"""Joint-native RAW trajectory recorder for datagen — Layer-1 primitive.

Writes the RAW curobo-collected data the user reviews BEFORE any LeRobot
conversion: per trajectory, a self-contained folder with the five video streams
(4 bench third-person + injected wrist) + one ``traj.hdf5`` (the joint trajectory
+ sim-state dump) + ``meta.json``. The MP4s match the bench spec byte-for-byte
(same PyAV ``h264`` / ``yuv420p`` encode at the camera's native render size,
256² @ 30 fps), so a reviewer just opens the videos. LeRobot v2.1 conversion for
SFT is a SEPARATE downstream step (passthrough — no re-encode), not done here.

Per env step the recorder captures: the 5 image streams, the robot's ACHIEVED
joint state, the cuRobo-COMMANDED joint target, the gripper command, and a
serialized sim-state dump (the MimicGen replay hook). ``traj.hdf5`` columns
(see ``data_format``):
  ``state``             (N,8) = ``[arm_q, gripper]``                  current joints
  ``actions``           (N,8) = ``[arm_q[t+1], gripper_cmd]``         (b) next-achieved
  ``actions_commanded`` (N,8) = ``[arm_q_cmd, gripper_cmd]``          (a) cuRobo command
  ``states``            (N,*) = serialized ``og.sim.dump_state`` per step (MimicGen)
  ``datagen_info/gripper_action`` (N,)

The recorder NEVER commands the arm — the caller (``execute_trajectory`` / the
family skeleton) steps the env with cuRobo joint targets and calls
``record_step(arm_q_cmd, gripper_cmd)`` after each ``env.step``; the recorder only
READS the achieved state + frames. Joint-native (no eef-8d / sim-state joint
reverse-engineering).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from maniguard.data.datagen import data_format
from maniguard.data.datagen.primitives.cameras import find_wrist_sensor


# --- PyAV h264/yuv420p writer (replicated from the bench task_generation video
#     writer so raw MP4s match the bench rollout spec exactly). ---------------
def _open_video(path: Path, fps: int, h: int, w: int) -> dict:
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = int(w)
    stream.height = int(h)
    stream.pix_fmt = "yuv420p"
    return {"container": container, "stream": stream}


def _write_frame(vw: dict, frame_uint8: np.ndarray) -> None:
    import av

    frame = np.ascontiguousarray(frame_uint8[..., :3].astype(np.uint8))
    vframe = av.VideoFrame.from_ndarray(frame, format="rgb24")
    for packet in vw["stream"].encode(vframe):
        vw["container"].mux(packet)


def _close_video(vw: dict) -> None:
    try:
        for packet in vw["stream"].encode():  # flush
            vw["container"].mux(packet)
        vw["container"].close()
    except Exception:  # noqa: BLE001
        pass


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _sensor_rgb_uint8(sensor, resolution: int) -> np.ndarray:
    """One RGB frame as (resolution, resolution, 3) uint8; zeros if unavailable."""
    if sensor is None:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    try:
        obs = sensor.get_obs()
        rgb = obs[0].get("rgb") if isinstance(obs, tuple) else obs.get("rgb")
    except Exception:  # noqa: BLE001
        rgb = None
    if rgb is None:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    return _to_np(rgb[..., :3]).astype(np.uint8)


def _arm_gripper_state(robot) -> tuple[np.ndarray, float]:
    """Achieved ``(arm_q (7,), gripper_mean)`` from the robot's current joints."""
    jp = _to_np(robot.get_joint_positions())
    arm = robot.default_arm
    arm_idx = _to_np(robot.arm_control_idx[arm]).astype(int)
    grip_idx = _to_np(robot.gripper_control_idx[arm]).astype(int)
    arm_q = jp[arm_idx].astype(np.float32)
    gripper = float(jp[grip_idx].mean())
    return arm_q, gripper


def _dump_sim_state() -> np.ndarray:
    import omnigibson as og

    s = og.sim.dump_state(serialized=True)
    return _to_np(s).astype(np.float32)


class Recorder:
    """One RAW datagen trajectory: 5 MP4s + ``traj.hdf5`` + ``meta.json`` in a folder.

    Lifecycle (per trajectory)::

        rec = Recorder()
        rec.attach(env, robot, out_dir, prompt)
        # in the execution loop, after each env.step(<curobo joint target>):
        rec.record_step(arm_q_cmd, gripper_cmd)
        rec.finalize(success=True, attrs={...})   # or success=False -> drop the folder
    """

    def __init__(self, *,
                 resolution: int = data_format.RESOLUTION,
                 fps: int = data_format.FPS,
                 record_sim_states: bool = True):
        self.resolution = int(resolution)
        self.fps = int(fps)
        self.record_sim_states = bool(record_sim_states)

        self.env = None
        self.robot = None
        self.out_dir: Path | None = None
        self.prompt = ""
        self._wrist = None
        self._writers: dict[str, dict] | None = None   # lazily opened on first frame
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._arm_q: list[np.ndarray] = []        # achieved arm joints (7,)
        self._gripper_pos: list[float] = []        # achieved gripper mean
        self._arm_q_cmd: list[np.ndarray] = []     # cuRobo commanded arm joints (7,)
        self._gripper_cmd: list[float] = []        # commanded gripper (binary)
        self._sim_states: list[np.ndarray] = []    # serialized sim dump per step
        self.n_steps = 0

    def attach(self, env, robot, out_dir: str | Path, prompt: str = "") -> None:
        self.env = env
        self.robot = robot
        self.out_dir = Path(out_dir)
        self.prompt = str(prompt)
        self._wrist = find_wrist_sensor(robot)
        self._writers = None
        self._reset_buffers()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        wname = getattr(self._wrist, "name", None)
        print(f"[datagen.record] raw trajectory -> {self.out_dir} (wrist={wname})",
              flush=True)

    def _grab_frames(self) -> dict[str, np.ndarray]:
        sens = self.env.external_sensors or {}
        frames = {
            img_key: _sensor_rgb_uint8(sens.get(og_name), self.resolution)
            for img_key, og_name in data_format.THIRD_PERSON_CAMS.items()
        }
        frames[data_format.WRIST_KEY] = _sensor_rgb_uint8(self._wrist, self.resolution)
        return frames

    def _open_writers(self, frames: dict[str, np.ndarray]) -> None:
        # Size each writer from the ACTUAL frame (sensor-kwarg resolution is
        # unreliable in OmniGibson; matches the bench render writer).
        self._writers = {}
        for key in data_format.IMAGE_KEYS:
            h, w = int(frames[key].shape[0]), int(frames[key].shape[1])
            self._writers[key] = _open_video(self.out_dir / f"{key}.mp4", self.fps, h, w)

    def record_step(self, arm_q_cmd: Sequence[float], gripper_cmd: float) -> None:
        """Capture one step. ``arm_q_cmd`` = the cuRobo-commanded arm joint target
        used to drive this step (7,); ``gripper_cmd`` = the binary gripper command.
        Achieved joints + frames + sim state are read from the env."""
        frames = self._grab_frames()
        if self._writers is None:
            self._open_writers(frames)
        for key, vw in self._writers.items():
            _write_frame(vw, frames[key])

        arm_q, gripper = _arm_gripper_state(self.robot)
        self._arm_q.append(arm_q)
        self._gripper_pos.append(gripper)
        self._arm_q_cmd.append(
            np.asarray(arm_q_cmd, dtype=np.float32).reshape(-1)[:data_format.ARM_DOF])
        self._gripper_cmd.append(float(gripper_cmd))
        if self.record_sim_states:
            self._sim_states.append(_dump_sim_state())
        self.n_steps += 1

    def _build_columns(self):
        achieved = np.stack(self._arm_q, axis=0)                       # (N,7)
        grip_pos = np.asarray(self._gripper_pos, dtype=np.float32)     # (N,)
        grip_cmd = np.asarray(self._gripper_cmd, dtype=np.float32)     # (N,)
        cmd = np.stack(self._arm_q_cmd, axis=0)                        # (N,7)

        state = np.concatenate([achieved, grip_pos[:, None]], axis=1)  # (N,8)
        # (b) next-achieved absolute joint; the final step holds (no t+1).
        next_arm = np.concatenate([achieved[1:], achieved[-1:]], axis=0)
        actions = np.concatenate([next_arm, grip_cmd[:, None]], axis=1)        # (N,8)
        # (a) cuRobo commanded joint target.
        actions_cmd = np.concatenate([cmd, grip_cmd[:, None]], axis=1)         # (N,8)
        return (state.astype(np.float32), actions.astype(np.float32),
                actions_cmd.astype(np.float32))

    def finalize(self, success: bool, attrs: dict | None = None) -> Path | None:
        """Close the 5 MP4 writers. On success write ``traj.hdf5`` + ``meta.json`` and
        return ``out_dir``; on failure/no-steps drop the whole trajectory folder and
        return None."""
        for vw in (self._writers or {}).values():
            _close_video(vw)
        self._writers = None

        if not success or self.n_steps == 0:
            if self.out_dir is not None and self.out_dir.exists():
                shutil.rmtree(self.out_dir, ignore_errors=True)
            print(f"[datagen.record] dropped {self.out_dir} "
                  f"(success={success}, steps={self.n_steps})", flush=True)
            self._reset_buffers()
            return None

        state, actions, actions_cmd = self._build_columns()
        self._write_hdf5(state, actions, actions_cmd, attrs or {})
        self._write_meta(attrs or {})
        out = self.out_dir
        print(f"[datagen.record] wrote {out} ({self.n_steps} steps, 5 videos + traj.hdf5)",
              flush=True)
        self._reset_buffers()
        return out

    def _write_hdf5(self, state, actions, actions_cmd, attrs: dict) -> None:
        import h5py

        with h5py.File(str(self.out_dir / "traj.hdf5"), "w") as f:
            for k, v in attrs.items():
                # h5py attrs take only scalars/strings — JSON-encode nested values (e.g. jitter dict)
                f.attrs[k] = v if isinstance(v, (str, int, float, bool, np.integer, np.floating)) \
                    else json.dumps(v, default=str)
            f.attrs["n_steps"] = int(self.n_steps)
            f.attrs["prompt"] = self.prompt
            f.attrs["fps"] = int(self.fps)
            f.attrs["resolution"] = int(self.resolution)
            f.create_dataset("state", data=state)
            f.create_dataset("actions", data=actions)
            f.create_dataset("actions_commanded", data=actions_cmd)
            g = f.create_group("datagen_info")
            g.create_dataset("gripper_action", data=actions_cmd[:, -1].astype(np.float32))
            if self.record_sim_states and self._sim_states:
                lens = {len(s) for s in self._sim_states}
                if len(lens) == 1:
                    f.create_dataset("states", data=np.stack(self._sim_states, axis=0),
                                     compression="gzip", compression_opts=4)
                else:
                    # ragged serialized state (AG attach/detach changes its length across steps)
                    # -> pad to max + store per-step lengths so it stays HDF5-writable.
                    L = max(lens)
                    padded = np.zeros((len(self._sim_states), L), dtype=np.float32)
                    slen = np.empty(len(self._sim_states), dtype=np.int32)
                    for i, s in enumerate(self._sim_states):
                        padded[i, : len(s)] = s
                        slen[i] = len(s)
                    f.create_dataset("states", data=padded, compression="gzip", compression_opts=4)
                    f.create_dataset("states_len", data=slen)

    def _write_meta(self, attrs: dict) -> None:
        meta = {
            "prompt": self.prompt,
            "success": True,
            "n_steps": int(self.n_steps),
            "fps": int(self.fps),
            "resolution": int(self.resolution),
            "video_keys": list(data_format.IMAGE_KEYS),
            **attrs,                       # keep nested dicts/lists (e.g. jitter) as real JSON
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
