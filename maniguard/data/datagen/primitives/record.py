"""Joint-native episode recorder for datagen — Layer-1 primitive.

Per env step the recorder captures: the five image streams (4 bench third-person +
injected wrist), the robot's ACHIEVED joint state, the cuRobo-COMMANDED joint
target, the gripper command, and a serialized sim-state dump (the MimicGen replay
hook). On success it writes one LeRobot v2.1 episode (schema =
``data_format.lerobot_features``) + a per-episode MimicGen sidecar HDF5; on failure
it aborts (drops the pre-streamed MP4s, writes nothing).

Two action columns (see ``data_format``):
  ``actions``           (b) next-achieved ``[arm_q[t+1], gripper_cmd]``  -- DEFAULT
  ``actions_commanded`` (a) cuRobo target  ``[arm_q_cmd[t], gripper_cmd]`` -- extra

The recorder NEVER commands the arm. The caller (``execute_trajectory`` / the family
skeleton) steps the env with cuRobo joint targets and calls
``record_step(arm_q_cmd, gripper_cmd)`` after each ``env.step``; the recorder only
READS the achieved state + frames. It reuses the bench LeRobot v2.1 MP4-passthrough
infra (``create_or_open_dataset``) so MP4s stream straight to their target paths and
``save_episode`` skips re-encoding. Joint-native (no eef-8d / sim-state
reverse-engineering — that was the old ``_sft_recorder``'s constraint, not ours).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from maniguard.data.datagen import data_format
from maniguard.data.datagen.primitives.cameras import find_wrist_sensor
from maniguard.data.lerobot.lerobot_writer import create_or_open_dataset


def make_dataset(repo_id: str, root: str | Path,
                 fps: int = data_format.FPS,
                 resolution: int = data_format.RESOLUTION):
    """Create or open the datagen LeRobot v2.1 dataset (5 video streams + joint
    state + joint actions [achieved] + actions_commanded). Reuses the bench
    MP4-passthrough infra so streamed MP4s are not re-encoded on commit."""
    return create_or_open_dataset(
        repo_id=repo_id, root=root, fps=fps, resolution=resolution,
        apply_passthrough=True,
        features=data_format.lerobot_features(resolution),
    )


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
    """One LeRobot v2.1 episode of joint-native datagen capture.

    Lifecycle (per task/episode)::

        rec = Recorder(dataset)
        rec.attach(env, robot, prompt)
        # in the execution loop, after each env.step(<curobo joint target>):
        rec.record_step(arm_q_cmd, gripper_cmd)
        rec.finalize(success=True, attrs={...})   # or success=False -> abort
    """

    def __init__(self, dataset, *,
                 resolution: int = data_format.RESOLUTION,
                 fps: int = data_format.FPS,
                 record_sim_states: bool = True):
        self.dataset = dataset
        self.resolution = int(resolution)
        self.fps = int(fps)
        self.record_sim_states = bool(record_sim_states)

        self.env = None
        self.robot = None
        self.prompt = ""
        self._wrist = None
        self._mp4_writers: dict[str, Any] = {}
        self._target_mp4: dict[str, Path] = {}
        self._episode_index = 0
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._arm_q: list[np.ndarray] = []        # achieved arm joints (7,)
        self._gripper_pos: list[float] = []        # achieved gripper mean
        self._arm_q_cmd: list[np.ndarray] = []     # cuRobo commanded arm joints (7,)
        self._gripper_cmd: list[float] = []        # commanded gripper (binary)
        self._sim_states: list[np.ndarray] = []    # serialized sim dump per step
        self.n_steps = 0

    def attach(self, env, robot, prompt: str = "") -> None:
        import imageio.v2 as imageio

        self.env = env
        self.robot = robot
        self.prompt = str(prompt)
        self._wrist = find_wrist_sensor(robot)
        self._reset_buffers()

        # Reserve this episode's slot + stream MP4s straight to LeRobot's target
        # video paths (the passthrough patch then skips re-encoding on commit).
        meta = self.dataset.meta
        self._episode_index = meta.total_episodes
        self._target_mp4 = {
            key: self.dataset.root / meta.get_video_file_path(self._episode_index, key)
            for key in meta.video_keys
        }
        for p in self._target_mp4.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        self._mp4_writers = {
            key: imageio.get_writer(str(p), fps=self.fps, codec="libx264", quality=7)
            for key, p in self._target_mp4.items()
        }
        wname = getattr(self._wrist, "name", None)
        print(f"[datagen.record] episode {self._episode_index}: streaming "
              f"{len(self._mp4_writers)} videos (wrist={wname})", flush=True)

    def _grab_frames(self) -> dict[str, np.ndarray]:
        sens = self.env.external_sensors or {}
        frames = {
            img_key: _sensor_rgb_uint8(sens.get(og_name), self.resolution)
            for img_key, og_name in data_format.THIRD_PERSON_CAMS.items()
        }
        frames[data_format.WRIST_KEY] = _sensor_rgb_uint8(self._wrist, self.resolution)
        return frames

    def record_step(self, arm_q_cmd: Sequence[float], gripper_cmd: float) -> None:
        """Capture one step. ``arm_q_cmd`` = the cuRobo-commanded arm joint target
        used to drive this step (7,); ``gripper_cmd`` = the binary gripper command.
        Achieved joints + frames + sim state are read from the env."""
        frames = self._grab_frames()
        for key, writer in self._mp4_writers.items():
            writer.append_data(frames[key])

        arm_q, gripper = _arm_gripper_state(self.robot)
        self._arm_q.append(arm_q)
        self._gripper_pos.append(gripper)
        self._arm_q_cmd.append(np.asarray(arm_q_cmd, dtype=np.float32).reshape(-1)[:data_format.ARM_DOF])
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
        return state.astype(np.float32), actions.astype(np.float32), actions_cmd.astype(np.float32)

    def finalize(self, success: bool, attrs: dict | None = None) -> Path | None:
        """Close MP4 writers. On success write the LeRobot episode + MimicGen
        sidecar and return the sidecar path; on failure/no-steps drop the MP4s and
        return None."""
        for w in self._mp4_writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        self._mp4_writers = {}

        if not success or self.n_steps == 0:
            for p in self._target_mp4.values():
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            print(f"[datagen.record] episode {self._episode_index}: aborted "
                  f"(success={success}, steps={self.n_steps})", flush=True)
            self._reset_buffers()
            return None

        state, actions, actions_cmd = self._build_columns()
        self._commit_lerobot(state, actions, actions_cmd)
        sidecar = self._write_sidecar(actions_cmd, attrs or {})
        print(f"[datagen.record] episode {self._episode_index}: committed "
              f"{self.n_steps} steps + sidecar {sidecar.name}", flush=True)
        self._reset_buffers()
        return sidecar

    def _commit_lerobot(self, state, actions, actions_cmd) -> None:
        for key, p in self._target_mp4.items():
            if not p.is_file():
                raise RuntimeError(f"target MP4 missing for {key!r}: {p}")
        dummy = np.zeros(self.dataset.features[data_format.IMAGE_KEYS[0]]["shape"],
                         dtype=np.uint8)
        for i in range(self.n_steps):
            frame = {key: dummy for key in data_format.IMAGE_KEYS}
            frame["state"] = state[i]
            frame["actions"] = actions[i]
            frame["actions_commanded"] = actions_cmd[i]
            # LeRobot 0.3.4: task is a separate positional arg, not a frame key.
            self.dataset.add_frame(frame, self.prompt)
        self.dataset.save_episode()

    def _write_sidecar(self, actions_cmd, attrs: dict) -> Path:
        import h5py

        sidecar_dir = Path(self.dataset.root) / "mimicgen_sidecar"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        path = sidecar_dir / f"episode_{self._episode_index:06d}.hdf5"
        with h5py.File(str(path), "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["n_steps"] = int(self.n_steps)
            f.attrs["prompt"] = self.prompt
            f.attrs["episode_index"] = int(self._episode_index)
            g = f.create_group("datagen_info")
            g.create_dataset("gripper_action", data=actions_cmd[:, -1].astype(np.float32))
            if self.record_sim_states and self._sim_states:
                f.create_dataset("states", data=np.stack(self._sim_states, axis=0),
                                 compression="gzip", compression_opts=4)
        return path
