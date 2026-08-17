#!/usr/bin/env python3
"""Replay a teleop HDF5 with physics and render SFT-ready observations.

Stage 1 of the teleop → SFT pipeline. Two orthogonal knobs control what each
step records (defaults: --controller joint, --cams 3):

  --controller  state convention recorded under obs/state (float32, 8D):
      joint (default): [arm_q(7), gripper_pos(1)]  -- absolute joint config for
                       a JointController policy; gripper_pos is the mean of the
                       two finger qpos.
      eef:             [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]  -- legacy
                       LIBERO / IsaacLab-Stack-Cube layout (both fingers kept).

  --cams        camera set recorded as image obs (see CAMERA_SETS):
      3 (default): image_left + image_right + wrist_image  (cam_left/cam_right
                   overviews + wrist).
      2:           image + wrist_image  (cam_opposite overview + wrist; LIBERO).

`action` handling depends on what the leader arm recorded:
  GELLO  (8D absolute joint target, JointController)      -> copied unchanged.
  SO-101 (7D EEF delta, InverseKinematicsController)      -> rewritten to the
         same 8D joint convention from the replayed joint states (a delta is
         meaningless as an SFT action); requires --controller joint.
Either way Stage 2 receives an 8D joint action and decides eef-delta vs joint.

Output HDF5 is consumed by Stage 2 (maniguard.data.lerobot.*), which writes a
LeRobot v2.1 dataset.

Run once per teleop trajectory (one traj per process keeps Isaac Sim's
scene-teardown quirks contained):

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
    python -m maniguard.data.playback \
        --input outputs/teleop_collected/dusty_transfer/task_0000_traj_001.hdf5 \
        --output outputs/teleop_rendered_joint/dusty_transfer/task_0000_traj_001.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm
from maniguard.utils.camera_setup import (
    CAMERA_RESOLUTION,
    build_external_camera_configs,
)


def _quat2axisangle(quat: th.Tensor) -> th.Tensor:
    """Convert (x, y, z, w) quaternion to axis-angle (3,)."""
    quat = quat.to(th.float32).clone()
    quat = th.clamp(quat, -1.0, 1.0)
    w = quat[3]
    sin_half = th.sqrt(th.clamp(1.0 - w * w, min=0.0))
    if float(sin_half) < 1e-6:
        return th.zeros(3, dtype=th.float32)
    angle = 2.0 * th.acos(th.clamp(w, -1.0, 1.0))
    axis = quat[:3] / sin_half
    return (axis * angle).to(th.float32)


# Camera composition per --cams choice, mapping each camera to its LeRobot image key. 
#   - The first entry is always the robot wrist camera; 
#   - the rest are external third-person views (built as external VisionSensors). 
# 2-cam is the LIBERO-style single overview; 
# 3-cam uses left/right overviews
CAMERA_SETS = {
    2: {"wrist": "wrist_image", "cam_opposite": "image"},
    3: {"wrist": "wrist_image", "cam_left": "image_left", "cam_right": "image_right"},
}


def _find_external_key(obs: dict, cam_name: str) -> str:
    """Flat-obs key for a named external camera's RGB."""
    exact = f"external::{cam_name}::rgb"
    if exact in obs:
        return exact
    for k in obs:
        if k.startswith("external::") and k.endswith("::rgb") and cam_name in k:
            return k
    raise RuntimeError(
        f"No {cam_name}::rgb key found. Available keys: {sorted(obs.keys())}"
    )


def _find_wrist_key(obs: dict, robot_name: str) -> str:
    """Flat-obs key for the robot's wrist camera RGB. Raises if absent
    (every teleop trajectory carries a wrist camera; a missing one is a bug)."""
    prefix = f"{robot_name}::"
    candidates = [k for k in obs if k.startswith(prefix) and k.endswith("::rgb")]
    if not candidates:
        raise RuntimeError(
            f"No wrist camera RGB key found for robot '{robot_name}'. "
            f"Available keys: {sorted(obs.keys())}"
        )
    lowered = {k.lower(): k for k in candidates}
    for pattern in (":eef_link:camera:0::rgb", ":camera_link:camera:0::rgb"):
        for low, full in lowered.items():
            if low.endswith(pattern):
                return full
    for token in ("eef_link", "wrist", "realsense", "d405", "hand", "camera"):
        for low, full in lowered.items():
            if token in low:
                return full
    return sorted(candidates)[0]


class ManiGuardPlaybackWriter(DataPlaybackWrapper):
    """DataPlaybackWrapper that emits the obs fields needed for SFT.

    The parent stores whatever `_process_obs` returns under
    `data/demo_X/obs/<key>`; `action` / sim `state` / reward / etc. are copied
    through unchanged. Two orthogonal knobs (overridden per-run by main() on the
    instance) select what `_process_obs` emits:
      _controller_mode: "eef"   -> eef_8d state [eef_pos(3), axisangle(3), grip(2)]
                        "joint" -> joint_8d state [arm_q(7), gripper_pos(1)]
      _n_cams:          2 | 3   -> picks a CAMERA_SETS entry (image keys + cams)
    Action is copied through by the parent and then normalized to the 8D joint
    convention by ``_normalize_actions_to_joint`` (GELLO passthrough / SO-101
    synthesis); the LeRobot export stage decides eef-delta vs joint handling.
    """

    _controller_mode = "joint"
    _n_cams = 3
    _keys_resolved = False

    def _process_obs(self, obs, info):
        robot = self.env.robots[0]

        # Resolve each camera's flat-dict key once ("wrist" -> robot wrist
        # sensor, others -> external sensors). DataPlaybackWrapper sets
        # flatten_obs_space=True so obs uses "::"-separated flat keys.
        if not self._keys_resolved:
            self._obs_keys = {
                cam: (_find_wrist_key(obs, robot.name) if cam == "wrist"
                      else _find_external_key(obs, cam))
                for cam in CAMERA_SETS[self._n_cams]
            }
            print(f"[Playback] controller={self._controller_mode}, n_cams={self._n_cams}")
            print(f"[Playback] obs keys = {self._obs_keys}")
            self._keys_resolved = True

        # State (controller-dependent).
        if self._controller_mode == "joint":
            qpos = robot.get_joint_positions()
            arm_q = qpos[robot.arm_control_idx[robot.default_arm]].to(th.float32)
            gripper_pos = qpos[robot.gripper_control_idx[robot.default_arm]].to(th.float32).mean().reshape(1)
            state = th.cat([arm_q, gripper_pos])
        else:  # "eef"
            eef_pos = robot.get_relative_eef_position().to(th.float32)
            eef_axisangle = _quat2axisangle(robot.get_relative_eef_orientation().to(th.float32))
            gripper_qpos = robot.get_joint_positions()[
                robot.gripper_control_idx[robot.default_arm]
            ].to(th.float32)
            state = th.cat([eef_pos, eef_axisangle, gripper_qpos])

        # Images: one entry per camera, keyed by its LeRobot output name
        # (drop alpha, uint8 HWC).
        out = {
            lerobot_key: obs[self._obs_keys[cam]][..., :3].to(th.uint8)
            for cam, lerobot_key in CAMERA_SETS[self._n_cams].items()
        }
        out["state"] = state
        return out


def _dump_mp4s(hdf5_path: str, fps: int = 30) -> None:
    """Dump one MP4 per recorded image stream next to an obs-rendered HDF5.

    Handles both schemas: 2-cam (image, wrist_image) and 3-cam (image_left,
    image_right, wrist_image) -- dumps whichever image obs are present.
    """
    import h5py
    import imageio.v2 as imageio

    # LeRobot image key -> mp4 suffix (only present keys are dumped).
    KEY_SUFFIX = {
        "image": "main", "image_left": "left",
        "image_right": "right", "wrist_image": "wrist",
    }
    stem, _ = os.path.splitext(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if not demo_keys:
            print("[Playback] No demos in HDF5, skipping MP4 export")
            return
        obs = f["data"][demo_keys[0]]["obs"]
        streams = {suf: obs[k][:] for k, suf in KEY_SUFFIX.items() if k in obs}

    for suffix, frames in streams.items():
        path = f"{stem}_{suffix}.mp4"
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"[Playback] MP4: {path}  ({len(frames)} frames)")


def _stamp_metadata(hdf5_path: str, controller_mode: str, n_cams: int) -> None:
    """Fingerprint the playback config into the output HDF5's `data` group.

    The LeRobot export stage reads this to auto-select the schema: eef and joint
    `obs/state` are both 8D float, indistinguishable from the array alone, so the
    controller mode must be recorded explicitly. n_cams is stored too (though it
    is also inferable from the obs image keys) to keep the record self-describing.
    """
    import h5py
    with h5py.File(hdf5_path, "a") as f:
        f["data"].attrs["controller_mode"] = controller_mode
        f["data"].attrs["n_cams"] = int(n_cams)


def _normalize_actions_to_joint(output_path: str, input_path: str, controller_mode: str) -> None:
    """Rewrite SO-101 raw actions (7D IK delta) as 8D joint-native actions.

    GELLO records the SFT convention directly (8D absolute joint target) and is
    passed through untouched. SO-101's InverseKinematicsController records
    [dpos(3), drot(3), gripper(1)] — a delta command is meaningless as an SFT
    action, but under ``--controller joint`` the output's ``obs/state`` holds the
    N+1 replayed absolute joint configs, so the faithful joint target for step t
    is the configuration the arm actually reached after applying raw action t:

        action_t = [state_{t+1}[:7], raw_gripper_cmd_t]

    (the same "commanded target ~ next config" relation the GELLO recordings
    exhibit). The result is stamped as ``data.attrs['action_source']`` so the
    provenance stays visible downstream.
    """
    import h5py

    with h5py.File(input_path, "r") as f:
        raw = {k: f["data"][k]["action"][:] for k in f["data"]}
    dims = {a.shape[1] for a in raw.values()}

    if dims == {8}:
        source = "joint_passthrough"
    elif dims == {7}:
        if controller_mode != "joint":
            raise SystemExit(
                "[Playback] ERROR: 7D (IK-delta) raw actions need --controller joint "
                "to synthesize joint actions from the replayed states; the eef state "
                "layout does not record joint configs."
            )
        source = "ik_delta_to_joint_from_states"
        with h5py.File(output_path, "r+") as f:
            for key, grp in f["data"].items():
                raw_a = raw.get(key)
                if raw_a is None:
                    raise SystemExit(f"[Playback] ERROR: episode {key!r} not in {input_path}")
                state = grp["obs"]["state"][:]
                n = raw_a.shape[0]
                if state.shape[0] != n + 1:
                    raise SystemExit(
                        f"[Playback] ERROR: {key}: expected {n + 1} obs/state frames for "
                        f"{n} actions, got {state.shape[0]} — cannot align joint targets."
                    )
                joint_a = np.concatenate([state[1:, :7], raw_a[:, -1:]], axis=1).astype(np.float32)
                del grp["action"]
                grp.create_dataset("action", data=joint_a)
        print(f"[Playback] SO-101 input: rewrote 7D IK-delta actions as 8D joint "
              f"targets from the replayed states ({len(raw)} episode(s)).")
    else:
        raise SystemExit(
            f"[Playback] ERROR: unexpected raw action width(s) {sorted(dims)}; "
            "expected 8 (GELLO joint) or 7 (SO-101 IK delta)."
        )

    with h5py.File(output_path, "a") as f:
        f["data"].attrs["action_source"] = source


def _setup_cameras_from_scene(env, diagnostics_path: str | None = None) -> None:
    """Position the external cameras at the task's PRESET poses.

    Thin wrapper over the shared load-side rule in maniguard.utils.camera_setup
    (place_recorded_task_cameras): if --diagnostics points at the source task's
    diagnostics.jsonl, its recorded ``cameras`` poses are applied — the same
    views datagen/eval/teleop use; without it, the canonical robot-frame
    recompute is the (warned) fallback.
    """
    import json

    from maniguard.utils.camera_setup import place_recorded_task_cameras

    diagnostics = None
    if diagnostics_path and os.path.isfile(diagnostics_path):
        with open(diagnostics_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    diagnostics = json.loads(line)
                    break
    place_recorded_task_cameras(env, diagnostics, set_viewer=True)


def _force_sensor_resolution(env, resolution: int) -> None:
    """Work around OmniGibson's Kit-viewport override of sensor_kwargs.

    VisionSensor `image_height`/`image_width` kwargs don't always survive
    Kit's viewport init (the new viewport falls back to the app default),
    so we explicitly set them on each sensor and reload the observation
    space before the first reset(). See StanfordVL/OmniGibson#266 and #1875.
    """
    for cam in (env.external_sensors or {}).values():
        cam.image_height = resolution
        cam.image_width = resolution
    robot = env.robots[0] if env.robots else None
    if robot is not None:
        from omnigibson.sensors import VisionSensor
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                sensor.image_height = resolution
                sensor.image_width = resolution
    env.load_observation_space()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Path to a teleop HDF5 recorded by so101_franka_teleop.py")
    p.add_argument("--output", required=True,
                   help="Path to write the rendered HDF5 (will be overwritten)")
    p.add_argument("--resolution", type=int, default=CAMERA_RESOLUTION,
                   help="Square side length for main and wrist cameras")
    p.add_argument("--n-render-iterations", type=int, default=3,
                   help="Render iterations per step (higher = cleaner but slower)")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode ID within the input HDF5 (teleop produces one)")
    p.add_argument("--all-episodes", action="store_true",
                   help="Process every demo_* in the input HDF5 instead of one")
    p.add_argument("--save-mp4", action="store_true",
                   help="Also dump per-camera MP4s next to the rendered HDF5 "
                        "for quick visual review.")
    p.add_argument("--controller", choices=["eef", "joint"], default="joint",
                   help="State convention to record (default: joint). "
                        "'joint' = joint_8d [arm_q(7), gripper_pos(1)] for a "
                        "JointController policy; 'eef' = eef_8d (legacy LIBERO "
                        "path). Independent of --cams.")
    p.add_argument("--cams", type=int, choices=[2, 3], default=3,
                   help="Camera set (default: 3; see CAMERA_SETS). "
                        "3 = cam_left + cam_right + wrist; "
                        "2 = cam_opposite + wrist. Independent of --controller.")
    p.add_argument("--diagnostics", type=str, default=None,
                   help="Path to the source task's diagnostics.jsonl; its recorded "
                        "cameras poses are applied (the shared load-side rule). "
                        "Omit only for legacy scenes without recorded poses.")
    args = p.parse_args()

    # DataPlaybackWrapper precondition.
    gm.ENABLE_TRANSITION_RULES = False

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Force 256x256 for both main and wrist cameras. The teleop HDF5 was
    # recorded at Kit's default (~1280x720) but we need square 256x256 for
    # the Pi0.5 / OmniGibsonDataConfig pipeline.
    external_names = [c for c in CAMERA_SETS[args.cams] if c != "wrist"]
    external_cfg = build_external_camera_configs(
        names=external_names,
        resolution=args.resolution,
    )
    robot_sensor_cfg = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_height": args.resolution,
                "image_width": args.resolution,
            },
        },
    }

    env = ManiGuardPlaybackWriter.create_from_hdf5(
        input_path=args.input,
        output_path=args.output,
        robot_obs_modalities=["rgb", "proprio"],
        external_sensors_config=external_cfg,
        robot_sensor_config=robot_sensor_cfg,
        n_render_iterations=args.n_render_iterations,
        only_successes=False,
        overwrite=True,
        include_robot_control=True,
        include_contacts=True,
    )

    # Thread the two orthogonal knobs into the writer's _process_obs.
    env._controller_mode = args.controller
    env._n_cams = args.cams

    _force_sensor_resolution(env.env, args.resolution)

    # Place the external cameras at the task's preset poses.
    # DataPlaybackWrapper doesn't position them, so without this cam_* sit at
    # the default (0,0,0) pose pointing at nothing -> uniform gray images.
    _setup_cameras_from_scene(env.env, diagnostics_path=args.diagnostics)

    if args.all_episodes:
        env.playback_dataset(record_data=True)
    else:
        env.playback_episode(episode_id=args.episode, record_data=True)

    env.save_data()
    _stamp_metadata(args.output, args.controller, args.cams)
    _normalize_actions_to_joint(args.output, args.input, args.controller)
    print(f"[Playback] Wrote: {args.output}  "
          f"(controller={args.controller}, n_cams={args.cams})")

    if args.save_mp4:
        _dump_mp4s(args.output)

    og.clear()


if __name__ == "__main__":
    main()
