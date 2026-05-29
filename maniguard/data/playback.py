#!/usr/bin/env python3
"""Replay a teleop HDF5 with physics and render SFT-ready observations.

Stage 1 of the teleop → SFT pipeline. Two orthogonal knobs control what each
step records (defaults: --controller joint, --cams 3):

  --controller  state convention recorded under obs/state (float32, 8D):
      joint (default): [arm_q(7), gripper_pos(1)]  -- absolute joint config for
                       a JointController policy; gripper_pos is the mean of the
                       two finger qpos. Matches sentinel-pnp-clutter-joint.
      eef:             [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]  -- legacy
                       LIBERO / IsaacLab-Stack-Cube layout (both fingers kept).

  --cams        camera set recorded as image obs (see CAMERA_SETS):
      3 (default): image_left + image_right + wrist_image  (cam_left/cam_right
                   overviews + wrist; matches sentinel-pnp-clutter-joint).
      2:           image + wrist_image  (cam_opposite overview + wrist; LIBERO).

`action` is copied from the input HDF5 unchanged (the raw teleop 8D joint
target); the Stage 2 LeRobot export decides eef-delta vs joint handling.

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
    Action is the raw teleop 8D joint passthrough either way; the LeRobot export
    stage decides eef-delta vs joint handling.
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


def _setup_cameras_from_scene(env) -> None:
    """Position cam_opposite like teleop did.

    Mirrors so101_franka_teleop.py: find support_surface + a target object,
    call build_video_view_specs + setup_cameras to place the external
    camera(s) at the canonical "opposite-side overview" pose, and
    synchronise the viewer camera to match.
    """
    try:
        from maniguard.task_generation.utils.video import (
            build_video_view_specs,
            setup_cameras,
        )
    except ImportError as e:
        print(f"[Playback] WARNING: camera setup helpers not importable ({e}); "
              f"cam_opposite will stay at its default pose.")
        return

    scene = env.scene
    robot = env.robots[0]
    support_obj = scene.object_registry("name", "support_surface")

    if support_obj is not None:
        # Pick any non-robot, non-support object as the "target" for the lookat.
        target_obj = next(
            (obj for obj in scene.objects if obj is not robot and obj is not support_obj),
            support_obj,
        )
        views = build_video_view_specs(
            None, robot, target_obj, support_obj=support_obj,
        )
        setup_cameras(env, views)
        print(f"[Playback] Camera mode = support_surface "
              f"(target={target_obj.name}, support={support_obj.name}).")
        return

    # Robot-frame fallback for HF furnished scenes that ship no 'support_surface'
    # (e.g. the 6fam-base benchmark scenes). Mirrors gello_franka_teleop's
    # _setup_cameras_for_scene fallback verbatim so the re-rendered camera poses
    # match the teleop-time poses. setup_cameras positions whichever of
    # cam_opposite / cam_left / cam_right actually exist in the env, so this
    # works unchanged for both the 2-cam and 3-cam external-sensor sets.
    import omnigibson.utils.transform_utils as _T

    rp_t, rq_t = robot.get_position_orientation()
    rp = np.asarray(rp_t.cpu().numpy() if hasattr(rp_t, "cpu") else rp_t,
                    dtype=np.float32)
    rmat_t = _T.quat2mat(rq_t)
    rmat = np.asarray(rmat_t.cpu().numpy() if hasattr(rmat_t, "cpu") else rmat_t,
                      dtype=np.float32)
    forward = rmat[:, 0].copy()
    forward[2] = 0.0
    n = float(np.linalg.norm(forward))
    forward = forward / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)

    cam_height_off = 0.9
    back_off = 1.2
    side_off = 1.0
    side_forward_off = 0.2

    workspace = rp + forward * 0.45 + np.array([0, 0, 0.05], dtype=np.float32)
    opp_eye = rp - forward * back_off + np.array([0, 0, cam_height_off], dtype=np.float32)
    left_eye = rp + left * side_off + forward * side_forward_off \
                  + np.array([0, 0, cam_height_off], dtype=np.float32)
    right_eye = rp - left * side_off + forward * side_forward_off \
                   + np.array([0, 0, cam_height_off], dtype=np.float32)
    views = [
        {"label": "opposite_side_front", "eye": opp_eye.tolist(),  "lookat": workspace.tolist()},
        {"label": "left_overview",       "eye": left_eye.tolist(), "lookat": workspace.tolist()},
        {"label": "right_overview",      "eye": right_eye.tolist(),"lookat": workspace.tolist()},
    ]
    setup_cameras(env, views)
    print(f"[Playback] Camera mode = robot-frame; "
          f"robot_pos=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}), "
          f"forward=({forward[0]:.2f},{forward[1]:.2f})")


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

    # Reproduce the teleop-time camera view. Teleop runs setup_cameras()
    # which places cam_opposite at an overhead "opposite-side" pose based
    # on robot/target/support geometry. DataPlaybackWrapper doesn't, so
    # without this, cam_opposite sits at the default (0,0,0) pose pointing
    # at nothing -> rendered obs is a uniform gray image.
    _setup_cameras_from_scene(env.env)

    if args.all_episodes:
        env.playback_dataset(record_data=True)
    else:
        env.playback_episode(episode_id=args.episode, record_data=True)

    env.save_data()
    _stamp_metadata(args.output, args.controller, args.cams)
    print(f"[Playback] Wrote: {args.output}  "
          f"(controller={args.controller}, n_cams={args.cams})")

    if args.save_mp4:
        _dump_mp4s(args.output)

    og.clear()


if __name__ == "__main__":
    main()
