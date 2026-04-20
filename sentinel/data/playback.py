#!/usr/bin/env python3
"""Replay a teleop HDF5 with physics and render SFT-ready observations.

Stage 1 of the teleop → SFT pipeline. For each step:
  obs/image       : cam_opposite RGB, uint8 (H, W, 3), 256x256 by default
  obs/wrist_image : wrist sensor RGB, uint8 (H, W, 3), 256x256 by default
  obs/state       : [eef_pos(3), eef_axisangle(3), gripper_qpos(2)] float32 (8D)
action            : copied from the input HDF5 (7D Franka IK delta, unchanged)

State layout matches IsaacLab-Stack-Cube convention so we can reuse the
norm_stats / weight init from Pi0.5 ckpts trained on that dataset
(see RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py:_wrap_obs). Both finger
qpos are kept separately, not averaged.

Output HDF5 is consumed by Stage 2 (tools/hdf5_to_lerobot.py), which writes
a LeRobot v2.1 dataset ready for RLinf's OmniGibsonDataConfig SFT path.

Run once per teleop trajectory (one traj per process keeps Isaac Sim's
scene-teardown quirks contained):

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
    python tools/playback_teleop_to_hdf5.py \
        --input outputs/teleop/traj_0.hdf5 \
        --output outputs/teleop_rendered/traj_0.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch as th

REPO_ROOT = Path(__file__).resolve().parents[2]
RLINF_ROOT = REPO_ROOT / "RLinf"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm
from sentinel.utils.camera_setup import (
    CAMERA_RESOLUTION,
    build_external_camera_configs,
)


def _quat2axisangle(quat: th.Tensor) -> th.Tensor:
    """Convert (x, y, z, w) quaternion to axis-angle (3,). Matches LIBERO/RLinf convention."""
    quat = quat.to(th.float32).clone()
    quat = th.clamp(quat, -1.0, 1.0)
    w = quat[3]
    sin_half = th.sqrt(th.clamp(1.0 - w * w, min=0.0))
    if float(sin_half) < 1e-6:
        return th.zeros(3, dtype=th.float32)
    angle = 2.0 * th.acos(th.clamp(w, -1.0, 1.0))
    axis = quat[:3] / sin_half
    return (axis * angle).to(th.float32)


def _find_main_key(obs: dict) -> str:
    """Flat-obs key for cam_opposite's RGB."""
    exact = "external::cam_opposite::rgb"
    if exact in obs:
        return exact
    for k in obs:
        if k.startswith("external::") and k.endswith("::rgb") and "cam_opposite" in k:
            return k
    raise RuntimeError(
        f"No cam_opposite::rgb key found. Available keys: {sorted(obs.keys())}"
    )


def _find_wrist_key(obs: dict, robot_name: str) -> str | None:
    """Flat-obs key for the robot's wrist camera RGB."""
    prefix = f"{robot_name}::"
    candidates = [k for k in obs if k.startswith(prefix) and k.endswith("::rgb")]
    if not candidates:
        return None
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


class SentinelPlaybackWriter(DataPlaybackWrapper):
    """DataPlaybackWrapper that emits exactly the obs fields needed for SFT.

    The parent stores whatever `_process_obs` returns under
    `data/demo_X/obs/<key>`. We return three keys: image, wrist_image, state.
    `action` / `state` (sim state for rollback) / reward / etc. are handled
    by the parent class and copied through unchanged.
    """

    def _process_obs(self, obs, info):
        robot = self.env.robots[0]

        # Resolve flat-dict keys once. DataPlaybackWrapper sets
        # flatten_obs_space=True so obs uses "::"-separated flat keys, not
        # nested dicts like RLinf's sentinel_env.
        if not getattr(self, "_keys_resolved", False):
            self._main_key = _find_main_key(obs)
            self._wrist_key = _find_wrist_key(obs, robot.name)
            print(f"[Playback] main_key = {self._main_key}")
            print(f"[Playback] wrist_key = {self._wrist_key or '<none — will emit zeros>'}")
            self._keys_resolved = True

        # 8D policy state: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2).
        # Matches IsaacLab-Stack-Cube's _wrap_obs so Pi0.5 ckpts trained on
        # that dataset (state/action projection dims, norm_stats layout)
        # transfer cleanly as SFT init.
        eef_pos = robot.get_relative_eef_position().to(th.float32)
        eef_quat = robot.get_relative_eef_orientation().to(th.float32)
        eef_axisangle = _quat2axisangle(eef_quat)
        gripper_idx = robot.gripper_control_idx[robot.default_arm]
        gripper_qpos = robot.get_joint_positions()[gripper_idx].to(th.float32)
        state = th.cat([eef_pos, eef_axisangle, gripper_qpos])

        # Main + wrist camera RGB (drop alpha, uint8 HWC).
        main_rgb = obs[self._main_key][..., :3].to(th.uint8)
        if self._wrist_key is not None:
            wrist_rgb = obs[self._wrist_key][..., :3].to(th.uint8)
        else:
            wrist_rgb = th.zeros_like(main_rgb)

        return {
            "image": main_rgb,
            "wrist_image": wrist_rgb,
            "state": state,
        }


def _dump_mp4s(hdf5_path: str, fps: int = 30) -> None:
    """Write {stem}_main.mp4 and {stem}_wrist.mp4 from an obs-rendered HDF5."""
    import h5py
    import imageio.v2 as imageio

    stem, _ = os.path.splitext(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if not demo_keys:
            print("[Playback] No demos in HDF5, skipping MP4 export")
            return
        demo = f["data"][demo_keys[0]]
        main = demo["obs/image"][:]
        wrist = demo["obs/wrist_image"][:]

    for suffix, frames in (("main", main), ("wrist", wrist)):
        path = f"{stem}_{suffix}.mp4"
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"[Playback] MP4: {path}  ({len(frames)} frames)")


def _setup_cameras_from_scene(env) -> None:
    """Position cam_opposite like teleop did.

    Mirrors so101_franka_teleop.py: find support_surface + a target object,
    call build_video_view_specs + setup_cameras to place the external
    camera(s) at the canonical "opposite-side overview" pose, and
    synchronise the viewer camera to match.
    """
    try:
        from sentinel.task_generation.utils.video import (
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
    if support_obj is None:
        print("[Playback] WARNING: scene has no 'support_surface' object; "
              "cam_opposite stays at default pose (teleop placement needs it).")
        return

    # Pick any non-robot, non-support object as the "target" for the lookat.
    target_obj = next(
        (obj for obj in scene.objects if obj is not robot and obj is not support_obj),
        support_obj,
    )
    views = build_video_view_specs(
        None, robot, target_obj, support_obj=support_obj,
    )
    setup_cameras(env, views)
    print(f"[Playback] Positioned cam_opposite via setup_cameras "
          f"(target={target_obj.name}, support={support_obj.name}).")


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
                   help="Also dump {stem}_main.mp4 and {stem}_wrist.mp4 next to "
                        "the rendered HDF5 for quick visual review.")
    args = p.parse_args()

    # DataPlaybackWrapper precondition.
    gm.ENABLE_TRANSITION_RULES = False

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Force 256x256 for both main and wrist cameras. The teleop HDF5 was
    # recorded at Kit's default (~1280x720) but we need square 256x256 for
    # the Pi0.5 / OmniGibsonDataConfig pipeline.
    external_cfg = build_external_camera_configs(
        names=("cam_opposite",),
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

    env = SentinelPlaybackWriter.create_from_hdf5(
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
    print(f"[Playback] Wrote: {args.output}")

    if args.save_mp4:
        _dump_mp4s(args.output)

    og.clear()


if __name__ == "__main__":
    main()
