#!/usr/bin/env python3
"""Replay a saved pick_and_place trajectory through OmniGibson, capturing
SFT-ready (obs, action) pairs at each env.step into an HDF5 file plus three
review MP4s.

Inputs (per seed):
    <collect_root>/task_XXXX/seed_YY/trajectory.pt        (from pick_and_place)
    <collect_root>/task_XXXX/seed_YY/result.json          (only used for sanity)

Outputs (per seed):
    <out_root>/task_XXXX/seed_YY/rollout.hdf5  with:
        data/demo_0/obs/image_left   uint8 (T, H, W, 3)   cam_left RGB
        data/demo_0/obs/image_right  uint8 (T, H, W, 3)   cam_right RGB
        data/demo_0/obs/wrist_image  uint8 (T, H, W, 3)   robot wrist RGB
        data/demo_0/obs/state        float32 (T, 8)       [eef_pos(3) | eef_aa(3) | gripper_q(2)]
        data/demo_0/action           float32 (T, 7)       OSC [dpos(3) | daa(3) | gripper_cmd(1)]
        data/demo_0/done             bool   (T,)
        data/<top-level attrs>: task_dir, seed, target_name, n_steps
    <out_root>/task_XXXX/seed_YY/rollout_image_left.mp4
    <out_root>/task_XXXX/seed_YY/rollout_image_right.mp4
    <out_root>/task_XXXX/seed_YY/rollout_wrist.mp4

Phase A (approach + close) is replayed via direct ``robot.set_joint_positions``
plus ``og.sim.step()``, NOT through ``env.step``. The HDF5 only contains
Phase B (held → goal transport + final settle), which is the SFT-relevant
half: "given a held object, navigate it to the goal sphere." Recording
discrete kinematic teleports as actions would be dishonest for SFT anyway.

Cameras:
    External: cam_left, cam_right       (positioned from diagnostics.cameras)
    Robot   : wrist (rgb on eef link)

Usage::

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        conda run -n behavior python -m tools.render_pnp_for_sft \\
            --collect-dir outputs/pnp_multitask_collect/task_0000/seed_00 \\
            --out-dir outputs/pnp_sft/task_0000/seed_00
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing ``sentinel`` triggers the OG patch path (long-finger Franka etc.).
import sentinel  # noqa: F401


_WRIST_CAM_PATCHED = False


def _install_wrist_camera_patch():
    """Monkey-patch FrankaPanda._load_sensors to attach a Camera prim under
    panda_hand BEFORE the sensor scanner walks link children. The pose
    matches franka_mounted.usda:2918 so the rendered wrist view is
    identical to the one task_generation already uses.
    """
    global _WRIST_CAM_PATCHED
    if _WRIST_CAM_PATCHED:
        return
    from omnigibson.robots.franka import FrankaPanda
    import omnigibson.lazy as lazy

    _orig_load_sensors = FrankaPanda._load_sensors

    def _load_sensors_with_wrist(self):
        try:
            stage = lazy.isaacsim.core.utils.stage.get_current_stage()
            hand = self._links.get("panda_hand") if self._links else None
            if hand is not None:
                cam_path = f"{hand.prim_path}/Camera"
                if not stage.GetPrimAtPath(cam_path).IsValid():
                    cam_prim = lazy.pxr.UsdGeom.Camera.Define(stage, cam_path).GetPrim()
                    xformable = lazy.pxr.UsdGeom.Xformable(cam_prim)
                    xformable.ClearXformOpOrder()
                    t_op = xformable.AddXformOp(
                        lazy.pxr.UsdGeom.XformOp.TypeTranslate,
                        lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
                    t_op.Set(lazy.pxr.Gf.Vec3d(0.05, 0.0, -0.13))
                    r_op = xformable.AddXformOp(
                        lazy.pxr.UsdGeom.XformOp.TypeOrient,
                        lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
                    r_op.Set(lazy.pxr.Gf.Quatd(-0.0923, -0.701, -0.701, -0.0923))
                    s_op = xformable.AddXformOp(
                        lazy.pxr.UsdGeom.XformOp.TypeScale,
                        lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
                    s_op.Set(lazy.pxr.Gf.Vec3d(1.0, 1.0, 1.0))
                    xformable.SetXformOpOrder([t_op, r_op, s_op])
                    cam_prim.CreateAttribute(
                        "focalLength", lazy.pxr.Sdf.ValueTypeNames.Float
                    ).Set(17.0)
                    cam_prim.CreateAttribute(
                        "clippingRange", lazy.pxr.Sdf.ValueTypeNames.Float2
                    ).Set(lazy.pxr.Gf.Vec2f(0.001, 1000000.0))
                    print(f"[render]   injected wrist Camera prim at {cam_path}",
                          flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[render]   wrist-cam injection failed: {e}", flush=True)
        return _orig_load_sensors(self)

    FrankaPanda._load_sensors = _load_sensors_with_wrist
    _WRIST_CAM_PATCHED = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collect-dir", type=Path, required=True,
                   help="Path to outputs/pnp_multitask_collect/task_XXXX/seed_YY")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write rollout.hdf5 + MP4s")
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--resolution", type=int, default=256,
                   help="Square frame side length for all three cameras.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--substeps-per-wp", type=int, default=6)
    p.add_argument("--inner-pos-tol", type=float, default=0.005)
    p.add_argument("--inner-rot-tol-rad", type=float, default=0.05)
    p.add_argument("--final-settle-steps", type=int, default=100)
    p.add_argument("--final-pos-tol", type=float, default=0.01)
    p.add_argument("--final-rot-tol-rad", type=float, default=0.05)
    return p.parse_args()


def _quat_canonical(q_xyzw):
    """Return ±q so that w >= 0 (canonical shortest-path rep)."""
    if float(q_xyzw[3].item()) < 0.0:
        return -q_xyzw
    return q_xyzw


def _quat2axisangle_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternion to axis-angle (3,) numpy."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    q = np.clip(q, -1.0, 1.0)
    w = float(q[3])
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(max(-1.0, min(1.0, w))))
    axis = q[:3] / sin_half
    return (axis * angle).astype(np.float32)


def _init_og(headless: bool = True):
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if headless:
        gm.HEADLESS = True
    import omnigibson as og
    return og


def _build_env_for_replay(task_dir: Path, episode: int, resolution: int):
    """Mirror pick_and_place_from_dataset._build_env but with our camera set:
    external = (cam_left, cam_right) at the diagnostics-recorded poses, plus
    wrist RGB on the robot. No other changes.
    """
    import omnigibson as og
    import torch as th
    from tools.replay_empty_from_dataset import (
        _build_object_cfg, _build_robot_cfg,
        _identify_task_objects, _load_diagnostics_row, _load_scene_info,
    )
    from sentinel.envs.frozen_task_runtime import (
        configure_review_sensors, extract_scene_robot_setup,
        position_diagnostics_cameras,
    )
    from sentinel.utils.camera_setup import build_external_camera_configs
    from sentinel.utils.goal_region import GoalRegionSpec, spawn_goal_region_marker

    diagnostics = _load_diagnostics_row(task_dir, episode)
    scene_info = _load_scene_info(task_dir, episode)
    task_names = _identify_task_objects(scene_info, diagnostics)

    object_cfgs = [_build_object_cfg(task_names[0], scene_info, fixed_base=True)]
    object_cfgs += [_build_object_cfg(n, scene_info, fixed_base=False)
                    for n in task_names[1:]]

    robot_setup = extract_scene_robot_setup(scene_info)
    robot_cfg = _build_robot_cfg(robot_setup)
    # Keep FrankaPanda — the trajectory was planned for FrankaPanda's base
    # frame. The wrist camera comes from a Camera prim we inject under
    # panda_hand via _install_wrist_camera_patch() before env construction.
    # Match pick_and_place_from_dataset's runtime overrides — OSC arm,
    # raw-units actions, MultiFingerGripper, assisted grasping.
    robot_cfg["grasping_mode"] = "assisted"
    robot_cfg["self_collisions"] = True
    robot_cfg["action_normalize"] = False
    # Robot obs = rgb so robot._load_sensors picks up the panda_hand camera.
    robot_cfg["obs_modalities"] = ["rgb"]
    # Force wrist resolution so it matches the externals.
    robot_cfg["sensor_config"] = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_height": resolution,
                "image_width": resolution,
            },
        },
    }
    _OSC_LIMITS = (
        (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
        (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
    )
    robot_cfg["controller_config"] = {
        "arm_0": {
            "name": "OperationalSpaceController",
            "command_input_limits": _OSC_LIMITS,
            "command_output_limits": _OSC_LIMITS,
        },
        "gripper_0": {"name": "MultiFingerGripperController"},
    }

    # External overviews: cam_left + cam_right (positioned from diagnostics).
    # The wrist camera comes from FrankaMounted's USD-parented panda_hand
    # camera — auto-discovered via robot._load_sensors(); no external prim
    # needed.
    diag_cam_names = [c["sensor_name"] for c in diagnostics.get("cameras", [])
                      if c.get("sensor_name")]
    diag_cam_names_for_overview = [n for n in ("cam_left", "cam_right")
                                   if n in diag_cam_names]
    if not diag_cam_names_for_overview:
        raise RuntimeError(
            f"Neither cam_left nor cam_right present in diagnostics cameras: "
            f"{diag_cam_names}")
    external_sensors = build_external_camera_configs(
        names=diag_cam_names_for_overview, resolution=resolution,
        modalities=("rgb",),
    )

    env_cfg = {
        "scene": {"type": "Scene"},
        "robots": [robot_cfg],
        "objects": object_cfgs,
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": external_sensors,
        },
    }

    env = og.Environment(configs=env_cfg)
    env.reset()

    for cfg in object_cfgs:
        obj = env.scene.object_registry("name", cfg["name"])
        if obj is None:
            continue
        obj.set_position_orientation(
            position=th.tensor(cfg["position"], dtype=th.float32),
            orientation=th.tensor(cfg["orientation"], dtype=th.float32),
        )
        if hasattr(obj, "keep_still"):
            obj.keep_still()

    robot = env.robots[0]
    if robot_setup.get("position") is not None:
        robot.set_position_orientation(
            position=th.tensor(robot_setup["position"], dtype=th.float32),
            orientation=th.tensor(robot_setup["orientation"], dtype=th.float32),
        )
    if hasattr(robot, "keep_still"):
        robot.keep_still()
    og.sim.step()

    gr_payload = diagnostics.get("goal_region")
    if gr_payload is not None:
        spawn_goal_region_marker(env, GoalRegionSpec.from_json(gr_payload))
        og.sim.step()

    configure_review_sensors(env)
    position_diagnostics_cameras(env, og, diagnostics, set_viewer=True)
    # Apply VisionSensor resolution to externals + robot sensors (Kit viewport
    # init can ignore sensor_kwargs).
    from omnigibson.sensors import VisionSensor
    for cam in (env.external_sensors or {}).values():
        cam.image_height = resolution
        cam.image_width = resolution
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = resolution
            sensor.image_width = resolution
    env.load_observation_space()
    for _ in range(10):
        og.sim.step()

    return env, og, diagnostics, diag_cam_names_for_overview


def _sensor_rgb_uint8(sensor, resolution: int) -> np.ndarray:
    """Read RGB uint8 HWC from a VisionSensor; returns zeros on failure."""
    try:
        obs_tuple = sensor.get_obs()
        rgb = obs_tuple[0].get("rgb") if isinstance(obs_tuple, tuple) else obs_tuple.get("rgb")
    except Exception:  # noqa: BLE001
        rgb = None
    if rgb is None:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rgb = rgb[..., :3]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach()
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.asarray(rgb, dtype=np.uint8)


def _find_wrist_sensor(robot):
    """Find the robot's wrist camera (panda_hand for FrankaMounted)."""
    from omnigibson.sensors import VisionSensor
    if not getattr(robot, "sensors", None):
        return None
    # Prefer one with 'hand' in the name (matches task_generation/utils/video.py).
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor) and "hand" in name.lower():
            return sensor
    # Any VisionSensor as fallback.
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            return sensor
    return None


def _grab_obs_frames(env, robot, overview_cam_names: list[str], resolution: int,
                     wrist_sensor=None):
    """Return uint8 frames for left, right, wrist via sensor objects directly.

    Externals (cam_left, cam_right) come from env.external_sensors; wrist
    comes from robot.sensors (FrankaMounted's panda_hand camera).
    """
    out = {}
    sens = env.external_sensors or {}
    for cam in overview_cam_names:
        s = sens.get(cam)
        if s is None:
            out[cam] = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        else:
            out[cam] = _sensor_rgb_uint8(s, resolution)
    out["wrist"] = (_sensor_rgb_uint8(wrist_sensor, resolution)
                    if wrist_sensor is not None
                    else np.zeros((resolution, resolution, 3), dtype=np.uint8))
    return out


def _grab_state(robot):
    """8D SFT state: [eef_pos(3) | eef_aa(3) | gripper_q(2)] float32 numpy."""
    import torch as th
    eef_pos = robot.get_relative_eef_position().detach().cpu().numpy().astype(np.float32)
    eef_quat = robot.get_relative_eef_orientation().detach().cpu().numpy().astype(np.float32)
    eef_aa = _quat2axisangle_np(eef_quat)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    if hasattr(gripper_idx, "cpu"):
        gripper_idx_np = gripper_idx.cpu().numpy()
    else:
        gripper_idx_np = np.asarray(gripper_idx)
    gripper_q = robot.get_joint_positions().detach().cpu().numpy()[gripper_idx_np].astype(np.float32)
    return np.concatenate([eef_pos, eef_aa, gripper_q], axis=0).astype(np.float32)


def _replay_phase_a(env, og, robot, target_obj, init_pos, init_quat,
                    approach_traj, *, settle_open_steps=8, close_steps=20,
                    frame_callback=None, prebuffer_steps=10):
    """Direct-set-joint-position replay of Phase A (NOT recorded in HDF5).

    Mirrors sentinel.rl.grasps.collector.run_grasp_attempt:
      - Pin target to init pose
      - Replay each approach waypoint via set_joint_positions + og.sim.step
      - Settle open + close gripper while pinned
      - Verify AG engaged on target

    If ``frame_callback`` is given, it is invoked after every og.sim.step()
    so the caller can append video frames (shows the home pose, approach,
    grasp, and close in the rendered MP4). HDF5 is not written here.

    ``prebuffer_steps`` adds ``N`` extra steps at the start with the robot
    at its restored initial joint state — the rendered video opens on a
    clearly recognizable home pose for a moment before motion begins.

    Returns True if AG holds the target at the end, else False.
    """
    import torch as th
    from omnigibson.controllers.controller_base import IsGraspingState
    from sentinel.rl.grasps.collector import (
        _reset_controller_goals, _build_hold_action, _phase1_step,
    )

    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    open_gripper_q = robot.joint_upper_limits[gripper_control_idx].clone()
    initial_q = robot.get_joint_positions().clone()

    robot.release_grasp_immediately(arm)
    robot.set_joint_positions(initial_q)
    target_obj.set_position_orientation(position=init_pos, orientation=init_quat)
    target_obj.root_link.disable_gravity()
    target_obj.root_link.set_linear_velocity(th.zeros(3))
    target_obj.root_link.set_angular_velocity(th.zeros(3))
    _reset_controller_goals(robot)
    # Update cam_wrist BEFORE the step so it captures the home pose, then
    # the sim step's own render baked the new camera pose into the frame.
    _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
    if frame_callback is not None:
        frame_callback()

    robot.set_joint_positions(open_gripper_q, gripper_control_idx)
    for waypoint in approach_traj:
        robot.set_joint_positions(waypoint, arm_control_idx)
        robot.set_joint_positions(open_gripper_q, gripper_control_idx)
        _reset_controller_goals(robot)
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
        if frame_callback is not None:
            frame_callback()

    robot.set_joint_positions(open_gripper_q, gripper_control_idx)
    _reset_controller_goals(robot)
    for _ in range(settle_open_steps):
        robot.apply_action(_build_hold_action(robot, gripper_open=True))
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
        if frame_callback is not None:
            frame_callback()
    for _ in range(close_steps):
        robot.apply_action(_build_hold_action(robot, gripper_open=False))
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
        if frame_callback is not None:
            frame_callback()

    held = robot.is_grasping(arm, target_obj) == IsGraspingState.TRUE
    # Release the pin (re-enable gravity AFTER Phase B starts is too late —
    # do it now so the transport replay sees real physics).
    target_obj.root_link.enable_gravity()
    return held


def _replay_phase_b_record(env, og, robot, eef_traj_base, *,
                           ext_cam_names, resolution,
                           wrist_sensor=None,
                           substeps_per_wp=6, pos_tol=0.005, rot_tol_rad=0.05,
                           final_settle_steps=100, final_pos_tol=0.01,
                           final_rot_tol_rad=0.05,
                           frame_callback=None):
    """OSC pose_delta_ori replay of Phase B with per-step (obs, action) recording.

    Mirrors tools.pick_and_place_from_dataset._replay_holding. Returns a dict
    of stacked arrays ready for HDF5 emission. If ``frame_callback`` is
    given, it is invoked with the dict ``{"image_left", "image_right",
    "wrist"}`` after every env.step so the caller can append video frames
    independently of the HDF5 buffering.
    """
    import torch as th
    import omnigibson.utils.transform_utils as T

    arm = robot.default_arm
    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]

    frames_left, frames_right, frames_wrist = [], [], []
    states, actions, dones = [], [], []

    def _step_and_record(target_pos_b, target_quat_b, last: bool):
        cur_pos_b, cur_quat_b = robot.get_relative_eef_pose(arm)
        cur_pos_b = cur_pos_b.float()
        cur_quat_b = cur_quat_b.float()
        dpos = target_pos_b - cur_pos_b
        q_target = _quat_canonical(target_quat_b)
        q_cur = _quat_canonical(cur_quat_b)
        q_inv = T.quat_inverse(q_cur)
        q_delta = _quat_canonical(T.quat_multiply(q_target, q_inv))
        daa = T.quat2axisangle(q_delta)
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[arm_action_idx] = th.cat([dpos, daa])
        action[gripper_action_idx] = -1.0
        env.step(action)
        obs_frames = _grab_obs_frames(env, robot, ext_cam_names, resolution,
                                      wrist_sensor=wrist_sensor)
        frames_left.append(obs_frames[ext_cam_names[0]])
        if len(ext_cam_names) > 1:
            frames_right.append(obs_frames[ext_cam_names[1]])
        else:
            frames_right.append(np.zeros_like(obs_frames[ext_cam_names[0]]))
        frames_wrist.append(obs_frames["wrist"])
        if frame_callback is not None:
            frame_callback(obs_frames)
        states.append(_grab_state(robot))
        # Action: [dpos(3) | daa(3) | gripper_cmd(1)] = 7D.
        act_np = np.zeros(7, dtype=np.float32)
        act_np[:3] = dpos.detach().cpu().numpy().astype(np.float32)
        act_np[3:6] = daa.detach().cpu().numpy().astype(np.float32)
        act_np[6] = -1.0
        actions.append(act_np)
        dones.append(bool(last))
        return float(th.norm(dpos).item()), float(th.norm(daa).item())

    n = len(eef_traj_base)
    print(f"[render] Phase B replay: {n} eef waypoints", flush=True)
    last_pos_err = last_rot_err = 0.0
    for wi in range(n):
        target_pos_b = eef_traj_base[wi, :3].float()
        target_quat_b = eef_traj_base[wi, 3:7].float()
        for k in range(substeps_per_wp):
            is_last = (wi == n - 1) and (k == substeps_per_wp - 1)
            last_pos_err, last_rot_err = _step_and_record(
                target_pos_b, target_quat_b, last=is_last)
            if last_pos_err < pos_tol and last_rot_err < rot_tol_rad:
                break

    # Final settle on last waypoint.
    target_pos_b = eef_traj_base[-1, :3].float()
    target_quat_b = eef_traj_base[-1, 3:7].float()
    for k in range(final_settle_steps):
        is_last = (k == final_settle_steps - 1)
        last_pos_err, last_rot_err = _step_and_record(
            target_pos_b, target_quat_b, last=is_last)
        if last_pos_err < final_pos_tol and last_rot_err < final_rot_tol_rad:
            # Mark this final step as done.
            dones[-1] = True
            break
    print(f"[render]   recorded {len(actions)} env.step events  "
          f"final pos_err={last_pos_err:.4f}m rot_err={last_rot_err:.4f}rad",
          flush=True)
    return {
        "image_left": np.stack(frames_left, axis=0),
        "image_right": np.stack(frames_right, axis=0),
        "wrist_image": np.stack(frames_wrist, axis=0),
        "state": np.stack(states, axis=0),
        "action": np.stack(actions, axis=0),
        "done": np.array(dones, dtype=bool),
    }


def _write_hdf5(out_path: Path, data: dict, attrs: dict):
    import h5py
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        grp = f.create_group("data/demo_0")
        obs = grp.create_group("obs")
        obs.create_dataset("image_left",  data=data["image_left"],  compression="gzip", compression_opts=4)
        obs.create_dataset("image_right", data=data["image_right"], compression="gzip", compression_opts=4)
        obs.create_dataset("wrist_image", data=data["wrist_image"], compression="gzip", compression_opts=4)
        obs.create_dataset("state",       data=data["state"])
        grp.create_dataset("action", data=data["action"])
        grp.create_dataset("done",   data=data["done"])
    print(f"[render] wrote {out_path}  ({len(data['action'])} steps)", flush=True)


class _MP4Streamer:
    """Three open MP4 writers (left, right, wrist). Frames are appended in
    real time so Phase A motion shows up in the final videos even though
    Phase A is not in the HDF5."""

    def __init__(self, out_dir: Path, fps: int):
        import imageio.v2 as imageio
        out_dir.mkdir(parents=True, exist_ok=True)
        self._paths = {
            "image_left":  out_dir / "rollout_image_left.mp4",
            "image_right": out_dir / "rollout_image_right.mp4",
            "wrist":       out_dir / "rollout_wrist.mp4",
        }
        self._writers = {
            k: imageio.get_writer(str(p), fps=fps, codec="libx264", quality=7)
            for k, p in self._paths.items()
        }
        self.n_frames = 0

    def append(self, obs_frames: dict):
        """obs_frames keys: <left_cam_name>, <right_cam_name>, 'wrist'."""
        # The overview cameras are keyed by their actual names in
        # _grab_obs_frames output. The caller passes a normalized dict with
        # the canonical keys 'image_left', 'image_right', 'wrist'.
        self._writers["image_left"].append_data(obs_frames["image_left"])
        self._writers["image_right"].append_data(obs_frames["image_right"])
        self._writers["wrist"].append_data(obs_frames["wrist"])
        self.n_frames += 1

    def close(self):
        for k, w in self._writers.items():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        for p in self._paths.values():
            print(f"[render]   wrote {p}", flush=True)


def _obs_to_canonical(obs_frames: dict, overview_cam_names: list[str]) -> dict:
    """Re-key _grab_obs_frames output to canonical {image_left,image_right,wrist}."""
    return {
        "image_left":  obs_frames[overview_cam_names[0]],
        "image_right": (obs_frames[overview_cam_names[1]]
                        if len(overview_cam_names) > 1
                        else np.zeros_like(obs_frames[overview_cam_names[0]])),
        "wrist":       obs_frames["wrist"],
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover task_dir from the collect_dir's result.json
    res_path = args.collect_dir / "result.json"
    if not res_path.is_file():
        raise SystemExit(f"no result.json at {res_path}")
    result = json.loads(res_path.read_text())
    task_dir = Path(result["task_dir"])
    target_name = result.get("target_name") or ""
    seed_label = args.collect_dir.name

    traj_path = args.collect_dir / "trajectory.pt"
    if not traj_path.is_file():
        raise SystemExit(f"no trajectory.pt at {traj_path}")
    import torch as th
    traj = th.load(traj_path, weights_only=False)
    approach_traj = traj["approach_traj"]
    transport_eef_traj = traj["transport_eef_traj_base"]
    print(f"[render] {args.collect_dir} → {args.out_dir}", flush=True)
    print(f"[render] task_dir={task_dir.name}  target={target_name}  "
          f"approach_len={len(approach_traj)}  transport_len={len(transport_eef_traj)}",
          flush=True)

    og = _init_og(headless=True)
    _install_wrist_camera_patch()
    env, og, diagnostics, ext_cam_names = _build_env_for_replay(
        task_dir, args.episode, args.resolution,
    )

    try:
        robot = env.robots[0]
        target_obj = env.scene.object_registry("name", target_name)
        if target_obj is None:
            raise RuntimeError(f"target {target_name!r} not in registry")
        init_pos, init_quat = target_obj.get_position_orientation()
        init_pos = init_pos.clone(); init_quat = init_quat.clone()

        wrist_sensor = _find_wrist_sensor(robot)
        wrist_name = (getattr(wrist_sensor, "name", None)
                      if wrist_sensor is not None else None)
        print(f"[render] overview_cams={ext_cam_names}  "
              f"wrist_sensor={wrist_name or '<none>'}", flush=True)

        # Open MP4 streamers BEFORE Phase A so the videos cover the full
        # rollout (home → approach → grasp → transport → settle).
        streamer = _MP4Streamer(args.out_dir, args.video_fps)

        def _phase_a_frame_cb():
            obs_frames = _grab_obs_frames(env, robot, ext_cam_names,
                                          args.resolution,
                                          wrist_sensor=wrist_sensor)
            streamer.append(_obs_to_canonical(obs_frames, ext_cam_names))

        def _phase_b_frame_cb(obs_frames):
            streamer.append(_obs_to_canonical(obs_frames, ext_cam_names))

        # Phase A — direct kinematic replay; frames go to MP4 only.
        held = _replay_phase_a(
            env, og, robot, target_obj, init_pos, init_quat, approach_traj,
            frame_callback=_phase_a_frame_cb,
        )
        if not held:
            print("[render] Phase A did not result in AG hold — "
                  "Phase B will execute but the demo is invalid.", flush=True)

        # Phase B — OSC replay with per-step (obs, action) HDF5 recording.
        # Frames also stream to the MP4s via the callback.
        data = _replay_phase_b_record(
            env, og, robot, transport_eef_traj,
            ext_cam_names=ext_cam_names,
            resolution=args.resolution,
            wrist_sensor=wrist_sensor,
            substeps_per_wp=args.substeps_per_wp,
            pos_tol=args.inner_pos_tol, rot_tol_rad=args.inner_rot_tol_rad,
            final_settle_steps=args.final_settle_steps,
            final_pos_tol=args.final_pos_tol,
            final_rot_tol_rad=args.final_rot_tol_rad,
            frame_callback=_phase_b_frame_cb,
        )

        attrs = {
            "task_dir": str(task_dir),
            "target_name": target_name,
            "seed": seed_label,
            "n_steps": int(len(data["action"])),
            "n_video_frames": int(streamer.n_frames),
            "phase_a_held": bool(held),
            "ext_cam_left": ext_cam_names[0] if len(ext_cam_names) > 0 else "",
            "ext_cam_right": ext_cam_names[1] if len(ext_cam_names) > 1 else "",
        }
        _write_hdf5(args.out_dir / "rollout.hdf5", data, attrs)
        streamer.close()
        print(f"[render]   video frames total: {streamer.n_frames}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[render] FAIL: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
