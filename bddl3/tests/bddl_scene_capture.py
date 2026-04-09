"""
Record BDDL-generated BEHAVIOR task scenes to video.

Loads a single BDDL activity by name, creates the OmniGibson Behavior env,
runs a short zero-action rollout, and records either the viewer (interface)
camera and/or the R1Pro agent cameras into separate videos under
activity-named subfolders.

Intended for headless use on SSH servers: set OMNIGIBSON_HEADLESS=1.
Default save path: videos/ next to this script (bddl3/tests/videos/).
"""

import argparse
import datetime
import os
import sys

import cv2
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BDDL3_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BDDL3_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video

DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
VIEWER_RESOLUTION = (720, 1280)
HEAD_RECORD_RES = 720  # upscale head camera to this before recording
AGENT_VIDEO_RESOLUTION = (HEAD_RECORD_RES, HEAD_RECORD_RES)  # head-only square
DEFAULT_FPS = 30
DEFAULT_DURATION_SEC = 5.0
gm.USE_GPU_DYNAMICS = True


# ---------------------------------------------------------------------------
# Head camera helpers
# ---------------------------------------------------------------------------

def _get_head_sensor(robot):
    """Find the head (zed) VisionSensor from robot.sensors by name substring."""
    for name, sensor in robot.sensors.items():
        if "zed" in name.lower():
            return name, sensor
    return None, None


def _upgrade_head_camera(robot, target_res=HEAD_RECORD_RES):
    """Increase head camera resolution at runtime for sharper recording."""
    name, sensor = _get_head_sensor(robot)
    if sensor is None:
        return
    sensor.image_height = target_res
    sensor.image_width = target_res
    print(f"[bddl_scene_capture] Head camera '{name}' resolution -> {target_res}x{target_res}")


def _get_head_rgb(obs, env):
    """Extract head camera RGB from obs. Returns (H,W,3) numpy or None."""
    robot_name = env.robots[0].name if (env and env.robots) else None
    if robot_name and robot_name in obs and isinstance(obs[robot_name], dict):
        for sensor_key, sensor_data in obs[robot_name].items():
            if "zed" in sensor_key.lower() and isinstance(sensor_data, dict) and "rgb" in sensor_data:
                return _to_np_rgb(sensor_data["rgb"])
    # Flattened fallback
    for key, val in obs.items():
        if "zed" in key.lower() and "rgb" in key.lower() and "::" in key:
            return _to_np_rgb(val)
    return None


def _to_np_rgb(rgb):
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    return rgb


def _resize_uint8(img, h, w):
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------------------
# Viewer camera placement
# ---------------------------------------------------------------------------

def _get_object_position(entity):
    """Get (x,y,z) numpy position from a BDDLEntity, or None on failure."""
    from omnigibson.object_states import Pose
    try:
        val = entity.get_state(Pose) if hasattr(entity, "get_state") else None
        if val is False or val is None:
            val = entity.states[Pose].get_value()
        if val is not None and val is not False:
            p = val[0]
            if hasattr(p, "cpu"):
                p = p.cpu().numpy()
            return np.asarray(p, dtype=np.float64).flatten()[:3]
    except Exception:
        pass
    return None


def _compute_focus_from_scope(env, focus_object_names=None):
    """
    Compute focus point from BDDL object_scope.

    Args:
        env: OmniGibson environment (after reset).
        focus_object_names: If provided, list of object name substrings to
            filter on (e.g. ["stove", "frying_pan"]). Only objects whose
            BDDL instance name contains any of these substrings are included.
            If None, all non-agent existing objects are used.

    Returns:
        numpy (3,) focus position, or None.
    """
    scope = getattr(getattr(env, "task", None), "object_scope", None)
    if not scope:
        return None

    positions = []
    matched_names = []
    for inst, entity in scope.items():
        if "agent" in inst.lower() or not getattr(entity, "exists", False):
            continue
        if focus_object_names is not None:
            inst_lower = inst.lower()
            if not any(name.lower() in inst_lower for name in focus_object_names):
                continue
        p = _get_object_position(entity)
        if p is not None:
            positions.append(p)
            matched_names.append(inst)

    if matched_names:
        print(f"[bddl_scene_capture] Focus objects ({len(matched_names)}): {matched_names}")
        for name, pos in zip(matched_names, positions):
            print(f"  {name}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    if not positions:
        if focus_object_names:
            all_insts = [k for k in scope.keys() if "agent" not in k.lower()]
            print(f"[bddl_scene_capture] WARNING: --focus_objects {focus_object_names} matched nothing. "
                  f"Available: {all_insts}")
        return None

    return np.mean(positions, axis=0)


def _compute_viewer_camera_pose(env, focus_object_names=None,
                                behind_offset=1.0, up_offset=0.3):
    """
    Place viewer camera slightly behind and above the robot's head,
    looking at the centroid of selected activity objects.

    Args:
        focus_object_names: Optional list of BDDL object name substrings
            to use as focus (e.g. ["stove", "frying_pan"]).
            If None, all non-agent objects are averaged.
    """
    robot = env.robots[0]
    head_name, head_sensor = _get_head_sensor(robot)
    if head_sensor is not None:
        try:
            head_pos, _ = head_sensor.get_position_orientation()
            if hasattr(head_pos, "cpu"):
                head_pos = head_pos.cpu().numpy()
            head_pos = np.asarray(head_pos, dtype=np.float64).flatten()[:3]
        except Exception:
            head_pos = None
    else:
        head_pos = None

    if head_pos is None:
        try:
            head_pos, _ = robot.get_position_orientation()
            if hasattr(head_pos, "cpu"):
                head_pos = head_pos.cpu().numpy()
            head_pos = np.asarray(head_pos, dtype=np.float64).flatten()[:3]
            head_pos[2] += 1.2
        except Exception:
            return None

    focus = _compute_focus_from_scope(env, focus_object_names)
    if focus is None:
        focus = head_pos.copy()
        focus[2] -= 0.5

    diff = head_pos - focus
    diff[2] = 0.0
    if np.linalg.norm(diff) < 0.05:
        diff = np.array([-1.0, 0.0, 0.0])
    diff = diff / np.linalg.norm(diff)

    cam_pos = head_pos + behind_offset * diff
    cam_pos[2] = head_pos[2] + up_offset

    forward = focus - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    quat = _look_at_quat(forward)

    return cam_pos.tolist(), quat.tolist()


def _apply_robot_offset(env, offset_meters, focus_object_names=None):
    """
    Move the robot toward (positive) or away from (negative) the focus point.
    This changes head camera distance to objects. Call after env.reset().
    """
    if offset_meters == 0.0:
        return
    focus = _compute_focus_from_scope(env, focus_object_names)
    if focus is None:
        return
    robot = env.robots[0]
    try:
        pos, ori = robot.get_position_orientation()
        if hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
        pos = np.asarray(pos, dtype=np.float64).flatten()[:3]
    except Exception:
        return
    diff = focus - pos
    diff[2] = 0.0
    if np.linalg.norm(diff) < 1e-6:
        return
    unit = diff / np.linalg.norm(diff)
    new_pos = pos + offset_meters * unit
    new_pos = new_pos.tolist()
    try:
        robot.set_position_orientation(new_pos, ori)
        print(f"[bddl_scene_capture] Robot offset {offset_meters} m toward focus -> head closer to objects.")
    except Exception as e:
        print(f"[bddl_scene_capture] Robot offset failed: {e}")


def _look_at_quat(forward, up=None):
    """Convert forward direction to (x,y,z,w) quaternion."""
    from scipy.spatial.transform import Rotation
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-9)
    new_up = np.cross(forward, right)
    R = np.stack([forward, right, new_up], axis=-1)
    return Rotation.from_matrix(R).as_quat()  # (x,y,z,w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Record BDDL activity scene to video(s) (viewer and/or R1Pro agent cameras)."
    )
    parser.add_argument(
        "--activity", type=str, required=True,
        help="BDDL activity name (e.g. transfer_hot_pan_safely).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_VIDEO_DIR,
        help=f"Root directory for video outputs. Default: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_SEC,
        help=f"Recording duration in seconds. Default: {DEFAULT_DURATION_SEC}",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=f"Frames per second. Default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--activity_definition_id", type=int, default=0,
    )
    parser.add_argument(
        "--no_viewer_video", action="store_true",
        help="Do not save the interface (viewer) video. Default: save both.",
    )
    parser.add_argument(
        "--no_agent_video", action="store_true",
        help="Do not save the agent (head camera) video. Default: save both.",
    )
    parser.add_argument(
        "--robot_offset", type=float, default=0.0,
        help="Move the robot this many meters toward (positive) or away from (negative) "
             "the focus point after reset. This changes head camera distance to objects. Default: 0.",
    )
    parser.add_argument(
        "--activity_instance_id", type=int, default=0,
    )
    parser.add_argument(
        "--no_online_sampling", action="store_true",
        help="Use pre-cached task template (only for official activities with pre-generated scene JSON).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to base behavior YAML config.",
    )
    parser.add_argument(
        "--scene_model", type=str, default=None,
        help="Override scene model (e.g. Rs_int).",
    )
    parser.add_argument(
        "--focus_objects", type=str, nargs="+", default=None,
        help="One or more BDDL object name substrings to focus the camera on "
             "(e.g. --focus_objects stove frying_pan). The viewer camera will "
             "look at the centroid of matched objects. Matching is case-insensitive "
             "substring. If omitted, all non-agent objects are averaged.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    activity_name = args.activity.strip()
    if not activity_name:
        raise ValueError("--activity must be non-empty.")

    # Resolve base config
    if args.config and os.path.isfile(args.config):
        config_path = args.config
    else:
        config_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Override task for this activity
    cfg["task"]["activity_name"] = activity_name
    cfg["task"]["activity_definition_id"] = args.activity_definition_id
    cfg["task"]["activity_instance_id"] = args.activity_instance_id
    # Default: online sampling (so custom BDDL works without pre-cached scene JSON)
    cfg["task"]["online_object_sampling"] = not args.no_online_sampling
    cfg["task"]["use_presampled_robot_pose"] = args.no_online_sampling
    if args.scene_model is not None:
        cfg["scene"]["scene_model"] = args.scene_model

    # Output dirs: output_dir / activity_name / viewer  and  output_dir / activity_name / agent
    base_out = os.path.join(args.output_dir, activity_name)
    viewer_dir = os.path.join(base_out, "viewer")
    agent_dir = os.path.join(base_out, "agent")
    os.makedirs(viewer_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)

    # Headless setup
    if os.environ.get("OMNIGIBSON_HEADLESS", "").lower() in ("1", "true", "t"):
        with gm.unlocked():
            gm.HEADLESS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.RENDER_VIEWER_CAMERA = True

    print(f"[bddl_scene_capture] activity={activity_name}, duration={args.duration}s, "
          f"fps={args.fps}, output={base_out}")

    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    print(f"[bddl_scene_capture] Environment created. Robot: {robot.name}")

    # Upgrade head camera resolution for sharper recording
    _upgrade_head_camera(robot, target_res=HEAD_RECORD_RES)
    env.load_observation_space()

    num_steps = max(1, int(round(args.duration * args.fps)))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    video_writer_viewer = None
    video_writer_agent = None

    def _zero_action():
        if getattr(env.action_space, "spaces", None) is not None:
            return {
                r.name: np.zeros(r.action_space.shape, dtype=getattr(r.action_space, "dtype", np.float32))
                for r in env.robots
            }
        return np.zeros(env.action_space.shape, dtype=getattr(env.action_space, "dtype", np.float32))

    try:
        obs, info = env.reset()
        print(f"[bddl_scene_capture] env.reset() done.")

        robot_name = robot.name
        if robot_name in obs and isinstance(obs[robot_name], dict):
            print(f"[bddl_scene_capture] obs['{robot_name}'] sensors: {list(obs[robot_name].keys())}")

        zero_action = _zero_action()

        # Optional: move robot toward/away from focus to change head camera distance to objects
        if getattr(args, "robot_offset", 0.0) != 0.0:
            _apply_robot_offset(env, args.robot_offset, args.focus_objects)

        # Viewer camera: near the robot head, looking at activity objects
        if gm.RENDER_VIEWER_CAMERA:
            cam_pose = _compute_viewer_camera_pose(env, focus_object_names=args.focus_objects)
            if cam_pose is not None:
                print(f"[bddl_scene_capture] Viewer camera pos={[round(x, 2) for x in cam_pose[0]]}")
                og.sim.viewer_camera.set_position_orientation(
                    position=cam_pose[0], orientation=cam_pose[1],
                )
            else:
                print("[bddl_scene_capture] Fallback viewer camera.")
                og.sim.viewer_camera.set_position_orientation(
                    position=[1.6, 6.15, 1.5],
                    orientation=[-0.2322, 0.5895, 0.7199, -0.2835],
                )
            if not gm.HEADLESS:
                og.sim.enable_viewer_camera_teleoperation()

        # Video writers (default: save both; use --no_viewer_video or --no_agent_video to skip one)
        save_viewer = not getattr(args, "no_viewer_video", False)
        save_agent = not getattr(args, "no_agent_video", False)

        if save_viewer:
            viewer_fpath = os.path.join(viewer_dir, f"viewer_{timestamp}.mp4")
            video_writer_viewer = create_video_writer(
                fpath=viewer_fpath, resolution=VIEWER_RESOLUTION, rate=args.fps,
            )
            print(f"[bddl_scene_capture] Viewer video: {viewer_fpath}")

        if save_agent:
            head_rgb = _get_head_rgb(obs, env)
            if head_rgb is not None:
                print(f"[bddl_scene_capture] Head camera shape: {head_rgb.shape}")
                agent_fpath = os.path.join(agent_dir, f"agent_head_{timestamp}.mp4")
                video_writer_agent = create_video_writer(
                    fpath=agent_fpath, resolution=AGENT_VIDEO_RESOLUTION, rate=args.fps,
                )
                print(f"[bddl_scene_capture] Agent (head) video: {agent_fpath}")
            else:
                print("[bddl_scene_capture] WARNING: Head camera RGB not found; skipping agent video.")

        print(f"[bddl_scene_capture] Starting {num_steps}-step rollout...")
        for step in range(num_steps):
            print(f"[bddl_scene_capture] Step {step} of {num_steps}")
            
            obs, reward, terminated, truncated, info = env.step(zero_action)

            if video_writer_viewer is not None and gm.RENDER_VIEWER_CAMERA:
                viewer_obs, _ = og.sim.viewer_camera.get_obs()
                if "rgb" in viewer_obs:
                    rgb = _to_np_rgb(viewer_obs["rgb"])
                    write_video(rgb[None, ...], video_writer_viewer, mode="rgb")

            if video_writer_agent is not None:
                head_rgb = _get_head_rgb(obs, env)
                if head_rgb is not None:
                    frame = _resize_uint8(head_rgb, HEAD_RECORD_RES, HEAD_RECORD_RES)
                    write_video(frame[None, ...], video_writer_agent, mode="rgb")

            if terminated or truncated:
                print(f"[bddl_scene_capture] Episode ended at step {step}")
                break

        print(f"[bddl_scene_capture] Rollout finished ({min(step + 1, num_steps)} steps).")

    except Exception as e:
        print(f"[bddl_scene_capture] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if video_writer_viewer is not None:
            video_writer_viewer[0].close()
            print("[bddl_scene_capture] Viewer video closed.")
        if video_writer_agent is not None:
            video_writer_agent[0].close()
            print("[bddl_scene_capture] Agent (head) video closed.")
        og.shutdown()

    print(f"[bddl_scene_capture] Done. Outputs under: {base_out}")


if __name__ == "__main__":
    main()

'''
# Default: save both viewer and agent videos
OMNIGIBSON_HEADLESS=1 python bddl3/tests/bddl_scene_capture.py --activity transfer_hot_pan_safely --focus_objects stove frying_pan

OMNIGIBSON_HEADLESS=1 python tests/bddl_scene_capture.py --activity transfer_hot_pan_safely --no_viewer_video --robot_offset 0.3 --focus_objects stove frying_pan newspaper paper_towel
'''
