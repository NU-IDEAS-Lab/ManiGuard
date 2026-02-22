"""
Record BDDL-generated BEHAVIOR task scenes to video.

Loads a single BDDL activity by name, creates the OmniGibson Behavior env,
runs a short zero-action rollout, and records either the viewer (interface)
camera and/or the R1Pro agent cameras (head, left_wrist, right_wrist) into
separate videos under activity-named subfolders.

Intended for headless use on SSH servers: set OMNIGIBSON_HEADLESS=1.
Default save path: videos/ next to this script (bddl3/tests/videos/).

Scene vs task:
  - With online sampling (default): the scene loads its base layout (e.g. scene_best.json),
    then objects are placed at runtime from your BDDL. No pre-cached task file needed.
  - With --no_online_sampling: the loader looks for a pre-generated file under
    datasets/2025-challenge-task-instances/scenes/<scene_model>/json/
    named <scene_model>_task_<activity>_<def_id>_<inst_id>_template.json.
    Only use this for official activities that have such files.

Scene models:
  - house_double_floor_lower (default in r1pro_behavior.yaml): full house, two floors.
  - Rs_int: smaller interactive scene, often used in tests; requires the scene to exist
    in behavior-1k-assets/scenes/. Override with --scene_model Rs_int in your config
    or by passing a custom --config YAML.
"""

import argparse
import datetime
import os
import sys

import cv2
import numpy as np
import yaml

# Ensure OmniGibson is on path when run from repo root or bddl3/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BDDL3_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BDDL3_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video


# Default video output: same-level folder as this script
DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
VIEWER_RESOLUTION = (720, 1280)  # (height, width) from r1pro_behavior render
# Agent combined frame: 3 panels (head, left_wrist, right_wrist) at 128 height each
AGENT_PANEL_HEIGHT = 128
AGENT_COMBINED_RESOLUTION = (AGENT_PANEL_HEIGHT, AGENT_PANEL_HEIGHT * 3)
DEFAULT_FPS = 30
DEFAULT_DURATION_SEC = 5.0
# for cloth system initialization
gm.USE_GPU_DYNAMICS = True

def _find_robot_rgb_views(obs, robot_name="robot0"):
    """
    From env obs (nested dict), collect R1Pro camera RGB by role.
    Returns dict: "head" | "left_wrist" | "right_wrist" -> numpy (H,W,3) uint8.
    """
    out = {}
    if robot_name not in obs or not isinstance(obs[robot_name], dict):
        return out
    robot_obs = obs[robot_name]
    for sensor_key, sensor_data in robot_obs.items():
        if not isinstance(sensor_data, dict) or "rgb" not in sensor_data:
            continue
        rgb = sensor_data["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        key_lower = sensor_key.lower()
        if "zed" in key_lower or "head" in key_lower:
            out["head"] = rgb
        elif "left" in key_lower and ("realsense" in key_lower or "wrist" in key_lower):
            out["left_wrist"] = rgb
        elif "right" in key_lower and ("realsense" in key_lower or "wrist" in key_lower):
            out["right_wrist"] = rgb
    return out


def _stack_agent_frames(agent_views, target_height=AGENT_PANEL_HEIGHT):
    """
    Stack head, left_wrist, right_wrist horizontally (same height) for one combined frame.
    agent_views: dict from _find_robot_rgb_views.
    """
    order = ["head", "left_wrist", "right_wrist"]
    imgs = []
    for k in order:
        if k not in agent_views:
            continue
        img = np.asarray(agent_views[k])
        if img.dtype != np.uint8 and img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        if h != target_height:
            scale = target_height / h
            new_w = max(1, int(round(w * scale)))
            img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
    if not imgs:
        return None
    return np.concatenate(imgs, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record BDDL activity scene to video(s) (viewer and/or R1Pro agent cameras)."
    )
    parser.add_argument(
        "--activity",
        type=str,
        required=True,
        help="BDDL activity name (e.g. transfer_hot_pan_safely, light_candle_near_flammables).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"Root directory for video outputs. Default: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help=f"Recording duration in seconds. Default: {DEFAULT_DURATION_SEC}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second (and sim steps per second). Default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["viewer", "agent", "both"],
        default="both",
        help="Which view(s) to record: viewer (interface), agent (R1Pro head + wrists), or both. Default: both",
    )
    parser.add_argument(
        "--activity_definition_id",
        type=int,
        default=0,
        help="BDDL activity definition id. Default: 0",
    )
    parser.add_argument(
        "--activity_instance_id",
        type=int,
        default=0,
        help="BDDL activity instance id. Default: 0",
    )
    parser.add_argument(
        "--no_online_sampling",
        action="store_true",
        help="Use pre-cached task template from 2025-challenge-task-instances (only for activities that have pre-generated scene JSON). Default: False (use online sampling for custom BDDL).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to base behavior YAML config. If not set, uses og.example_config_path/r1pro_behavior.yaml",
    )
    parser.add_argument(
        "--scene_model",
        type=str,
        default=None,
        help="Override scene model (e.g. Rs_int). Must exist in behavior-1k-assets/scenes/. Default: use config value (e.g. house_double_floor_lower).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    activity_name = args.activity.strip()
    if not activity_name:
        raise ValueError("--activity must be non-empty.")

    # Resolve base config
    if args.config and os.path.isfile(args.config):
        config_path = args.config
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Default behavior config not found: {config_path}. Set --config to a valid r1pro_behavior YAML."
            )
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

    # Optional: headless-friendly (viewer camera still renders)
    if os.environ.get("OMNIGIBSON_HEADLESS", "").lower() in ("1", "true", "t"):
        with gm.unlocked():
            gm.HEADLESS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.RENDER_VIEWER_CAMERA = True

    og.log.info(
        f"bddl_scene_capture: activity={activity_name}, duration={args.duration}s, "
        f"fps={args.fps}, view={args.view}, output_base={base_out}"
    )

    env = og.Environment(configs=cfg)
    num_steps = max(1, int(round(args.duration * args.fps)))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Viewer camera: fixed pose for headless (no teleop)
    if gm.RENDER_VIEWER_CAMERA:
        og.sim.viewer_camera.set_position_orientation(
            position=[1.6, 6.15, 1.5],
            orientation=[-0.2322, 0.5895, 0.7199, -0.2835],
        )
        if not gm.HEADLESS:
            og.sim.enable_viewer_camera_teleoperation()

    video_writer_viewer = None
    video_writer_agent = None

    def _zero_action():
        if getattr(env.action_space, "spaces", None) is not None:
            return {
                r.name: np.zeros(r.action_space.shape, dtype=getattr(r.action_space, "dtype", np.float32))
                for r in env.robots
            }
        return np.zeros(
            env.action_space.shape,
            dtype=getattr(env.action_space, "dtype", np.float32),
        )

    try:
        obs, info = env.reset()
        zero_action = _zero_action()

        if args.view in ("viewer", "both"):
            viewer_fpath = os.path.join(viewer_dir, f"viewer_{timestamp}.mp4")
            video_writer_viewer = create_video_writer(
                fpath=viewer_fpath,
                resolution=VIEWER_RESOLUTION,
                rate=args.fps,
            )
            og.log.info(f"Viewer video: {viewer_fpath}")

        if args.view in ("agent", "both"):
            agent_views = _find_robot_rgb_views(obs)
            if agent_views:
                agent_fpath = os.path.join(agent_dir, f"agent_combined_{timestamp}.mp4")
                video_writer_agent = create_video_writer(
                    fpath=agent_fpath,
                    resolution=AGENT_COMBINED_RESOLUTION,
                    rate=args.fps,
                )
                og.log.info(f"Agent (combined) video: {agent_fpath}")
            else:
                og.log.warning("No R1Pro RGB views found in obs; skipping agent video.")

        for step in range(num_steps):
            # Zero action rollout
            obs, reward, terminated, truncated, info = env.step(zero_action)

            if args.view in ("viewer", "both") and video_writer_viewer is not None and gm.RENDER_VIEWER_CAMERA:
                viewer_obs, _ = og.sim.viewer_camera.get_obs()
                if "rgb" in viewer_obs:
                    rgb = viewer_obs["rgb"][..., :3]
                    if hasattr(rgb, "cpu"):
                        rgb = rgb.cpu().numpy()
                    write_video(rgb[None, ...], video_writer_viewer, mode="rgb")

            if args.view in ("agent", "both") and video_writer_agent is not None:
                agent_views = _find_robot_rgb_views(obs)
                combined = _stack_agent_frames(agent_views)
                if combined is not None:
                    write_video(combined[None, ...], video_writer_agent, mode="rgb")

            if terminated or truncated:
                break

    finally:
        if video_writer_viewer is not None:
            video_writer_viewer[0].close()
            og.log.info("Viewer video closed.")
        if video_writer_agent is not None:
            video_writer_agent[0].close()
            og.log.info("Agent video closed.")
        og.shutdown()

    og.log.info(f"Scene capture finished. Outputs under: {base_out}")


if __name__ == "__main__":
    main()

'''
OMNIGIBSON_HEADLESS=1 python bddl3/tests/bddl_scene_capture.py --activity transfer_hot_pan_safely

OMNIGIBSON_HEADLESS=1 python bddl3/tests/bddl_scene_capture.py --activity transfer_hot_pan_safely --scene_model Rs_int

OMNIGIBSON_HEADLESS=1 python bddl3/tests/bddl_scene_capture.py --activity picking_up_trash --no_online_sampling
'''