#!/usr/bin/env python3
"""
Standalone test: FrankaPanda at a table with IK controller + LIBERO-style observations.
Uses the OmniGibson teleop demo scene setup. No frozen scene or benchmark registry needed.

Usage:
    # Reset only - verify scene and camera
    conda run -n behavior env OMNI_KIT_ACCEPT_EULA=yes python tools/test_libero_standalone.py --reset-only

    # With policy server (start serve_pi05_franka_websocket.py first)
    conda run -n behavior env OMNI_KIT_ACCEPT_EULA=yes python tools/test_libero_standalone.py --steps 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = REPO_ROOT / "RLinf"
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


def parse_args():
    p = argparse.ArgumentParser(description="Standalone LIBERO-style FrankaPanda tabletop test.")
    p.add_argument("--steps", type=int, default=0, help="Number of env steps to run. 0 = reset only.")
    p.add_argument("--reset-only", action="store_true", help="Just reset and save images.")
    p.add_argument("--host", default="127.0.0.1", help="Policy server host.")
    p.add_argument("--port", type=int, default=8000, help="Policy server port.")
    p.add_argument("--headless", action="store_true", help="Run headless.")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "standalone_test"), help="Output directory.")
    p.add_argument("--save-video", action="store_true", help="Save video of the rollout.")
    p.add_argument("--execute-horizon", type=int, default=5, help="Execute N steps per action chunk before replanning.")
    return p.parse_args()


def quat2axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3D)."""
    quat = np.array(quat, dtype=np.float32)
    quat = np.clip(quat, -1.0, 1.0)
    w = quat[3]
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    axis = quat[:3] / sin_half
    return (axis * angle).astype(np.float32)


def build_env_config(headless: bool):
    """Build a minimal OmniGibson env config mimicking the teleop demo."""
    if headless:
        gm.HEADLESS = True

    scene_cfg = {"type": "Scene"}

    robot_cfg = {
        "type": "FrankaPanda",
        "name": "franka",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "physical",
        "position": [0, 0, 0.5],  # Raise to table-top height
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": 256,
                    "image_width": 256,
                },
            },
        },
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_delta_ori",
                "command_input_limits": [[-1.0] * 6, [1.0] * 6],
                "command_output_limits": [
                    [-0.2, -0.2, -0.2, -0.5, -0.5, -0.5],
                    [0.2, 0.2, 0.2, 0.5, 0.5, 0.5],
                ],
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": [-1.0, 1.0],
                "command_output_limits": "default",
                "mode": "smooth",
                "inverted": True,
            },
        },
    }

    object_cfg = [
        {
            "type": "DatasetObject",
            "prim_path": "/World/breakfast_table",
            "name": "breakfast_table",
            "category": "breakfast_table",
            "model": "kwmfdg",
            "bounding_box": [2, 1, 0.4],
            "position": [0.8, 0, 0.3],
            "orientation": [0, 0, 0.707, 0.707],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/frail",
            "name": "frail",
            "category": "frail",
            "model": "zmjovr",
            "scale": [2, 2, 2],
            "position": [0.6, -0.35, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure1",
            "name": "toy_figure1",
            "category": "toy_figure",
            "model": "issvzv",
            "scale": [0.75, 0.75, 0.75],
            "position": [0.6, 0, 0.55],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure2",
            "name": "toy_figure2",
            "category": "toy_figure",
            "model": "nncqfn",
            "scale": [0.75, 0.75, 0.75],
            "position": [0.6, 0.2, 0.55],
        },
    ]

    # External camera (agentview-style)
    env_cfg = {
        "action_frequency": 20,
        "rendering_frequency": 20,
        "physics_frequency": 120,
        "external_sensors": [
            {
                "sensor_type": "VisionSensor",
                "name": "agentview",
                "relative_prim_path": "/agentview",
                "modalities": ["rgb"],
                "sensor_kwargs": {
                    "image_height": 256,
                    "image_width": 256,
                },
            },
        ],
    }

    return dict(scene=scene_cfg, robots=[robot_cfg], objects=object_cfg, env=env_cfg)


# Camera: front overhead, looking down at table/objects/robot
AGENTVIEW_POSITION = [1.7, 0.0, 1.1]
AGENTVIEW_LOOKAT = [0.5, 0.0, 0.4]


def extract_obs(env, robot):
    """Extract LIBERO-compatible observation."""
    raw_obs, _ = env.get_obs()

    # Main camera (agentview)
    external = raw_obs.get("external", {})
    main_rgb = external["agentview"]["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    # Wrist camera
    robot_obs = raw_obs.get(robot.name, {})
    wrist_sensor_name = None
    for name, obs in robot_obs.items():
        if isinstance(obs, dict) and "rgb" in obs:
            wrist_sensor_name = name
            break
    if wrist_sensor_name:
        wrist_rgb = robot_obs[wrist_sensor_name]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    else:
        wrist_rgb = np.zeros_like(main_rgb)

    # EEF state: pos(3) + axisangle(3) + gripper(1) = 7D
    eef_pos = robot.get_relative_eef_position().cpu().numpy().astype(np.float32)
    eef_quat = robot.get_relative_eef_orientation().cpu().numpy().astype(np.float32)
    eef_axisangle = quat2axisangle(eef_quat)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    gripper_qpos = robot.get_joint_positions()[gripper_idx].cpu().numpy().astype(np.float32)
    gripper_scalar = np.mean(gripper_qpos).reshape(1).astype(np.float32)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_scalar])

    return {
        "main_image": main_rgb,
        "wrist_image": wrist_rgb,
        "state": state,
        "prompt": "Pick up the toy figure and put it in the basket.",
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import omnigibson as og

    cfg = build_env_config(headless=args.headless)
    env = og.Environment(configs=cfg)
    env.reset()

    # Set agentview camera
    import omnigibson.utils.transform_utils as T
    import math
    cam_eye = np.array(AGENTVIEW_POSITION, dtype=np.float32)
    cam_lookat = np.array(AGENTVIEW_LOOKAT, dtype=np.float32)
    direction = cam_lookat - cam_eye
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    cam_quat = T.euler2quat(torch.tensor([
        math.pi / 2 + float(np.arcsin(np.clip(direction[2], -1.0, 1.0))),
        0.0,
        float(np.arctan2(-direction[0], direction[1])),
    ], dtype=torch.float32))
    agentview = env.external_sensors["agentview"]
    agentview.set_position_orientation(
        position=cam_eye.tolist(),
        orientation=cam_quat.tolist(),
        frame="world",
    )
    og.sim.viewer_camera.set_position_orientation(
        position=cam_eye.tolist(),
        orientation=cam_quat.tolist(),
    )

    # Settle physics
    robot = env.robots[0]
    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    for _ in range(2):
        og.sim.render()

    obs = extract_obs(env, robot)

    print(f"Main image shape: {obs['main_image'].shape}")
    print(f"Wrist image shape: {obs['wrist_image'].shape}")
    print(f"State shape: {obs['state'].shape}")
    print(f"State: {obs['state'].tolist()}")
    print(f"Prompt: {obs['prompt']}")

    # Save images
    imageio.imwrite(str(output_dir / "agentview.png"), obs["main_image"])
    imageio.imwrite(str(output_dir / "wrist.png"), obs["wrist_image"])
    print(f"Saved images to {output_dir}")

    if args.reset_only or args.steps == 0:
        print("Reset-only mode. Exiting.")
        sys.stdout.flush()
        os._exit(0)

    # Connect to policy server
    from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
    from PIL import Image

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    policy.reset()
    server_meta = policy.get_server_metadata() or {}
    print(f"Policy server metadata: {server_meta}")

    frames = [obs["main_image"]]
    action_space = robot.action_space
    execute_horizon = args.execute_horizon

    step_idx = 0
    done = False
    while step_idx < args.steps and not done:
        main_img = obs["main_image"]
        wrist_img = obs["wrist_image"]
        main_img_resized = np.array(Image.fromarray(main_img).resize((224, 224)))
        wrist_img_resized = np.array(Image.fromarray(wrist_img).resize((224, 224)))

        policy_obs = {
            "main_images": main_img_resized,
            "wrist_images": wrist_img_resized,
            "states": obs["state"],
            "task_descriptions": obs["prompt"],
        }
        action_chunk = policy.act(policy_obs).detach().cpu().numpy().astype(np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[np.newaxis, :]

        chunk_len = min(execute_horizon, len(action_chunk), args.steps - step_idx)
        for chunk_idx in range(chunk_len):
            action = action_chunk[chunk_idx].copy()
            action[-1] = np.sign(action[-1]) if abs(action[-1]) > 0.01 else -1.0
            action_clipped = np.clip(action[:action_space.shape[0]], action_space.low, action_space.high)
            raw_obs, reward, terminated, truncated, info = env.step(
                torch.from_numpy(action_clipped).unsqueeze(0)
            )
            obs = extract_obs(env, robot)
            frames.append(obs["main_image"])
            step_idx += 1

            if step_idx % 20 == 0 or step_idx == 1:
                print(f"Step {step_idx}/{args.steps} (chunk {chunk_idx+1}/{chunk_len}): action={[round(float(a), 3) for a in action[:7]]}")

            if terminated or truncated:
                print(f"Episode ended at step {step_idx}")
                done = True
                break

    if args.save_video and frames:
        video_path = output_dir / "rollout.mp4"
        imageio.mimsave(str(video_path), frames, fps=10)
        print(f"Saved video to {video_path}")

    print("Done.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
