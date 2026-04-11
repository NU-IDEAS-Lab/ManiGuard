#!/usr/bin/env python3
"""
Standalone test: FrankaPanda with openpi DROID policy server.
DROID uses joint_position(7) + gripper(1) for state, 8D delta EEF actions.

Usage:
    # Reset only
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
        conda run -n behavior python tools/test_droid_standalone.py --reset-only --headless

    # With openpi droid server (start server first)
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
        conda run -n behavior python tools/test_droid_standalone.py --steps 500 --save-video
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    p = argparse.ArgumentParser(description="Standalone DROID-format FrankaPanda test.")
    p.add_argument("--steps", type=int, default=0, help="Number of env steps. 0 = reset only.")
    p.add_argument("--reset-only", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "droid_test"))
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--execute-horizon", type=int, default=5)
    p.add_argument("--prompt", default="Pick up the red cube.")
    p.add_argument("--scene", choices=["cubes", "toys"], default="cubes")
    return p.parse_args()


def build_env_config(headless: bool, scene: str):
    if headless:
        gm.HEADLESS = True

    scene_cfg = {"type": "Scene"}
    robot_cfg = {
        "type": "FrankaPanda",
        "name": "franka",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "physical",
        "position": [0, 0, 0.5],
        # Match DROID training data mean joint pose
        "reset_joint_pos": [0.016, 0.267, -0.017, -2.026, -0.033, 2.345, 0.083, 0.04, 0.04],
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {"image_height": 256, "image_width": 256},
            },
        },
        "controller_config": {
            "arm_0": {
                "name": "JointController",
                "motor_type": "position",
                "use_delta_commands": True,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": None,
                "command_output_limits": "default",
                "mode": "smooth",
            },
        },
    }

    if scene == "cubes":
        table_height = 0.3
        cube_z = table_height + 0.2 + 0.02 + 0.01
        object_cfg = [
            {
                "type": "DatasetObject",
                "prim_path": "/World/breakfast_table",
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "kwmfdg",
                "bounding_box": [2, 1, 0.4],
                "position": [0.8, 0, table_height],
                "orientation": [0, 0, 0.707, 0.707],
            },
            {
                "type": "PrimitiveObject",
                "prim_path": "/World/red_cube",
                "name": "red_cube",
                "primitive_type": "Cube",
                "size": 0.04,
                "position": [0.55, 0.1, cube_z],
                "rgba": [0.9, 0.1, 0.1, 1.0],
            },
            {
                "type": "PrimitiveObject",
                "prim_path": "/World/blue_cube",
                "name": "blue_cube",
                "primitive_type": "Cube",
                "size": 0.04,
                "position": [0.55, -0.1, cube_z],
                "rgba": [0.1, 0.1, 0.9, 1.0],
            },
        ]
    else:
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
        ]

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
                "sensor_kwargs": {"image_height": 256, "image_width": 256},
            },
        ],
    }

    return dict(scene=scene_cfg, robots=[robot_cfg], objects=object_cfg, env=env_cfg)


# Camera position
AGENTVIEW_POSITION = [1.7, 0.0, 1.1]
AGENTVIEW_LOOKAT = [0.5, 0.0, 0.4]


def extract_obs(env, robot, prompt):
    """Extract DROID-format observation."""
    raw_obs, _ = env.get_obs()

    # Main camera
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

    # DROID state: joint_position(7) + gripper_position(1) = 8D
    arm_idx = robot.arm_control_idx[robot.default_arm]
    joint_pos = robot.get_joint_positions()[arm_idx].cpu().numpy().astype(np.float32)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    gripper_qpos = robot.get_joint_positions()[gripper_idx].cpu().numpy().astype(np.float32)
    gripper_scalar = np.mean(gripper_qpos).reshape(1).astype(np.float32)

    return {
        # DROID keys for openpi native server
        "observation/exterior_image_1_left": main_rgb,
        "observation/wrist_image_left": wrist_rgb,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_scalar,
        "prompt": prompt,
        # Also keep raw for video
        "_main_rgb": main_rgb,
        "_wrist_rgb": wrist_rgb,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    import math

    cfg = build_env_config(headless=args.headless, scene=args.scene)
    env = og.Environment(configs=cfg)
    env.reset()

    # Set camera
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
    agentview.set_position_orientation(position=cam_eye.tolist(), orientation=cam_quat.tolist(), frame="world")
    og.sim.viewer_camera.set_position_orientation(position=cam_eye.tolist(), orientation=cam_quat.tolist())

    # Settle
    robot = env.robots[0]
    for _ in range(20):
        robot.keep_still()
        og.sim.step()
    for _ in range(2):
        og.sim.render()

    obs = extract_obs(env, robot, args.prompt)

    print(f"Main image shape: {obs['observation/exterior_image_1_left'].shape}")
    print(f"Wrist image shape: {obs['observation/wrist_image_left'].shape}")
    print(f"Joint position: {obs['observation/joint_position'].tolist()}")
    print(f"Gripper position: {obs['observation/gripper_position'].tolist()}")
    print(f"Prompt: {obs['prompt']}")

    imageio.imwrite(str(output_dir / "agentview.png"), obs["_main_rgb"])
    imageio.imwrite(str(output_dir / "wrist.png"), obs["_wrist_rgb"])
    print(f"Saved images to {output_dir}")

    if args.reset_only or args.steps == 0:
        print("Reset-only mode. Exiting.")
        sys.stdout.flush()
        os._exit(0)

    # Connect to openpi native server using openpi_client
    from openpi_client import websocket_client_policy as _wcp
    from PIL import Image

    policy = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    server_meta = {}
    print(f"Connected to openpi server")

    frames = [obs["_main_rgb"]]
    wrist_frames = [obs["_wrist_rgb"]]
    action_space = robot.action_space
    execute_horizon = args.execute_horizon

    step_idx = 0
    done = False
    while step_idx < args.steps and not done:
        # Resize images to 224x224 for policy
        main_resized = np.array(Image.fromarray(obs["_main_rgb"]).resize((224, 224)))
        wrist_resized = np.array(Image.fromarray(obs["_wrist_rgb"]).resize((224, 224)))

        policy_obs = {
            "observation/exterior_image_1_left": main_resized,
            "observation/wrist_image_left": wrist_resized,
            "observation/joint_position": obs["observation/joint_position"],
            "observation/gripper_position": obs["observation/gripper_position"],
            "prompt": obs["prompt"],
        }
        result = policy.infer(policy_obs)
        action_chunk = np.asarray(result["actions"], dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[np.newaxis, :]

        # DROID outputs 8D absolute: joint_position(7) + gripper_position(1)
        # Send directly to JointController (arm 7D) + GripperController (1D)
        chunk_len = min(execute_horizon, len(action_chunk), args.steps - step_idx)
        for chunk_idx in range(chunk_len):
            action_8d = action_chunk[chunk_idx].copy()
            # 8D action: 7 joint targets + 1 gripper target
            action_clipped = np.clip(action_8d[:action_space.shape[0]], action_space.low, action_space.high)

            raw_obs, reward, terminated, truncated, info = env.step(
                torch.from_numpy(action_clipped).unsqueeze(0)
            )
            obs = extract_obs(env, robot, args.prompt)
            frames.append(obs["_main_rgb"])
            wrist_frames.append(obs["_wrist_rgb"])
            step_idx += 1

            if step_idx % 20 == 0 or step_idx == 1:
                print(f"Step {step_idx}/{args.steps} (chunk {chunk_idx+1}/{chunk_len}): "
                      f"action_8d={[round(float(a), 3) for a in action_8d]}")

            if terminated or truncated:
                print(f"Episode ended at step {step_idx}")
                done = True
                break

    if args.save_video and frames:
        video_path = output_dir / "rollout.mp4"
        imageio.mimsave(str(video_path), frames, fps=10)
        print(f"Saved agentview video to {video_path}")
        wrist_video_path = output_dir / "rollout_wrist.mp4"
        imageio.mimsave(str(wrist_video_path), wrist_frames, fps=10)
        print(f"Saved wrist video to {wrist_video_path}")

    print("Done.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
