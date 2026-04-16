#!/usr/bin/env python3
"""Test OmniGibson VectorEnvironment initialization with benchmark scenes.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
        python scripts/test_vec_env.py \
            --benchmark-root sentinel-lite-taskgen-staging/transfer_trial20_benchmark_safe \
            --num-envs 2 --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Test VectorEnvironment init with benchmark scenes.")
    p.add_argument("--benchmark-root", required=True, help="Path to benchmark scene directory.")
    p.add_argument("--num-envs", type=int, default=2, help="Number of parallel envs to create.")
    p.add_argument("--max-scenes", type=int, default=None, help="Limit scene discovery.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--action-frequency", type=int, default=20)
    p.add_argument("--rendering-frequency", type=int, default=20)
    p.add_argument("--physics-frequency", type=int, default=120)
    p.add_argument("--camera-resolution", type=int, default=128)
    p.add_argument("--steps", type=int, default=5, help="Number of random-action steps to test.")
    return p.parse_args()


def discover_scenes(benchmark_root: str, max_scenes=None):
    """Minimal scene discovery matching benchmark.py logic."""
    root = Path(benchmark_root)
    scenes = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        is_scene = (current / "scene_ep1.json").is_file() and (current / "diagnostics.jsonl").is_file()
        if is_scene:
            scene_file = current / "scene_ep1.json"
            diag_file = current / "diagnostics.jsonl"
            diag = json.loads(diag_file.read_text(encoding="utf-8").strip().split("\n")[0])
            scene_info = json.loads(scene_file.read_text(encoding="utf-8"))
            init_info = scene_info.get("objects_info", {}).get("init_info", {})

            # Find target object and rooms
            target_name = None
            for entry in diag.get("active_object_summary", []):
                if entry.get("role") == "target":
                    target_name = entry.get("scene_object_name")
                    break
            target_rooms = []
            if target_name and target_name in init_info:
                target_rooms = init_info[target_name].get("args", {}).get("in_rooms", [])
            surface_name = diag.get("surface", "")
            if surface_name and surface_name in init_info:
                surface_rooms = init_info[surface_name].get("args", {}).get("in_rooms", [])
                target_rooms = list(set(target_rooms + surface_rooms))

            scenes.append({
                "name": current.name,
                "scene_file": str(scene_file),
                "scene_model": diag.get("scene_model", current.name),
                "target_name": target_name,
                "surface_name": surface_name,
                "target_rooms": target_rooms,
                "scene_class": scene_info.get("init_info", {}).get("class_name", "InteractiveTraversableScene"),
            })
            continue
        if depth >= 3:
            continue
        for entry in entries:
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                stack.append((entry, depth + 1))

    scenes.sort(key=lambda s: s["name"])
    if max_scenes:
        scenes = scenes[:max_scenes]
    return scenes


def build_og_config(scene_info: dict, args):
    """Build a single OmniGibson env config from a benchmark scene."""
    if scene_info["scene_class"] == "Scene":
        scene_cfg = {
            "type": "Scene",
            "scene_file": scene_info["scene_file"],
        }
    else:
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_info["scene_model"],
            "scene_file": scene_info["scene_file"],
            "scene_instance": None,
            "include_robots": False,
        }
        if scene_info["target_rooms"]:
            scene_cfg["load_room_instances"] = scene_info["target_rooms"]

    # Extract robot pose from scene snapshot
    scene_data = json.loads(Path(scene_info["scene_file"]).read_text(encoding="utf-8"))
    sd_init_info = scene_data.get("objects_info", {}).get("init_info", {})
    state_registry = scene_data.get("state", {}).get("registry", {}).get("object_registry", {})

    robot_cfg = {
        "type": "FrankaMounted",
        "name": "agent_0",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": args.camera_resolution,
                    "image_width": args.camera_resolution,
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

    for obj_name, obj_info in sd_init_info.items():
        if "robot" in str(obj_info.get("class_module", "")).lower():
            state = state_registry.get(obj_name, {})
            root_link = state.get("root_link", {})
            if root_link.get("pos"):
                robot_cfg["position"] = root_link["pos"]
            if root_link.get("ori"):
                robot_cfg["orientation"] = root_link["ori"]
            if state.get("joint_pos"):
                robot_cfg["reset_joint_pos"] = state["joint_pos"]
            break

    robots_block = [] if scene_info["scene_class"] == "Scene" else [robot_cfg]

    return {
        "scene": scene_cfg,
        "robots": robots_block,
        "objects": [],
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": args.action_frequency,
            "rendering_frequency": args.rendering_frequency,
            "physics_frequency": args.physics_frequency,
        },
    }


def main():
    args = parse_args()

    # Discover scenes
    scenes = discover_scenes(args.benchmark_root, max_scenes=args.max_scenes)
    if not scenes:
        print(f"No scenes found under {args.benchmark_root}")
        sys.exit(1)

    num_envs = min(args.num_envs, len(scenes))
    print(f"Discovered {len(scenes)} scenes, will create {num_envs} parallel envs")
    for i, s in enumerate(scenes[:num_envs]):
        print(f"  env {i}: {s['name']} (model={s['scene_model']}, rooms={s['target_rooms']})")

    # --- OmniGibson imports (after arg parsing to fail fast) ---
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if args.headless:
        gm.HEADLESS = True

    import omnigibson as og

    # Build per-env configs
    configs = [build_og_config(scenes[i], args) for i in range(num_envs)]

    print(f"\n{'='*60}")
    print("Creating VectorEnvironment...")
    print(f"{'='*60}")

    try:
        vec_env = og.VectorEnvironment(num_envs=num_envs, config=configs[0])
        print(f"VectorEnvironment created with {num_envs} envs (same config)")

        # Position the viewer camera to see all envs.
        # Scenes are laid out along the X axis with ~10m margin.
        # We pull the camera up and back to frame all of them.
        import torch as th
        sim = og.sim
        if sim.viewer_camera is not None:
            # Compute a centroid across all scene robots
            positions = []
            for env in vec_env.envs:
                if env.robots:
                    pos = env.robots[0].get_position_orientation()[0]
                    positions.append(pos.cpu().tolist())
            if positions:
                cx = sum(p[0] for p in positions) / len(positions)
                cy = sum(p[1] for p in positions) / len(positions)
                cz = sum(p[2] for p in positions) / len(positions)
                # Spread: how far apart the outermost envs are
                spread = max(abs(p[0] - cx) for p in positions) if len(positions) > 1 else 5.0
                pullback = max(spread * 2.5, 8.0)
                cam_pos = th.tensor([cx - pullback * 0.4, cy - pullback * 0.6, cz + pullback * 0.8])
                # Look-at via a simple forward-facing quaternion (top-down oblique)
                # We'll just set position and let the user orbit with keyboard
                sim.viewer_camera.set_position_orientation(
                    position=cam_pos,
                    orientation=th.tensor([0.35, 0.08, -0.02, 0.93]),
                )
                print(f"Viewer camera repositioned to see all {num_envs} envs")
                print(f"  centroid=({cx:.1f}, {cy:.1f}, {cz:.1f}), pullback={pullback:.1f}")
                print(f"  camera pos={cam_pos.tolist()}")

            # Enable keyboard teleoperation of the viewer camera
            camera_mover = sim.enable_viewer_camera_teleoperation()
            print("Viewer camera teleoperation enabled (WASD/QE to move, mouse to look)")

        print("Resetting...")
        observations, infos = vec_env.reset()
        print(f"Reset complete. Got {len(observations)} observation dicts.")
        for i, obs in enumerate(observations):
            robot_keys = [k for k in obs if k.startswith("agent") or k.startswith("robot")]
            external_keys = [k for k in obs if k.startswith("external") or k.startswith("cam")]
            print(f"  env {i}: obs keys={list(obs.keys())[:5]}... robot={robot_keys} ext={external_keys}")

        print(f"\nStepping {args.steps} times with random actions...")
        print("(Use WASD + mouse to orbit the viewport camera)")
        for step in range(args.steps):
            actions = []
            for env in vec_env.envs:
                action_space = env.robots[0].action_space
                action = action_space.sample()
                actions.append(action)
            observations, rewards, terminates, truncates, info_list = vec_env.step(actions)
            if (step + 1) % 50 == 0 or step == 0:
                print(f"  step {step+1}/{args.steps}")

        print("\nVectorEnvironment test PASSED")

    except Exception as e:
        print(f"\nVectorEnvironment FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    print("\nDone. Exiting.")
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
