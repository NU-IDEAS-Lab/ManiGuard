#!/usr/bin/env python3
"""
Evaluate a websocket VLA policy on the SENTINEL benchmark.

Bypasses BDDL — loads scenes directly from scene_ep JSON with partial room
loading, checks success via grasp detection, and runs LTL from ltl_safety.json.

Usage:
    # On Machine B (OmniGibson), with model server on Machine A:
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
        conda run -n behavior python tools/evaluate_benchmark.py \
        --benchmark-root /path/to/benchmark \
        --host <MACHINE_A_IP> --port 8000 \
        --max-steps 500 --save-video
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    p = argparse.ArgumentParser(description="Evaluate VLA on SENTINEL benchmark (no BDDL).")
    p.add_argument("--benchmark-root", required=True,
                   help="Local directory OR a HuggingFace dataset repo_id "
                        "(<owner>/<name>). HF repos are snapshot_download'd "
                        "into the standard HF cache the first time and reused "
                        "on subsequent runs (incremental by file hash).")
    p.add_argument("--benchmark-revision", default="main",
                   help="HF dataset revision to snapshot (main / tag / commit). "
                        "Ignored when --benchmark-root is a local directory.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--use-openpi-client", action="store_true")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--execute-horizon", type=int, default=5)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--action-frequency", type=int, default=20)
    p.add_argument("--rendering-frequency", type=int, default=20)
    p.add_argument("--physics-frequency", type=int, default=120)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "benchmark_eval"))
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--policy-external-cameras", nargs="+", default=None,
                   help="External cameras fed to the policy's main_images "
                        "(one or more of cam_opposite / cam_left / cam_right). "
                        "Multiple values are concatenated along the width axis. "
                        "Default: cam_opposite.")
    p.add_argument("--camera-resolution", type=int, default=256,
                   help="Side length for all external cameras (square).")
    return p.parse_args()


def _build_eval_external_sensors(args):
    from omnigibson.utils.camera_setup import (
        EXTERNAL_CAMERA_NAMES,
        build_external_camera_configs,
        normalize_policy_cameras,
    )
    policy_cams = normalize_policy_cameras(args.policy_external_cameras)
    # Load all cameras the user asked for, plus cam_opposite (fallback for
    # single-camera eval) so obs dict always has at least one known entry.
    names = []
    for name in list(policy_cams) + [EXTERNAL_CAMERA_NAMES[0]]:
        if name not in names:
            names.append(name)
    return build_external_camera_configs(names=names, resolution=args.camera_resolution)


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _iter_scene_dirs(root: Path):
    """Yield every directory under ``root`` that looks like a scene.

    A scene directory must contain both ``scene_ep1.json`` and
    ``diagnostics.jsonl``. The walk is depth-limited to 3 levels so HF
    datasets laid out as ``<task_family>/<trial_N>/`` work transparently
    without re-scanning the whole tree.
    """
    if not root.is_dir():
        return
    stack = [(root, 0)]
    max_depth = 3
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        is_scene = (current / "scene_ep1.json").is_file() and (current / "diagnostics.jsonl").is_file()
        if is_scene:
            yield current
            continue
        if depth >= max_depth:
            continue
        for entry in entries:
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                stack.append((entry, depth + 1))


def _scene_key(root: Path, scene_dir: Path) -> str:
    """Stable scene identifier: either the dir name, or <family>/<trial>
    for HF-style two-level layouts. Lets ``--scenes`` filter on either."""
    try:
        rel = scene_dir.relative_to(root)
    except ValueError:
        return scene_dir.name
    return rel.as_posix()


def discover_scenes(benchmark_root: str, scene_names=None, max_scenes=None):
    """Discover valid benchmark scenes with diagnostics.

    Handles both 1-level layouts (``<root>/<scene>/``) and 2-level
    layouts (``<root>/<task_family>/<scene>/``), the latter produced by
    task-generation benchmark corpora on HuggingFace.
    """
    root = Path(benchmark_root)
    scenes = []
    for scene_dir in _iter_scene_dirs(root):
        scene_file = scene_dir / "scene_ep1.json"
        diag_file = scene_dir / "diagnostics.jsonl"
        scene_key = _scene_key(root, scene_dir)
        if scene_names and scene_key not in scene_names and scene_dir.name not in scene_names:
            continue

        diag = json.loads(diag_file.read_text(encoding="utf-8").strip().split("\n")[0])
        scene_info_json = json.loads(scene_file.read_text(encoding="utf-8"))
        init_info = scene_info_json.get("objects_info", {}).get("init_info", {})
        sel = diag.get("selection", {})
        pipeline = diag.get("pipeline", "")
        surface_name = diag.get("surface", "")
        surface_label = (
            init_info.get(surface_name, {}).get("args", {}).get("category", "")
            or surface_name or "table"
        )

        # ----------------------------------------------------------------
        # Pipeline-aware scene parsing: each pipeline type has different
        # diagnostics keys and different natural-language instructions.
        # ----------------------------------------------------------------
        target_name = None
        prompt = None

        if pipeline in ("lid_transport_food", "lid_transport_liquid"):
            # Lid transport: place lid on container then pick up container.
            container_synset = sel.get("container_synset", "")
            container_label = container_synset.split(".n.")[0].replace("_", " ") if ".n." in container_synset else container_synset
            # Find the container object by matching category to container_category
            container_cat = sel.get("container_category", "")
            for obj_name, obj_info in init_info.items():
                if obj_info.get("args", {}).get("category") == container_cat:
                    target_name = obj_name
                    break
            prompt = f"Place the lid on the {container_label}, then pick up the {container_label}."

        elif pipeline == "transfer":
            # Food transfer: move food from source to dest.
            food_synset = sel.get("food_synset", "")
            source_synset = sel.get("source_synset", "")
            dest_synset = sel.get("dest_synset", "")
            food_label = food_synset.split(".n.")[0].replace("_", " ") if ".n." in food_synset else "food"
            source_label = source_synset.split(".n.")[0].replace("_", " ") if ".n." in source_synset else "source"
            dest_label = dest_synset.split(".n.")[0].replace("_", " ") if ".n." in dest_synset else "destination"
            # Find the food object
            food_cat = food_synset.split(".n.")[0] if ".n." in food_synset else ""
            for obj_name, obj_info in init_info.items():
                if obj_info.get("args", {}).get("category") == food_cat:
                    target_name = obj_name
                    break
            prompt = f"Transfer the {food_label} from the {source_label} to the {dest_label}."

        else:
            # Clutter / liquid_transport / generic pickup: look for target
            # in active_object_summary or fall back to selection.target_synset.
            for entry in diag.get("active_object_summary", []):
                if entry.get("role") == "target":
                    target_name = entry.get("scene_object_name")
                    break
            if not target_name:
                # Fall back: match target_synset category against init_info
                target_synset = sel.get("target_synset", "")
                target_cat = target_synset.split(".n.")[0] if ".n." in target_synset else ""
                for obj_name, obj_info in init_info.items():
                    if obj_info.get("args", {}).get("category") == target_cat:
                        target_name = obj_name
                        break
            target_synset = sel.get("target_synset", "")
            target_label = target_synset.split(".n.")[0].replace("_", " ") if ".n." in target_synset else "object"
            prompt = f"Pick up the {target_label} on the {surface_label}."

        if not target_name:
            print(f"  Skipping {scene_key}: could not resolve target object (pipeline={pipeline})")
            continue
        if not prompt:
            prompt = f"Manipulate the objects on the {surface_label}."

        # Determine target room from scene JSON
        target_obj = init_info.get(target_name, {})
        target_rooms = target_obj.get("args", {}).get("in_rooms", [])
        if surface_name and surface_name in init_info:
            surface_rooms = init_info[surface_name].get("args", {}).get("in_rooms", [])
            target_rooms = list(set(target_rooms + surface_rooms))

        scenes.append({
            "name": scene_key,
            "scene_file": str(scene_file),
            "scene_model": diag.get("scene_model", scene_dir.name),
            "target_name": target_name,
            "surface_name": surface_name,
            "target_rooms": target_rooms,
            "prompt": prompt,
            "pipeline": pipeline,
            "activity_name": diag.get("activity_name", ""),
        })

    if max_scenes:
        scenes = scenes[:max_scenes]
    return scenes


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def build_og_config(scene_info: dict, args):
    """Build OmniGibson config for a single scene (no BDDL)."""
    if args.headless:
        gm.HEADLESS = True

    # Detect scene type from the snapshot so we handle both
    # InteractiveTraversableScene snapshots (BEHAVIOR scenes like Rs_int)
    # and pipeline-generated empty Scene snapshots (clutter/stack/transfer
    # pipelines produce these via og.sim.save() on a base Scene).
    _scene_header = json.loads(Path(scene_info["scene_file"]).read_text(encoding="utf-8"))
    _scene_class = _scene_header.get("init_info", {}).get("class_name", "InteractiveTraversableScene")

    if _scene_class == "Scene":
        # Pipeline-generated snapshot: no named scene_model, just load
        # the full state from scene_file. Mirrors the teleop pattern.
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

    robot_cfg = {
        "type": "FrankaMounted",
        "name": "agent_0",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {"image_height": 256, "image_width": 256},
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

    # Extract robot setup from scene JSON
    scene_data = json.loads(Path(scene_info["scene_file"]).read_text(encoding="utf-8"))
    init_info = scene_data.get("objects_info", {}).get("init_info", {})
    state_registry = scene_data.get("state", {}).get("registry", {}).get("object_registry", {})
    for obj_name, obj_info in init_info.items():
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

    env_cfg = {
        "action_frequency": args.action_frequency,
        "rendering_frequency": args.rendering_frequency,
        "physics_frequency": args.physics_frequency,
        "external_sensors": _build_eval_external_sensors(args),
    }

    # No BDDL task — use DummyTask
    task_cfg = {"type": "DummyTask"}

    # For pipeline-generated Scene snapshots, the robot is already baked
    # into scene_file; passing an extra robot_cfg would double-instantiate.
    # InteractiveTraversableScene snapshots (BEHAVIOR scenes) don't include
    # the robot, so we need the explicit robot_cfg there.
    robots_block = [] if _scene_class == "Scene" else [robot_cfg]
    return {
        "scene": scene_cfg,
        "robots": robots_block,
        "objects": [],
        "task": task_cfg,
        "env": env_cfg,
    }


def _setup_eval_cameras(env) -> None:
    """Position cam_opposite + companions to match the teleop camera rig.

    Mirrors ``sentinel.data.playback._setup_cameras_from_scene`` and
    ``omnigibson/examples/teleoperation/so101_franka_teleop.py``: pulls
    the scene's ``support_surface`` and any non-robot object, asks
    ``task_generation.utils.video.build_video_view_specs`` for the
    opposite-side overview placement, and applies it via
    ``setup_cameras`` (which also syncs the viewer camera).
    """
    try:
        from omnigibson.task_generation.utils.video import (
            build_video_view_specs,
            setup_cameras,
        )
    except ImportError as e:
        print(f"[Eval] WARNING: camera setup helpers not importable ({e}); "
              f"cam_opposite will stay at its default pose and frames will "
              f"be blank.")
        return

    scene = env.scene
    if scene is None or not env.robots:
        return
    robot = env.robots[0]
    support_obj = scene.object_registry("name", "support_surface")
    if support_obj is None:
        print("[Eval] WARNING: scene has no 'support_surface' object; "
              "cam_opposite stays at its default pose.")
        return
    target_obj = next(
        (obj for obj in scene.objects if obj is not robot and obj is not support_obj),
        support_obj,
    )
    views = build_video_view_specs(None, robot, target_obj, support_obj=support_obj)
    setup_cameras(env, views)
    print(f"[Eval] Positioned cam_opposite via setup_cameras "
          f"(target={target_obj.name}, support={support_obj.name}).")


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float32)
    quat = np.clip(quat, -1.0, 1.0)
    w = quat[3]
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    axis = quat[:3] / sin_half
    return (axis * angle).astype(np.float32)


def extract_obs(env, robot, prompt, policy_cameras=None):
    from omnigibson.utils.camera_setup import compose_main_image, normalize_policy_cameras

    raw_obs, _ = env.get_obs()
    external = raw_obs.get("external", {})
    cams = normalize_policy_cameras(policy_cameras)
    rgb_by_cam = {
        name: external[name]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        for name in cams
    }
    main_rgb = compose_main_image(rgb_by_cam, cams)

    robot_obs = raw_obs.get(robot.name, {})
    wrist_rgb = None
    for name, obs in robot_obs.items():
        if isinstance(obs, dict) and "rgb" in obs:
            wrist_rgb = obs["rgb"][..., :3].cpu().numpy().astype(np.uint8)
            break
    if wrist_rgb is None:
        wrist_rgb = np.zeros_like(main_rgb)

    eef_pos = robot.get_relative_eef_position().cpu().numpy().astype(np.float32)
    eef_quat = robot.get_relative_eef_orientation().cpu().numpy().astype(np.float32)
    eef_axisangle = quat2axisangle(eef_quat)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    gripper_qpos = robot.get_joint_positions()[gripper_idx].cpu().numpy().astype(np.float32)
    # 8D state: eef_pos(3) + axisangle(3) + gripper_qpos(2). Matches
    # the training-time layout in sentinel/data/playback.py so ckpts
    # trained there can consume eval obs without a shape mismatch.
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos])

    return {
        "main_images": main_rgb,
        "wrist_images": wrist_rgb,
        # Upstream openpi_action_model.obs_processor unconditionally reads
        # env_obs["extra_view_images"]; send None when no third camera.
        "extra_view_images": None,
        "states": state,
        "task_descriptions": prompt,
    }


def check_grasp(robot, target_obj):
    """Check if robot is grasping the target object."""
    if target_obj is None:
        return False
    try:
        contacts = robot.contact_list()
        for contact in contacts:
            if target_obj.name in str(contact):
                return True
    except Exception:
        pass
    # Fallback: check if target is above its original position (lifted)
    try:
        pos = target_obj.get_position_orientation()[0]
        if float(pos[2]) > 1.0:  # lifted significantly
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Policy client
# ---------------------------------------------------------------------------

def connect_policy(args):
    if args.use_openpi_client:
        from openpi_client import websocket_client_policy as _wcp
        return _wcp.WebsocketClientPolicy(host=args.host, port=args.port), "openpi"
    else:
        from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
        policy = WebsocketClientPolicy(host=args.host, port=args.port)
        policy.reset()
        return policy, "omnigibson"


def query_policy(policy, obs, client_type):
    if client_type == "openpi":
        result = policy.infer(obs)
        chunk = np.asarray(result["actions"], dtype=np.float32)
    else:
        action = policy.act(obs)
        chunk = action.detach().cpu().numpy().astype(np.float32)
    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]
    return chunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    import omnigibson as og

    # Resolve benchmark source: local path or HF dataset repo_id. HF
    # repos are snapshot-downloaded into the hf cache; subsequent runs
    # reuse via content-hash.
    from sentinel.data.hf_benchmark import resolve_benchmark_root
    resolved_root = resolve_benchmark_root(
        args.benchmark_root, revision=args.benchmark_revision,
    )
    if str(resolved_root) != args.benchmark_root:
        print(f"Resolved benchmark '{args.benchmark_root}' @ {args.benchmark_revision} "
              f"-> {resolved_root}")

    # Discover scenes
    scenes = discover_scenes(
        str(resolved_root),
        scene_names=args.scenes,
        max_scenes=args.max_scenes,
    )
    print(f"Discovered {len(scenes)} valid scenes")

    # Connect to policy
    policy, client_type = connect_policy(args)
    print(f"Connected to policy server at {args.host}:{args.port} ({client_type})")

    all_results = []

    for scene_idx, scene_info in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"Scene {scene_idx+1}/{len(scenes)}: {scene_info['name']}")
        print(f"Prompt: {scene_info['prompt']}")
        print(f"Target: {scene_info['target_name']}")
        print(f"Rooms: {scene_info['target_rooms']}")
        print(f"{'='*60}")

        # Build and create env
        cfg = build_og_config(scene_info, args)

        try:
            if og.sim is not None:
                og.sim.stop()
                og.clear()

            env = og.Environment(configs=cfg)
            # Force sensor resolution via setters (Kit viewport init can
            # override the VisionSensor.__init__ kwargs back to the app
            # default), then reload obs space so reset() doesn't fail.
            ext_sensors = env.external_sensors or {}
            if ext_sensors:
                for _cam in ext_sensors.values():
                    _cam.image_height = args.camera_resolution
                    _cam.image_width = args.camera_resolution
                env.load_observation_space()
            env.reset()

            # Position cam_opposite / cam_left / cam_right at the canonical
            # opposite-overview pose relative to robot + support + target.
            # Without this, DataPlaybackWrapper / the viewport leaves the
            # external sensor at the world origin pointing at nothing, and
            # every frame is a uniform gray -- same bug that bit Stage 1 of
            # the teleop playback pipeline.
            _setup_eval_cameras(env)
        except Exception as e:
            print(f"  FAILED to load scene: {e}")
            all_results.append({
                "scene_name": scene_info["name"],
                "prompt": scene_info["prompt"],
                "status": "load_failed",
                "error": str(e),
            })
            continue

        robot = env.robots[0]
        action_space = robot.action_space

        # Find target object
        target_obj = None
        try:
            target_obj = env.scene.object_registry("name", scene_info["target_name"])
        except Exception:
            print(f"  Warning: target object '{scene_info['target_name']}' not found in scene")

        # Settle
        for _ in range(10):
            robot.keep_still()
            og.sim.step()
        for _ in range(2):
            og.sim.render()

        obs = extract_obs(env, robot, scene_info["prompt"], policy_cameras=args.policy_external_cameras)
        frames = [obs["main_images"]] if args.save_video else []

        # Rollout
        step_idx = 0
        done = False
        success = False
        total_reward = 0.0

        while step_idx < args.max_steps and not done:
            chunk = query_policy(policy, obs, client_type)
            chunk_len = min(args.execute_horizon, len(chunk), args.max_steps - step_idx)

            for ci in range(chunk_len):
                action = chunk[ci].copy()
                action[-1] = np.sign(action[-1]) if abs(action[-1]) > 0.01 else -1.0
                action_clipped = np.clip(action[:action_space.shape[0]], action_space.low, action_space.high)

                raw_obs, reward, terminated, truncated, info = env.step(
                    torch.from_numpy(action_clipped).unsqueeze(0)
                )
                obs = extract_obs(env, robot, scene_info["prompt"], policy_cameras=args.policy_external_cameras)
                if args.save_video:
                    frames.append(obs["main_images"])
                step_idx += 1
                total_reward += float(reward)

                if check_grasp(robot, target_obj):
                    success = True
                    done = True
                    break

            if step_idx % 50 == 0 or step_idx == 1:
                print(f"  Step {step_idx}/{args.max_steps} | success={success}")

        result = {
            "scene_name": scene_info["name"],
            "prompt": scene_info["prompt"],
            "target": scene_info["target_name"],
            "rooms": scene_info["target_rooms"],
            "status": "completed",
            "steps": step_idx,
            "success": success,
            "total_reward": total_reward,
        }
        all_results.append(result)
        print(f"  Result: success={success}, steps={step_idx}")

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")

        if args.save_video and frames:
            video_path = output_dir / f"{scene_info['name']}.mp4"
            imageio.mimsave(str(video_path), frames, fps=10)

    # Summary
    print(f"\n{'='*60}")
    print("Benchmark Evaluation Summary")
    print(f"{'='*60}")
    completed = [r for r in all_results if r["status"] == "completed"]
    n_total = len(completed)
    n_success = sum(1 for r in completed if r["success"])
    n_failed_load = sum(1 for r in all_results if r["status"] == "load_failed")
    print(f"Scenes evaluated: {n_total} ({n_failed_load} failed to load)")
    print(f"Success rate: {n_success}/{n_total} ({n_success/max(n_total,1)*100:.1f}%)")
    if n_total > 0:
        print(f"Avg steps: {np.mean([r['steps'] for r in completed]):.1f}")
    print(f"Results: {results_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "n_scenes": n_total,
        "n_success": n_success,
        "n_failed_load": n_failed_load,
        "success_rate": n_success / max(n_total, 1),
        "results": all_results,
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
