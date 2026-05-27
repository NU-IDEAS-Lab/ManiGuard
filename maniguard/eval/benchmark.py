#!/usr/bin/env python3
"""Evaluate a websocket VLA policy on the ManiGuard benchmark.

Usage:
    python -m maniguard.eval.benchmark --config configs/eval/sim_table_25k.yaml
    python -m maniguard.eval.benchmark --config configs/eval/sim_table_25k.yaml --max-steps 500
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniguard.eval.eval_config import EvalConfig, config_from_cli
from maniguard.eval.scene_discovery import discover_scenes



# ---------------------------------------------------------------------------
# Isaac Sim / OmniGibson bootstrap
# ---------------------------------------------------------------------------

def _init_omnigibson(cfg: EvalConfig):
    if not cfg.longfinger:
        os.environ["SENTINEL_SKIP_LONGFINGER"] = "1"
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        pass
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if cfg.headless:
        gm.HEADLESS = True


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _build_eval_external_sensors(cfg: EvalConfig):
    from maniguard.utils.camera_setup import (
        EXTERNAL_CAMERA_NAMES,
        build_external_camera_configs,
        normalize_policy_cameras,
    )
    policy_cams = normalize_policy_cameras(cfg.policy_cameras)
    names = []
    for name in list(policy_cams) + [EXTERNAL_CAMERA_NAMES[0]]:
        if name not in names:
            names.append(name)
    return build_external_camera_configs(names=names, resolution=cfg.camera_resolution)


def build_og_config(scene_info: dict, cfg: EvalConfig):
    _scene_header = json.loads(Path(scene_info["scene_file"]).read_text(encoding="utf-8"))
    _scene_class = _scene_header.get("init_info", {}).get("class_name", "")

    if _scene_class == "InteractiveTraversableScene" and scene_info.get("scene_model"):
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_info["scene_model"],
            "scene_file": scene_info["scene_file"],
            "scene_instance": None,
            "include_robots": True,
        }
        _objects_info = _scene_header.get("objects_info", {}).get("init_info", {})
        _robot_has_rooms = any(
            obj.get("class_name") == "FrankaPanda" and obj.get("args", {}).get("in_rooms")
            for obj in _objects_info.values()
        )
        if scene_info.get("target_rooms") and _robot_has_rooms:
            scene_cfg["load_room_instances"] = scene_info["target_rooms"]
    else:
        scene_cfg = {
            "type": "Scene",
            "scene_file": scene_info["scene_file"],
        }

    env_cfg = {
        "action_frequency": cfg.action_frequency,
        "rendering_frequency": cfg.rendering_frequency,
        "physics_frequency": cfg.physics_frequency,
        "external_sensors": _build_eval_external_sensors(cfg),
    }

    return {
        "scene": scene_cfg,
        "robots": [],
        "objects": [],
        "task": {"type": "DummyTask"},
        "env": env_cfg,
    }


def _setup_eval_cameras(env, scene_info: dict) -> None:
    import omnigibson as og
    from maniguard.task_generation.utils.video import eye_lookat_to_quat

    cameras = scene_info.get("cameras", [])
    if not cameras:
        print("[Eval] WARNING: no cameras in scene_info; frames will be default pose.")
        return

    ext_sensors = env.external_sensors or {}
    placed = 0
    for cam_info in cameras:
        sensor_name = cam_info.get("sensor_name")
        sensor = ext_sensors.get(sensor_name)
        if sensor is None:
            continue
        eye = cam_info["eye"]
        lookat = cam_info["lookat"]
        orientation = cam_info.get("orientation") or eye_lookat_to_quat(eye, lookat).tolist()
        sensor.set_position_orientation(position=eye, orientation=orientation, frame="world")
        placed += 1

    opp = next((c for c in cameras if c.get("sensor_name") == "cam_opposite"), cameras[0])
    ori = opp.get("orientation") or eye_lookat_to_quat(opp["eye"], opp["lookat"]).tolist()
    og.sim.viewer_camera.set_position_orientation(position=opp["eye"], orientation=ori)
    print(f"[Eval] Positioned {placed} cameras from diagnostics.")


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

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


def extract_obs(env, robot, prompt, cfg: EvalConfig):
    from maniguard.utils.camera_setup import compose_main_image, normalize_policy_cameras

    raw_obs, _ = env.get_obs()
    external = raw_obs.get("external", {})
    cams = normalize_policy_cameras(cfg.policy_cameras)
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
        if not getattr(extract_obs, "_wrist_warned", False):
            print(f"[Eval] WARNING: no wrist camera found in robot obs keys={list(robot_obs.keys())}; using black image")
            extract_obs._wrist_warned = True
        wrist_rgb = np.zeros_like(main_rgb)

    eef_pos = robot.get_relative_eef_position().cpu().numpy().astype(np.float32)
    eef_quat = robot.get_relative_eef_orientation().cpu().numpy().astype(np.float32)
    eef_axisangle = quat2axisangle(eef_quat)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    gripper_qpos = robot.get_joint_positions()[gripper_idx].cpu().numpy().astype(np.float32)

    if cfg.state_mode == "eef_8d":
        import torch as _torch
        from omnigibson.utils.transform_utils import quat2euler as _quat2euler
        eef_euler = _quat2euler(_torch.as_tensor(eef_quat)).cpu().numpy().astype(np.float32)
        state = np.concatenate([eef_pos, eef_euler, gripper_qpos])
    elif cfg.state_mode == "eef_8d_axisangle":
        state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos])
    elif cfg.state_mode == "eef_7d":
        gripper_scalar = np.mean(gripper_qpos).reshape(1).astype(np.float32)
        state = np.concatenate([eef_pos, eef_axisangle, gripper_scalar])
    elif cfg.state_mode == "joint":
        arm_positions = robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]]
        gripper_scalar = np.mean(gripper_qpos).reshape(1).astype(np.float32)
        state = np.concatenate([arm_positions.cpu().numpy().astype(np.float32), gripper_scalar])
    else:
        raise ValueError(f"Unknown state_mode: {cfg.state_mode}")

    return {
        "main_images": main_rgb,
        "wrist_images": wrist_rgb,
        "states": state,
        "task_descriptions": prompt,
    }


# ---------------------------------------------------------------------------
# Policy client
# ---------------------------------------------------------------------------

class _RandomPolicy:
    def __init__(self, action_dim=7):
        self._dim = action_dim
    def act(self, obs):
        return torch.from_numpy(
            np.random.uniform(-0.05, 0.05, size=(1, self._dim)).astype(np.float32)
        )


def connect_policy(cfg: EvalConfig):
    if cfg.random_policy:
        return _RandomPolicy(action_dim=cfg.action_dim), "random"
    if cfg.use_openpi_client:
        from openpi_client import websocket_client_policy as _wcp
        return _wcp.WebsocketClientPolicy(host=cfg.host, port=cfg.port), "openpi"
    else:
        from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
        policy = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
        policy.reset()
        return policy, "omnigibson"


def _remap_obs_for_openpi(obs: dict) -> dict:
    return {
        "observation/image": obs["main_images"],
        "observation/wrist_image": obs["wrist_images"],
        "observation/state": obs["states"],
        "prompt": obs["task_descriptions"],
    }


def query_policy(policy, obs, client_type):
    if client_type == "random":
        action = policy.act(obs)
        chunk = action.numpy() if hasattr(action, "numpy") else np.asarray(action)
    elif client_type == "openpi":
        result = policy.infer(_remap_obs_for_openpi(obs))
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
    cfg = config_from_cli()

    if not cfg.benchmark_root:
        raise ValueError("benchmark_root must be set in config or via --benchmark-root")
    if not cfg.output_dir:
        cfg.output_dir = str(REPO_ROOT / "outputs" / "benchmark_eval")

    print(f"[Eval] Config: {cfg.name}")
    print(f"[Eval] state={cfg.state_mode}, action_dim={cfg.action_dim}, "
          f"horizon={cfg.execute_horizon}, max_steps={cfg.max_steps}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    cfg.save_json(output_dir / "eval_config.json")
    print(f"[Eval] Config saved to {output_dir / 'eval_config.json'}")

    _init_omnigibson(cfg)
    import omnigibson as og

    from maniguard.data.hf_benchmark import resolve_benchmark_root
    resolved_root = resolve_benchmark_root(
        cfg.benchmark_root, revision=cfg.benchmark_revision,
    )
    if str(resolved_root) != cfg.benchmark_root:
        print(f"Resolved benchmark '{cfg.benchmark_root}' @ {cfg.benchmark_revision} "
              f"-> {resolved_root}")

    scenes = discover_scenes(
        str(resolved_root),
        scene_names=cfg.scenes,
        max_scenes=cfg.max_scenes,
    )
    if cfg.scene_filter:
        import fnmatch
        scenes = [s for s in scenes if fnmatch.fnmatch(s["name"], cfg.scene_filter)]
    print(f"Discovered {len(scenes)} valid scenes")

    policy, client_type = connect_policy(cfg)
    if client_type == "random":
        print("Using random policy (smoke test mode)")
    else:
        print(f"Connected to policy server at {cfg.host}:{cfg.port} ({client_type})")

    all_results = []

    for scene_idx, scene_info in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"Scene {scene_idx+1}/{len(scenes)}: {scene_info['name']}")
        print(f"Prompt: {scene_info['prompt']}")
        print(f"Target: {scene_info['target_name']}")
        print(f"Rooms: {scene_info['target_rooms']}")
        print(f"{'='*60}")

        og_cfg = build_og_config(scene_info, cfg)

        try:
            if og.sim is not None:
                og.sim.stop()
                og.clear()

            env = og.Environment(configs=og_cfg)
            ext_sensors = env.external_sensors or {}
            if ext_sensors:
                for _cam in ext_sensors.values():
                    _cam.image_height = cfg.camera_resolution
                    _cam.image_width = cfg.camera_resolution
                env.load_observation_space()
            env.reset()
            _setup_eval_cameras(env, scene_info=scene_info)
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

        if cfg.override_controller_config:
            print(f"  Overriding controllers: {list(cfg.override_controller_config.keys())}")
            robot.reload_controllers(cfg.override_controller_config)

        action_space = robot.action_space

        from maniguard.eval.goal_checker import build_goal_checker
        goal_checker = build_goal_checker(scene_info)
        if goal_checker is not None:
            goal_checker.resolve(env)
            if hasattr(goal_checker, "raw_region"):
                print(f"  Goal region: {goal_checker.raw_region.to_json()}")
            else:
                print(f"  Goals: {goal_checker.raw_conditions}")
        else:
            print(f"  Warning: no goal_region or goal_conditions in diagnostics — success will always be False")

        for _ in range(10):
            robot.keep_still()
            og.sim.step()
        for _ in range(2):
            og.sim.render()

        obs = extract_obs(env, robot, scene_info["prompt"], cfg)
        frames = [obs["main_images"]] if cfg.save_video else []

        step_idx = 0
        done = False
        success = False
        total_reward = 0.0
        goal_detail = {}

        while step_idx < cfg.max_steps and not done:
            chunk = query_policy(policy, obs, client_type)
            chunk_len = min(cfg.execute_horizon, len(chunk), cfg.max_steps - step_idx)

            for ci in range(chunk_len):
                action = chunk[ci].copy()
                if cfg.gripper_binarize:
                    action[-1] = np.sign(action[-1]) if abs(action[-1]) > 0.01 else -1.0
                action_clipped = np.clip(action[:action_space.shape[0]], action_space.low, action_space.high)

                _, reward, _, _, _ = env.step(
                    torch.from_numpy(action_clipped).unsqueeze(0)
                )
                obs = extract_obs(env, robot, scene_info["prompt"], cfg)
                if cfg.save_video:
                    frames.append(obs["main_images"])
                step_idx += 1
                total_reward += float(reward)

                if goal_checker is not None:
                    success, goal_detail = goal_checker.check(env)
                if success:
                    done = True
                    break

            if step_idx % 50 == 0 or step_idx == 1:
                print(f"  Step {step_idx}/{cfg.max_steps} | success={success} | goals={goal_detail}")

        result = {
            "scene_name": scene_info["name"],
            "prompt": scene_info["prompt"],
            "target": scene_info["target_name"],
            "pipeline": scene_info.get("pipeline", ""),
            "rooms": scene_info["target_rooms"],
            "status": "completed",
            "steps": step_idx,
            "success": success,
            "goal_detail": goal_detail,
            "total_reward": total_reward,
        }
        all_results.append(result)
        print(f"  Result: success={success}, steps={step_idx}")

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")

        if cfg.save_video and frames:
            video_path = output_dir / f"{scene_info['name']}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
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
