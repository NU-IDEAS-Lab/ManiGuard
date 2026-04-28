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

def _init_omnigibson():
    """Lazy-init Isaac Sim + OmniGibson. Called once before env creation."""
    try:
        import isaacsim  # noqa: F401
    except ImportError as exc:
        log.debug("isaacsim bootstrap import skipped (OmniGibson will self-bootstrap): %s", exc)
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
    p.add_argument("--random-policy", action="store_true",
                   help="Use uniform random actions instead of a policy "
                        "server. For pipeline smoke-testing only.")
    p.add_argument("--eval-profile", type=str, default="pi05_stack_cube",
                   help="Eval profile name (matches a yaml in sentinel/eval/profiles/ "
                        "or a built-in). Controls state_mode, action_dim, "
                        "execute_horizon, gripper binarization. "
                        "Available: pi05_stack_cube, pi0_libero, pi05_libero, gr00t.")
    return p.parse_args()


def _build_eval_external_sensors(args):
    from sentinel.utils.camera_setup import (
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
# Scene discovery — delegated to lightweight module (no torch/imageio deps)
# so scripts/run_benchmark_all_scenes.sh can import it from any Python env.
# ---------------------------------------------------------------------------

from sentinel.eval.scene_discovery import discover_scenes  # noqa: F401
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def build_og_config(scene_info: dict, args):
    """Build OmniGibson config for a single scene (no BDDL)."""
    if args.headless:
        gm.HEADLESS = True

    # Detect scene type. New pipeline snapshots have correct in_rooms on
    # all task objects + robot, so InteractiveTraversableScene partial
    # room loading works. Trimmed scenes also work as plain Scene.
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
        if scene_info.get("target_rooms"):
            scene_cfg["load_room_instances"] = scene_info["target_rooms"]
    else:
        scene_cfg = {
            "type": "Scene",
            "scene_file": scene_info["scene_file"],
        }

    env_cfg = {
        "action_frequency": args.action_frequency,
        "rendering_frequency": args.rendering_frequency,
        "physics_frequency": args.physics_frequency,
        "external_sensors": _build_eval_external_sensors(args),
    }

    task_cfg = {"type": "DummyTask"}

    # Robot is baked into scene_file — no separate robot_cfg needed.
    return {
        "scene": scene_cfg,
        "robots": [],
        "objects": [],
        "task": task_cfg,
        "env": env_cfg,
    }


def _setup_eval_cameras(env, scene_info: dict) -> None:
    """Position external cameras from diagnostics-saved poses.

    The pipeline saves exact camera eye/lookat/orientation in
    ``diagnostics.jsonl`` (field ``cameras``). We apply those directly —
    no re-computation, no support-surface lookup, no fallback.
    """
    import omnigibson as og
    from sentinel.task_generation.utils.video import eye_lookat_to_quat

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

    # Sync viewer camera to cam_opposite.
    opp = next((c for c in cameras if c.get("sensor_name") == "cam_opposite"), cameras[0])
    ori = opp.get("orientation") or eye_lookat_to_quat(opp["eye"], opp["lookat"]).tolist()
    og.sim.viewer_camera.set_position_orientation(position=opp["eye"], orientation=ori)

    print(f"[Eval] Positioned {placed} cameras from diagnostics.")


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


def extract_obs(env, robot, prompt, policy_cameras=None, state_mode="eef_8d",
                table_top_z=0.0):
    from sentinel.utils.camera_setup import compose_main_image, normalize_policy_cameras

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
    # Re-reference z to be height above the support surface (matches IsaacLab
    # stack-cube convention where state.z ∈ [0.01, 0.29] = above-table height).
    if table_top_z != 0.0:
        eef_pos = eef_pos.copy()
        eef_pos[2] = float(eef_pos[2]) - float(table_top_z)
    eef_quat = robot.get_relative_eef_orientation().cpu().numpy().astype(np.float32)
    eef_axisangle = quat2axisangle(eef_quat)
    # IsaacLab stack-cube data uses Euler (rpy) — see GR00T/pi05_stack_cube norm
    # stats: state.pitch.std≈0.04 (gripper down), roll/yaw wrap ±π. Axis-angle
    # would be smooth and bounded; Euler matches the bimodal distribution.
    import torch as _torch
    from omnigibson.utils.transform_utils import quat2euler as _quat2euler
    eef_euler = _quat2euler(_torch.as_tensor(eef_quat)).cpu().numpy().astype(np.float32)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    gripper_qpos = robot.get_joint_positions()[gripper_idx].cpu().numpy().astype(np.float32)

    if state_mode == "eef_8d":
        # eef_pos(3) + euler_rpy(3) + gripper_qpos(2) — IsaacLab stack-cube layout
        state = np.concatenate([eef_pos, eef_euler, gripper_qpos])
    elif state_mode == "eef_7d":
        # eef_pos(3) + axisangle(3) + gripper_scalar(1) — LIBERO layout
        gripper_scalar = np.mean(gripper_qpos).reshape(1).astype(np.float32)
        state = np.concatenate([eef_pos, eef_axisangle, gripper_scalar])
    elif state_mode == "joint":
        # joint_positions(7) + gripper_scalar(1)
        arm_positions = robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]]
        gripper_scalar = np.mean(gripper_qpos).reshape(1).astype(np.float32)
        state = np.concatenate([arm_positions.cpu().numpy().astype(np.float32), gripper_scalar])
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")

    return {
        "main_images": main_rgb,
        "wrist_images": wrist_rgb,
        # Upstream openpi_action_model.obs_processor unconditionally reads
        # env_obs["extra_view_images"]; send None when no third camera.
        "extra_view_images": None,
        "states": state,
        "task_descriptions": prompt,
    }



# ---------------------------------------------------------------------------
# Policy client
# ---------------------------------------------------------------------------

class _RandomPolicy:
    """Uniform random actions for smoke-testing the eval pipeline."""
    def __init__(self, action_dim=7):
        self._dim = action_dim
    def act(self, obs):
        return torch.from_numpy(
            np.random.uniform(-0.05, 0.05, size=(1, self._dim)).astype(np.float32)
        )


def connect_policy(args, action_dim=7):
    if getattr(args, "random_policy", False):
        return _RandomPolicy(action_dim=action_dim), "random"
    if args.use_openpi_client:
        from openpi_client import websocket_client_policy as _wcp
        return _wcp.WebsocketClientPolicy(host=args.host, port=args.port), "openpi"
    else:
        from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
        policy = WebsocketClientPolicy(host=args.host, port=args.port)
        policy.reset()
        return policy, "omnigibson"


def query_policy(policy, obs, client_type):
    if client_type == "random":
        action = policy.act(obs)
        chunk = action.numpy() if hasattr(action, "numpy") else np.asarray(action)
    elif client_type == "openpi":
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

    from sentinel.eval.profiles import get_profile
    profile = get_profile(args.eval_profile)
    print(f"[Eval] Profile: {profile.name} (state={profile.state_mode}, "
          f"action_dim={profile.action_dim}, horizon={profile.execute_horizon})")

    # Profile overrides CLI defaults (CLI still wins if explicitly set).
    if args.execute_horizon == 5 and profile.execute_horizon != 5:
        args.execute_horizon = profile.execute_horizon
    if args.policy_external_cameras is None:
        args.policy_external_cameras = profile.policy_cameras

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    _init_omnigibson()
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
    policy, client_type = connect_policy(args, action_dim=profile.action_dim)
    if client_type == "random":
        print("Using random policy (smoke test mode)")
    else:
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

        # If the profile requires a different controller (e.g. GR00T N1.6-DROID
        # wants a delta-JointController but the scene was generated with OSC),
        # swap controllers now. This preserves the loaded joint state; only the
        # controller logic + action_space are rebuilt. See reload_controllers().
        if profile.override_controller_config:
            print(f"  Overriding controllers: {list(profile.override_controller_config.keys())}")
            robot.reload_controllers(profile.override_controller_config)

        action_space = robot.action_space

        # Find target object
        target_obj = None
        try:
            target_obj = env.scene.object_registry("name", scene_info["target_name"])
        except Exception:
            print(f"  Warning: target object '{scene_info['target_name']}' not found in scene")

        # Look up the support surface and read its top z. Used to re-reference
        # eef_pos.z so the policy sees "height above table" — matches the
        # IsaacLab stack-cube training distribution (z ∈ [0.01, 0.29]).
        table_top_z = 0.0
        surface_name = scene_info.get("surface_name")
        if surface_name:
            try:
                surf_obj = env.scene.object_registry("name", surface_name)
                if surf_obj is not None:
                    _, aabb_max = surf_obj.aabb
                    table_top_z = float(aabb_max[2])
                    print(f"  Support surface '{surface_name}' top z = {table_top_z:.4f}")
                else:
                    print(f"  Warning: support surface '{surface_name}' not found; "
                          f"eef_pos.z will not be re-referenced")
            except Exception as e:
                print(f"  Warning: could not read AABB of '{surface_name}': {e}")

        # Build success checker from diagnostics goal_region / goal_conditions fields
        from sentinel.eval.goal_checker import build_goal_checker
        goal_checker = build_goal_checker(scene_info)
        if goal_checker is not None:
            goal_checker.resolve(env)
            if hasattr(goal_checker, "raw_region"):
                print(f"  Goal region: {goal_checker.raw_region.to_json()}")
            else:
                print(f"  Goals: {goal_checker.raw_conditions}")
        else:
            print(f"  Warning: no goal_region or goal_conditions in diagnostics — success will always be False")

        # Settle
        for _ in range(10):
            robot.keep_still()
            og.sim.step()
        for _ in range(2):
            og.sim.render()

        obs = extract_obs(env, robot, scene_info["prompt"], policy_cameras=args.policy_external_cameras, state_mode=profile.state_mode, table_top_z=table_top_z)
        frames = [obs["main_images"]] if args.save_video else []

        # Rollout
        step_idx = 0
        done = False
        success = False
        total_reward = 0.0
        goal_detail = {}

        while step_idx < args.max_steps and not done:
            chunk = query_policy(policy, obs, client_type)
            chunk_len = min(args.execute_horizon, len(chunk), args.max_steps - step_idx)

            for ci in range(chunk_len):
                action = chunk[ci].copy()
                if profile.gripper_binarize:
                    action[-1] = np.sign(action[-1]) if abs(action[-1]) > 0.01 else -1.0
                action_clipped = np.clip(action[:action_space.shape[0]], action_space.low, action_space.high)

                raw_obs, reward, terminated, truncated, info = env.step(
                    torch.from_numpy(action_clipped).unsqueeze(0)
                )
                obs = extract_obs(env, robot, scene_info["prompt"], policy_cameras=args.policy_external_cameras, state_mode=profile.state_mode, table_top_z=table_top_z)
                if args.save_video:
                    frames.append(obs["main_images"])
                step_idx += 1
                total_reward += float(reward)

                if goal_checker is not None:
                    success, goal_detail = goal_checker.check(env)
                if success:
                    done = True
                    break

            if step_idx % 50 == 0 or step_idx == 1:
                print(f"  Step {step_idx}/{args.max_steps} | success={success} | goals={goal_detail}")

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

        if args.save_video and frames:
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
