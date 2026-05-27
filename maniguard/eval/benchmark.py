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

    # three_cam models consume the wrist view (right_wrist_0_rgb, unmasked).
    # SFT data was recorded with the wrist Camera relocated to a canonical
    # pose; reuse that exact patch so the eval wrist matches training.
    if getattr(cfg, "obs_layout", "single_plus_wrist") == "three_cam":
        from maniguard.data.curobo._sft_recorder import install_wrist_camera_patch
        install_wrist_camera_patch()
        print("[Eval] three_cam: installed canonical wrist-camera patch.")


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


def _eef_jacobian_arm(robot, arm, arm_cols_np):
    """(6, n_arm_dof) base-frame eef Jacobian (numpy), replicating OmniGibson's
    controllable_object eef-task extraction + arm-DOF column selection."""
    link_name = robot.eef_link_names[arm]
    start_idx = 0 if robot.fixed_base else 6
    link_idx = robot._articulation_view.get_body_index(link_name)
    jac = robot.get_relative_jacobian().cpu().numpy()  # (n_links, 6, n_joints[+6])
    j_link = jac[-(robot.n_links - link_idx), :, start_idx:start_idx + robot.n_joints]
    return j_link[:, arm_cols_np].astype(np.float64)


def eef_delta_to_joint_action(robot, eef_delta, action_space):
    """Convert a 7-D policy action [dpos(3), daa(3), gripper(1)] into the
    robot's JointController action [target_arm_joints(7), gripper(1)].

    One damped-least-squares Jacobian step maps the commanded base-frame eef
    twist to a joint delta; the absolute joint target (current + dq) is what
    the JointController PD-tracks each step. This reproduces the SFT mechanism
    (cuRobo joint targets tracked by a JointController), so the joint path that
    realizes the eef delta matches training instead of diverging like OSC.

    The 6-D eef delta IS the base-frame twist to realize (orientation delta is
    axis-angle, since R_target = dR @ R_cur)."""
    arm = robot.default_arm
    err6 = np.asarray(eef_delta[:6], dtype=np.float64)

    arm_cols_t = robot.arm_control_idx[arm]
    arm_cols_np = arm_cols_t.cpu().numpy() if hasattr(arm_cols_t, "cpu") else np.asarray(arm_cols_t)
    J = _eef_jacobian_arm(robot, arm, arm_cols_np)
    lam = 0.05
    dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), err6)

    q_all = robot.get_joint_positions().cpu().numpy()
    target_arm = (q_all[arm_cols_np] + dq).astype(np.float32)

    # Action vector is ordered [arm_0 (7), gripper_0 (1)]; arm command is the
    # absolute joint target, gripper is the (binarized) command passed through.
    out = np.zeros(action_space.shape[0], dtype=np.float32)
    out[: len(target_arm)] = target_arm
    out[-1] = float(eef_delta[-1])
    return out


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
        "images_by_cam": rgb_by_cam,
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


def _remap_obs_for_openpi(obs: dict, cfg: EvalConfig) -> dict:
    layout = getattr(cfg, "obs_layout", "single_plus_wrist")
    if layout == "three_cam":
        from maniguard.utils.camera_setup import normalize_policy_cameras
        cams = normalize_policy_cameras(cfg.policy_cameras)
        if len(cams) < 2:
            raise ValueError(
                "obs_layout='three_cam' needs 2 external policy_cameras in order "
                f"(e.g. [cam_left, cam_right]); got {cams}"
            )
        # ManiGuardInputs assigns slots: cams[0]/image_left -> base_0_rgb,
        # wrist -> left_wrist_0_rgb, cams[1]/image_right -> right_wrist_0_rgb
        # (matches openpi pi05_pnp_clutter_3cam_lora).
        return {
            "observation/image_left": obs["images_by_cam"][cams[0]],
            "observation/image_right": obs["images_by_cam"][cams[1]],
            "observation/wrist_image": obs["wrist_images"],
            "observation/state": obs["states"],
            "prompt": obs["task_descriptions"],
        }
    return {
        "observation/image": obs["main_images"],
        "observation/wrist_image": obs["wrist_images"],
        "observation/state": obs["states"],
        "prompt": obs["task_descriptions"],
    }


def query_policy(policy, obs, client_type, cfg):
    if client_type == "random":
        action = policy.act(obs)
        chunk = action.numpy() if hasattr(action, "numpy") else np.asarray(action)
    elif client_type == "openpi":
        result = policy.infer(_remap_obs_for_openpi(obs, cfg))
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

    from maniguard.data.scene.hf_benchmark import resolve_benchmark_root
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
        if cfg.prompt_template:
            from maniguard.data.lerobot.lerobot_writer import episode_prompt
            scene_info["prompt"] = episode_prompt(scene_info["target_name"], cfg.prompt_template)
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
            robot = env.robots[0]

            # Resize external (cam_left/right) AND the robot's onboard wrist
            # camera to the policy resolution. The wrist cam defaults to
            # 128x128; SFT recorded it at cfg.camera_resolution, so leaving it
            # at the default feeds the wrist (left_wrist_0) slot a degraded
            # half-res image vs training.
            from omnigibson.sensors import VisionSensor as _VisionSensor
            _resized = False
            for _cam in (env.external_sensors or {}).values():
                _cam.image_height = cfg.camera_resolution
                _cam.image_width = cfg.camera_resolution
                _resized = True
            for _sensor in (robot.sensors or {}).values():
                if isinstance(_sensor, _VisionSensor):
                    _sensor.image_height = cfg.camera_resolution
                    _sensor.image_width = cfg.camera_resolution
                    _resized = True
            if _resized:
                env.load_observation_space()
            env.reset()
            _setup_eval_cameras(env, scene_info=scene_info)

            # Reload the eval controller AFTER env.reset(): reset restores the
            # scene snapshot's (JointController) controller-state into the
            # matching baked controller; reloading after avoids loading that
            # state into the new controller (IK's control_filter / state-length
            # mismatch -> KeyError). The fresh controller's goal is then
            # initialized by the zero-delta warmup below.
            override_cc = cfg.override_controller_config
            if override_cc is None and cfg.controller_preset:
                from maniguard.envs.frozen_task_runtime import CONTROLLER_PRESETS
                if cfg.controller_preset not in CONTROLLER_PRESETS:
                    raise ValueError(
                        f"unknown controller_preset {cfg.controller_preset!r}; "
                        f"choices: {sorted(CONTROLLER_PRESETS)}"
                    )
                override_cc = json.loads(json.dumps(CONTROLLER_PRESETS[cfg.controller_preset]))
            if override_cc and cfg.joint_pos_kp is not None:
                arm0 = override_cc.get("arm_0", {})
                if arm0.get("name") == "JointController":
                    # Implicit position drive (stable at high stiffness) instead
                    # of explicit impedance torque. joint_pos_kp -> isaac drive
                    # stiffness; isaac_kd ~ critically damped (unit inertia).
                    kp = float(cfg.joint_pos_kp)
                    arm0["use_impedances"] = False
                    arm0["isaac_kp"] = kp
                    arm0["isaac_kd"] = 2.0 * (kp ** 0.5)
                    print(f"  JointController position drive isaac_kp -> {kp}", flush=True)
            if override_cc:
                print(f"  Overriding controllers ({cfg.controller_preset or 'custom'}): "
                      f"{list(override_cc.keys())}", flush=True)
                robot.reload_controllers(override_cc)
        except Exception as e:
            print(f"  FAILED to load scene: {e}")
            all_results.append({
                "scene_name": scene_info["name"],
                "prompt": scene_info["prompt"],
                "status": "load_failed",
                "error": str(e),
            })
            continue

        action_space = robot.action_space
        print(f"  Action space dim: {action_space.shape[0]}", flush=True)
        print(f"  Action space low:  {np.array2string(np.asarray(action_space.low), precision=3)}", flush=True)
        print(f"  Action space high: {np.array2string(np.asarray(action_space.high), precision=3)}", flush=True)

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

        # Warm up by stepping a hold command, initializing the freshly-reloaded
        # controller's goal from the current pose and settling physics. For the
        # JointController (ik_eef_to_joint), "hold" is the current joint targets
        # (a zero action would command joints to 0 and fling the arm); a zero
        # eef-delta maps to current joints. For a delta controller, zeros hold.
        _hold_eef = np.zeros(7, dtype=np.float32)
        _hold_eef[-1] = 1.0  # gripper open during warmup
        for _ in range(10):
            if cfg.ik_eef_to_joint:
                wa = eef_delta_to_joint_action(robot, _hold_eef, action_space)
            else:
                wa = np.zeros(action_space.shape[0], dtype=np.float32)
            wa = np.clip(wa, action_space.low, action_space.high)
            env.step(torch.from_numpy(wa).unsqueeze(0))
        for _ in range(2):
            og.sim.render()

        obs = extract_obs(env, robot, scene_info["prompt"], cfg)
        frames = [obs["main_images"]] if cfg.save_video else []

        step_idx = 0
        done = False
        success = False
        total_reward = 0.0
        goal_detail = {}
        status = "completed"

        # A flailing policy can drive the arm into a PhysX blowup that
        # invalidates the articulation mid-rollout (get_joint_positions ->
        # None). Catch it so the partial video + result are still saved and
        # the batch moves on to the next scene instead of dying.
        try:
            while step_idx < cfg.max_steps and not done:
                chunk = query_policy(policy, obs, client_type, cfg)
                if os.environ.get("EVAL_DEBUG_IMG") and step_idx == 0:
                    for _nm, _im in obs.get("images_by_cam", {}).items():
                        imageio.imwrite(f"outputs/dbg_{_nm}.png", np.asarray(_im))
                    imageio.imwrite("outputs/dbg_wrist.png", np.asarray(obs["wrist_images"]))
                    print("[img] dumped policy input images to outputs/dbg_*.png", flush=True)
                if os.environ.get("EVAL_DEBUG_IO"):
                    with open(os.environ["EVAL_DEBUG_IO"], "a", encoding="utf-8") as _f:
                        _f.write(json.dumps({
                            "step": step_idx,
                            "state": np.asarray(obs["states"], dtype=np.float32).tolist(),
                            "act0": np.asarray(chunk[0], dtype=np.float32).tolist(),
                        }) + "\n")
                chunk_len = min(cfg.execute_horizon, len(chunk), cfg.max_steps - step_idx)

                for ci in range(chunk_len):
                    action = chunk[ci].copy()  # policy 7-D eef delta + gripper
                    if cfg.gripper_binarize:
                        action[-1] = np.sign(action[-1]) if abs(action[-1]) > 0.01 else -1.0
                    if cfg.ik_eef_to_joint:
                        # eef delta -> absolute joint targets for the JointController
                        ctrl_action = eef_delta_to_joint_action(robot, action, action_space)
                    else:
                        ctrl_action = action[:action_space.shape[0]]
                    action_clipped = np.clip(ctrl_action, action_space.low, action_space.high)

                    _eef_before = np.asarray(obs["states"][:3], dtype=np.float32)
                    _, reward, _, _, _ = env.step(
                        torch.from_numpy(action_clipped).unsqueeze(0)
                    )
                    obs = extract_obs(env, robot, scene_info["prompt"], cfg)
                    if os.environ.get("EVAL_DEBUG_STEP"):
                        _eef_after = np.asarray(obs["states"][:3], dtype=np.float32)
                        with open(os.environ["EVAL_DEBUG_STEP"], "a", encoding="utf-8") as _f:
                            _f.write(json.dumps({
                                "step": step_idx,
                                "raw_act": np.asarray(chunk[ci], dtype=np.float32).tolist(),
                                "clipped_act": np.asarray(action_clipped, dtype=np.float32).tolist(),
                                "eef_before": _eef_before.tolist(),
                                "eef_after": _eef_after.tolist(),
                            }) + "\n")
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
                    print(f"  Step {step_idx}/{cfg.max_steps} | success={success} | goals={goal_detail}", flush=True)
        except Exception as e:  # noqa: BLE001 - want the partial video regardless of cause
            status = "crashed"
            import traceback as _tb
            print(f"  ROLLOUT CRASHED at step {step_idx}: {type(e).__name__}: {e}", flush=True)
            print(_tb.format_exc(), flush=True)

        result = {
            "scene_name": scene_info["name"],
            "prompt": scene_info["prompt"],
            "target": scene_info["target_name"],
            "pipeline": scene_info.get("pipeline", ""),
            "rooms": scene_info["target_rooms"],
            "status": status,
            "steps": step_idx,
            "success": success,
            "goal_detail": goal_detail,
            "total_reward": total_reward,
        }
        all_results.append(result)
        print(f"  Result: success={success}, steps={step_idx}, status={status}", flush=True)

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
