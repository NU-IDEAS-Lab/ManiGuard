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

def _overview_cam_name(cfg: EvalConfig) -> str:
    """Physical external camera that supplies the policy's single overview,
    selected by cfg.external_cam (must match the checkpoint's train config)."""
    if cfg.external_cam not in ("left", "right"):
        raise ValueError(f"external_cam must be 'left' or 'right', got {cfg.external_cam!r}")
    return "cam_left" if cfg.external_cam == "left" else "cam_right"


def _build_eval_external_sensors(cfg: EvalConfig):
    """Build ONLY the one external overview camera the policy consumes
    (cam_left or cam_right per cfg.external_cam) — matching the 2-cam training
    convention. The wrist camera comes from the robot USD; no cam_opposite."""
    from maniguard.utils.camera_setup import build_external_camera_configs
    return build_external_camera_configs(
        names=[_overview_cam_name(cfg)], resolution=cfg.camera_resolution
    )


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
    raw_obs, _ = env.get_obs()
    external = raw_obs.get("external", {})
    # Single external overview = the one camera selected by cfg.external_cam
    # (cam_left or cam_right); it is the only external camera built/rendered.
    overview_name = _overview_cam_name(cfg)
    overview_rgb = external[overview_name]["rgb"][..., :3].cpu().numpy().astype(np.uint8)

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
        wrist_rgb = np.zeros_like(overview_rgb)

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
        "overview_image": overview_rgb,
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
    """Pack the 2-cam policy observation (LIBERO convention). The single external
    overview (cam_left or cam_right per cfg.external_cam) goes to the fixed key
    observation/image_left; the server (Sim2CamInputs) maps image_left->base_0,
    wrist->left_wrist_0, and zero-fills+masks the third slot. Matches every
    ManiGuard joint checkpoint's train config."""
    return {
        "observation/image_left": obs["overview_image"],
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
# LTL safety
# ---------------------------------------------------------------------------

_OBJECT_TAXONOMY = None


def _category_synset_lemma(category: str) -> str:
    """OmniGibson category -> its BDDL synset lemma (e.g. ``roasting_pan`` ->
    ``roaster``, ``milk_carton`` -> ``milk__carton``).

    LTL patterns name objects by synset lemma, which is NOT always the OG
    category; bridging via the object taxonomy lets those patterns resolve.
    Returns ``""`` if unavailable (taxonomy missing or category has no synset).
    """
    global _OBJECT_TAXONOMY
    if not category:
        return ""
    try:
        if _OBJECT_TAXONOMY is None:
            from bddl.object_taxonomy import ObjectTaxonomy
            _OBJECT_TAXONOMY = ObjectTaxonomy()
        syn = _OBJECT_TAXONOMY.get_synset_from_category(category)
        return syn.split(".n.")[0] if syn else ""
    except Exception:
        return ""


def _build_active_objects_for_ltl(env, ltl_safety, surface_name):
    """Reconstruct ``{inst_id: obj}`` so the diagnostics LTL patterns resolve to
    loaded scene objects.

    6fam-base scenes carry no ``inst_to_name`` / ``active_object_summary``, so
    the per-scene LTL spec (embedded in diagnostics) references objects only by
    glob pattern, e.g. ``teacup_*`` (category), ``roaster_*`` (synset lemma of a
    ``roasting_pan``), ``desk.n.01_*`` (synset), ``target_paper_towel_holder_*``
    (role+category). Map each pattern to the matching loaded objects under a key
    that fnmatches it:

      * ``agent.*``        -> the robot
      * else               -> objects whose category OR synset lemma == the
                              pattern prefix (synset base ``.split('.n.')[0]``
                              stripped), plus any whose name fnmatches the
                              pattern (role+category)
      * unresolved synset  -> the diagnostics ``surface`` object (support
                              backstop, e.g. ``breakfast_table.n.01_*`` filled by
                              an OG ``desk`` — a task role substitution the
                              taxonomy does not link)
    """
    import fnmatch

    patterns = set()
    for pdef in ((ltl_safety or {}).get("propositions") or {}).values():
        for key in ("over", "relative_to"):
            v = pdef.get(key)
            if isinstance(v, list):
                patterns.update(v)
            elif isinstance(v, str):
                patterns.add(v)

    robot = env.robots[0] if env.robots else None
    objs = list(env.scene.objects)
    # category -> synset lemma, so synset-lemma-named patterns resolve too.
    cat2lemma = {}
    for o in objs:
        c = getattr(o, "category", "")
        if c and c not in cat2lemma:
            cat2lemma[c] = _category_synset_lemma(c)
    surface_obj = (
        env.scene.object_registry("name", surface_name) if surface_name else None
    )

    active = {}
    for pat in patterns:
        prefix = pat[:-2] if pat.endswith("_*") else pat
        if prefix.startswith("agent"):
            if robot is not None:
                active[f"{prefix}_0"] = robot
            continue
        base = prefix.split(".n.")[0]
        matched = [
            o for o in objs
            if getattr(o, "category", "") == base
            or cat2lemma.get(getattr(o, "category", "")) == base
        ]
        matched += [
            o for o in objs
            if o not in matched and fnmatch.fnmatch(getattr(o, "name", ""), pat)
        ]
        if not matched and ".n." in prefix:
            # Synset pattern (e.g. lid.n.02_*) whose object is spawned under a
            # ROLE name (lid_cap_ep1_1, category 'cap') rather than the synset
            # category — match the synset lemma as a role-name prefix before
            # falling back to the support surface.
            role_matched = [
                o for o in objs if getattr(o, "name", "").startswith(base + "_")
            ]
            if role_matched:
                matched = role_matched
            elif surface_obj is not None:
                print(f"  [LTL] pattern {pat!r} unresolved by category/synset; "
                      f"using diagnostics surface {surface_name!r}")
                matched = [surface_obj]
        for i, obj in enumerate(matched):
            active[f"{prefix}_{i}"] = obj
    return active


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

    # Which metrics to evaluate (subset of {success, safety}). Gates which
    # checkers run, whether Spot is required, and what the summary reports.
    metrics = [m.lower() for m in (cfg.metrics or [])]
    if not metrics or any(m not in ("success", "safety") for m in metrics):
        raise ValueError(
            f"metrics must be a non-empty subset of ['success', 'safety']; "
            f"got {cfg.metrics!r}"
        )
    run_success = "success" in metrics
    run_safety = "safety" in metrics
    print(f"[Eval] metrics: success={run_success}, safety={run_safety}")

    # Each run gets its own subfolder under output_dir so successive runs never
    # overwrite each other. run_name is an explicit leaf when given (the batch
    # runner passes one shared name so all per-scene processes land together);
    # otherwise it is a timestamp, loud-suffixed with the tag for smoke/test
    # runs and uniquified on the (near-impossible) same-second collision.
    base_dir = Path(cfg.output_dir)
    run_name = (cfg.run_name or "").strip()
    if run_name:
        output_dir = base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if (cfg.tag or "").strip():
            stamp += "_" + cfg.tag.strip().upper()
        output_dir = base_dir / stamp
        n = 2
        while output_dir.exists():
            output_dir = base_dir / f"{stamp}_{n}"
            n += 1
        output_dir.mkdir(parents=True)
    print(f"[Eval] Run dir: {output_dir}")
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

    # When evaluating safety, LTL monitoring is mandatory whenever the benchmark
    # carries a spec — fail fast if the Spot runtime is missing/broken rather
    # than silently producing safety-free results. (Real Spot: conda-forge, NOT
    # pip.) Skipped entirely for a success-only run (no Spot dependency).
    if run_safety and any(s.get("ltl_safety") for s in scenes):
        from maniguard.utils.ltl_utils import (
            get_spot_runtime_status,
            spot_runtime_available,
        )
        if not spot_runtime_available(require_buddy=True):
            status = get_spot_runtime_status(require_buddy=True)
            raise RuntimeError(
                "LTL safety monitoring is required for this benchmark but the "
                f"Spot runtime is not functional: {status.get('error')}.\n"
                "Install the real Spot in this env:\n"
                "  conda install -c conda-forge spot   (do NOT 'pip install spot')"
            )
        print("LTL safety: Spot runtime OK — monitoring enabled")

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
            # Place external cameras at the SAME robot-frame poses used during
            # teleop collection + playback re-render (shared single source of
            # truth in camera_setup), so eval image_left / image_right match the
            # training views exactly. Replaces the old diagnostics-pose reader
            # (_setup_eval_cameras) — those stored poses were a different, stale
            # set and would put the policy out of distribution.
            from maniguard.utils.camera_setup import setup_external_cameras_robot_frame
            setup_external_cameras_robot_frame(env)

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

            # Force grasping semantics to match the training data's grasp mode.
            # The 6fam-base scene bakes its own grasping_mode (often 'assisted'),
            # but the policy was trained under the teleop mode (joint families =
            # 'sticky'); a mismatch means the learned gripper actions never grasp.
            # OmniGibson reads grasping_mode per step, so setting it now takes
            # effect immediately (the per-mode _ag_* state is created at init for
            # all modes, so switching post-load is safe).
            if cfg.grasping_mode not in ("physical", "assisted", "sticky"):
                raise ValueError(
                    f"grasping_mode must be physical/assisted/sticky, got "
                    f"{cfg.grasping_mode!r}"
                )
            if robot.grasping_mode != cfg.grasping_mode:
                print(f"  Grasping mode: {robot.grasping_mode} -> {cfg.grasping_mode}", flush=True)
                robot._grasping_mode = cfg.grasping_mode
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
        goal_checker = build_goal_checker(scene_info) if run_success else None
        if goal_checker is not None:
            goal_checker.resolve(env)
            if hasattr(goal_checker, "raw_region"):
                print(f"  Goal region: {goal_checker.raw_region.to_json()}")
            else:
                print(f"  Goals: {goal_checker.raw_conditions}")
        else:
            print(f"  Warning: no goal_region or goal_conditions in diagnostics — success will always be False")

        # Warm up by stepping a HOLD command, initializing the freshly-reloaded
        # controller's goal from the current pose and settling physics. The hold
        # must keep the arm where it is: for an absolute JointController
        # (ik_eef_to_joint=False) a ZERO action would command every joint to 0
        # and FLING the arm, so hold the CURRENT arm joints (action layout is
        # [arm_q(n), ..., gripper(last)]); the ik path maps a zero eef-delta to
        # the current joints. Gripper open throughout.
        _hold_eef = np.zeros(7, dtype=np.float32)
        _hold_eef[-1] = 1.0  # gripper open during warmup
        _hold_arm = robot.get_joint_positions()[
            robot.arm_control_idx[robot.default_arm]
        ].cpu().numpy().astype(np.float32)
        for _ in range(10):
            if cfg.ik_eef_to_joint:
                wa = eef_delta_to_joint_action(robot, _hold_eef, action_space)
            else:
                wa = np.zeros(action_space.shape[0], dtype=np.float32)
                wa[:_hold_arm.shape[0]] = _hold_arm
                wa[-1] = 1.0  # gripper open
            wa = np.clip(wa, action_space.low, action_space.high)
            env.step(torch.from_numpy(wa).unsqueeze(0))
        for _ in range(2):
            og.sim.render()

        obs = extract_obs(env, robot, scene_info["prompt"], cfg)
        # Two separate streams recorded as two mp4s: the policy's overview
        # (cam_left/right) and the wrist — same resolution the policy sees.
        main_frames = [obs["overview_image"]] if cfg.save_video else []
        wrist_frames = [obs["wrist_images"]] if cfg.save_video else []

        # LTL safety monitor — records throughout the rollout but NEVER ends it
        # (success / max_steps govern termination). scene_model=None: evaluate
        # exactly the task-level spec embedded in this scene's diagnostics.
        ltl_safety = scene_info.get("ltl_safety") or {}
        monitor = None
        if run_safety and ltl_safety:
            from maniguard.utils.safety_monitor import TaskLTLMonitor
            monitor = TaskLTLMonitor(
                env,
                ltl_safety=ltl_safety,
                activity_name=scene_info.get("activity_name", ""),
                scene_model=None,
                active_objects_by_inst=_build_active_objects_for_ltl(
                    env, ltl_safety, scene_info.get("surface_name"),
                ),
            )
            monitor.reset()
            monitor.step(0)

        step_idx = 0
        done = False
        success = False
        consec_success = 0          # consecutive instantaneous-goal steps
        success_step = None         # step the goal was CONFIRMED (held K steps)
        success_first_step = None   # step the goal first held instantaneously
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
                    imageio.imwrite("outputs/dbg_overview.png", np.asarray(obs["overview_image"]))
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
                        main_frames.append(obs["overview_image"])
                        wrist_frames.append(obs["wrist_images"])
                    step_idx += 1
                    total_reward += float(reward)

                    # Safety: advance the LTL monitor every executed step. It
                    # only records (no early-stop); a transient monitor hiccup
                    # must not abort an otherwise-fine rollout.
                    if monitor is not None:
                        try:
                            monitor.step(step_idx)
                        except Exception as _ltl_e:  # noqa: BLE001
                            print(f"  [LTL] monitor.step failed at {step_idx}: {_ltl_e}")

                    if goal_checker is not None:
                        inst_success, goal_detail = goal_checker.check(env)
                        if inst_success:
                            if success_first_step is None:
                                success_first_step = step_idx
                            consec_success += 1
                        else:
                            consec_success = 0
                        # Confirm success only when the goal holds for
                        # success_hold_steps consecutive steps — a single-frame
                        # brush / AG-grasp flicker / pass-through must not count
                        # (DEV_LOG 2026-05-09 false positive).
                        if consec_success >= cfg.success_hold_steps:
                            success = True
                            success_step = step_idx
                            done = True
                            break

                if step_idx % 50 == 0 or step_idx == 1:
                    print(f"  Step {step_idx}/{cfg.max_steps} | success={success} | goals={goal_detail}", flush=True)
        except Exception as e:  # noqa: BLE001 - want the partial video regardless of cause
            status = "crashed"
            import traceback as _tb
            print(f"  ROLLOUT CRASHED at step {step_idx}: {type(e).__name__}: {e}", flush=True)
            print(_tb.format_exc(), flush=True)

        ltl_summary = monitor.summary() if monitor is not None else None
        result = {
            "scene_name": scene_info["name"],
            "prompt": scene_info["prompt"],
            "target": scene_info["target_name"],
            "pipeline": scene_info.get("pipeline", ""),
            "rooms": scene_info["target_rooms"],
            "status": status,
            "steps": step_idx,
            "metrics": metrics,
            "success": success if run_success else None,
            "success_step": success_step,
            "success_first_step": success_first_step,
            "goal_detail": goal_detail,
            "total_reward": total_reward,
            "ltl_monitored": monitor is not None,
            "ltl_violated": (monitor.violated if monitor is not None else None),
            "ltl_violation_step": (monitor.violation_step if monitor is not None else None),
            "ltl_violation_count": (monitor.violation_count if monitor is not None else 0),
            "ltl_formula": (ltl_summary.get("formula", "") if ltl_summary else ""),
        }
        all_results.append(result)
        _ltl_str = "" if monitor is None else f", ltl_violated={monitor.violated}"
        print(f"  Result: success={success}, steps={step_idx}, status={status}{_ltl_str}", flush=True)

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")

        if cfg.save_video and main_frames:
            # Two separate mp4s — the policy's overview stream and the wrist
            # stream — at the training data's 30 fps, same 256-res the policy
            # sees (no upscaling: keeps the saved video faithful to the input).
            (output_dir / scene_info["name"]).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(output_dir / f"{scene_info['name']}_main.mp4"), main_frames, fps=30)
            imageio.mimsave(str(output_dir / f"{scene_info['name']}_wrist.mp4"), wrist_frames, fps=30)

        # Full per-step LTL log to a sidecar (kept out of the main results to
        # avoid bloating them with thousands of per-step AP dicts).
        if ltl_summary is not None:
            (output_dir / scene_info["name"]).parent.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{scene_info['name']}_ltl.json").write_text(
                json.dumps(ltl_summary, ensure_ascii=True, indent=2), encoding="utf-8",
            )

    # Summary
    print(f"\n{'='*60}")
    print("Benchmark Evaluation Summary")
    print(f"{'='*60}")
    completed = [r for r in all_results if r["status"] == "completed"]
    n_total = len(completed)
    n_failed_load = sum(1 for r in all_results if r["status"] == "load_failed")
    print(f"Scenes evaluated: {n_total} ({n_failed_load} failed to load)")
    if run_success:
        n_success = sum(1 for r in completed if r.get("success"))
        print(f"Success rate: {n_success}/{n_total} ({n_success/max(n_total,1)*100:.1f}%)")
    else:
        n_success = None
    if n_total > 0:
        print(f"Avg steps: {np.mean([r['steps'] for r in completed]):.1f}")
    n_ltl = sum(1 for r in completed if r.get("ltl_monitored"))
    n_violated = sum(1 for r in completed if r.get("ltl_violated"))
    if n_ltl:
        print(f"Safety (LTL): {n_violated}/{n_ltl} scenes had a violation")
    print(f"Results: {results_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "metrics": metrics,
        "n_scenes": n_total,
        "n_success": n_success,
        "n_failed_load": n_failed_load,
        "success_rate": (n_success / max(n_total, 1)) if run_success else None,
        "n_ltl_monitored": n_ltl,
        "n_ltl_violated": n_violated,
        "results": all_results,
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
