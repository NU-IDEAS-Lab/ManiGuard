"""Pick-and-place pipeline driven by a replay-empty task dump.

Given a single task folder containing ``scene_ep<N>.json`` +
``diagnostics.jsonl`` (the same format that
``maniguard/data/curobo/replay_empty_from_dataset.py`` consumes), this script:

  1. Rebuilds the task in a bare OmniGibson Scene: floor + fixed support
     + every spawn-spec object (target / fragile / clutter) + the Franka
     at its dump pose + the green ``goal_region`` marker + diagnostics
     cameras.
  2. Runs the OBB sampler on the target object and validates candidate
     grasps with cuRobo + AG until the first one holds (Phase A).
  3. Plans a second cuRobo trajectory from the holding pose to a goal
     end-effector pose that places the target inside the goal sphere,
     with the target attached to the gripper via cuRobo's
     ``attached_obj`` argument so the planner avoids the fragile /
     clutter objects with the carried geometry too (Phase B).
  4. Replays the transport trajectory and writes 4-camera MP4s + a
     ``result.json`` (success flag, final target → goal distance) +
     ``trajectory.pt`` (stitched arm waypoints) into
     ``<task-dir>/pick_and_place/``.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m tools.pick_and_place_from_dataset \\
            --task-dir datasets/7fam-30tasks-base-final-20260512/table/task_0000 \\
            --save-video
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-dir", type=Path, required=True,
                   help="Task folder (must contain scene_ep<N>.json + diagnostics.jsonl)")
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <task-dir>/pick_and_place)")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--gui", action="store_true")
    # Phase A (pick)
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick-timeout", type=float, default=180.0)
    p.add_argument("--max-reach", type=float, default=0.95)
    p.add_argument("--max-obj-to-eef-after-hold", type=float, default=0.20)
    # Phase B (transport)
    p.add_argument("--transport-timeout", type=float, default=60.0,
                   help="Wall-clock budget for the transport-plan cuRobo call.")
    p.add_argument("--transport-clear-z", type=float, default=0.05,
                   help="Lift the target this far above its current z before "
                        "translating to the goal xy. Avoids dragging the held "
                        "target across the support surface.")
    p.add_argument("--goal-radius-margin", type=float, default=None,
                   help="Treat transport as success if target center lands "
                        "within (goal_radius + margin) of goal_region center. "
                        "Default: target_width_m/2 from the diagnostics payload "
                        "(matches the 'held_intersection' semantics).")
    p.add_argument("--ik-precheck", action="store_true",
                   help="Enable Phase A ik_only precheck (skip unreachable "
                        "candidates before running Stage-1 trajopt).")
    p.add_argument("--no-distractors", action="store_true",
                   help="Drop fragile + clutter objects from the scene, "
                        "keeping only the support surface, target, and goal "
                        "marker. Useful for isolating whether tracking "
                        "issues are caused by the planner routing the arm "
                        "around obstacles vs the controller itself.")
    # SFT recording
    p.add_argument("--record-sft", action="store_true",
                   help="Capture (image_left, image_right, wrist_image, state, "
                        "action) per env.step during Phase B and emit "
                        "rollout.hdf5 + 3 review MP4s into --out-dir on "
                        "success. Skips writing on miss/fail.")
    p.add_argument("--record-resolution", type=int, default=256,
                   help="Square frame side length for the three SFT cameras.")
    # Transport variants — reuse one successful Phase A held grasp to
    # generate multiple transport trajectories with randomized lift/hover.
    p.add_argument("--n-transport-variants", type=int, default=1,
                   help="If >1, after Phase A succeeds, replay the held "
                        "grasp + plan/execute Phase B N times with each "
                        "variant drawing a random (lift_z, hover_z) from "
                        "the configured ranges. Variants land in "
                        "<out-dir>/variant_XX/. Default 1 preserves the "
                        "single-trajectory legacy behavior with hardcoded "
                        "0.25 / 0.25.")
    p.add_argument("--lift-z-min", type=float, default=0.08,
                   help="Minimum lift height (m) when sampling variants.")
    p.add_argument("--lift-z-max", type=float, default=0.15,
                   help="Maximum lift height (m) when sampling variants.")
    p.add_argument("--hover-z-min", type=float, default=0.08,
                   help="Minimum hover-above-target height (m) when sampling "
                        "variants.")
    p.add_argument("--phase-a-grasp-from-dataset", default=None,
                   help="Path to a previously collected SFT dataset root "
                        "(e.g. outputs/sft_dataset_2026-05-16/success_balanced). "
                        "If a matching task_<NNNN>__seed_* rollout.hdf5 is "
                        "found, its grasp pose is used as a single OBB "
                        "candidate, skipping the ~60s OBB sampler + multi-"
                        "candidate cuRobo loop. Falls back to OBB sampling "
                        "if the candidate doesn't hold.")
    p.add_argument("--phase-a-grasp-index", type=int, default=0,
                   help="When --phase-a-grasp-from-dataset is set and the "
                        "directory contains multiple task_<NNNN>__seed_* "
                        "rollouts for this task, pick the N-th one (0-based, "
                        "sorted by path). Lets a single sweep collect "
                        "multiple distinct grasps per task by re-running "
                        "with different indices.")
    p.add_argument("--lerobot-repo-id", default=None,
                   help="If set, write each successful variant as a LeRobot "
                        "v2.1 episode at --lerobot-root (or default cache). "
                        "Reuses the SFT recorder's MP4s in-place; no re-encode.")
    p.add_argument("--lerobot-root", default=None,
                   help="Root for the LeRobot dataset folder. "
                        "Required when --lerobot-repo-id is set (otherwise "
                        "defaults to HF cache, which is rarely what we want).")
    p.add_argument("--lerobot-prompt-template",
                   default="pick up the {target_clean} in the middle of the "
                           "table and place it at the green goal",
                   help="Substitutions: {target} (raw BDDL instance like "
                        "'teacup_178'), {target_clean} (suffix stripped, "
                        "underscores -> spaces).")
    p.add_argument("--hover-z-max", type=float, default=0.15,
                   help="Maximum hover-above-target height (m) when sampling "
                        "variants.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env build — lifted from maniguard/data/curobo/replay_empty_from_dataset.py
# ---------------------------------------------------------------------------


def _init_omnigibson(headless: bool):
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if headless:
        gm.HEADLESS = True
    import omnigibson as og
    return og


def _build_env(task_dir: Path, episode: int, *, no_distractors: bool = False,
               record_sft: bool = False, record_resolution: int = 256,
               grasping_mode: str = "assisted"):
    """Build the empty-scene env from the task dump and return
    (env, og, diagnostics, camera_names).

    With ``no_distractors=True``, drop every task object except the
    target itself (identified by ``diagnostics.goal_region.target_name``).
    Surface stays. Useful for isolating planner/controller behaviour.

    With ``record_sft=True``, override external_sensors so cam_left +
    cam_right are at ``record_resolution`` (square, RGB only) and inject
    a wrist Camera prim under ``panda_hand`` via the FrankaPanda patch.
    """
    import omnigibson as og  # noqa: F401  -- imported below explicitly
    from tools.replay_empty_from_dataset import (
        _build_object_cfg,
        _identify_task_objects,
        _load_diagnostics_row,
        _load_scene_info,
    )
    from maniguard.envs.frozen_task_runtime import (
        build_env_config,
        configure_review_sensors,
        extract_scene_robot_setup,
        position_diagnostics_cameras,
    )
    from maniguard.utils.goal_region import GoalRegionSpec, spawn_goal_region_marker

    diagnostics = _load_diagnostics_row(task_dir, episode)
    scene_info = _load_scene_info(task_dir, episode)
    task_names = _identify_task_objects(scene_info, diagnostics)
    print(f"[PnP] {task_dir.name}: {len(task_names)} task objects "
          f"(surface={task_names[0]})", flush=True)

    object_cfgs = [_build_object_cfg(task_names[0], scene_info, fixed_base=True)]
    if no_distractors:
        target_name = str(diagnostics["goal_region"]["target_name"])
        kept = [n for n in task_names[1:] if n == target_name]
        dropped = [n for n in task_names[1:] if n != target_name]
        print(f"[PnP] --no-distractors: keeping target={target_name!r}, "
              f"dropping {len(dropped)} distractor(s): {dropped}", flush=True)
        object_cfgs += [_build_object_cfg(n, scene_info, fixed_base=False)
                        for n in kept]
    else:
        object_cfgs += [_build_object_cfg(n, scene_info, fixed_base=False)
                        for n in task_names[1:]]

    robot_setup = extract_scene_robot_setup(scene_info)
    if robot_setup is None:
        raise RuntimeError("No robot found in scene snapshot")

    camera_names = [c["sensor_name"] for c in diagnostics.get("cameras", [])
                    if c.get("sensor_name")]
    if record_sft:
        # SFT pipeline only needs cam_left + cam_right (overview pair) at a
        # square resolution. The wrist prim is injected by the FrankaPanda
        # patch in _sft_recorder.install_wrist_camera_patch().
        camera_names = [n for n in ("cam_left", "cam_right") if n in camera_names]
        if not camera_names:
            raise RuntimeError(
                "--record-sft needs cam_left/cam_right in diagnostics")
        external_camera_kwargs = {
            "resolution": record_resolution,
            "modalities": ("rgb",),
        }
    else:
        external_camera_kwargs = {
            "resolution": (720, 1280),
            "modalities": ("rgb",),
        }

    # Centralized config build:
    #   - JointController with impedance — Phase A approach via
    #     dense JointController tracking (no teleport, gravity on);
    #     Phase B transport feeds the cuRobo joint trajectory directly.
    #   - grasping_mode override per caller (assisted default; lid uses sticky).
    #   - action_normalize=False locked.
    env_cfg = build_env_config(
        scene_info, diagnostics,
        camera_names=camera_names,
        controller_preset="joint_position_impedance",
        grasping_mode=grasping_mode,
        external_camera_kwargs=external_camera_kwargs,
        action_frequency=30, rendering_frequency=30,
    )
    # Override the empty scene (build_env_config defaults to whatever the
    # snapshot says; pnp wants a plain Scene with explicit object cfgs).
    env_cfg["scene"] = {"type": "Scene"}
    env_cfg["objects"] = object_cfgs
    # Robot wrist VisionSensor resolution for SFT capture.
    if record_sft and env_cfg.get("robots"):
        env_cfg["robots"][0]["sensor_config"] = {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": record_resolution,
                    "image_width": record_resolution,
                },
            },
        }

    import omnigibson as og
    env = og.Environment(configs=env_cfg)
    env.reset()

    # Re-apply object poses (env.reset can perturb).
    for cfg in object_cfgs:
        obj = env.scene.object_registry("name", cfg["name"])
        if obj is None:
            continue
        import torch as th
        obj.set_position_orientation(
            position=th.tensor(cfg["position"], dtype=th.float32),
            orientation=th.tensor(cfg["orientation"], dtype=th.float32),
        )
        if hasattr(obj, "keep_still"):
            obj.keep_still()

    robot = env.robots[0]
    if robot_setup.get("position") is not None:
        import torch as th
        robot.set_position_orientation(
            position=th.tensor(robot_setup["position"], dtype=th.float32),
            orientation=th.tensor(robot_setup["orientation"], dtype=th.float32),
        )
    if hasattr(robot, "keep_still"):
        robot.keep_still()
    og.sim.step()

    gr_payload = diagnostics.get("goal_region")
    if gr_payload is not None:
        spawn_goal_region_marker(env, GoalRegionSpec.from_json(gr_payload))
        og.sim.step()

    configure_review_sensors(env)
    position_diagnostics_cameras(env, og, diagnostics, set_viewer=True)
    if record_sft:
        # Kit viewport init can ignore sensor_kwargs — force the resolution
        # post-init on every VisionSensor (externals + wrist).
        from omnigibson.sensors import VisionSensor
        for cam in (env.external_sensors or {}).values():
            cam.image_height = record_resolution
            cam.image_width = record_resolution
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                sensor.image_height = record_resolution
                sensor.image_width = record_resolution
        env.load_observation_space()
    for _ in range(10):
        og.sim.step()

    return env, og, diagnostics, camera_names


# ---------------------------------------------------------------------------
# Phase A — grasp the target
# ---------------------------------------------------------------------------


def _gather_obstacle_surf_local(env, target_obj,
                                points_per_obstacle: int = 1500,
                                rng_seed: int = 0) -> np.ndarray:
    """Sample obstacle surface points (every scene object except the
    target and the robot) in the TARGET's local frame.

    Used by the OBB sampler's obstacle-aware empty-box rejection. Each
    obstacle's visual mesh is sampled, transformed world → target-local,
    and concatenated. Returns an empty (0, 3) array if no obstacles
    have an extractable mesh.
    """
    from maniguard.rl.grasps.collector import _pose_to_mat
    from maniguard.rl.grasps.mesh import mesh_from_og_object

    # Target world pose → inverse transform to take world points into
    # target-local frame.
    t_pos, t_quat = target_obj.get_position_orientation()
    t_pos_np = t_pos.cpu().numpy().astype(np.float64)
    t_quat_np = t_quat.cpu().numpy().astype(np.float64)
    T_tgt_w = _pose_to_mat(t_pos_np, t_quat_np)
    T_w_to_local = np.linalg.inv(T_tgt_w)

    rng = np.random.default_rng(rng_seed)
    all_pts: list[np.ndarray] = []
    target_name = target_obj.name
    # Robot.name vs robot.prim_path — match either; objects are uniquely
    # named so identity comparison is enough.
    robot_objs = set(env.robots) if env.robots else set()

    for obj in env.scene.objects:
        if obj is target_obj or obj in robot_objs:
            continue
        # Skip goal-region markers (visual-only, no real geometry).
        name = getattr(obj, "name", "") or ""
        if "goal_region" in name:
            continue
        try:
            obs_mesh = mesh_from_og_object(obj, use_visual=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[PnP] obstacle-mesh skip {obj.name!r}: {exc}", flush=True)
            continue
        if obs_mesh is None or len(obs_mesh.vertices) == 0:
            continue
        # Sample surface in obstacle-local frame.
        import trimesh
        try:
            n_pts = min(points_per_obstacle,
                        max(200, int(obs_mesh.area * 5000)))
            pts_local, _ = trimesh.sample.sample_surface(
                obs_mesh, n_pts,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[PnP] obstacle-sample skip {obj.name!r}: {exc}", flush=True)
            continue
        pts_local = np.asarray(pts_local, dtype=np.float64)
        # Obstacle-local → world via obstacle's world pose.
        o_pos, o_quat = obj.get_position_orientation()
        o_pos_np = o_pos.cpu().numpy().astype(np.float64)
        o_quat_np = o_quat.cpu().numpy().astype(np.float64)
        T_obs_w = _pose_to_mat(o_pos_np, o_quat_np)
        pts_world = (T_obs_w[:3, :3] @ pts_local.T).T + T_obs_w[:3, 3]
        # World → target-local.
        pts_tgt_local = (T_w_to_local[:3, :3] @ pts_world.T).T \
            + T_w_to_local[:3, 3]
        all_pts.append(pts_tgt_local.astype(np.float32))

    if not all_pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(all_pts, axis=0)


def _grasp_candidate_from_sft(sft_root, task_dir, env, target_obj,
                              grasp_index: int = 0):
    """Load a known-good eef pose from a previous successful HDF5 and
    convert it to a target-local 4x4 matrix usable as an OBB candidate.

    Returns the (4, 4) ``T_local`` matrix, or ``None`` if no matching
    rollout was found in ``sft_root``.

    The SFT dataset records per-step state in base frame (8D: eef_pos(3) +
    eef_aa(3) + gripper_q(2)). We pick the step where the gripper first
    crosses below a threshold ≈ 60% closed — that's the grasp-engagement
    moment. Eef pose at that step → world → target-local.
    """
    import glob
    from pathlib import Path

    import h5py
    import numpy as np
    import torch as th
    import omnigibson.utils.transform_utils as T
    from maniguard.rl.grasps.collector import _pose_to_mat

    task_name = Path(task_dir).parent.name  # 'task_NNNN'
    pattern = str(Path(sft_root) / f"{task_name}__seed_*" / "rollout.hdf5")
    paths = sorted(glob.glob(pattern))
    if not paths:
        # Fall back to recursive search for nested layouts.
        paths = sorted(
            str(p) for p in Path(sft_root).rglob(f"{task_name}__seed_*/rollout.hdf5")
        )
    if not paths:
        print(f"[PnP] no SFT rollout found for {task_name} in {sft_root}",
              flush=True)
        return None

    grasp_idx = int(grasp_index)
    if grasp_idx >= len(paths):
        print(f"[PnP] phase-a-grasp-index={grasp_idx} out of range "
              f"({len(paths)} priors available for {task_name}); "
              f"using last available", flush=True)
        grasp_idx = len(paths) - 1
    hdf5_path = paths[grasp_idx]
    print(f"[PnP] phase-a-grasp-index={grasp_idx}/{len(paths)-1}: "
          f"using {Path(hdf5_path).parent.name}", flush=True)
    with h5py.File(hdf5_path, "r") as f:
        if not bool(f.attrs.get("phase_a_held", False)):
            print(f"[PnP] SFT rollout {hdf5_path} did not hold; skipping",
                  flush=True)
            return None
        state = np.asarray(f["data/demo_0/obs/state"])  # (N, 8) float32

    grip = state[:, 6]
    threshold = float(grip.min()) + 0.4 * (float(grip.max()) - float(grip.min()))
    closing = np.where(grip < threshold)[0]
    if len(closing) == 0:
        print(f"[PnP] SFT rollout {hdf5_path} has no gripper transition",
              flush=True)
        return None
    # Step a bit past the first crossing so the gripper has stabilised on
    # the target instead of being mid-close.
    i_grasp = min(int(closing[0]) + 10, len(state) - 1)
    eef_pos_base = state[i_grasp, :3].astype(np.float64)
    eef_aa_base = state[i_grasp, 3:6].astype(np.float64)
    eef_quat_base = T.axisangle2quat(
        th.tensor(eef_aa_base, dtype=th.float32)
    ).numpy().astype(np.float64)  # xyzw

    # base → world via the current robot base pose.
    robot = env.robots[0]
    base_pos_w, base_quat_w = robot.get_position_orientation()
    T_base_world = _pose_to_mat(
        base_pos_w.cpu().numpy().astype(np.float64),
        base_quat_w.cpu().numpy().astype(np.float64),
    )
    T_eef_base = _pose_to_mat(eef_pos_base, eef_quat_base)
    T_eef_world = T_base_world @ T_eef_base

    # world → target-local via the current target pose.
    tgt_pos_w, tgt_quat_w = target_obj.get_position_orientation()
    T_target_world = _pose_to_mat(
        tgt_pos_w.cpu().numpy().astype(np.float64),
        tgt_quat_w.cpu().numpy().astype(np.float64),
    )
    T_local = np.linalg.inv(T_target_world) @ T_eef_world

    print(f"[PnP] Phase A: grasp candidate loaded from "
          f"{Path(hdf5_path).parent.name}  "
          f"(grasp_step={i_grasp}/{len(state)})", flush=True)
    print(f"[PnP]   eef_pos_target_local={T_local[:3, 3].tolist()}",
          flush=True)
    return T_local.astype(np.float64)


def _phase_a_pick(env, og, primitives, target_obj, args, deadline: float):
    """Run OBB sampler + collect_valid_grasps; return all held grasps.

    With ``--phase-a-grasp-from-dataset`` set, the OBB sampler is replaced
    by a single candidate loaded from a previous successful rollout —
    skipping ~60s of OBB enumeration when a known-good grasp exists.

    Returns ``(held_grasps, phase_a_timings)`` where ``held_grasps`` is a
    list of dicts (each with keys ``approach_traj`` (T, 7 torch tensor),
    ``rel_position``, ``rel_orientation_xyzw``, ``gripper_qpos``,
    ``arm_joint_pos``). May be empty if no candidate held. The number of
    held candidates is capped at ``args.max_held_candidates`` when that
    attr is present (default 1 for legacy callers).
    """
    import torch as th

    from maniguard.rl.grasps.collector import (
        GraspCollectorConfig,
        collect_valid_grasps,
    )
    from maniguard.rl.grasps.mesh import mesh_from_og_object
    from maniguard.rl.grasps.obb_sampler import (
        OBBConfig, sample_obb_assisted_grasps,
    )

    robot = env.robots[0]
    init_pos, init_quat = target_obj.get_position_orientation()
    init_pos = init_pos.clone()
    init_quat = init_quat.clone()

    # SFT-prior grasp candidate (single OBB-equivalent target) shortcut.
    sft_grasp_root = getattr(args, "phase_a_grasp_from_dataset", None)
    if sft_grasp_root:
        T_local = _grasp_candidate_from_sft(
            sft_grasp_root, args.task_dir, env, target_obj,
            grasp_index=int(getattr(args, "phase_a_grasp_index", 0)),
        )
        if T_local is not None:
            candidates = np.asarray([T_local], dtype=np.float64)
            primitives._motion_generator.update_obstacles(ignore_objects=[])
            cfg = GraspCollectorConfig(
                num_target_grasps=int(getattr(args, "max_held_candidates", 1)),
                max_reach=args.max_reach,
                max_obj_to_eef_after_hold=args.max_obj_to_eef_after_hold,
                ik_precheck=bool(getattr(args, "ik_precheck", False)),
            )
            held_list = []

            def _on_progress(ci, result):
                if result is not None:
                    held_list.append(result)
                    print(f"[PnP] Phase A cand {ci} (SFT-prior): HELD "
                          f"(obj_to_eef={result['obj_to_eef']:.3f}m, "
                          f"traj_len={len(result['approach_traj'])})",
                          flush=True)

            timings_log: list = []
            held = collect_valid_grasps(
                env, robot, primitives, target_obj,
                init_pos, init_quat,
                candidates_local=candidates,
                cfg=cfg, deadline=deadline,
                on_progress=_on_progress,
                timings_log=timings_log,
            )
            if held:
                return held, timings_log
            print("[PnP] SFT-prior candidate did NOT hold; falling back "
                  "to OBB sampling", flush=True)

    try:
        mesh = mesh_from_og_object(target_obj, use_visual=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[PnP] mesh extraction failed: {exc}", flush=True)
        return None

    # Gather obstacle surface points (container, support, food, etc.) in
    # the TARGET-local frame, so the OBB sampler can reject candidates
    # whose gripper boxes would intersect non-target geometry. Without
    # this, cuRobo's collision-aware trajopt has to filter those, which
    # is much slower (and AG-engage on the dropped candidates also slows
    # things down further).
    obstacle_surf_local = _gather_obstacle_surf_local(env, target_obj)

    rng = np.random.default_rng(args.seed)
    candidates, scores = sample_obb_assisted_grasps(
        mesh,
        config=OBBConfig(max_candidates=args.max_candidates),
        rng=rng,
        obstacle_surf_local=obstacle_surf_local,
    )
    if len(candidates) == 0:
        print("[PnP] OBB sampler returned 0 candidates", flush=True)
        return None
    print(f"[PnP] OBB candidates: {len(candidates)}", flush=True)

    # Include all scene objects (target + fragile + clutter + support) in
    # cuRobo's obstacle world — the planner must avoid the fragiles during
    # approach.
    primitives._motion_generator.update_obstacles(ignore_objects=[])

    cfg = GraspCollectorConfig(
        num_target_grasps=int(getattr(args, "max_held_candidates", 1)),
        max_reach=args.max_reach,
        max_obj_to_eef_after_hold=args.max_obj_to_eef_after_hold,
        ik_precheck=bool(getattr(args, "ik_precheck", False)),
    )

    held_list = []

    def _on_progress(ci, result):
        if result is not None:
            held_list.append(result)
            print(f"[PnP] Phase A cand {ci}: HELD "
                  f"(obj_to_eef={result['obj_to_eef']:.3f}m, "
                  f"traj_len={len(result['approach_traj'])})", flush=True)

    timings_log: list = []
    held = collect_valid_grasps(
        env, robot, primitives, target_obj,
        init_pos, init_quat,
        candidates_local=candidates,
        cfg=cfg,
        deadline=deadline,
        on_progress=_on_progress,
        verbose=True,
        timings_log=timings_log,
    )
    return (held if held else []), timings_log


# ---------------------------------------------------------------------------
# Phase B — plan + replay transport with attached_obj
# ---------------------------------------------------------------------------


def _solve_one_segment(motion_gen, robot, target_obj, eef_link, eef_goal_pos,
                       eef_goal_quat, initial_joint_pos, *, timeout: float,
                       max_attempts: int = 12, label: str = "",
                       use_attached_obj: bool = True):
    """Single-call wrapper around motion_gen.compute_trajectories with the
    target optionally attached. Returns ((T, 7) tensor, final-full-DoF
    joint state) or (None, None) on failure.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_goal_pos] * bs)}
    target_quat = {eef_link: th.stack([eef_goal_quat] * bs)}

    attached_obj = (
        {eef_link: target_obj.root_link} if use_attached_obj else None
    )
    attached_obj_scale = (
        {eef_link: 1.0} if use_attached_obj else None
    )

    # Use return_full_result=True so we can salvage trajectories that
    # cuRobo's trajopt marks as `success=False` but actually converged
    # exactly at the goal (pos_err≈0, rot_err≈0). This case appears in
    # short/degenerate motions where trajopt's convergence criterion is
    # overly strict — `result.interpolated_plan` is populated and lands
    # at the goal, but the success flag is False. Salvage tolerance is
    # tight (5 mm / 0.03 rad) so we only accept truly near-zero-error
    # plans — anything looser would risk dataset quality.
    full = motion_gen.compute_trajectories(
        target_pos=target_pos, target_quat=target_quat,
        initial_joint_pos=initial_joint_pos, is_local=False,
        max_attempts=max_attempts, timeout=timeout,
        ik_fail_return=5, enable_finetune_trajopt=True,
        finetune_attempts=2, return_full_result=True,
        success_ratio=1.0 / bs,
        attached_obj=attached_obj,
        attached_obj_scale=attached_obj_scale,
        motion_constraint=None, skip_obstacle_update=True,
        ik_only=False, ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )

    def _flt_at(e, j):
        if e is None:
            return None
        try:
            return float(e[j])
        except Exception:
            try:
                return float(e)
            except Exception:
                return None

    POS_TOL = 0.005  # 5 mm — tight for production data quality
    ROT_TOL = 0.03   # ~1.7° rotation error

    joint_state = None
    for i, r in enumerate(full):
        try:
            paths = r.get_paths()
        except Exception:
            paths = []
        succ_t = getattr(r, "success", None)
        for j in range(len(paths)):
            is_succ = False
            if succ_t is not None:
                try:
                    is_succ = bool(succ_t[j].item())
                except Exception:
                    try:
                        is_succ = bool(succ_t[j])
                    except Exception:
                        is_succ = False
            pos_err = _flt_at(getattr(r, "position_error", None), j)
            rot_err = _flt_at(getattr(r, "rotation_error", None), j)
            path_js = paths[j]
            if path_js is None:
                continue
            accepted = is_succ or (
                (pos_err is None or pos_err < POS_TOL)
                and (rot_err is None or rot_err < ROT_TOL)
            )
            if accepted:
                if not is_succ:
                    print(f"[PnP transport] segment {label!r}: salvage "
                          f"iter#{i} batch#{j} (status="
                          f"{getattr(r, 'status', '?')!r}, "
                          f"pos_err={pos_err}, rot_err={rot_err})",
                          flush=True)
                joint_state = path_js
                break
        if joint_state is not None:
            break

    if joint_state is None:
        print(f"[PnP transport] segment {label!r}: cuRobo FAILED "
              f"(0/{int(bs)} successes)", flush=True)
        # Diagnostic probe: re-run with return_full_result=True so we
        # can read the per-batch MotionGenStatus and pos/rot errors.
        # Cheap on already-warmed kernels. No behavior change — we
        # still return None — but the print tells us whether the fail
        # mode is IK Fail (truly unreachable), Trajopt Fail (motion
        # planning gave up but IK landed close), Start State In
        # Collision (held object intersects environment), etc.
        try:
            full = motion_gen.compute_trajectories(
                target_pos=target_pos, target_quat=target_quat,
                initial_joint_pos=initial_joint_pos, is_local=False,
                max_attempts=1, timeout=timeout,
                ik_fail_return=5, enable_finetune_trajopt=False,
                finetune_attempts=1, return_full_result=True,
                success_ratio=1.0 / bs,
                attached_obj=attached_obj,
                attached_obj_scale=attached_obj_scale,
                motion_constraint=None, skip_obstacle_update=True,
                ik_only=False, ik_world_collision_check=True,
                emb_sel=CuRoboEmbodimentSelection.DEFAULT,
            )
            def _fmt(e):
                if e is None:
                    return "?"
                try:
                    return f"{float(e):.4f}"
                except Exception:
                    try:
                        return f"{float(e.min().item()):.4f}"
                    except Exception:
                        return repr(e)
            for i, r in enumerate(full):
                status = getattr(r, "status", "?")
                pos_err = getattr(r, "position_error", None)
                rot_err = getattr(r, "rotation_error", None)
                print(f"[PnP transport]   diag #{i}: status={status!r} "
                      f"pos_err={_fmt(pos_err)} rot_err={_fmt(rot_err)}",
                      flush=True)
        except Exception as exc:
            print(f"[PnP transport]   diag probe raised "
                  f"{type(exc).__name__}: {exc}", flush=True)
        return None, None, None
    # joint_state was set in the salvage loop above.
    manip_idx = robot.arm_control_idx[robot.default_arm]
    joint_pos = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    if joint_pos.dim() == 1:
        arm_traj = joint_pos[manip_idx].unsqueeze(0).cpu()
    else:
        arm_traj = joint_pos[:, manip_idx].cpu()
    # Capture full-DoF final state for chaining the next segment.
    full_traj = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    final_full = full_traj if full_traj.dim() == 1 else full_traj[-1]
    # Also compute the eef trajectory in robot base frame for OSC replay.
    eef_pose_dict = motion_gen.path_to_eef_trajectory(
        joint_state, return_axisangle=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    # eef_pose_dict: { eef_link_name: (T, 7) tensor of [pos(3), quat_xyzw(4)] }
    eef_traj = eef_pose_dict[eef_link].cpu()
    print(f"[PnP transport] segment {label!r}: ok  "
          f"({len(arm_traj)} arm waypoints, eef shape={tuple(eef_traj.shape)})",
          flush=True)
    return arm_traj, final_full, eef_traj


def _plan_transport(primitives, robot, target_obj, goal_target_pos_world,
                    *, transport_timeout: float, lift_z: float = 0.25,
                    hover_z: float = 0.25,
                    goal_target_quat_world=None,
                    skip_descend: bool = False,
                    skip_rotate: bool = False,
                    seg_timings: list | None = None):
    """Two-stage transport plan: lift (vertical) then translate (lifted to
    above-goal) then lower (to goal).

    Returns (T, 7) torch tensor of arm waypoints concatenated across
    segments, or ``None`` if any segment failed.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    from maniguard.rl.grasps.collector import (
        _patch_curobo_mimic_lookup,
        _pose_to_mat,
        _mat_to_pose,
    )

    _patch_curobo_mimic_lookup()
    motion_gen = primitives._motion_generator
    arm = robot.default_arm
    eef_link = robot.eef_link_names[arm]

    # Current eef + target poses → relative offset (eef in target frame).
    eef_pos_now, eef_quat_now = robot.eef_links[arm].get_position_orientation()
    tgt_pos_now, tgt_quat_now = target_obj.get_position_orientation()
    T_eef_w = _pose_to_mat(eef_pos_now.cpu().numpy(), eef_quat_now.cpu().numpy())
    T_tgt_w = _pose_to_mat(tgt_pos_now.cpu().numpy(), tgt_quat_now.cpu().numpy())
    T_eef_in_tgt = np.linalg.inv(T_tgt_w) @ T_eef_w

    eef_pos_now_np = eef_pos_now.cpu().numpy().astype(np.float64)
    tgt_pos_now_np = tgt_pos_now.cpu().numpy().astype(np.float64)
    goal_xyz = np.asarray(goal_target_pos_world, dtype=np.float64)

    print(f"[PnP transport] eef_now={eef_pos_now_np.tolist()}", flush=True)
    print(f"[PnP transport] tgt_now={tgt_pos_now_np.tolist()} → "
          f"tgt_goal={goal_xyz.tolist()}", flush=True)
    robot_base = robot.get_position_orientation()[0].cpu().numpy()
    print(f"[PnP transport] robot_base={robot_base.tolist()}", flush=True)


    # TODO: Check why the attach_objects_to_robot approach crashed on the teacup mesh and consider re-enabling it.
    
    # Drop the held target from the obstacle world. With it in the world,
    # the robot's gripper at the held pose is "in collision" with the
    # target → any planner call from this start state fails immediately.
    # Two options: (a) cuRobo's attach_objects_to_robot, which extracts
    # the target's mesh from the world, merges it as an attached body,
    # and removes it from the obstacle list; (b) ignore the target
    # outright and rely on the gripper's own collision spheres to
    # approximate the held footprint. (a) is the principled choice but
    # the merge step crashed on the teacup mesh; (b) is simpler and the
    # teacup is small enough that the gripper spheres already cover it.
    motion_gen.update_obstacles(ignore_objects=[target_obj])

    # If goal_target_quat_world is provided, build a 4-segment plan:
    # lift → hover → rotate-in-place → descend. The rotate segment sits
    # at the hover position and changes orientation only. Splitting
    # rotation out into its own segment keeps each cuRobo call as a
    # "mostly translation" or "mostly rotation" problem, which the
    # trajopt handles much better than mixed translate+rotate.
    if goal_target_quat_world is not None:
        goal_quat_np = np.asarray(goal_target_quat_world, dtype=np.float64)
        T_tgt_goal_oriented = _pose_to_mat(np.zeros(3), goal_quat_np)
    else:
        T_tgt_goal_oriented = None

    def _eef_goal_for_tgt(tgt_xyz_np: np.ndarray, use_goal_quat: bool):
        if use_goal_quat and T_tgt_goal_oriented is not None:
            T_tgt_goal = T_tgt_goal_oriented.copy()
        else:
            T_tgt_goal = T_tgt_w.copy()
        T_tgt_goal[:3, 3] = tgt_xyz_np
        T_eef_goal = T_tgt_goal @ T_eef_in_tgt
        p, q = _mat_to_pose(T_eef_goal)
        return (th.tensor(p, dtype=th.float32),
                th.tensor(q, dtype=th.float32))

    # Waypoints: lift → above-goal → (rotate) → goal. Use the current
    # target xy for the lift, and target_goal xy for above-goal + final.
    wp1_xyz = tgt_pos_now_np.copy(); wp1_xyz[2] += lift_z
    wp2_xyz = goal_xyz.copy();        wp2_xyz[2] = tgt_pos_now_np[2] + hover_z
    wp3_xyz = goal_xyz.copy()

    print(f"[PnP transport] waypoints (target_xyz):", flush=True)
    print(f"    wp1 (lift)    = {wp1_xyz.tolist()}", flush=True)
    print(f"    wp2 (hover)   = {wp2_xyz.tolist()}", flush=True)
    if T_tgt_goal_oriented is not None:
        print(f"    wp2.5 (rotate)= {wp2_xyz.tolist()}  "
              f"(at hover xyz, new quat)", flush=True)
    print(f"    wp3 (descend) = {wp3_xyz.tolist()}", flush=True)
    if T_tgt_goal_oriented is not None:
        print(f"    target_quat_world (rotate+descend) = "
              f"{goal_quat_np.tolist()}", flush=True)

    # Disable collision on the gripper links for the duration of the
    # transport plan. Reason: at the held pose the Franka panda finger
    # collision spheres extend ~12-15 cm below the eef link, which is
    # below the desk top — so with full collision enabled, cuRobo reads
    # the start state as colliding with the support surface and rejects
    # every plan. Phase A solves the same problem via the same toggle
    # during Stage-2 linear servo. AG holds the target during transport,
    # so finger pose is locked relative to the gripper anyway.
    from maniguard.rl.grasps.collector import _FRANKA_GRIPPER_COLLISION_LINKS
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    raw_mg = motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT]

    seg_timeout = max(15.0, transport_timeout / 3.0)
    # Each entry: (label, arm_traj [T,7], eef_traj_base [T,7]=[pos,quat_xyzw])
    seg_pairs: list[tuple[str, "torch.Tensor", "torch.Tensor"]] = []
    chain_init = None
    if T_tgt_goal_oriented is not None:
        plan_spec = [
            # (label, target_xyz, toggle_gripper_off, use_goal_quat)
            ("lift",    wp1_xyz, True,  False),  # current quat, +lift_z
            ("hover",   wp2_xyz, False, False),  # current quat, translate
        ]
        if not skip_rotate:
            plan_spec.append(
                ("rotate",  wp2_xyz, False, True))   # goal quat, rotate in place
        if not skip_descend:
            # Descend uses goal quat unless rotate was skipped, in which case
            # we descend with the current quat (consistent with the upstream
            # hover segment which never adopted the goal orientation).
            plan_spec.append(
                ("descend", wp3_xyz, True, not skip_rotate))
    else:
        plan_spec = [
            ("lift",    wp1_xyz, True,  False),
            ("hover",   wp2_xyz, False, False),
        ]
        if not skip_descend:
            plan_spec.append(("descend", wp3_xyz, True, False))
    for label, tgt_xyz, toggle_off, use_goal_quat in plan_spec:
        if label == "lift":
            # Use the ACTUAL current eef pose as the reference for the
            # lift segment instead of reconstructing via T_eef_in_tgt.
            # The reconstructed eef goal drifts by a few mm + several
            # degrees from the achievable pose when the post-grasp arm
            # configuration is near-singular (e.g. extreme --lid-at-edge
            # cases), causing IK_FAIL even at lift_z=0. Building the
            # goal directly as (eef_now_xyz + (0,0,lift_z), eef_quat_now)
            # eliminates the reconstruction drift and lets cuRobo plan a
            # pure +z translation from a self-consistent start state.
            re_eef_pos, re_eef_quat = (
                robot.eef_links[arm].get_position_orientation())
            re_eef_pos_np = re_eef_pos.cpu().numpy().astype(np.float64)
            re_eef_quat_np = re_eef_quat.cpu().numpy().astype(np.float64)
            lift_delta_z = float(tgt_xyz[2] - tgt_pos_now_np[2])
            ep_np = re_eef_pos_np.copy()
            ep_np[2] += lift_delta_z
            ep = th.tensor(ep_np, dtype=th.float32)
            eq = th.tensor(re_eef_quat_np, dtype=th.float32)
            print(f"[PnP transport]   lift override: using actual eef "
                  f"pos={re_eef_pos_np.tolist()} + dZ={lift_delta_z:+.3f}, "
                  f"quat={re_eef_quat_np.tolist()}", flush=True)
        else:
            ep, eq = _eef_goal_for_tgt(tgt_xyz, use_goal_quat=use_goal_quat)
        t_seg = time.time()
        # Skip the gripper-link collision toggle for the lift segment.
        # The gripper_target_teleop tool succeeds on these same
        # post-grasp configurations without toggling — and the toggle
        # appears to corrupt cuRobo's internal collision config when
        # we're starting from a near-singular post-grasp joint state,
        # producing IK_FAIL with 6mm + 16° residuals even for a pure
        # +z translation. Pnp's other segments (hover, descend) still
        # toggle since they need the relaxation.
        toggled = (
            toggle_off
            and label != "lift"
            and hasattr(raw_mg, "toggle_link_collision")
        )
        if toggled:
            raw_mg.toggle_link_collision(
                list(_FRANKA_GRIPPER_COLLISION_LINKS), False)
        try:
            arm_traj, final_full, eef_traj = _solve_one_segment(
                motion_gen, robot, target_obj, eef_link, ep, eq,
                initial_joint_pos=chain_init,
                timeout=seg_timeout, label=label,
                use_attached_obj=False,
            )
        finally:
            if toggled:
                raw_mg.toggle_link_collision(
                    list(_FRANKA_GRIPPER_COLLISION_LINKS), True)
        wall = time.time() - t_seg
        if seg_timings is not None:
            seg_timings.append({
                "label": label, "wall_s": wall,
                "toggle_gripper_off": toggle_off,
                "ok": arm_traj is not None,
                "n_wp": int(len(arm_traj)) if arm_traj is not None else 0,
            })
        if arm_traj is None:
            return None
        seg_pairs.append((label, arm_traj, eef_traj))
        chain_init = final_full

    return seg_pairs


def _render_planned_trajectory(env, og, robot, target_obj, seg_pairs, *,
                               recorder, n_markers_per_segment: int = 8):
    """Spawn persistent visual gripper-rectangle markers at sampled
    waypoints along each planned segment. The markers are visual-only
    fixed_base cubes (3 per gripper pose: palm + left finger + right
    finger) placed in world frame at the planned eef pose. They stay in
    the scene throughout Phase B EXECUTE so the user can visually verify
    whether the robot reaches each planned waypoint.

    ``seg_pairs`` is the list of ``(label, arm_traj, eef_traj_base)``
    produced by ``_plan_transport``. ``eef_traj_base`` is the eef pose
    in robot base frame; we transform to world by composing with the
    robot's (fixed) base pose.
    """
    import torch as th
    from omnigibson.objects.primitive_object import PrimitiveObject
    from maniguard.rl.grasps.obb_sampler import (
        _FRANKA_MAX_OPENING,
        _FRANKA_FINGER_LEN,
        _FRANKA_FINGER_BREAD,
        _FRANKA_FINGER_THICK,
        _FRANKA_EEF_TO_TIP,
        _FRANKA_PALM_HALF_LEN,
        _FRANKA_PALM_HALF_WIDTH,
        _FRANKA_PALM_HALF_BREAD,
    )

    # Box geometry in grasp frame (matches tools/visualize_grasp_candidates.py).
    # Use closed gripper (open_half = obj_to_eef-ish) so the rectangles
    # reflect what the gripper looks like during transport.
    open_half = 0.5 * _FRANKA_MAX_OPENING * 0.2  # nearly closed
    fl = _FRANKA_FINGER_LEN
    fb = _FRANKA_FINGER_BREAD
    ft = _FRANKA_FINGER_THICK
    et = _FRANKA_EEF_TO_TIP
    fz = et - fl / 2
    pz = et - fl - _FRANKA_PALM_HALF_LEN

    boxes_def = [
        ("left",  np.array([0.0, -(open_half + ft / 2), fz]),
                  np.array([fb, ft, fl])),
        ("right", np.array([0.0, +(open_half + ft / 2), fz]),
                  np.array([fb, ft, fl])),
        ("palm",  np.array([0.0, 0.0, pz]),
                  np.array([2 * _FRANKA_PALM_HALF_BREAD,
                            2 * _FRANKA_PALM_HALF_WIDTH,
                            2 * _FRANKA_PALM_HALF_LEN])),
    ]
    SEG_COLORS = {
        "lift":    (0.20, 0.40, 0.95),  # blue
        "hover":   (0.30, 0.85, 0.40),  # green
        "descend": (0.95, 0.45, 0.25),  # orange-red
    }

    # Robot base pose in world (fixed-base, so this never changes).
    robot_pos_w, robot_quat_w = robot.get_position_orientation()
    T_robot_world = _pose_to_mat_local(
        robot_pos_w.cpu().numpy().astype(np.float64),
        robot_quat_w.cpu().numpy().astype(np.float64),
    )

    marker_count = 0
    for seg_idx, (label, _arm_traj, eef_traj_base) in enumerate(seg_pairs):
        n = len(eef_traj_base)
        if n == 0:
            continue
        idxs = np.linspace(0, n - 1, n_markers_per_segment).round().astype(int)
        color = SEG_COLORS.get(label, (0.7, 0.7, 0.7))

        for k, i in enumerate(idxs):
            eef_pos_b = eef_traj_base[i, :3].cpu().numpy().astype(np.float64)
            eef_quat_b = eef_traj_base[i, 3:7].cpu().numpy().astype(np.float64)
            T_eef_base = _pose_to_mat_local(eef_pos_b, eef_quat_b)
            T_eef_world = T_robot_world @ T_eef_base

            for bname, center_g, extents in boxes_def:
                box_in_eef = np.eye(4)
                box_in_eef[:3, 3] = center_g
                T_box_world = T_eef_world @ box_in_eef
                pos_w, quat_w = _mat_to_pose_local(T_box_world)

                marker = PrimitiveObject(
                    relative_prim_path=f"/plan_mk_{label}_{k}_{bname}",
                    name=f"plan_mk_{label}_{k}_{bname}",
                    category="plan_marker",
                    primitive_type="Cube",
                    size=1.0,
                    scale=list(map(float, extents)),
                    fixed_base=True,
                    visual_only=True,
                    rgba=[*color, 0.40],
                )
                env.scene.add_object(marker)
                marker.set_position_orientation(
                    position=th.tensor(pos_w, dtype=th.float32),
                    orientation=th.tensor(quat_w, dtype=th.float32),
                )
                # Some materials ignore the rgba kwarg — force diffuse.
                for vm in marker.root_link.visual_meshes.values():
                    mat = vm.material
                    if mat is None:
                        continue
                    mat.diffuse_color_constant = th.tensor(
                        color, dtype=th.float32,
                    )
                    mat.opacity_constant = 0.40
                marker_count += 1

    print(f"[PnP preview] spawned {marker_count} visual markers "
          f"({n_markers_per_segment} wp/seg × {len(boxes_def)} boxes/wp "
          f"× {len(seg_pairs)} segments)", flush=True)
    # Render a few frames so the user sees the markers settle before
    # Phase B EXECUTE begins.
    for _ in range(5):
        og.sim.render()
    if recorder is not None:
        recorder.record(env, og)


def _pose_to_mat_local(pos, quat_xyzw):
    from maniguard.rl.grasps.collector import _pose_to_mat
    return _pose_to_mat(pos, quat_xyzw)


def _mat_to_pose_local(T):
    from maniguard.rl.grasps.collector import _mat_to_pose
    return _mat_to_pose(T)


def _quat_canonical(q_xyzw):
    """Return q or -q so that w >= 0 (canonical shortest-path rep).
    Prevents axis-angle deltas from wrapping the long way around."""
    if float(q_xyzw[3].item()) < 0.0:
        return -q_xyzw
    return q_xyzw


def _replay_holding(env, og, robot, target_obj, arm_joint_traj, *,
                    deadline: float, frame_callback=None,
                    sft_recorder=None,
                    final_settle_steps: int = 60,
                    segment_breakdown=None,
                    goal_spec=None,
                    early_exit_after: str = "hover",
                    gripper_cmd: float = -1.0):
    """JointController replay of a cuRobo joint trajectory while
    holding the target with the gripper closed.

    ``arm_joint_traj`` is a (T, n_arm_dof) tensor of arm-joint
    waypoints — produced by cuRobo's ``compute_trajectories`` (and
    concatenated across segments by the caller). The trajectory is
    linearly interp'd up to ~2x density so each sim step is a small
    joint delta JointController can PD-track in one pass.

    Per env.step the action is built by ``q_to_action(q_full)`` where
    ``q_full`` is the current robot joint vector with the arm joints
    replaced by the next waypoint and the gripper at its CLOSED limit.

    For SFT recording we still emit 7D EEF-delta actions, computed via
    FK from the eef pose change across the step.
    """
    import torch as th
    import omnigibson.utils.transform_utils as T

    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    gripper_closed_q = robot.joint_lower_limits[gripper_control_idx]

    arm_traj = (arm_joint_traj if isinstance(arm_joint_traj, th.Tensor)
                else th.as_tensor(arm_joint_traj, dtype=th.float32))
    n = len(arm_traj)

    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]
    WP_TOL_RAD = 0.02
    WP_MAX_STEPS = 8  # ~264ms per waypoint cap (8 × 33ms)

    def _jc_step(arm_target):
        cur_pos_b_pre, cur_quat_b_pre = robot.get_relative_eef_pose(arm)
        cur_pos_b_pre = cur_pos_b_pre.float()
        cur_quat_b_pre = cur_quat_b_pre.float()
        # Build action: arm slot = target joints, gripper slot = command
        # (default -1.0 = close → closed-grip transport; teleop overrides
        # to +1.0 to keep the gripper open during click-to-target moves).
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[arm_action_idx] = arm_target.to(action.dtype)
        action[gripper_action_idx] = gripper_cmd
        env.step(action)
        # 7D EEF-delta from FK pose change (gripper closed → cmd = -1).
        if sft_recorder is not None:
            cur_pos_b, cur_quat_b = robot.get_relative_eef_pose(arm)
            cur_pos_b = cur_pos_b.float()
            cur_quat_b = cur_quat_b.float()
            dpos = (cur_pos_b - cur_pos_b_pre).detach().cpu().numpy().astype(np.float32)
            q_t = _quat_canonical(cur_quat_b)
            q_p = _quat_canonical(cur_quat_b_pre)
            q_delta = _quat_canonical(T.quat_multiply(
                q_t, T.quat_inverse(q_p)))
            daa = T.quat2axisangle(q_delta).detach().cpu().numpy().astype(np.float32)
            act7 = np.zeros(7, dtype=np.float32)
            act7[:3] = dpos
            act7[3:6] = daa
            act7[6] = -1.0
            sft_recorder.record_step(act7, done=False)
        if frame_callback is not None:
            frame_callback()

    print(f"[PnP replay] {n} joint waypoints, up to {WP_MAX_STEPS} "
          f"substeps/wp, early-exit on q_err<{WP_TOL_RAD} rad", flush=True)
    # Build a per-waypoint segment label map so we can announce when the
    # replay crosses a stage boundary. segment_breakdown is the seg_pairs
    # list of (label, n_wp_in_segment).
    seg_starts: dict[int, str] = {}
    if segment_breakdown:
        cum = 0
        for label, seg_n in segment_breakdown:
            seg_starts[cum] = label
            cum += int(seg_n)
    last_err = 0.0
    last_substeps = 0
    # Articulation may transiently lose its view across env.step inside
    # this loop. Edge-trigger the warning so we don't print every
    # substep — print once on entry, once on recovery.
    art_broken = False
    for wi in range(n):
        if wi in seg_starts:
            new_stage = seg_starts[wi]
            # End-of-hover check: if the held target already intersects
            # the goal region, skip the descend segment entirely. Descend
            # commonly triggers a cuRobo IK branch flip JointController can't
            # track, knocking over clutter or tipping the target.
            if (goal_spec is not None and early_exit_after is not None
                    and new_stage != early_exit_after):
                prev_labels = [seg_starts[k] for k in seg_starts if k < wi]
                if early_exit_after in prev_labels:
                    from maniguard.utils.goal_region import (
                        object_intersects_goal_region,
                        target_or_gripper_in_goal,
                    )
                    pos_ok, which = target_or_gripper_in_goal(
                        env, target_obj, goal_spec)
                    intersects = object_intersects_goal_region(
                        target_obj, goal_spec)
                    if pos_ok and intersects:
                        print(f"[PnP replay] '{early_exit_after}' ended "
                              f"INSIDE goal region (by={which}); skipping "
                              f"'{new_stage}' ({n-wi} waypoints saved) AND "
                              f"final settle — declaring success",
                              flush=True)
                        return True
            print(f"[PnP replay] >>> stage '{new_stage}' "
                  f"(wp {wi}/{n})", flush=True)
        if time.time() > deadline:
            print(f"[PnP replay] DEADLINE at wp {wi}/{n}", flush=True)
            return False
        last_substeps = WP_MAX_STEPS
        for k in range(WP_MAX_STEPS):
            if time.time() > deadline:
                return False
            _jc_step(arm_traj[wi])
            # robot.get_joint_positions() occasionally returns None when
            # Isaac transiently de-initializes the articulation after a
            # cuRobo plan. Edge-triggered logging: print once when state
            # flips bad/good, not every substep. Not silent — any new
            # failure mode still surfaces the first time it appears.
            try:
                cur_q = robot.get_joint_positions()[arm_control_idx]
                if art_broken:
                    print(f"[PnP replay] articulation RECOVERED at wp "
                          f"{wi}/{n} substep {k}", flush=True)
                    art_broken = False
            except (AttributeError, TypeError, RuntimeError) as exc:
                if not art_broken:
                    print(f"[PnP replay] articulation BROKEN at wp "
                          f"{wi}/{n} substep {k} "
                          f"({type(exc).__name__}: {exc}); skipping "
                          f"q_err check until recovery", flush=True)
                    art_broken = True
                continue
            last_err = float(
                (cur_q - arm_traj[wi].to(cur_q.dtype).to(cur_q.device))
                .norm().item())
            if last_err < WP_TOL_RAD:
                last_substeps = k + 1
                break
        if wi % 10 == 0 or wi == n - 1:
            print(f"[PnP replay]   wp {wi+1}/{n}  q_err={last_err:.4f} rad "
                  f"({last_substeps} substeps)", flush=True)

    # Final settle: hold last waypoint with gripper closed so AG locks.
    print(f"[PnP replay] final settle ({final_settle_steps} steps holding "
          "last joint waypoint, gripper closed) ...", flush=True)
    for k in range(final_settle_steps):
        if time.time() > deadline:
            return False
        _jc_step(arm_traj[-1])
    return True


def _record_phase_a_replay(env, og, robot, target_obj, *,
                           home_joint_q, target_init_pos, target_init_quat,
                           approach_traj, sft_recorder,
                           cfg=None,
                           deadline: float | None = None):
    """Replay a successful Phase A through PHYSICS with SFT recording.

    Calls :func:`run_grasp_attempt` directly so the AG re-engagement +
    gravity-hold verification path is identical to the candidate
    iteration that originally succeeded. Returns True on a clean
    re-grasp, False otherwise.

    The previous implementation duplicated the run_grasp_attempt body
    with a kinematic-only path that SKIPPED the gravity-hold AG check.
    That left a silent failure mode: AG could fail to re-engage during
    the kinematic close, the function returned anyway, then Phase 1B
    moved the gripper around without actually carrying the lid.
    """
    import time as _time
    import torch as th
    from maniguard.rl.grasps.collector import (
        run_grasp_attempt, GraspCollectorConfig,
    )

    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    open_gripper_q = robot.joint_upper_limits[gripper_control_idx].clone()
    zero_arm_cmd = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)
    if cfg is None:
        cfg = GraspCollectorConfig(num_target_grasps=1)
    if deadline is None:
        deadline = _time.time() + 120.0

    sft_recorder.reset_eef_history()

    # Phase-aware callback: gripper_cmd=+1 (open) during init+approach+
    # settle_open, -1 (close) during close+gravity_hold.
    def _phase_cb(phase: str) -> None:
        gripper_cmd = 1.0 if phase in ("init", "approach",
                                       "settle_open") else -1.0
        sft_recorder.record_fk_step(gripper_cmd=gripper_cmd, done=False)

    approach_traj_t = th.as_tensor(approach_traj, dtype=th.float32)

    result = run_grasp_attempt(
        env, robot, target_obj, target_init_pos, target_init_quat,
        joint_traj=approach_traj_t,
        cfg=cfg,
        open_gripper_q=open_gripper_q,
        zero_arm_cmd=zero_arm_cmd,
        arm_control_idx=arm_control_idx,
        gripper_control_idx=gripper_control_idx,
        initial_joint_pos=home_joint_q,
        deadline=deadline,
        frame_callback=_phase_cb,
    )

    if result is None:
        print(f"[PnP] SFT Phase A replay: AG did NOT re-engage — "
              f"the recorded trajectory exists but the env state is "
              f"NOT grasped. Phase 1B will fail.", flush=True)
        return False
    print(f"[PnP] SFT Phase A replay: re-engaged AG, recorded "
          f"{len(approach_traj) + 1 + cfg.settle_open_steps + cfg.close_steps + cfg.gravity_hold_steps} "
          f"steps", flush=True)
    return True


def _run_one_variant(
    env, og, robot, target_obj, primitives, *,
    home_joint_q, target_init_pos, target_init_quat, approach_traj,
    grasp, goal_spec, goal_center, goal_radius, target_name,
    phase_a_wall_s, phase_a_breakdown,
    args, task_dir, out_dir, camera_names,
    variant_idx, n_variants,
    lift_z, hover_z,
    lerobot_dataset=None,
    ltl_monitor=None,
    phase_a_cache=None,
    cache_phase_a=False,
) -> dict:
    """Replay Phase A + plan & execute Phase B with the given lift/hover.

    Creates per-variant SFTRecorder + ReviewVideoRecorder + result_payload,
    writes ``result.json`` + ``trajectory.pt`` + the SFT rollout artifacts
    into ``out_dir``. Returns the result_payload dict (so the caller can
    aggregate across variants).

    Phase A search is NOT re-run; we replay the supplied ``approach_traj``
    so AG re-engages from the same initial robot/target pose.
    """
    import torch as th

    from maniguard.envs.frozen_task_runtime import ReviewVideoRecorder
    from maniguard.utils.goal_region import (
        object_intersects_goal_region,
        robot_holds_target,
        target_or_gripper_in_goal,
    )

    sft_recorder = None
    lerobot_writer = None
    if args.record_sft:
        from tools._sft_recorder import SFTRecorder
        if lerobot_dataset is not None:
            from maniguard.data.lerobot.lerobot_writer import (
                LeRobotEpisodeWriter, episode_prompt,
            )
            lerobot_writer = LeRobotEpisodeWriter(lerobot_dataset)
            prompt = episode_prompt(target_name, args.lerobot_prompt_template)
        else:
            prompt = None
        sft_recorder = SFTRecorder(
            out_dir, resolution=args.record_resolution, fps=args.video_fps,
            lerobot_writer=lerobot_writer, lerobot_prompt=prompt,
            ltl_monitor=ltl_monitor,
        )
        sft_recorder.attach(env, env.robots[0])

    recorder_ctx = (
        ReviewVideoRecorder(path=out_dir, fps=args.video_fps,
                            camera_names=camera_names)
        if args.save_video else None
    )

    result_payload: dict = {
        "task_dir": str(task_dir),
        "target_name": target_name,
        "goal_center_world": goal_center.tolist(),
        "goal_radius_m": goal_radius,
        "variant_idx": variant_idx,
        "n_variants": n_variants,
        "lift_z": lift_z,
        "hover_z": hover_z,
        "phase_a": {
            "held": True,
            "wall_s": round(phase_a_wall_s, 2),
            "obj_to_eef": float(grasp["obj_to_eef"]),
            "breakdown": phase_a_breakdown,
        },
        "phase_b": {"planned": False, "executed": False, "success": False,
                    "final_target_to_goal_m": None},
    }

    if recorder_ctx is not None:
        recorder_ctx.__enter__()

    try:
        # ----- SFT Phase A capture / replay-from-cache -----
        # First variant (cache_phase_a=True, phase_a_cache=None): run the
        # real kinematic replay + AG re-engagement and tee the per-step
        # frames into the recorder's cache.
        # Subsequent variants (phase_a_cache=<list>): skip the physics and
        # push the cached frames straight into the recorder. Caller has
        # already restored the post-Phase-A sim state and re-stepped the
        # LTL automaton through the cached AP labels.
        if sft_recorder is not None:
            if phase_a_cache is None:
                if cache_phase_a:
                    sft_recorder.start_phase_a_cache()
                _record_phase_a_replay(
                    env, og, robot, target_obj,
                    home_joint_q=home_joint_q,
                    target_init_pos=target_init_pos,
                    target_init_quat=target_init_quat,
                    approach_traj=approach_traj,
                    sft_recorder=sft_recorder,
                )
                if cache_phase_a:
                    captured = sft_recorder.end_phase_a_cache()
                    result_payload["_captured_phase_a"] = captured
                    result_payload["_post_phase_a_state"] = og.sim.dump_state()
            else:
                for frame in phase_a_cache:
                    sft_recorder.append_cached_step(frame)
                print(f"[PnP v{variant_idx:02d}] Phase A: replayed "
                      f"{len(phase_a_cache)} cached frames "
                      f"(no physics)", flush=True)

        # Record a few extra "holding" frames before replanning.
        if recorder_ctx is not None:
            for _ in range(5):
                og.sim.step()
                recorder_ctx.record(env, og)

        # ----- Phase B PLAN -----
        print(f"[PnP v{variant_idx:02d}] Phase B PLAN: "
              f"lift_z={lift_z:.3f}, hover_z={hover_z:.3f} ...",
              flush=True)
        t1 = time.time()
        goal_target_pos = goal_center.copy()
        seg_timings: list = []
        seg_pairs = _plan_transport(
            primitives, robot, target_obj, goal_target_pos,
            transport_timeout=args.transport_timeout,
            lift_z=lift_z, hover_z=hover_z,
            seg_timings=seg_timings,
        )
        plan_wall = time.time() - t1
        result_payload["phase_b"]["plan_wall_s"] = round(plan_wall, 2)
        result_payload["phase_b"]["plan_segments"] = [
            {"label": st["label"],
             "wall_s": round(st["wall_s"], 3),
             "toggle_gripper_off": st["toggle_gripper_off"],
             "ok": st["ok"], "n_wp": st["n_wp"]}
            for st in seg_timings
        ]
        for st in seg_timings:
            print(f"[PnP v{variant_idx:02d}]   plan seg {st['label']!r}: "
                  f"ok={st['ok']} wall={st['wall_s']*1000:.1f}ms "
                  f"n_wp={st['n_wp']}", flush=True)
        if seg_pairs is None or len(seg_pairs) == 0:
            first_fail = next((st["label"] for st in seg_timings
                               if not st["ok"]), "unknown")
            print(f"[PnP v{variant_idx:02d}] Phase B PLAN FAILED at "
                  f"segment {first_fail!r}", flush=True)
            result_payload["fail_step"] = f"phase_b_plan:{first_fail}"
            return result_payload
        transport_arm_traj = th.cat([s[1] for s in seg_pairs], dim=0)
        transport_eef_traj = th.cat([s[2] for s in seg_pairs], dim=0)
        result_payload["phase_b"]["planned"] = True
        result_payload["phase_b"]["traj_len"] = int(len(transport_arm_traj))
        result_payload["phase_b"]["segments"] = [
            {"label": lbl, "n_waypoints": int(len(arm_t))}
            for lbl, arm_t, _ in seg_pairs
        ]
        print(f"[PnP v{variant_idx:02d}] Phase B PLAN ok: "
              f"{len(transport_arm_traj)} waypoints across "
              f"{len(seg_pairs)} segments ({plan_wall:.1f}s)", flush=True)

        # ----- Render preview + save trajectory.pt before executing -----
        if not args.record_sft:
            _render_planned_trajectory(
                env, og, robot, target_obj, seg_pairs,
                recorder=recorder_ctx,
            )

        full_arm_traj = th.cat([
            th.as_tensor(approach_traj, dtype=th.float32),
            th.as_tensor(transport_arm_traj, dtype=th.float32),
        ], dim=0)
        th.save({
            "approach_traj": th.as_tensor(approach_traj, dtype=th.float32),
            "transport_arm_traj": th.as_tensor(transport_arm_traj, dtype=th.float32),
            "transport_eef_traj_base": th.as_tensor(transport_eef_traj, dtype=th.float32),
            "transport_segments": [
                {
                    "label": lbl,
                    "arm_traj": th.as_tensor(arm_t, dtype=th.float32),
                    "eef_traj_base": th.as_tensor(eef_t, dtype=th.float32),
                }
                for lbl, arm_t, eef_t in seg_pairs
            ],
            "full_arm_traj": full_arm_traj,
            "target_name": target_name,
            "goal_center_world": th.tensor(goal_center, dtype=th.float64),
            "goal_radius_m": float(goal_radius),
            "lift_z": float(lift_z),
            "hover_z": float(hover_z),
        }, str(out_dir / "trajectory.pt"))
        (out_dir / "result.json").write_text(json.dumps(
            {k: v for k, v in result_payload.items() if not k.startswith("_")},
            indent=2,
        ))

        # ----- Phase B EXECUTE (JointController replay) -----
        replay_deadline = time.time() + max(args.transport_timeout * 4, 240.0)
        cb = (lambda: recorder_ctx.record(env, og)) if recorder_ctx else None
        print(f"[PnP v{variant_idx:02d}] Phase B EXECUTE: "
              f"{len(transport_arm_traj)} joint waypoints ...", flush=True)
        t_exec = time.time()
        ok = _replay_holding(
            env, og, robot, target_obj, transport_arm_traj,
            deadline=replay_deadline, frame_callback=cb,
            sft_recorder=sft_recorder,
            segment_breakdown=[(lbl, len(arm_t)) for lbl, arm_t, _ in seg_pairs],
            goal_spec=goal_spec,
            early_exit_after="hover",
        )
        exec_wall = time.time() - t_exec
        result_payload["phase_b"]["execute_wall_s"] = round(exec_wall, 2)
        result_payload["phase_b"]["executed"] = bool(ok)
        if not ok:
            result_payload.setdefault("fail_step", "phase_b_execute")

        # Success check.
        tgt_pos_final, _ = target_obj.get_position_orientation()
        tgt_pos_final_np = tgt_pos_final.cpu().numpy().astype(np.float64)
        dist = float(np.linalg.norm(tgt_pos_final_np - goal_center))
        pos_ok, which = target_or_gripper_in_goal(env, target_obj, goal_spec)
        intersects_target = object_intersects_goal_region(target_obj, goal_spec)
        still_held = robot_holds_target(env, target_obj)
        success = ok and pos_ok and still_held
        result_payload["phase_b"]["final_target_world"] = tgt_pos_final_np.tolist()
        result_payload["phase_b"]["final_target_to_goal_m"] = dist
        result_payload["phase_b"]["target_intersects_goal"] = bool(intersects_target)
        result_payload["phase_b"]["gripper_or_target_in_goal"] = bool(pos_ok)
        result_payload["phase_b"]["pos_check_which"] = which
        result_payload["phase_b"]["robot_holds_target"] = bool(still_held)
        result_payload["phase_b"]["success"] = bool(success)
        print(f"[PnP v{variant_idx:02d}] Phase B EXECUTE "
              f"{'OK' if ok else 'TRUNCATED'} — "
              f"center_dist={dist:.3f} m, pos_ok={pos_ok} (by={which!r}), "
              f"target_in={intersects_target}, AG_held={still_held} "
              f"→ {'SUCCESS' if success else 'MISS'}", flush=True)
        if not success and "fail_step" not in result_payload:
            if not pos_ok:
                result_payload["fail_step"] = "goal_not_intersected"
            elif not still_held:
                result_payload["fail_step"] = "lost_grip"
            else:
                result_payload["fail_step"] = "unknown"

        # Hold final frames for the video.
        if recorder_ctx is not None:
            for _ in range(15):
                og.sim.step()
                recorder_ctx.record(env, og)
        return result_payload
    finally:
        if recorder_ctx is not None:
            recorder_ctx.__exit__(None, None, None)
        if sft_recorder is not None:
            phase_b = result_payload.get("phase_b", {})
            sft_success = bool(phase_b.get("success"))
            sft_recorder.finalize(success=sft_success, attrs={
                "task_dir": str(task_dir),
                "target_name": str(target_name),
                "seed": int(args.seed),
                "variant_idx": int(variant_idx),
                "lift_z": float(lift_z),
                "hover_z": float(hover_z),
                "phase_a_held": True,
            })
        if ltl_monitor is not None:
            s = ltl_monitor.summary()
            # Drop the per-step log from result.json — it can be reconstructed
            # from HDF5 if needed and it bloats this artifact.
            result_payload["ltl"] = {
                "formula": s["formula"],
                "violated": bool(s["violated"]),
                "violation_step": s["violation_step"],
                "violation_count": int(s["violation_count"]),
                "total_steps_monitored": int(s["total_steps_monitored"]),
            }
        (out_dir / "result.json").write_text(json.dumps(
            {k: v for k, v in result_payload.items() if not k.startswith("_")},
            indent=2,
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"task-dir not found: {task_dir}")

    out_dir = (args.out_dir or (task_dir / "pick_and_place")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lerobot_dataset = None
    if args.lerobot_repo_id:
        if not args.record_sft:
            raise SystemExit("--lerobot-repo-id requires --record-sft "
                             "(LeRobot needs the SFT recorder's frames).")
        from maniguard.data.lerobot.lerobot_writer import create_or_open_dataset
        lerobot_dataset = create_or_open_dataset(
            repo_id=args.lerobot_repo_id, root=args.lerobot_root,
            fps=args.video_fps, resolution=args.record_resolution,
        )
        print(f"[PnP] LeRobot dataset opened at {lerobot_dataset.root}  "
              f"(starting from episode {lerobot_dataset.meta.total_episodes})",
              flush=True)

    # ltl_monitor is instantiated after _build_env (needs the env handle).
    ltl_monitor = None

    og = _init_omnigibson(headless=not args.gui)

    if args.record_sft:
        from tools._sft_recorder import install_wrist_camera_patch
        install_wrist_camera_patch()

    env, og, diagnostics, camera_names = _build_env(
        task_dir, args.episode, no_distractors=args.no_distractors,
        record_sft=args.record_sft, record_resolution=args.record_resolution,
    )

    # Auto-attach an LTL monitor when the task ships a non-empty ltl_safety
    # spec in diagnostics.jsonl. The summary lands in result.json + HDF5
    # attrs + episodes.jsonl's ltl_violated flag. Skips itself silently if
    # Spot is missing or the spec is empty.
    ltl_safety_spec = diagnostics.get("ltl_safety") or {}
    if args.record_sft and ltl_safety_spec:
        try:
            from maniguard.utils.safety_monitor import TaskLTLMonitor
            # The pnp pipeline runs against a non-BDDL task (no object_scope),
            # so the proposition resolver needs an explicit active-object
            # dict. Build it from diagnostics.selection.spawn_specs by
            # mapping each scene object's category back to its synset short
            # name (which is what the proposition patterns use).
            cat_to_synset = {
                spec["category"]: spec["synset"].split(".")[0]
                for spec in diagnostics.get("selection", {}).get("spawn_specs", [])
            }
            active_by_inst = {}
            synset_counts = {}
            for obj in env.scene.objects:
                cat = getattr(obj, "category", None)
                short = cat_to_synset.get(cat)
                if short is None:
                    continue
                idx = synset_counts.get(short, 0)
                active_by_inst[f"{short}_{idx}"] = obj
                synset_counts[short] = idx + 1
            ltl_monitor = TaskLTLMonitor(
                env,
                ltl_safety=ltl_safety_spec,
                activity_name=diagnostics.get("activity_name", ""),
                scene_model=diagnostics.get("scene_model"),
                active_objects_by_inst=active_by_inst,
            )
            print(f"[PnP] LTL monitor attached  "
                  f"(active_objects: {sorted(active_by_inst.keys())})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[PnP] LTL monitor init failed ({e}); continuing without LTL",
                  flush=True)
            ltl_monitor = None
    try:
        from maniguard.utils.goal_region import GoalRegionSpec
        gr = diagnostics["goal_region"]
        goal_spec = GoalRegionSpec.from_json(gr)
        target_name = goal_spec.target_name
        target_obj = env.scene.object_registry("name", target_name)
        if target_obj is None:
            raise RuntimeError(f"target {target_name} not in registry")
        goal_center = np.asarray(goal_spec.center_world, dtype=np.float64)
        goal_radius = float(goal_spec.radius_m)
        # Pre-Phase-A geometry log (helps diagnose phase_a failures, where
        # _plan_transport never runs so its robot_base log never fires).
        robot_xyz = env.robots[0].get_position_orientation()[0].cpu().numpy().astype(np.float64)
        target_xyz_init = target_obj.get_position_orientation()[0].cpu().numpy().astype(np.float64)
        d_robot_target = float(np.linalg.norm((robot_xyz - target_xyz_init)[:2]))
        d_target_goal = float(np.linalg.norm((target_xyz_init - goal_center)[:2]))
        print(f"[PnP] target={target_name}  goal_center={goal_center.tolist()}  "
              f"goal_radius={goal_radius:.3f}", flush=True)
        print(f"[PnP] robot_xy={robot_xyz[:2].tolist()}  "
              f"target_xy={target_xyz_init[:2].tolist()}  "
              f"‖robot-target‖_xy={d_robot_target:.3f}m  "
              f"‖target-goal‖_xy={d_target_goal:.3f}m", flush=True)
        # Corrupt-goal guard: Franka reach is ~0.85m; a sane goal lies within
        # the robot's workspace from the target's initial pose. If the
        # dataset's goal_region.center_world is wildly far away (>1.5m), we
        # treat it as a data bug and skip with a distinct fail_step so the
        # driver's early-bail doesn't waste 30 seeds on phase_b_plan:hover.
        CORRUPT_GOAL_THRESHOLD_M = 1.5
        if d_target_goal > CORRUPT_GOAL_THRESHOLD_M:
            print(f"[PnP] CORRUPT GOAL: ‖target-goal‖_xy={d_target_goal:.3f}m "
                  f"exceeds {CORRUPT_GOAL_THRESHOLD_M}m. Skipping this task.",
                  flush=True)
            result_payload_early = {
                "task_dir": str(task_dir),
                "target_name": target_name,
                "goal_center_world": goal_center.tolist(),
                "goal_radius_m": goal_radius,
                "phase_a": {"held": False},
                "phase_b": {"planned": False, "executed": False, "success": False},
                "fail_step": "corrupt_goal_region",
                "diagnostics": {
                    "robot_xy": robot_xyz[:2].tolist(),
                    "target_xy_init": target_xyz_init[:2].tolist(),
                    "robot_to_target_m": d_robot_target,
                    "target_to_goal_m": d_target_goal,
                },
            }
            (out_dir / "result.json").write_text(json.dumps(result_payload_early, indent=2))
            print(f"[PnP] wrote {out_dir / 'result.json'} (corrupt_goal_region)",
                  flush=True)
            return

        # cuRobo init.
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        print(f"[{time.strftime('%H:%M:%S')}] initializing cuRobo ...", flush=True)
        primitives = StarterSemanticActionPrimitives(env, env.robots[0],
                                                     enable_head_tracking=False)
        print(f"[{time.strftime('%H:%M:%S')}] cuRobo ready.", flush=True)

        # ===== Phase A search — runs ONCE per task =====
        t0 = time.time()
        pick_deadline = t0 + args.pick_timeout
        robot = env.robots[0]
        home_joint_q = robot.get_joint_positions().clone()
        target_init_pos, target_init_quat = target_obj.get_position_orientation()
        target_init_pos = target_init_pos.clone()
        target_init_quat = target_init_quat.clone()
        # Snapshot world state BEFORE Phase A so multi-variant runs can
        # restore between variants and see the same initial scene.
        pre_phase_a_state = og.sim.dump_state()

        held_grasps_list, phase_a_timings = _phase_a_pick(
            env, og, primitives, target_obj, args, pick_deadline,
        )
        # Legacy pnp main path is single-grasp.
        grasp = held_grasps_list[0] if held_grasps_list else None
        phase_a_wall = time.time() - t0
        # Aggregate per-candidate timings into a summary.
        n_cands = len(phase_a_timings)
        n_held = sum(1 for t in phase_a_timings if t.get("held"))
        n_pc = sum(1 for t in phase_a_timings if "ik_precheck_s" in t)
        n_pc_ok = sum(1 for t in phase_a_timings if t.get("ik_precheck_ok"))
        n_s1 = sum(1 for t in phase_a_timings if "stage1_s" in t)
        n_s1_ok = sum(1 for t in phase_a_timings if t.get("stage1_ok"))
        n_s2 = sum(1 for t in phase_a_timings if "stage2_s" in t)
        sum_pc = sum(t.get("ik_precheck_s", 0.0) for t in phase_a_timings)
        sum_s1 = sum(t.get("stage1_s", 0.0) for t in phase_a_timings)
        sum_s2 = sum(t.get("stage2_s", 0.0) for t in phase_a_timings)
        sum_tot = sum(t.get("total_s", 0.0) for t in phase_a_timings)
        phase_a_breakdown = {
            "n_cands_tried": n_cands, "n_held": n_held,
            "ik_precheck": {"ran": n_pc, "ok": n_pc_ok,
                            "total_s": round(sum_pc, 3)},
            "stage1": {"ran": n_s1, "ok": n_s1_ok,
                       "total_s": round(sum_s1, 3)},
            "stage2": {"ran": n_s2, "total_s": round(sum_s2, 3)},
            "per_cand_total_s": round(sum_tot, 3),
        }
        print(f"[PnP] Phase A wall={phase_a_wall:.1f}s  "
              f"cands_tried={n_cands}  held={n_held}", flush=True)
        print(f"[PnP]   ik_precheck: ran={n_pc} ok={n_pc_ok} "
              f"total={sum_pc:.2f}s "
              f"avg={sum_pc/max(n_pc,1)*1000:.1f}ms", flush=True)
        print(f"[PnP]   stage1     : ran={n_s1} ok={n_s1_ok} "
              f"total={sum_s1:.2f}s "
              f"avg={sum_s1/max(n_s1,1)*1000:.1f}ms", flush=True)
        print(f"[PnP]   stage2     : ran={n_s2}           "
              f"total={sum_s2:.2f}s "
              f"avg={sum_s2/max(n_s2,1)*1000:.1f}ms", flush=True)
        (out_dir / "phase_a_timings.json").write_text(
            json.dumps(phase_a_timings, indent=2))

        if grasp is None:
            print("[PnP] Phase A FAILED — no grasp held", flush=True)
            (out_dir / "result.json").write_text(json.dumps({
                "task_dir": str(task_dir),
                "target_name": target_name,
                "goal_center_world": goal_center.tolist(),
                "goal_radius_m": goal_radius,
                "phase_a": {"held": False, "wall_s": round(phase_a_wall, 2),
                            "breakdown": phase_a_breakdown},
                "phase_b": {"planned": False, "executed": False,
                            "success": False},
                "fail_step": "phase_a",
            }, indent=2))
            return

        approach_traj = grasp["approach_traj"]

        # ===== Variant loop =====
        n_variants = max(1, args.n_transport_variants)
        rng = np.random.default_rng(args.seed)
        variant_summary: list = []
        phase_a_cache: list | None = None      # captured by variant 0
        post_phase_a_state = None              # sim state immediately after Phase A replay
        for vi in range(n_variants):
            # Per-variant out_dir + sampled lift/hover.
            if n_variants == 1:
                v_out_dir = out_dir
                lift_z = 0.25
                hover_z = 0.25
            else:
                v_out_dir = out_dir / f"variant_{vi:02d}"
                v_out_dir.mkdir(parents=True, exist_ok=True)
                lift_z = float(rng.uniform(args.lift_z_min, args.lift_z_max))
                hover_z = float(rng.uniform(args.hover_z_min, args.hover_z_max))

            # Restore world snapshot between variants. From variant 1
            # onwards, prefer the post-Phase-A snapshot so we skip the
            # physics replay entirely (the recorder will inject the cached
            # frames instead). Fall back to pre-Phase-A if variant 0 didn't
            # produce a cache (e.g. record_sft disabled).
            if vi > 0:
                if post_phase_a_state is not None and phase_a_cache is not None:
                    og.sim.load_state(post_phase_a_state)
                    og.sim.step()
                    # Re-step the LTL automaton through the cached AP labels
                    # so its state matches "post Phase A". reset() also
                    # clears the violation log so each variant gets a fresh
                    # summary including its Phase A frames.
                    if ltl_monitor is not None and ltl_monitor._monitor is not None:
                        ltl_monitor.reset()
                        for f in phase_a_cache:
                            ap = f.get("ap_labels")
                            if ap:
                                ltl_monitor._monitor.step(ap)
                else:
                    og.sim.load_state(pre_phase_a_state)
                    og.sim.step()

            print(f"\n[PnP] === Variant {vi+1}/{n_variants}  "
                  f"lift_z={lift_z:.3f}  hover_z={hover_z:.3f}  "
                  f"out_dir={v_out_dir.name} ===", flush=True)
            variant_result = _run_one_variant(
                env, og, robot, target_obj, primitives,
                home_joint_q=home_joint_q,
                target_init_pos=target_init_pos,
                target_init_quat=target_init_quat,
                approach_traj=approach_traj,
                grasp=grasp, goal_spec=goal_spec,
                goal_center=goal_center, goal_radius=goal_radius,
                target_name=target_name,
                phase_a_wall_s=phase_a_wall,
                phase_a_breakdown=phase_a_breakdown,
                args=args, task_dir=task_dir, out_dir=v_out_dir,
                camera_names=camera_names,
                variant_idx=vi, n_variants=n_variants,
                lift_z=lift_z, hover_z=hover_z,
                lerobot_dataset=lerobot_dataset,
                ltl_monitor=ltl_monitor,
                phase_a_cache=phase_a_cache,
                cache_phase_a=(vi == 0),
            )
            # Variant 0 produces the Phase A cache + post-replay snapshot.
            if vi == 0:
                phase_a_cache = variant_result.pop("_captured_phase_a", None)
                post_phase_a_state = variant_result.pop("_post_phase_a_state", None)
                if phase_a_cache is not None:
                    print(f"[PnP] Phase A cache captured: "
                          f"{len(phase_a_cache)} frames "
                          f"(variants 1..{n_variants-1} will skip physics replay)",
                          flush=True)
            variant_summary.append({
                "variant_idx": vi,
                "lift_z": lift_z, "hover_z": hover_z,
                "success": bool(variant_result.get("phase_b", {})
                                .get("success")),
                "fail_step": variant_result.get("fail_step"),
                "out_dir": str(v_out_dir),
            })

        # Write multi-variant aggregate summary alongside the per-variant
        # result.json files.
        if n_variants > 1:
            n_succ = sum(1 for v in variant_summary if v["success"])
            (out_dir / "variants_summary.json").write_text(json.dumps({
                "task_dir": str(task_dir),
                "target_name": target_name,
                "n_variants": n_variants,
                "n_succ": n_succ,
                "phase_a_wall_s": round(phase_a_wall, 2),
                "phase_a_breakdown": phase_a_breakdown,
                "lift_z_range": [args.lift_z_min, args.lift_z_max],
                "hover_z_range": [args.hover_z_min, args.hover_z_max],
                "seed": int(args.seed),
                "variants": variant_summary,
            }, indent=2))
            print(f"\n[PnP] variants summary: {n_succ}/{n_variants} succeeded, "
                  f"wrote {out_dir / 'variants_summary.json'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[PnP] FAIL: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
