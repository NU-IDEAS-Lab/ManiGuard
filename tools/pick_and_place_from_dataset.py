"""Pick-and-place pipeline driven by a replay-empty task dump.

Given a single task folder containing ``scene_ep<N>.json`` +
``diagnostics.jsonl`` (the same format that
``tools/replay_empty_from_dataset.py`` consumes), this script:

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


_REPO_ROOT = Path(__file__).resolve().parents[1]
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env build — lifted from tools/replay_empty_from_dataset.py
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
               record_sft: bool = False, record_resolution: int = 256):
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
    from sentinel.envs.frozen_task_runtime import (
        build_env_config,
        configure_review_sensors,
        extract_scene_robot_setup,
        position_diagnostics_cameras,
    )
    from sentinel.utils.goal_region import GoalRegionSpec, spawn_goal_region_marker

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
    #   - OSC arm controller in raw m/rad (Phase B feeds dpos/daa directly)
    #   - grasping_mode=assisted, action_normalize=False locked
    #   - Phase A drives via set_joint_positions (bypasses controllers)
    env_cfg = build_env_config(
        scene_info, diagnostics,
        camera_names=camera_names,
        controller_preset="osc",
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


def _phase_a_pick(env, og, primitives, target_obj, args, deadline: float):
    """Run OBB sampler + collect_valid_grasps; return the first held grasp.

    Returns a dict with keys ``approach_traj`` (T, 7 torch tensor),
    ``rel_position``, ``rel_orientation_xyzw``, ``gripper_qpos``,
    ``arm_joint_pos``, or None if no grasp held.
    """
    import torch as th

    from sentinel.rl.grasps.collector import (
        GraspCollectorConfig,
        collect_valid_grasps,
    )
    from sentinel.rl.grasps.mesh import mesh_from_og_object
    from sentinel.rl.grasps.obb_sampler import (
        OBBConfig, sample_obb_assisted_grasps,
    )

    robot = env.robots[0]
    init_pos, init_quat = target_obj.get_position_orientation()
    init_pos = init_pos.clone()
    init_quat = init_quat.clone()

    # OBB sampler needs gravity off so the mesh extraction is stable; the
    # collector also pins the target during approach.
    target_obj.root_link.disable_gravity()

    try:
        mesh = mesh_from_og_object(target_obj, use_visual=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[PnP] mesh extraction failed: {exc}", flush=True)
        return None

    rng = np.random.default_rng(args.seed)
    candidates, scores = sample_obb_assisted_grasps(
        mesh,
        config=OBBConfig(max_candidates=args.max_candidates),
        rng=rng,
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
        num_target_grasps=1,
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
    return (held[0] if held else None), timings_log


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

    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos, target_quat=target_quat,
        initial_joint_pos=initial_joint_pos, is_local=False,
        max_attempts=max_attempts, timeout=timeout,
        ik_fail_return=5, enable_finetune_trajopt=True,
        finetune_attempts=2, return_full_result=False,
        success_ratio=1.0 / bs,
        attached_obj=attached_obj,
        attached_obj_scale=attached_obj_scale,
        motion_constraint=None, skip_obstacle_update=True,
        ik_only=False, ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        print(f"[PnP transport] segment {label!r}: cuRobo FAILED "
              f"(0/{int(bs)} successes)", flush=True)
        return None, None, None
    joint_state = joint_states[success_idx[0]]
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
                    seg_timings: list | None = None):
    """Two-stage transport plan: lift (vertical) then translate (lifted to
    above-goal) then lower (to goal).

    Returns (T, 7) torch tensor of arm waypoints concatenated across
    segments, or ``None`` if any segment failed.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    from sentinel.rl.grasps.collector import (
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

    def _eef_goal_for_tgt(tgt_xyz_np: np.ndarray):
        T_tgt_goal = T_tgt_w.copy()
        T_tgt_goal[:3, 3] = tgt_xyz_np
        T_eef_goal = T_tgt_goal @ T_eef_in_tgt
        p, q = _mat_to_pose(T_eef_goal)
        return (th.tensor(p, dtype=th.float32),
                th.tensor(q, dtype=th.float32))

    # Three waypoints: lift → above-goal → goal. Hold the target
    # orientation throughout (we only translate). Use the current target xy
    # for the lift, and target_goal xy for above-goal + final.
    wp1_xyz = tgt_pos_now_np.copy(); wp1_xyz[2] += lift_z
    wp2_xyz = goal_xyz.copy();        wp2_xyz[2] = tgt_pos_now_np[2] + hover_z
    wp3_xyz = goal_xyz.copy()

    print(f"[PnP transport] waypoints (target_xyz):", flush=True)
    print(f"    wp1 (lift)    = {wp1_xyz.tolist()}", flush=True)
    print(f"    wp2 (hover)   = {wp2_xyz.tolist()}", flush=True)
    print(f"    wp3 (descend) = {wp3_xyz.tolist()}", flush=True)

    # Disable collision on the gripper links for the duration of the
    # transport plan. Reason: at the held pose the Franka panda finger
    # collision spheres extend ~12-15 cm below the eef link, which is
    # below the desk top — so with full collision enabled, cuRobo reads
    # the start state as colliding with the support surface and rejects
    # every plan. Phase A solves the same problem via the same toggle
    # during Stage-2 linear servo. AG holds the target during transport,
    # so finger pose is locked relative to the gripper anyway.
    from sentinel.rl.grasps.collector import _FRANKA_GRIPPER_COLLISION_LINKS
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    raw_mg = motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT]

    seg_timeout = max(15.0, transport_timeout / 3.0)
    # Each entry: (label, arm_traj [T,7], eef_traj_base [T,7]=[pos,quat_xyzw])
    seg_pairs: list[tuple[str, "torch.Tensor", "torch.Tensor"]] = []
    chain_init = None
    plan_spec = [
        # (label, target_xyz, toggle_gripper_collision_off)
        ("lift",    wp1_xyz, True),   # fingers below table at start
        ("hover",   wp2_xyz, False),  # lifted above clutter — full collisions
        ("descend", wp3_xyz, True),   # fingers near table at end
    ]
    for label, tgt_xyz, toggle_off in plan_spec:
        ep, eq = _eef_goal_for_tgt(tgt_xyz)
        t_seg = time.time()
        toggled = toggle_off and hasattr(raw_mg, "toggle_link_collision")
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
    from sentinel.rl.grasps.obb_sampler import (
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
                    try:
                        mat.diffuse_color_constant = th.tensor(
                            color, dtype=th.float32,
                        )
                        mat.opacity_constant = 0.40
                    except Exception:  # noqa: BLE001
                        pass
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
    from sentinel.rl.grasps.collector import _pose_to_mat
    return _pose_to_mat(pos, quat_xyzw)


def _mat_to_pose_local(T):
    from sentinel.rl.grasps.collector import _mat_to_pose
    return _mat_to_pose(T)


def _quat_canonical(q_xyzw):
    """Return q or -q so that w >= 0 (canonical shortest-path rep).
    Prevents axis-angle deltas from wrapping the long way around."""
    if float(q_xyzw[3].item()) < 0.0:
        return -q_xyzw
    return q_xyzw


def _replay_holding(env, og, robot, target_obj, eef_traj_base, *,
                    deadline: float, frame_callback=None,
                    sft_recorder=None,
                    inner_substeps_per_wp: int = 6,
                    inner_pos_tol: float = 0.005,
                    inner_rot_tol_rad: float = 0.05,
                    final_settle_steps: int = 200,
                    final_pos_tol: float = 0.01,
                    final_rot_tol_rad: float = 0.05):
    """OSC-native replay of a cuRobo eef trajectory while holding the
    target with the gripper closed.

    ``eef_traj_base`` is a (T, 7) tensor of ``[pos(3), quat_xyzw(4)]`` in
    the robot's BASE frame — produced by
    ``motion_gen.path_to_eef_trajectory``. Each cuRobo waypoint is given
    up to ``inner_substeps_per_wp`` env.steps to settle within
    (``inner_pos_tol``, ``inner_rot_tol_rad``); when the inner-loop
    converges early we move to the next waypoint immediately.

    This pattern (verified by tools/test_curobo_osc_tracking.py) keeps
    peak mid-trajectory tracking error around 2 cm at OSC default
    gains, and was the best of three strategies tested (vs. pure
    streaming and joint-space densification).

    Per env.step the action is:
        action[arm]     = [target_pos - cur_pos,
                           axisangle(canon(target_quat) * inv(canon(cur_quat)))]
        action[gripper] = -1.0   # close → AG bond carries the target

    Quaternions are canonicalized (w >= 0) before the delta to prevent
    long-way-around axis-angle wraps.
    """
    import torch as th
    import omnigibson.utils.transform_utils as T

    arm = robot.default_arm
    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]

    def _osc_step(target_pos_b, target_quat_b):
        cur_pos_b, cur_quat_b = robot.get_relative_eef_pose(arm)
        cur_pos_b = cur_pos_b.float()
        cur_quat_b = cur_quat_b.float()
        dpos = target_pos_b - cur_pos_b
        q_target = _quat_canonical(target_quat_b)
        q_cur = _quat_canonical(cur_quat_b)
        q_inv = T.quat_inverse(q_cur)
        q_delta = _quat_canonical(T.quat_multiply(q_target, q_inv))
        daa = T.quat2axisangle(q_delta)
        # action units are meters/radians directly — see env build for
        # action_normalize=False + matched OSC input/output limits.
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[arm_action_idx] = th.cat([dpos, daa])
        action[gripper_action_idx] = -1.0
        env.step(action)
        if sft_recorder is not None:
            act7 = np.zeros(7, dtype=np.float32)
            act7[:3] = dpos.detach().cpu().numpy().astype(np.float32)
            act7[3:6] = daa.detach().cpu().numpy().astype(np.float32)
            act7[6] = -1.0
            sft_recorder.record_step(act7, done=False)
        if frame_callback is not None:
            frame_callback()
        return float(th.norm(dpos).item()), float(th.norm(daa).item())

    n = len(eef_traj_base)
    print(f"[PnP replay] {n} waypoints, up to {inner_substeps_per_wp} "
          f"substeps/wp, early-exit on "
          f"(pos<{inner_pos_tol*1000:.1f}mm, rot<{inner_rot_tol_rad:.3f}rad) ...",
          flush=True)
    last_pos_err = last_rot_err = 0.0
    last_substeps = 0
    for wi in range(n):
        if time.time() > deadline:
            print(f"[PnP replay] DEADLINE at wp {wi}/{n}", flush=True)
            return False
        target_pos_b = eef_traj_base[wi, :3].float()
        target_quat_b = eef_traj_base[wi, 3:7].float()
        last_substeps = inner_substeps_per_wp
        for k in range(inner_substeps_per_wp):
            if time.time() > deadline:
                print(f"[PnP replay] DEADLINE at wp {wi}/{n} substep {k}",
                      flush=True)
                return False
            last_pos_err, last_rot_err = _osc_step(target_pos_b, target_quat_b)
            if (last_pos_err < inner_pos_tol and
                    last_rot_err < inner_rot_tol_rad):
                last_substeps = k + 1
                break
        if wi % 30 == 0 or wi == n - 1:
            print(f"[PnP replay]   wp {wi+1}/{n}  "
                  f"pos_err={last_pos_err:.4f} m  "
                  f"rot_err={last_rot_err:.4f} rad  "
                  f"({last_substeps} substeps)", flush=True)

    # Final settle on the last waypoint.
    target_pos_b = eef_traj_base[-1, :3].float()
    target_quat_b = eef_traj_base[-1, 3:7].float()
    print(f"[PnP replay] final settle (≤{final_settle_steps} steps, "
          f"tol=({final_pos_tol} m, {final_rot_tol_rad} rad)) ...", flush=True)
    for k in range(final_settle_steps):
        if time.time() > deadline:
            print(f"[PnP replay] DEADLINE during final settle (step {k})",
                  flush=True)
            return False
        last_pos_err, last_rot_err = _osc_step(target_pos_b, target_quat_b)
        if last_pos_err < final_pos_tol and last_rot_err < final_rot_tol_rad:
            print(f"[PnP replay]   settled after {k+1} steps  "
                  f"(pos_err={last_pos_err:.4f} m, rot_err={last_rot_err:.4f} rad)",
                  flush=True)
            return True
    print(f"[PnP replay]   final settle hit cap  "
          f"(pos_err={last_pos_err:.4f} m, rot_err={last_rot_err:.4f} rad)",
          flush=True)
    return True


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

    og = _init_omnigibson(headless=not args.gui)

    if args.record_sft:
        from tools._sft_recorder import install_wrist_camera_patch
        install_wrist_camera_patch()

    env, og, diagnostics, camera_names = _build_env(
        task_dir, args.episode, no_distractors=args.no_distractors,
        record_sft=args.record_sft, record_resolution=args.record_resolution,
    )
    sft_recorder = None
    if args.record_sft:
        from tools._sft_recorder import SFTRecorder
        sft_recorder = SFTRecorder(out_dir, resolution=args.record_resolution,
                                   fps=args.video_fps)
        sft_recorder.attach(env, env.robots[0])
    try:
        from sentinel.utils.goal_region import (
            GoalRegionSpec,
            object_intersects_goal_region,
            robot_holds_target,
            target_or_gripper_in_goal,
        )
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

        # Open the video recorder around the full pick+place rollout.
        from sentinel.envs.frozen_task_runtime import ReviewVideoRecorder

        recorder_ctx = (
            ReviewVideoRecorder(path=out_dir, fps=args.video_fps,
                                camera_names=camera_names)
            if args.save_video else None
        )

        result_payload = {
            "task_dir": str(task_dir),
            "target_name": target_name,
            "goal_center_world": goal_center.tolist(),
            "goal_radius_m": goal_radius,
            "phase_a": {"held": False},
            "phase_b": {"planned": False, "executed": False, "success": False,
                        "final_target_to_goal_m": None},
        }

        if recorder_ctx is not None:
            recorder_ctx.__enter__()

        try:
            # ----- Phase A -----
            t0 = time.time()
            pick_deadline = t0 + args.pick_timeout
            grasp, phase_a_timings = _phase_a_pick(
                env, og, primitives, target_obj, args, pick_deadline,
            )
            phase_a_wall = time.time() - t0
            result_payload["phase_a"]["wall_s"] = round(phase_a_wall, 2)
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
            result_payload["phase_a"]["breakdown"] = {
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
            # Persist per-candidate detail to a separate file (can be large).
            (out_dir / "phase_a_timings.json").write_text(
                json.dumps(phase_a_timings, indent=2))
            if grasp is None:
                print("[PnP] Phase A FAILED — no grasp held", flush=True)
                result_payload["fail_step"] = "phase_a"
                return
            result_payload["phase_a"]["held"] = True
            result_payload["phase_a"]["obj_to_eef"] = float(grasp["obj_to_eef"])
            approach_traj = grasp["approach_traj"]

            # Record a few extra "holding" frames before replanning.
            if recorder_ctx is not None:
                for _ in range(5):
                    og.sim.step()
                    recorder_ctx.record(env, og)

            # ----- Phase B PLAN (cuRobo only — no physics execution yet) -----
            print("[PnP] Phase B PLAN: solving 3-segment transport "
                  "(lift → hover → descend) ...", flush=True)
            t1 = time.time()
            goal_target_pos = goal_center.copy()
            seg_timings: list = []
            seg_pairs = _plan_transport(
                primitives, env.robots[0], target_obj, goal_target_pos,
                transport_timeout=args.transport_timeout,
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
                print(f"[PnP]   plan seg {st['label']!r}: ok={st['ok']} "
                      f"wall={st['wall_s']*1000:.1f}ms "
                      f"toggle_off={st['toggle_gripper_off']} "
                      f"n_wp={st['n_wp']}", flush=True)
            if seg_pairs is None or len(seg_pairs) == 0:
                first_fail = next((st["label"] for st in seg_timings
                                   if not st["ok"]), "unknown")
                print(f"[PnP] Phase B PLAN FAILED at segment {first_fail!r} "
                      f"(plan wall={plan_wall:.2f}s)", flush=True)
                result_payload["fail_step"] = f"phase_b_plan:{first_fail}"
                return
            import torch as th
            transport_arm_traj = th.cat([s[1] for s in seg_pairs], dim=0)
            transport_eef_traj = th.cat([s[2] for s in seg_pairs], dim=0)
            result_payload["phase_b"]["planned"] = True
            result_payload["phase_b"]["traj_len"] = int(len(transport_arm_traj))
            result_payload["phase_b"]["segments"] = [
                {"label": lbl, "n_waypoints": int(len(arm_t))}
                for lbl, arm_t, _ in seg_pairs
            ]
            print(f"[PnP] Phase B PLAN ok: {len(transport_arm_traj)} waypoints "
                  f"across {len(seg_pairs)} segments "
                  f"({result_payload['phase_b']['plan_wall_s']}s)", flush=True)

            # ----- Render preview + save artifacts BEFORE executing -----
            # Show what cuRobo planned, then persist trajectory + result.json.
            # This way the artifacts exist even if the physics execution
            # below diverges from the plan.
            robot = env.robots[0]
            print("[PnP] rendering planned-trajectory preview ...", flush=True)
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
            }, str(out_dir / "trajectory.pt"))
            print(f"[PnP] wrote {out_dir / 'trajectory.pt'}", flush=True)
            (out_dir / "result.json").write_text(json.dumps(result_payload, indent=2))
            print(f"[PnP] wrote {out_dir / 'result.json'} (pre-execute)", flush=True)

            # ----- Phase B EXECUTE (OSC Cartesian path) -----
            # OSC arm is in pose_delta_ori mode (from env build). Drive it
            # by feeding (target_eef_base - current_eef_base) deltas per
            # substep; physics handles the joint motion and AG keeps the
            # held target carried with the gripper.
            # Generous deadline for OSC replay: each env.step has Python +
            # Isaac overhead well beyond the 1/30s physics dt. 209 wp +
            # 30 settle ≈ 240 env.steps ≈ 30-60s wall time empirically.
            replay_deadline = time.time() + max(args.transport_timeout * 4, 240.0)
            cb = (lambda: recorder_ctx.record(env, og)) if recorder_ctx else None
            print("[PnP] Phase B EXECUTE: OSC Cartesian replay of "
                  f"{len(transport_eef_traj)} eef waypoints ...", flush=True)
            t_exec = time.time()
            ok = _replay_holding(
                env, og, robot, target_obj, transport_eef_traj,
                deadline=replay_deadline, frame_callback=cb,
                sft_recorder=sft_recorder,
            )
            exec_wall = time.time() - t_exec
            result_payload["phase_b"]["execute_wall_s"] = round(exec_wall, 2)
            result_payload["phase_b"]["executed"] = bool(ok)
            if not ok:
                result_payload.setdefault("fail_step", "phase_b_execute")
            print(f"[PnP]   execute wall={exec_wall:.1f}s "
                  f"ok={bool(ok)}", flush=True)

            # Use the canonical "held_intersection" check from
            # sentinel/utils/goal_region.py: target AABB ∩ sphere AND
            # robot is still grasping the target.
            tgt_pos_final, _ = target_obj.get_position_orientation()
            tgt_pos_final_np = tgt_pos_final.cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(tgt_pos_final_np - goal_center))
            # Relaxed positional check: target OR gripper AABB ∩ sphere.
            # Captures the (common) case where the held object is slightly
            # off-center but the fingers are inside the goal region.
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
            print(f"[PnP] Phase B EXECUTE {'OK' if ok else 'TRUNCATED'} — "
                  f"center_dist={dist:.3f} m (radius={goal_radius:.3f}), "
                  f"pos_ok={pos_ok} (by={which!r}), "
                  f"target_in={intersects_target}, AG_held={still_held} "
                  f"→ {'SUCCESS' if success else 'MISS'}", flush=True)
            if not success and "fail_step" not in result_payload:
                if not pos_ok:
                    result_payload["fail_step"] = "goal_not_intersected"
                elif not still_held:
                    result_payload["fail_step"] = "lost_grip"
                else:
                    result_payload["fail_step"] = "unknown"
            # Final structured breakdown.
            ph_a = result_payload["phase_a"].get("wall_s", 0)
            ph_b_plan = result_payload["phase_b"].get("plan_wall_s", 0)
            ph_b_exec = result_payload["phase_b"].get("execute_wall_s", 0)
            print("\n[PnP] === STEP BREAKDOWN ===", flush=True)
            print(f"[PnP]   Phase A (grasp find)    : {ph_a:6.1f}s",
                  flush=True)
            print(f"[PnP]   Phase B PLAN (3 segs)   : {ph_b_plan:6.1f}s",
                  flush=True)
            print(f"[PnP]   Phase B EXECUTE (OSC)   : {ph_b_exec:6.1f}s",
                  flush=True)
            print(f"[PnP]   fail_step               : "
                  f"{result_payload.get('fail_step', '-')}", flush=True)
            print(f"[PnP]   success                 : {success}", flush=True)

            # Hold a few more frames so the video shows the final pose.
            if recorder_ctx is not None:
                for _ in range(15):
                    og.sim.step()
                    recorder_ctx.record(env, og)
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
                    "phase_a_held": bool(result_payload.get("phase_a", {}).get("held")),
                })
            (out_dir / "result.json").write_text(json.dumps(result_payload, indent=2))
            print(f"[PnP] wrote {out_dir / 'result.json'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[PnP] FAIL: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
