"""Minimal end-to-end pick-and-place demo.

Scene: floor + breakfast_table (fixed) + target object on the table +
a single green goal-region sphere offset on the table. No fragiles, no
clutter, no diagnostics-cameras, no replay-empty plumbing.

Pipeline:
  1. OBB sampler → ``collector.collect_valid_grasps`` → first held grasp.
  2. cuRobo plans a 3-segment transport (lift → hover-translate → descend)
     from the held pose so the target ends inside the goal sphere.
     Gripper-link collisions are toggled OFF only for the lift and
     descend segments (where fingers cross the support surface); the
     horizontal hover segment runs with full collisions enabled.
  3. OSC Cartesian replay drives the planned eef trajectory through
     ``robot.eef_links[...]`` via OSC ``pose_delta_ori``.
  4. Success check via ``sentinel.utils.goal_region``:
     ``object_intersects_goal_region(target, spec)`` AND
     ``robot_holds_target(env, target)``.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m tools.minimal_pick_place
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

# Importing ``sentinel`` triggers ``sentinel/__init__.py`` →
# ``_apply_omnigibson_patches()`` which installs the long-finger Franka
# patch (``franka_panda_longfinger`` asset bundle + AG raycast points
# shifted to match the longer fingers). This MUST happen BEFORE we
# import omnigibson, otherwise the Franka class is loaded with stock
# geometry and the OBB sampler's AG-zone constants no longer line up
# with the physical fingers — every grasp candidate fires the dreaded
# `'NoneType' object has no attribute 'view'` from cuRobo trying to
# read joint state for non-existent finger links.
import sentinel  # noqa: F401  (side effect: apply OG patches)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-category", default="teacup")
    p.add_argument("--target-model", default="tfzfam",
                   help="DatasetObject model id (must exist in BEHAVIOR).")
    p.add_argument("--surface-category", default="breakfast_table")
    p.add_argument("--surface-model", default=None,
                   help="None → first available model for the category.")
    p.add_argument("--goal-offset", type=float, nargs=2, default=[0.10, 0.20],
                   help="Goal sphere center xy offset (m) from the target's "
                        "initial xy on the table, in world frame. Default "
                        "places it 10 cm further forward (+x, away from "
                        "robot) and 20 cm to the side (+y) so the goal is "
                        "comfortably inside the Franka's forward workspace.")
    p.add_argument("--goal-radius", type=float, default=0.03)
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick-timeout", type=float, default=300.0)
    p.add_argument("--transport-timeout", type=float, default=60.0)
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/minimal_pick_place"))
    p.add_argument("--gui", action="store_true")
    p.add_argument("--ik-precheck", action="store_true",
                   help="Enable ik_only precheck before Stage 1 trajopt.")
    p.add_argument("--bench-tag", default="run",
                   help="Tag used in result.json + timings filename.")
    p.add_argument("--bench-iterate-all", action="store_true",
                   help="Iterate every candidate even after the first hold "
                        "(apples-to-apples bench).")
    p.add_argument("--lift-height", type=float, default=0.25,
                   help="Vertical lift (m) before horizontal transport. "
                        "Three-segment plan: lift → hover → descend. "
                        "0.25 matches Phase A's pre-grasp standoff.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _init_og(headless: bool):
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if headless:
        gm.HEADLESS = True
    import omnigibson as og
    return og


def _build_env(og, surface_category: str, surface_model: str | None,
               target_category: str, target_model: str):
    """Build env with floor + table (fixed) + target floating in air
    (so we can drop it onto the table at the right z after we know
    the table's top). Robot is mounted on the table top.
    """
    # OSC pose_delta_ori in matched-limits mode (input==output) so action
    # units are raw meters/radians.
    _OSC_LIMITS = ((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
                   ( 0.2,  0.2,  0.2,  0.5,  0.5,  0.5))
    env_cfg = {
        "scene": {"type": "Scene"},
        "robots": [{
            "type": "FrankaPanda",
            "name": "agent_0",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": False,
            "fixed_base": True,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": "assisted",
            "self_collisions": True,
            "controller_config": {
                "arm_0": {
                    "name": "OperationalSpaceController",
                    "command_input_limits": _OSC_LIMITS,
                    "command_output_limits": _OSC_LIMITS,
                },
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
        "objects": [
            {
                "type": "DatasetObject", "name": "support",
                "category": surface_category, "model": surface_model,
                "scale": [1.0, 1.0, 1.0], "fixed_base": True,
                "position": [0.55, 0.0, 0.0], "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject", "name": "target",
                "category": target_category, "model": target_model,
                "scale": [1.0, 1.0, 1.0], "fixed_base": False,
                "position": [0.55, 0.0, 1.0], "orientation": [0, 0, 0, 1],
            },
        ],
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
    }
    env = og.Environment(configs=env_cfg)
    env.reset()
    og.sim.step()
    return env


def _place_target_on_table(env, og, table_obj, target_obj, *,
                           lift_above_top: float = 0.005):
    """Place the target so its bottom sits ``lift_above_top`` above the
    support's top face, centered above the table center."""
    import torch as th
    aabb_min, aabb_max = table_obj.aabb
    top_z = float(aabb_max[2])
    cx = float((aabb_min[0] + aabb_max[0]) * 0.5)
    cy = float((aabb_min[1] + aabb_max[1]) * 0.5)
    # Settle target to compute its own AABB at identity orientation.
    target_obj.set_position_orientation(
        position=th.tensor([cx, cy, top_z + 0.1], dtype=th.float32),
        orientation=th.tensor([0, 0, 0, 1], dtype=th.float32),
    )
    for _ in range(3):
        og.sim.step()
    t_min, t_max = target_obj.aabb
    center_to_bottom = float(target_obj.get_position_orientation()[0][2]
                             - t_min[2])
    new_z = top_z + lift_above_top + center_to_bottom
    target_obj.set_position_orientation(
        position=th.tensor([cx, cy, new_z], dtype=th.float32),
        orientation=th.tensor([0, 0, 0, 1], dtype=th.float32),
    )
    target_obj.root_link.set_linear_velocity(th.zeros(3, dtype=th.float32))
    target_obj.root_link.set_angular_velocity(th.zeros(3, dtype=th.float32))
    target_obj.root_link.disable_gravity()
    for _ in range(5):
        og.sim.step()
    return (cx, cy, top_z)


def _mount_robot_on_table(robot, top_z: float, table_cx: float, table_cy: float,
                          *, back_offset: float = -0.25):
    """Place the Franka 2 mm above the table top, set back from the
    table center (so the eef faces the target)."""
    import torch as th
    robot.set_position_orientation(
        position=th.tensor([table_cx + back_offset, table_cy,
                            top_z + 0.002], dtype=th.float32),
        orientation=th.tensor([0, 0, 0, 1], dtype=th.float32),
    )


def _build_goal_spec(target_xy, top_z, offset_xy, radius):
    """Build a minimal GoalRegionSpec (only the fields the checker needs
    are populated; bookkeeping fields are zeros)."""
    from sentinel.utils.goal_region import GoalRegionSpec
    cx, cy = target_xy
    ox, oy = offset_xy
    gx = float(cx + ox)
    gy = float(cy + oy)
    gz = float(top_z + radius)  # sphere center sits one radius above the table
    return GoalRegionSpec(
        mode="held_intersection",
        shape="sphere",
        family="table",
        target_name="target",
        support_name="support",
        marker_name="goal_region__target",
        center_world=(gx, gy, gz),
        radius_m=float(radius),
        color_rgba=(0.10, 0.80, 0.20, 0.60),
        target_width_m=0.0,
        anchor_local_xy=(0.0, 0.0),
        pack_bbox_robot_local_xy=((0.0, 0.0), (0.0, 0.0)),
        support_bounds_robot_local_xy=((0.0, 0.0), (0.0, 0.0)),
        clamped_to_support_bounds=False,
    )


# ---------------------------------------------------------------------------
# Phase A — find a held grasp
# ---------------------------------------------------------------------------


def _phase_a_pick(env, og, primitives, target_obj, args, deadline):
    import torch as th
    from sentinel.rl.grasps.collector import (
        GraspCollectorConfig, collect_valid_grasps,
    )
    from sentinel.rl.grasps.mesh import mesh_from_og_object
    from sentinel.rl.grasps.obb_sampler import (
        OBBConfig, sample_obb_assisted_grasps,
    )

    robot = env.robots[0]
    init_pos, init_quat = target_obj.get_position_orientation()
    init_pos = init_pos.clone(); init_quat = init_quat.clone()
    target_obj.root_link.disable_gravity()

    mesh = mesh_from_og_object(target_obj, use_visual=True)
    rng = np.random.default_rng(args.seed)
    candidates, scores = sample_obb_assisted_grasps(
        mesh, config=OBBConfig(max_candidates=args.max_candidates), rng=rng,
    )
    print(f"[A] {len(candidates)} OBB candidates", flush=True)
    if len(candidates) == 0:
        return None

    primitives._motion_generator.update_obstacles(ignore_objects=[])
    cfg = GraspCollectorConfig(
        num_target_grasps=10_000 if args.bench_iterate_all else 1,
        ik_precheck=bool(args.ik_precheck),
    )
    held = []
    def _on(ci, result):
        if result is not None:
            held.append(result)
            print(f"[A] cand {ci}: HELD "
                  f"(obj_to_eef={result['obj_to_eef']:.3f} m)", flush=True)

    timings_log: list = []
    out = collect_valid_grasps(
        env, robot, primitives, target_obj,
        init_pos, init_quat,
        candidates_local=candidates, cfg=cfg,
        deadline=deadline, on_progress=_on, verbose=True,
        timings_log=timings_log,
    )
    return (out[0] if out else None), timings_log


# ---------------------------------------------------------------------------
# Phase B PLAN — 3-segment cuRobo transport (lift → hover → descend)
# ---------------------------------------------------------------------------


def _solve_segment(motion_gen, robot, target_obj, eef_link,
                   eef_target_pos, eef_target_quat, initial_q, *,
                   timeout, label):
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_target_pos] * bs)}
    target_quat = {eef_link: th.stack([eef_target_quat] * bs)}
    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos, target_quat=target_quat,
        initial_joint_pos=initial_q, is_local=False,
        max_attempts=8, timeout=timeout, ik_fail_return=5,
        enable_finetune_trajopt=True, finetune_attempts=2,
        return_full_result=False, success_ratio=1.0 / bs,
        attached_obj=None, attached_obj_scale=None,
        motion_constraint=None, skip_obstacle_update=True,
        ik_only=False, ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        print(f"[B] segment {label!r}: cuRobo FAILED", flush=True)
        return None, None
    js = joint_states[success_idx[0]]
    full_traj = motion_gen.path_to_joint_trajectory(
        js, get_full_js=True, emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    final_full = full_traj if full_traj.dim() == 1 else full_traj[-1]
    eef_dict = motion_gen.path_to_eef_trajectory(
        js, return_axisangle=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    eef_traj = eef_dict[eef_link].cpu()
    print(f"[B] segment {label!r}: ok ({len(eef_traj)} eef waypoints)",
          flush=True)
    return eef_traj, final_full


def _plan_transport(primitives, robot, target_obj, goal_center, *,
                    transport_timeout, lift_height: float = 0.25,
                    seg_timings: list | None = None):
    """Three-segment cuRobo transport: lift → hover-translate → descend.

    Collision strategy: gripper-link collisions are toggled OFF only for
    the segments whose start or end has the fingers near/below the
    support surface (lift, descend). The horizontal hover segment runs
    with FULL collisions enabled — by then the held target is
    `lift_height` above the table, well clear of typical tabletop
    obstacles, so we get a genuinely safe long horizontal plan instead
    of feasibility-by-suppression.
    """
    import torch as th
    from sentinel.rl.grasps.collector import (
        _FRANKA_GRIPPER_COLLISION_LINKS, _patch_curobo_mimic_lookup,
        _pose_to_mat, _mat_to_pose,
    )
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    _patch_curobo_mimic_lookup()
    motion_gen = primitives._motion_generator
    arm = robot.default_arm
    eef_link = robot.eef_link_names[arm]

    # Held offset: where the eef sits relative to the target. We freeze the
    # target's orientation to identity for the goal mapping; this preserves
    # the held relative-pose throughout the transport.
    eef_pos_now, eef_quat_now = robot.eef_links[arm].get_position_orientation()
    tgt_pos_now, _ = target_obj.get_position_orientation()
    T_eef = _pose_to_mat(eef_pos_now.cpu().numpy(), eef_quat_now.cpu().numpy())
    T_tgt = _pose_to_mat(tgt_pos_now.cpu().numpy(),
                         np.array([0, 0, 0, 1.0]))
    T_eef_in_tgt = np.linalg.inv(T_tgt) @ T_eef

    tgt_now_np = tgt_pos_now.cpu().numpy().astype(np.float64)
    goal_np = np.asarray(goal_center, dtype=np.float64)

    # Three target-xyz waypoints. Lift = current target xy at +lift_height.
    # Hover = goal xy at the same lift altitude. Descend = goal.
    wp_lift = tgt_now_np.copy(); wp_lift[2] += lift_height
    wp_hover = goal_np.copy();   wp_hover[2] = tgt_now_np[2] + lift_height
    wp_descend = goal_np.copy()
    print(f"[B] tgt_now={tgt_now_np.tolist()} goal={goal_np.tolist()}  "
          f"lift_h={lift_height:.3f}m", flush=True)
    print(f"[B]   wp_lift   ={wp_lift.tolist()}", flush=True)
    print(f"[B]   wp_hover  ={wp_hover.tolist()}", flush=True)
    print(f"[B]   wp_descend={wp_descend.tolist()}", flush=True)

    def _eef_goal_for_tgt(tgt_xyz):
        T_tgt_goal = T_tgt.copy()
        T_tgt_goal[:3, 3] = tgt_xyz
        return _mat_to_pose(T_tgt_goal @ T_eef_in_tgt)

    motion_gen.update_obstacles(ignore_objects=[target_obj])
    raw_mg = motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT]

    seg_timeout = max(15.0, transport_timeout / 3.0)
    plan_spec = [
        # (label, target_xyz, toggle_gripper_collision_off)
        ("lift",    wp_lift,    True),   # start has fingers in contact regime
        ("hover",   wp_hover,   False),  # fully above table — collisions ON
        ("descend", wp_descend, True),   # end has fingers in contact regime
    ]
    segments: list[tuple[str, "th.Tensor"]] = []
    chain_init = None
    for label, tgt_xyz, toggle_off in plan_spec:
        ep, eq = _eef_goal_for_tgt(tgt_xyz)
        t_seg = time.time()
        if toggle_off:
            raw_mg.toggle_link_collision(
                list(_FRANKA_GRIPPER_COLLISION_LINKS), False)
        try:
            eef_traj, final_full = _solve_segment(
                motion_gen, robot, target_obj, eef_link,
                th.tensor(ep, dtype=th.float32),
                th.tensor(eq, dtype=th.float32),
                chain_init, timeout=seg_timeout, label=label,
            )
        finally:
            if toggle_off:
                raw_mg.toggle_link_collision(
                    list(_FRANKA_GRIPPER_COLLISION_LINKS), True)
        seg_wall = time.time() - t_seg
        if seg_timings is not None:
            seg_timings.append({
                "label": label, "wall_s": seg_wall,
                "toggle_gripper_off": toggle_off,
                "ok": eef_traj is not None,
                "n_wp": int(len(eef_traj)) if eef_traj is not None else 0,
            })
        if eef_traj is None:
            return None
        segments.append((label, eef_traj))
        chain_init = final_full

    return segments


# ---------------------------------------------------------------------------
# Phase B EXECUTE — OSC Cartesian replay
# ---------------------------------------------------------------------------


def _q_canon(q):
    import torch as th
    return -q if float(q[3].item()) < 0.0 else q


def _osc_replay(env, og, robot, eef_traj_base, *, deadline,
                substeps_per_wp=6, pos_tol=0.005, rot_tol=0.05,
                settle_steps=200):
    import torch as th
    import omnigibson.utils.transform_utils as T
    arm = robot.default_arm
    a_idx = robot.arm_action_idx[arm]
    g_idx = robot.gripper_action_idx[arm]

    def _step(tp, tq):
        cp, cq = robot.get_relative_eef_pose(arm)
        dpos = tp - cp.float()
        q_t = _q_canon(tq)
        q_c = _q_canon(cq.float())
        q_d = _q_canon(T.quat_multiply(q_t, T.quat_inverse(q_c)))
        daa = T.quat2axisangle(q_d)
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[a_idx] = th.cat([dpos, daa])
        action[g_idx] = -1.0
        env.step(action)
        return float(th.norm(dpos)), float(th.norm(daa))

    n = len(eef_traj_base)
    pe = re = 0.0
    print(f"[E] {n} eef waypoints (≤{substeps_per_wp} substeps/wp) "
          f"+ {settle_steps}-step final settle", flush=True)
    for wi in range(n):
        if time.time() > deadline:
            print(f"[E] DEADLINE at wp {wi}/{n}", flush=True)
            return False
        tp = eef_traj_base[wi, :3].float()
        tq = eef_traj_base[wi, 3:7].float()
        for k in range(substeps_per_wp):
            pe, re = _step(tp, tq)
            if pe < pos_tol and re < rot_tol:
                break
        if wi % 30 == 0 or wi == n - 1:
            print(f"[E]   wp {wi+1}/{n}  pe={pe:.4f} m  re={re:.4f} rad",
                  flush=True)

    tp = eef_traj_base[-1, :3].float()
    tq = eef_traj_base[-1, 3:7].float()
    for k in range(settle_steps):
        if time.time() > deadline:
            return False
        pe, re = _step(tp, tq)
        if pe < pos_tol and re < rot_tol:
            print(f"[E] settled after {k+1} steps  pe={pe:.4f} re={re:.4f}",
                  flush=True)
            return True
    print(f"[E] settle cap hit  pe={pe:.4f} re={re:.4f}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    og = _init_og(headless=not args.gui)
    env = _build_env(og, args.surface_category, args.surface_model,
                     args.target_category, args.target_model)
    try:
        robot = env.robots[0]
        table = env.scene.object_registry("name", "support")
        target = env.scene.object_registry("name", "target")

        # Settle target on table, mount robot.
        cx, cy, top_z = _place_target_on_table(env, og, table, target)
        _mount_robot_on_table(robot, top_z, cx, cy)
        for _ in range(8):
            og.sim.step()
        print(f"[scene] table top z={top_z:.3f}, target xy=({cx:.3f},{cy:.3f})",
              flush=True)

        # Goal sphere offset from the target on the same table.
        from sentinel.utils.goal_region import (
            object_intersects_goal_region, robot_holds_target,
            spawn_goal_region_marker,
        )
        spec = _build_goal_spec((cx, cy), top_z, args.goal_offset,
                                args.goal_radius)
        spawn_goal_region_marker(env, spec)
        og.sim.step()
        print(f"[scene] goal_center={spec.center_world}  "
              f"radius={spec.radius_m:.3f}", flush=True)

        # cuRobo.
        from omnigibson.action_primitives.starter_semantic_action_primitives \
            import StarterSemanticActionPrimitives
        print("[setup] initializing cuRobo ...", flush=True)
        primitives = StarterSemanticActionPrimitives(
            env, robot, enable_head_tracking=False,
        )
        print("[setup] cuRobo ready", flush=True)

        # Phase A.
        t0 = time.time()
        grasp, timings_log = _phase_a_pick(env, og, primitives, target, args,
                                           t0 + args.pick_timeout)
        phase_a_wall = time.time() - t0
        # Aggregate per-candidate timings.
        n_cands = len(timings_log)
        n_precheck = sum(1 for t in timings_log if "ik_precheck_s" in t)
        n_precheck_ok = sum(1 for t in timings_log if t.get("ik_precheck_ok"))
        n_stage1 = sum(1 for t in timings_log if "stage1_s" in t)
        n_stage1_ok = sum(1 for t in timings_log if t.get("stage1_ok"))
        n_stage2 = sum(1 for t in timings_log if "stage2_s" in t)
        n_held = sum(1 for t in timings_log if t.get("held"))
        sum_pc = sum(t.get("ik_precheck_s", 0.0) for t in timings_log)
        sum_s1 = sum(t.get("stage1_s", 0.0) for t in timings_log)
        sum_s2 = sum(t.get("stage2_s", 0.0) for t in timings_log)
        sum_tot = sum(t.get("total_s", 0.0) for t in timings_log)
        print(f"[A] phase-a wall={phase_a_wall:.1f}s  "
              f"cands={n_cands}  held={n_held}", flush=True)
        print(f"[A]   ik_precheck: ran={n_precheck} ok={n_precheck_ok} "
              f"total={sum_pc:.2f}s "
              f"avg={sum_pc/max(n_precheck,1)*1000:.1f}ms", flush=True)
        print(f"[A]   stage1    : ran={n_stage1} ok={n_stage1_ok} "
              f"total={sum_s1:.2f}s "
              f"avg={sum_s1/max(n_stage1,1)*1000:.1f}ms", flush=True)
        print(f"[A]   stage2    : ran={n_stage2}            "
              f"total={sum_s2:.2f}s "
              f"avg={sum_s2/max(n_stage2,1)*1000:.1f}ms", flush=True)
        print(f"[A]   per-cand total avg="
              f"{sum_tot/max(n_cands,1)*1000:.1f}ms", flush=True)
        # Dump per-candidate timings for offline comparison.
        timings_path = args.out_dir / f"timings_{args.bench_tag}.json"
        timings_path.write_text(json.dumps({
            "ik_precheck_enabled": bool(args.ik_precheck),
            "phase_a_wall_s": phase_a_wall,
            "n_cands": n_cands, "n_held": n_held,
            "n_ik_precheck_ran": n_precheck, "n_ik_precheck_ok": n_precheck_ok,
            "n_stage1_ran": n_stage1, "n_stage1_ok": n_stage1_ok,
            "n_stage2_ran": n_stage2,
            "sum_ik_precheck_s": sum_pc, "sum_stage1_s": sum_s1,
            "sum_stage2_s": sum_s2, "sum_total_s": sum_tot,
            "per_cand": timings_log,
        }, indent=2))
        print(f"[A]   wrote {timings_path}", flush=True)
        if args.bench_iterate_all:
            print("[bench] iterate-all mode → skipping Phase B", flush=True)
            return
        if grasp is None:
            print("[FAIL] no held grasp", flush=True)
            return

        # Phase B PLAN.
        t1 = time.time()
        seg_timings: list = []
        eef_segs = _plan_transport(
            primitives, robot, target, spec.center_world,
            transport_timeout=args.transport_timeout,
            lift_height=args.lift_height,
            seg_timings=seg_timings,
        )
        plan_wall = time.time() - t1
        for st in seg_timings:
            print(f"[B]   seg {st['label']!r}: ok={st['ok']} "
                  f"wall={st['wall_s']*1000:.1f}ms "
                  f"toggle_off={st['toggle_gripper_off']} "
                  f"n_wp={st['n_wp']}", flush=True)
        if eef_segs is None:
            print(f"[FAIL] cuRobo transport plan failed "
                  f"(plan wall={plan_wall:.2f}s)", flush=True)
            phaseb_path = args.out_dir / f"phaseb_timings_{args.bench_tag}.json"
            phaseb_path.write_text(json.dumps({
                "ok": False, "plan_wall_s": plan_wall,
                "lift_height_m": args.lift_height,
                "segments": seg_timings,
            }, indent=2))
            print(f"[B]   wrote {phaseb_path}", flush=True)
            return
        import torch as th
        full_eef = th.cat([s for _, s in eef_segs], dim=0)
        sum_seg = sum(st["wall_s"] for st in seg_timings)
        print(f"[B] plan wall={plan_wall:.2f}s "
              f"(sum-segs={sum_seg:.2f}s)  "
              f"{len(full_eef)} waypoints across {len(eef_segs)} segments",
              flush=True)
        phaseb_path = args.out_dir / f"phaseb_timings_{args.bench_tag}.json"
        phaseb_path.write_text(json.dumps({
            "ok": True, "plan_wall_s": plan_wall,
            "sum_seg_wall_s": sum_seg,
            "lift_height_m": args.lift_height,
            "n_wp_total": int(len(full_eef)),
            "segments": seg_timings,
        }, indent=2))
        print(f"[B]   wrote {phaseb_path}", flush=True)

        # Phase B EXECUTE.
        ok = _osc_replay(env, og, robot, full_eef,
                         deadline=time.time() + 240.0)

        # Goal check via the canonical helpers.
        intersects = object_intersects_goal_region(target, spec)
        still_held = robot_holds_target(env, target)
        tp_final, _ = target.get_position_orientation()
        tp_np = tp_final.cpu().numpy().astype(np.float64).tolist()
        dist = float(np.linalg.norm(
            np.array(tp_np) - np.array(spec.center_world)))
        success = bool(ok and intersects and still_held)
        print(f"\n[RESULT] success={success}  "
              f"AABB∩sphere={intersects}  AG_held={still_held}  "
              f"center_dist={dist:.3f} m  "
              f"target_world={tp_np}", flush=True)

        result = {
            "target": {"category": args.target_category,
                       "model": args.target_model},
            "goal_center_world": list(spec.center_world),
            "goal_radius_m": spec.radius_m,
            "phase_a_held": True,
            "phase_b_executed": bool(ok),
            "target_intersects_goal": bool(intersects),
            "robot_holds_target": bool(still_held),
            "center_dist_m": dist,
            "final_target_world": tp_np,
            "success": success,
        }
        (args.out_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[done] wrote {args.out_dir / 'result.json'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
