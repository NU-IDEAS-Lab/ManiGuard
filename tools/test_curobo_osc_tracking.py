"""Minimal cuRobo + OSC tracking test.

Goal: verify that a cuRobo joint trajectory, after applying OG's canonical
``add_linearly_interpolated_waypoints`` densification, can be tracked by
the OSC controller at one waypoint per ``env.step`` with small tracking
error. No grasping, no AG, no held object — just empty scene + Franka.

Two passes by default:
  A. stream the cuRobo interpolated_plan directly (50 Hz spacing).
  B. densify the joint trajectory to ``max_inter_dist`` rad spacing,
     FK each densified waypoint, stream that.

For each pass we print per-N waypoint tracking error and the final eef
pose error.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m tools.test_curobo_osc_tracking
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/test_curobo_osc"))
    p.add_argument("--max-inter-dist", type=float, default=0.005,
                   help="add_linearly_interpolated_waypoints' max joint "
                        "change between adjacent waypoints (rad).")
    p.add_argument("--target-dpos", type=float, nargs=3, default=[0.0, 0.2, 0.15],
                   help="Target eef pose delta from start (m), in robot base frame.")
    return p.parse_args()


def _init_og():
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    gm.HEADLESS = True
    import omnigibson as og
    return og


def _build_env(og):
    env_cfg = {
        "scene": {"type": "Scene"},
        "robots": [{
            "type": "FrankaPanda",
            "name": "agent_0",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": True,
            "fixed_base": True,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": "physical",
            "self_collisions": True,
            "controller_config": {
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
        "objects": [],
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


def _plan_to_eef_delta(primitives, robot, dpos_base):
    """Plan a single cuRobo trajectory from current pose to (current + dpos)
    keeping eef orientation fixed. Returns the JointState path."""
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    motion_gen = primitives._motion_generator
    arm = robot.default_arm
    eef_link = robot.eef_link_names[arm]
    bs = motion_gen.batch_size

    eef_pos, eef_quat = robot.eef_links[arm].get_position_orientation()
    target_pos_world = eef_pos.clone().float() + th.tensor(dpos_base, dtype=th.float32)
    target_quat_world = eef_quat.clone().float()

    target_pos = {eef_link: th.stack([target_pos_world] * bs)}
    target_quat = {eef_link: th.stack([target_quat_world] * bs)}

    motion_gen.update_obstacles(ignore_objects=[])
    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos, target_quat=target_quat,
        initial_joint_pos=None, is_local=False,
        max_attempts=8, timeout=20.0,
        ik_fail_return=5, enable_finetune_trajopt=True,
        finetune_attempts=2, return_full_result=False,
        success_ratio=1.0 / bs,
        skip_obstacle_update=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        return None, None
    return joint_states[success_idx[0]], (target_pos_world, target_quat_world)


def _eef_traj_from_joint_state(motion_gen, joint_state, eef_link, *, densify=None):
    """Convert a cuRobo JointState path into an (T, 7) eef trajectory in
    robot base frame. If ``densify`` is not None, first densify the joint
    trajectory via add_linearly_interpolated_waypoints, rebuild a
    JointState, then FK that.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    if densify is None:
        eef_dict = motion_gen.path_to_eef_trajectory(
            joint_state, return_axisangle=False,
            emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )
        return eef_dict[eef_link].cpu()

    # Path: joint -> densified joint -> JointState -> FK
    q_full = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    ).cpu().float()
    q_dense = motion_gen.add_linearly_interpolated_waypoints(
        traj=q_full, max_inter_dist=densify,
    )

    # Build a JointState in OG full-joint order, then reorder to the
    # kinematics-side joint subset (locked joints excluded) — same trick
    # OG's curobo wrapper uses everywhere else (cf. check_collisions:
    # `JointState(... joint_names=robot_joint_names).get_ordered_joint_state(
    #     mg[emb_sel].kinematics.joint_names)`).
    from curobo.types.state import JointState
    js_dense = JointState(
        position=q_dense.to(motion_gen.tensor_args.device),
        joint_names=motion_gen.robot_joint_names,
    ).get_ordered_joint_state(
        motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT].kinematics.joint_names
    )
    eef_dict = motion_gen.path_to_eef_trajectory(
        js_dense, return_axisangle=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    return eef_dict[eef_link].cpu(), q_dense


def _quat_canonical(q_xyzw):
    """Return q or -q so that w >= 0 (canonical shortest-path rep)."""
    import torch as th
    if q_xyzw[3].item() < 0:
        return -q_xyzw
    return q_xyzw


def _osc_stream(env, og, robot, eef_traj_base, *, label, deadline,
                pos_threshold=0.005, rot_threshold=0.05,
                max_settle_steps=60,
                inner_substeps_per_wp=1, inner_pos_tol=0.005,
                inner_rot_tol=0.05):
    """One env.step per cuRobo eef waypoint, then settle on the last
    waypoint. Prints per-30 wp tracking error. Returns the final
    (pos_err, rot_err)."""
    import torch as th
    import omnigibson.utils.transform_utils as T

    arm = robot.default_arm
    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]

    def _step_toward(p_b, q_b_xyzw):
        cur_p, cur_q = robot.get_relative_eef_pose(arm)
        cur_p = cur_p.float()
        cur_q = cur_q.float()
        dpos = p_b - cur_p
        # Ensure shortest-path quaternion delta.
        q_target = _quat_canonical(q_b_xyzw)
        q_cur = _quat_canonical(cur_q)
        q_inv = T.quat_inverse(q_cur)
        q_delta = T.quat_multiply(q_target, q_inv)
        q_delta = _quat_canonical(q_delta)
        daa = T.quat2axisangle(q_delta)
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[arm_action_idx] = th.cat([dpos, daa])
        action[gripper_action_idx] = 1.0  # open
        env.step(action)
        return float(th.norm(dpos)), float(th.norm(daa))

    n = len(eef_traj_base)
    print(f"\n[{label}] streaming {n} eef waypoints ...", flush=True)
    last_pos_err = last_rot_err = 0.0
    t0 = time.time()
    for wi in range(n):
        if time.time() > deadline:
            print(f"[{label}]   DEADLINE at wp {wi}/{n}", flush=True)
            return last_pos_err, last_rot_err
        p_b = eef_traj_base[wi, :3].float()
        q_b = eef_traj_base[wi, 3:7].float()
        # Inner loop: up to inner_substeps_per_wp env.steps, break early
        # when within (inner_pos_tol, inner_rot_tol).
        for _ in range(inner_substeps_per_wp):
            if time.time() > deadline:
                break
            last_pos_err, last_rot_err = _step_toward(p_b, q_b)
            if (last_pos_err < inner_pos_tol and
                    last_rot_err < inner_rot_tol):
                break
        if wi % 30 == 0 or wi == n - 1:
            print(f"[{label}]   wp {wi+1}/{n}  "
                  f"pos_err={last_pos_err:.4f} m  "
                  f"rot_err={last_rot_err:.4f} rad", flush=True)

    # Final settle
    p_b = eef_traj_base[-1, :3].float()
    q_b = eef_traj_base[-1, 3:7].float()
    settled_at = max_settle_steps
    for k in range(max_settle_steps):
        if time.time() > deadline:
            break
        last_pos_err, last_rot_err = _step_toward(p_b, q_b)
        if last_pos_err < pos_threshold and last_rot_err < rot_threshold:
            settled_at = k + 1
            break
    wall = time.time() - t0
    print(f"[{label}] DONE  final pos_err={last_pos_err:.4f} m  "
          f"rot_err={last_rot_err:.4f} rad  settle_steps={settled_at}  "
          f"wall={wall:.1f} s", flush=True)
    return last_pos_err, last_rot_err


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    og = _init_og()
    env = _build_env(og)
    robot = env.robots[0]
    arm = robot.default_arm
    eef_link = robot.eef_link_names[arm]
    print(f"[setup] robot at base {robot.get_position_orientation()[0].cpu().tolist()}",
          flush=True)
    eef_p, eef_q = robot.eef_links[arm].get_position_orientation()
    print(f"[setup] eef world pos {eef_p.cpu().tolist()}, "
          f"quat_xyzw {eef_q.cpu().tolist()}", flush=True)

    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )
    print("[setup] initializing cuRobo ...", flush=True)
    primitives = StarterSemanticActionPrimitives(
        env, robot, enable_head_tracking=False,
    )
    motion_gen = primitives._motion_generator
    print("[setup] cuRobo ready.", flush=True)

    print(f"[plan] planning eef delta {args.target_dpos} in base frame ...",
          flush=True)
    joint_state, target_pose = _plan_to_eef_delta(
        primitives, robot, args.target_dpos,
    )
    if joint_state is None:
        print("[plan] FAILED", flush=True)
        return
    print(f"[plan] success: cuRobo path len = "
          f"{joint_state.position.shape}", flush=True)

    # --- Pass A: cuRobo interpolated_plan directly ---
    eef_traj_A = _eef_traj_from_joint_state(
        motion_gen, joint_state, eef_link, densify=None,
    )
    print(f"[A] eef_traj shape {tuple(eef_traj_A.shape)} (cuRobo "
          f"interpolated_plan, ~50 Hz)", flush=True)

    # Reset robot for pass A
    initial_q = robot.get_joint_positions().clone()
    deadline_A = time.time() + 60.0
    pe_A, re_A = _osc_stream(env, og, robot, eef_traj_A,
                             label="A:noinp", deadline=deadline_A)

    # Reset for pass B
    robot.set_joint_positions(initial_q)
    for _ in range(10):
        og.sim.step()

    # --- Pass B: densified joint traj -> FK -> stream ---
    eef_traj_B, q_dense = _eef_traj_from_joint_state(
        motion_gen, joint_state, eef_link, densify=args.max_inter_dist,
    )
    print(f"[B] eef_traj shape {tuple(eef_traj_B.shape)} (densified to "
          f"max_inter_dist={args.max_inter_dist} rad)", flush=True)
    deadline_B = time.time() + 120.0
    pe_B, re_B = _osc_stream(env, og, robot, eef_traj_B,
                             label="B:densify", deadline=deadline_B)

    # Reset for pass C: same cuRobo interpolated_plan as A, but with a
    # per-waypoint inner-loop settle (the v11 pattern).
    robot.set_joint_positions(initial_q)
    for _ in range(10):
        og.sim.step()
    deadline_C = time.time() + 120.0
    pe_C, re_C = _osc_stream(env, og, robot, eef_traj_A,
                             label="C:inner_settle", deadline=deadline_C,
                             inner_substeps_per_wp=6)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  A (cuRobo interpolated_plan,  {len(eef_traj_A)} wp, 1 step/wp): "
          f"pos_err={pe_A:.4f} m, rot_err={re_A:.4f} rad", flush=True)
    print(f"  B (densified joints,           {len(eef_traj_B)} wp, 1 step/wp): "
          f"pos_err={pe_B:.4f} m, rot_err={re_B:.4f} rad", flush=True)
    print(f"  C (cuRobo plan + inner settle, {len(eef_traj_A)} wp, ≤6 steps/wp): "
          f"pos_err={pe_C:.4f} m, rot_err={re_C:.4f} rad", flush=True)


if __name__ == "__main__":
    main()
