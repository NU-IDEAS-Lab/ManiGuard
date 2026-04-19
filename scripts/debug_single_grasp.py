#!/usr/bin/env python3
"""Debug a SINGLE antipodal grasp candidate end-to-end.

Prints every intermediate transformation so we can pinpoint exactly where
the gripper ends up vs where we intended it to go:

  1. Target world pose
  2. Candidate T_local (obj-frame)
  3. Composed T_eef_world (what we hand to IK)
  4. cuRobo IK result (joint positions)
  5. After teleport — actual eef_link world pose
  6. Diff between requested and actual eef pose
  7. Actual fingertip world positions
  8. Distance from each fingertip to the nearest point on the target mesh

Also exports a visualization PNG showing the mesh + candidate frame + actual
gripper pose after teleport.

Usage:
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=... CUDA_VISIBLE_DEVICES=0 \
        python scripts/debug_single_grasp.py \
            --scene-dir datasets/safety-benchmark/clutter_goblet_00 \
            --candidate-idx 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Debug a single grasp candidate end-to-end.")
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--candidate-idx", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/debug_grasp"))
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False

    import omnigibson as og
    import torch as th
    import trimesh

    from sentinel.rl.config import build_config
    from sentinel.rl.resets.grasp_sampler import AntipodalConfig, sample_antipodal_grasps
    from sentinel.rl.resets.mesh_loader import (
        franka_gripper_params,
        gripper_mesh_local_to_eef,
        mesh_from_og_object,
    )
    from sentinel.rl.resets.grasp_collector import _curobo_ik, _mat_to_pose, _pose_to_mat
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )

    print("=" * 70)
    print(f"STEP 1: Build config + load scene")
    print("=" * 70)
    cfg = build_config(args.scene_dir)
    target_name = cfg["task"]["obj_name"]
    print(f"  target: {target_name}")

    env = og.Environment(configs=cfg)
    env.reset()
    target = env.scene.object_registry("name", target_name)
    robot = env.robots[0]
    arm = robot.default_arm
    eef_link = robot.eef_links[arm]

    print(f"  robot: {robot.name}")
    print(f"  grasping_mode: {robot.grasping_mode}")

    tgt_pos, tgt_quat = target.get_position_orientation()
    print(f"\n  target world pos={tgt_pos.tolist()}")
    print(f"  target world quat_xyzw={tgt_quat.tolist()}")

    print("\n" + "=" * 70)
    print(f"STEP 2: Extract mesh + sample antipodal candidates")
    print("=" * 70)
    mesh = mesh_from_og_object(target, use_visual=True)
    print(f"  mesh extents={mesh.extents.tolist()}  bounds={mesh.bounds.tolist()}")

    acfg = AntipodalConfig(
        num_surface_samples=64, num_orientations=16, num_standoff_samples=8,
        top_bias=False, **franka_gripper_params(),
    )
    candidates = sample_antipodal_grasps(mesh, acfg, rng=np.random.default_rng(0))
    print(f"  {len(candidates)} candidates generated")
    # Pick a candidate whose approach points reasonably down in world frame
    # (otherwise cuRobo + Franka wrist likely can't reach it). We check in obj
    # frame; goblet quat is ~identity so obj z ≈ world z.
    tgt_pos_tmp = target.get_position_orientation()[0]
    tgt_quat_tmp = target.get_position_orientation()[1]
    tgt_pos_pre = tgt_pos_tmp.cpu().numpy() if hasattr(tgt_pos_tmp, "cpu") else np.asarray(tgt_pos_tmp)
    tgt_quat_pre = tgt_quat_tmp.cpu().numpy() if hasattr(tgt_quat_tmp, "cpu") else np.asarray(tgt_quat_tmp)
    T_target_pre = _pose_to_mat(tgt_pos_pre, tgt_quat_pre)

    # Score each candidate: prefer strongly downward approach in world
    approach_world_list = []
    for T in candidates:
        T_w = T_target_pre @ T
        approach_world_list.append(T_w[2, 2])  # z-component of approach in world
    approach_world_arr = np.asarray(approach_world_list)
    down_mask = approach_world_arr < -0.5  # strong top-down
    down_indices = np.where(down_mask)[0]
    print(f"  candidates with approach z<-0.5 in world: {len(down_indices)} / {len(candidates)}")
    if args.candidate_idx < len(down_indices):
        pick = down_indices[args.candidate_idx]
    else:
        pick = args.candidate_idx
    print(f"  using candidate idx={pick}")
    T_local = np.asarray(candidates[pick], dtype=np.float64)

    print(f"\n  candidate {args.candidate_idx} T_local:")
    print(f"    position (obj-frame): {T_local[:3, 3].tolist()}")
    print(f"    rotation (obj-frame):")
    for row in T_local[:3, :3]:
        print(f"      {row.tolist()}")
    print(f"    local +Z (approach direction in obj frame): {T_local[:3, 2].tolist()}")
    print(f"    local +Y (finger closing direction in obj frame): {T_local[:3, 1].tolist()}")

    print("\n" + "=" * 70)
    print(f"STEP 3: Compose to world frame")
    print("=" * 70)
    tgt_pos_np = tgt_pos.cpu().numpy() if hasattr(tgt_pos, "cpu") else np.asarray(tgt_pos)
    tgt_quat_np = tgt_quat.cpu().numpy() if hasattr(tgt_quat, "cpu") else np.asarray(tgt_quat)
    T_target_world = _pose_to_mat(tgt_pos_np, tgt_quat_np)
    T_eef_world_requested = T_target_world @ T_local
    req_pos, req_quat_xyzw = _mat_to_pose(T_eef_world_requested)

    print(f"  T_target_world:")
    for row in T_target_world:
        print(f"    {row.tolist()}")
    print(f"\n  T_eef_world_requested:")
    for row in T_eef_world_requested:
        print(f"    {row.tolist()}")
    print(f"\n  requested eef pos (world): {req_pos.tolist()}")
    print(f"  requested eef quat_xyzw:    {req_quat_xyzw.tolist()}")
    print(f"  approach direction (world): {T_eef_world_requested[:3, 2].tolist()}")

    print("\n" + "=" * 70)
    print(f"STEP 4: Fingertip offset check (from OG runtime)")
    print("=" * 70)
    for fname, length in robot.eef_to_fingertip_lengths[arm].items():
        print(f"  {fname}: {length:.4f} m (positive = fingertip is at +z of eef_link)")

    # Predict where fingertip will be if teleport is perfect
    ft_offset = list(robot.eef_to_fingertip_lengths[arm].values())[0]  # both fingers same
    approach_world = T_eef_world_requested[:3, 2]
    predicted_fingertip = req_pos + ft_offset * approach_world
    print(f"  predicted fingertip world (req_pos + {ft_offset} * approach): {predicted_fingertip.tolist()}")
    print(f"  distance fingertip → target center: {np.linalg.norm(predicted_fingertip - tgt_pos_np):.4f} m")

    print("\n" + "=" * 70)
    print(f"STEP 5: cuRobo IK solve")
    print("=" * 70)
    primitives = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
    req_pos_t = th.tensor(req_pos, dtype=th.float32)
    req_quat_t = th.tensor(req_quat_xyzw, dtype=th.float32)
    joint_pos = _curobo_ik(
        primitives._motion_generator, robot, arm,
        req_pos_t, req_quat_t, skip_obstacle_update=False,
    )
    if joint_pos is None:
        print(f"  IK FAILED (unreachable)")
        os._exit(1)
    print(f"  IK success. Joint positions: {joint_pos.tolist()}")

    print("\n" + "=" * 70)
    print(f"STEP 6: Teleport + verify actual eef pose")
    print("=" * 70)
    arm_idx = robot.arm_control_idx[arm]
    gripper_idx = robot.gripper_control_idx[arm]
    open_gripper_q = robot.joint_upper_limits[gripper_idx]
    robot.set_joint_positions(joint_pos, arm_idx)
    robot.set_joint_positions(open_gripper_q, gripper_idx)
    for ctrl in robot._controllers.values():
        ctrl._goal = None
    for _ in range(5):
        og.sim.step()

    actual_pos, actual_quat = eef_link.get_position_orientation()
    actual_pos_np = actual_pos.cpu().numpy() if hasattr(actual_pos, "cpu") else np.asarray(actual_pos)
    actual_quat_np = actual_quat.cpu().numpy() if hasattr(actual_quat, "cpu") else np.asarray(actual_quat)

    pos_diff = np.linalg.norm(actual_pos_np - req_pos)
    print(f"  requested eef pos: {req_pos.tolist()}")
    print(f"  actual    eef pos: {actual_pos_np.tolist()}")
    print(f"  pos error: {pos_diff:.4f} m")
    print(f"  requested eef quat_xyzw: {req_quat_xyzw.tolist()}")
    print(f"  actual    eef quat_xyzw: {actual_quat_np.tolist()}")
    # quaternion angular distance
    q_dot = abs(float(np.dot(req_quat_xyzw, actual_quat_np)))
    q_dot = min(1.0, q_dot)
    q_angle = 2 * np.arccos(q_dot)
    print(f"  orientation error: {np.degrees(q_angle):.2f} deg")

    print("\n" + "=" * 70)
    print(f"STEP 7: Actual fingertip positions after teleport")
    print("=" * 70)
    for fl in robot.finger_links[arm]:
        fp, _ = fl.get_position_orientation()
        fp_np = fp.cpu().numpy() if hasattr(fp, "cpu") else np.asarray(fp)
        dist_to_target = np.linalg.norm(fp_np - tgt_pos_np)
        # Fingertip in target's local frame (mesh is in obj-local, target at identity init)
        # For simplicity: rotate fingertip by inverse of target quat, then translate by -target_pos
        fp_in_obj = fp_np - tgt_pos_np  # Approximate — ignores target rotation (quat near identity for goblet)
        closest_pt, surface_dist, _ = mesh.nearest.on_surface(fp_in_obj.reshape(1, 3))
        sd = float(surface_dist[0])
        print(f"  {fl.name}: pos_world={fp_np.tolist()}")
        print(f"    dist to target CENTER: {dist_to_target:.4f} m")
        print(f"    dist to target SURFACE (mesh.nearest): {sd:.4f} m  {'  (penetrating!)' if sd < -0.001 else ''}")

    print("\n" + "=" * 70)
    print(f"STEP 8: Close gripper + hold (so you can see finger contact)")
    print("=" * 70)
    # Build close-gripper action (zero arm, gripper cmd = +1 = close binary)
    arm_dim = len(robot.arm_action_idx[arm])
    zero_arm = th.zeros(arm_dim, dtype=th.float32)
    # MultiFingerGripperController binary mode (default, inverted=False):
    # target >= 0 → OPEN (upper limit); target < 0 → CLOSE (lower limit).
    # Franka finger joints: lower=0 (closed), upper=0.04 (open).
    close_action = th.zeros(robot.action_dim, dtype=th.float32)
    close_action[robot.arm_action_idx[arm]] = zero_arm
    close_action[robot.gripper_action_idx[arm]] = -1.0

    for step in range(30):
        robot.apply_action(close_action)
        og.sim.step()

    # Use OG's physics-based Touching predicate (reads real PhysX contacts via
    # ContactBodies object state). This is the ground-truth contact check —
    # independent of any fingertip-position math we'd compute by hand.
    from omnigibson.object_states import ContactBodies, Touching
    touching = robot.states[Touching].get_value(target)
    target_contacts = target.states[ContactBodies].get_value()
    contact_names = sorted({c.name if hasattr(c, "name") else str(c) for c in target_contacts})
    gripper_q = robot.get_joint_positions()[gripper_idx].detach().cpu().numpy()
    print(f"  After close: gripper_q={gripper_q.tolist()}")
    print(f"  Touching(robot, target) = {touching}")
    print(f"  target is contacted by {len(contact_names)} bodies:")
    for c in contact_names[:12]:
        print(f"    - {c}")
    if touching:
        shared = sorted({l.name for l in (set(robot.links.values()) & set(target_contacts))})
        print(f"  robot links touching target: {shared}")

    print("\n  Holding closed state for ~60s so you can visually inspect. Ctrl-C to exit.")
    sys.stdout.flush()
    import time
    for _ in range(600):
        robot.apply_action(close_action)
        og.sim.step()
        time.sleep(0.1)
    os._exit(0)


if __name__ == "__main__":
    main()
