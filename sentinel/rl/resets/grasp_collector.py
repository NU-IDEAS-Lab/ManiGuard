"""Physics-validated grasp collector.

Mirrors UW Lab OmniReset's grasp sampling pipeline (``record_grasps.py`` +
``check_grasp_success``) but operates on a full robot arm rather than a
free-floating gripper:

  1. Load antipodal candidates (object-local eef poses) from
     ``grasp_sampler.sample_antipodal_grasps``.
  2. For each candidate, compose with the target's current world pose to
     get an eef world target.
  3. Solve cuRobo IK via ``StarterSemanticActionPrimitives._ik_solver_cartesian_to_joint_space``;
     skip unreachable candidates.
  4. Teleport the arm, reset controller goals so OSC does not snap back,
     open the gripper, settle a few physics steps.
  5. Drive the gripper **closed through the controller** (``robot.apply_action``)
     with ``grasping_mode="physical"`` so real contact friction decides whether
     the grasp holds. No sticky shortcut.
  6. Run an OmniReset-style shake test: every few steps apply a small random
     linear velocity to the target (matches ``global_physics_control_event``'s
     0.01N perturbation), then check multi-factor stability
     (velocity under threshold for N consecutive steps, position drift bounded,
     object above floor) — same criteria as OmniReset's ``check_grasp_success``
     (terminations.py:116-217).
  7. Export the set of validated grasps as rel_position + rel_orientation +
     gripper_qpos (plus arm_joint_pos for fast runtime reset).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class GraspCollectorConfig:
    """Collection hyperparameters. Defaults pulled from OmniReset's
    ``check_grasp_success`` + ``grasp_sampling_cfg`` values."""

    num_target_grasps: int = 100
    max_attempts: int | None = None
    """Cap on candidates to iterate (independent of num_target_grasps).
    None = iterate until all candidates tried."""

    settle_steps: int = 10
    """Zero-action sim steps after teleport, before closing gripper."""

    close_steps: int = 20
    """Sim steps driving the gripper closed (enough for real contact to form)."""

    shake_steps: int = 40
    """Shake-test duration (~4 s at 10 Hz)."""

    shake_interval: int = 4
    """Apply a new random perturbation every N steps (OmniReset: ~every 0.1 s)."""

    shake_force_magnitude: float = 0.01
    """Random linear-velocity magnitude applied to the target each perturbation
    (m/s). OmniReset uses 0.01 N force; at low target mass, velocity impulse
    of the same scale has similar effect."""

    stability_vel_threshold: float = 0.05
    """Object linear velocity must stay below this to count as stable (m/s)."""

    stability_ang_vel_threshold: float = 1.0
    """Object angular velocity threshold (rad/s)."""

    stability_consecutive_steps: int = 5
    """Consecutive stable sim steps required for the candidate to pass."""

    max_pos_drift: float = 0.10
    """Hard cap on target displacement from init pose over the whole test."""

    min_z_height: float = 0.05
    """If target z drops below this (e.g. fell to floor), the grasp failed."""

    ik_timeout: float = 10.0
    """cuRobo IK timeout per candidate (seconds)."""

    max_reach_from_base: float = 1.1
    """Prefilter: skip candidates whose eef world pos is further than this
    from the robot base. ``get_position_orientation`` on FrankaMounted returns
    the pedestal's ground-level base; the arm's link0 is lifted ~0.35 m, so
    effective max reach from ground base is ~0.855 + 0.35 = 1.2 m. We use 1.1 m
    for a conservative cutoff before hitting cuRobo IK."""

    min_approach_down_component: float = -0.3
    """Prefilter: keep candidates whose gripper approach axis (+Z in eef frame)
    has world-z component ≤ this. Negative = pointing downward. Default -0.3
    accepts approaches up to ~70° off vertical; tighten to -0.7 for almost-
    straight-down only. Franka wrist can't reach grippers pointing strongly
    upward — antipodal sampler generates these but they're mostly unreachable."""

    lift_amount: float = 0.15
    """Straight-up world displacement applied to the eef after closing, to
    verify the grasp holds under gravity (object leaves the support surface
    and hangs on the fingers alone). 15 cm clears any typical tabletop."""

    lift_settle_steps: int = 30
    """Sim steps after lifting eef, to let physics integrate whether the
    object follows or falls back."""

    min_lift_rise: float = 0.04
    """Object z must increase by at least this much vs pre-lift position.
    4 cm comfortably clears any support surface (typical table thickness <2 cm)
    and leaves margin for slight slip. Looser than ``lift_amount`` because
    OSC pose_delta tracking doesn't achieve the full commanded displacement
    in a short settle window under load."""


def _pose_to_mat(pos, quat_xyzw) -> np.ndarray:
    """Build 4x4 from (position, quaternion xyzw)."""
    import trimesh.transformations as tra
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    # trimesh expects (w, x, y, z)
    T = tra.quaternion_matrix([q[3], q[0], q[1], q[2]])
    T[:3, 3] = p
    return T


def _mat_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose 4x4 into (position, quaternion xyzw)."""
    import trimesh.transformations as tra
    p = T[:3, 3].copy()
    q_wxyz = tra.quaternion_from_matrix(T)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return p, q_xyzw


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to xyzw quaternion via 4x4 wrap."""
    T = np.eye(4)
    T[:3, :3] = R
    return _mat_to_pose(T)[1]


def _build_action(robot, arm_cmd: "torch.Tensor", gripper_cmd: float) -> "torch.Tensor":
    """Assemble a flat action vector with zero arm delta and binary gripper."""
    import torch as th

    arm = robot.default_arm
    action = th.zeros(robot.action_dim, dtype=th.float32)
    arm_idx = robot.arm_action_idx[arm]
    gripper_idx = robot.gripper_action_idx[arm]
    action[arm_idx] = arm_cmd
    action[gripper_idx] = gripper_cmd
    return action


def _reset_controller_goals(robot) -> None:
    """Zero out controller goals so OSC / IK controllers don't spring the arm
    back to a pre-teleport commanded pose on the next step."""
    for ctrl in robot._controllers.values():
        ctrl._goal = None


def _curobo_ik(motion_gen, robot, arm: str, eef_pos, eef_quat, skip_obstacle_update: bool):
    """Bypass ``StarterSemanticActionPrimitives._ik_solver_cartesian_to_joint_space``
    so we can use ``CuRoboEmbodimentSelection.DEFAULT`` for single-arm robots.

    The primitive wrapper hardcodes ``emb_sel=ARM``, which is only configured for
    multi-embodiment robots (Tiago / R1). FrankaMounted registers only DEFAULT,
    so the wrapper raises ``KeyError(<ARM>)``. We replicate the wrapper's logic
    against ``DEFAULT``.

    Returns: torch.Tensor of arm joint positions, or ``None`` on IK failure.
    """
    import math
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        m as primitives_macros,
    )

    eef_link = robot.eef_link_names[arm]
    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_pos for _ in range(bs)])}
    target_quat = {eef_link: th.stack([eef_quat for _ in range(bs)])}

    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        initial_joint_pos=None,
        is_local=False,
        max_attempts=math.ceil(primitives_macros.MAX_PLANNING_ATTEMPTS / bs),
        timeout=60.0,
        ik_fail_return=primitives_macros.MAX_IK_FAILURES_BEFORE_RETURN,
        enable_finetune_trajopt=False,
        finetune_attempts=0,
        return_full_result=False,
        success_ratio=1.0 / bs,
        attached_obj=None,
        attached_obj_scale=None,
        motion_constraint=None,
        skip_obstacle_update=skip_obstacle_update,
        ik_only=True,
        ik_world_collision_check=False,  # disable for pure-kinematic reachability; physics/shake test catches actual collisions later
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )

    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        return None
    joint_state = joint_states[success_idx[0]]
    joint_pos = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=False, emb_sel=CuRoboEmbodimentSelection.DEFAULT
    )
    manip_idx = th.cat([robot.arm_control_idx[arm]])
    return joint_pos[manip_idx].cpu()


def collect_valid_grasps(
    env,
    target_obj,
    candidates_local: np.ndarray,
    cfg: GraspCollectorConfig = GraspCollectorConfig(),
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Validate antipodal candidates and return those that hold a physical grasp.

    Args:
        env: a running ``og.Environment`` (the robot must be ``env.robots[0]``).
        target_obj: the ``BaseObject`` to grasp.
        candidates_local: ``(N, 4, 4)`` float array from ``sample_antipodal_grasps``
            (eef poses in target-local frame).
        cfg: hyperparameters.
        rng: numpy Generator for reproducible shake perturbations.
        verbose: print per-candidate outcomes.

    Returns:
        List of dicts, each with:
          - ``rel_position``: (3,) eef pos in target-local frame
          - ``rel_orientation_xyzw``: (4,) eef quat xyzw in target-local frame
          - ``gripper_qpos``: (F,) finger joint positions after closing
          - ``arm_joint_pos``: (7,) arm joint config from IK (skips re-solve at runtime)
    """
    import omnigibson as og
    import torch as th
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )

    if rng is None:
        rng = np.random.default_rng(0)

    robot = env.robots[0]
    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]

    # Snapshot initial robot state so we can reset between candidates.
    initial_joint_pos = robot.get_joint_positions().clone()
    init_target_pos, init_target_quat = target_obj.get_position_orientation()
    init_target_pos = init_target_pos.clone() if hasattr(init_target_pos, "clone") else np.asarray(init_target_pos)
    init_target_quat = init_target_quat.clone() if hasattr(init_target_quat, "clone") else np.asarray(init_target_quat)
    T_target_world = _pose_to_mat(
        init_target_pos.cpu().numpy() if hasattr(init_target_pos, "cpu") else init_target_pos,
        init_target_quat.cpu().numpy() if hasattr(init_target_quat, "cpu") else init_target_quat,
    )

    # Lazy cuRobo init (first call is the slow one; subsequent are fast).
    if verbose:
        print(f"  [collector] initializing StarterSemanticActionPrimitives + cuRobo...", flush=True)
    primitives = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
    if verbose:
        robot_pos_w = robot.get_position_orientation()[0]
        target_pos_w = target_obj.get_position_orientation()[0]
        print(f"  [collector] primitives + cuRobo ready", flush=True)
        print(f"  [collector] robot base world pos: {robot_pos_w.tolist()}", flush=True)
        print(f"  [collector] target world pos:     {target_pos_w.tolist()}", flush=True)
        print(f"  [collector] reach (base->target): {float((target_pos_w - robot_pos_w).norm()):.3f} m", flush=True)

    # Open-gripper joint value (upper limit = fingers fully apart for Franka).
    joint_upper = robot.joint_upper_limits
    open_gripper_q = joint_upper[gripper_control_idx].clone()
    zero_arm_cmd = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)

    robot_base_pos = robot.get_position_orientation()[0]
    robot_base_pos_np = (robot_base_pos.cpu().numpy() if hasattr(robot_base_pos, "cpu")
                        else np.asarray(robot_base_pos)).astype(np.float64).reshape(3)

    # Sanity check: can cuRobo solve an IK for a canonical top-down pose
    # 15 cm above the target? If this fails, the IK wiring is broken — no
    # point iterating candidates.
    if verbose:
        import trimesh.transformations as tra
        tgt_np = (init_target_pos.cpu().numpy() if hasattr(init_target_pos, "cpu")
                 else np.asarray(init_target_pos)).astype(np.float64).reshape(3)
        sanity_eef_pos = th.tensor(
            [tgt_np[0], tgt_np[1], tgt_np[2] + 0.15], dtype=th.float32
        )
        # Top-down eef: +Z down in world = rotate identity 180° about X (or Y).
        q_topdown_wxyz = tra.quaternion_about_axis(math.pi, [1.0, 0.0, 0.0])
        sanity_eef_quat = th.tensor(
            [q_topdown_wxyz[1], q_topdown_wxyz[2], q_topdown_wxyz[3], q_topdown_wxyz[0]],
            dtype=th.float32,
        )
        sanity_joint = _curobo_ik(
            primitives._motion_generator, robot, arm,
            sanity_eef_pos, sanity_eef_quat, skip_obstacle_update=False,
        )
        print(f"  [collector] sanity IK (15 cm top-down above target): "
              f"{'OK' if sanity_joint is not None else 'FAIL'}", flush=True)

    valid: list[dict] = []
    total = len(candidates_local)
    iter_cap = cfg.max_attempts if cfg.max_attempts is not None else total
    skipped_unreachable = 0

    for i in range(min(total, iter_cap)):
        if len(valid) >= cfg.num_target_grasps:
            break

        T_local = np.asarray(candidates_local[i], dtype=np.float64)
        T_eef_world = T_target_world @ T_local
        eef_pos_np, eef_quat_np = _mat_to_pose(T_eef_world)

        # Cheap reach prefilter — skip candidates provably beyond arm reach.
        dist_to_base = float(np.linalg.norm(eef_pos_np - robot_base_pos_np))
        if dist_to_base > cfg.max_reach_from_base:
            skipped_unreachable += 1
            continue

        # Orientation prefilter — gripper approach (eef local +Z) must point
        # reasonably downward in world frame for Franka wrist to reach.
        approach_world_z = float(T_eef_world[2, 2])
        if approach_world_z > cfg.min_approach_down_component:
            skipped_unreachable += 1
            continue

        eef_pos = th.tensor(eef_pos_np, dtype=th.float32)
        eef_quat = th.tensor(eef_quat_np, dtype=th.float32)

        # Step 1: restore scene state before trying this candidate.
        robot.set_joint_positions(initial_joint_pos)
        target_obj.set_position_orientation(init_target_pos, init_target_quat)
        target_obj.root_link.set_linear_velocity(th.zeros(3))
        target_obj.root_link.set_angular_velocity(th.zeros(3))
        _reset_controller_goals(robot)
        og.sim.step()  # let physics settle the reset

        # Step 2: IK (direct cuRobo call, using DEFAULT embodiment selection
        # since single-arm FrankaMounted doesn't register the ARM variant).
        if verbose and i < 3:
            print(f"  [{i:4d}/{total}] IK start: eef_pos={eef_pos.tolist()}", flush=True)
        try:
            joint_pos = _curobo_ik(
                primitives._motion_generator, robot, arm,
                eef_pos, eef_quat,
                skip_obstacle_update=(i > 0),
            )
        except Exception as e:
            if verbose:
                print(f"  [{i:4d}/{total}] IK raised {type(e).__name__}: {e}", flush=True)
            continue
        if verbose and i < 3:
            print(f"  [{i:4d}/{total}] IK {'returned joints' if joint_pos is not None else 'unreachable'}", flush=True)
        if joint_pos is None:
            if verbose and i < 10:
                print(f"  [{i:4d}/{total}] IK unreachable")
            continue

        # Step 3: teleport arm + open gripper
        robot.set_joint_positions(joint_pos, arm_control_idx)
        robot.set_joint_positions(open_gripper_q, gripper_control_idx)
        _reset_controller_goals(robot)

        # Step 4: settle with open-gripper, zero-arm action.
        # MultiFingerGripperController binary mode (default): target >= 0 = OPEN
        # (upper limit = fingers apart), target < 0 = CLOSE (lower limit = fingers
        # together). Franka finger joints range [0 closed, 0.04 open].
        settle_action = _build_action(robot, zero_arm_cmd, gripper_cmd=+1.0)  # open
        for _ in range(cfg.settle_steps):
            robot.apply_action(settle_action)
            og.sim.step()

        # Step 5: drive gripper closed through the controller (triggers real contact).
        close_action = _build_action(robot, zero_arm_cmd, gripper_cmd=-1.0)  # close
        for _ in range(cfg.close_steps):
            robot.apply_action(close_action)
            og.sim.step()

        # Step 5.5: PhysX contact gate — if neither finger is actually touching
        # the target after closing, this candidate is a geometric placement
        # that doesn't produce a physical grasp. Skip shake (saves time +
        # avoids false-positive "stable" grasps propped up by the table).
        from omnigibson.object_states import Touching
        if not robot.states[Touching].get_value(target_obj):
            if verbose:
                print(f"  [{i:4d}/{total}] no contact after close, skip", flush=True)
            continue

        # Step 6: LIFT PHASE — drive arm +Z through OSC so real contact forces
        # carry the object. If grasp is holding, target rises with gripper;
        # if object was just propped on the table, it stays while gripper flies.

        target_z_before_lift = float(target_obj.get_position_orientation()[0][2])
        eef_z_before_lift = float(robot.eef_links[arm].get_position_orientation()[0][2])

        # Drive the arm upward via OSC pose-delta (NOT set_joint_positions) so
        # PhysX integrates real contact forces between fingers and the object.
        # Teleporting joints (even in sub-mm steps) is still a discontinuity:
        # object inertia leaves it behind the first frame, contact breaks.
        #
        # OSC pose_delta_ori: first 3 dims = eef [dx, dy, dz] commanded delta.
        # Input semantics depend on command_input_limits; for scene-loaded
        # Franka we send a large positive Z (max-out the up command) and let
        # the controller track over many sim steps.
        # OSC pose_delta_ori accumulates: each apply_action sets goal =
        # current_eef + delta, so a constant delta advances the commanded goal
        # each step. Motion rate is NON-linear in input because PID tracking
        # compresses small values. Measured on Franka + goblet:
        #   input 1.0  → ~27 mm/step, ~82 cm total (too fast, contact breaks)
        #   input 0.02 → ~0.07 mm/step, ~2 mm total (too slow, no meaningful lift)
        # 0.1 is a middle-ground estimate; calibrate empirically per robot/object.
        lift_delta_arm = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)
        lift_delta_arm[2] = 0.1
        lift_action = _build_action(robot, lift_delta_arm, gripper_cmd=-1.0)
        for _ in range(cfg.lift_settle_steps):
            robot.apply_action(lift_action)
            og.sim.step()

        eef_z_after_lift = float(robot.eef_links[arm].get_position_orientation()[0][2])
        eef_rise = eef_z_after_lift - eef_z_before_lift
        if verbose and i < 20:
            print(f"  [{i:4d}/{total}] lift: eef_rise={eef_rise:.3f}m", flush=True)

        lifted_touching = robot.states[Touching].get_value(target_obj)
        target_z_after_lift = float(target_obj.get_position_orientation()[0][2])
        z_rise = target_z_after_lift - target_z_before_lift
        if not lifted_touching or z_rise < cfg.min_lift_rise:
            if verbose:
                reason = "lost contact" if not lifted_touching else f"no rise (z_rise={z_rise:.3f}m)"
                print(f"  [{i:4d}/{total}] LIFT fail: {reason}", flush=True)
            continue

        # Step 7: shake test with OmniReset-style multi-factor stability check.
        shake_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        init_shake_pos = target_obj.get_position_orientation()[0].clone()
        consecutive_stable = 0
        failed = False
        final_drift = 0.0
        final_z = 0.0

        for s in range(cfg.shake_steps):
            if s % cfg.shake_interval == 0:
                d = shake_rng.normal(size=3)
                d /= (np.linalg.norm(d) + 1e-8)
                impulse = th.tensor(d * cfg.shake_force_magnitude, dtype=th.float32)
                target_obj.root_link.set_linear_velocity(impulse)

            robot.apply_action(close_action)
            og.sim.step()

            lin_v = float(th.norm(target_obj.root_link.get_linear_velocity()))
            ang_v = float(th.norm(target_obj.root_link.get_angular_velocity()))
            cur_pos = target_obj.get_position_orientation()[0]
            drift = float(th.norm(cur_pos - init_shake_pos))
            z = float(cur_pos[2])
            final_drift = drift
            final_z = z

            stable = (
                lin_v < cfg.stability_vel_threshold
                and ang_v < cfg.stability_ang_vel_threshold
                and drift < cfg.max_pos_drift
                and z > cfg.min_z_height
            )
            if stable:
                consecutive_stable += 1
            else:
                consecutive_stable = 0

            # Early-exit if object falls — no point continuing.
            if z < cfg.min_z_height - 0.02 or drift > cfg.max_pos_drift * 2:
                failed = True
                break

        passed = (
            not failed
            and consecutive_stable >= cfg.stability_consecutive_steps
            and final_drift < cfg.max_pos_drift
            and final_z > cfg.min_z_height
        )

        if verbose:
            status = "PASS" if passed else "fail"
            print(f"  [{i:4d}/{total}] {status}  drift={final_drift:.3f}m z={final_z:.3f}m "
                  f"stable_consec={consecutive_stable}")

        if passed:
            final_gripper_q = robot.get_joint_positions()[gripper_control_idx].detach().cpu().numpy()
            valid.append({
                "rel_position": T_local[:3, 3].copy().astype(np.float32),
                "rel_orientation_xyzw": _mat_to_quat_xyzw(T_local[:3, :3]).astype(np.float32),
                "gripper_qpos": final_gripper_q.astype(np.float32),
                "arm_joint_pos": (joint_pos.detach().cpu().numpy() if hasattr(joint_pos, "cpu")
                                  else np.asarray(joint_pos)).astype(np.float32),
            })

    if verbose:
        print(f"  [collector] done. valid={len(valid)}  "
              f"unreachable_prefilter={skipped_unreachable}/{min(total, iter_cap)}", flush=True)
    return valid


def save_grasp_dataset(valid: list[dict], path: Path | str, target_name: str) -> None:
    """Serialize validated grasps to a ``.pt`` file matching OmniReset's
    recorder schema: rel_position / rel_orientation / gripper_qpos dicts."""
    import torch as th

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not valid:
        raise ValueError("No valid grasps to save.")

    dataset = {
        "target_name": target_name,
        "rel_position": th.stack([th.as_tensor(g["rel_position"]) for g in valid]),
        "rel_orientation_xyzw": th.stack([th.as_tensor(g["rel_orientation_xyzw"]) for g in valid]),
        "gripper_qpos": th.stack([th.as_tensor(g["gripper_qpos"]) for g in valid]),
        "arm_joint_pos": th.stack([th.as_tensor(g["arm_joint_pos"]) for g in valid]),
    }
    th.save(dataset, str(path))
