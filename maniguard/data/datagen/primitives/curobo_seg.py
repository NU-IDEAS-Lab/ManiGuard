"""Solve one cuRobo motion segment — Layer-1 primitive (family-agnostic).

Given a configured motion generator + a world-frame eef goal pose + the current
full joint state, return a collision-free joint trajectory to that goal.

Replicated clean from ``pick_and_place_from_dataset._solve_one_segment`` (the
reusable solver the old code already cross-imported into other scripts — its
family-agnostic core). KEPT: the salvage logic that recovers trajectories cuRobo's
trajopt flags ``success=False`` but which actually converged within tol
(5 mm / 0.03 rad). DROPPED: the ``eef_traj`` (OSC-replay leftover — datagen executes
joints directly), the ``[PnP transport]`` labels, and the always-on failure re-probe
(now opt-in via ``diagnose_on_fail``). ADDED seams: ``attach_obj`` (carry a held
object so the planner avoids collisions with its geometry too) + ``motion_constraint``
(the partial-pose / linear-servo safety levers P3/P7 drive).

This function ONLY solves: the motion generator, its obstacle world, and any
constraint state must already be configured by the caller (P7). It does not toggle
gripper collisions, choose waypoints, or update obstacles — those are the Layer-2
high-level motion + Layer-1 obstacle/constraint concerns, kept out on purpose (the
old ``_plan_transport`` tangled them in, which is the orchestration we are NOT
inheriting).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SegmentResult:
    """One solved segment. ``final_full`` chains into the next segment's
    ``initial_joint_pos``; ``arm_traj`` (T,7) is fed to the JointController."""

    arm_traj: Any              # (T, 7) arm joint waypoints (cpu tensor)
    final_full: Any            # full-DoF final joint state (for chaining)
    salvaged: bool             # accepted via salvage (cuRobo success flag was False)
    pos_err: float | None
    rot_err: float | None
    n_waypoints: int


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


def _salvage(full, pos_tol: float, rot_tol: float, label: str):
    """Pick the first batch path that succeeded, or that converged within tol
    despite ``success=False``. Returns ``(joint_state|None, salvaged, pos_err, rot_err)``.
    """
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
                (pos_err is None or pos_err < pos_tol)
                and (rot_err is None or rot_err < rot_tol)
            )
            if accepted:
                salvaged = not is_succ
                if salvaged:
                    print(f"[datagen.curobo] segment {label!r}: salvage "
                          f"iter#{i} batch#{j} (pos_err={pos_err}, rot_err={rot_err})",
                          flush=True)
                return path_js, salvaged, pos_err, rot_err
    return None, False, None, None


def _diagnose(motion_gen, target_pos, target_quat, initial_joint_pos, bs,
              timeout, attached_obj, attached_obj_scale) -> None:
    """Opt-in failure probe: re-run with full results to print the per-batch
    MotionGenStatus + pos/rot errors (IK Fail vs Trajopt Fail vs Start-In-Collision).
    Cheap on warmed kernels; no behaviour change (caller already has None)."""
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

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

    try:
        full = motion_gen.compute_trajectories(
            target_pos=target_pos, target_quat=target_quat,
            initial_joint_pos=initial_joint_pos, is_local=False,
            max_attempts=1, timeout=timeout, ik_fail_return=5,
            enable_finetune_trajopt=False, finetune_attempts=1,
            return_full_result=True, success_ratio=1.0 / bs,
            attached_obj=attached_obj, attached_obj_scale=attached_obj_scale,
            motion_constraint=None, skip_obstacle_update=True, ik_only=False,
            ik_world_collision_check=True, emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )
        for i, r in enumerate(full):
            print(f"[datagen.curobo]   diag #{i}: status={getattr(r, 'status', '?')!r} "
                  f"pos_err={_fmt(getattr(r, 'position_error', None))} "
                  f"rot_err={_fmt(getattr(r, 'rotation_error', None))}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[datagen.curobo]   diag probe raised {type(exc).__name__}: {exc}", flush=True)


def solve_segment(motion_gen, robot, eef_goal_pos, eef_goal_quat, initial_joint_pos, *,
                  timeout: float, max_attempts: int = 12,
                  attach_obj=None, motion_constraint=None,
                  eef_link: str | None = None, label: str = "",
                  pos_tol: float = 0.005, rot_tol: float = 0.03,
                  ik_rot_relax: float | None = None, ik_pos_relax: float | None = None,
                  diagnose_on_fail: bool = False) -> SegmentResult | None:
    """Plan one collision-free segment to ``(eef_goal_pos, eef_goal_quat)`` (world
    frame) from ``initial_joint_pos`` (full-DoF). Returns a :class:`SegmentResult`
    or ``None`` on failure.

    ``attach_obj`` (OG object, optional) is attached to the gripper so the planner
    avoids collisions with the carried geometry too. ``motion_constraint`` is a
    cuRobo ``PoseCostMetric`` (e.g. partial-pose hold / linear servo). The obstacle
    world is assumed pre-configured by the caller (``skip_obstacle_update=True``).

    For a deliberate straight push INTO contact (no trajopt, no collision avoidance) use
    :func:`solve_ik` per interpolated waypoint instead."""
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    if eef_link is None:
        eef_link = robot.eef_link_names[robot.default_arm]

    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_goal_pos] * bs)}
    target_quat = {eef_link: th.stack([eef_goal_quat] * bs)}
    attached_obj = {eef_link: attach_obj.root_link} if attach_obj is not None else None
    attached_obj_scale = {eef_link: 1.0} if attach_obj is not None else None

    # ik_rot_relax / ik_pos_relax: temporarily widen cuRobo's IK convergence rotation_threshold (rad) /
    # position_threshold (m) just for THIS plan, restored in finally below. cuRobo bakes
    # rotation_threshold=0.05 + position_threshold=0.005 at MotionGen construction; the IK success gate
    # reads them at call time (ik_solver._get_success -> self.rotation_threshold / self.position_threshold),
    # so mutating the live solver attributes is the per-call lever (there is NO MotionGenPlanConfig field
    # for them). Needed for a far-reach handle grasp at the arm's reach ENVELOPE, where the best IK solution
    # sits a hair past the strict gate on BOTH axes (measured: pos 0.0052-0.0058 m, rot 0.043-0.057 rad) —
    # whichever binds first IK_FAILs (no solution -> no trajectory). At a STANDOFF pre-grasp the sub-cm /
    # few-degree residual is harmless (the next SERVO re-aims from the LIVE handle pose). We widen the
    # salvage tols to match so the trajopt path (which still optimises to ~that residual) is kept.
    ik_solver = motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT].ik_solver
    _saved_rot_thresh = _saved_pos_thresh = None
    salvage_rot_tol, salvage_pos_tol = rot_tol, pos_tol
    if ik_rot_relax is not None:
        _saved_rot_thresh = ik_solver.rotation_threshold
        ik_solver.rotation_threshold = max(_saved_rot_thresh, ik_rot_relax)
        salvage_rot_tol = max(rot_tol, ik_rot_relax)
    if ik_pos_relax is not None:
        _saved_pos_thresh = ik_solver.position_threshold
        ik_solver.position_threshold = max(_saved_pos_thresh, ik_pos_relax)
        salvage_pos_tol = max(pos_tol, ik_pos_relax)

    # return_full_result=True so the salvage pass can recover trajectories trajopt
    # flags success=False but which actually converged at the goal (short/degenerate
    # motions where its convergence criterion is overly strict).
    try:
        full = motion_gen.compute_trajectories(
            target_pos=target_pos, target_quat=target_quat,
            initial_joint_pos=initial_joint_pos, is_local=False,
            max_attempts=max_attempts, timeout=timeout, ik_fail_return=5,
            enable_finetune_trajopt=True, finetune_attempts=2,
            return_full_result=True, success_ratio=1.0 / bs,
            attached_obj=attached_obj, attached_obj_scale=attached_obj_scale,
            motion_constraint=motion_constraint, skip_obstacle_update=True,
            ik_only=False, ik_world_collision_check=True,
            emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )
    except TypeError as exc:
        # cuRobo (Stanford fork) crashes building the failure result when a
        # hold_partial_pose query is invalid: `[False for _ in batch_size]` where
        # batch_size is an int (motion_gen.py plan_batch). It only fires when
        # motion_constraint is set + update_pose_cost_metric rejects the query
        # (start/goal orientation mismatch or motion off the free axis). Treat an
        # invalid constrained query as a plan failure (None) instead of crashing
        # collection — the caller seeds constrained segments to stay valid.
        if motion_constraint is not None:
            print(f"[datagen.curobo] segment {label!r}: invalid partial-pose "
                  f"constraint query -> None ({exc})", flush=True)
            return None
        raise
    finally:
        if _saved_rot_thresh is not None:
            ik_solver.rotation_threshold = _saved_rot_thresh
        if _saved_pos_thresh is not None:
            ik_solver.position_threshold = _saved_pos_thresh

    joint_state, salvaged, pos_err, rot_err = _salvage(full, salvage_pos_tol, salvage_rot_tol, label)
    if joint_state is None:
        print(f"[datagen.curobo] segment {label!r}: FAILED (0/{int(bs)} successes)",
              flush=True)
        if diagnose_on_fail:
            _diagnose(motion_gen, target_pos, target_quat, initial_joint_pos,
                      bs, timeout, attached_obj, attached_obj_scale)
        return None

    manip_idx = robot.arm_control_idx[robot.default_arm]
    jp = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=False, emb_sel=CuRoboEmbodimentSelection.DEFAULT)
    arm_traj = (jp[manip_idx].unsqueeze(0) if jp.dim() == 1 else jp[:, manip_idx]).cpu()

    full_traj = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=True, emb_sel=CuRoboEmbodimentSelection.DEFAULT)
    final_full = full_traj if full_traj.dim() == 1 else full_traj[-1]

    print(f"[datagen.curobo] segment {label!r}: ok ({len(arm_traj)} waypoints)"
          f"{' (salvaged)' if salvaged else ''}", flush=True)
    return SegmentResult(arm_traj=arm_traj, final_full=final_full, salvaged=salvaged,
                         pos_err=pos_err, rot_err=rot_err, n_waypoints=len(arm_traj))


def solve_ik(motion_gen, robot, eef_goal_pos, eef_goal_quat, initial_joint_pos, *,
             timeout: float, max_attempts: int = 8, ik_collision: bool = False,
             eef_link: str | None = None, label: str = "") -> SegmentResult | None:
    """Single-pose IK to ``(eef_goal_pos, eef_goal_quat)`` (world) from ``initial_joint_pos``
    (full-DoF). Returns a :class:`SegmentResult` whose ``arm_traj`` is the 1-waypoint IK config
    (or ``None`` if no IK solution). This is the building block for a straight Cartesian SERVO:
    interpolate the eef line and IK each pose with ``ik_collision=False`` so the solve goes
    straight INTO contact (a deliberate push the collision-avoiding planner would refuse).

    Unlike :func:`solve_segment` this uses cuRobo's IK solver (``ik_only`` => ``solve_ik_batch``),
    which returns an ``IKResult`` (no ``get_paths`` / trajopt salvage) — so we read the IK joint
    state directly from the ``return_full_result=False`` ``(successes, joint_states)`` form."""
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    if eef_link is None:
        eef_link = robot.eef_link_names[robot.default_arm]
    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_goal_pos] * bs)}
    target_quat = {eef_link: th.stack([eef_goal_quat] * bs)}
    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos, target_quat=target_quat,
        initial_joint_pos=initial_joint_pos, is_local=False,
        max_attempts=max_attempts, timeout=timeout, ik_fail_return=5,
        return_full_result=False, success_ratio=1.0 / bs,
        ik_only=True, ik_world_collision_check=ik_collision,
        skip_obstacle_update=True, emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    manip_idx = robot.arm_control_idx[robot.default_arm]
    seed_arm = initial_joint_pos[manip_idx].detach().cpu().reshape(-1)

    def _arm_of(js):
        jp = motion_gen.path_to_joint_trajectory(
            js, get_full_js=False, emb_sel=CuRoboEmbodimentSelection.DEFAULT)
        return (jp[manip_idx] if jp.dim() == 1 else jp[:, manip_idx]).detach().cpu().reshape(-1)

    # Pick the valid IK branch CLOSEST to the seed (the previous servo waypoint), NOT cuRobo's
    # "first valid" — for a redundant 7-DoF arm the first-valid solution can jump to a far branch
    # (wrist/elbow skew) for nearly the same eef pose, which is the unnatural sideways twist seen on
    # a straight servo lift. Nearest-to-seed keeps the per-waypoint configs continuous.
    best_arm, best_d = None, None
    for i in range(len(joint_states)):
        try:
            ok = bool(successes[i].item())
        except Exception:  # noqa: BLE001
            ok = bool(successes[i])
        if not ok or joint_states[i] is None:
            continue
        arm_i = _arm_of(joint_states[i])
        d = float((arm_i - seed_arm).norm())
        if best_d is None or d < best_d:
            best_d, best_arm = d, arm_i
    if best_arm is None:
        print(f"[datagen.curobo] ik {label!r}: no IK solution", flush=True)
        return None

    # NOTE: get_full_js=True crashes cuRobo here ("lock_joints is also listed in self.joint_names")
    # for an IK solution, so we do NOT compute final_full — the caller seeds the next IK by
    # splicing this arm config into the prior full-DoF state.
    return SegmentResult(arm_traj=best_arm.reshape(1, -1), final_full=None, salvaged=False,
                         pos_err=None, rot_err=None, n_waypoints=1)
