"""Shared sim/cuRobo helpers + grasp validator.

Two layers:

* low-level helpers (pose <-> matrix, action assembly, controller goal
  reset, Franka-tolerant cuRobo motion plan) — used by the rest of
  the grasp pipeline and from outside if needed.
* ``GraspCollectorConfig`` + ``collect_valid_grasps`` + ``save_grasp_dataset``
  — the search-only side of the per-object pipeline. Takes
  ``(N, 4, 4)`` candidate eef poses (in target-local frame) from any
  source (currently GraspGen), returns the subset that survives a
  full physics validation: cuRobo motion plan → close → assisted-grasp
  fire check → gravity hold → eef-distance check. Returned dicts have
  the format ``GraspDatasetResetter`` (sentinel/rl/grasps/reset.py)
  consumes, plus a saved ``approach_traj`` so the video renderer can
  replay the cuRobo path without re-solving IK.

The validator never captures frames — that's the renderer's job. This
keeps the search loop fast and lets ``render_grasps.py`` re-render any
saved grasp at video time without re-running physics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import torch


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


def _build_action(robot, arm_cmd: "torch.Tensor",
                  gripper_cmd: float) -> "torch.Tensor":
    """Assemble a flat action vector with zero arm delta and binary gripper."""
    import torch as th

    arm = robot.default_arm
    action = th.zeros(robot.action_dim, dtype=th.float32)
    arm_idx = robot.arm_action_idx[arm]
    gripper_idx = robot.gripper_action_idx[arm]
    action[arm_idx] = arm_cmd
    action[gripper_idx] = gripper_cmd
    return action


def _all_joint_controllers(robot) -> bool:
    """True iff every controller on the robot is a JointController in
    absolute-position mode (use_delta_commands=False). When this holds,
    ``robot.q_to_action`` is callable and we can hand it an arbitrary
    full-DoF joint setpoint to build an action."""
    from omnigibson.controllers.joint_controller import JointController
    for ctrl in robot._controllers.values():
        if not isinstance(ctrl, JointController):
            return False
        if getattr(ctrl, "use_delta_commands", False):
            return False
    return True


def _build_hold_action(robot, gripper_open: bool) -> "torch.Tensor":
    """Build an "arm-stationary + gripper-{open,close}" action.

    Three controller setups handled:

    * **OSC arm + any gripper**: arm-action slot = zero (no pose delta).
    * **All JointController** (arm + gripper): full-DoF
      ``q_to_action(current_arm_q + gripper_at_limit)``.
    * **JointController arm + MultiFingerGripper**: arm slot = current
      arm joint positions (so the absolute-position controller holds
      the arm where it is), gripper slot = ±1.0.

    Gripper command is +1.0 (open) or -1.0 (close); both
    MultiFingerGripperController (smooth mode) and JointController
    interpret ±1.0 as joint upper/lower limit.
    """
    import torch as th
    from omnigibson.controllers.joint_controller import JointController

    arm = robot.default_arm
    gripper_idx = robot.gripper_action_idx[arm]
    arm_idx = robot.arm_action_idx[arm]
    gripper_cmd = +1.0 if gripper_open else -1.0

    if _all_joint_controllers(robot):
        q_full = robot.get_joint_positions().clone()
        gripper_ctrl_idx = robot.gripper_control_idx[arm]
        if gripper_open:
            q_full[gripper_ctrl_idx] = robot.joint_upper_limits[gripper_ctrl_idx]
        else:
            q_full[gripper_ctrl_idx] = robot.joint_lower_limits[gripper_ctrl_idx]
        return robot.q_to_action(q_full)

    arm_ctrl = robot._controllers[f"arm_{arm}"]
    is_joint_arm = (isinstance(arm_ctrl, JointController)
                    and not getattr(arm_ctrl, "use_delta_commands", False))
    if is_joint_arm:
        # Mixed: JointController arm + (non-JointController) gripper.
        # Arm slot holds the current joint positions; gripper slot is ±1.
        arm_ctrl_idx = robot.arm_control_idx[arm]
        cur_q = robot.get_joint_positions().clone()
        return _build_action(robot,
                             cur_q[arm_ctrl_idx].to(th.float32),
                             gripper_cmd)

    # OSC fallback — zero arm delta + scalar gripper command.
    zero_arm = th.zeros(len(arm_idx), dtype=th.float32)
    return _build_action(robot, zero_arm, gripper_cmd)


def _reset_controller_goals(robot) -> None:
    """Zero out controller goals so OSC / IK controllers don't spring the arm
    back to a pre-teleport commanded pose on the next step."""
    for ctrl in robot._controllers.values():
        ctrl._goal = None


_CUROBO_MIMIC_PATCHED = False


def _patch_curobo_mimic_lookup() -> None:
    """Make cuRobo's mimic-joint resolution tolerant of missing source joints.

    cuRobo (Stanford fork) calls ``js.joint_names.index(j)`` for every
    mimic source joint ``j`` defined in the robot config. For Franka,
    ``panda_finger_joint1`` is a mimic source (panda_finger_joint2
    mimics it), but OG's default joint state passed into
    ``compute_trajectories`` doesn't always carry the finger joint
    (gripper DOFs are handled separately by ``MultiFingerGripperController``),
    so ``index()`` raises ``ValueError: 'panda_finger_joint1' is not in
    list`` and the whole motion plan errors out.

    The patch fills any missing joints with sane defaults instead of
    raising. For Franka fingers, default to 0.04 m (fully open) so the
    motion plan validates the correct gripper geometry; collision spheres
    of a closed gripper are too thin and pass through obstacles a real
    open gripper would clip.
    """
    global _CUROBO_MIMIC_PATCHED
    if _CUROBO_MIMIC_PATCHED:
        return
    from curobo.types.state import JointState
    import torch as _th
    _orig_reindex = JointState.inplace_reindex

    _MISSING_JOINT_FILLS = {
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    def inplace_reindex_tolerant(self, joint_names):
        if self.joint_names is None:
            raise ValueError("joint names are not specified in JointState")
        missing = [j for j in joint_names if j not in self.joint_names]
        if not missing:
            return _orig_reindex(self, joint_names)
        device = self.position.device
        fill_shape = list(self.position.shape)
        fill_shape[-1] = 1
        for j in missing:
            val = _MISSING_JOINT_FILLS.get(j, 0.0)
            col = _th.full(fill_shape, val, device=device,
                           dtype=self.position.dtype)
            self.position = _th.cat([self.position, col], dim=-1)
        if self.velocity is not None:
            self.velocity = _th.cat(
                [self.velocity,
                 _th.zeros(list(self.position.shape[:-1]) + [len(missing)],
                           device=device, dtype=self.velocity.dtype)],
                dim=-1)
        if self.acceleration is not None:
            self.acceleration = _th.cat(
                [self.acceleration,
                 _th.zeros(list(self.position.shape[:-1]) + [len(missing)],
                           device=device, dtype=self.acceleration.dtype)],
                dim=-1)
        if getattr(self, "jerk", None) is not None:
            self.jerk = _th.cat(
                [self.jerk,
                 _th.zeros(list(self.position.shape[:-1]) + [len(missing)],
                           device=device, dtype=self.jerk.dtype)],
                dim=-1)
        self.joint_names = list(self.joint_names) + missing
        return _orig_reindex(self, joint_names)

    JointState.inplace_reindex = inplace_reindex_tolerant
    _CUROBO_MIMIC_PATCHED = True


# Franka panda gripper link names — used to disable collision on the
# in-contact leg so the fingers can close around the target.
# Names must match the URDF that cuRobo loaded for the Franka asset.
_FRANKA_GRIPPER_COLLISION_LINKS = (
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)


def _curobo_ik_fast(motion_gen, robot, arm, eef_pos, eef_quat,
                    skip_obstacle_update,
                    pregrasp_standoff_m: float = 0.25,
                    ik_precheck: bool = False,
                    single_stage: bool = False,
                    timings: dict | None = None):
    """Two-stage cuRobo plan: obstacle-aware approach to a pre-grasp
    standoff, then a linear-constrained continuation to the grasp pose
    with the Franka gripper links removed from the collision world.

    Why this shape:
      * Phase A's caller keeps the target in cuRobo's obstacle world.
        The OBB sampler designs the grasp pose with the open gripper
        ~2 mm clear of the target — but cuRobo's Franka panda collision
        spheres have ~13 mm effective radius and span the chord midline.
        So planning all the way to the grasp pose with the full robot
        collision model registers as colliding with the target every
        time. The "right" answer is a pre-grasp standoff + linear servo
        with gripper-link collision disabled on the last segment.
      * cuRobo's own ``plan_grasp`` API does exactly this internally,
        but on CUDA 11.x its goal-type switch (plan_single ↔ plan_goalset
        within a single call) trips the cuda-graph reset path that only
        works on CUDA >= 12, so we re-implement the same logic using
        the ``compute_trajectories`` path that OmniGibson's wrapper has
        already warmed up cleanly. Single goal type throughout, no
        graph switching.

    Returns ``(T, J)`` torch tensor of arm-joint waypoints (stage 1 +
    stage 2 stitched), or ``None`` on failure.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    _patch_curobo_mimic_lookup()

    eef_link = robot.eef_link_names[arm]
    bs = motion_gen.batch_size

    # Compute the pre-grasp standoff once (also used by Stage 1 below).
    # In single-stage mode, the "standoff" IS the grasp pose — Stage 1
    # plans directly to it with full collision and there is no Stage 2.
    T_grasp = _pose_to_mat(eef_pos.cpu().numpy(), eef_quat.cpu().numpy())
    approach_w = T_grasp[:3, 2]
    if single_stage:
        standoff_pos_np = eef_pos.cpu().numpy()
    else:
        standoff_pos_np = eef_pos.cpu().numpy() - pregrasp_standoff_m * approach_w
    standoff_pos_t = th.tensor(standoff_pos_np, dtype=eef_pos.dtype)

    # --- Precheck: ik_only at the STANDOFF with world-collision ON.
    # Same goal Stage 1 reaches via trajopt, but skips the trajopt cost.
    # If even IK to the standoff is infeasible, the collision-aware trajopt
    # downstream will also fail — fast-reject here.
    if ik_precheck:
        t_pc0 = time.time()
        pc_pos = {eef_link: th.stack([standoff_pos_t for _ in range(bs)])}
        pc_quat = {eef_link: th.stack([eef_quat for _ in range(bs)])}
        pc_successes, _pc_js = motion_gen.compute_trajectories(
            target_pos=pc_pos, target_quat=pc_quat,
            initial_joint_pos=None, is_local=False,
            max_attempts=4, timeout=5.0, ik_fail_return=2,
            enable_finetune_trajopt=False, finetune_attempts=0,
            return_full_result=False, success_ratio=1.0 / bs,
            attached_obj=None, attached_obj_scale=None,
            motion_constraint=None,
            skip_obstacle_update=skip_obstacle_update,
            ik_only=True, ik_world_collision_check=True,
            emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )
        ik_ok = bool(th.any(pc_successes).item())
        if timings is not None:
            timings["ik_precheck_s"] = time.time() - t_pc0
            timings["ik_precheck_ok"] = ik_ok
        if not ik_ok:
            return None

    # --- Stage 1: full collision-aware plan to the pre-grasp standoff
    # (standoff_pos_t computed above).
    t_s1 = time.time()
    target_pos = {eef_link: th.stack([standoff_pos_t for _ in range(bs)])}
    target_quat = {eef_link: th.stack([eef_quat for _ in range(bs)])}
    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        initial_joint_pos=None,
        is_local=False,
        max_attempts=16,
        timeout=60.0,
        ik_fail_return=4,
        enable_finetune_trajopt=True,
        finetune_attempts=2,
        return_full_result=False,
        success_ratio=1.0 / bs,
        attached_obj=None,
        attached_obj_scale=None,
        motion_constraint=None,
        skip_obstacle_update=skip_obstacle_update,
        ik_only=False,
        ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    if timings is not None:
        timings["stage1_s"] = time.time() - t_s1
    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        if timings is not None:
            timings["stage1_ok"] = False
        return None
    if timings is not None:
        timings["stage1_ok"] = True
    joint_state = joint_states[success_idx[0]]
    joint_pos = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    manip_idx = th.cat([robot.arm_control_idx[arm]])
    if joint_pos.dim() == 1:
        stage1_arm = joint_pos[manip_idx].unsqueeze(0).cpu()
    else:
        stage1_arm = joint_pos[:, manip_idx].cpu()

    # In single-stage mode, Stage 1 already reached the grasp pose with
    # full collision — there is no linear servo to append.
    if single_stage:
        return stage1_arm

    # Capture Stage 1's FINAL full-DoF joint state — this is the seed for
    # Stage 2 so trajopt plans the linear segment from the standoff (where
    # the arm actually lands after Stage 1) to the grasp pose, NOT from
    # home. Without this, Stage 2's hold_partial_pose constraint compares
    # home_quat vs grasp_quat and fails with "Partial orientation between
    # start and goal is not equal" → TypeError on update_pose_cost_metric.
    joint_pos_full = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    stage1_end_full = (joint_pos_full if joint_pos_full.dim() == 1
                       else joint_pos_full[-1])

    # --- Stage 2: linear-constrained continuation from standoff to grasp,
    # with Franka gripper links REMOVED from the collision world so the
    # final segment can clamp around the target.
    #
    # ``motion_constraint`` is forwarded by the OG wrapper as a
    # PoseCostMetric(hold_partial_pose=True, hold_vec_weight=...). The
    # [0.1, 0.1, 0.1, 0.1, 0.1, 0.0] weights (orientation, then position)
    # with zero on the approach (z) axis tell trajopt to hold the relative
    # orientation + perpendicular position constant while letting the
    # arm advance along the approach axis only — a pure linear servo.
    t_s2 = time.time()
    raw_mg = motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT]
    if hasattr(raw_mg, "toggle_link_collision"):
        raw_mg.toggle_link_collision(list(_FRANKA_GRIPPER_COLLISION_LINKS), False)
    try:
        target_pos = {eef_link: th.stack([eef_pos for _ in range(bs)])}
        target_quat = {eef_link: th.stack([eef_quat for _ in range(bs)])}
        g_successes, g_joint_states = motion_gen.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=stage1_end_full,
            is_local=False,
            max_attempts=8,
            timeout=30.0,
            ik_fail_return=4,
            enable_finetune_trajopt=True,
            finetune_attempts=2,
            return_full_result=False,
            success_ratio=1.0 / bs,
            attached_obj=None,
            attached_obj_scale=None,
            motion_constraint=[0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            skip_obstacle_update=True,
            ik_only=False,
            ik_world_collision_check=True,
            emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )
    finally:
        if hasattr(raw_mg, "toggle_link_collision"):
            raw_mg.toggle_link_collision(list(_FRANKA_GRIPPER_COLLISION_LINKS), True)
    if timings is not None:
        timings["stage2_s"] = time.time() - t_s2

    g_success_idx = th.where(g_successes)[0].cpu()
    if len(g_success_idx) == 0:
        # Stage 1 reached but stage 2 (linear segment into the target)
        # failed. Return stage 1 only — close+hold downstream will fail
        # the AG check, but the trajectory itself is valid.
        return stage1_arm
    g_joint_state = g_joint_states[g_success_idx[0]]
    g_joint_pos = motion_gen.path_to_joint_trajectory(
        g_joint_state, get_full_js=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    if g_joint_pos.dim() == 1:
        stage2_arm = g_joint_pos[manip_idx].unsqueeze(0).cpu()
    else:
        stage2_arm = g_joint_pos[:, manip_idx].cpu()

    return th.cat([stage1_arm, stage2_arm], dim=0)


# ---------------------------------------------------------------------------
# Validator: physics-checks an array of candidate eef poses, returns the
# subset that survives a real grasp.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraspCollectorConfig:
    """Hyperparameters for ``collect_valid_grasps``.

    Defaults match the values render_grasps.py was running with for the
    GraspGen pipeline (default panda fingers, no support surface). Adjust
    via constructor; everything is plain data so safe to pickle.
    """

    ik_precheck: bool = False
    """If True, run an ik_only=True query at the grasp pose (with world
    collision OFF) before launching the Stage 1 trajopt. Fast-rejects
    candidates that have no IK solution at all, skipping the expensive
    trajopt cost on dead candidates."""

    single_stage_grasp: bool = False
    """If True, skip the standoff + linear-servo stages and plan a single
    cuRobo trajectory directly to the grasp pose with full collision
    checking (gripper links INCLUDED). Useful when the two-stage approach
    produces phantom plans through thin objects that then fail the AG
    engage check downstream — single-stage rejects those plans up-front
    by treating gripper-target overlap as a collision."""

    num_target_grasps: int = 10
    """Stop iterating candidates once this many holding grasps have been
    collected. Larger numbers give the RL reset more variety but cost
    proportional sim time per object."""

    settle_open_steps: int = 8
    """Sim steps with gripper open + target hard-pinned, after the cuRobo
    waypoint replay finishes. Lets PhysX stabilise before close."""

    close_steps: int = 20
    """Sim steps driving the gripper closed (target still hard-pinned).
    Long enough for assisted-grasp's contact-ray check to fire."""

    gravity_hold_steps: int = 30
    """Sim steps with gravity re-enabled and the target unpinned. Real
    grasps survive; phantom AG fires let the target slip out."""

    max_reach: float = 0.95
    """Skip candidates whose eef world position is further than this from
    the robot base. Cheap rejection before invoking cuRobo (Franka panda
    arm reach is ~0.855 m; 0.95 leaves margin for pose noise)."""

    max_obj_to_eef_after_hold: float = 0.15
    """After the gravity-hold settles, the target's distance to the
    eef link must be at most this much for the grasp to count. Catches
    phantom AG fires where ``is_grasping`` returns True but the target
    has been flung away from the gripper by the constraint solver."""


def _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin: bool):
    """Single physics step with the target velocity-zeroed (and optionally
    position-pinned) so gravity does not affect it.

    ``hard_pin=True`` resets the target's pose to ``(init_pos, init_quat)``
    after the step so finger close can reliably engage AG even when
    contact forces would otherwise push the target out of the AG ray's
    reach. The phantom-contact "grasps" this lets through are weeded out
    by the gravity-hold check below.

    ``hard_pin=False`` is for the approach motion: target is stationary
    against gravity, but contact resolution is left untouched because
    the gripper isn't touching it yet.
    """
    import omnigibson as og
    import torch as th

    target_obj.root_link.set_linear_velocity(th.zeros(3, dtype=th.float32))
    target_obj.root_link.set_angular_velocity(th.zeros(3, dtype=th.float32))
    og.sim.step()
    target_obj.root_link.set_linear_velocity(th.zeros(3, dtype=th.float32))
    target_obj.root_link.set_angular_velocity(th.zeros(3, dtype=th.float32))
    if hard_pin:
        target_obj.set_position_orientation(position=init_pos, orientation=init_quat)


def run_grasp_attempt(
    env, robot, target_obj,
    init_pos, init_quat,
    joint_traj,
    cfg: GraspCollectorConfig,
    open_gripper_q, zero_arm_cmd,
    arm_control_idx, gripper_control_idx,
    initial_joint_pos,
    deadline: float,
    frame_callback=None,
    verbose_idx: int | None = None,
) -> dict | None:
    """Replay ``joint_traj`` + close + gravity-hold and return a saved-grasp
    dict if the grasp survives, else ``None``.

    Shared kernel for two callers: :func:`_try_grasp_candidate` (Phase A,
    no video) gets a fresh cuRobo trajectory; :mod:`render_grasps` Phase B
    re-runs a previously saved trajectory with a frame-capture callback.

    Args:
        joint_traj: ``(T, 7)`` torch tensor or numpy array of arm-joint
            waypoints. Must end at the desired grasp eef pose.
        frame_callback: optional callable invoked once after every simulator
            step. Receives a single ``phase`` string ('init', 'approach',
            'settle_open', 'close', 'gravity_hold'). Phase B uses it to push
            frames into a buffer; SFT recording uses it to pick the right
            gripper_cmd per step.
        verbose_idx: when set, prints a stage-specific reject reason
            (``AG didn't engage`` / ``phase2 drop``).
    """
    import omnigibson as og
    import torch as th
    import numpy as np
    from omnigibson.controllers.controller_base import IsGraspingState

    arm = robot.default_arm

    def _cb(phase: str) -> None:
        if frame_callback is None:
            return
        # Backward compatibility: callers that registered a 0-arg callback
        # (e.g. render_grasps._capture) should accept the phase kwarg via a
        # default. New callers should declare ``def fn(phase): ...``.
        frame_callback(phase)

    # Reset for this attempt — clears any stale AG bond, returns arm to home,
    # snaps target back to its init pose. Target gravity stays ON; no
    # hard pin during approach (gravity holds the target on the surface
    # while the gripper approaches dynamically).
    robot.release_grasp_immediately(arm)
    robot.set_joint_positions(initial_joint_pos)
    target_obj.set_position_orientation(position=init_pos, orientation=init_quat)
    target_obj.root_link.set_linear_velocity(th.zeros(3))
    target_obj.root_link.set_angular_velocity(th.zeros(3))
    _reset_controller_goals(robot)
    # A few settle steps so the target rests on the surface under gravity
    # before we start moving the gripper.
    for _ in range(4):
        og.sim.step()
    _cb("init")

    # Stage 1: dynamic approach via JointController tracking cuRobo's
    # joint trajectory. We DON'T densify — instead we hold each cuRobo
    # waypoint as the controller target for up to N env.steps until
    # joint error is below tol. JointController PD-tracks each target;
    # under impedance gains it may take a few env.steps to fully
    # converge. Each env.step = 4 physics substeps at our default
    # (120 Hz physics / 30 Hz action) = ~33 ms wall-time of physics
    # simulation per action.
    arm_traj_torch = (joint_traj if isinstance(joint_traj, th.Tensor)
                      else th.as_tensor(joint_traj, dtype=th.float32))

    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]
    WP_TOL_RAD = 0.02      # accept arrival within ~1.1°/joint norm
    WP_MAX_STEPS = 8       # cap per-waypoint substeps; 8 × 33ms = 264ms

    # Track per-waypoint convergence so we can diagnose tracking issues.
    wp_substeps_used: list[int] = []
    wp_final_err: list[float] = []
    for wi, waypoint in enumerate(arm_traj_torch):
        if time.time() > deadline:
            return None
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[arm_action_idx] = waypoint.to(action.dtype)
        action[gripper_action_idx] = 1.0
        # Final waypoint: tighter tol + more substeps. AG needs the
        # fingers exactly where the OBB sampler placed them (~2mm
        # clearance) or its raycasts miss.
        is_final = (wi == len(arm_traj_torch) - 1)
        tol = 0.005 if is_final else WP_TOL_RAD
        max_substeps = 20 if is_final else WP_MAX_STEPS
        substeps_used = max_substeps
        err = float("inf")
        for k in range(max_substeps):
            env.step(action)
            _cb("approach")
            cur_q = robot.get_joint_positions()[arm_control_idx]
            err = float((cur_q - waypoint.to(cur_q.dtype).to(cur_q.device))
                        .norm().item())
            if err < tol:
                substeps_used = k + 1
                break
            if time.time() > deadline:
                return None
        wp_substeps_used.append(substeps_used)
        wp_final_err.append(err)

    # Approach diagnostics: how well did we track the cuRobo joint plan?
    if verbose_idx is not None and len(wp_final_err) > 0:
        avg_substeps = float(np.mean(wp_substeps_used))
        max_err = float(max(wp_final_err))
        final_err = wp_final_err[-1]
        # Compute eef-space error at planned final pose vs achieved.
        planned_final_q = arm_traj_torch[-1]
        cur_q = robot.get_joint_positions()[arm_control_idx]
        joint_diff = (cur_q - planned_final_q.to(cur_q.dtype).to(cur_q.device))
        joint_l1 = float(joint_diff.abs().sum().item())
        joint_max = float(joint_diff.abs().max().item())
        eef_pos, eef_quat = robot.eef_links[arm].get_position_orientation()
        print(f"    cand {verbose_idx}: approach tracking — "
              f"avg_substeps={avg_substeps:.1f}/{WP_MAX_STEPS}, "
              f"max_q_err={max_err:.4f}, final_q_err_norm={final_err:.4f} rad, "
              f"final_q_err_L1={joint_l1:.4f}, final_q_err_max_joint="
              f"{joint_max:.4f}", flush=True)
        print(f"    cand {verbose_idx}: eef-after-approach world pos="
              f"{eef_pos.cpu().numpy().tolist()}", flush=True)

    # Stage 2: settle (HOLD the planned final waypoint, gripper open)
    # then close. Important: hold the PLANNED final joint pose, not
    # the current robot pose — otherwise the controller locks in any
    # undershoot and the fingers won't be at the grasp pose AG expects.
    final_waypoint = arm_traj_torch[-1]
    settle_open_action = th.zeros(robot.action_dim, dtype=th.float32)
    settle_open_action[arm_action_idx] = final_waypoint.to(settle_open_action.dtype)
    settle_open_action[gripper_action_idx] = 1.0
    for _ in range(cfg.settle_open_steps):
        if time.time() > deadline:
            return None
        env.step(settle_open_action)
        _cb("settle_open")

    # Post-settle: how close did we get to the planned final pose?
    if verbose_idx is not None:
        cur_q = robot.get_joint_positions()[arm_control_idx]
        joint_diff = (cur_q - final_waypoint.to(cur_q.dtype).to(cur_q.device))
        joint_max = float(joint_diff.abs().max().item())
        joint_l2 = float(joint_diff.norm().item())
        eef_pos_now, eef_quat_now = robot.eef_links[arm].get_position_orientation()
        print(f"    cand {verbose_idx}: post-settle — "
              f"q_err_L2={joint_l2:.4f}, q_err_max_joint={joint_max:.4f}, "
              f"eef_pos={eef_pos_now.cpu().numpy().tolist()}",
              flush=True)

    close_action = settle_open_action.clone()
    close_action[gripper_action_idx] = -1.0
    for _ in range(cfg.close_steps):
        if time.time() > deadline:
            return None
        env.step(close_action)
        _cb("close")

    if robot.is_grasping(arm, target_obj) != IsGraspingState.TRUE:
        if verbose_idx is not None:
            print(f"    cand {verbose_idx}: AG didn't engage after close",
                  flush=True)
        return None

    # Stage 3: gravity hold — extra steps to verify the grasp survives
    # dynamic forces (no pin, no gravity-toggle since gravity is always
    # on now).
    # Gravity hold target = planned final waypoint with gripper closed
    # (NOT current arm pose) so the controller keeps holding the grasp
    # configuration even if dynamic forces nudge the arm.
    for _ in range(cfg.gravity_hold_steps):
        if time.time() > deadline:
            return None
        env.step(close_action)
        _cb("gravity_hold")

    still_grasping = robot.is_grasping(arm, target_obj) == IsGraspingState.TRUE
    obj_to_eef = float(
        (target_obj.get_position_orientation()[0]
         - robot.eef_links[arm].get_position_orientation()[0]).norm()
    )
    if not (still_grasping and obj_to_eef <= cfg.max_obj_to_eef_after_hold):
        if verbose_idx is not None:
            print(f"    cand {verbose_idx}: phase2 drop "
                  f"(still_grasping={still_grasping}, "
                  f"obj_to_eef={obj_to_eef:.3f})", flush=True)
        return None

    # Capture the saved-grasp dict.
    eef_pos_t, eef_quat_t = robot.eef_links[arm].get_position_orientation()
    tgt_pos_t, tgt_quat_t = target_obj.get_position_orientation()
    T_eef_w = _pose_to_mat(eef_pos_t.cpu().numpy(), eef_quat_t.cpu().numpy())
    T_tgt_w = _pose_to_mat(tgt_pos_t.cpu().numpy(), tgt_quat_t.cpu().numpy())
    T_eef_local = np.linalg.inv(T_tgt_w) @ T_eef_w
    rel_pos, rel_quat = _mat_to_pose(T_eef_local)

    full_q = robot.get_joint_positions().cpu().numpy()
    traj_np = (joint_traj.cpu().numpy() if hasattr(joint_traj, "cpu")
               else np.asarray(joint_traj))
    return {
        "rel_position": rel_pos.astype(np.float32),
        "rel_orientation_xyzw": rel_quat.astype(np.float32),
        "gripper_qpos": full_q[gripper_control_idx.cpu().numpy()].astype(np.float32),
        "arm_joint_pos": full_q[arm_control_idx.cpu().numpy()].astype(np.float32),
        "approach_traj": traj_np.astype(np.float32),
        "obj_to_eef": float(obj_to_eef),
    }


def _try_grasp_candidate(
    env, robot, primitives, target_obj,
    init_pos, init_quat, T_target_world,
    T_local: np.ndarray,
    cfg: GraspCollectorConfig,
    open_gripper_q, zero_arm_cmd,
    arm_control_idx, gripper_control_idx,
    initial_joint_pos,
    deadline: float,
    verbose_idx: int | None = None,
    cand_timings: dict | None = None,
) -> dict | None:
    """Phase A per-candidate: reach prefilter → cuRobo motion plan →
    :func:`run_grasp_attempt`.
    """
    import torch as th

    arm = robot.default_arm

    # Stage 0: reset the world to a clean state BEFORE planning.
    # Without this, after the first successful candidate the robot ends in
    # a "still gripping" pose (gripper closed around target, fingers in
    # collision with the target). cuRobo then plans from that constrained
    # start state and reports "no path" for every subsequent candidate —
    # the reset that normally runs inside ``run_grasp_attempt`` never
    # fires because we never get past the motion plan.
    robot.release_grasp_immediately(arm)
    robot.set_joint_positions(initial_joint_pos)
    target_obj.set_position_orientation(position=init_pos, orientation=init_quat)
    target_obj.root_link.set_linear_velocity(th.zeros(3))
    target_obj.root_link.set_angular_velocity(th.zeros(3))
    _reset_controller_goals(robot)
    # Settle so the target rests on the surface under gravity (no
    # hard_pin — the gripper isn't touching it yet, gravity holds it).
    import omnigibson as og
    for _ in range(2):
        og.sim.step()

    # Stage 1: reach prefilter.
    T_eef_world = T_target_world @ T_local
    eef_pos_np, eef_quat_np = _mat_to_pose(T_eef_world)
    robot_base_np = (robot.get_position_orientation()[0]
                     .cpu().numpy().astype(np.float64).reshape(3))
    if float(np.linalg.norm(eef_pos_np - robot_base_np)) > cfg.max_reach:
        if verbose_idx is not None:
            print(f"    cand {verbose_idx}: reach skip", flush=True)
        return None

    # Stage 2: cuRobo motion plan.
    eef_pos_t = th.tensor(eef_pos_np, dtype=th.float32)
    eef_quat_t = th.tensor(eef_quat_np, dtype=th.float32)
    try:
        # cuRobo's collision world is refreshed ONCE at the top of
        # ``collect_valid_grasps`` (which pins the target to ``init_pos``
        # before refreshing). Every candidate resets the target back to
        # the same pose before this point, so the obstacle world is
        # bit-identical and re-running ``update_obstacles`` per candidate
        # just costs 50-200 ms of prim iteration for no benefit.
        joint_traj = _curobo_ik_fast(
            primitives._motion_generator, robot, arm,
            eef_pos_t, eef_quat_t, skip_obstacle_update=True,
            ik_precheck=cfg.ik_precheck,
            single_stage=cfg.single_stage_grasp,
            timings=cand_timings,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose_idx is not None:
            # Verbose during debug: full traceback on the FIRST candidate,
            # condensed thereafter (otherwise tracebacks flood the log).
            import traceback as _tb
            if verbose_idx == 0:
                print(f"    cand {verbose_idx}: cuRobo raised "
                      f"{type(exc).__name__}: {exc}", flush=True)
                _tb.print_exc()
            else:
                print(f"    cand {verbose_idx}: cuRobo raised "
                      f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return None
    if joint_traj is None or len(joint_traj) == 0:
        if verbose_idx is not None:
            print(f"    cand {verbose_idx}: motion plan: no path", flush=True)
        return None
    if verbose_idx is not None:
        print(f"    cand {verbose_idx}: motion plan ok "
              f"(traj_len={len(joint_traj)})", flush=True)
    if time.time() > deadline:
        return None

    return run_grasp_attempt(
        env, robot, target_obj,
        init_pos, init_quat,
        joint_traj=joint_traj,
        cfg=cfg,
        open_gripper_q=open_gripper_q,
        zero_arm_cmd=zero_arm_cmd,
        arm_control_idx=arm_control_idx,
        gripper_control_idx=gripper_control_idx,
        initial_joint_pos=initial_joint_pos,
        deadline=deadline,
        frame_callback=None,
        verbose_idx=verbose_idx,
    )


def collect_valid_grasps(
    env, robot, primitives, target_obj,
    init_pos, init_quat,
    candidates_local: np.ndarray,
    cfg: GraspCollectorConfig,
    deadline: float,
    on_progress: Callable[[int, dict | None], None] | None = None,
    verbose: bool = False,
    timings_log: list | None = None,
) -> list[dict]:
    """Iterate through ``candidates_local`` and return up to
    ``cfg.num_target_grasps`` validated grasps (or as many as fit
    within ``deadline``).

    No frame capture. Caller is responsible for spawning ``target_obj``,
    setting it to ``(init_pos, init_quat)``, disabling gravity on it,
    and updating cuRobo obstacles to ignore the target. The function
    leaves the env state dirty — caller cleans up.
    """
    import torch as th

    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    open_gripper_q = robot.joint_upper_limits[gripper_control_idx].clone()
    # Action vector dim != joint dim — OSC delta uses 6-D pose deltas, not
    # 7-D joints. Build zero-cmd from the action-side index.
    zero_arm_cmd = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)
    initial_joint_pos = robot.get_joint_positions().clone()
    T_target_world = _pose_to_mat(
        init_pos.cpu().numpy() if hasattr(init_pos, "cpu") else init_pos,
        init_quat.cpu().numpy() if hasattr(init_quat, "cpu") else init_quat,
    )

    # Place the target at (init_pos, init_quat) and refresh cuRobo's
    # collision world ONCE, then keep ``skip_obstacle_update=True`` for
    # every per-candidate plan. The target is reset to the same pose at
    # the start of every candidate so the obstacle world is identical
    # each time — re-scanning per candidate cost 50-200 ms × N for no
    # behavioural change.
    target_obj.set_position_orientation(position=init_pos, orientation=init_quat)
    primitives._motion_generator.update_obstacles(ignore_objects=[])

    # Snapshot the FULL sim state right now (post-target-placement,
    # pre-first-candidate). With the new dynamic approach, the gripper
    # can brush against scene objects (fragiles, clutter) during a
    # failed candidate — those displacements would otherwise persist.
    # We restore this snapshot at the start of every candidate so
    # every attempt sees the same scene state.
    import omnigibson as og
    pre_candidate_state = og.sim.dump_state()

    held: list[dict] = []
    for ci, T_local in enumerate(candidates_local):
        # Restore the full scene snapshot before each candidate. This
        # zeros velocities, returns every object to its position at the
        # top of this function, and clears any stale AG bond.
        if ci > 0:
            og.sim.load_state(pre_candidate_state)
            og.sim.step()
        if time.time() > deadline:
            break
        if len(held) >= cfg.num_target_grasps:
            break
        cand_timings: dict = {}
        t_cand0 = time.time()
        result = _try_grasp_candidate(
            env, robot, primitives, target_obj,
            init_pos, init_quat, T_target_world,
            np.asarray(T_local, dtype=np.float64),
            cfg, open_gripper_q, zero_arm_cmd,
            arm_control_idx, gripper_control_idx,
            initial_joint_pos, deadline,
            verbose_idx=ci if verbose else None,
            cand_timings=cand_timings,
        )
        cand_timings["total_s"] = time.time() - t_cand0
        cand_timings["held"] = result is not None
        if timings_log is not None:
            timings_log.append(cand_timings)
        if on_progress is not None:
            on_progress(ci, result)
        if result is not None:
            held.append(result)

    return held


def save_grasp_dataset(held: list[dict], path: Path | str,
                       target_name: str) -> None:
    """Serialize ``held`` grasps to ``path`` in the format
    :class:`sentinel.rl.grasps.reset.GraspDatasetResetter` consumes.

    Pads variable-length ``approach_traj`` entries into a list rather
    than stacking, so trajectories of different lengths coexist.
    """
    import torch as th

    if not held:
        raise ValueError("Cannot save an empty grasp list")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_name": target_name,
        "rel_position": th.from_numpy(
            np.stack([g["rel_position"] for g in held])),
        "rel_orientation_xyzw": th.from_numpy(
            np.stack([g["rel_orientation_xyzw"] for g in held])),
        "gripper_qpos": th.from_numpy(
            np.stack([g["gripper_qpos"] for g in held])),
        "arm_joint_pos": th.from_numpy(
            np.stack([g["arm_joint_pos"] for g in held])),
        # List rather than stacked tensor because cuRobo trajectories
        # are variable-length per candidate.
        "approach_traj": [
            th.from_numpy(g["approach_traj"]) for g in held
        ],
    }
    th.save(payload, str(path))
