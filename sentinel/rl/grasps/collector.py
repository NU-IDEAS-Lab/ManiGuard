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


def _curobo_ik_fast(motion_gen, robot, arm, eef_pos, eef_quat,
                    skip_obstacle_update):
    """Full cuRobo motion plan to a Cartesian eef target.

    Returns ``(T, J)`` torch tensor of arm-joint waypoints, or ``None``
    if no plan was found within budget. Without table or any other
    obstacle (target ignored, floor far below Franka mount), cuRobo's
    sphere-based collision check has plenty of free space to plan an
    approach trajectory. The mimic-joint monkey-patch above keeps the
    planner from choking on missing finger-joint state.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    _patch_curobo_mimic_lookup()

    eef_link = robot.eef_link_names[arm]
    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_pos for _ in range(bs)])}
    target_quat = {eef_link: th.stack([eef_quat for _ in range(bs)])}

    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        initial_joint_pos=None,
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
        motion_constraint=None,
        skip_obstacle_update=skip_obstacle_update,
        ik_only=False,
        ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        return None
    joint_state = joint_states[success_idx[0]]
    joint_pos = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    manip_idx = th.cat([robot.arm_control_idx[arm]])
    if joint_pos.dim() == 1:
        return joint_pos[manip_idx].unsqueeze(0).cpu()
    return joint_pos[:, manip_idx].cpu()


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
        frame_callback: optional zero-arg callable invoked once after every
            simulator step. Phase B uses it to push frames into a buffer.
        verbose_idx: when set, prints a stage-specific reject reason
            (``AG didn't engage`` / ``phase2 drop``).
    """
    import omnigibson as og
    from omnigibson.controllers.controller_base import IsGraspingState

    arm = robot.default_arm

    # Reset for this attempt — clears any stale AG bond, returns arm to home,
    # snaps target back to its init pose.
    robot.release_grasp_immediately(arm)
    robot.set_joint_positions(initial_joint_pos)
    target_obj.set_position_orientation(position=init_pos, orientation=init_quat)
    _reset_controller_goals(robot)
    _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
    if frame_callback is not None:
        frame_callback()

    # Stage 1: replay trajectory with hard pin so the gripper sweeping
    # toward the grasp pose can't push the target around.
    robot.set_joint_positions(open_gripper_q, gripper_control_idx)
    for waypoint in joint_traj:
        if time.time() > deadline:
            return None
        robot.set_joint_positions(waypoint, arm_control_idx)
        robot.set_joint_positions(open_gripper_q, gripper_control_idx)
        _reset_controller_goals(robot)
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
        if frame_callback is not None:
            frame_callback()

    # Stage 2: open settle + close, still pinned.
    robot.set_joint_positions(open_gripper_q, gripper_control_idx)
    _reset_controller_goals(robot)
    settle_action = _build_action(robot, zero_arm_cmd, gripper_cmd=+1.0)
    for _ in range(cfg.settle_open_steps):
        if time.time() > deadline:
            return None
        robot.apply_action(settle_action)
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
        if frame_callback is not None:
            frame_callback()

    close_action = _build_action(robot, zero_arm_cmd, gripper_cmd=-1.0)
    for _ in range(cfg.close_steps):
        if time.time() > deadline:
            return None
        robot.apply_action(close_action)
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)
        if frame_callback is not None:
            frame_callback()

    if robot.is_grasping(arm, target_obj) != IsGraspingState.TRUE:
        if verbose_idx is not None:
            print(f"    cand {verbose_idx}: AG didn't engage after close",
                  flush=True)
        return None

    # Stage 3: gravity hold without pin — real verification.
    target_obj.root_link.enable_gravity()
    try:
        for _ in range(cfg.gravity_hold_steps):
            if time.time() > deadline:
                return None
            robot.apply_action(close_action)
            og.sim.step()
            if frame_callback is not None:
                frame_callback()
    finally:
        target_obj.root_link.disable_gravity()

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
) -> dict | None:
    """Phase A per-candidate: reach prefilter → cuRobo motion plan →
    :func:`run_grasp_attempt`.
    """
    import torch as th

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
    arm = robot.default_arm
    eef_pos_t = th.tensor(eef_pos_np, dtype=th.float32)
    eef_quat_t = th.tensor(eef_quat_np, dtype=th.float32)
    try:
        # Refresh cuRobo's collision world per candidate. Skipping the
        # update is faster but a previous candidate's gravity hold can
        # leave PhysX state that cuRobo's cached world disagrees with,
        # making subsequent motion plans return "no path".
        primitives._motion_generator.update_obstacles(ignore_objects=[target_obj])
        joint_traj = _curobo_ik_fast(
            primitives._motion_generator, robot, arm,
            eef_pos_t, eef_quat_t, skip_obstacle_update=True,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose_idx is not None:
            print(f"    cand {verbose_idx}: cuRobo raised "
                  f"{type(exc).__name__}", flush=True)
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

    held: list[dict] = []
    for ci, T_local in enumerate(candidates_local):
        if time.time() > deadline:
            break
        if len(held) >= cfg.num_target_grasps:
            break
        result = _try_grasp_candidate(
            env, robot, primitives, target_obj,
            init_pos, init_quat, T_target_world,
            np.asarray(T_local, dtype=np.float64),
            cfg, open_gripper_q, zero_arm_cmd,
            arm_control_idx, gripper_control_idx,
            initial_joint_pos, deadline,
            verbose_idx=ci if verbose else None,
        )
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
