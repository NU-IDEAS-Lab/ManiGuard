"""Reset arm+gripper to a saved grasp pose for the target object.

Port of UW Lab OmniReset's ``reset_end_effector_from_grasp_dataset``
(``source/uwlab_tasks/.../mdp/events.py:606-779``) to OmniGibson + cuRobo.

At each episode reset we:
    1. Read the target's current world pose (pos, quat xyzw).
    2. Sample a random grasp from ``grasps_<target>.pt`` — ``rel_position``
       (3,), ``rel_orientation_xyzw`` (4,), ``gripper_qpos`` (F,).
    3. Compose ``T_gripper_world = T_target_world @ T_relative`` to get the
       saved grasp pose in world coordinates (adapts to the target's current
       pose — OmniReset's key trick for reusing grasps across episodes with
       randomized object placement).
    4. Optional ±pose perturbation in the gripper body frame.
    5. cuRobo IK solves arm joint positions that realize this eef pose.
    6. ``set_joint_positions`` teleports arm + writes saved gripper_qpos.
    7. On IK failure, retry with a different grasp (up to ``max_retries``).
       If all retries fail, return False so the caller falls back to the
       default precached-ready-pose reset.

Differences from OmniReset:
    - Quat convention is xyzw (OG/PhysX) rather than wxyz (IsaacLab).
    - IK is cuRobo (binary success/fail per call) rather than the Jacobian-
      DLS iterative solver OmniReset uses. OmniReset's 25-iter ``joint_pos =
      current + 0.25 * (target - current)`` converges softly to whatever the
      DLS produces; we just get a clean binary success and retry on fail.
    - We don't re-use OmniReset's ``gripper_joint_positions`` dict layout
      (per-joint-name); our collector saves a flat ``gripper_qpos`` tensor
      already aligned to ``robot.gripper_control_idx``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th


class GraspDatasetResetter:
    """Loads ``grasps_<target>.pt`` and configures arm+gripper at a saved grasp.

    Lifetime matches the task: construct once per task load, call
    ``reset_eef`` at each episode reset.
    """

    def __init__(
        self,
        env,
        robot,
        target_obj,
        dataset_path: Path | str,
        reset_mode: str = "cached",
        pose_range_b: Dict[str, Tuple[float, float]] | None = None,
        max_retries: int = 5,
        zero_gravity_settle_steps: int = 5,
        post_gravity_settle_steps: int = 3,
    ):
        """
        Args:
            reset_mode: either ``"cached"`` or ``"ik"``.

                - ``"cached"`` (default): use the arm joint angles stored at
                  collection time (``grasps.pt::arm_joint_pos``) directly.
                  Skips cuRobo entirely, which (a) removes the ~100-500 ms
                  online IK cost per reset and (b) sidesteps OG's
                  ``assert len(og.sim.scenes) == 1`` in
                  ``CuRoboMotionGenerator.__init__`` — so this mode is the
                  one that unblocks ``num_envs > 1``. Requires the target
                  object to be at (approximately) the same world pose at
                  reset time as it was at collection time; our current
                  ``_restore_target`` always resets to the scene-init pose,
                  so cached joints stay valid across episodes.
                - ``"ik"``: compose ``T_gripper_world = T_obj_world @ T_rel``
                  freshly each reset and run cuRobo IK. Supports
                  ``pose_range_b`` per-reset randomization (the saved
                  ``arm_joint_pos`` is only valid for the original object
                  pose; any perturbation requires a re-IK). Slower per
                  reset + single-env only, but necessary for
                  OmniReset-style generalization training.
            pose_range_b: body-frame ±(x, y, z, roll, pitch, yaw) perturbation
                applied per reset. Only meaningful under ``reset_mode="ik"``;
                ``"cached"`` mode rejects any non-zero range.
        """
        self._env = env
        self._robot = robot
        self._target_obj = target_obj
        self._dataset_path = Path(dataset_path).resolve()
        self._max_retries = max_retries
        # OmniReset-style gravity management: float the target during teleport
        # so that tiny gripper-target penetrations can be resolved without
        # PhysX pushing the object away. Gravity is re-enabled before return.
        self._zero_gravity_settle_steps = int(zero_gravity_settle_steps)
        self._post_gravity_settle_steps = int(post_gravity_settle_steps)

        if reset_mode not in {"cached", "ik"}:
            raise ValueError(f"reset_mode must be 'cached' or 'ik', got {reset_mode!r}")
        self._reset_mode = reset_mode

        self._arm = robot.default_arm
        self._arm_control_idx = robot.arm_control_idx[self._arm]
        self._gripper_control_idx = robot.gripper_control_idx[self._arm]

        # Pose range for per-reset perturbation (body-frame translation +
        # roll/pitch/yaw). Mirrors OmniReset's ``pose_range_b``.
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        pose_range_b = pose_range_b or {}
        self._ranges = np.asarray(
            [pose_range_b.get(k, (0.0, 0.0)) for k in keys], dtype=np.float64
        )
        if reset_mode == "cached" and np.any(self._ranges != 0.0):
            raise ValueError(
                "reset_mode='cached' is incompatible with pose_range_b "
                "(per-reset perturbation requires fresh IK). "
                "Switch to reset_mode='ik' to use pose_range_b."
            )

        self._load_grasps()
        self._primitives = None  # lazy cuRobo (only constructed in ``ik`` mode)

    # ------------------------------------------------------------------ load

    def _load_grasps(self):
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Grasp dataset not found: {self._dataset_path}")
        data = th.load(str(self._dataset_path), map_location="cpu")
        self._rel_positions = data["rel_position"].float()  # (N, 3)
        self._rel_orientations_xyzw = data["rel_orientation_xyzw"].float()  # (N, 4)
        self._gripper_qpos = data["gripper_qpos"].float()  # (N, F)
        # Arm joints at collection time — required for cached-mode reset.
        # Older datasets without this key fall back to ik mode only.
        if "arm_joint_pos" in data:
            self._arm_joint_pos = data["arm_joint_pos"].float()  # (N, J)
        else:
            self._arm_joint_pos = None
            if self._reset_mode == "cached":
                raise ValueError(
                    f"Dataset {self._dataset_path} has no 'arm_joint_pos' — "
                    f"re-run collector or switch to reset_mode='ik'."
                )
        self._num_grasps = int(self._rel_positions.shape[0])
        if self._num_grasps == 0:
            raise ValueError(f"Grasp dataset is empty: {self._dataset_path}")

    def _ensure_curobo(self):
        if self._primitives is not None:
            return
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        self._primitives = StarterSemanticActionPrimitives(
            self._env, self._robot, enable_head_tracking=False
        )

    # --------------------------------------------------------------- public

    def reset_eef(self, rng: np.random.Generator | None = None) -> bool:
        """Attempt to configure the arm+gripper at a saved grasp pose.

        Returns:
            True if IK succeeded on one of the sampled grasps and joints have
            been written; False if all retries failed (caller should fall
            back to a default reset).
        """
        if rng is None:
            rng = np.random.default_rng()

        tgt_pos, tgt_quat = self._target_obj.get_position_orientation()
        tgt_pos = tgt_pos.detach().cpu().numpy().astype(np.float64)
        tgt_quat = tgt_quat.detach().cpu().numpy().astype(np.float64)  # xyzw

        n_try = min(self._max_retries, self._num_grasps)
        indices = rng.permutation(self._num_grasps)[:n_try]

        # Only "ik" mode needs cuRobo; "cached" mode uses saved joints directly
        # and therefore works under multi-env (``num_envs > 1``) where cuRobo's
        # ``assert len(og.sim.scenes) == 1`` would otherwise fail.
        if self._reset_mode == "ik":
            self._ensure_curobo()
        import omnigibson as og
        from omnigibson.object_states import Touching

        # Pre-build a "hold" action so the arm doesn't collapse under gravity
        # during settle steps. OSC pose_delta_ori with zero delta = keep
        # current eef pose; gripper_cmd=-1 = stay closed.
        hold_action = th.zeros(self._robot.action_dim, dtype=th.float32)
        hold_action[self._robot.arm_action_idx[self._arm]] = th.zeros(
            len(self._robot.arm_action_idx[self._arm]), dtype=th.float32
        )
        hold_action[self._robot.gripper_action_idx[self._arm]] = -1.0  # close

        # Snapshot target pose so we can restore between retries (a failed
        # attempt may have bumped or dropped the object).
        init_tgt_pos, init_tgt_quat = self._target_obj.get_position_orientation()
        init_tgt_pos = init_tgt_pos.detach().clone()
        init_tgt_quat = init_tgt_quat.detach().clone()
        zero3 = th.zeros(3)

        def _restore_target():
            self._target_obj.set_position_orientation(
                position=init_tgt_pos, orientation=init_tgt_quat
            )
            self._target_obj.root_link.set_linear_velocity(zero3)
            self._target_obj.root_link.set_angular_velocity(zero3)

        for idx in indices:
            idx = int(idx)
            rel_pos = self._rel_positions[idx].numpy().astype(np.float64)
            rel_quat = self._rel_orientations_xyzw[idx].numpy().astype(np.float64)

            # Restore target pose at start of each attempt so the world-frame
            # grasp target stays consistent across retries.
            _restore_target()

            if self._reset_mode == "cached":
                # Fast path: use the arm joint angles the collector saved for
                # this exact grasp — valid because ``_restore_target`` puts
                # the object back at its scene-init pose (same pose as at
                # collection time). Zero IK cost per reset, no cuRobo ref,
                # and therefore works under multi-env.
                joint_pos = self._arm_joint_pos[idx]
            else:
                # Compose world eef target: T_eef_world = T_obj_world @ T_rel
                eef_pos, eef_quat = _compose_pose(tgt_pos, tgt_quat, rel_pos, rel_quat)

                # Optional body-frame perturbation (``reset_mode='ik'`` only).
                if np.any(self._ranges != 0.0):
                    samples = rng.uniform(self._ranges[:, 0], self._ranges[:, 1])
                    perturb_quat = _euler_to_quat_xyzw(
                        float(samples[3]), float(samples[4]), float(samples[5])
                    )
                    eef_pos, eef_quat = _compose_pose(
                        eef_pos, eef_quat, samples[:3].astype(np.float64), perturb_quat
                    )

                joint_pos = self._curobo_ik(eef_pos, eef_quat)
                if joint_pos is None:
                    continue

            # Zero-gravity teleport + settle so tiny finger/target overlap
            # is resolved without the object flying off.
            self._target_obj.root_link.disable_gravity()
            try:
                self._robot.set_joint_positions(joint_pos, self._arm_control_idx)
                self._robot.set_joint_positions(
                    self._gripper_qpos[idx], self._gripper_control_idx
                )
                # Zero controller goals so OSC / IK controllers don't snap the
                # arm back to a pre-teleport commanded pose on the next step.
                for ctrl in self._robot._controllers.values():
                    ctrl._goal = None

                for _ in range(self._zero_gravity_settle_steps):
                    self._robot.apply_action(hold_action)
                    og.sim.step()
            finally:
                self._target_obj.root_link.enable_gravity()

            # Under gravity, verify the grasp actually holds the target. This
            # is the "real" success gate: IK found a kinematic solution AND
            # the resulting fingertips make contact sufficient for friction
            # to keep the object from falling.
            for _ in range(self._post_gravity_settle_steps):
                self._robot.apply_action(hold_action)
                og.sim.step()

            if self._robot.states[Touching].get_value(self._target_obj):
                return True

            # Not touching — grasp didn't physically hold. Clean up so the
            # next retry starts from a clean state.
            for arm in self._robot.arm_names:
                self._robot.release_grasp_immediately(arm=arm)

        # All retries exhausted. Leave target at its initial pose + gravity on
        # so the caller sees a consistent state before falling back to the
        # default precached reset.
        _restore_target()
        return False

    # ---------------------------------------------------------------- curobo

    def _curobo_ik(self, eef_pos_np: np.ndarray, eef_quat_np: np.ndarray):
        """cuRobo IK via DEFAULT embodiment (single-arm Franka).

        Same implementation as ``sentinel.rl.grasps.collector._curobo_ik``
        — duplicated here to avoid importing a module-private helper and to
        keep the resetter self-contained.
        """
        import torch
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            m as primitives_macros,
        )

        motion_gen = self._primitives._motion_generator
        eef_link = self._robot.eef_link_names[self._arm]
        bs = motion_gen.batch_size

        eef_pos = torch.tensor(eef_pos_np, dtype=torch.float32)
        eef_quat = torch.tensor(eef_quat_np, dtype=torch.float32)
        target_pos = {eef_link: torch.stack([eef_pos for _ in range(bs)])}
        target_quat = {eef_link: torch.stack([eef_quat for _ in range(bs)])}

        successes, joint_states = motion_gen.compute_trajectories(
            target_pos=target_pos, target_quat=target_quat,
            initial_joint_pos=None, is_local=False,
            max_attempts=math.ceil(primitives_macros.MAX_PLANNING_ATTEMPTS / bs),
            timeout=60.0,
            ik_fail_return=primitives_macros.MAX_IK_FAILURES_BEFORE_RETURN,
            enable_finetune_trajopt=False, finetune_attempts=0,
            return_full_result=False, success_ratio=1.0 / bs,
            attached_obj=None, attached_obj_scale=None, motion_constraint=None,
            skip_obstacle_update=False, ik_only=True,
            ik_world_collision_check=False,
            emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )

        success_idx = torch.where(successes)[0].cpu()
        if len(success_idx) == 0:
            return None
        joint_state = joint_states[success_idx[0]]
        joint_pos = motion_gen.path_to_joint_trajectory(
            joint_state, get_full_js=False,
            emb_sel=CuRoboEmbodimentSelection.DEFAULT,
        )
        manip_idx = torch.cat([self._robot.arm_control_idx[self._arm]])
        return joint_pos[manip_idx].cpu()


# -------------------------------------------------------- pose-math helpers


def _quat_xyzw_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quat → 3×3 rotation matrix."""
    import trimesh.transformations as tra
    return tra.quaternion_matrix(
        [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
    )[:3, :3]


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (x, y, z, w) quat."""
    import trimesh.transformations as tra
    T = np.eye(4)
    T[:3, :3] = R
    q_wxyz = tra.quaternion_from_matrix(T)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def _compose_pose(
    pos_a: np.ndarray, quat_a_xyzw: np.ndarray,
    pos_b: np.ndarray, quat_b_xyzw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose ``T_a @ T_b``: pose_b is interpreted in pose_a's local frame.

    Same semantics as IsaacLab's ``math_utils.combine_frame_transforms``
    used in OmniReset's reset event.
    """
    Ra = _quat_xyzw_to_mat(quat_a_xyzw)
    Rb = _quat_xyzw_to_mat(quat_b_xyzw)
    Rc = Ra @ Rb
    pos_c = np.asarray(pos_a, dtype=np.float64) + Ra @ np.asarray(pos_b, dtype=np.float64)
    return pos_c, _mat_to_quat_xyzw(Rc)


def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ Euler → (x, y, z, w) quat."""
    import trimesh.transformations as tra
    q_wxyz = tra.quaternion_from_euler(roll, pitch, yaw, axes="sxyz")
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


__all__ = ["GraspDatasetResetter"]
