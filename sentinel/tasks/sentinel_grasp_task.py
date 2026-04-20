"""Sentinel's extension of :class:`omnigibson.tasks.grasp_task.GraspTask`.

Three differences from the upstream v3.7.2 task:

1. ``_create_termination_conditions`` actually registers a ``GraspGoal``
   termination (upstream has it commented out) and plumbs a ``hold_steps``
   counter through so success requires N consecutive steps of a valid grasp
   rather than a single frame. The ``GraspGoal`` class itself is runtime-
   patched by :mod:`sentinel._omnigibson_patches` to accept that kwarg.

2. ``_reset_agent`` handles robots that lack ``trunk_control_idx`` (e.g. a
   Franka mounted on a base without a torso) by falling back to arm-only
   control-idx, and coerces reset-pose values to tensors so yaml-loaded
   lists don't blow up the call to ``robot.set_joint_positions``.

3. ``default_termination_config`` adds ``grasp_hold_steps: 1`` so YAMLs that
   don't override ``termination_config`` still get sane defaults.

Registered as ``"SentinelGraspTask"`` via the inherited ``Registerable``
machinery. Task configs should use ``type: SentinelGraspTask``.
"""
from __future__ import annotations

import random

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.reward_functions.grasp_reward import GraspReward
from omnigibson.tasks.grasp_task import GraspTask, MAX_JOINT_RANDOMIZATION_ATTEMPTS
from omnigibson.termination_conditions.grasp_goal import GraspGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.python_utils import classproperty


def _resolve_joint_control_idx(robot):
    """Concat trunk + default-arm control idx, falling back to arm-only."""
    if hasattr(robot, "trunk_control_idx"):
        return th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    return robot.arm_control_idx[robot.default_arm]


class SentinelGraspTask(GraspTask):
    def _create_termination_conditions(self):
        terminations = dict()
        hold_steps = 1
        if self._termination_config is not None:
            hold_steps = self._termination_config.get("grasp_hold_steps", 1)
        terminations["graspgoal"] = GraspGoal(self.obj_name, hold_steps=hold_steps)
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        return terminations

    def _create_reward_functions(self):
        rewards = dict()
        rewards["grasp"] = GraspReward(self.obj_name, **self._reward_config)
        return rewards

    def _reset_agent(self, env):
        robot = env.robots[0]
        for arm in robot.arm_names:
            robot.release_grasp_immediately(arm=arm)

        # Cached reset poses (the fast path — always used by sentinel configs).
        if self._reset_poses is not None:
            joint_control_idx = _resolve_joint_control_idx(robot)
            robot_pose = random.choice(self._reset_poses)
            joint_pos = robot_pose["joint_pos"]
            if not isinstance(joint_pos, th.Tensor):
                joint_pos = th.tensor(joint_pos)
            robot.set_joint_positions(joint_pos, joint_control_idx)
            robot_pos = robot_pose["base_pos"]
            if not isinstance(robot_pos, th.Tensor):
                robot_pos = th.tensor(robot_pos)
            robot_orn = robot_pose["base_ori"]
            if not isinstance(robot_orn, th.Tensor):
                robot_orn = th.tensor(robot_orn)
            robot.set_position_orientation(
                position=robot_pos, orientation=robot_orn, frame="scene"
            )
            return

        # Primitive-controller fallback (randomized pose via curobo).
        if self._primitive_controller is None:
            self._primitive_controller = StarterSemanticActionPrimitives(
                env, robot, enable_head_tracking=False
            )

        joint_control_idx = _resolve_joint_control_idx(robot)
        dim = len(joint_control_idx)
        if "combined" in robot.robot_arm_descriptor_yamls:
            joint_combined_idx = th.cat(
                [robot.trunk_control_idx, robot.arm_control_idx["combined"]]
            )
            initial_joint_pos = th.tensor(robot.get_joint_positions()[joint_combined_idx])
            control_idx_in_joint_pos = th.where(th.isin(joint_combined_idx, joint_control_idx))[0]
        else:
            initial_joint_pos = th.tensor(robot.get_joint_positions()[joint_control_idx])
            control_idx_in_joint_pos = th.arange(dim)

        for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
            joint_pos, joint_control_idx = self._get_random_joint_position(robot)
            initial_joint_pos[control_idx_in_joint_pos] = joint_pos
            collision_detected = self._primitive_controller._motion_generator.check_collisions(
                [initial_joint_pos],
            ).cpu()[0]
            if not collision_detected:
                robot.set_joint_positions(joint_pos, joint_control_idx)
                og.sim.step()
                break

        obj = env.scene.object_registry("name", self.obj_name)
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        grasp_pose = random.choice(grasp_poses)
        sampled_pose_2d = self._primitive_controller._sample_pose_near_object(obj, pose_on_obj=grasp_pose)
        robot_pose = self._primitive_controller._get_robot_pose_from_2d_pose(sampled_pose_2d)
        robot.set_position_orientation(*robot_pose)

        for _ in range(10):
            og.sim.step()
        for _ in range(100):
            og.sim.step()
            if th.norm(robot.get_linear_velocity()) > 1e-2:
                continue
            if th.norm(robot.get_angular_velocity()) > 1e-2:
                continue
            break
        else:
            raise ValueError("Robot could not settle")

        robot_up = T.quat_apply(
            robot.get_position_orientation()[1], th.tensor([0, 0, 1], dtype=th.float32)
        )
        if robot_up[2] < 0.75:
            raise ValueError("Robot has toppled over")

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 100000, "grasp_hold_steps": 1}


__all__ = ["SentinelGraspTask"]
