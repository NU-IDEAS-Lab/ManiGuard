"""Privileged state PickAndLift task for proprio-only RL.

This variant intentionally leaves ``PickAndLiftTask`` unchanged. It provides a
state-based observation with explicit goal information and a reward that treats
physical grasp, lifted holding, carry progress, and dropping as distinct
events.
"""

from __future__ import annotations

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.controllers import IsGraspingState
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.python_utils import classproperty

from sentinel.rl.tasks.pick_and_lift import PickAndLiftTask


def _pos_in_robot_frame(robot, world_pos):
    robot_pos, robot_quat = robot.get_position_orientation()
    rel_pos, _ = T.relative_pose_transform(
        world_pos,
        th.tensor([0.0, 0.0, 0.0, 1.0], dtype=world_pos.dtype, device=world_pos.device),
        robot_pos,
        robot_quat,
    )
    return rel_pos


def _is_grasp_candidate(robot, obj) -> bool:
    """Return True only when OG's gripper-level grasp inference says target is grasped."""
    try:
        return robot.is_grasping(
            arm=robot.default_arm,
            candidate_obj=obj,
        ) == IsGraspingState.TRUE
    except Exception:  # noqa: BLE001 - physical-grasp inference is best-effort
        return False


def _is_lifted(obj, init_obj_z: float | None, lift_threshold: float) -> bool:
    if init_obj_z is None:
        return False
    try:
        obj_z = float(obj.get_position_orientation()[0][2])
    except Exception:  # noqa: BLE001
        return False
    return obj_z >= float(init_obj_z) + float(lift_threshold)


def _is_holding_for_carry(robot, obj, init_obj_z: float | None, lift_threshold: float) -> bool:
    return _is_grasp_candidate(robot, obj) and _is_lifted(obj, init_obj_z, lift_threshold)


class PrivilegedInGoalRegion(SuccessCondition):
    """Success requires goal intersection / proximity while still lifted-held."""

    def __init__(self, obj_name: str, success_radius: float = 0.05,
                 goal_region_spec=None, lift_threshold: float = 0.01):
        self.obj_name = obj_name
        self.success_radius = float(success_radius)
        self._goal_region_spec = goal_region_spec
        self._lift_threshold = float(lift_threshold)
        self._init_obj_z: float | None = None

    def _step(self, task, env, action):
        obj = env.scene.object_registry("name", self.obj_name)
        if obj is None or task._goal_world_pos is None:
            return False

        if self._goal_region_spec is not None:
            from sentinel.utils.goal_region import object_intersects_goal_region

            in_goal = object_intersects_goal_region(obj, self._goal_region_spec)
        else:
            obj_pos = obj.get_position_orientation()[0]
            in_goal = bool(th.norm(obj_pos - task._goal_world_pos) < self.success_radius)

        if not in_goal:
            return False
        return _is_holding_for_carry(
            env.robots[0],
            obj,
            self._init_obj_z,
            self._lift_threshold,
        )

    def reset(self, task, env):
        obj = env.scene.object_registry("name", self.obj_name)
        self._init_obj_z = (
            float(obj.get_position_orientation()[0][2]) if obj is not None else None
        )


class PrivilegedPickAndLiftReward(BaseRewardFunction):
    """Reward for approach, grasp acquisition, lifted carry progress, and drop."""

    def __init__(
        self,
        obj_name: str,
        pregrasp_dist_coeff: float = 1.0,
        carry_dist_coeff: float = 4.0,
        grasp_acquire_reward: float = 2.0,
        drop_penalty: float = 1.0,
        goal_bonus: float = 50.0,
        collision_penalty: float = 0.0,
        regularization_coef: float = 0.0,
        lift_threshold: float = 0.01,
    ):
        super().__init__()
        self.obj_name = obj_name
        self.pregrasp_dist_coeff = float(pregrasp_dist_coeff)
        self.carry_dist_coeff = float(carry_dist_coeff)
        self.grasp_acquire_reward = float(grasp_acquire_reward)
        self.drop_penalty = float(drop_penalty)
        self.goal_bonus = float(goal_bonus)
        self.collision_penalty = float(collision_penalty)
        self.regularization_coef = float(regularization_coef)
        self.lift_threshold = float(lift_threshold)

        self._obj = None
        self._init_obj_z: float | None = None
        self._prev_eef_to_obj = None
        self._prev_obj_to_goal = None
        self._was_grasp_candidate = False
        self._had_grasp_candidate = False
        self._was_holding = False
        self._had_holding = False

    def _step(self, task, env, action):
        if self._obj is None:
            self._obj = env.scene.object_registry("name", self.obj_name)
        robot = env.robots[0]
        obj_pos = self._obj.get_position_orientation()[0]
        eef_pos = robot.get_eef_position(robot.default_arm)

        if self._init_obj_z is None:
            self._init_obj_z = float(obj_pos[2])

        grasp_candidate = _is_grasp_candidate(robot, self._obj)
        lifted = _is_lifted(self._obj, self._init_obj_z, self.lift_threshold)
        holding = bool(grasp_candidate and lifted)

        reward = 0.0
        info = {
            "grasp_candidate": grasp_candidate,
            "lifted": lifted,
            "holding": holding,
        }

        if detect_robot_collision_in_sim(robot, filter_objs=[self._obj]):
            reward -= self.collision_penalty
            info["collision"] = True

        if self.regularization_coef > 0:
            action_mag = th.sum(th.abs(action))
            regularization = -float(action_mag) * self.regularization_coef
            reward += regularization
            info["regularization"] = regularization

        if (
            grasp_candidate
            and not self._was_grasp_candidate
            and not self._had_grasp_candidate
        ):
            reward += self.grasp_acquire_reward
            info["grasp_acquire_reward"] = self.grasp_acquire_reward
            self._had_grasp_candidate = True

        if self._had_holding and self._was_holding and not holding:
            reward -= self.drop_penalty
            info["drop_penalty"] = -self.drop_penalty

        obj_to_goal = th.norm(obj_pos - task._goal_world_pos)
        info["obj_to_goal"] = float(obj_to_goal)

        if holding:
            if self._prev_obj_to_goal is not None:
                progress = self._prev_obj_to_goal - obj_to_goal
                carry_shaping = float(progress) * self.carry_dist_coeff
                reward += carry_shaping
                info["carry_shaping"] = carry_shaping
                info["carry_progress"] = float(progress)
            self._prev_obj_to_goal = obj_to_goal
            self._prev_eef_to_obj = None
            self._had_holding = True

            if self._in_goal(task, self._obj, obj_pos):
                reward += self.goal_bonus
                info["goal_bonus"] = self.goal_bonus
        else:
            if not grasp_candidate:
                eef_to_obj = th.norm(eef_pos - obj_pos)
                if self._prev_eef_to_obj is not None:
                    progress = self._prev_eef_to_obj - eef_to_obj
                    pregrasp_shaping = float(progress) * self.pregrasp_dist_coeff
                    reward += pregrasp_shaping
                    info["pregrasp_shaping"] = pregrasp_shaping
                    info["pregrasp_progress"] = float(progress)
                self._prev_eef_to_obj = eef_to_obj
            self._prev_obj_to_goal = None

        self._was_grasp_candidate = grasp_candidate
        self._was_holding = holding
        return reward, info

    def _in_goal(self, task, obj, obj_pos) -> bool:
        if task._goal_region_spec is not None:
            from sentinel.utils.goal_region import object_intersects_goal_region

            return bool(object_intersects_goal_region(obj, task._goal_region_spec))
        return bool(th.norm(obj_pos - task._goal_world_pos) < task.success_radius)

    def reset(self, task, env):
        super().reset(task, env)
        self._obj = env.scene.object_registry("name", self.obj_name)
        self._init_obj_z = (
            float(self._obj.get_position_orientation()[0][2])
            if self._obj is not None
            else None
        )
        self._prev_eef_to_obj = None
        self._prev_obj_to_goal = None
        self._was_grasp_candidate = False
        self._had_grasp_candidate = False
        self._was_holding = False
        self._had_holding = False


class PickAndLiftPrivilegedTask(PickAndLiftTask):
    """Pick-and-lift task with explicit goal-region state observations."""

    def _get_obs(self, env):
        obj = env.scene.object_registry("name", self.obj_name)
        robot = env.robots[0]

        obj_world = obj.get_position_orientation()[0]
        goal_world = self._goal_world_pos.to(obj_world.device)

        target_pos_robot_frame = _pos_in_robot_frame(robot, obj_world)
        goal_pos_robot_frame = _pos_in_robot_frame(robot, goal_world)
        obj_to_goal_vector = goal_pos_robot_frame - target_pos_robot_frame
        goal_radius = th.tensor(
            [float(self.success_radius)],
            dtype=obj_world.dtype,
            device=obj_world.device,
        )

        return {
            "target_pos_robot_frame": target_pos_robot_frame,
            "goal_pos_robot_frame": goal_pos_robot_frame,
            "obj_to_goal_vector": obj_to_goal_vector,
            "goal_radius": goal_radius,
        }, dict()

    @classproperty
    def default_reward_config(cls):
        return {
            "pregrasp_dist_coeff": 1.0,
            "carry_dist_coeff": 4.0,
            "grasp_acquire_reward": 2.0,
            "drop_penalty": 1.0,
            "goal_bonus": 50.0,
            "collision_penalty": 0.0,
            "regularization_coef": 0.0,
            "lift_threshold": 0.01,
        }

    def _create_termination_conditions(self):
        return {
            "goal": PrivilegedInGoalRegion(
                self.obj_name,
                success_radius=self.success_radius,
                goal_region_spec=self._goal_region_spec,
                lift_threshold=self._reward_config["lift_threshold"],
            ),
            "timeout": Timeout(max_steps=self._termination_config["max_steps"]),
        }

    def _create_reward_functions(self):
        return {
            "pick_and_lift_privileged": PrivilegedPickAndLiftReward(
                obj_name=self.obj_name,
                **self._reward_config,
            )
        }


__all__ = [
    "PickAndLiftPrivilegedTask",
    "PrivilegedPickAndLiftReward",
    "PrivilegedInGoalRegion",
]
