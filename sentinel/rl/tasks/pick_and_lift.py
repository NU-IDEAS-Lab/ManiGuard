"""PickAndLiftTask — grasp target object and move it to a goal region in the air.

Difference from ``GraspTask``:
- Success is NOT defined by sustained grasp contact alone. The agent must
  deliver the target object into a configurable goal ball above its spawn
  pose. This matches the standard pick-and-place evaluation used in
  LIBERO, ManiSkill, RoboSuite — closes the "touch-and-quit" reward hack
  that pure contact termination allows.
- A visual goal marker (translucent sphere, ``PrimitiveObject`` with
  ``visual_only=True``) is spawned at the goal position so the policy's
  RGB observations include a spatial cue toward the target region.

Structure mirrors ``GraspTask`` for _load / _reset_scene / _reset_agent so
collectors and diverse-reset samplers that target grasp poses on this
object (``sentinel.rl.resets`` pipeline) transfer directly.
"""

from __future__ import annotations

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.tasks.grasp_task import GraspTask
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.python_utils import classproperty


class InGoalRegion(SuccessCondition):
    """Success when the target object sits within ``success_radius`` of the
    task's ``_goal_world_pos`` AND the robot is still grasping it."""

    def __init__(self, obj_name: str, success_radius: float = 0.05):
        self.obj_name = obj_name
        self.success_radius = success_radius

    def _step(self, task, env, action):
        obj = env.scene.object_registry("name", self.obj_name)
        if obj is None or task._goal_world_pos is None:
            return False
        obj_pos = obj.get_position_orientation()[0]
        dist = th.norm(obj_pos - task._goal_world_pos)
        if dist >= self.success_radius:
            return False
        # Require the object to still be in hand (guards against "launch it
        # toward the goal and let it fly" exploits).
        robot = env.robots[0]
        in_hand = robot._ag_obj_in_hand.get(robot.default_arm)
        return in_hand is obj

    def reset(self, task, env):
        pass


class PickAndLiftReward(BaseRewardFunction):
    """Reward: approach → grasp → carry to goal.

    Dense shaping on (eef → obj) distance while not grasping and on
    (obj → goal) distance while grasping, plus a large sparse bonus on
    entering the goal region (terminal). All distances are shaped
    *delta-wise* (prev − cur) so hovering yields 0 reward — same design as
    the delta-distance fix we applied earlier to GraspReward.
    """

    def __init__(
        self,
        obj_name: str,
        pregrasp_dist_coeff: float = 1.0,
        carry_dist_coeff: float = 2.0,
        grasp_reward: float = 1.0,
        goal_bonus: float = 50.0,
        collision_penalty: float = 1.0,
        regularization_coef: float = 0.0,
    ):
        super().__init__()
        self.obj_name = obj_name
        self.pregrasp_dist_coeff = pregrasp_dist_coeff
        self.carry_dist_coeff = carry_dist_coeff
        self.grasp_reward = grasp_reward
        self.goal_bonus = goal_bonus
        self.collision_penalty = collision_penalty
        self.regularization_coef = regularization_coef
        self._prev_eef_to_obj = None
        self._prev_obj_to_goal = None
        self._was_grasping = False
        self._obj = None

    def _step(self, task, env, action):
        if self._obj is None:
            self._obj = env.scene.object_registry("name", self.obj_name)
        robot = env.robots[0]
        obj_pos = self._obj.get_position_orientation()[0]
        eef_pos = robot.get_eef_position(robot.default_arm)

        in_hand = robot._ag_obj_in_hand.get(robot.default_arm)
        grasping = in_hand is self._obj

        reward = 0.0
        info = {"grasping": grasping}

        # Collision penalty
        if detect_robot_collision_in_sim(robot, filter_objs=[self._obj]):
            reward -= self.collision_penalty
            info["collision"] = True

        # Action-magnitude regularization
        if self.regularization_coef > 0:
            action_mag = th.sum(th.abs(action))
            reward -= float(action_mag) * self.regularization_coef
            info["regularization"] = -float(action_mag) * self.regularization_coef

        if grasping:
            reward += self.grasp_reward
            info["grasp_reward"] = self.grasp_reward
            # Shape on moving object toward goal (delta distance)
            obj_to_goal = th.norm(obj_pos - task._goal_world_pos)
            if self._was_grasping and self._prev_obj_to_goal is not None:
                delta = self._prev_obj_to_goal - obj_to_goal
                shaping = float(delta) * self.carry_dist_coeff
                reward += shaping
                info["carry_shaping"] = shaping
            self._prev_obj_to_goal = obj_to_goal
            self._prev_eef_to_obj = None

            # Sparse goal bonus (also terminal via InGoalRegion)
            if float(obj_to_goal) < task.success_radius:
                reward += self.goal_bonus
                info["goal_bonus"] = self.goal_bonus
        else:
            # Shape on eef → object approach
            eef_to_obj = th.norm(eef_pos - obj_pos)
            if not self._was_grasping and self._prev_eef_to_obj is not None:
                delta = self._prev_eef_to_obj - eef_to_obj
                shaping = float(delta) * self.pregrasp_dist_coeff
                reward += shaping
                info["pregrasp_shaping"] = shaping
            self._prev_eef_to_obj = eef_to_obj
            self._prev_obj_to_goal = None

        self._was_grasping = grasping
        return reward, info

    def reset(self, task, env):
        super().reset(task, env)
        self._prev_eef_to_obj = None
        self._prev_obj_to_goal = None
        self._was_grasping = False
        self._obj = None


class PickAndLiftTask(GraspTask):
    """Grasp target + carry it to a goal ball above the spawn location.

    Extends ``GraspTask``: reuses the precached-reset-pose / primitive-based
    arm reset logic, and the `objects_config`-driven object spawning. Adds
    the goal-region termination + visualization and swaps in a carry-shaped
    reward.

    Args:
        obj_name: name of the target object in the scene (as in GraspTask).
        goal_offset: (3,) offset from the target's initial world position at
            task load time. Default (0, 0, 0.15) = 15 cm straight up.
        success_radius: target must reach within this radius of the goal
            to trigger InGoalRegion termination. Default 5 cm.
        visualize_goal: if True, spawn a translucent ``PrimitiveObject``
            sphere at the goal position (rendered in env cameras).
        goal_marker_rgba: color + alpha of the goal marker.
        termination_config, reward_config, include_obs, precached_reset_pose_path,
        objects_config: forwarded to GraspTask.
    """

    def __init__(
        self,
        obj_name,
        goal_offset=(0.0, 0.0, 0.15),
        success_radius=0.05,
        visualize_goal=True,
        goal_marker_rgba=(0.1, 0.8, 0.2, 0.35),
        termination_config=None,
        reward_config=None,
        include_obs=True,
        precached_reset_pose_path=None,
        objects_config=None,
    ):
        self.goal_offset = th.tensor(goal_offset, dtype=th.float32)
        self.success_radius = float(success_radius)
        self.visualize_goal = bool(visualize_goal)
        self.goal_marker_rgba = tuple(goal_marker_rgba)
        self._goal_world_pos = None
        self._goal_marker = None
        super().__init__(
            obj_name=obj_name,
            termination_config=termination_config,
            reward_config=reward_config,
            include_obs=include_obs,
            precached_reset_pose_path=precached_reset_pose_path,
            objects_config=objects_config,
        )

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 500}

    @classproperty
    def default_reward_config(cls):
        return {
            "pregrasp_dist_coeff": 1.0,
            "carry_dist_coeff": 2.0,
            "grasp_reward": 1.0,
            "goal_bonus": 50.0,
            "collision_penalty": 1.0,
            "regularization_coef": 0.0,
        }

    def _create_termination_conditions(self):
        return {
            "goal": InGoalRegion(self.obj_name, success_radius=self.success_radius),
            "timeout": Timeout(max_steps=self._termination_config["max_steps"]),
        }

    def _create_reward_functions(self):
        return {"pick_and_lift": PickAndLiftReward(obj_name=self.obj_name, **self._reward_config)}

    def _load(self, env):
        # GraspTask spawns objects + precached reset pose config.
        super()._load(env)

        target_obj = env.scene.object_registry("name", self.obj_name)
        if target_obj is None:
            raise ValueError(f"PickAndLiftTask: target object {self.obj_name!r} not in scene.")
        target_init_pos = target_obj.get_position_orientation()[0]
        self._goal_world_pos = target_init_pos + self.goal_offset.to(target_init_pos.device)

        if self.visualize_goal:
            marker_name = f"{self.obj_name}_goal_marker"
            marker = PrimitiveObject(
                relative_prim_path=f"/{marker_name}",
                name=marker_name,
                primitive_type="Sphere",
                radius=self.success_radius,
                visual_only=True,
                rgba=self.goal_marker_rgba,
            )
            env.scene.add_object(marker)
            marker.set_position_orientation(position=self._goal_world_pos)
            self._goal_marker = marker

    def _reset_scene(self, env):
        super()._reset_scene(env)
        # Recompute goal in case the target's init pose drifted (e.g. new scene
        # state on reset). Marker stays static at load-time goal position by
        # design: randomizing the goal each reset would require respawning
        # the marker prim which is expensive.
        target_obj = env.scene.object_registry("name", self.obj_name)
        if target_obj is not None:
            init_pos = target_obj.get_position_orientation()[0]
            self._goal_world_pos = init_pos + self.goal_offset.to(init_pos.device)
            if self._goal_marker is not None:
                self._goal_marker.set_position_orientation(position=self._goal_world_pos)
