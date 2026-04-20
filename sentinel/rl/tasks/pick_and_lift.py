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
object (``sentinel.rl.grasps`` pipeline) transfer directly.
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
from omnigibson.object_states import Touching
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.python_utils import classproperty


# Physical grasping_mode compatibility: OG's ``_ag_obj_in_hand`` attribute is
# only populated by the AssistiveGrasp / StickyGrasp controllers. Under
# ``grasping_mode="physical"`` (what the grasp-dataset collector + resetter
# operate on) it is always empty. We substitute a physics-derived check:
# "gripper is still in contact with the target AND the target has risen off
# the initial resting surface". This captures the same semantic intent as AG
# ("robot is currently holding the object") while remaining valid for
# physical-friction grasps, and it still blocks the ``launch-and-fly``
# exploit that the AG check originally guarded against.
_LIFT_HOLD_THRESHOLD = 0.02  # 2 cm above initial z to count as "lifted"


def _is_physically_holding(robot, obj, init_obj_z: float) -> bool:
    """Return True if the robot is in contact with ``obj`` and ``obj`` has
    risen at least ``_LIFT_HOLD_THRESHOLD`` above its initial z."""
    try:
        if not robot.states[Touching].get_value(obj):
            return False
    except Exception:  # noqa: BLE001 — conservative fallback if state unavailable
        return False
    obj_z = float(obj.get_position_orientation()[0][2])
    return obj_z > init_obj_z + _LIFT_HOLD_THRESHOLD


class InGoalRegion(SuccessCondition):
    """Success when the target object sits within ``success_radius`` of the
    task's ``_goal_world_pos`` AND the robot is still physically holding it."""

    def __init__(self, obj_name: str, success_radius: float = 0.05):
        self.obj_name = obj_name
        self.success_radius = success_radius
        self._init_obj_z: float | None = None

    def _step(self, task, env, action):
        obj = env.scene.object_registry("name", self.obj_name)
        if obj is None or task._goal_world_pos is None:
            return False
        obj_pos = obj.get_position_orientation()[0]
        dist = th.norm(obj_pos - task._goal_world_pos)
        if dist >= self.success_radius:
            return False
        # Guard against launch-and-fly: object must still be held.
        if self._init_obj_z is None:
            return False  # reset not yet captured init_z
        return _is_physically_holding(env.robots[0], obj, self._init_obj_z)

    def reset(self, task, env):
        obj = env.scene.object_registry("name", self.obj_name)
        if obj is not None:
            self._init_obj_z = float(obj.get_position_orientation()[0][2])


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
        self._init_obj_z: float | None = None

    def _step(self, task, env, action):
        if self._obj is None:
            self._obj = env.scene.object_registry("name", self.obj_name)
        robot = env.robots[0]
        obj_pos = self._obj.get_position_orientation()[0]
        eef_pos = robot.get_eef_position(robot.default_arm)

        if self._init_obj_z is None:
            # Hit only if reset() somehow ran before init_obj_z was set.
            self._init_obj_z = float(obj_pos[2])

        grasping = _is_physically_holding(robot, self._obj, self._init_obj_z)

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
        self._obj = env.scene.object_registry("name", self.obj_name)
        if self._obj is not None:
            self._init_obj_z = float(self._obj.get_position_orientation()[0][2])
        else:
            self._init_obj_z = None


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
        grasp_dataset_path=None,
        grasp_reset_pose_range_b=None,
        grasp_reset_max_retries=5,
    ):
        self.goal_offset = th.tensor(goal_offset, dtype=th.float32)
        self.success_radius = float(success_radius)
        self.visualize_goal = bool(visualize_goal)
        self.goal_marker_rgba = tuple(goal_marker_rgba)
        self._goal_world_pos = None
        self._goal_marker = None

        # Optional reset-from-grasp wiring (OmniReset-style). If set, each
        # _reset_agent call tries to IK a saved grasp and teleport the arm+
        # gripper into it; falls back to the precached ready pose on IK fail.
        self._grasp_dataset_path = grasp_dataset_path
        self._grasp_reset_pose_range_b = grasp_reset_pose_range_b
        self._grasp_reset_max_retries = int(grasp_reset_max_retries)
        self._grasp_resetter = None  # lazy construction (needs env + robot)

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

    def _reset_agent(self, env):
        """Reset agent — precached base/joint pose, then optional grasp reset.

        Overrides ``GraspTask._reset_agent`` to (a) fix the upstream bug that
        assumes ``trunk_control_idx`` exists on every robot (FrankaMounted has
        no trunk) and (b) layer OmniReset-style grasp-dataset reset on top.

        Flow:
          1. Release any active grasps.
          2. If ``_reset_poses`` (precached ready-pose) is available, apply
             base pose + joint pose. Arm-only join idx fallback for robots
             without a trunk.
          3. If ``grasp_dataset_path`` is set, lazily build a
             ``GraspDatasetResetter`` and overwrite the arm+gripper joint
             config with a saved grasp. Silently falls back to the precached
             ready-pose on IK failure.
        """
        import random
        robot = env.robots[0]
        for arm in robot.arm_names:
            robot.release_grasp_immediately(arm=arm)

        # Precached ready-pose (base + arm joints). Supports Franka-style
        # robots with no trunk joint.
        if self._reset_poses is not None:
            if hasattr(robot, "trunk_control_idx"):
                joint_control_idx = th.cat(
                    [robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]]
                )
            else:
                joint_control_idx = robot.arm_control_idx[robot.default_arm]

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

        # Optional: overwrite arm+gripper with an IK-solved saved grasp.
        if self._grasp_dataset_path is None:
            return

        target_obj = env.scene.object_registry("name", self.obj_name)
        if target_obj is None:
            return

        if self._grasp_resetter is None:
            from sentinel.rl.grasps.reset import GraspDatasetResetter
            self._grasp_resetter = GraspDatasetResetter(
                env=env,
                robot=robot,
                target_obj=target_obj,
                dataset_path=self._grasp_dataset_path,
                pose_range_b=self._grasp_reset_pose_range_b,
                max_retries=self._grasp_reset_max_retries,
            )

        self._grasp_resetter.reset_eef()
