from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from omnigibson.macros import gm

from rlinf.envs.sentinel.embodiment_profile import get_sentinel_embodiment_profile
from rlinf.envs.sentinel.registry import (
    DEFAULT_SENTINEL_ROBOT_NAME,
    SentinelSceneSpec,
    build_runtime_scene_info,
    build_scene_registry,
    extract_scene_robot_setup,
    slice_scene_registry_for_worker,
    strip_scene_robots_from_scene_info,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True

DEFAULT_POLICY_WRIST_LOCAL_POSITION_OFFSET = (0.02, 0.0, 0.04)


def policy_gripper_scalar_from_joint_positions(
    gripper_positions: torch.Tensor,
    gripper_lower: torch.Tensor,
    gripper_upper: torch.Tensor,
) -> torch.Tensor:
    """Convert OmniGibson finger joints to Pi0.5 gripper scalar (0=open, 1=closed)."""
    gripper_positions = gripper_positions.to(torch.float32)
    gripper_lower = gripper_lower.to(torch.float32)
    gripper_upper = gripper_upper.to(torch.float32)
    gripper_denom = torch.clamp(gripper_upper - gripper_lower, min=1e-6)
    open_fraction = ((gripper_positions - gripper_lower) / gripper_denom).clamp(0.0, 1.0)
    close_fraction = 1.0 - open_fraction
    return torch.mean(close_fraction).reshape(1)


def gripper_joint_positions_from_policy_scalar(
    gripper_scalar: torch.Tensor | float,
    gripper_lower: torch.Tensor,
    gripper_upper: torch.Tensor,
) -> torch.Tensor:
    """Convert Pi0.5 gripper scalar (0=open, 1=closed) to OmniGibson finger joints."""
    gripper_lower = gripper_lower.to(torch.float32)
    gripper_upper = gripper_upper.to(torch.float32)
    gripper_scalar = torch.as_tensor(gripper_scalar, dtype=torch.float32).reshape(1).clamp(0.0, 1.0)
    open_fraction = 1.0 - gripper_scalar
    return gripper_lower + (gripper_upper - gripper_lower) * open_fraction


def resolve_wrist_sensor_name(
    robot_obs: dict,
    *,
    suffix_priority: tuple[str, ...] | None = None,
    token_priority: tuple[tuple[str, ...], ...] | None = None,
) -> tuple[str | None, str, list[str]]:
    """Resolve the wrist RGB sensor deterministically."""

    rgb_sensor_names = [
        sensor_name
        for sensor_name, sensor_obs in robot_obs.items()
        if isinstance(sensor_obs, dict) and "rgb" in sensor_obs
    ]
    if not rgb_sensor_names:
        return None, "no_rgb_sensor", []

    lowered_to_name = {sensor_name.lower(): sensor_name for sensor_name in rgb_sensor_names}

    exact_suffix_priority = suffix_priority or (
        ":camera_link:camera:0",
        ":eef_link:camera:0",
    )
    for suffix in exact_suffix_priority:
        for lowered_name, sensor_name in lowered_to_name.items():
            if lowered_name.endswith(suffix):
                return sensor_name, f"exact_suffix:{suffix}", rgb_sensor_names

    token_priority = token_priority or (
        ("camera_link", "camera"),
        ("eef_link", "camera"),
        ("wrist",),
        ("realsense",),
        ("d405",),
        ("hand",),
    )
    for tokens in token_priority:
        for lowered_name, sensor_name in lowered_to_name.items():
            if all(token in lowered_name for token in tokens):
                return sensor_name, f"token_match:{'+'.join(tokens)}", rgb_sensor_names

    if len(rgb_sensor_names) == 1:
        return rgb_sensor_names[0], "single_rgb_fallback", rgb_sensor_names

    sensor_name = sorted(rgb_sensor_names)[0]
    return sensor_name, "sorted_rgb_fallback", rgb_sensor_names


def compute_policy_wrist_local_pose(
    base_local_position,
    *,
    local_position_offset=None,
    local_position_override=None,
) -> torch.Tensor:
    """Compute the runtime wrist-camera local pose override."""

    base_local_position = torch.as_tensor(base_local_position, dtype=torch.float32).reshape(3)
    if local_position_override is not None:
        return torch.as_tensor(local_position_override, dtype=torch.float32).reshape(3)
    if local_position_offset is None:
        local_position_offset = DEFAULT_POLICY_WRIST_LOCAL_POSITION_OFFSET
    return base_local_position + torch.as_tensor(local_position_offset, dtype=torch.float32).reshape(3)


def compute_policy_wrist_local_orientation(
    base_local_orientation,
    *,
    local_orientation_override=None,
) -> torch.Tensor:
    """Resolve the runtime wrist-camera local orientation override."""

    if local_orientation_override is None:
        return torch.as_tensor(base_local_orientation, dtype=torch.float32).reshape(4)
    return torch.as_tensor(local_orientation_override, dtype=torch.float32).reshape(4)


def synthesize_scene_relative_ready_eef_position(
    robot_base_position,
    workspace_center,
    *,
    surface_bounds_xy=None,
    table_top_z: float,
    object_top_z: float,
    workspace_standoff_m: float,
    height_above_table_m: float,
    min_object_clearance_m: float,
    surface_margin_m: float,
) -> torch.Tensor:
    """Build a benchmark-owned ready EEF position from scene geometry instead of static joints."""

    robot_base_position = torch.as_tensor(robot_base_position, dtype=torch.float32).reshape(3)
    workspace_center = torch.as_tensor(workspace_center, dtype=torch.float32).reshape(3)

    toward_workspace = workspace_center[:2] - robot_base_position[:2]
    norm = torch.linalg.norm(toward_workspace)
    if float(norm) < 1e-6:
        forward_xy = torch.tensor([0.0, 1.0], dtype=torch.float32)
    else:
        forward_xy = toward_workspace / norm

    desired_xy = workspace_center[:2] - forward_xy * float(workspace_standoff_m)
    if surface_bounds_xy is not None:
        (x0, y0), (x1, y1) = surface_bounds_xy
        desired_xy[0] = desired_xy[0].clamp(float(x0) + float(surface_margin_m), float(x1) - float(surface_margin_m))
        desired_xy[1] = desired_xy[1].clamp(float(y0) + float(surface_margin_m), float(y1) - float(surface_margin_m))

    desired_z = max(
        float(table_top_z) + float(height_above_table_m),
        float(object_top_z) + float(min_object_clearance_m),
    )
    return torch.tensor([desired_xy[0], desired_xy[1], desired_z], dtype=torch.float32)


def _workspace_xy_basis(
    robot_base_position,
    workspace_center,
    *,
    fallback_forward=(0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    robot_base_position = np.asarray(robot_base_position, dtype=np.float32).reshape(3)
    workspace_center = np.asarray(workspace_center, dtype=np.float32).reshape(3)
    toward_workspace = workspace_center[:2] - robot_base_position[:2]
    norm = float(np.linalg.norm(toward_workspace))
    if norm < 1e-6:
        forward = np.asarray(fallback_forward, dtype=np.float32)
    else:
        forward = toward_workspace / norm
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    return forward.astype(np.float32), lateral.astype(np.float32)


def _surface_diag_xy(surface_bounds_xy) -> float:
    if surface_bounds_xy is None:
        return 0.0
    (x0, y0), (x1, y1) = surface_bounds_xy
    return float(np.linalg.norm(np.asarray([x1 - x0, y1 - y0], dtype=np.float32)))


def pull_back_camera_eye(
    eye,
    lookat,
    *,
    pullback_m: float,
) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32).reshape(3)
    lookat = np.asarray(lookat, dtype=np.float32).reshape(3)
    if float(pullback_m) <= 0.0:
        return eye

    ray = eye - lookat
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm < 1e-6:
        return eye

    return (lookat + ray / ray_norm * (ray_norm + float(pullback_m))).astype(np.float32)


def build_tabletop_oblique_support_view(
    robot_base_position,
    workspace_center,
    *,
    table_top_z: float,
    surface_bounds_xy=None,
    backoff_m: float,
    lateral_offset_m: float,
    height_above_table_m: float,
    lookat_height_above_table_m: float,
    pullback_m: float = 0.0,
) -> dict:
    forward, lateral = _workspace_xy_basis(robot_base_position, workspace_center)
    workspace_center = np.asarray(workspace_center, dtype=np.float32).reshape(3)
    backoff = float(backoff_m)
    lateral_mag = float(lateral_offset_m)

    eye = np.asarray(
        [
            float(workspace_center[0] - forward[0] * backoff + lateral[0] * lateral_mag),
            float(workspace_center[1] - forward[1] * backoff + lateral[1] * lateral_mag),
            float(table_top_z + float(height_above_table_m)),
        ],
        dtype=np.float32,
    )
    lookat = np.asarray(
        [
            float(workspace_center[0]),
            float(workspace_center[1]),
            float(table_top_z + float(lookat_height_above_table_m)),
        ],
        dtype=np.float32,
    )
    eye = pull_back_camera_eye(eye, lookat, pullback_m=pullback_m)
    return {
        "label": "tabletop_oblique_support",
        "eye": [float(v) for v in eye],
        "lookat": [float(v) for v in lookat],
    }


def build_droid_left_shoulder_view(
    robot_base_position,
    target_position,
    workspace_center,
    *,
    table_top_z: float,
    forward_backoff_m: float,
    lateral_offset_m: float,
    height_above_table_m: float,
    pullback_m: float = 0.0,
) -> dict:
    robot_base_position = np.asarray(robot_base_position, dtype=np.float32).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    workspace_center = np.asarray(workspace_center, dtype=np.float32).reshape(3)
    forward, lateral = _workspace_xy_basis(robot_base_position, workspace_center, fallback_forward=(0.0, 1.0))
    eye = np.asarray(
        [
            float(robot_base_position[0] - forward[0] * float(forward_backoff_m) + lateral[0] * float(lateral_offset_m)),
            float(robot_base_position[1] - forward[1] * float(forward_backoff_m) + lateral[1] * float(lateral_offset_m)),
            float(max(table_top_z + float(height_above_table_m), robot_base_position[2] + 0.96, target_position[2] + 0.58)),
        ],
        dtype=np.float32,
    )
    eye = pull_back_camera_eye(eye, workspace_center, pullback_m=pullback_m)
    return {
        "label": "droid_left_shoulder",
        "eye": [float(v) for v in eye],
        "lookat": [float(v) for v in workspace_center],
    }


def build_workspace_camera_lookat_point(
    workspace_center,
    *,
    table_top_z: float,
    object_top_z: float,
    height_above_table_m: float,
) -> np.ndarray:
    workspace_center = np.asarray(workspace_center, dtype=np.float32).reshape(3)
    return np.asarray(
        [
            float(workspace_center[0]),
            float(workspace_center[1]),
            float(max(table_top_z + float(height_above_table_m), min(object_top_z, table_top_z + 0.18))),
        ],
        dtype=np.float32,
    )


class SentinelEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.video_cfg = cfg.video_cfg
        self.sentinel_cfg = cfg.sentinel_cfg
        self.embodiment_profile = get_sentinel_embodiment_profile(
            self.sentinel_cfg.get("embodiment_profile")
        )
        self.reset_settle_steps = int(
            self.sentinel_cfg.get("reset_settle_steps", self.embodiment_profile.reset_settle_steps)
        )
        self.ltl_violation_grace_steps = int(self.cfg.get("ltl_violation_grace_steps", 0))
        self.terminate_on_ltl_violation = bool(self.cfg.get("terminate_on_ltl_violation", False))
        self._is_start = True

        gm.HEADLESS = bool(self.sentinel_cfg.get("headless", False))
        gm.USE_GPU_DYNAMICS = bool(self.sentinel_cfg.get("use_gpu_dynamics", False))
        gm.ENABLE_FLATCACHE = not gm.USE_GPU_DYNAMICS

        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self._frames = [[] for _ in range(self.num_envs)]
        self._camera_views = [None for _ in range(self.num_envs)]
        self._sensor_debug_infos = [None for _ in range(self.num_envs)]
        self._policy_start_frames = [None for _ in range(self.num_envs)]
        self._wrist_start_frames = [None for _ in range(self.num_envs)]
        self._reset_infos = [None for _ in range(self.num_envs)]
        self._reset_object_summaries = [None for _ in range(self.num_envs)]
        self._ltl_monitors = [None for _ in range(self.num_envs)]
        self._ltl_violation_first_steps = [None for _ in range(self.num_envs)]
        self._episode_indices = [0 for _ in range(self.num_envs)]

        self._scene_registry = build_scene_registry(
            benchmark_root=self.sentinel_cfg.benchmark_root,
            activity_root=self.sentinel_cfg.activity_root,
            scene_names=self.sentinel_cfg.get("scene_names"),
            max_scenes=self.sentinel_cfg.get("max_scenes"),
        )
        self._scene_specs = slice_scene_registry_for_worker(
            registry=self._scene_registry,
            num_envs=self.num_envs,
            seed_offset=self.seed_offset,
        )

        self._results_dir = Path(self.video_cfg.video_base_dir).resolve().parent
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self._results_dir / "sentinel_eval_results.jsonl"
        self._prompt_log_path = self._results_dir / "sentinel_prompts.jsonl"
        self._runtime_scene_root = self._results_dir / "_runtime_scene_cache"
        self._runtime_scene_root.mkdir(parents=True, exist_ok=True)
        self._reset_policy_state_override = self._load_reset_policy_state_override()

        self._init_envs()

    def _debug_reset_trace(self, message: str) -> None:
        if bool(self.sentinel_cfg.get("debug_reset_trace", False)):
            print(f"[SentinelEnv.reset] {message}", flush=True)

    def _init_envs(self):
        import omnigibson as og

        if og.sim is not None:
            og.sim.stop()

        self._patch_activity_root()
        base_cfg = OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True)
        self.envs = []
        for spec in self._scene_specs:
            env_cfg = self._make_env_cfg(base_cfg, spec)
            self.envs.append(og.Environment(configs=env_cfg, in_vec_env=True))

        og.sim.play()
        for env in self.envs:
            env.post_play_load()

    def _make_env_cfg(self, base_cfg: dict, spec: SentinelSceneSpec) -> dict:
        env_cfg = copy.deepcopy(base_cfg)
        self._apply_embodiment_profile_to_env_cfg(env_cfg)
        env_cfg["scene"]["scene_model"] = spec.scene_name
        runtime_scene_file, scene_robot_setup = self._prepare_runtime_scene_file(spec)
        env_cfg["scene"]["scene_file"] = runtime_scene_file
        env_cfg["scene"]["scene_instance"] = None
        env_cfg["scene"]["include_robots"] = False
        with open(spec.problem_file, "r", encoding="utf-8") as handle:
            env_cfg["task"]["predefined_problem"] = handle.read()
        env_cfg["task"]["activity_name"] = spec.activity_name
        env_cfg["task"]["activity_definition_id"] = 0
        env_cfg["task"]["activity_instance_id"] = 0
        env_cfg["task"]["online_object_sampling"] = False
        env_cfg["task"]["use_presampled_robot_pose"] = False
        env_cfg["task"].setdefault("termination_config", {})
        env_cfg["task"]["termination_config"]["max_steps"] = int(self.cfg.max_episode_steps) + 1
        self._configure_runtime_robot(env_cfg, scene_robot_setup)
        return env_cfg

    def _apply_embodiment_profile_to_env_cfg(self, env_cfg: dict) -> None:
        external_sensors = env_cfg.setdefault("env", {}).setdefault("external_sensors", [])
        if external_sensors:
            sensor_kwargs = external_sensors[0].setdefault("sensor_kwargs", {})
            sensor_kwargs["image_height"] = int(self.embodiment_profile.external_camera_resolution[0])
            sensor_kwargs["image_width"] = int(self.embodiment_profile.external_camera_resolution[1])

        robots_cfg = env_cfg.setdefault("robots", [])
        if robots_cfg:
            sensor_kwargs = (
                robots_cfg[0]
                .setdefault("sensor_config", {})
                .setdefault("VisionSensor", {})
                .setdefault("sensor_kwargs", {})
            )
            sensor_kwargs["image_height"] = int(self.embodiment_profile.wrist_camera_resolution[0])
            sensor_kwargs["image_width"] = int(self.embodiment_profile.wrist_camera_resolution[1])

    def _configure_runtime_robot(self, env_cfg: dict, scene_robot_setup: dict | None) -> None:
        robots_cfg = env_cfg.setdefault("robots", [])
        if not robots_cfg:
            raise ValueError("Sentinel env config must define at least one robot.")

        robot_cfg = robots_cfg[0]
        robot_cfg["type"] = self.embodiment_profile.robot_type
        robot_cfg["name"] = self.embodiment_profile.robot_name or DEFAULT_SENTINEL_ROBOT_NAME
        robot_cfg["action_normalize"] = False
        if scene_robot_setup is not None:
            if scene_robot_setup.get("position") is not None:
                robot_cfg["position"] = scene_robot_setup["position"]
            if scene_robot_setup.get("orientation") is not None:
                robot_cfg["orientation"] = scene_robot_setup["orientation"]
            if scene_robot_setup.get("reset_joint_pos") is not None:
                robot_cfg["reset_joint_pos"] = scene_robot_setup["reset_joint_pos"]
        controller_cfg = robot_cfg.setdefault("controller_config", {})
        arm_cfg = controller_cfg.setdefault("arm_0", {})
        arm_cfg.update(
            {
                "name": self.embodiment_profile.arm_controller_name,
                "motor_type": self.embodiment_profile.arm_motor_type,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_delta_commands": self.embodiment_profile.arm_use_delta_commands,
                "use_impedances": self.embodiment_profile.arm_use_impedances,
            }
        )

        gripper_cfg = controller_cfg.setdefault("gripper_0", {})
        gripper_cfg.update(
            {
                "name": self.embodiment_profile.gripper_controller_name,
                "mode": self.embodiment_profile.gripper_mode,
                "inverted": self.embodiment_profile.gripper_inverted,
                "command_input_limits": [
                    list(self.embodiment_profile.gripper_input_limits[0]),
                    list(self.embodiment_profile.gripper_input_limits[1]),
                ],
                "command_output_limits": "default",
            }
        )

    def _load_reset_policy_state_override(self) -> torch.Tensor | None:
        values = self.sentinel_cfg.get("robot_reset_policy_state")
        if values is None:
            return None
        state = torch.as_tensor(values, dtype=torch.float32)
        if state.numel() != 8:
            raise ValueError(
                "Sentinel robot_reset_policy_state must contain 8 values "
                f"(7 joints + 1 gripper), got {state.numel()}"
            )
        return state.reshape(8)

    def _apply_profile_post_reset_ready_pose(self, env_idx: int, env, spec: SentinelSceneSpec) -> dict | None:
        if self.embodiment_profile.reset_pose_mode != "scene_relative_ready_eef_v1":
            return None

        import omnigibson.utils.transform_utils as T
        from omnigibson.controllers.ik_controller import _compute_ik_qpos_torch

        robot = env.robots[0]
        target_obj = self._object_by_name(env, spec.target_object_name)
        support_obj = self._object_by_name(env, spec.support_object_name)
        if target_obj is None or support_obj is None:
            return {
                "source": f"{self.embodiment_profile.name}:{self.embodiment_profile.reset_pose_mode}",
                "applied": False,
                "reason": "missing_target_or_support",
            }

        geometry = self._compute_workspace_geometry(env, target_obj, support_obj)
        desired_world_pos = synthesize_scene_relative_ready_eef_position(
            geometry["robot_base_pos"],
            geometry["workspace_center"],
            surface_bounds_xy=geometry["surface_bounds_xy"],
            table_top_z=geometry["table_top_z"],
            object_top_z=geometry["object_top_z"],
            workspace_standoff_m=self.embodiment_profile.ready_eef_standoff_m,
            height_above_table_m=self.embodiment_profile.ready_eef_height_above_table_m,
            min_object_clearance_m=self.embodiment_profile.ready_eef_min_object_clearance_m,
            surface_margin_m=self.embodiment_profile.ready_eef_surface_margin_m,
        )

        base_pos, base_quat = robot.get_position_orientation()
        _, eef_quat_world = robot.eef_links[robot.default_arm].get_position_orientation()
        desired_local_pos, desired_local_quat = T.relative_pose_transform(
            desired_world_pos,
            eef_quat_world,
            base_pos,
            base_quat,
        )

        arm_idx = robot.arm_control_idx[robot.default_arm]
        gripper_idx = robot.gripper_control_idx[robot.default_arm]
        full_joint_positions = robot.get_joint_positions().to(torch.float32).clone()
        solved_arm = full_joint_positions[arm_idx].clone()
        solve_mode = "jacobian_iterative"
        goal_ori_mat = T.quat2mat(desired_local_quat.to(torch.float32))
        for _ in range(3):
            full_joint_positions[arm_idx] = solved_arm
            robot.set_joint_positions(positions=full_joint_positions, drive=False)
            robot.keep_still()
            control_dict = robot.get_control_dict()
            task_name = f"eef_{robot.default_arm}"
            solved_arm = _compute_ik_qpos_torch(
                q=torch.as_tensor(control_dict["joint_position"][arm_idx], dtype=torch.float32),
                j_eef=torch.as_tensor(control_dict[f"{task_name}_jacobian_relative"][:, arm_idx], dtype=torch.float32),
                ee_pos=torch.as_tensor(control_dict[f"{task_name}_pos_relative"], dtype=torch.float32),
                ee_mat=T.quat2mat(torch.as_tensor(control_dict[f"{task_name}_quat_relative"], dtype=torch.float32)),
                goal_pos=desired_local_pos.to(torch.float32),
                goal_ori_mat=goal_ori_mat,
                q_lower_limit=robot.joint_lower_limits[arm_idx].to(torch.float32),
                q_upper_limit=robot.joint_upper_limits[arm_idx].to(torch.float32),
            )

        full_joint_positions[arm_idx] = solved_arm.to(torch.float32)
        if self.embodiment_profile.ready_gripper_scalar is not None:
            full_joint_positions[gripper_idx] = gripper_joint_positions_from_policy_scalar(
                self.embodiment_profile.ready_gripper_scalar,
                robot.joint_lower_limits[gripper_idx],
                robot.joint_upper_limits[gripper_idx],
            )

        robot.set_joint_positions(positions=full_joint_positions, drive=False)
        robot.keep_still()
        final_local_pos = robot.get_relative_eef_position().to(torch.float32)
        final_local_quat = robot.get_relative_eef_orientation().to(torch.float32)
        try:
            robot.reset_joint_pos = full_joint_positions.clone()
        except Exception:
            pass

        return {
            "source": f"{self.embodiment_profile.name}:{self.embodiment_profile.reset_pose_mode}",
            "applied": True,
            "solve_mode": solve_mode,
            "workspace_center": [float(v) for v in geometry["workspace_center"].tolist()],
            "desired_world_position": [float(v) for v in desired_world_pos.tolist()],
            "desired_local_position": [float(v) for v in desired_local_pos.to(torch.float32).tolist()],
            "desired_local_orientation": [float(v) for v in desired_local_quat.to(torch.float32).tolist()],
            "arm_joint_positions": [float(v) for v in solved_arm.to(torch.float32).tolist()],
            "final_local_position": [float(v) for v in final_local_pos.tolist()],
            "final_local_orientation": [float(v) for v in final_local_quat.tolist()],
            "gripper_scalar": self.embodiment_profile.ready_gripper_scalar,
        }

    def _apply_policy_reset_state_override(self, env) -> dict | None:
        if self._reset_policy_state_override is None:
            return None

        robot = env.robots[0]
        state = self._reset_policy_state_override.to(torch.float32)
        arm_idx = robot.arm_control_idx[robot.default_arm]
        gripper_idx = robot.gripper_control_idx[robot.default_arm]
        if len(arm_idx) != 7:
            raise ValueError(f"Expected 7 Franka arm DOFs, got {len(arm_idx)}")

        joint_positions = robot.get_joint_positions().to(torch.float32).clone()
        joint_positions[arm_idx] = state[:7]
        joint_positions[gripper_idx] = gripper_joint_positions_from_policy_scalar(
            state[7],
            robot.joint_lower_limits[gripper_idx],
            robot.joint_upper_limits[gripper_idx],
        )
        robot.set_joint_positions(positions=joint_positions, drive=False)
        robot.keep_still()
        try:
            robot.reset_joint_pos = joint_positions.clone()
        except Exception:
            pass

        return {
            "source": str(self.sentinel_cfg.get("robot_reset_source", "custom")),
            "policy_state": [float(v) for v in state.tolist()],
            "joint_positions": [float(v) for v in joint_positions.tolist()],
        }

    def _prepare_runtime_scene_file(self, spec: SentinelSceneSpec) -> tuple[str, dict | None]:
        runtime_scene_dir = self._runtime_scene_root / spec.scene_name
        runtime_scene_dir.mkdir(parents=True, exist_ok=True)
        runtime_scene_path = runtime_scene_dir / "scene_ep1.runtime.json"

        with open(spec.scene_file, "r", encoding="utf-8") as handle:
            scene_info = json.load(handle)
        with open(spec.diagnostics_file, "r", encoding="utf-8") as handle:
            diagnostics = json.loads(next(line for line in handle if line.strip()))
        with open(spec.problem_file, "r", encoding="utf-8") as handle:
            problem_text = handle.read()

        scene_robot_setup = extract_scene_robot_setup(scene_info, robot_name=DEFAULT_SENTINEL_ROBOT_NAME)
        runtime_scene_base = strip_scene_robots_from_scene_info(scene_info)
        runtime_scene_info = build_runtime_scene_info(
            scene_info=runtime_scene_base,
            diagnostics=diagnostics,
            problem_text=problem_text,
        )
        with runtime_scene_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_scene_info, handle, ensure_ascii=True, indent=2)
        return str(runtime_scene_path), scene_robot_setup

    @property
    def device(self):
        return "cpu"

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _patch_activity_root(self):
        import bddl.activity
        import bddl.config
        import omnigibson.tasks.behavior_task as behavior_task
        import omnigibson.utils.bddl_utils as bddl_utils

        activity_root = str(Path(self.sentinel_cfg.activity_root).resolve())
        repo_root = Path(__file__).resolve().parents[4]
        activity_root_path = Path(activity_root)
        domain_src_dir = repo_root / "bddl3" / "bddl" / "activity_definitions"
        for domain_name in ("domain_igibson.bddl", "domain_omnigibson.bddl"):
            domain_src = domain_src_dir / domain_name
            domain_dst = activity_root_path / domain_name
            if domain_src.is_file() and not domain_dst.exists():
                os.symlink(domain_src, domain_dst)

        bddl.config.ACTIVITY_CONFIGS_PATH = activity_root
        bddl.activity.ACTIVITY_CONFIGS_PATH = activity_root
        behavior_task.ACTIVITY_CONFIGS_PATH = activity_root

        activities = sorted(
            p.name for p in activity_root_path.iterdir() if p.is_dir() and (p / "problem0.bddl").is_file()
        )
        bddl_utils.BEHAVIOR_ACTIVITIES.clear()
        bddl_utils.BEHAVIOR_ACTIVITIES.extend(activities)
        behavior_task.BEHAVIOR_ACTIVITIES.clear()
        behavior_task.BEHAVIOR_ACTIVITIES.extend(activities)

    def _object_by_name(self, env, name: str | None):
        if not name:
            return None
        try:
            return env.scene.object_registry("name", name)
        except Exception:
            return None

    def _build_camera_pose(self, eye, lookat):
        import omnigibson.utils.transform_utils as T

        eye = np.asarray(eye, dtype=np.float32)
        lookat = np.asarray(lookat, dtype=np.float32)
        direction = lookat - eye
        direction = direction / max(np.linalg.norm(direction), 1e-6)
        quat = T.euler2quat(
            torch.tensor(
                [
                    math.pi / 2 + float(np.arcsin(np.clip(direction[2], -1.0, 1.0))),
                    0.0,
                    float(np.arctan2(-direction[0], direction[1])),
                ],
                dtype=torch.float32,
            )
        )
        return torch.tensor(eye, dtype=torch.float32), quat.to(torch.float32)

    def _compute_workspace_geometry(self, env, target_obj, support_obj=None):
        robot = env.robots[0]
        rp = np.asarray(robot.get_position_orientation()[0][:3], dtype=np.float32)
        tp = np.asarray(target_obj.get_position_orientation()[0][:3], dtype=np.float32)
        surface_bounds_xy = None

        if support_obj is not None:
            try:
                aabb_min, aabb_max = support_obj.aabb
                x0, y0, _ = [float(v) for v in aabb_min[:3]]
                x1, y1, z1 = [float(v) for v in aabb_max[:3]]
                support_center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z1], dtype=np.float32)
                table_top_z = float(z1)
                surface_bounds_xy = ((x0, y0), (x1, y1))
            except Exception:
                support_center = np.asarray([(rp[0] + tp[0]) * 0.5, (rp[1] + tp[1]) * 0.5, tp[2]], dtype=np.float32)
                table_top_z = float(tp[2])
        else:
            support_center = np.asarray([(rp[0] + tp[0]) * 0.5, (rp[1] + tp[1]) * 0.5, tp[2]], dtype=np.float32)
            table_top_z = float(tp[2])

        cluster_positions = []
        object_top_z = float(tp[2])
        object_scope = getattr(env.task, "object_scope", {}) or {}
        for entity in object_scope.values():
            if entity is None or not getattr(entity, "exists", False):
                continue
            wrapped_obj = getattr(entity, "wrapped_obj", None)
            if wrapped_obj is None:
                continue
            try:
                cluster_positions.append(np.asarray(wrapped_obj.get_position_orientation()[0][:3], dtype=np.float32))
                _, aabb_max = wrapped_obj.aabb
                object_top_z = max(object_top_z, float(aabb_max[2]))
            except Exception:
                continue
        cluster_center = np.mean(np.stack(cluster_positions, axis=0), axis=0) if cluster_positions else tp

        workspace_blend = float(self.embodiment_profile.ready_eef_workspace_blend)
        workspace_center = np.asarray(
            [
                float((1.0 - workspace_blend) * support_center[0] + workspace_blend * cluster_center[0]),
                float((1.0 - workspace_blend) * support_center[1] + workspace_blend * cluster_center[1]),
                float(max(table_top_z + 0.12, cluster_center[2], tp[2])),
            ],
            dtype=np.float32,
        )
        return {
            "robot_base_pos": rp,
            "target_pos": tp,
            "support_center": support_center,
            "cluster_center": cluster_center,
            "workspace_center": workspace_center,
            "table_top_z": table_top_z,
            "object_top_z": object_top_z,
            "surface_bounds_xy": surface_bounds_xy,
        }

    def _build_policy_main_camera_view(self, geometry: dict) -> dict:
        if self.embodiment_profile.main_camera_mode == "droid_left_shoulder_v1":
            return build_droid_left_shoulder_view(
                geometry["robot_base_pos"],
                geometry["target_pos"],
                geometry["workspace_center"],
                table_top_z=geometry["table_top_z"],
                forward_backoff_m=self.embodiment_profile.main_camera_backoff_m,
                lateral_offset_m=self.embodiment_profile.main_camera_lateral_offset_m,
                height_above_table_m=self.embodiment_profile.main_camera_height_above_table_m,
                pullback_m=self.embodiment_profile.main_camera_pullback_m,
            )
        if self.embodiment_profile.main_camera_mode == "tabletop_oblique_support_v1":
            return build_tabletop_oblique_support_view(
                geometry["robot_base_pos"],
                geometry["workspace_center"],
                table_top_z=geometry["table_top_z"],
                surface_bounds_xy=geometry["surface_bounds_xy"],
                backoff_m=self.embodiment_profile.main_camera_backoff_m,
                lateral_offset_m=self.embodiment_profile.main_camera_lateral_offset_m,
                height_above_table_m=self.embodiment_profile.main_camera_height_above_table_m,
                lookat_height_above_table_m=self.embodiment_profile.main_camera_lookat_height_above_table_m,
                pullback_m=self.embodiment_profile.main_camera_pullback_m,
            )
        raise ValueError(
            f"Unsupported Sentinel main camera mode: {self.embodiment_profile.main_camera_mode}"
        )

    def _build_policy_wrist_lookat(self, geometry: dict) -> np.ndarray:
        if self.embodiment_profile.wrist_camera_mode != "workspace_lookat_v1":
            raise ValueError(
                f"Unsupported Sentinel wrist lookat mode: {self.embodiment_profile.wrist_camera_mode}"
            )
        return build_workspace_camera_lookat_point(
            geometry["workspace_center"],
            table_top_z=geometry["table_top_z"],
            object_top_z=geometry["object_top_z"],
            height_above_table_m=self.embodiment_profile.wrist_camera_lookat_height_above_table_m,
        )

    def _apply_policy_wrist_camera(self, env, raw_obs, spec: SentinelSceneSpec):
        robot = env.robots[0]
        robot_obs = raw_obs.get(robot.name, {})
        wrist_sensor_name, resolution_reason, rgb_sensor_names = resolve_wrist_sensor_name(
            robot_obs,
            suffix_priority=self.embodiment_profile.wrist_sensor_suffix_priority,
            token_priority=self.embodiment_profile.wrist_sensor_token_priority,
        )
        if wrist_sensor_name is None:
            return {
                "applied": False,
                "reason": "no_rgb_sensor",
                "available_robot_rgb_sensors": list(rgb_sensor_names),
            }

        sensor = robot.sensors[wrist_sensor_name]
        base_position, base_orientation = sensor.get_position_orientation(frame="parent")
        local_position_offset = self.sentinel_cfg.get(
            "policy_wrist_local_position_offset",
            list(self.embodiment_profile.wrist_local_position_offset),
        )
        local_position_override = self.sentinel_cfg.get("policy_wrist_local_position_override")
        local_orientation_override = self.sentinel_cfg.get(
            "policy_wrist_local_orientation_override",
            self.embodiment_profile.wrist_local_orientation_override,
        )
        applied_local_position = compute_policy_wrist_local_pose(
            base_position,
            local_position_offset=local_position_offset,
            local_position_override=local_position_override,
        )
        sensor.set_position_orientation(
            position=applied_local_position,
            frame="parent",
        )

        orientation_mode = "base_local_orientation"
        desired_world_lookat = None
        if local_orientation_override is not None:
            applied_local_orientation = compute_policy_wrist_local_orientation(
                base_orientation,
                local_orientation_override=local_orientation_override,
            )
            sensor.set_position_orientation(
                orientation=applied_local_orientation,
                frame="parent",
            )
            orientation_mode = "explicit_local_override"
        elif self.embodiment_profile.wrist_camera_mode == "workspace_lookat_v1":
            target_obj = self._object_by_name(env, spec.target_object_name)
            support_obj = self._object_by_name(env, spec.support_object_name)
            if target_obj is not None:
                geometry = self._compute_workspace_geometry(env, target_obj, support_obj)
                desired_world_lookat = self._build_policy_wrist_lookat(geometry)
                world_position, _ = sensor.get_position_orientation()
                _, desired_world_orientation = self._build_camera_pose(
                    eye=world_position.detach().cpu().tolist(),
                    lookat=desired_world_lookat.tolist(),
                )
                sensor.set_position_orientation(
                    orientation=desired_world_orientation,
                    frame="world",
                )
                orientation_mode = self.embodiment_profile.wrist_camera_mode
            else:
                applied_local_orientation = compute_policy_wrist_local_orientation(base_orientation)
                sensor.set_position_orientation(
                    orientation=applied_local_orientation,
                    frame="parent",
                )
        elif self.embodiment_profile.wrist_camera_mode == "mounted_local_v1":
            applied_local_orientation = compute_policy_wrist_local_orientation(base_orientation)
            sensor.set_position_orientation(
                orientation=applied_local_orientation,
                frame="parent",
            )
            orientation_mode = "mounted_local_v1"
        else:
            raise ValueError(
                f"Unsupported Sentinel wrist camera mode: {self.embodiment_profile.wrist_camera_mode}"
            )

        applied_local_position_actual, applied_local_orientation_actual = sensor.get_position_orientation(frame="parent")
        applied_world_position, applied_world_orientation = sensor.get_position_orientation()
        return {
            "applied": True,
            "sensor_name": wrist_sensor_name,
            "resolution_reason": resolution_reason,
            "available_robot_rgb_sensors": list(rgb_sensor_names),
            "orientation_mode": orientation_mode,
            "base_local_position": [float(v) for v in base_position.detach().cpu().tolist()],
            "base_local_orientation": [float(v) for v in base_orientation.detach().cpu().tolist()],
            "applied_local_position": [float(v) for v in applied_local_position_actual.detach().cpu().tolist()],
            "applied_local_orientation": [float(v) for v in applied_local_orientation_actual.detach().cpu().tolist()],
            "applied_world_position": [float(v) for v in applied_world_position.detach().cpu().tolist()],
            "applied_world_orientation": [float(v) for v in applied_world_orientation.detach().cpu().tolist()],
            "local_position_offset": [float(v) for v in torch.as_tensor(local_position_offset, dtype=torch.float32).tolist()],
            "local_position_override": None
            if local_position_override is None
            else [float(v) for v in torch.as_tensor(local_position_override, dtype=torch.float32).tolist()],
            "local_orientation_override": None
            if local_orientation_override is None
            else [float(v) for v in torch.as_tensor(local_orientation_override, dtype=torch.float32).tolist()],
            "workspace_lookat_world": None
            if desired_world_lookat is None
            else [float(v) for v in desired_world_lookat.tolist()],
        }

    def _apply_policy_camera(self, env, spec: SentinelSceneSpec):
        from omnigibson.task_generation.pipeline_common import set_viewer_camera_pose

        target_obj = self._object_by_name(env, spec.target_object_name)
        support_obj = self._object_by_name(env, spec.support_object_name)
        if target_obj is None:
            return
        geometry = self._compute_workspace_geometry(env, target_obj, support_obj)
        canonical_view = self._build_policy_main_camera_view(geometry)
        sensor = env.external_sensors["external_sensor0"]
        position, orientation = self._build_camera_pose(
            eye=canonical_view["eye"],
            lookat=canonical_view["lookat"],
        )
        sensor.set_position_orientation(position=position, orientation=orientation, frame="parent")
        set_viewer_camera_pose(canonical_view["eye"], canonical_view["lookat"])
        actual_local_position, actual_local_orientation = sensor.get_position_orientation(frame="parent")
        actual_world_position, actual_world_orientation = sensor.get_position_orientation()
        return {
            "label": canonical_view.get("label"),
            "eye": [float(v) for v in canonical_view["eye"]],
            "lookat": [float(v) for v in canonical_view["lookat"]],
            "actual_local_position": [float(v) for v in actual_local_position.detach().cpu().tolist()],
            "actual_local_orientation": [float(v) for v in actual_local_orientation.detach().cpu().tolist()],
            "actual_world_position": [float(v) for v in actual_world_position.detach().cpu().tolist()],
            "actual_world_orientation": [float(v) for v in actual_world_orientation.detach().cpu().tolist()],
            "workspace_center": [float(v) for v in geometry["workspace_center"].tolist()],
            "support_center": [float(v) for v in geometry["support_center"].tolist()],
        }

    def _extract_obs(self, env, raw_obs, spec: SentinelSceneSpec):
        external = raw_obs.get("external", {})
        main_rgb = external["external_sensor0"]["rgb"].to(torch.uint8)[..., :3]

        robot = env.robots[0]
        robot_obs = raw_obs[robot.name]
        wrist_sensor_name, wrist_sensor_resolution, rgb_sensor_names = resolve_wrist_sensor_name(
            robot_obs,
            suffix_priority=self.embodiment_profile.wrist_sensor_suffix_priority,
            token_priority=self.embodiment_profile.wrist_sensor_token_priority,
        )
        if wrist_sensor_name is not None:
            wrist_rgb = robot_obs[wrist_sensor_name]["rgb"].to(torch.uint8)[..., :3]
        else:
            wrist_rgb = torch.zeros_like(main_rgb)
            wrist_sensor_name = "synthetic_zeros"
            wrist_sensor_resolution = "synthetic_zeros"
            rgb_sensor_names = []

        arm_positions = robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]]
        gripper_indices = robot.gripper_control_idx[robot.default_arm]
        gripper_positions = robot.get_joint_positions()[gripper_indices]
        gripper_scalar = policy_gripper_scalar_from_joint_positions(
            gripper_positions,
            robot.joint_lower_limits[gripper_indices],
            robot.joint_upper_limits[gripper_indices],
        )
        state = torch.cat([arm_positions.to(torch.float32), gripper_scalar.to(torch.float32)], dim=0)

        return {
            "main_images": main_rgb,
            "wrist_images": wrist_rgb,
            "state": state,
            "task_description": spec.prompt,
            "_debug": {
                "wrist_sensor_name": wrist_sensor_name,
                "wrist_sensor_resolution": wrist_sensor_resolution,
                "available_robot_rgb_sensors": list(rgb_sensor_names),
                "main_rgb_shape": [int(v) for v in main_rgb.shape],
                "wrist_rgb_shape": [int(v) for v in wrist_rgb.shape],
            },
        }

    def _wrap_obs(self, obs_items):
        return {
            "main_images": torch.stack([item["main_images"] for item in obs_items], dim=0),
            "wrist_images": torch.stack([item["wrist_images"] for item in obs_items], dim=0),
            "states": torch.stack([item["state"] for item in obs_items], dim=0),
            "task_descriptions": [item["task_description"] for item in obs_items],
        }

    def _record_metrics(self, rewards, successes, infos):
        info_list = []
        for env_idx, (reward, success, info) in enumerate(zip(rewards, successes, infos)):
            self.returns[env_idx] += float(reward)
            self.success_once[env_idx] = self.success_once[env_idx] | bool(success)
            info_list.append(
                {
                    "success_once": self.success_once[env_idx].clone(),
                    "return": self.returns[env_idx].clone(),
                    "episode_len": self.elapsed_steps[env_idx].clone(),
                    "reward": self.returns[env_idx].clone()
                    / max(int(self.elapsed_steps[env_idx].item()), 1),
                }
            )
        return {"episode": to_tensor(list_of_dict_to_dict_of_list(info_list))}

    def _append_jsonl(self, path: Path, payload: dict):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _serialize_ltl_info(self, ltl_info: dict | None) -> dict:
        if not ltl_info:
            return {
                "violation": False,
                "state": None,
                "accepting": None,
                "ap": None,
            }
        label_dict = ltl_info.get("ap")
        if isinstance(label_dict, dict):
            label_dict = {str(k): bool(v) for k, v in label_dict.items()}
        return {
            "violation": bool(ltl_info.get("doomed", False)),
            "state": ltl_info.get("state"),
            "accepting": ltl_info.get("accepting"),
            "ap": label_dict,
        }

    def _goal_text_lookup(self, task, item) -> str | None:
        value = item.item() if hasattr(item, "item") else item
        if isinstance(value, str) and value.isdigit():
            value = int(value)

        natural_goals = getattr(task, "activity_natural_language_goal_conditions", None)
        if natural_goals is None:
            return None

        if isinstance(value, int):
            try:
                if isinstance(natural_goals, dict):
                    return natural_goals.get(value) or natural_goals.get(str(value))
                return natural_goals[value]
            except Exception:
                return None
        return None

    def _serialize_goal_status(self, goal_status: dict | None, task=None) -> dict | None:
        if not goal_status:
            return None

        satisfied = [str(item) for item in goal_status.get("satisfied", [])]
        unsatisfied = [str(item) for item in goal_status.get("unsatisfied", [])]
        satisfied_text = [self._goal_text_lookup(task, item) for item in goal_status.get("satisfied", [])]
        unsatisfied_text = [self._goal_text_lookup(task, item) for item in goal_status.get("unsatisfied", [])]
        return {
            "satisfied": satisfied,
            "unsatisfied": unsatisfied,
            "satisfied_text": [text for text in satisfied_text if text is not None],
            "unsatisfied_text": [text for text in unsatisfied_text if text is not None],
            "satisfied_count": len(satisfied),
            "unsatisfied_count": len(unsatisfied),
        }

    def _current_goal_status(self, env, info: dict | None = None) -> dict | None:
        task = getattr(env, "task", None)
        if info is not None and info.get("goal_status") is not None:
            return self._serialize_goal_status(info.get("goal_status"), task=task)

        if task is None:
            return None
        termination_conditions = getattr(task, "_termination_conditions", {}) or {}
        predicate_goal = termination_conditions.get("predicate")
        if predicate_goal is None:
            return None
        return self._serialize_goal_status(getattr(predicate_goal, "goal_status", None), task=task)

    def _termination_reason(self, info: dict, terminated: bool, truncated: bool) -> str:
        if bool(info.get("success", False)):
            return "success"
        if truncated:
            return "time_limit"
        if bool(info.get("ltl_violation_terminated", False)):
            return "ltl_violation"
        if terminated:
            return "env_terminated"
        return "running"

    def _collect_object_state_summary(self, env) -> list[dict]:
        from omnigibson.object_states import Dropped, Upright

        summary = []
        object_scope = getattr(env.task, "object_scope", {}) or {}
        for inst_id, entity in object_scope.items():
            if entity is None or not getattr(entity, "exists", False):
                continue
            wrapped_obj = getattr(entity, "wrapped_obj", None)
            if wrapped_obj is None:
                continue
            position = None
            try:
                position = [float(v) for v in wrapped_obj.get_position_orientation()[0][:3]]
            except Exception:
                position = None
            upright = None
            dropped = None
            try:
                if Upright in wrapped_obj.states:
                    upright = bool(wrapped_obj.states[Upright].get_value())
            except Exception:
                upright = None
            try:
                if Dropped in wrapped_obj.states:
                    dropped = bool(wrapped_obj.states[Dropped].get_value())
            except Exception:
                dropped = None
            summary.append(
                {
                    "inst_id": inst_id,
                    "scene_object_name": getattr(wrapped_obj, "name", None),
                    "category": getattr(wrapped_obj, "category", None),
                    "upright": upright,
                    "dropped": dropped,
                    "position": position,
                }
            )
        return summary

    def _build_ltl_monitor(self, env, spec: SentinelSceneSpec):
        from omnigibson.utils.safety_monitor import TaskLTLMonitor

        active_objects = {}
        object_scope = getattr(env.task, "object_scope", {}) or {}
        for inst_id, entity in object_scope.items():
            if entity is None or not getattr(entity, "exists", False):
                continue
            wrapped_obj = getattr(entity, "wrapped_obj", None)
            if wrapped_obj is not None:
                active_objects[inst_id] = wrapped_obj

        monitor = TaskLTLMonitor(
            env=env,
            activity_name=spec.activity_name,
            scene_model=spec.scene_name,
            active_objects_by_inst=active_objects,
        )
        monitor.reset()
        return monitor

    def _save_episode_artifacts(self, env_idx: int, terminated: bool, truncated: bool, info: dict | None = None):
        scene_name = self._scene_specs[env_idx].scene_name
        episode_idx = self._episode_indices[env_idx]
        if self.video_cfg.save_video and self._frames[env_idx]:
            policy_path = self._results_dir / f"{scene_name}_episode_{episode_idx}.mp4"
            imageio.mimsave(policy_path, self._frames[env_idx], fps=10)
        if self._policy_start_frames[env_idx] is not None:
            imageio.imwrite(
                self._results_dir / f"{scene_name}_episode_{episode_idx}_policy_start.png",
                self._policy_start_frames[env_idx],
            )
        if self._wrist_start_frames[env_idx] is not None:
            imageio.imwrite(
                self._results_dir / f"{scene_name}_episode_{episode_idx}_wrist_start.png",
                self._wrist_start_frames[env_idx],
            )
        spec = self._scene_specs[env_idx]
        info = info or {}
        ltl_info = info.get("ltl") or {}
        reset_ltl = self._serialize_ltl_info((self._reset_infos[env_idx] or {}).get("ltl"))
        episode_camera = self._camera_views[env_idx] or {}
        camera_path = self._results_dir / f"{scene_name}_episode_{episode_idx}_camera.json"
        camera_path.write_text(
            json.dumps(
                {
                    "scene_name": scene_name,
                    "episode_idx": episode_idx,
                    "policy_camera": episode_camera,
                    "sensor_debug": self._sensor_debug_infos[env_idx],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        reset_state_path = self._results_dir / f"{scene_name}_episode_{episode_idx}_reset_state.json"
        reset_state_path.write_text(
            json.dumps(
                {
                    "scene_name": scene_name,
                    "episode_idx": episode_idx,
                    "objects": self._reset_object_summaries[env_idx] or [],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._append_jsonl(
            self._results_path,
            {
                "scene_name": spec.scene_name,
                "activity_name": spec.activity_name,
                "target_synset": spec.target_synset,
                "success": bool(info.get("success", False)),
                "success_once": bool(self.success_once[env_idx]),
                "ltl_violation": bool(info.get("ltl_violation", False)),
                "ltl_violation_terminated": bool(info.get("ltl_violation_terminated", False)),
                "ltl_state": ltl_info.get("state"),
                "ltl_accepting": ltl_info.get("accepting"),
                "ltl_ap": self._serialize_ltl_info(ltl_info)["ap"],
                "reset_ltl_violation": reset_ltl["violation"],
                "reset_ltl_state": reset_ltl["state"],
                "reset_ltl_accepting": reset_ltl["accepting"],
                "reset_ltl_ap": reset_ltl["ap"],
                "goal_status": info.get("goal_status"),
                "termination_reason": self._termination_reason(info, terminated, truncated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "ltl_violation_first_step": self._ltl_violation_first_steps[env_idx],
                "episode_len": int(self.elapsed_steps[env_idx].item()),
                "return": float(self.returns[env_idx].item()),
                "policy_camera": episode_camera,
            },
        )
        self._append_jsonl(
            self._prompt_log_path,
            {
                "scene_name": spec.scene_name,
                "prompt": spec.prompt,
            },
        )
        self._frames[env_idx] = []
        self._camera_views[env_idx] = None
        self._sensor_debug_infos[env_idx] = None
        self._policy_start_frames[env_idx] = None
        self._wrist_start_frames[env_idx] = None
        self._reset_infos[env_idx] = None
        self._reset_object_summaries[env_idx] = None
        self._ltl_monitors[env_idx] = None
        self._ltl_violation_first_steps[env_idx] = None
        self._episode_indices[env_idx] += 1

    def _reset_metrics(self, env_idx: int | None = None):
        if env_idx is None:
            self._elapsed_steps.zero_()
            self.success_once.zero_()
            self.returns.zero_()
            self._ltl_violation_first_steps = [None for _ in range(self.num_envs)]
            return
        self._elapsed_steps[env_idx] = 0
        self.success_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._ltl_violation_first_steps[env_idx] = None

    def reset(self):
        import omnigibson as og

        obs_items = []
        for env_idx, (env, spec) in enumerate(zip(self.envs, self._scene_specs)):
            self._debug_reset_trace(f"env_idx={env_idx} scene={spec.scene_name} entering env.reset()")
            raw_obs, reset_info = env.reset()
            self._debug_reset_trace(f"env_idx={env_idx} env.reset() returned")
            reset_info = dict(reset_info or {})
            settle_steps = max(self.reset_settle_steps, 1)
            self._debug_reset_trace(f"env_idx={env_idx} settling scene pose for {settle_steps} steps")
            for _ in range(settle_steps):
                env.robots[0].keep_still()
                og.sim.step()
            self._debug_reset_trace(f"env_idx={env_idx} scene settle done")
            profile_reset_override = self._apply_profile_post_reset_ready_pose(env_idx, env, spec)
            if profile_reset_override is not None:
                self._debug_reset_trace(f"env_idx={env_idx} applied profile post-reset override")
                reset_info["profile_post_reset_override"] = profile_reset_override
            policy_reset_override = self._apply_policy_reset_state_override(env)
            if policy_reset_override is not None:
                self._debug_reset_trace(f"env_idx={env_idx} applied policy-state reset override")
                reset_info["policy_reset_override"] = policy_reset_override
            wrist_policy_view = self._apply_policy_wrist_camera(env, raw_obs, spec)
            self._debug_reset_trace(f"env_idx={env_idx} applied wrist camera")
            self._camera_views[env_idx] = self._apply_policy_camera(env, spec)
            self._debug_reset_trace(f"env_idx={env_idx} applied main camera")
            self._debug_reset_trace(f"env_idx={env_idx} flushing camera pose for {settle_steps} steps")
            for _ in range(settle_steps):
                env.robots[0].keep_still()
                og.sim.step()
            self._debug_reset_trace(f"env_idx={env_idx} camera pose flush done")
            for _ in range(2):
                og.sim.render()
            self._debug_reset_trace(f"env_idx={env_idx} render complete, fetching obs")
            raw_obs, _ = env.get_obs()
            self._debug_reset_trace(f"env_idx={env_idx} get_obs complete")
            self._reset_infos[env_idx] = dict((reset_info or {}).get("obs_info", {}))
            self._reset_object_summaries[env_idx] = self._collect_object_state_summary(env)
            self._ltl_monitors[env_idx] = self._build_ltl_monitor(env, spec)
            self._reset_infos[env_idx]["ltl"] = self._ltl_monitors[env_idx].step(step_idx=0)
            self._frames[env_idx] = []
            obs_item = self._extract_obs(env, raw_obs, spec)
            self._sensor_debug_infos[env_idx] = dict(obs_item.get("_debug") or {})
            self._sensor_debug_infos[env_idx]["policy_wrist_camera"] = wrist_policy_view
            self._policy_start_frames[env_idx] = obs_item["main_images"].cpu().numpy().astype(np.uint8)
            self._wrist_start_frames[env_idx] = obs_item["wrist_images"].cpu().numpy().astype(np.uint8)
            if self.video_cfg.save_video:
                self._frames[env_idx].append(self._policy_start_frames[env_idx])
            obs_items.append(obs_item)
        self._reset_metrics()
        return self._wrap_obs(obs_items), {}

    def step(self, actions=None):
        obs_items = []
        rewards = []
        terminations = []
        truncations = []
        info_payloads = []
        for env_idx, (env, spec, action) in enumerate(zip(self.envs, self._scene_specs, actions)):
            raw_obs, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            info["goal_status"] = self._current_goal_status(env, info)
            ltl_monitor = self._ltl_monitors[env_idx]
            ltl_info = None
            current_step = int(self._elapsed_steps[env_idx].item()) + 1
            if ltl_monitor is not None:
                ltl_info = ltl_monitor.step(step_idx=current_step)
                info["ltl"] = ltl_info
            if ltl_info is not None:
                if bool(ltl_info.get("doomed", False)) and self._ltl_violation_first_steps[env_idx] is None:
                    self._ltl_violation_first_steps[env_idx] = current_step
                latched_violation = self._ltl_violation_first_steps[env_idx] is not None
                info["ltl_violation"] = latched_violation
                info["ltl_violation_first_step"] = self._ltl_violation_first_steps[env_idx]
                info["ltl_violation_terminated"] = False
                if self.terminate_on_ltl_violation and latched_violation and current_step >= (
                    self._ltl_violation_first_steps[env_idx] + self.ltl_violation_grace_steps
                ):
                    terminated = True
                    info["ltl_violation_terminated"] = True
            else:
                info["ltl_violation_terminated"] = False
            info["success"] = bool(info.get("success", False))
            if info["success"]:
                terminated = True
            if self.video_cfg.save_video:
                frame = raw_obs["external"]["external_sensor0"]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                self._frames[env_idx].append(frame)
            obs_items.append(self._extract_obs(env, raw_obs, spec))
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            reached_time_limit = bool(self._elapsed_steps[env_idx] + 1 >= self.cfg.max_episode_steps)
            truncations.append(bool((not terminated) and (truncated or reached_time_limit)))
            info_payloads.append(info)

        self._elapsed_steps += 1
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        terminations_t = torch.tensor(terminations, dtype=torch.bool)
        truncations_t = torch.tensor(truncations, dtype=torch.bool)
        success_t = torch.tensor(
            [bool(info.get("success", False)) for info in info_payloads], dtype=torch.bool
        )
        infos = self._record_metrics(rewards_t, success_t, info_payloads)
        infos["ltl_violation"] = torch.tensor(
            [bool(info.get("ltl_violation", False)) for info in info_payloads], dtype=torch.bool
        )
        infos["ltl_violation_terminated"] = torch.tensor(
            [bool(info.get("ltl_violation_terminated", False)) for info in info_payloads], dtype=torch.bool
        )
        infos["ltl_violation_first_step"] = [
            info.get("ltl_violation_first_step") for info in info_payloads
        ]
        infos["goal_status"] = [info.get("goal_status") for info in info_payloads]

        for env_idx, (terminated, truncated) in enumerate(zip(terminations_t, truncations_t)):
            if bool(terminated or truncated):
                self._save_episode_artifacts(
                    env_idx,
                    bool(terminated),
                    bool(truncated),
                    info_payloads[env_idx],
                )

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations_t.clone()
            terminations_t[:] = False

        return self._wrap_obs(obs_items), rewards_t, terminations_t, truncations_t, infos

    def chunk_step(self, chunk_actions):
        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []
        infos = {}
        extracted_obs = None
        for step_idx in range(chunk_actions.shape[1]):
            extracted_obs, rewards, terminations, truncations, infos = self.step(
                chunk_actions[:, step_idx]
            )
            chunk_rewards.append(rewards)
            chunk_terminations.append(terminations)
            chunk_truncations.append(truncations)
        return (
            extracted_obs,
            torch.stack(chunk_rewards, dim=1),
            torch.stack(chunk_terminations, dim=1),
            torch.stack(chunk_truncations, dim=1),
            infos,
        )

    def flush_video(self):
        return None

    def update_reset_state_ids(self):
        return None
