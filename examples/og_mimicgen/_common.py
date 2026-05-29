"""Shared helpers for the OmniGibson MimicGen-style example scripts."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T


def load_json_or_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping config in {path}")
    return data


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    return value


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(data), f, indent=2)


_MISSING = object()


def read_json_attr(attrs: h5py.AttributeManager, key: str, default: Any = _MISSING) -> Any:
    if key not in attrs:
        if default is not _MISSING:
            return default
        raise KeyError(f"HDF5 attribute {key!r} is missing")
    raw = attrs[key]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def load_env_config_from_hdf5(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        config = read_json_attr(f["data"].attrs, "config")
        if "scene_file" in f["data"].attrs:
            config = copy.deepcopy(config)
            config.setdefault("scene", {})["scene_file"] = read_json_attr(f["data"].attrs, "scene_file")
    return config


def configure_playback_config(config: dict[str, Any], frequency: float = 1000.0) -> dict[str, Any]:
    config = copy.deepcopy(config)
    env_cfg = config.setdefault("env", {})
    env_cfg["action_frequency"] = frequency
    env_cfg["rendering_frequency"] = frequency
    env_cfg["physics_frequency"] = frequency
    env_cfg["flatten_obs_space"] = True
    return config


def create_env(config: dict[str, Any]):
    return og.Environment(configs=copy.deepcopy(config))


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def robot_action_offset(env, robot_id: int) -> int:
    return int(sum(robot.action_dim for robot in env.robots[:robot_id]))


def make_full_action(env, robot_id: int, robot_action: th.Tensor | np.ndarray) -> th.Tensor:
    actions = []
    for robot in env.robots:
        action = th.zeros(robot.action_dim, dtype=th.float32)
        gripper_idx = getattr(robot, "gripper_action_idx", {}).get(robot.default_arm)
        if gripper_idx is not None:
            action[gripper_idx] = 1.0
        actions.append(action)
    full = th.cat(actions, dim=0)
    start = robot_action_offset(env, robot_id)
    robot_action = as_tensor(robot_action)
    full[start : start + robot_action.shape[0]] = robot_action
    return full


def as_tensor(value: Any, *, dtype=th.float32) -> th.Tensor:
    if isinstance(value, th.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return th.tensor(value, dtype=dtype)


def maybe_reload_ik_controllers(env, robot_id: int, *, gripper_mode: str = "smooth") -> None:
    """Put the active robot into an IK delta-pose controller if OG supports it."""
    robot = env.robots[robot_id]
    arm_name = f"arm_{robot.default_arm}"
    gripper_name = f"gripper_{robot.default_arm}"
    controller_config = {
        arm_name: {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
            "mode": "pose_delta_ori",
            "smoothing_filter_size": 5,
        },
        gripper_name: {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": gripper_mode,
        },
    }
    if hasattr(robot, "reload_controllers"):
        robot.reload_controllers(controller_config=controller_config)


def build_ik_delta_action(
    robot,
    target_pos_robot: th.Tensor,
    target_quat_robot: th.Tensor,
    gripper_cmd: float,
) -> th.Tensor:
    curr_pos, curr_quat = robot.get_relative_eef_pose(arm="default")
    curr_pos = as_tensor(curr_pos)
    curr_quat = as_tensor(curr_quat)
    target_pos_robot = as_tensor(target_pos_robot)
    target_quat_robot = as_tensor(target_quat_robot)

    delta_pos = target_pos_robot - curr_pos
    delta_ori = T.orientation_error(T.quat2mat(target_quat_robot), T.quat2mat(curr_quat))
    arm_cmd = th.cat([delta_pos, delta_ori], dim=0).to(dtype=th.float32)

    action = th.zeros(robot.action_dim, dtype=th.float32)
    arm_key = f"arm_{robot.default_arm}"
    gripper_key = f"gripper_{robot.default_arm}"
    if hasattr(robot, "controller_action_idx") and arm_key in robot.controller_action_idx:
        action[robot.controller_action_idx[arm_key]] = arm_cmd
        if gripper_key in robot.controller_action_idx:
            idx = robot.controller_action_idx[gripper_key]
            action[idx] = float(gripper_cmd)
        else:
            action[-1] = float(gripper_cmd)
    else:
        if robot.action_dim < 7:
            raise RuntimeError(f"Robot action_dim={robot.action_dim} is too small for IK delta-pose actions")
        action[:6] = arm_cmd
        action[6] = float(gripper_cmd)
    return action


def transform_waypoints_to_robot_frame(
    waypoints: list[dict[str, Any]],
    ref_obj,
    robot,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    obj_pos, obj_quat = ref_obj.get_position_orientation()
    robot_pos, robot_quat = robot.get_position_orientation()

    obj_t_world = T.pose2mat((as_tensor(obj_pos), as_tensor(obj_quat)))
    robot_t_world = T.pose2mat((as_tensor(robot_pos), as_tensor(robot_quat)))
    world_t_robot = T.pose_inv(robot_t_world)

    pos_robot = []
    quat_robot = []
    gripper = []
    for waypoint in waypoints:
        eef_t_obj = T.pose2mat((as_tensor(waypoint["eef_pos_obj"]), as_tensor(waypoint["eef_quat_obj"])))
        eef_t_world = obj_t_world @ eef_t_obj
        eef_t_robot = world_t_robot @ eef_t_world
        eef_pos_robot, eef_quat_robot = T.mat2pose(eef_t_robot)
        pos_robot.append(eef_pos_robot)
        quat_robot.append(eef_quat_robot)
        gripper.append(float(waypoint["gripper_action"]))

    return th.stack(pos_robot), th.stack(quat_robot), th.tensor(gripper, dtype=th.float32)


def apply_eef_z_offset(eef_pos: th.Tensor, eef_quat: th.Tensor, z_offset: float) -> tuple[th.Tensor, th.Tensor]:
    if abs(float(z_offset)) < 1e-9:
        return eef_pos, eef_quat
    offset_local = th.tensor([0.0, 0.0, float(z_offset)], dtype=th.float32)
    rot = T.quat2mat(eef_quat)
    if eef_pos.ndim == 1:
        return eef_pos + rot @ offset_local, eef_quat
    return eef_pos + th.einsum("nij,j->ni", rot, offset_local), eef_quat


def linear_freespace_trajectory(
    robot,
    target_pos_robot: th.Tensor,
    target_quat_robot: th.Tensor,
    *,
    velocity: float,
    action_frequency: float,
) -> tuple[th.Tensor, th.Tensor]:
    curr_pos, curr_quat = robot.get_relative_eef_pose(arm="default")
    curr_pos = as_tensor(curr_pos)
    curr_quat = as_tensor(curr_quat)
    target_pos_robot = as_tensor(target_pos_robot)
    target_quat_robot = as_tensor(target_quat_robot)
    distance = th.linalg.norm(target_pos_robot - curr_pos).item()
    travel_time = distance / max(float(velocity), 1e-6)
    n_steps = max(2, int(math.ceil(travel_time * float(action_frequency))))
    s = th.linspace(0.0, 1.0, n_steps)
    pos = (1.0 - s[:, None]) * curr_pos[None, :] + s[:, None] * target_pos_robot[None, :]
    quat = T.quat_slerp(
        curr_quat[None, :].repeat(n_steps, 1),
        target_quat_robot[None, :].repeat(n_steps, 1),
        s[:, None],
    )
    return pos, quat


@dataclass(frozen=True)
class PoseRandomization:
    object_name: str
    max_xyz_offset: tuple[float, float, float]
    max_yaw: float


def parse_pose_randomization(value: str) -> PoseRandomization:
    try:
        object_name, raw_values = value.split(":", 1)
        dx, dy, dz, dyaw = [float(v) for v in raw_values.split(",")]
    except ValueError as exc:
        raise ValueError(
            "Expected randomization as OBJECT_NAME:DX,DY,DZ,DYAW_RAD, "
            f"got {value!r}"
        ) from exc
    if not object_name:
        raise ValueError(f"Object name is empty in {value!r}")
    return PoseRandomization(object_name, (dx, dy, dz), dyaw)


def randomize_pose(
    pos: Any,
    quat: Any,
    max_xyz_offset: Iterable[float],
    max_yaw: float,
) -> tuple[th.Tensor, th.Tensor]:
    pos = as_tensor(pos)
    quat = as_tensor(quat)
    max_xyz = as_tensor(list(max_xyz_offset))
    pos_offset = th.rand(3) * (2.0 * max_xyz) - max_xyz
    yaw_offset = th.rand(1).item() * 2.0 * float(max_yaw) - float(max_yaw)
    rot_offset = T.euler2mat(th.tensor([0.0, 0.0, yaw_offset], dtype=th.float32))
    return pos + pos_offset, T.mat2quat(rot_offset @ T.quat2mat(quat))


def apply_pose_randomizations(env, specs: Iterable[PoseRandomization]) -> None:
    for spec in specs:
        obj = env.scene.object_registry("name", spec.object_name, None)
        if obj is None:
            raise RuntimeError(f"Cannot randomize missing object {spec.object_name!r}")
        pos, quat = obj.get_position_orientation()
        new_pos, new_quat = randomize_pose(pos, quat, spec.max_xyz_offset, spec.max_yaw)
        obj.set_position_orientation(
            new_pos,
            new_quat,
        )
        if hasattr(obj, "keep_still"):
            obj.keep_still()


def wake_scene_objects(env) -> None:
    for obj in env.scene.objects:
        if hasattr(obj, "wake"):
            obj.wake()


def check_task_success(env, last_action: Any | None = None) -> bool:
    base_env = unwrap_env(env)
    task = getattr(base_env, "task", None)
    if task is not None and last_action is not None and hasattr(task, "step"):
        try:
            task.step(base_env, last_action)
        except Exception:
            pass
    if hasattr(base_env, "is_success"):
        try:
            value = base_env.is_success()
            if isinstance(value, dict) and "task" in value:
                return bool(value["task"])
            if isinstance(value, bool):
                return value
        except Exception:
            pass
    if task is not None:
        for attr in ("success", "_success"):
            if hasattr(task, attr):
                try:
                    return bool(getattr(task, attr))
                except Exception:
                    pass
    return False


def prepare_output_path(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    return path


def shutdown_og() -> None:
    try:
        og.clear()
    finally:
        og.shutdown()


def env_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    th.manual_seed(seed)
    try:
        og.set_seed(seed)
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
