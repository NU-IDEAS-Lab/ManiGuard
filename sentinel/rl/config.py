"""Build an OmniGibson env config from a sentinel benchmark scene directory.

Reuses ``sentinel.eval.scene_discovery.discover_scenes`` so RL training and
VLA eval agree on target-object / scene-file resolution.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from sentinel.envs.registry import extract_scene_robot_setup
from sentinel.eval.scene_discovery import discover_scenes


def _scene_info_for_dir(scene_dir: Path) -> Dict[str, Any]:
    scene_dir = scene_dir.resolve()
    scenes = discover_scenes(str(scene_dir.parent), scene_names=[scene_dir.name])
    if not scenes:
        raise FileNotFoundError(
            f"No benchmark scene discovered at {scene_dir} "
            f"(need scene_ep1.json + diagnostics.jsonl)"
        )
    return scenes[0]


def _write_precached_reset_pose(path: str, scene_file: str) -> None:
    """Write a 1-entry precached reset pose file from the scene's robot state.

    GraspTask._reset_agent expects ``joint_pos`` with arm DoF only (7 for
    Franka); gripper joints come from the scene init_state directly.
    """
    scene_data = json.loads(Path(scene_file).read_text(encoding="utf-8"))
    robot_setup = extract_scene_robot_setup(scene_data)
    if robot_setup is None:
        raise ValueError(f"No robot found in {scene_file}")

    joint_pos = robot_setup.get("reset_joint_pos")
    if isinstance(joint_pos, list):
        joint_pos = list(joint_pos)[:7]

    payload = [{
        "joint_pos": joint_pos,
        "base_pos": robot_setup.get("position") or [0.0, 0.0, 0.0],
        "base_ori": robot_setup.get("orientation") or [0.0, 0.0, 0.0, 1.0],
    }]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def build_config(scene_dir: Path | str) -> Dict[str, Any]:
    """Build an OmniGibson Environment config for GraspTask on a clutter scene.

    Scene, robot, and objects all come from the scene_file; the task targets
    whichever object ``scene_discovery`` resolves as the pickup target.
    """
    scene_dir = Path(scene_dir).resolve()
    scene_info = _scene_info_for_dir(scene_dir)
    scene_file = scene_info["scene_file"]
    target_name = scene_info["target_name"]

    reset_pose_path = os.path.join(tempfile.gettempdir(), f"sentinel_rl_reset_pose_{scene_dir.name}.json")
    _write_precached_reset_pose(reset_pose_path, scene_file)

    task_cfg = {
        "type": "SentinelGraspTask",
        "obj_name": target_name,
        "objects_config": [],
        "precached_reset_pose_path": reset_pose_path,
        "termination_config": {"max_steps": 500, "grasp_hold_steps": 10},
        "reward_config": {
            "dist_coeff": 1.0,
            "grasp_reward": 10.0,
            "collision_penalty": 1.0,
            "eef_position_penalty_coef": 0.0,
            "eef_orientation_penalty_coef": 0.0,
            "regularization_coef": 0.0,
        },
    }

    return {
        "scene": {"type": "Scene", "scene_file": scene_file},
        "robots": [],
        "objects": [],
        "task": task_cfg,
        "env": {"flatten_action_space": True, "flatten_obs_space": True},
    }
