"""Load frozen task snapshots into simulation, step them, and record taskgen-style review videos."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_VIEWER_CAMERA_PATH = "/OmniverseKit_Persp"
TASKGEN_REVIEW_FRAME_HW = (512, 512)
DEFAULT_REVIEW_CAMERA_NAMES = ("cam_opposite", "cam_left", "cam_right", "cam_left_shoulder")
REVIEW_CAMERA_LABELS = {
    "cam_opposite": "opposite_side_front",
    "cam_left": "left_overview",
    "cam_right": "right_overview",
    "cam_left_shoulder": "left_shoulder",
}


@lru_cache(maxsize=1)
def resolve_runtime_python() -> Path:
    candidates: list[Path] = []
    override = os.environ.get("SENTINEL_RUNTIME_PYTHON")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(Path(sys.executable).expanduser())
    candidates.append(Path.home() / "miniconda3" / "envs" / "behavior" / "bin" / "python")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        probe = subprocess.run(
            [str(candidate), "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(
        "Could not find a runtime python with torch. Set SENTINEL_RUNTIME_PYTHON or install/use the behavior env."
    )


def _scene_registry(scene_info: dict[str, Any]) -> dict[str, Any]:
    state = scene_info.get("state", {})
    if "registry" in state:
        return state["registry"].setdefault("object_registry", {})
    return state.setdefault("object_registry", {})


def _is_scene_robot(obj_info: dict[str, Any]) -> bool:
    class_module = str(obj_info.get("class_module", ""))
    class_name = str(obj_info.get("class_name", ""))
    return class_module.startswith("omnigibson.robots.") or class_name.endswith(("Robot", "Mounted", "Panda"))


def strip_scene_robots_from_scene_info(scene_info: dict[str, Any]) -> dict[str, Any]:
    runtime_scene_info = json.loads(json.dumps(scene_info))
    init_info = runtime_scene_info.get("objects_info", {}).get("init_info", {})
    state_registry = _scene_registry(runtime_scene_info)
    robot_names = [name for name, obj_info in init_info.items() if _is_scene_robot(obj_info)]
    for robot_name in robot_names:
        init_info.pop(robot_name, None)
        state_registry.pop(robot_name, None)
    return runtime_scene_info


def extract_scene_robot_setup(scene_info: dict[str, Any], robot_name: str = "agent_0") -> dict[str, Any] | None:
    init_info = scene_info.get("objects_info", {}).get("init_info", {})
    state_registry = _scene_registry(scene_info)
    for scene_object_name, obj_info in init_info.items():
        if not _is_scene_robot(obj_info):
            continue
        state_info = state_registry.get(scene_object_name, {})
        root_link = state_info.get("root_link", {})
        robot_args = json.loads(json.dumps(obj_info.get("args", {}) or {}))
        robot_args.pop("expected_file_hash", None)
        return {
            "scene_object_name": scene_object_name,
            "name": robot_name,
            "robot_type": str(obj_info.get("class_name") or robot_args.get("type") or "FrankaPanda"),
            "robot_args": robot_args,
            "position": root_link.get("pos"),
            "orientation": root_link.get("ori"),
            "reset_joint_pos": state_info.get("joint_pos"),
        }
    return None


def _build_runtime_robot_cfg(scene_robot_setup: dict[str, Any] | None) -> dict[str, Any]:
    robot_cfg = {
        "type": "FrankaPanda",
        "name": "agent_0",
        "obs_modalities": [],
        "include_sensor_names": None,
        "exclude_sensor_names": None,
        "scale": 1.0,
        "self_collisions": True,
        "action_normalize": True,
        "action_type": "continuous",
        "grasping_mode": "physical",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "fixed_base": True,
        "controller_config": {
            "arm_0": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": "default",
                "command_output_limits": "default",
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                "command_input_limits": "default",
                "command_output_limits": "default",
            },
        },
    }
    if scene_robot_setup is None:
        return robot_cfg
    robot_args = scene_robot_setup.get("robot_args")
    if isinstance(robot_args, dict):
        robot_cfg.update(json.loads(json.dumps(robot_args)))
    if scene_robot_setup.get("robot_type"):
        robot_cfg["type"] = scene_robot_setup["robot_type"]
    if scene_robot_setup.get("name"):
        robot_cfg["name"] = scene_robot_setup["name"]
    if scene_robot_setup.get("position") is not None:
        robot_cfg["position"] = list(scene_robot_setup["position"])
    if scene_robot_setup.get("orientation") is not None:
        robot_cfg["orientation"] = list(scene_robot_setup["orientation"])
    if scene_robot_setup.get("reset_joint_pos") is not None:
        robot_cfg["reset_joint_pos"] = list(scene_robot_setup["reset_joint_pos"])
    return robot_cfg


def _init_omnigibson(headless: bool = True):
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root, repo_root / "OmniGibson"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    data_path = os.environ.get("OMNIGIBSON_DATA_PATH", "")
    candidate_roots = [
        repo_root / "datasets",
        repo_root.parent / "SENTINEL-Lite-data" / "datasets",
    ]
    if not data_path or not Path(data_path).exists():
        for candidate in candidate_roots:
            if candidate.exists():
                os.environ["OMNIGIBSON_DATA_PATH"] = str(candidate.resolve())
                break

    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    if headless:
        gm.HEADLESS = True

    import omnigibson as og

    return og


@dataclass
class FrozenTaskRuntimeSession:
    headless: bool = True
    og: Any | None = None

    def __enter__(self):
        self.og = _init_omnigibson(headless=self.headless)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.og is not None and self.og.sim is not None:
            try:
                viewer_camera = getattr(self.og.sim, "viewer_camera", None)
                if viewer_camera is not None:
                    viewer_camera.active_camera_path = DEFAULT_VIEWER_CAMERA_PATH
            except Exception:
                pass
            try:
                self.og.sim.stop()
            except Exception:
                pass
        return False


def build_env_config(scene_info: dict[str, Any], diagnostics: dict[str, Any], *, camera_names: Sequence[str] | None = None) -> dict[str, Any]:
    runtime_scene_info = strip_scene_robots_from_scene_info(scene_info)
    scene_class = runtime_scene_info.get("init_info", {}).get("class_name", "")
    scene_model = diagnostics.get("scene_model")
    robot_setup = extract_scene_robot_setup(scene_info)

    if scene_class == "InteractiveTraversableScene":
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "scene_file": runtime_scene_info,
            "scene_instance": None,
            "include_robots": False,
        }
    else:
        scene_cfg = {
            "type": "Scene",
            "scene_file": runtime_scene_info,
        }

    cfg = {
        "scene": scene_cfg,
        "robots": [_build_runtime_robot_cfg(robot_setup)] if robot_setup is not None else [],
        "objects": [],
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": 20,
            "rendering_frequency": 20,
            "physics_frequency": 120,
        },
    }
    if camera_names:
        from omnigibson.utils.camera_setup import build_external_camera_configs

        cfg["env"]["external_sensors"] = build_external_camera_configs(names=list(camera_names))
    return cfg


def position_diagnostics_cameras(
    env,
    og,
    diagnostics: dict[str, Any],
    *,
    preferred_camera: str | None = None,
    set_viewer: bool = True,
) -> int:
    from omnigibson.task_generation.utils.video import eye_lookat_to_quat

    cameras = list(diagnostics.get("cameras", []) or [])
    viewer_entry = None
    if preferred_camera is not None:
        viewer_entry = next((c for c in cameras if c.get("sensor_name") == preferred_camera), None)
    if viewer_entry is None and cameras:
        viewer_entry = next((c for c in cameras if c.get("canonical")), cameras[0])

    placed = 0
    for cam_info in cameras:
        sensor_name = cam_info.get("sensor_name")
        if not sensor_name:
            continue
        sensor = (env.external_sensors or {}).get(sensor_name)
        if sensor is None:
            continue
        eye = cam_info.get("eye")
        lookat = cam_info.get("lookat")
        if eye is None or lookat is None:
            continue
        orientation = cam_info.get("orientation") or eye_lookat_to_quat(eye, lookat).tolist()
        sensor.set_position_orientation(position=eye, orientation=orientation, frame="world")
        placed += 1

    if set_viewer and viewer_entry is not None:
        eye = viewer_entry.get("eye")
        lookat = viewer_entry.get("lookat")
        if eye is not None and lookat is not None:
            orientation = viewer_entry.get("orientation") or eye_lookat_to_quat(eye, lookat).tolist()
            og.sim.viewer_camera.set_position_orientation(position=eye, orientation=orientation)

    return placed


class ReviewVideoRecorder:
    def __init__(self, *, path: Path, fps: int, camera_names: Sequence[str] | None = None):
        self.path = path
        self.fps = int(fps)
        self.camera_names = [str(name) for name in (camera_names or []) if str(name)]
        self.writer = None
        self.writers: dict[str, Any] = {}

    def __enter__(self):
        import imageio.v2 as imageio

        self.path.mkdir(parents=True, exist_ok=True)
        for camera_name in self.camera_names:
            label = REVIEW_CAMERA_LABELS.get(camera_name, camera_name.replace("cam_", ""))
            video_path = self.path / f"rollout_{label}_ep1.mp4"
            self.writers[camera_name] = imageio.get_writer(str(video_path), fps=self.fps)
        return self

    @staticmethod
    def _to_uint8_frame(rgb: Any):
        import numpy as np

        frame = rgb[..., :3]
        if hasattr(frame, "detach"):
            frame = frame.detach()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        else:
            frame = np.asarray(frame)
        return frame.astype("uint8")

    def record(self, env, og) -> None:
        og.sim.render()
        for camera_name, writer in self.writers.items():
            sensor = (env.external_sensors or {}).get(camera_name)
            if sensor is None:
                continue
            try:
                rgb = sensor.get_obs()[0].get("rgb")
            except Exception:
                rgb = None
            if rgb is None:
                continue
            writer.append_data(self._to_uint8_frame(rgb))

    def __exit__(self, exc_type, exc, tb):
        for writer in self.writers.values():
            writer.close()
        return False


def configure_review_sensors(env) -> None:
    from omnigibson.task_generation.utils.video import configure_taskgen_review_sensors

    configure_taskgen_review_sensors(env, TASKGEN_REVIEW_FRAME_HW)


def compute_floor_z(env) -> float:
    floor_z = 0.0
    for obj in env.scene.objects:
        category = str(getattr(obj, "category", ""))
        name = str(getattr(obj, "name", ""))
        if category != "floors" and not name.startswith("floors_"):
            continue
        try:
            _, aabb_max = obj.aabb
        except Exception:
            continue
        floor_z = max(floor_z, float(aabb_max[2]))
    return floor_z


def zero_action(robot) -> Any:
    import numpy as np

    return np.zeros_like(np.asarray(robot.action_space.sample(), dtype=np.float32))


def step_idle(env, og, *, steps: int, video_recorder: ReviewVideoRecorder | None = None) -> None:
    robot = env.robots[0] if env.robots else None
    action = zero_action(robot) if robot is not None else None
    for _ in range(max(0, int(steps))):
        if action is not None:
            env.step(action)
        else:
            og.sim.step()
        if video_recorder is not None:
            video_recorder.record(env, og)


def save_scene_snapshot(env, scene_path: Path) -> None:
    env.scene.save(json_path=str(scene_path))
