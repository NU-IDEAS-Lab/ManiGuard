"""Eval profiles — per-model configuration for benchmark evaluation.

Each profile captures the model-specific knobs that differ across VLA
checkpoints (state shape, action dim, gripper handling, etc.) while
the rest of the eval pipeline stays model-agnostic.

Add a new model by dropping a YAML file in sentinel/eval/profiles/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class EvalProfile:
    name: str

    # -- Observation --
    # "eef_8d": eef_pos(3) + axisangle(3) + gripper_qpos(2)  (IsaacLab stack-cube)
    # "eef_7d": eef_pos(3) + axisangle(3) + gripper_scalar(1) (LIBERO)
    state_mode: str = "eef_8d"
    policy_cameras: List[str] = field(default_factory=lambda: ["cam_opposite"])

    # -- Action --
    action_dim: int = 7
    execute_horizon: int = 5
    gripper_binarize: bool = True   # sign() on last action dim

    # -- Server (informational — used by launcher scripts, not benchmark.py) --
    serve_config_name: str = ""
    serve_script: str = "sentinel/serve/pi05_franka.py"

    # -- Env controller override --
    # If set, benchmark.py will call robot.reload_controllers(...) after scene
    # load to swap the scene-baked controller (typically OSC from the pipeline)
    # for the one the policy expects. The robot's joint state is preserved; only
    # the controller logic + action_space are rebuilt.
    override_controller_config: Optional[Dict[str, Any]] = None


_PROFILES_DIR = Path(__file__).parent / "profiles"


def load_profile(name: str) -> EvalProfile:
    """Load an EvalProfile from sentinel/eval/profiles/<name>.yaml."""
    yaml_path = _PROFILES_DIR / f"{name}.yaml"
    if not yaml_path.is_file():
        available = [p.stem for p in _PROFILES_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Eval profile '{name}' not found at {yaml_path}. "
            f"Available: {sorted(available)}"
        )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data.setdefault("name", name)
    return EvalProfile(**data)


# Built-in defaults for quick use without a yaml file.
BUILTIN_PROFILES = {
    "pi05_stack_cube": EvalProfile(
        name="pi05_stack_cube",
        state_mode="eef_8d",
        action_dim=7,
        execute_horizon=5,
        serve_config_name="pi05_isaaclab_stack_cube",
    ),
    "pi0_libero": EvalProfile(
        name="pi0_libero",
        state_mode="eef_7d",
        action_dim=7,
        execute_horizon=4,
        serve_config_name="pi0_libero",
    ),
}


def get_profile(name: str) -> EvalProfile:
    """Get profile by name: try yaml first, then built-in, then error."""
    yaml_path = _PROFILES_DIR / f"{name}.yaml"
    if yaml_path.is_file():
        return load_profile(name)
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    available = sorted(
        set([p.stem for p in _PROFILES_DIR.glob("*.yaml")] + list(BUILTIN_PROFILES))
    )
    raise FileNotFoundError(
        f"Eval profile '{name}' not found. Available: {available}"
    )
