"""Unified eval configuration.

One YAML file controls the entire eval run: benchmark source, policy
connection, model-specific knobs (state/action), sim frequencies, and
output settings.  CLI args override any field for quick ad-hoc tweaks.

Usage:
    python -m maniguard.eval.benchmark --config configs/eval/sim_table_25k.yaml
    python -m maniguard.eval.benchmark --config configs/eval/sim_table_25k.yaml --max-steps 500
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class EvalConfig:
    name: str = "unnamed"

    # -- Benchmark source --
    benchmark_root: str = ""
    benchmark_revision: str = "main"
    scene_filter: str = ""
    scenes: Optional[List[str]] = None
    max_scenes: Optional[int] = None

    # -- Policy connection --
    host: str = "127.0.0.1"
    port: int = 8000
    use_openpi_client: bool = True
    random_policy: bool = False

    # -- Model / observation --
    state_mode: str = "eef_8d_axisangle"
    policy_cameras: List[str] = field(default_factory=lambda: ["cam_opposite"])
    action_dim: int = 7
    execute_horizon: int = 5
    gripper_binarize: bool = True
    override_controller_config: Optional[Dict[str, Any]] = None

    # -- Simulation --
    action_frequency: int = 20
    rendering_frequency: int = 20
    physics_frequency: int = 120
    headless: bool = False
    longfinger: bool = True

    # -- Eval --
    max_steps: int = 1000
    camera_resolution: int = 256
    save_video: bool = True

    # -- Output --
    output_dir: str = ""

    # -- Informational (not used by benchmark.py directly) --
    checkpoint: str = ""
    serve_config_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        d = self.to_dict()
        for k in list(d):
            if d[k] is None:
                del d[k]
        return yaml.dump(d, default_flow_style=False, sort_keys=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def load_eval_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvalConfig(**{k: v for k, v in data.items() if k in EvalConfig.__dataclass_fields__})


def config_from_cli() -> EvalConfig:
    """Parse CLI args into an EvalConfig.

    --config loads a YAML base; all other flags override it.
    """
    import argparse

    p = argparse.ArgumentParser(description="Evaluate VLA on ManiGuard benchmark.")
    p.add_argument("--config", type=str, required=True, help="Path to eval config YAML.")
    # Every EvalConfig field can be overridden from CLI.
    p.add_argument("--benchmark-root", type=str, default=None)
    p.add_argument("--benchmark-revision", type=str, default=None)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--use-openpi-client", action="store_true", default=None)
    p.add_argument("--random-policy", action="store_true", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--execute-horizon", type=int, default=None)
    p.add_argument("--action-frequency", type=int, default=None)
    p.add_argument("--rendering-frequency", type=int, default=None)
    p.add_argument("--physics-frequency", type=int, default=None)
    p.add_argument("--headless", action="store_true", default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-video", action="store_true", default=None)
    p.add_argument("--policy-cameras", nargs="+", default=None)
    p.add_argument("--camera-resolution", type=int, default=None)

    args = p.parse_args()
    cfg = load_eval_config(args.config)

    cli_map = {
        "benchmark_root": "benchmark_root",
        "benchmark_revision": "benchmark_revision",
        "scenes": "scenes",
        "max_scenes": "max_scenes",
        "host": "host",
        "port": "port",
        "use_openpi_client": "use_openpi_client",
        "random_policy": "random_policy",
        "max_steps": "max_steps",
        "execute_horizon": "execute_horizon",
        "action_frequency": "action_frequency",
        "rendering_frequency": "rendering_frequency",
        "physics_frequency": "physics_frequency",
        "headless": "headless",
        "output_dir": "output_dir",
        "save_video": "save_video",
        "policy_cameras": "policy_cameras",
        "camera_resolution": "camera_resolution",
    }
    for cli_name, cfg_name in cli_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)

    return cfg
