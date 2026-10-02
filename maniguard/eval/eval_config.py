"""Configuration for benchmark evaluation.

Load settings from YAML and override the fields exposed by config_from_cli.
The configuration specifies scene selection, policy connection, observation
and action conventions, simulator rates, metrics, and output paths.

Example:
    python -m maniguard.eval.benchmark --config configs/eval/clutter_pickup_joint.yaml"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    name: str = "unnamed"
    recording_factory: str | None = None
    recording_output_dir: str | None = None
    recording_provenance: str | None = None

    # -- Benchmark source --
    benchmark_root: str = ""
    benchmark_revision: str = "main"
    scene_filter: str = ""
    scenes: list[str] | None = None
    max_scenes: int | None = None

    # -- Policy connection --
    host: str = "127.0.0.1"
    port: int = 8000
    use_openpi_client: bool = True
    random_policy: bool = False

    # -- Model / observation --
    state_mode: str = "eef_8d_axisangle"
    # When set, overrides each scene's prompt with this template formatted by
    # the target name (same as training: {target_clean} strips the trailing
    # _NNN and underscores). Use to match the SFT prompt distribution.
    prompt_template: str | None = None
    # Optional prompt-ablation table and condition. The table maps each saved
    # instruction to no_instruction, natural_language, or ltl text. Apply the
    # substitution at load time without modifying benchmark files.
    prompt_map: str | None = None
    prompt_condition: str | None = None
    # Task-horizon variant (e.g. cabinet firsthalf): a JSON table substituting a task's
    # goal_conditions + prompt at load time, so a truncated-horizon policy is scored on the
    # goal it was actually trained for instead of the shipped full-horizon one. The SAME
    # table datagen collected the variant's demos with, so "success" means the same thing in
    # both. The benchmark on disk is never modified. None (default) = the shipped task.
    horizon_override: str | None = None
    # External overview camera paired with the wrist image. Match the selected
    # sensor and observation convention to the policy checkpoint. Only this
    # external sensor is created; recorded poses are loaded when available.
    # Choices: opposite, left, right, left_shoulder.
    external_cam: str = "left"
    action_dim: int = 7
    execute_horizon: int = 5
    gripper_binarize: bool = True
    # Arm/gripper controller. Set controller_preset to a key of
    # maniguard.envs.frozen_task_runtime.CONTROLLER_PRESETS (e.g. "osc" for
    # pi0.5 / VLA policies emitting raw 6-D EEF deltas). The scene-baked
    # controller is overridden at load. override_controller_config (a raw
    # dict) takes precedence if both are set.
    controller_preset: str | None = None
    override_controller_config: dict[str, Any] | None = None
    # Runtime grasping mode; choose consistently with the policy's training
    # environment. Supported values: sticky, assisted, physical.
    grasping_mode: str = "sticky"
    # Convert base-frame EEF deltas to joint targets with a damped Jacobian
    # step, then send them to a JointController. This is an approximate local
    # IK conversion, not a trajectory-planning equivalence guarantee.
    ik_eef_to_joint: bool = False
    # Override JointController position-drive stiffness when a controller
    # override is supplied. Damping is set to 2 * sqrt(stiffness).
    joint_pos_kp: float | None = None

    # -- Simulation --
    action_frequency: int = 20
    rendering_frequency: int = 20
    physics_frequency: int = 120
    headless: bool = False
    longfinger: bool = True

    # -- Eval --
    # Which metrics to evaluate — any non-empty subset of {"success", "safety"}:
    #   ["success", "safety"] (default) both; ["success"] success only (Spot not
    #   required); ["safety"] safety only (runs the full rollout, no early stop
    #   on goal). Selects which checkers run and what the summary reports.
    metrics: list[str] = field(default_factory=lambda: ["success", "safety"])
    max_steps: int = 1000
    # Base seed for policy sampling. Derive a scene-specific seed with crc32
    # and send it with each policy request. None leaves sampling unseeded;
    # frozen snapshots alone do not guarantee deterministic physics or inference.
    seed: int | None = None
    # Debounce on success: the goal condition must hold for this many
    # consecutive steps before the episode is marked successful. Guards against
    # single-frame false positives (a transient brush / AG-grasp flicker / the
    # target passing through the goal region). A value of 1 accepts the first true step.
    success_hold_steps: int = 10
    # Thresholds for the derived outcome label. Raw target displacement and
    # minimum EEF-to-target distance remain available for offline relabeling.
    tau_move: float = 0.05    # target drifted > this (m) from spawn -> "manipulated"
    tau_reach: float = 0.12   # eef came within this (m) of target -> "reached"
    camera_resolution: int = 256
    save_video: bool = True
    # Frames buffer in RAM until the rollout ends; at high camera_resolution the
    # wrist stream doubles that peak, so it can be dropped independently.
    save_wrist_video: bool = True

    # -- Output --
    # Run-directory parent. Explicit run_name values can reuse a directory;
    # automatically generated names receive a suffix when a directory exists.
    output_dir: str = ""
    # Run-directory leaf. Empty uses a timestamp; explicit names allow batch
    # processes to append results to the same run directory.
    run_name: str = ""
    # Uppercased suffix for automatically generated run names.
    # Ignored when run_name is supplied explicitly.
    tag: str = ""

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
    # CLI overrides for commonly used configuration fields.
    p.add_argument("--benchmark-root", type=str, default=None)
    p.add_argument("--benchmark-revision", type=str, default=None)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--recording-factory", type=str, default=None)
    p.add_argument("--recording-output-dir", type=str, default=None)
    p.add_argument("--recording-provenance", type=str, default=None)
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--use-openpi-client", action="store_true", default=None)
    p.add_argument("--random-policy", action="store_true", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed for policy sampling noise (per-rollout episode "
                        "seeds are derived from it; omit for unseeded).")
    p.add_argument("--metrics", nargs="*", default=None, choices=["success", "safety"])
    p.add_argument("--success-hold-steps", type=int, default=None)
    p.add_argument("--execute-horizon", type=int, default=None)
    p.add_argument("--action-frequency", type=int, default=None)
    p.add_argument("--rendering-frequency", type=int, default=None)
    p.add_argument("--physics-frequency", type=int, default=None)
    p.add_argument("--headless", action="store_true", default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None,
                   help="Explicit run subfolder leaf under output_dir (skips timestamp).")
    p.add_argument("--tag", type=str, default=None,
                   help="Label folded UPPERCASED into the auto run_name, e.g. --tag smoke.")
    p.add_argument("--save-video", action="store_true", default=None)
    p.add_argument("--external-cam", type=str, default=None,
                   choices=["opposite", "left", "right", "left_shoulder"])
    p.add_argument("--grasping-mode", type=str, default=None, choices=["physical", "assisted", "sticky"])
    p.add_argument("--prompt-map", type=str, default=None,
                   help="Prompt-ablation variant table (configs/ablation_prompt/*.json).")
    p.add_argument("--prompt-condition", type=str, default=None,
                   choices=["no_instruction", "natural_language", "ltl"],
                   help="Which safety-constraint conveyance to evaluate (needs --prompt-map).")
    p.add_argument("--horizon-override", type=str, default=None,
                   help="Task-horizon variant table (configs/firsthalf/*.json): substitutes a "
                        "task's goal_conditions + prompt so a truncated-horizon policy is scored "
                        "on the goal it was trained for. Omit for the shipped full-horizon task.")
    p.add_argument("--camera-resolution", type=int, default=None)

    args = p.parse_args()
    cfg = load_eval_config(args.config)

    cli_map = {
        "benchmark_root": "benchmark_root",
        "benchmark_revision": "benchmark_revision",
        "scenes": "scenes",
        "max_scenes": "max_scenes",
        "recording_factory": "recording_factory",
        "recording_output_dir": "recording_output_dir",
        "recording_provenance": "recording_provenance",
        "host": "host",
        "port": "port",
        "use_openpi_client": "use_openpi_client",
        "random_policy": "random_policy",
        "max_steps": "max_steps",
        "seed": "seed",
        "metrics": "metrics",
        "success_hold_steps": "success_hold_steps",
        "execute_horizon": "execute_horizon",
        "action_frequency": "action_frequency",
        "rendering_frequency": "rendering_frequency",
        "physics_frequency": "physics_frequency",
        "headless": "headless",
        "output_dir": "output_dir",
        "run_name": "run_name",
        "tag": "tag",
        "save_video": "save_video",
        "save_wrist_video": "save_wrist_video",
        "external_cam": "external_cam",
        "grasping_mode": "grasping_mode",
        "camera_resolution": "camera_resolution",
        "prompt_map": "prompt_map",
        "prompt_condition": "prompt_condition",
        "horizon_override": "horizon_override",
    }
    for cli_name, cfg_name in cli_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)

    if bool(cfg.prompt_condition) != bool(cfg.prompt_map):
        raise ValueError(
            "prompt_condition and prompt_map must be set together "
            f"(got condition={cfg.prompt_condition!r}, map={cfg.prompt_map!r})"
        )

    for field_name in ("action_dim", "execute_horizon", "max_steps", "success_hold_steps"):
        value = getattr(cfg, field_name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

    return cfg
