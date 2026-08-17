"""Unified eval configuration.

One YAML file controls the entire eval run: benchmark source, policy
connection, model-specific knobs (state/action), sim frequencies, and
output settings.  CLI args override any field for quick ad-hoc tweaks.

Usage:
    python -m maniguard.eval.benchmark --config configs/eval/clutter_pickup_joint.yaml
    python -m maniguard.eval.benchmark --config configs/eval/clutter_pickup_joint.yaml --max-steps 500
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    name: str = "unnamed"

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
    # Prompt-ablation (Q2): replace each scene's instruction with the variant that
    # conveys the safety constraint differently -- "no_instruction" (today's data),
    # "natural_language", or "ltl". The variants come from prompt_map, the SAME table
    # the ablation's SFT datasets are rewritten from, so training and eval see
    # byte-identical prompts. The benchmark itself is never modified; the swap happens
    # at load time. Leave both None for a normal run.
    prompt_map: str | None = None
    prompt_condition: str | None = None
    # Task-horizon variant (e.g. cabinet firsthalf): a JSON table substituting a task's
    # goal_conditions + prompt at load time, so a truncated-horizon policy is scored on the
    # goal it was actually trained for instead of the shipped full-horizon one. The SAME
    # table datagen collected the variant's demos with, so "success" means the same thing in
    # both. The benchmark on disk is never modified. None (default) = the shipped task.
    horizon_override: str | None = None
    # Which third-person overview the policy consumes. The model is fed exactly
    # ONE external overview + the wrist (LIBERO 2-cam convention): the policy
    # server reads observation/image_left + observation/wrist_image (+ state).
    # This selects which physical camera supplies that single overview, and MUST
    # match the checkpoint's training config (Sim2CamLiberoDataConfig.external_cam)
    # so the policy stays in distribution. Choices = the datagen contract
    # (data_format.EXTERNAL_CAM_CHOICES): opposite / left / right / left_shoulder;
    # cam_<name> is rendered and sent as observation/image_left. Only that one
    # external camera is rendered (the others are never created); its POSE is
    # loaded from the task's diagnostics["cameras"] (same as datagen), never
    # recomputed.
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
    # Robot grasping semantics, forced on the eval robot AFTER scene load (the
    # ManiGuard-Bench scene's baked grasping_mode is overridden). MUST match the grasp
    # mode used to COLLECT the training data, or the policy's learned gripper
    # behaviour won't grasp: "sticky" welds an object on ANY single-finger
    # contact, "assisted" requires two fingers, "physical" is pure friction.
    # The training datasets do NOT record this, so it is set explicitly here —
    # the joint teleop families were collected in "sticky".
    #   choices: "sticky" | "assisted" | "physical"
    grasping_mode: str = "sticky"
    # When true, treat the policy output as a 6-D eef delta (+ gripper) and
    # convert it to absolute joint targets via a Jacobian IK step, then feed
    # those to a JointController (PD position tracking). This matches how the
    # SFT data was generated (cuRobo joint targets tracked by a JointController)
    # so the realized joint path follows training instead of diverging.
    # Use with controller_preset: joint_position_impedance.
    ik_eef_to_joint: bool = False
    # Override the JointController impedance stiffness (pos_kp). The default
    # (50) is too soft to reach the per-step joint target within one control
    # step (~10% tracking), which double-softens the already-PD-tracked SFT
    # deltas. A high value makes the controller reach its target each step
    # (achieved eef == commanded delta). Only applied to a JointController arm.
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
    # Base seed for the policy's action-sampling noise (the only substantive
    # randomness at eval: scenes are frozen snapshots). Per rollout the client
    # derives episode_seed = crc32(f"{seed}:{scene_name}") and sends it in every
    # request; each policy server re-seeds its sampler (JAX key / torch RNG)
    # when the value changes. None (default) = unseeded, previous behavior.
    # Distinct base seeds give independent repeat trials of the same task.
    seed: int | None = None
    # Debounce on success: the goal condition must hold for this many
    # consecutive steps before the episode is marked successful. Guards against
    # single-frame false positives (a transient brush / AG-grasp flicker / the
    # target passing through the goal region). 1 = legacy first-frame behaviour.
    success_hold_steps: int = 10
    # -- Engagement metric outcome thresholds (docs/evaluation/engagement_metric.md).
    # These only affect the derived `outcome` label, NOT the raw per-rollout signals
    # (target2spawn_max_dist / eef2target_min_dist), which are always logged, so the
    # labels can be recomputed offline. The defaults separate cleanly across families.
    tau_move: float = 0.05    # target drifted > this (m) from spawn -> "manipulated"
    tau_reach: float = 0.12   # eef came within this (m) of target -> "reached"
    camera_resolution: int = 256
    save_video: bool = True

    # -- Output --
    # Per-CONFIG base directory. Each run writes to <output_dir>/<run_name> so
    # successive runs never overwrite each other (results, videos, LTL/summary
    # sidecars all land in the run subfolder).
    output_dir: str = ""
    # Per-RUN subfolder leaf under output_dir. Empty (default) -> benchmark.py
    # auto-generates a "YYYYmmdd_HHMMSS" timestamp, loud-suffixed with `tag`.
    # Set explicitly (e.g. --run-name baseline_v2) to name a run, or to make a
    # multi-scene batch share ONE folder (run_benchmark_all_scenes.sh generates
    # one run_name and passes the same --run-name to every per-scene process).
    run_name: str = ""
    # Free-form label folded UPPERCASED into the auto-generated run_name, e.g.
    # --tag smoke -> "20260607_143000_SMOKE". Always tag smoke/test runs so a
    # throwaway folder can never be mistaken for a real eval. Ignored when
    # run_name is set explicitly.
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

    return cfg
