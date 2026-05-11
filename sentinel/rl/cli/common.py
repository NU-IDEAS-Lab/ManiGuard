"""Argparse builders shared across sentinel RL algorithm entry points.

Split into four self-contained groups so each algorithm file can opt in:

    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_training_args(parser)
    add_wandb_args(parser)
    add_video_args(parser)
    # ... then add algorithm-specific args (lagrange lr, focops k, ...)
    args = parser.parse_args()

``validate_env_args(args)`` raises ``SystemExit`` with a clear message if a
combination is unsupported (e.g. ``--num-envs > 1`` without ``--scene-file``),
and should be called once after ``parse_args()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_env_args(parser: argparse.ArgumentParser) -> None:
    """Scene / grasp dataset / robot-controller wiring.

    Shared by every algo that trains on the grasp-reset setup.
    """
    g = parser.add_argument_group("env")
    g.add_argument("--category", default="mug")
    g.add_argument("--model", default="kewbyf")
    g.add_argument("--target-name", type=str, default=None,
                   help="Override target object name in scene. Defaults to "
                        "target_<category>_<model>. When --scene-file is set, "
                        "must match the name inside the saved scene.")
    g.add_argument("--grasp-dataset-dir", type=Path,
                   default=Path("outputs/grasp_datasets/batch"),
                   help="Directory containing grasps_<cat>_<model>.pt.")
    g.add_argument("--scene-file", type=Path, default=None,
                   help="Pre-baked OG scene JSON. Required for --num-envs > 1.")
    g.add_argument("--num-envs", type=int, default=1,
                   help="Parallel env count via SentinelSB3VectorEnvironment. "
                        "Requires --reset-mode cached + --scene-file.")
    g.add_argument("--reset-mode", choices=["cached", "ik"], default="cached",
                   help="'cached' (default): load arm_joint_pos directly, no "
                        "online IK, multi-env works. 'ik': compose "
                        "T_eef_world @ reset and run cuRobo IK; supports "
                        "pose-range perturbation but single-env only.")
    g.add_argument("--arm-controller", choices=["joint", "osc"], default="joint",
                   help="'joint' (default): JointController (position delta, "
                        "no Jacobian inversion). 'osc': OperationalSpace; "
                        "crashes with LinAlgError on kinematic singularity.")
    g.add_argument("--max-steps", type=int, default=200,
                   help="Episode length (task timeout in env steps).")
    g.add_argument("--diagnostics-file", type=Path, default=None,
                   help="Path to a benchmark task's diagnostics.jsonl. "
                        "When provided, the goal region from the benchmark "
                        "is used instead of the hardcoded goal_offset. "
                        "Also infers --scene-file from the sibling "
                        "scene_ep1.json if not explicitly set.")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Common SB3-style training knobs + output paths."""
    g = parser.add_argument_group("training")
    g.add_argument("--total-timesteps", type=int, default=20_000)
    g.add_argument("--n-steps", type=int, default=128,
                   help="Rollout length per env before on-policy update.")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--learning-rate", type=float, default=3e-4)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--output-dir", type=Path,
                   default=Path("outputs/rl_run"),
                   help="Run artifacts: ckpts/, videos/, tb_*, monitor.csv.")
    g.add_argument("--save-freq", type=int, default=250_000,
                   help="Save a checkpoint every N ENV-steps (not vec-steps). "
                        "Divided by --num-envs internally because SB3's "
                        "CheckpointCallback counts vec-step calls.")
    g.add_argument("--resume-from", type=Path, default=None,
                   help="Path to a checkpoint .zip to continue training from. "
                        "Timesteps counter preserved (reset_num_timesteps=False) "
                        "so TB / wandb stays continuous with the original run.")
    g.add_argument("--eval-freq", type=int, default=0,
                   help="Run K deterministic eval episodes every N env-steps "
                        "(at rollout boundaries, to avoid disturbing the "
                        "training rollout buffer). 0 = disabled. "
                        "Typical: 500_000 with --n-eval-episodes 10.")
    g.add_argument("--n-eval-episodes", type=int, default=10,
                   help="Episodes per periodic eval sweep (only used when "
                        "--eval-freq > 0).")


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Weights & Biases opt-in logging (mirrors SB3 TB via sync_tensorboard)."""
    g = parser.add_argument_group("wandb")
    g.add_argument("--wandb", action="store_true",
                   help="Log metrics + run config to Weights & Biases.")
    g.add_argument("--wandb-project", type=str, default="sentinel-grasp-reset")
    g.add_argument("--wandb-entity", type=str, default=None)
    g.add_argument("--wandb-run-name", type=str, default=None)
    g.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-upload-ckpts", action="store_true",
                   help="Upload checkpoints as W&B artifacts.")


def add_video_args(parser: argparse.ArgumentParser) -> None:
    """Periodic viewer-camera clip recording (ViewerVideoCallback)."""
    g = parser.add_argument_group("video")
    g.add_argument("--video-freq", type=int, default=0,
                   help="Record a clip every N env-steps. 0 = disabled.")
    g.add_argument("--video-length", type=int, default=200,
                   help="Frames per clip (≈ one episode at max_steps=200).")


def validate_env_args(args) -> None:
    """Reject combinations the env builder can't satisfy.

    Raises SystemExit with a clear message (not an exception — we want the
    user to see the error cleanly, not a traceback).
    """
    if getattr(args, "diagnostics_file", None) is not None:
        if not args.diagnostics_file.exists():
            raise SystemExit(
                f"--diagnostics-file does not exist: {args.diagnostics_file}"
            )
        if args.scene_file is None:
            inferred = args.diagnostics_file.parent / "scene_ep1.json"
            if not inferred.exists():
                raise SystemExit(
                    f"--diagnostics-file was provided but sibling scene_ep1.json "
                    f"not found at {inferred}. Set --scene-file explicitly."
                )
            args.scene_file = inferred
    if args.num_envs > 1 and args.scene_file is None:
        raise SystemExit(
            "--num-envs > 1 requires --scene-file pointing at a pre-baked OG "
            "scene JSON. Empty-scene + runtime-spawn breaks OG's scene tiling "
            "at idx != 0."
        )
    if args.num_envs > 1 and args.reset_mode == "ik":
        raise SystemExit(
            "--num-envs > 1 requires --reset-mode cached. Online cuRobo IK "
            "trips ``assert len(og.sim.scenes) == 1`` in "
            "CuRoboMotionGenerator.__init__."
        )
    if args.scene_file is not None and not args.scene_file.exists():
        raise SystemExit(f"--scene-file does not exist: {args.scene_file}")


__all__ = [
    "add_env_args",
    "add_training_args",
    "add_wandb_args",
    "add_video_args",
    "validate_env_args",
]
