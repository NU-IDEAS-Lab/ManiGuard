#!/usr/bin/env python3
"""Vanilla PPO on the grasp-reset PickAndLiftTask.

Baseline before the constrained variants (PPO-Lag, FOCOPS, CUPS). All
non-algorithm bookkeeping (CLI args, env build, callback + wandb wiring,
final save) is shared via ``sentinel.rl.cli.common``,
``sentinel.rl.envs.wrappers``, and ``sentinel.rl.training.trainer`` — so this
file contains only the PPO-specific model construction.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.algorithms.ppo \\
            --category mug --model kewbyf --target-name target_mug \\
            --scene-file outputs/pipeline_runs/.../scene_ep1.json \\
            --grasp-dataset-dir outputs/grasp_datasets/mug_scene_ep1 \\
            --num-envs 4 --total-timesteps 10000000 \\
            --n-steps 256 --batch-size 128 \\
            --wandb --video-freq 100000
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from sentinel.rl.cli.common import (
    add_env_args, add_training_args, add_video_args, add_wandb_args,
    validate_env_args,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Vanilla PPO on PickAndLiftTask with grasp-reset."
    )
    add_env_args(p)
    add_training_args(p)
    add_wandb_args(p)
    add_video_args(p)
    # (PPO-specific knobs go here — gamma/gae_lambda stay at SB3 defaults
    # since this is the baseline.)
    return p.parse_args()


def main():
    args = parse_args()
    validate_env_args(args)

    # OG macros must be set before omnigibson is imported (which happens inside
    # build_vec_env via sb3_vec).
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    from sentinel.rl.envs.wrappers import build_vec_env
    from sentinel.rl.training.trainer import run_training

    out = Path(args.output_dir).resolve()
    vec_env = build_vec_env(args, out_dir=out)
    set_random_seed(args.seed)

    tensorboard_log_dir = out / f"tb_{time.strftime('%Y%m%d-%H%M%S')}"

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"--resume-from does not exist: {args.resume_from}")
        print(f"  resuming from:          {args.resume_from}", flush=True)
        # PPO.load restores weights + optimizer state + num_timesteps. We
        # explicitly pass env + tensorboard_log + device because those aren't
        # preserved in the zip (they're env/runtime-dependent).
        model = PPO.load(
            str(args.resume_from),
            env=vec_env,
            tensorboard_log=str(tensorboard_log_dir),
            device="cuda",
        )
        print(f"  timesteps at ckpt:      {model.num_timesteps:,}", flush=True)
    else:
        # MultiInputPolicy handles OG's dict observation space (proprio + task
        # components stay separate even under flatten_obs_space=True because
        # the flatten applies within each dict entry, not across them).
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(tensorboard_log_dir),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            device="cuda",
            seed=args.seed,
        )

    run_training(model, args, algo_name="ppo")


if __name__ == "__main__":
    main()
