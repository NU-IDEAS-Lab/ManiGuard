#!/usr/bin/env python3
"""Minimal PPO training on the grasp-reset PickAndLiftTask setup.

Trains a proprio-only MlpPolicy on the empty scene + breakfast_table +
single DatasetObject setup, with PickAndLiftTask resetting every episode
into a saved grasp from ``grasps_<category>_<model>.pt``.

Intended as a pipeline sanity check:
  - PPO successfully consumes OG's dict/flat obs + action spaces
  - Reward signal (grasp_reward + carry_shaping + goal_bonus) drives
    learning
  - InGoalRegion terminates episodes and contributes terminal reward

Not a full training run — ``--total-timesteps`` defaults to ~20k which
completes in minutes and is enough to see mean_episode_reward trend.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.training.ppo_grasp_reset \\
            --category soda_cup --model fsfsas --total-timesteps 20000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="soda_cup")
    p.add_argument("--model", default="fsfsas")
    p.add_argument("--grasp-dataset-dir", type=Path,
                   default=Path("outputs/grasp_datasets/batch"))
    p.add_argument("--total-timesteps", type=int, default=20_000)
    p.add_argument("--n-steps", type=int, default=128,
                   help="PPO rollout length per env before update.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_grasp_reset"))
    return p.parse_args()




def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    target_name = f"target_{args.category}_{args.model}"
    grasp_path = args.grasp_dataset_dir / f"grasps_{args.category}_{args.model}.pt"
    if not grasp_path.exists():
        raise SystemExit(f"Missing grasp dataset: {grasp_path}")

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv

    import sentinel.rl.tasks  # noqa: F401  (register PickAndLiftTask)
    from sentinel.rl.envs.grasp_reset_scene import build_config

    cfg = build_config(
        target_name=target_name,
        category=args.category,
        model=args.model,
        grasp_dataset_path=grasp_path,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Booting OG environment...", flush=True)
    env = og.Environment(configs=cfg)

    # Wrap for SB3 — Monitor records episode returns, DummyVecEnv gives a
    # vectorized interface with one instance.
    print(f"  obs_space: {env.observation_space}", flush=True)
    print(f"  action_space: {env.action_space}", flush=True)

    vec_env = DummyVecEnv([lambda: Monitor(env, filename=str(out / "monitor.csv"))])

    set_random_seed(args.seed)

    tensorboard_log_dir = out / f"tb_{time.strftime('%Y%m%d-%H%M%S')}"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # MultiInputPolicy handles OG's dict observation space (proprio + task
    # components stay separate even with flatten_obs_space=True because the
    # flatten applies within each dict entry, not across them).
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

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.total_timesteps // 5),
            save_path=str(out / "ckpts"),
            name_prefix="ppo_grasp_reset",
        ),
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Starting PPO training "
          f"(target={args.total_timesteps} steps)...", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    dt = time.time() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] Training done in {dt:.1f}s "
          f"({args.total_timesteps / max(1, dt):.1f} steps/s)", flush=True)

    final_ckpt = out / "ppo_grasp_reset_final.zip"
    model.save(str(final_ckpt))
    print(f"  saved final checkpoint: {final_ckpt}", flush=True)
    print(f"  tensorboard logs:       {tensorboard_log_dir}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
