#!/usr/bin/env python3
"""Evaluate a saved RL checkpoint on a scene.

Two surfaces, one implementation:

1. ``evaluate_policy(model, vec_env, n_episodes, deterministic=True) -> dict``
   — shared helper. Used by both this script's ``main()`` and
   ``sentinel.rl.training.callbacks.PeriodicEvalCallback`` so the metric
   definitions stay consistent between training-time periodic eval and
   post-hoc standalone eval.

2. ``python -m sentinel.rl.algorithms.eval`` — standalone CLI. Boots OG from
   ``--scene-file``, loads ``--checkpoint``, runs ``--n-episodes`` of
   deterministic rollout, prints + writes ``metrics.json``.

Metric definitions:
    - ``mean_reward`` / ``std_reward``: cumulative per-episode reward.
    - ``mean_length`` / ``std_length``: per-episode step count.
    - ``success_rate``: fraction of episodes ending via terminated=True
      (InGoalRegion fired); truncated=True (timeout) is not success.
    - ``mean_cost`` / ``std_cost``: sum of ``info["cost"]`` over each episode.
      Currently 0 across the board because PickAndLiftTask doesn't emit
      ``cost`` yet — this field is ready for the constrained-RL Phase 1
      wiring.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.algorithms.eval \\
            --category mug --model kewbyf --target-name target_mug \\
            --scene-file outputs/pipeline_runs/.../scene_ep1.json \\
            --grasp-dataset-dir outputs/grasp_datasets/mug_scene_ep1 \\
            --checkpoint outputs/ppo_grasp_reset/.../ckpts/ppo_5000000_steps.zip \\
            --n-episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

from sentinel.rl.cli.common import add_env_args, validate_env_args


def evaluate_policy(
    model,
    vec_env,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Run ``n_episodes`` rollouts; return aggregate metrics.

    Uses the provided ``vec_env`` as-is (caller owns its lifecycle). Loops
    until ``n_episodes`` completed episodes have been collected across all
    parallel envs.
    """
    n_envs = vec_env.num_envs
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_successes: list[bool] = []
    ep_costs: list[float] = []

    # Per-env rolling accumulators (persist across env.step boundaries until
    # the env hits done).
    current_reward = np.zeros(n_envs)
    current_length = np.zeros(n_envs, dtype=int)
    current_cost = np.zeros(n_envs)

    obs = vec_env.reset()
    while len(ep_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        current_reward += rewards
        current_length += 1
        for i, info in enumerate(infos):
            current_cost[i] += float(info.get("cost", 0.0))
            if not dones[i]:
                continue
            truncated = bool(info.get("TimeLimit.truncated", False))
            ep_rewards.append(float(current_reward[i]))
            ep_lengths.append(int(current_length[i]))
            ep_successes.append(not truncated)
            ep_costs.append(float(current_cost[i]))
            current_reward[i] = 0.0
            current_length[i] = 0
            current_cost[i] = 0.0
            if len(ep_rewards) >= n_episodes:
                break

    rewards_arr = np.asarray(ep_rewards[:n_episodes])
    lengths_arr = np.asarray(ep_lengths[:n_episodes])
    costs_arr = np.asarray(ep_costs[:n_episodes])
    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_length": float(lengths_arr.mean()),
        "std_length": float(lengths_arr.std()),
        "success_rate": float(sum(ep_successes[:n_episodes]) / n_episodes),
        "mean_cost": float(costs_arr.mean()),
        "std_cost": float(costs_arr.std()),
        "n_episodes": int(n_episodes),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a saved RL checkpoint on a scene."
    )
    add_env_args(p)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a saved SB3 checkpoint (.zip).")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/rl_eval"))
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic policy (default: deterministic).")
    # Eval always runs single-env (cleaner episode accounting); override
    # --num-envs if the user passes it explicitly.
    return p.parse_args()


def main():
    args = parse_args()
    # Force single-env for eval — episode boundaries are trivial this way, and
    # we don't need the multi-env speedup when only running N=20-ish episodes.
    # (If the user explicitly wants multi-env eval, they can pass --num-envs
    # >1; we only override when it's still the default of 1.)
    validate_env_args(args)

    if not args.checkpoint.exists():
        raise SystemExit(f"--checkpoint does not exist: {args.checkpoint}")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    from stable_baselines3 import PPO

    from sentinel.rl.envs.wrappers import build_vec_env

    vec_env = build_vec_env(args, out_dir=out)

    print(f"[{time.strftime('%H:%M:%S')}] Loading {args.checkpoint}...",
          flush=True)
    model = PPO.load(str(args.checkpoint), env=vec_env, device="cuda")
    print(f"  model timesteps:   {model.num_timesteps:,}", flush=True)

    deterministic = not args.stochastic
    print(f"[{time.strftime('%H:%M:%S')}] Running {args.n_episodes} "
          f"{'deterministic' if deterministic else 'stochastic'} episodes...",
          flush=True)
    t0 = time.time()
    metrics = evaluate_policy(
        model, vec_env,
        n_episodes=args.n_episodes,
        deterministic=deterministic,
    )
    dt = time.time() - t0

    print(f"\n=== Eval results ({dt:.1f}s) ===")
    for k, v in metrics.items():
        print(f"  {k:16s} {v}")

    out_json = out / (
        f"eval_{args.checkpoint.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    with open(out_json, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "scene_file": str(args.scene_file) if args.scene_file else None,
            "category": args.category, "model": args.model,
            "target_name": args.target_name,
            "deterministic": deterministic,
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nWrote {out_json}")

    sys.stdout.flush()
    os._exit(0)


__all__ = ["evaluate_policy"]


if __name__ == "__main__":
    main()
