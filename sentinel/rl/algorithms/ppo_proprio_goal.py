#!/usr/bin/env python3
"""Privileged goal-state + proprio PPO on PickAndLiftPrivilegedTask."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from sentinel.rl.algorithms.ppo_proprio import (
    _KitLogTailer,
    _log_stage,
    _make_proprio_scene_copy,
)
from sentinel.rl.cli.common import (
    add_env_args,
    add_training_args,
    add_video_args,
    add_wandb_args,
    validate_env_args,
)


EXPECTED_TASK_LOW_DIM = 10


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Privileged target/goal state + robot proprio PPO on "
            "PickAndLiftPrivilegedTask."
        )
    )
    add_env_args(p)
    add_training_args(p)
    add_wandb_args(p)
    add_video_args(p)
    p.add_argument(
        "--grasping-mode",
        choices=["physical", "assisted", "sticky"],
        default="physical",
        help=(
            "Robot grasping_mode to write into the per-run scene override. "
            "Scene-file runs otherwise inherit OG's default physical mode."
        ),
    )
    return p.parse_args()


def _validate_privileged_obs_space(vec_env) -> None:
    obs_space = getattr(vec_env, "observation_space", None)
    spaces = getattr(obs_space, "spaces", None)
    if not isinstance(spaces, dict):
        raise SystemExit(f"Expected Dict observation space, got: {obs_space}")

    keys = sorted(str(key) for key in spaces)
    print("  obs keys:", flush=True)
    for key in keys:
        print(f"    - {key}", flush=True)

    rgb_keys = [key for key in keys if key.split("::")[-1] == "rgb"]
    proprio_keys = [key for key in keys if key.split("::")[-1] == "proprio"]
    task_low_dim_space = spaces.get("task::low_dim")

    if rgb_keys:
        raise SystemExit(
            "ppo_proprio_goal expected no RGB observations, but found: "
            + ", ".join(rgb_keys)
        )
    if not proprio_keys:
        raise SystemExit(
            "ppo_proprio_goal expected robot proprio observations, but none were found"
        )
    if task_low_dim_space is None:
        raise SystemExit(
            "ppo_proprio_goal expected task::low_dim with target/goal state, "
            "but it was not found"
        )
    if tuple(getattr(task_low_dim_space, "shape", ())) != (EXPECTED_TASK_LOW_DIM,):
        raise SystemExit(
            "ppo_proprio_goal expected task::low_dim shape "
            f"({EXPECTED_TASK_LOW_DIM},), got {task_low_dim_space}"
        )


def main():
    start_time = time.time()
    _log_stage("parsing CLI and validating env args")
    args = parse_args()
    validate_env_args(args)
    args.task_type = "PickAndLiftPrivilegedTask"

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _log_stage("creating proprio scene override")
    source_scene_file = Path(args.scene_file).resolve()
    proprio_scene_file, patched_robots = _make_proprio_scene_copy(
        source_scene_file,
        out,
        grasping_mode=args.grasping_mode,
    )
    args.scene_file = proprio_scene_file
    print(f"  source scene:           {source_scene_file}", flush=True)
    print(f"  proprio scene:          {proprio_scene_file}", flush=True)
    print(f"  patched robot obs:      {patched_robots}", flush=True)
    print(f"  task type:              {args.task_type}", flush=True)
    print(f"  grasping mode:          {args.grasping_mode}", flush=True)

    _log_stage("starting Kit log watcher")
    kit_tailer = _KitLogTailer(since=start_time)
    kit_tailer.start()

    _log_stage("importing OmniGibson and setting macros")
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    _log_stage("importing PPO / SB3 helpers")
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    from sentinel.rl.envs.wrappers import build_vec_env
    from sentinel.rl.training.trainer import run_training

    _log_stage(f"building OmniGibson vec env (num_envs={args.num_envs})")
    vec_env = build_vec_env(args, out_dir=out)
    _validate_privileged_obs_space(vec_env)
    _log_stage("vec env ready")
    set_random_seed(args.seed)

    tensorboard_log_dir = out / f"tb_{time.strftime('%Y%m%d-%H%M%S')}"
    _log_stage(f"tensorboard dir: {tensorboard_log_dir}")

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"--resume-from does not exist: {args.resume_from}")
        _log_stage("loading PPO checkpoint")
        print(f"  resuming from:          {args.resume_from}", flush=True)
        model = PPO.load(
            str(args.resume_from),
            env=vec_env,
            tensorboard_log=str(tensorboard_log_dir),
            device="cuda",
        )
        print(f"  timesteps at ckpt:      {model.num_timesteps:,}", flush=True)
    else:
        _log_stage("constructing fresh PPO model")
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

    try:
        _log_stage("entering shared training loop")
        run_training(
            model,
            args,
            algo_name="ppo_proprio_goal",
            extra_wandb_config={
                "obs_mode": "proprio_target_goal_state",
                "task_type": args.task_type,
                "grasping_mode": args.grasping_mode,
                "task_low_dim": (
                    "target_pos_robot_frame[3], goal_pos_robot_frame[3], "
                    "obj_to_goal_vector[3], goal_radius[1]"
                ),
                "reward_mode": "grasp_acquire_carry_drop_goal",
                "source_scene_file": str(source_scene_file),
                "patched_scene_file": str(proprio_scene_file),
                "patched_robot_obs_count": patched_robots,
            },
        )
    finally:
        kit_tailer.stop()


if __name__ == "__main__":
    main()
