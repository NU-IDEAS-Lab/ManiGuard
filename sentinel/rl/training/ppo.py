"""Train a PPO agent on a GraspTask over a clutter benchmark scene.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
        python -m sentinel.rl.training.ppo \
            --scene-dir datasets/safety-benchmark/clutter_goblet_00 \
            --num-envs 4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import omnigibson as og
from omnigibson.macros import gm

# Speed / stability settings applied before simulator starts.
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import is_image_space
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
except ModuleNotFoundError:
    og.log.error(
        "stable-baselines3 is not installed. "
        "Run: pip install stable-baselines3[extra]"
    )
    raise

from omnigibson.utils.python_utils import meets_minimum_version

from sentinel.rl.envs.benchmark_scene import build_config
from sentinel.rl.envs.sb3_vec import SentinelSB3VectorEnvironment
from sentinel.rl.training.callbacks import GCCallback, SafetyCallback
from sentinel.rl.training.extractors import RGBCombinedExtractor

assert meets_minimum_version(gym.__version__, "0.28.1"), "gymnasium >= 0.28.1 required"


def _force_sticky_grasp(env) -> None:
    """Override the scene-loaded robot's grasping_mode to ``sticky``.

    Clutter scene files don't set ``grasping_mode`` on FrankaMounted, so it
    defaults to ``physical`` — requires precise finger contact + friction,
    which PPO from pixels cannot learn. Sticky mode attaches the object once
    the gripper is close enough and closed, giving a tractable grasp signal.
    """
    if env.robots and hasattr(env.robots[0], "_grasping_mode"):
        env.robots[0]._grasping_mode = "sticky"


def make_env(scene_dir: Path):
    env = og.Environment(configs=build_config(scene_dir))
    _force_sticky_grasp(env)
    return env


def make_vec_env(scene_dir: Path, num_envs: int = 1):
    if num_envs > 1:
        env = SentinelSB3VectorEnvironment(
            num_envs=num_envs, config=build_config(scene_dir), render_on_step=False
        )
        for inner in env.envs:
            _force_sticky_grasp(inner)
    else:
        env = DummyVecEnv([lambda: make_env(scene_dir)])
    env = VecMonitor(env)

    space = env.observation_space
    needs_transpose = False
    if hasattr(space, "spaces"):
        needs_transpose = any(is_image_space(s) for s in space.spaces.values())
    else:
        needs_transpose = is_image_space(space)
    if needs_transpose:
        env = VecTransposeImage(env)
    return env


def parse_args():
    p = argparse.ArgumentParser(description="PPO GraspTask training on a clutter scene.")
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Path to a benchmark scene dir (scene_ep1.json + diagnostics.jsonl).")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Parallel env count via SentinelSB3VectorEnvironment.")
    p.add_argument("--checkpoint", type=str, default=None, help="PPO checkpoint .zip path.")
    p.add_argument("--eval", action="store_true", help="Evaluate --checkpoint instead of training.")
    p.add_argument("--resume", action="store_true", help="Resume training from --checkpoint.")
    p.add_argument("--print-checkpoint-keys", action="store_true",
                   help="Print observation keys in --checkpoint and exit.")
    p.add_argument("--eval-during-training", action="store_true",
                   help="Run EvalCallback during training (single sim → eval env shares process).")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    # wandb (opt-in). Login via `wandb login` or WANDB_API_KEY; WANDB_MODE=offline
    # works too for air-gapped hosts (sync later with `wandb sync`).
    p.add_argument("--wandb", action="store_true", help="Log metrics + checkpoints to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="sentinel-lite-grasp")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B team / user (default: personal).")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Override auto-generated run name.")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-upload-ckpts", action="store_true",
                   help="Upload checkpoints as W&B artifacts (off by default — .zip per 1k steps is noisy).")
    return p.parse_args()


def main():
    args = parse_args()

    # No-env paths first — avoid spinning up a training env just to inspect a checkpoint.
    if args.print_checkpoint_keys:
        assert args.checkpoint is not None, "--print-checkpoint-keys requires --checkpoint"
        model = PPO.load(args.checkpoint)
        og.log.info("Checkpoint observation keys:")
        if hasattr(model.observation_space, "spaces"):
            for key in sorted(model.observation_space.spaces.keys()):
                og.log.info("  %s", key)
        else:
            og.log.info("  (non-dict observation space)")
        return

    if args.eval:
        assert args.checkpoint is not None, "--eval requires --checkpoint"
        model = PPO.load(args.checkpoint)
        eval_env = make_vec_env(args.scene_dir)
        og.log.info("Starting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        og.log.info("Finished evaluation!")
        og.log.info(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
        return

    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    env = make_vec_env(args.scene_dir, num_envs=args.num_envs)
    set_random_seed(args.seed)
    env.reset()

    policy_kwargs = dict(features_extractor_class=RGBCombinedExtractor)

    if args.resume:
        assert args.checkpoint is not None, "--resume requires --checkpoint"
        model = PPO.load(args.checkpoint, env=env, device="cuda")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            policy_kwargs=policy_kwargs,
            n_steps=20 * 10,
            batch_size=8,
            device="cuda",
        )

    callbacks = [
        CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix="grasp"),
        SafetyCallback(),
        GCCallback(collect_every=1000, verbose=1),
    ]
    if args.eval_during_training:
        og.log.warning(
            "EvalCallback enabled during training — OmniGibson has one active sim; "
            "running eval in a separate process is recommended."
        )
        callbacks.append(EvalCallback(eval_env=make_vec_env(args.scene_dir), eval_freq=1000, n_eval_episodes=20))

    wandb_run = None
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run_name = args.wandb_run_name or f"{args.scene_dir.name}_n{args.num_envs}_{time.strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            sync_tensorboard=True,  # mirror SB3's TB events → W&B panels
            save_code=False,
            config={
                "scene_dir": str(args.scene_dir),
                "num_envs": args.num_envs,
                "total_timesteps": args.total_timesteps,
                "seed": args.seed,
                "resume": args.resume,
                "checkpoint": args.checkpoint,
                "algo": "ppo",
                "n_steps": 200,
                "batch_size": 8,
                "learning_rate": 3e-4,
            },
        )
        callbacks.append(WandbCallback(
            model_save_path=tensorboard_log_dir if args.wandb_upload_ckpts else None,
            model_save_freq=5000 if args.wandb_upload_ckpts else 0,
            gradient_save_freq=0,
            verbose=2,
        ))

    try:
        og.log.info("Starting training...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not args.resume,
        )
        og.log.info("Finished training!")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
