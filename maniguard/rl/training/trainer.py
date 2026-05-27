"""Shared training loop for every maniguard RL algorithm.

Each algorithm's main() builds its own SB3-compatible model (PPO, PPO-Lag,
FOCOPS, CUPS, …) then hands it off here. This keeps callback wiring, wandb
lifecycle, and final-save logic in exactly one place — so adding a new
constrained algorithm means writing ~60 lines of algo-specific code, not
copy-pasting 100 lines of bookkeeping.

Usage:
    from maniguard.rl.training.trainer import run_training
    run_training(model, args, algo_name="ppo_lag",
                 extra_wandb_config={"cost_limit": args.cost_limit, ...})
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _build_common_callbacks(args, out_dir: Path, algo_name: str) -> list:
    """Checkpoint + GC + optional ViewerVideo, in that order.

    Ordering note: ViewerVideoCallback must run after env.step so that
    viewer_camera has the up-to-date frame. SB3 runs callbacks in list order
    per step; all three are post-step so order is cosmetic.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback

    from maniguard.rl.training.callbacks import (
        GCCallback, PeriodicEvalCallback, ViewerVideoCallback,
    )

    # SB3's CheckpointCallback counts ``n_calls`` (one per vec-step), not env
    # steps — so args.save_freq (in env-step units) must be divided by the
    # number of parallel envs. Without this, ``save_freq=2_000_000`` with
    # ``num_envs=4`` fires only every 8M env-steps (and won't fire at all in
    # runs shorter than that).
    save_freq_vec = max(1, args.save_freq // max(1, args.num_envs))
    callbacks = [
        CheckpointCallback(
            save_freq=save_freq_vec,
            save_path=str(out_dir / "ckpts"),
            name_prefix=algo_name,
        ),
        # Python GC every 1k vec-step calls — clears cycle-held obs buffers
        # from OG's per-step path. Does not address C-level PhysX leaks.
        GCCallback(collect_every=1000, verbose=1),
    ]
    if args.video_freq > 0:
        callbacks.append(ViewerVideoCallback(
            save_freq=args.video_freq,
            clip_length=args.video_length,
            save_dir=out_dir / "videos",
            fps=30,
            verbose=1,
        ))
    if args.eval_freq > 0:
        callbacks.append(PeriodicEvalCallback(
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ))
    return callbacks


def _init_wandb(args, algo_name: str, extra_config: dict | None = None):
    """Initialise a wandb run and return (wandb_run, WandbCallback) or (None, None).

    If ``<output-dir>/wandb_run_id.txt`` already exists (e.g. a watchdog
    re-entered after a crash), resume that run instead of starting a new
    one. Persists the run ID on first launch so subsequent attempts find it.
    """
    if not args.wandb:
        return None, None
    import wandb
    from wandb.integration.sb3 import WandbCallback

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id_file = out_dir / "wandb_run_id.txt"
    existing_run_id = run_id_file.read_text().strip() if run_id_file.exists() else None

    run_name = args.wandb_run_name or (
        f"{algo_name}_{args.category}_{args.model}_n{args.num_envs}_"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    config = {
        "algo": algo_name,
        "policy": "MultiInputPolicy",
        "category": args.category,
        "model": args.model,
        "target_name": args.target_name,
        "scene_file": str(args.scene_file) if args.scene_file else None,
        "num_envs": args.num_envs,
        "reset_mode": args.reset_mode,
        "arm_controller": args.arm_controller,
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    if extra_config:
        config.update(extra_config)

    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        sync_tensorboard=True,  # mirror SB3 TB events → wandb panels
        save_code=False,
        config=config,
    )
    if existing_run_id:
        # ``resume="allow"`` attaches to the existing run; the timeline
        # continues seamlessly. The run name + config from the original
        # ``init`` are preserved server-side.
        init_kwargs["id"] = existing_run_id
        init_kwargs["resume"] = "allow"
        print(f"  resuming wandb run id:  {existing_run_id}", flush=True)

    run = wandb.init(**init_kwargs)
    if not existing_run_id:
        run_id_file.write_text(run.id)
    cb = WandbCallback(
        model_save_path=str(Path(args.output_dir).resolve() / "ckpts")
        if args.wandb_upload_ckpts else None,
        model_save_freq=max(1, args.total_timesteps // 5)
        if args.wandb_upload_ckpts else 0,
        gradient_save_freq=0,
        verbose=2,
    )
    print(f"  wandb run:              {run.get_url()}", flush=True)
    return run, cb


def run_training(model, args, *, algo_name: str,
                 extra_wandb_config: dict | None = None) -> None:
    """Run ``model.learn()`` with shared callbacks + wandb + final save.

    Algorithm entry points should call this exactly once after constructing
    their model. Exits the process on completion (``os._exit(0)``) to cut
    through OG's noisy shutdown phase.

    Args:
        model: an SB3-compatible algorithm instance with ``.learn()`` and
            ``.save()``. PPO from ``stable_baselines3`` is the canonical case;
            constrained variants (PPO-Lag, FOCOPS, CUPS) must conform to the
            same contract.
        args: parsed argparse Namespace. Expected fields come from
            ``maniguard.rl.cli.common.add_{env,training,wandb,video}_args``.
        algo_name: short tag (``"ppo"``, ``"ppo_lag"``, …). Used for
            checkpoint filename prefix, wandb config, and default run name.
        extra_wandb_config: algorithm-specific hyperparameters to log alongside
            the shared config (e.g. ``{"cost_limit": ..., "lagrange_lr": ...}``
            for PPO-Lag).
    """
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = _build_common_callbacks(args, out_dir, algo_name)
    wandb_run, wandb_cb = _init_wandb(args, algo_name, extra_wandb_config)
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    # Preserve SB3's internal timesteps counter on resume so TB/wandb
    # continue from the original run's step count instead of restarting at 0.
    resume = getattr(args, "resume_from", None) is not None

    try:
        print(f"[{time.strftime('%H:%M:%S')}] Starting {algo_name} training "
              f"(target={args.total_timesteps} steps"
              f"{', resumed' if resume else ''})...", flush=True)
        t0 = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not resume,
        )
        dt = time.time() - t0
        print(f"\n[{time.strftime('%H:%M:%S')}] Training done in {dt:.1f}s "
              f"({args.total_timesteps / max(1, dt):.1f} steps/s)", flush=True)

        final_ckpt = out_dir / f"{algo_name}_final.zip"
        model.save(str(final_ckpt))
        print(f"  saved final checkpoint: {final_ckpt}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    sys.stdout.flush()
    os._exit(0)


__all__ = ["run_training"]
