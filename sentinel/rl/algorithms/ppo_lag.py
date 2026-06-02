#!/usr/bin/env python3
"""PPO-Lag: Lagrangian-relaxation constrained PPO.

Algorithm design ported from omnisafe's PPOLag
(omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py, Apache-2.0).

The surrogate advantage is:

    adv = (adv_r − λ · adv_c) / (1 + λ)

where λ is updated after every rollout by:

    λ ← λ + lr_λ × (ep_cost_mean − cost_limit)   (clamped to ≥ 0)

Everything else (two-critic rollout, GAE on cost stream, cost-critic MSE
loss) is inherited from ConstrainedPPO.

Entry point mirrors ppo_proprio_goal.py.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch as th

from sentinel.rl.cli.common import (
    add_cost_args,
    add_env_args,
    add_lagrange_args,
    add_training_args,
    add_video_args,
    add_wandb_args,
    validate_env_args,
)
from sentinel.rl.safe.constrained_ppo import ConstrainedPPO
from sentinel.rl.safe.lagrange import Lagrange


class PPOLag(ConstrainedPPO):
    """Lagrangian PPO — smallest safe-RL delta above vanilla PPO.

    Inherits the full two-critic + cost-stream machinery from
    ``ConstrainedPPO`` and overrides only the two hooks:

    * ``_before_update`` — update λ from the just-collected rollout cost.
    * ``_compute_adv_surrogate`` — weight reward/cost advantages by λ.

    All SB3 ``.learn()`` / ``.save()`` / ``.load()`` contracts are
    preserved so ``run_training`` works without modification.
    """

    def __init__(
        self,
        *args: Any,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.035,
        lambda_optimizer: str = "Adam",
        lagrangian_upper_bound: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lagrange = Lagrange(
            cost_limit=self.cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            lagrangian_upper_bound=lagrangian_upper_bound,
        )

    # ------------------------------------------------------------------
    def _before_update(self) -> None:
        ep_cost = getattr(self, "_last_ep_cost_mean", 0.0)
        self.lagrange.update_lagrange_multiplier(ep_cost)
        lam = float(self.lagrange.lagrangian_multiplier.item())
        self.logger.record("train/lagrangian_multiplier", lam)

    def _compute_adv_surrogate(
        self, adv_r: th.Tensor, adv_c: th.Tensor
    ) -> th.Tensor:
        lam = self.lagrange.lagrangian_multiplier.to(adv_r.device)
        return (adv_r - lam * adv_c) / (1.0 + lam)

    # ------------------------------------------------------------------
    # Save / load: persist λ state alongside SB3's model zip
    # ------------------------------------------------------------------
    def save(self, path: str, **kwargs: Any) -> None:  # type: ignore[override]
        super().save(path, **kwargs)
        th.save(
            self.lagrange.state_dict_extra(),
            str(path).replace(".zip", "") + "_lagrange.pt",
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "PPOLag":  # type: ignore[override]
        model: PPOLag = super().load(path, **kwargs)  # type: ignore[assignment]
        lagrange_path = str(path).replace(".zip", "") + "_lagrange.pt"
        import os as _os

        if _os.path.exists(lagrange_path):
            state = th.load(lagrange_path, map_location="cpu")
            model.lagrange.load_state_dict_extra(state)
        return model


# ------------------------------------------------------------------ CLI
def _build_cost_fns(args):
    from sentinel.rl.costs import ConstantCost, ZeroCost

    if args.cost_source == "constant":
        return [ConstantCost(1.0)]
    return [ZeroCost()]


def parse_args():
    p = argparse.ArgumentParser(
        description="PPO-Lag: Lagrangian constrained PPO on PickAndLiftPrivilegedTask."
    )
    add_env_args(p)
    add_training_args(p)
    add_wandb_args(p)
    add_video_args(p)
    add_lagrange_args(p)
    add_cost_args(p)
    p.add_argument(
        "--grasping-mode",
        choices=["physical", "assisted", "sticky"],
        default="physical",
    )
    return p.parse_args()


def main():
    start_time = time.time()

    from sentinel.rl.algorithms.ppo_proprio import (
        _KitLogTailer,
        _log_stage,
        _make_proprio_scene_copy,
    )

    _log_stage("parsing CLI and validating env args")
    args = parse_args()
    validate_env_args(args)
    args.task_type = "PickAndLiftPrivilegedTask"

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _log_stage("creating proprio scene override")
    source_scene_file = Path(args.scene_file).resolve()
    proprio_scene_file, patched_robots = _make_proprio_scene_copy(
        source_scene_file, out, grasping_mode=args.grasping_mode
    )
    args.scene_file = proprio_scene_file
    print(f"  source scene:      {source_scene_file}", flush=True)
    print(f"  proprio scene:     {proprio_scene_file}", flush=True)
    print(f"  task type:         {args.task_type}", flush=True)
    print(f"  cost source:       {args.cost_source}", flush=True)
    print(f"  cost limit:        {args.cost_limit}", flush=True)
    print(f"  lambda init:       {args.lambda_init}", flush=True)
    print(f"  lambda lr:         {args.lambda_lr}", flush=True)

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

    _log_stage("importing PPO-Lag / SB3 helpers")
    from stable_baselines3.common.utils import set_random_seed

    from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper
    from sentinel.rl.envs.wrappers import build_vec_env
    from sentinel.rl.training.trainer import run_training

    _log_stage(f"building OmniGibson vec env (num_envs={args.num_envs})")
    vec_env = build_vec_env(args, out_dir=out)
    cost_fns = _build_cost_fns(args)
    vec_env = CostInjectingVecEnvWrapper(vec_env, cost_fns)
    _log_stage("vec env + cost wrapper ready")
    set_random_seed(args.seed)

    tensorboard_log_dir = out / f"tb_{time.strftime('%Y%m%d-%H%M%S')}"

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"--resume-from does not exist: {args.resume_from}")
        _log_stage("loading PPO-Lag checkpoint")
        model = PPOLag.load(
            str(args.resume_from),
            env=vec_env,
            tensorboard_log=str(tensorboard_log_dir),
            device="cuda",
        )
        print(f"  timesteps at ckpt: {model.num_timesteps:,}", flush=True)
    else:
        _log_stage("constructing fresh PPO-Lag model")
        model = PPOLag(
            env=vec_env,
            verbose=1,
            tensorboard_log=str(tensorboard_log_dir),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            cost_limit=args.cost_limit,
            cost_gamma=args.cost_gamma,
            cost_gae_lambda=args.cost_gae_lambda,
            vf_cost_coef=args.vf_cost_coef,
            lagrangian_multiplier_init=args.lambda_init,
            lambda_lr=args.lambda_lr,
            lambda_optimizer=args.lambda_optimizer,
            lagrangian_upper_bound=args.lambda_upper_bound,
            device="cuda",
            seed=args.seed,
        )

    try:
        _log_stage("entering shared training loop")
        run_training(
            model,
            args,
            algo_name="ppo_lag",
            extra_wandb_config={
                "obs_mode": "proprio_target_goal_state",
                "task_type": args.task_type,
                "grasping_mode": args.grasping_mode,
                "cost_source": args.cost_source,
                "cost_limit": args.cost_limit,
                "lambda_init": args.lambda_init,
                "lambda_lr": args.lambda_lr,
                "lambda_optimizer": args.lambda_optimizer,
                "lambda_upper_bound": args.lambda_upper_bound,
                "cost_gamma": args.cost_gamma,
                "cost_gae_lambda": args.cost_gae_lambda,
                "vf_cost_coef": args.vf_cost_coef,
                "source_scene_file": str(source_scene_file),
            },
        )
    finally:
        kit_tailer.stop()


if __name__ == "__main__":
    main()
