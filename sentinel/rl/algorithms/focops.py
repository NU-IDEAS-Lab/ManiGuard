#!/usr/bin/env python3
"""FOCOPS: First-Order Constrained Optimization in Policy Space.

Algorithm design ported from omnisafe's FOCOPS
(omnisafe/algorithms/on_policy/first_order/focops.py, Apache-2.0).

The actor loss replaces the clipped PPO surrogate with a KL-projected objective:

    L = E_s,a[ (KL(π_θ(·|s) || π_θ_old(·|s)) - (1/ζ) × ratio × adv)
               × 1[KL.detach() ≤ η] ] - ent_coef × H(π_θ)

where:
  η  (``focops_eta``)  — per-update trust-region size (KL hard constraint)
  ζ  (``focops_lam``)  — Lagrangian-style scale on the advantage (inner loop)
  adv = (adv_r − λ·adv_c) / (1+λ)  — same Lagrangian combination as PPO-Lag

The outer Lagrange multiplier λ is updated identically to PPO-Lag: after each
rollout based on ep_cost_mean vs cost_limit.

Per-sample KL requires the old-policy distribution parameters (μ, σ) for each
step. We store them in ``FOCOPSDictRolloutBuffer`` during rollout collection
and reconstruct a ``torch.distributions.Normal`` at training time.

Entry point mirrors ppo_lag.py / ppo_proprio_goal.py.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.distributions as td
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from sentinel.rl.buffers.focops_rollout_buffer import (
    FOCOPSDictRolloutBuffer,
    FOCOPSDictRolloutBufferSamples,
)
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


class FOCOPS(ConstrainedPPO):
    """First-Order Constrained Optimization in Policy Space.

    Inherits the two-critic + cost-stream machinery from ``ConstrainedPPO``
    and overrides:

    * ``_setup_model``        — swap the rollout buffer for ``FOCOPSDictRolloutBuffer``
    * ``collect_rollouts``    — also stores (mu_old, log_std_old) per step
    * ``_before_update``      — Lagrange update, same as PPO-Lag
    * ``_compute_adv_surrogate`` — same as PPO-Lag (λ-weighted)
    * ``_compute_policy_loss``— KL-projected FOCOPS loss instead of clipped PPO

    All SB3 ``.learn()`` / ``.save()`` / ``.load()`` contracts are
    preserved so ``run_training`` works without modification.
    """

    def __init__(
        self,
        *args: Any,
        focops_eta: float = 0.02,
        focops_lam: float = 1.5,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.035,
        lambda_optimizer: str = "Adam",
        lagrangian_upper_bound: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        self.focops_eta = float(focops_eta)
        self.focops_lam = float(focops_lam)
        # Set before super().__init__ so _setup_model (called inside super)
        # can read them when we override it.
        self._focops_lagrange_init = lagrangian_multiplier_init
        self._focops_lambda_lr = lambda_lr
        self._focops_lambda_optimizer = lambda_optimizer
        self._focops_lambda_upper_bound = lagrangian_upper_bound
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # Replace the buffer with the FOCOPS variant that stores dist params.
        self.rollout_buffer = FOCOPSDictRolloutBuffer(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
        )
        self.lagrange = Lagrange(
            cost_limit=self.cost_limit,
            lagrangian_multiplier_init=self._focops_lagrange_init,
            lambda_lr=self._focops_lambda_lr,
            lambda_optimizer=self._focops_lambda_optimizer,
            lagrangian_upper_bound=self._focops_lambda_upper_bound,
        )

    # ------------------------------------------------------------------
    # Rollout: store old distribution params per step
    # ------------------------------------------------------------------
    def collect_rollouts(  # type: ignore[override]
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: FOCOPSDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Extends ConstrainedPPO.collect_rollouts to store mu/log_std."""
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, values_cost, log_probs = self.policy(obs_tensor)
                # _last_distribution is set as a side effect of policy.forward
                mu_old, log_std_old = _extract_dist_params(self.policy)

            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            cost_rewards = np.array(
                [float(info.get("cost", 0.0)) for info in infos],
                dtype=np.float32,
            )

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_value_c = self.policy.predict_values_cost(terminal_obs)[0]
                    rewards[idx] += self.gamma * float(terminal_value)
                    cost_rewards[idx] += rollout_buffer.cost_gamma * float(terminal_value_c)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                cost_reward=cost_rewards,
                cost_value=values_cost,
                mu_old=mu_old,
                log_std_old=log_std_old,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(new_obs_tensor)
            values_cost = self.policy.predict_values_cost(new_obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones, last_cost_values=values_cost
        )

        ep_cost_mean = float(rollout_buffer.cost_rewards.sum(axis=0).mean())
        ep_cost_max = float(rollout_buffer.cost_rewards.sum(axis=0).max())
        self.logger.record("rollout/ep_cost_mean", ep_cost_mean)
        self.logger.record("rollout/ep_cost_max", ep_cost_max)
        self._last_ep_cost_mean = ep_cost_mean

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Training hooks
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

    def _compute_policy_loss(
        self,
        rollout_data: FOCOPSDictRolloutBufferSamples,
        log_prob: th.Tensor,
        advantages: th.Tensor,
        clip_range: float,
    ) -> tuple[th.Tensor, float]:
        """KL-projected FOCOPS actor loss.

        Loss = E[(KL(π_θ || π_old) - (1/ζ) × ratio × adv) × I[KL ≤ η]]
        """
        # Current distribution set as side-effect of evaluate_actions
        curr_dist_raw = getattr(self.policy, "_last_distribution", None)
        if curr_dist_raw is None or not hasattr(curr_dist_raw, "distribution"):
            # Fallback: PPO clipped surrogate (e.g., Discrete action space)
            return super()._compute_policy_loss(
                rollout_data, log_prob, advantages, clip_range
            )

        curr_dist: td.Normal = curr_dist_raw.distribution
        old_dist = td.Normal(
            loc=rollout_data.mu_old,
            scale=rollout_data.log_std_old.exp(),
        )
        # KL(π_θ || π_θ_old), summed over action dimensions
        kl = td.kl_divergence(curr_dist, old_dist).sum(dim=-1)

        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        loss = (kl - (1.0 / self.focops_lam) * ratio * advantages) * (
            kl.detach() <= self.focops_eta
        )
        policy_loss = loss.mean()
        # clip_fraction is meaningless for FOCOPS but we keep the logging slot
        return policy_loss, 0.0

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str, **kwargs: Any) -> None:  # type: ignore[override]
        super().save(path, **kwargs)
        th.save(
            self.lagrange.state_dict_extra(),
            str(path).replace(".zip", "") + "_lagrange.pt",
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "FOCOPS":  # type: ignore[override]
        model: FOCOPS = super().load(path, **kwargs)  # type: ignore[assignment]
        lagrange_path = str(path).replace(".zip", "") + "_lagrange.pt"
        import os as _os

        if _os.path.exists(lagrange_path):
            state = th.load(lagrange_path, map_location="cpu")
            model.lagrange.load_state_dict_extra(state)
        return model


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _extract_dist_params(
    policy,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract (mu_old, log_std_old) from policy._last_distribution.

    Returns (None, None) if the distribution is not a Gaussian or hasn't
    been set yet (e.g., at the very first step).
    """
    dist = getattr(policy, "_last_distribution", None)
    if dist is None or not hasattr(dist, "distribution"):
        return None, None
    raw: td.Normal = dist.distribution
    if not isinstance(raw, td.Normal):
        return None, None
    with th.no_grad():
        mu = raw.loc.cpu().numpy()
        log_std = raw.scale.log().cpu().numpy()
    return mu, log_std


# ------------------------------------------------------------------ CLI
def _build_cost_fns(args):
    from sentinel.rl.costs import ConstantCost, ZeroCost

    if args.cost_source == "constant":
        return [ConstantCost(1.0)]
    return [ZeroCost()]


def add_focops_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("focops")
    g.add_argument("--focops-eta", type=float, default=0.02,
                   help="Per-update KL trust-region size η.")
    g.add_argument("--focops-lam", type=float, default=1.5,
                   help="Inner-loop Lagrangian scale ζ on the advantage.")


def parse_args():
    p = argparse.ArgumentParser(
        description="FOCOPS: First-Order Constrained Optimization in Policy Space."
    )
    add_env_args(p)
    add_training_args(p)
    add_wandb_args(p)
    add_video_args(p)
    add_lagrange_args(p)
    add_cost_args(p)
    add_focops_args(p)
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
    print(f"  task type:         {args.task_type}", flush=True)
    print(f"  cost source:       {args.cost_source}", flush=True)
    print(f"  cost limit:        {args.cost_limit}", flush=True)
    print(f"  focops_eta:        {args.focops_eta}", flush=True)
    print(f"  focops_lam:        {args.focops_lam}", flush=True)

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

    _log_stage("importing FOCOPS / SB3 helpers")
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
        _log_stage("loading FOCOPS checkpoint")
        model = FOCOPS.load(
            str(args.resume_from),
            env=vec_env,
            tensorboard_log=str(tensorboard_log_dir),
            device="cuda",
        )
    else:
        _log_stage("constructing fresh FOCOPS model")
        model = FOCOPS(
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
            focops_eta=args.focops_eta,
            focops_lam=args.focops_lam,
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
            algo_name="focops",
            extra_wandb_config={
                "obs_mode": "proprio_target_goal_state",
                "task_type": args.task_type,
                "grasping_mode": args.grasping_mode,
                "cost_source": args.cost_source,
                "cost_limit": args.cost_limit,
                "focops_eta": args.focops_eta,
                "focops_lam": args.focops_lam,
                "lambda_init": args.lambda_init,
                "lambda_lr": args.lambda_lr,
                "source_scene_file": str(source_scene_file),
            },
        )
    finally:
        kit_tailer.stop()


if __name__ == "__main__":
    main()
