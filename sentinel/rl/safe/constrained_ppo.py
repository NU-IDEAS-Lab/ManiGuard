"""Constrained PPO base — shared by PPO-Lag and FOCOPS.

What it adds on top of SB3's ``PPO``:

* uses :class:`MultiInputConstrainedActorCriticPolicy` (two value heads) and
  :class:`ConstrainedDictRolloutBuffer` (parallel cost stream + GAE);
* during rollouts, reads ``info["cost"]`` per env per step, bootstraps the
  cost-value on truncation symmetrically with the reward-value bootstrap, and
  passes both reward and cost to the buffer;
* during training, computes a cost-critic MSE loss and adds it to the total
  loss with ``vf_cost_coef``;
* exposes a ``_compute_adv_surrogate(adv_r, adv_c) -> adv`` hook that
  subclasses MUST override — this is the single line that distinguishes
  PPO-Lag (weighted by Lagrange λ) from FOCOPS (KL-projected) from a
  hypothetical constraint-free baseline.

What it does NOT do: change the optimizer / clipping / target_kl / batching
machinery of SB3 PPO. We only fork what we have to.

Compatible with the existing ``sentinel.rl.training.trainer.run_training``
because we keep SB3's ``.learn / .save / .load`` contract intact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sentinel.rl.buffers.constrained_rollout_buffer import (
    ConstrainedDictRolloutBuffer,
)
from sentinel.rl.policies.constrained_actor_critic import (
    MultiInputConstrainedActorCriticPolicy,
)


class ConstrainedPPO(PPO):
    """Two-critic PPO base. Subclasses must override ``_compute_adv_surrogate``."""

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy] = MultiInputConstrainedActorCriticPolicy,
        env: VecEnv | str | None = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        vf_cost_coef: float = 0.5,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        cost_limit: float = 25.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        self.vf_cost_coef = float(vf_cost_coef)
        self.cost_limit = float(cost_limit)
        self.cost_gamma = float(cost_gamma)
        self.cost_gae_lambda = float(cost_gae_lambda)
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=ConstrainedDictRolloutBuffer,
            rollout_buffer_kwargs={
                "cost_gamma": cost_gamma,
                "cost_gae_lambda": cost_gae_lambda,
            },
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    # --- hooks for subclasses --------------------------------------------
    def _compute_adv_surrogate(
        self, adv_r: th.Tensor, adv_c: th.Tensor
    ) -> th.Tensor:
        """Weight reward and cost advantages into a single surrogate advantage.

        PPO-Lag: ``(adv_r - λ adv_c) / (1 + λ)``
        FOCOPS:  same form, but used inside a KL-projected loss
        """
        raise NotImplementedError(
            "ConstrainedPPO subclasses must override _compute_adv_surrogate "
            "(see PPOLag / FOCOPS)."
        )

    def _compute_policy_loss(
        self,
        rollout_data: Any,
        log_prob: th.Tensor,
        advantages: th.Tensor,
        clip_range: float,
    ) -> tuple[th.Tensor, float]:
        """Clipped-surrogate policy loss for standard PPO.

        Returns (policy_loss, clip_fraction). Overridden by FOCOPS to use the
        KL-projected loss instead of the clipped surrogate.
        """
        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(
            ratio, 1 - clip_range, 1 + clip_range
        )
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        return policy_loss, clip_fraction

    def _before_update(self) -> None:
        """Hook called inside ``train()`` before the SB3 epoch loop runs.

        PPO-Lag / FOCOPS use this to update the Lagrange multiplier based on
        the just-collected rollout's mean cost. Default no-op.
        """
        return

    # --- rollout ---------------------------------------------------------
    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: ConstrainedDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """SB3.collect_rollouts + cost stream + cost-value bootstrap.

        Mirrors ``OnPolicyAlgorithm.collect_rollouts`` line-by-line and adds:
          * unpack 4-tuple from policy.forward (values_c too);
          * read ``info["cost"]`` per env per step (default 0.0 if absent —
            though ``CostInjectingVecEnvWrapper`` always provides it);
          * cost-value bootstrap on truncation symmetrically with reward;
          * compute_returns_and_advantage with both streams.
        """
        assert self._last_obs is not None, "No previous observation was provided"
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

            # Per-env cost from info[]. CostInjectingVecEnvWrapper always
            # populates it; default 0.0 for safety if some env path doesn't.
            cost_rewards = np.array(
                [float(info.get("cost", 0.0)) for info in infos],
                dtype=np.float32,
            )

            # Bootstrap both value heads on truncation (mirror SB3's reward
            # bootstrap at https://github.com/DLR-RM/stable-baselines3/issues/633).
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
                        terminal_value_c = self.policy.predict_values_cost(
                            terminal_obs
                        )[0]
                    rewards[idx] += self.gamma * float(terminal_value)
                    cost_rewards[idx] += rollout_buffer.cost_gamma * float(
                        terminal_value_c
                    )

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                cost_reward=cost_rewards,
                cost_value=values_cost,
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

        # Cost rollout statistics, logged per rollout (not per env step) so the
        # subclass _before_update can read 'rollout/ep_cost_mean' from
        # self.logger / from the rollout_buffer directly.
        ep_cost_mean = float(rollout_buffer.cost_rewards.sum(axis=0).mean())
        ep_cost_max = float(rollout_buffer.cost_rewards.sum(axis=0).max())
        self.logger.record("rollout/ep_cost_mean", ep_cost_mean)
        self.logger.record("rollout/ep_cost_max", ep_cost_max)
        self._last_ep_cost_mean = ep_cost_mean  # cached for _before_update

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # --- training --------------------------------------------------------
    def train(self) -> None:  # noqa: C901 — mirrors SB3 structure
        """SB3 PPO.train + cost-critic loss + surrogate advantage hook."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._before_update()

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        cost_value_losses: list[float] = []
        clip_fractions: list[float] = []

        continue_training = True

        for _epoch in range(self.n_epochs):
            approx_kl_divs: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, values_cost, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                values_cost = values_cost.flatten()

                adv_r = rollout_data.advantages
                adv_c = rollout_data.cost_advantages
                if self.normalize_advantage and len(adv_r) > 1:
                    adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
                    # Cost advantage is centred but not std-normalized in
                    # omnisafe; we follow their convention.
                    adv_c = adv_c - adv_c.mean()

                advantages = self._compute_adv_surrogate(adv_r, adv_c)

                policy_loss, clip_fraction = self._compute_policy_loss(
                    rollout_data, log_prob, advantages, clip_range
                )
                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                    values_cost_pred = values_cost
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    values_cost_pred = rollout_data.old_cost_values + th.clamp(
                        values_cost - rollout_data.old_cost_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                cost_value_loss = F.mse_loss(
                    rollout_data.cost_returns, values_cost_pred
                )
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.vf_cost_coef * cost_value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {_epoch} "
                            f"due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        explained_var_cost = explained_variance(
            self.rollout_buffer.cost_values.flatten(),
            self.rollout_buffer.cost_returns.flatten(),
        )

        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record(
            "train/policy_gradient_loss", float(np.mean(pg_losses))
        )
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record(
            "train/cost_value_loss", float(np.mean(cost_value_losses))
        )
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/explained_variance_cost", explained_var_cost)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


__all__ = ["ConstrainedPPO"]
