"""Dict-obs rollout buffer that carries both reward and cost streams.

Extends SB3's ``DictRolloutBuffer`` with a parallel ``cost_*`` set of arrays
and GAE pass, matching omnisafe's ``VectorOnPolicyBuffer``. The cost stream
uses its own ``(gamma_c, lam_c)`` per the on-policy safe-RL convention
(omnisafe defaults to the same values as the reward stream — 0.99 / 0.95 —
but keeps them separate so a future user can decouple).

The buffer is a drop-in replacement when constructed via
``OnPolicyAlgorithm.__init__(rollout_buffer_class=...,
rollout_buffer_kwargs={"cost_gamma": ..., "cost_gae_lambda": ...})``: SB3
forwards everything through to ``__init__``. The constrained collect_rollouts
and train loops in ``sentinel.rl.safe.constrained_ppo`` know to call the
extended ``add`` / ``compute_returns_and_advantage`` signatures.
"""

from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict


class ConstrainedDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_cost_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    cost_advantages: th.Tensor
    returns: th.Tensor
    cost_returns: th.Tensor


class ConstrainedDictRolloutBuffer(DictRolloutBuffer):
    """``DictRolloutBuffer`` + per-step cost stream and per-step cost-value.

    Storage layout (each ``(buffer_size, n_envs)``):

    * ``rewards / values / advantages / returns`` — inherited reward stream.
    * ``cost_rewards / cost_values / cost_advantages / cost_returns`` — added
      here.

    ``compute_returns_and_advantage`` now also accepts ``last_cost_values`` and
    runs the same GAE recursion on the cost stream with ``cost_gamma`` /
    ``cost_gae_lambda``.
    """

    cost_rewards: np.ndarray
    cost_values: np.ndarray
    cost_returns: np.ndarray
    cost_advantages: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
    ):
        self.cost_gamma = float(cost_gamma)
        self.cost_gae_lambda = float(cost_gae_lambda)
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs)
        self.cost_rewards = np.zeros(shape, dtype=np.float32)
        self.cost_values = np.zeros(shape, dtype=np.float32)
        self.cost_returns = np.zeros(shape, dtype=np.float32)
        self.cost_advantages = np.zeros(shape, dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        cost_reward: np.ndarray | None = None,
        cost_value: th.Tensor | None = None,
    ) -> None:
        if cost_reward is None or cost_value is None:
            raise ValueError(
                "ConstrainedDictRolloutBuffer.add requires both cost_reward "
                "and cost_value; pass them from collect_rollouts."
            )
        # pos is incremented by super().add — capture before it moves.
        write_idx = self.pos
        self.cost_rewards[write_idx] = np.asarray(cost_reward, dtype=np.float32)
        self.cost_values[write_idx] = cost_value.clone().cpu().numpy().flatten()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_returns_and_advantage(  # type: ignore[override]
        self,
        last_values: th.Tensor,
        dones: np.ndarray,
        last_cost_values: th.Tensor | None = None,
    ) -> None:
        """Run GAE on both reward and cost streams."""
        super().compute_returns_and_advantage(last_values=last_values, dones=dones)

        if last_cost_values is None:
            raise ValueError(
                "compute_returns_and_advantage requires last_cost_values; "
                "ConstrainedPPO.collect_rollouts is responsible for passing it."
            )
        last_cost = last_cost_values.clone().cpu().numpy().flatten()
        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_cost
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.cost_values[step + 1]
            delta = (
                self.cost_rewards[step]
                + self.cost_gamma * next_values * next_non_terminal
                - self.cost_values[step]
            )
            last_gae = (
                delta
                + self.cost_gamma * self.cost_gae_lambda * next_non_terminal * last_gae
            )
            self.cost_advantages[step] = last_gae
        self.cost_returns = self.cost_advantages + self.cost_values

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[ConstrainedDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for tensor in (
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "cost_values",
                "cost_advantages",
                "cost_returns",
            ):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env=None,
    ) -> ConstrainedDictRolloutBufferSamples:
        return ConstrainedDictRolloutBufferSamples(
            observations={
                key: self.to_torch(obs[batch_inds])
                for key, obs in self.observations.items()
            },
            actions=self.to_torch(
                self.actions[batch_inds].astype(np.float32, copy=False)
            ),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_cost_values=self.to_torch(self.cost_values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            cost_advantages=self.to_torch(self.cost_advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            cost_returns=self.to_torch(self.cost_returns[batch_inds].flatten()),
        )


__all__ = [
    "ConstrainedDictRolloutBuffer",
    "ConstrainedDictRolloutBufferSamples",
]
