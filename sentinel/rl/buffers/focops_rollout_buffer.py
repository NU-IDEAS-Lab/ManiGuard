"""FOCOPS rollout buffer — extends ConstrainedDictRolloutBuffer with
per-step old-distribution parameters (mu_old, log_std_old) for the
KL-projected actor loss.

FOCOPS needs KL(π_θ || π_θ_old) per sample at training time. Because
SB3's buffer doesn't store distribution params, we add two extra arrays:

* ``mu_old``       — shape ``(buffer_size, n_envs, act_dim)`` float32
* ``log_std_old``  — same shape, the log of the policy std at collection time

Only valid for continuous (Box) action spaces. For Discrete actions the
arrays are allocated but unused and FOCOPS falls back to the PPO surrogate.
"""

from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict

from sentinel.rl.buffers.constrained_rollout_buffer import (
    ConstrainedDictRolloutBuffer,
)


class FOCOPSDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_cost_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    cost_advantages: th.Tensor
    returns: th.Tensor
    cost_returns: th.Tensor
    mu_old: th.Tensor       # (batch, act_dim)
    log_std_old: th.Tensor  # (batch, act_dim)


class FOCOPSDictRolloutBuffer(ConstrainedDictRolloutBuffer):
    """``ConstrainedDictRolloutBuffer`` + old-policy distribution params.

    Additional arrays (only for Gaussian / Box action spaces):
    * ``mu_old``       — stored action-distribution mean at collection time
    * ``log_std_old``  — stored log-std at collection time
    """

    mu_old: np.ndarray
    log_std_old: np.ndarray

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
        self._act_dim = (
            int(np.prod(action_space.shape))
            if isinstance(action_space, spaces.Box)
            else 1
        )
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
            cost_gamma=cost_gamma,
            cost_gae_lambda=cost_gae_lambda,
        )

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs, self._act_dim)
        self.mu_old = np.zeros(shape, dtype=np.float32)
        self.log_std_old = np.zeros(shape, dtype=np.float32)

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
        mu_old: np.ndarray | None = None,
        log_std_old: np.ndarray | None = None,
    ) -> None:
        write_idx = self.pos
        if mu_old is not None:
            self.mu_old[write_idx] = np.asarray(mu_old, dtype=np.float32)
        if log_std_old is not None:
            self.log_std_old[write_idx] = np.asarray(log_std_old, dtype=np.float32)
        super().add(obs, action, reward, episode_start, value, log_prob,
                    cost_reward=cost_reward, cost_value=cost_value)

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[FOCOPSDictRolloutBufferSamples, None, None]:
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
                "mu_old",
                "log_std_old",
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
    ) -> FOCOPSDictRolloutBufferSamples:
        return FOCOPSDictRolloutBufferSamples(
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
            mu_old=self.to_torch(self.mu_old[batch_inds]),
            log_std_old=self.to_torch(self.log_std_old[batch_inds]),
        )


__all__ = ["FOCOPSDictRolloutBuffer", "FOCOPSDictRolloutBufferSamples"]
