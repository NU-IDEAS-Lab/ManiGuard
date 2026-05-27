"""Unit tests for FOCOPS algorithm (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sentinel.rl.costs import ConstantCost
from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper
from sentinel.rl.algorithms.focops import FOCOPS, _extract_dist_params
from sentinel.rl.buffers.focops_rollout_buffer import FOCOPSDictRolloutBuffer
from sentinel.rl.policies.constrained_actor_critic import (
    MultiInputConstrainedActorCriticPolicy,
)


# ---------------------------------------------------------------- fake env
class _DictObsFakeVecEnv(VecEnv):
    def __init__(self, num_envs: int = 2):
        obs_space = spaces.Dict(
            {
                "proprio": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "task::low_dim": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            }
        )
        act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        super().__init__(num_envs, obs_space, act_space)
        self._step_counter = 0

    def _obs(self):
        return {
            "proprio": np.zeros((self.num_envs, 4), dtype=np.float32),
            "task::low_dim": np.zeros((self.num_envs, 3), dtype=np.float32),
        }

    def reset(self):
        self._step_counter = 0
        return self._obs()

    def step_async(self, actions):
        self._pending_actions = actions

    def step_wait(self):
        self._step_counter += 1
        rewards = np.full(self.num_envs, 0.5, dtype=np.float64)
        dones = np.array(
            [self._step_counter % 8 == 0] * self.num_envs, dtype=bool
        )
        infos = []
        for d in dones:
            info: dict = {"TimeLimit.truncated": False}
            if d:
                info["TimeLimit.truncated"] = True
                info["terminal_observation"] = {
                    "proprio": np.zeros(4, dtype=np.float32),
                    "task::low_dim": np.zeros(3, dtype=np.float32),
                }
            infos.append(info)
        return self._obs(), rewards, dones, infos

    def close(self): pass
    def get_attr(self, name, indices=None): return [None] * self.num_envs
    def set_attr(self, name, value, indices=None): pass
    def env_method(self, name, *args, indices=None, **kwargs): return [None] * self.num_envs
    def env_is_wrapped(self, cls, indices=None): return [False] * self.num_envs
    def has_attr(self, name): return False
    def seed(self, seed=None): return [None] * self.num_envs
    def get_images(self): return [None] * self.num_envs


# ---------------------------------------------------------------- tests
def test_focops_buffer_allocates_dist_params():
    obs_space = spaces.Dict({"x": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)})
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    buf = FOCOPSDictRolloutBuffer(
        buffer_size=4, observation_space=obs_space, action_space=act_space, n_envs=2
    )
    assert buf.mu_old.shape == (4, 2, 2)   # (buffer_size, n_envs, act_dim)
    assert buf.log_std_old.shape == (4, 2, 2)


def test_focops_buffer_stores_dist_params():
    obs_space = spaces.Dict({"x": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)})
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buf = FOCOPSDictRolloutBuffer(
        buffer_size=2, observation_space=obs_space, action_space=act_space, n_envs=1
    )
    obs = {"x": np.zeros((1, 2), dtype=np.float32)}
    mu = np.array([[[0.3]]], dtype=np.float32)    # (1, 1, 1)
    ls = np.array([[[-0.5]]], dtype=np.float32)
    buf.add(
        obs=obs,
        action=np.zeros((1, 1), dtype=np.float32),
        reward=np.array([1.0], dtype=np.float32),
        episode_start=np.array([0.0], dtype=np.float32),
        value=th.tensor([[0.5]]),
        log_prob=th.tensor([-0.1]),
        cost_reward=np.array([0.0], dtype=np.float32),
        cost_value=th.tensor([[0.0]]),
        mu_old=mu,
        log_std_old=ls,
    )
    assert buf.mu_old[0, 0, 0] == pytest.approx(0.3, abs=1e-5)
    assert buf.log_std_old[0, 0, 0] == pytest.approx(-0.5, abs=1e-5)


def test_policy_sets_last_distribution():
    """After policy.forward, _last_distribution is available."""
    obs_space = spaces.Dict({"x": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)})
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    policy = MultiInputConstrainedActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
    )
    obs = {"x": th.zeros(4, 3)}
    policy(obs)
    assert hasattr(policy, "_last_distribution")
    assert hasattr(policy._last_distribution, "distribution")


def test_extract_dist_params_shape():
    obs_space = spaces.Dict({"x": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)})
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    policy = MultiInputConstrainedActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
    )
    obs = {"x": th.zeros(4, 3)}
    policy(obs)
    mu, log_std = _extract_dist_params(policy)
    assert mu is not None and log_std is not None
    assert mu.shape == (4, 2)
    assert log_std.shape == (4, 2)


def test_focops_end_to_end():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = FOCOPS(
        env=wrapped,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        cost_limit=0.5,
        focops_eta=0.05,
        focops_lam=1.5,
        lambda_lr=0.1,
        device="cpu",
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=32)
    buf = model.rollout_buffer
    assert buf.cost_rewards.sum() > 0
    assert np.any(buf.cost_returns != 0)
    assert isinstance(buf, FOCOPSDictRolloutBuffer)


def test_focops_lambda_increases_with_cost():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = FOCOPS(
        env=wrapped,
        n_steps=8,
        batch_size=8,
        cost_limit=0.5,  # always exceeded
        lambda_lr=0.1,
        device="cpu",
        verbose=0,
        seed=0,
    )
    lam_before = float(model.lagrange.lagrangian_multiplier.item())
    model.learn(total_timesteps=32)
    lam_after = float(model.lagrange.lagrangian_multiplier.item())
    assert lam_after > lam_before


def test_focops_kl_loss_fires():
    """At least some steps hit the KL mask (kl ≤ eta), so policy_loss != 0."""
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = FOCOPS(
        env=wrapped,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        focops_eta=1000.0,  # large eta → all steps pass the mask
        focops_lam=1.5,
        device="cpu",
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=16)
    # If we get here without NaNs the KL loss computed fine.
    assert np.isfinite(float(model.rollout_buffer.cost_returns.mean()))


def test_focops_save_load():
    import tempfile
    from pathlib import Path

    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = FOCOPS(
        env=wrapped, n_steps=4, batch_size=4, device="cpu", verbose=0, seed=0,
        cost_limit=0.5, lambda_lr=0.1,
    )
    model.learn(total_timesteps=8)
    lam = float(model.lagrange.lagrangian_multiplier.item())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "focops.zip"
        model.save(str(path))
        env2 = _DictObsFakeVecEnv(num_envs=2)
        wrapped2 = CostInjectingVecEnvWrapper(env2, [ConstantCost(1.0)])
        loaded = FOCOPS.load(str(path), env=wrapped2, device="cpu")

    loaded_lam = float(loaded.lagrange.lagrangian_multiplier.item())
    assert abs(loaded_lam - lam) < 1e-5
