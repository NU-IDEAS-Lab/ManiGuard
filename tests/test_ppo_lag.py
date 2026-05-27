"""Unit tests for Lagrange multiplier and PPOLag algorithm (Phase 2)."""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sentinel.rl.costs import ConstantCost
from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper
from sentinel.rl.safe.lagrange import Lagrange
from sentinel.rl.algorithms.ppo_lag import PPOLag


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
def test_lagrange_increases_when_cost_exceeds_limit():
    lag = Lagrange(cost_limit=1.0, lambda_lr=0.1)
    lam_before = float(lag.lagrangian_multiplier.item())
    lag.update_lagrange_multiplier(ep_cost=2.0)  # cost > limit
    lam_after = float(lag.lagrangian_multiplier.item())
    assert lam_after > lam_before


def test_lagrange_decreases_when_under_limit():
    lag = Lagrange(cost_limit=10.0, lagrangian_multiplier_init=1.0, lambda_lr=0.1)
    lam_before = float(lag.lagrangian_multiplier.item())
    lag.update_lagrange_multiplier(ep_cost=0.0)  # cost < limit
    lam_after = float(lag.lagrangian_multiplier.item())
    assert lam_after < lam_before


def test_lagrange_clamps_to_zero():
    lag = Lagrange(cost_limit=10.0, lagrangian_multiplier_init=0.0, lambda_lr=1.0)
    for _ in range(5):
        lag.update_lagrange_multiplier(ep_cost=0.0)  # repeatedly push down
    assert float(lag.lagrangian_multiplier.item()) >= 0.0


def test_lagrange_upper_bound():
    lag = Lagrange(cost_limit=0.0, lagrangian_upper_bound=0.5, lambda_lr=10.0)
    for _ in range(20):
        lag.update_lagrange_multiplier(ep_cost=100.0)
    assert float(lag.lagrangian_multiplier.item()) <= 0.5 + 1e-6


def test_lagrange_softplus_stays_positive():
    lag = Lagrange(
        cost_limit=10.0, lagrangian_multiplier_init=0.0,
        lambda_lr=1.0, softplus=True
    )
    for _ in range(10):
        lag.update_lagrange_multiplier(ep_cost=0.0)
    assert float(lag.lagrangian_multiplier.item()) > 0.0


def test_lagrange_state_dict_roundtrip():
    lag = Lagrange(cost_limit=1.0, lambda_lr=0.1)
    lag.update_lagrange_multiplier(ep_cost=5.0)
    original = float(lag.lagrangian_multiplier.item())

    state = lag.state_dict_extra()
    lag2 = Lagrange(cost_limit=1.0, lambda_lr=0.1)
    lag2.load_state_dict_extra(state)
    assert abs(float(lag2.lagrangian_multiplier.item()) - original) < 1e-6


def test_ppo_lag_surrogate_uses_lambda():
    # Make a minimal PPOLag just to test the math (no env needed)
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = PPOLag(
        env=wrapped, n_steps=4, batch_size=4, device="cpu", verbose=0, seed=0,
        lagrangian_multiplier_init=2.0, lambda_lr=0.0,  # λ stays at 2
    )
    adv_r = th.tensor([1.0, 2.0, 3.0])
    adv_c = th.tensor([0.5, 0.5, 0.5])
    adv = model._compute_adv_surrogate(adv_r, adv_c)
    # Expected: (adv_r - 2*adv_c) / 3
    expected = (adv_r - 2.0 * adv_c) / 3.0
    assert th.allclose(adv, expected, atol=1e-5)


def test_ppo_lag_end_to_end():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = PPOLag(
        env=wrapped,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        cost_limit=0.5,  # cost=1.0 always exceeds, so λ should grow
        lambda_lr=0.1,
        device="cpu",
        verbose=0,
        seed=0,
    )
    lam_before = float(model.lagrange.lagrangian_multiplier.item())
    model.learn(total_timesteps=32)
    lam_after = float(model.lagrange.lagrangian_multiplier.item())
    # With ConstantCost(1.0) and cost_limit=0.5, λ must have grown
    assert lam_after > lam_before


def test_ppo_lag_save_load_preserves_lambda():
    import tempfile
    from pathlib import Path

    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = PPOLag(
        env=wrapped, n_steps=4, batch_size=4, device="cpu", verbose=0, seed=0,
        cost_limit=0.5, lambda_lr=0.1,
    )
    model.learn(total_timesteps=8)
    lam = float(model.lagrange.lagrangian_multiplier.item())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ppo_lag.zip"
        model.save(str(path))
        env2 = _DictObsFakeVecEnv(num_envs=2)
        wrapped2 = CostInjectingVecEnvWrapper(env2, [ConstantCost(1.0)])
        loaded = PPOLag.load(str(path), env=wrapped2, device="cpu")

    loaded_lam = float(loaded.lagrange.lagrangian_multiplier.item())
    assert abs(loaded_lam - lam) < 1e-5
