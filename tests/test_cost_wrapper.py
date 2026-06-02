"""Unit tests for the cost vec-env wrapper + cost-function base.

These tests deliberately avoid OmniGibson — a fake ``VecEnv`` lets us verify the
``CostInjectingVecEnvWrapper`` contract (per-env isolation, episode reset,
sim-fault handling) deterministically in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sentinel.rl.costs import BaseCostFunction, ConstantCost, ZeroCost
from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper


# ---------------------------------------------------------- test fixtures
class _FakeVecEnv(VecEnv):
    """Minimal VecEnv that replays a queue of (obs, rew, done, info) tuples."""

    def __init__(self, num_envs: int = 2, action_dim: int = 1):
        obs_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        super().__init__(num_envs=num_envs, observation_space=obs_space, action_space=act_space)
        self._next_step_returns: list = []
        self._pending_actions: np.ndarray | None = None
        self.received_actions: list[np.ndarray] = []

    def push(self, rewards, dones, infos):
        self._next_step_returns.append((rewards, dones, infos))

    def reset(self):
        return np.zeros((self.num_envs, 2), dtype=np.float32)

    def step_async(self, actions):
        self._pending_actions = np.asarray(actions)
        self.received_actions.append(self._pending_actions.copy())

    def step_wait(self):
        rewards, dones, infos = self._next_step_returns.pop(0)
        obs = np.zeros((self.num_envs, 2), dtype=np.float32)
        return obs, np.asarray(rewards, dtype=np.float64), np.asarray(dones, dtype=bool), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def has_attr(self, attr_name):
        return False

    def seed(self, seed=None):
        return [None] * self.num_envs

    def get_images(self):
        return [None] * self.num_envs


class _StepCountingCost(BaseCostFunction):
    """Per-instance step counter — proves per-env state isolation.

    Overrides ``reset`` to clear the per-episode counter so we can also verify
    the wrapper calls ``reset`` on episode boundaries / sim-fault recovery.
    """

    def __init__(self):
        super().__init__()
        self.steps = 0

    def _step(self, task, env, action):
        self.steps += 1
        return float(self.steps), {"count": self.steps}

    def reset(self, task, env):
        super().reset(task, env)
        self.steps = 0


# ----------------------------------------------------------------- tests
def test_zero_cost_writes_zero():
    venv = _FakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(venv, [ZeroCost()])
    wrapped.reset()
    venv.push(rewards=[1.0, 1.0], dones=[False, False], infos=[{}, {}])
    wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
    _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == 0.0
    assert infos[1]["cost"] == 0.0
    assert infos[0]["cost_breakdown"]["ZeroCost"] == 0.0


def test_constant_cost_sums_per_step():
    venv = _FakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(venv, [ConstantCost(0.7), ConstantCost(0.3)])
    wrapped.reset()
    venv.push(rewards=[0, 0], dones=[False, False], infos=[{}, {}])
    wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
    _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == pytest.approx(1.0)
    assert infos[1]["cost"] == pytest.approx(1.0)
    assert set(infos[0]["cost_breakdown"]) >= {"ConstantCost"}


def test_per_env_isolation():
    venv = _FakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(venv, [_StepCountingCost()])
    wrapped.reset()
    # 3 steps without dones — each env's counter increments independently
    for _ in range(3):
        venv.push(rewards=[0, 0], dones=[False, False], infos=[{}, {}])
        wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
        _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == 3.0
    assert infos[1]["cost"] == 3.0
    # End env 0 only; env 1 should keep counting
    venv.push(rewards=[0, 0], dones=[True, False], infos=[{}, {}])
    wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
    _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == 4.0  # 4th step still counted, then reset for next
    assert infos[1]["cost"] == 4.0
    # Next step: env 0 starts fresh (1), env 1 continues (5)
    venv.push(rewards=[0, 0], dones=[False, False], infos=[{}, {}])
    wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
    _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == 1.0
    assert infos[1]["cost"] == 5.0


def test_sim_fault_zeroes_cost_and_resets_state():
    venv = _FakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(venv, [_StepCountingCost()])
    wrapped.reset()
    # Step env 1 a few times so its counter is nonzero
    for _ in range(2):
        venv.push(rewards=[0, 0], dones=[False, False], infos=[{}, {}])
        wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
        wrapped.step_wait()
    # Now sim-fault recover env 0; env 1 keeps stepping
    venv.push(
        rewards=[0, 0],
        dones=[True, False],
        infos=[{"sim_fault_recovered": True, "sim_step_skipped": True}, {}],
    )
    wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
    _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == 0.0
    assert infos[0]["cost_breakdown"] == {}
    assert infos[1]["cost"] == 3.0
    # After fault recovery env 0 starts from a clean cost-fn state
    venv.push(rewards=[0, 0], dones=[False, False], infos=[{}, {}])
    wrapped.step_async(np.zeros((2, 1), dtype=np.float32))
    _, _, _, infos = wrapped.step_wait()
    assert infos[0]["cost"] == 1.0
    assert infos[1]["cost"] == 4.0


def test_rejects_empty_cost_list():
    venv = _FakeVecEnv(num_envs=1)
    with pytest.raises(ValueError, match="at least one cost function"):
        CostInjectingVecEnvWrapper(venv, [])


def test_rejects_non_base_cost_function():
    venv = _FakeVecEnv(num_envs=1)
    with pytest.raises(TypeError):
        CostInjectingVecEnvWrapper(venv, [object()])
