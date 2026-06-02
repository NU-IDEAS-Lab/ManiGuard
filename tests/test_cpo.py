"""Unit tests for CPO algorithm and TRPO utilities (Phase 4)."""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sentinel.rl.costs import ConstantCost
from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper
from sentinel.rl.safe.trpo_utils import (
    conjugate_gradients,
    flat_grad,
    flat_params,
    set_flat_params,
)
from sentinel.rl.algorithms.cpo import CPO, _cpo_step_direction
from sentinel.rl.buffers.focops_rollout_buffer import FOCOPSDictRolloutBuffer


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
        dones = np.array([self._step_counter % 8 == 0] * self.num_envs, dtype=bool)
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


# ---------------------------------------------------------------- trpo_utils tests

def test_flat_params_and_set():
    """flat_params + set_flat_params are exact inverse of each other."""
    net = th.nn.Linear(4, 3)
    params = list(net.parameters())
    flat = flat_params(params)

    # Zero out and restore
    original = flat.clone()
    set_flat_params(params, th.zeros_like(flat))
    assert flat_params(params).norm() == 0.0

    set_flat_params(params, original)
    assert th.allclose(flat_params(params), original)


def test_flat_grad_shape():
    net = th.nn.Linear(4, 3)
    x = th.randn(8, 4)
    loss = net(x).sum()
    g = flat_grad(loss, list(net.parameters()))
    total_params = sum(p.numel() for p in net.parameters())
    assert g.shape == (total_params,)


def test_conjugate_gradients_solves_spd():
    """CG should solve Ax=b exactly for small SPD A."""
    A = th.tensor([[4.0, 1.0], [1.0, 3.0]])  # SPD
    b = th.tensor([1.0, 2.0])
    # True solution: A^-1 b
    true_x = th.linalg.solve(A, b)

    Avp = lambda v: A @ v  # noqa: E731
    x = conjugate_gradients(Avp, b, n_steps=50, tol=1e-12)
    assert th.allclose(x, true_x, atol=1e-4)


def test_cpo_step_direction_feasible_trpo():
    """If c ≤ 0 and TRPO step is cost-safe, case should be feasible_trpo."""
    x_a = th.tensor([1.0, 0.0])
    x_b = th.tensor([0.0, 1.0])
    # p=1, q=1, r=0, s=1, c=-1 (feasible), delta=0.5
    # TRPO step: lam_a = sqrt(1/(2*0.5)) = 1, cost_at_trpo = -1 + 0/1 = -1 ≤ 0
    step, case = _cpo_step_direction(x_a, x_b, p=1.0, q=1.0, r=0.0, s=1.0,
                                     c=-1.0, target_kl=0.5)
    assert case == "feasible_trpo"
    assert step is not None


def test_cpo_step_direction_infeasible():
    """If c > 0 and QP has no solution, case should be feasibility."""
    x_a = th.tensor([1.0, 0.0])
    x_b = th.tensor([0.0, 1.0])
    # c=5, very large violation; s=1, delta=0.01 → 2*delta*s=0.02 < c^2=25
    step, case = _cpo_step_direction(x_a, x_b, p=1.0, q=1.0, r=0.0, s=1.0,
                                     c=5.0, target_kl=0.01)
    assert case == "feasibility"


def test_cpo_end_to_end():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = CPO(
        env=wrapped,
        n_steps=8,
        batch_size=8,
        cost_limit=0.5,
        target_kl_delta=0.05,
        cg_iters=3,          # fewer iterations for speed
        n_value_epochs=2,
        device="cpu",
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=32)
    buf = model.rollout_buffer
    assert isinstance(buf, FOCOPSDictRolloutBuffer)
    assert buf.cost_rewards.sum() > 0
    assert np.any(buf.cost_returns != 0)


def test_cpo_uses_separate_critic_optimizer():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = CPO(
        env=wrapped, n_steps=4, batch_size=4, device="cpu", verbose=0, seed=0
    )
    assert hasattr(model, "_critic_optimizer")
    assert isinstance(model._critic_optimizer, th.optim.Adam)


def test_cpo_save_load():
    import tempfile
    from pathlib import Path

    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = CPO(
        env=wrapped, n_steps=4, batch_size=4, device="cpu", verbose=0, seed=0,
        cost_limit=0.5, cg_iters=3, n_value_epochs=2,
    )
    model.learn(total_timesteps=8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cpo.zip"
        model.save(str(path))
        env2 = _DictObsFakeVecEnv(num_envs=2)
        wrapped2 = CostInjectingVecEnvWrapper(env2, [ConstantCost(1.0)])
        loaded = CPO.load(str(path), env=wrapped2, device="cpu")

    assert hasattr(loaded, "_critic_optimizer")
    assert hasattr(loaded.policy, "cost_value_net")
