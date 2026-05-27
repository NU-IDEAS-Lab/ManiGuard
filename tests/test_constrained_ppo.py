"""Unit tests for the constrained PPO building blocks (Phase 1).

We sidestep OmniGibson with a fake dict-obs vec env and verify:
1. ``ConstrainedDictRolloutBuffer`` allocates the cost stream, stores it via
   ``add``, and produces finite cost advantages on
   ``compute_returns_and_advantage``.
2. ``MultiInputConstrainedActorCriticPolicy`` adds a ``cost_value_net`` whose
   parameters end up in the optimizer's param groups (otherwise training
   wouldn't update it).
3. ``ConstrainedPPO`` (with a concrete subclass that uses
   ``adv_r`` only as the surrogate, i.e. unconstrained sanity check) can run
   ``collect_rollouts`` + ``train`` end-to-end on the fake env without raising.
4. The cost stream actually moves: with ``ConstantCost(1.0)`` injected,
   ``cost_returns`` is bounded away from zero after one rollout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sentinel.rl.buffers.constrained_rollout_buffer import (
    ConstrainedDictRolloutBuffer,
)
from sentinel.rl.costs import ConstantCost
from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper
from sentinel.rl.policies.constrained_actor_critic import (
    MultiInputConstrainedActorCriticPolicy,
)
from sentinel.rl.safe.constrained_ppo import ConstrainedPPO


# ---------------------------------------------------------- fixtures
class _DictObsFakeVecEnv(VecEnv):
    """Fake dict-obs vec env yielding constant obs/rewards/dones."""

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
        # Episode ends every 8 steps so we exercise the truncation bootstrap
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

    def close(self):
        pass

    def get_attr(self, name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, name, value, indices=None):
        pass

    def env_method(self, name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, cls, indices=None):
        return [False] * self.num_envs

    def has_attr(self, name):
        return False

    def seed(self, seed=None):
        return [None] * self.num_envs

    def get_images(self):
        return [None] * self.num_envs


class _AdvROnlyPPO(ConstrainedPPO):
    """Concrete ConstrainedPPO that ignores cost — for unconstrained sanity."""

    def _compute_adv_surrogate(self, adv_r, adv_c):
        return adv_r


# ----------------------------------------------------------- tests
def test_buffer_allocates_cost_stream():
    obs_space = spaces.Dict(
        {"x": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)}
    )
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buf = ConstrainedDictRolloutBuffer(
        buffer_size=4,
        observation_space=obs_space,
        action_space=act_space,
        n_envs=2,
        cost_gamma=0.99,
        cost_gae_lambda=0.95,
    )
    assert buf.cost_rewards.shape == (4, 2)
    assert buf.cost_returns.shape == (4, 2)
    assert buf.cost_advantages.shape == (4, 2)


def test_buffer_compute_returns_with_cost_stream():
    obs_space = spaces.Dict(
        {"x": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)}
    )
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buf = ConstrainedDictRolloutBuffer(
        buffer_size=3,
        observation_space=obs_space,
        action_space=act_space,
        n_envs=2,
    )
    obs = {"x": np.zeros((2, 2), dtype=np.float32)}
    for _ in range(3):
        buf.add(
            obs=obs,
            action=np.zeros((2, 1), dtype=np.float32),
            reward=np.array([1.0, 1.0], dtype=np.float32),
            episode_start=np.array([0.0, 0.0], dtype=np.float32),
            value=th.tensor([[0.5], [0.5]]),
            log_prob=th.tensor([-0.1, -0.1]),
            cost_reward=np.array([0.3, 0.3], dtype=np.float32),
            cost_value=th.tensor([[0.2], [0.2]]),
        )
    buf.compute_returns_and_advantage(
        last_values=th.tensor([0.5, 0.5]),
        dones=np.array([False, False]),
        last_cost_values=th.tensor([0.2, 0.2]),
    )
    assert np.all(np.isfinite(buf.cost_advantages))
    assert np.all(np.isfinite(buf.cost_returns))
    # With positive cost_rewards and finite cost_values, cost_returns > 0
    assert (buf.cost_returns > 0).all()


def test_policy_cost_value_net_in_optimizer():
    obs_space = spaces.Dict(
        {"x": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
    )
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    policy = MultiInputConstrainedActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
    )
    # cost_value_net exists and is a Linear
    assert isinstance(policy.cost_value_net, th.nn.Linear)
    # And its parameters are in the optimizer
    opt_params = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
    for p in policy.cost_value_net.parameters():
        assert id(p) in opt_params, "cost_value_net params missing from optimizer"


def test_policy_forward_returns_four():
    obs_space = spaces.Dict(
        {"x": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
    )
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    policy = MultiInputConstrainedActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
    )
    obs = {"x": th.zeros(5, 3)}
    actions, values, values_cost, log_prob = policy(obs)
    assert actions.shape == (5, 2)
    assert values.shape == (5, 1)
    assert values_cost.shape == (5, 1)
    assert log_prob.shape == (5,)


def test_subclass_must_override_adv_surrogate():
    obs_space = spaces.Dict(
        {"x": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)}
    )
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    class _NoOverride(ConstrainedPPO):
        pass

    env = _DictObsFakeVecEnv(num_envs=2)
    model = _NoOverride(env=env, n_steps=4, batch_size=4, device="cpu", verbose=0)
    with pytest.raises(NotImplementedError, match="must override"):
        model._compute_adv_surrogate(th.zeros(4), th.zeros(4))


def test_end_to_end_rollout_and_train_unconstrained_path():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = _AdvROnlyPPO(
        env=wrapped,
        n_steps=8,  # 1 episode end inside the rollout (every 8 steps)
        batch_size=8,
        n_epochs=2,
        cost_limit=10.0,
        device="cpu",
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=32)
    # Cost stream populated and accumulated by training stats
    buf = model.rollout_buffer
    assert buf.cost_rewards.sum() > 0
    # Cost returns aren't all zero — GAE actually ran on the cost stream
    assert np.any(buf.cost_returns != 0)


def test_save_load_roundtrip():
    env = _DictObsFakeVecEnv(num_envs=2)
    wrapped = CostInjectingVecEnvWrapper(env, [ConstantCost(1.0)])
    model = _AdvROnlyPPO(
        env=wrapped, n_steps=4, batch_size=4, device="cpu", verbose=0, seed=0
    )
    model.learn(total_timesteps=8)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.zip"
        model.save(str(path))
        env2 = _DictObsFakeVecEnv(num_envs=2)
        wrapped2 = CostInjectingVecEnvWrapper(env2, [ConstantCost(1.0)])
        # Reload as the same subclass — SB3 needs the class for unpickling.
        loaded = _AdvROnlyPPO.load(str(path), env=wrapped2, device="cpu")
    assert isinstance(
        loaded.policy, MultiInputConstrainedActorCriticPolicy
    )
    assert hasattr(loaded.policy, "cost_value_net")
