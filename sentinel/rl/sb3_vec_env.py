"""SB3 vec-env adapter for OmniGibson.

Subclasses upstream ``omnigibson.envs.sb3_vec_env.SB3VectorEnvironment`` and
fixes two numpy/torch bridging bugs in the upstream adapter:

1. ``step_async`` stored the raw SB3 numpy action, which then reached downstream
   task/reward code that expected a torch.Tensor (e.g. GraspReward's
   ``th.abs(action)``).
2. ``step_wait`` returned ``th.clone(self.buf_rews / buf_dones)``, but those
   buffers are numpy arrays allocated by SB3's DummyVecEnv.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch as th
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

import omnigibson as og
from omnigibson.envs.sb3_vec_env import SB3VectorEnvironment, last_stepped_env, last_stepped_time


class SentinelSB3VectorEnvironment(SB3VectorEnvironment):
    """Drop-in replacement for upstream SB3VectorEnvironment."""

    def step_async(self, actions) -> None:
        with og.sim.render_on_step(self.render_on_step):
            global last_stepped_env, last_stepped_time
            if last_stepped_env != self:
                assert (
                    last_stepped_time is None or self.last_reset_time > last_stepped_time
                ), "You must call reset() before using a different environment."
                last_stepped_env = self
                last_stepped_time = time.time()

            if not isinstance(actions, th.Tensor):
                actions = th.as_tensor(actions)
            self.actions = actions
            for i, action in enumerate(actions):
                self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        with og.sim.render_on_step(self.render_on_step):
            og.sim.step()

            for env_idx in range(self.num_envs):
                obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[
                    env_idx
                ]._post_step(self.actions[env_idx])
                self.buf_dones[env_idx] = terminated or truncated
                self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

                if self.buf_dones[env_idx]:
                    self.buf_infos[env_idx]["terminal_observation"] = obs
                    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)

            return (
                self._obs_from_buf(),
                np.copy(self.buf_rews),
                np.copy(self.buf_dones),
                copy.deepcopy(self.buf_infos),
            )
