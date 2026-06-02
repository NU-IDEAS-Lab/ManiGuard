"""VecEnv wrapper that injects a per-env, per-step cost scalar into ``info["cost"]``.

Sits between ``SentinelSB3VectorEnvironment`` (or the single-env
``DummyVecEnv(Monitor(...))``) and the constrained-PPO / CPO / FOCOPS trainer.
Holds one independent copy of each ``BaseCostFunction`` per env slot so cost
functions can accumulate per-episode state (e.g. "cost only after first grasp")
without leaking across the parallel slots.

The wrapper handles two non-obvious cases:

* **Sim-fault recovery**: when the underlying
  ``SentinelSB3VectorEnvironment._recover_from_sim_step_fault`` returns a
  synthetic skipped step (``info["sim_step_skipped"]`` or
  ``info["sim_fault_recovered"]`` set), we write ``info["cost"] = 0`` for that
  slot without invoking the cost functions — both because the action wasn't
  really applied and because cost-fn internal state may have been invalidated by
  the fault.
* **Episode boundaries**: on ``dones[i]``, we call ``reset(task, env)`` on each
  cost-fn for that slot so the next episode starts with a clean ledger. The
  underlying OG env has already auto-reset by the time ``step_wait`` returns;
  the wrapper just mirrors that boundary into the cost functions.

For the MVP, cost functions get ``task=None, env=None`` because ``ZeroCost`` /
``ConstantCost`` don't need them. When real cost functions land that read scene
state (LTL monitor, collision probes), we'll resolve per-slot ``task`` / ``env``
via ``self.venv.get_attr(...)`` then.
"""

from __future__ import annotations

import copy
import logging
from typing import Iterable

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
    VecEnvWrapper,
)

from sentinel.rl.costs.cost_function_base import BaseCostFunction

log = logging.getLogger(__name__)


class CostInjectingVecEnvWrapper(VecEnvWrapper):
    """Add a scalar ``info["cost"]`` (and ``info["cost_breakdown"]``) per env step.

    Args:
        venv: the underlying vectorised env (typically
            ``SentinelSB3VectorEnvironment`` or ``DummyVecEnv(Monitor(...))``).
        cost_functions: list of cost-function instances. They are deep-copied
            once per env slot so per-episode state in cost fns doesn't bleed
            across parallel rollouts.
    """

    def __init__(
        self,
        venv: VecEnv,
        cost_functions: Iterable[BaseCostFunction],
    ) -> None:
        super().__init__(venv)
        cost_functions = list(cost_functions)
        if not cost_functions:
            raise ValueError(
                "CostInjectingVecEnvWrapper requires at least one cost function "
                "(use ZeroCost for an MVP stub)."
            )
        for fn in cost_functions:
            if not isinstance(fn, BaseCostFunction):
                raise TypeError(
                    f"cost function {fn!r} is not a BaseCostFunction subclass"
                )
        self._cost_fns_per_env: list[list[BaseCostFunction]] = [
            [copy.deepcopy(fn) for fn in cost_functions]
            for _ in range(self.num_envs)
        ]
        self._last_actions: np.ndarray | None = None

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        actions = self._last_actions
        for env_idx, info in enumerate(infos):
            # Sim-fault skipped steps: action wasn't really applied; cost-fn
            # internal state may be stale. Mirror the zero-reward semantics.
            if info.get("sim_step_skipped") or info.get("sim_fault_recovered"):
                info["cost"] = 0.0
                info["cost_breakdown"] = {}
                # Fault recovery resets the underlying env; reset cost fns too.
                if info.get("sim_fault_recovered"):
                    for fn in self._cost_fns_per_env[env_idx]:
                        fn.reset(task=None, env=None)
                continue

            action = actions[env_idx] if actions is not None else None
            total_cost = 0.0
            breakdown: dict = {}
            for fn in self._cost_fns_per_env[env_idx]:
                cost_value, fn_info = fn.step(task=None, env=None, action=action)
                total_cost += cost_value
                breakdown[type(fn).__name__] = cost_value
                for k, v in fn_info.items():
                    breakdown[f"{type(fn).__name__}.{k}"] = v
            info["cost"] = float(total_cost)
            info["cost_breakdown"] = breakdown

            # Episode boundary: OG vec env has already auto-reset the slot;
            # reset our cost-fn state to match.
            if dones[env_idx]:
                for fn in self._cost_fns_per_env[env_idx]:
                    fn.reset(task=None, env=None)

        return obs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        for env_fns in self._cost_fns_per_env:
            for fn in env_fns:
                fn.reset(task=None, env=None)
        return obs


__all__ = ["CostInjectingVecEnvWrapper"]
