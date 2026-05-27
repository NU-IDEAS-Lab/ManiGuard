"""Cost-function module for safe-RL training.

Mirrors OmniGibson's ``reward_functions/`` layout: each cost is a ``BaseCostFunction``
subclass with the same ``reset(task, env)`` / ``_step(task, env, action) -> (cost, info)``
lifecycle. Costs are aggregated per step by
``sentinel.rl.envs.cost_wrapper.CostInjectingVecEnvWrapper`` and surfaced through
``info["cost"]``, which the constrained-PPO buffer reads in lockstep with the
reward stream.

The design follows the cost-interface from PKU-Alignment/omnisafe (Apache-2.0):
algorithms see a scalar cost per env per step; the source of that scalar is the
plug-in point. The MVP source is ``ZeroCost`` so the algorithm plumbing can be
verified end-to-end before real cost functions (LTL, collision, drop) land.
"""

from sentinel.rl.costs.cost_function_base import BaseCostFunction
from sentinel.rl.costs.zero_cost import ConstantCost, ZeroCost

__all__ = ["BaseCostFunction", "ZeroCost", "ConstantCost"]
