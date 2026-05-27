"""Safe-RL algorithm internals: shared bases + math utilities.

* ``ConstrainedPPO`` — shared base for PPO-Lag and FOCOPS. Adds a cost critic,
  cost-stream rollout buffer, and a ``_compute_adv_surrogate`` hook subclasses
  override to weight the policy gradient.
* ``Lagrange`` — scalar λ multiplier with gradient updates.
* ``trpo_utils`` — conjugate gradient + flat-param helpers for CPO.
"""

from sentinel.rl.safe.constrained_ppo import ConstrainedPPO
from sentinel.rl.safe.lagrange import Lagrange
from sentinel.rl.safe.trpo_utils import (
    conjugate_gradients,
    flat_grad,
    flat_params,
    set_flat_params,
)

__all__ = [
    "ConstrainedPPO",
    "Lagrange",
    "conjugate_gradients",
    "flat_grad",
    "flat_params",
    "set_flat_params",
]
