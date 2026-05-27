"""Safe-RL algorithm internals: shared bases + math utilities.

* ``ConstrainedPPO`` — shared base for PPO-Lag and FOCOPS. Adds a cost critic,
  cost-stream rollout buffer, and a ``_compute_adv_surrogate`` hook subclasses
  override to weight the policy gradient.
* ``Lagrange`` — scalar λ multiplier with gradient updates.
* ``trpo_utils`` (Phase 4) — conjugate gradient + Fisher-vector product +
  flat-param helpers for the CPO line search.
"""

from sentinel.rl.safe.constrained_ppo import ConstrainedPPO
from sentinel.rl.safe.lagrange import Lagrange

__all__ = ["ConstrainedPPO", "Lagrange"]
