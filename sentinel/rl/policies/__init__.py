"""Policies for safe-RL training.

Currently contains ``MultiInputConstrainedActorCriticPolicy`` — an SB3
``MultiInputActorCriticPolicy`` with a parallel cost-value head. Mirrors
omnisafe's ``ConstraintActorCritic``: shared MLP / features extractor by
default, separate Linear value heads for reward and cost.
"""

from sentinel.rl.policies.constrained_actor_critic import (
    MultiInputConstrainedActorCriticPolicy,
)

__all__ = ["MultiInputConstrainedActorCriticPolicy"]
