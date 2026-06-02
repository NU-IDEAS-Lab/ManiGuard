"""Lagrange multiplier for constrained on-policy safe-RL.

Ported from omnisafe's ``omnisafe/common/lagrange.py`` (Apache-2.0).

The multiplier λ is projected onto [0, lambda_upper_bound] after every update.
Two parameterisations are supported:

* ``softplus=False`` (default) — λ stored directly as a ``nn.Parameter``
  clamped to [0, upper_bound] after each gradient step; matches omnisafe's
  default mode and is simpler to inspect in logs.
* ``softplus=True`` — λ = softplus(x) so the unconstrained variable x is
  optimised; the clamp applies to λ after materialisation.

The learning-rate schedule is separate from the policy optimizer because λ
typically needs a much smaller LR than the policy (1e-3 vs 3e-4 range) and
updating it on a different schedule is fine.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Lagrange(nn.Module):
    """Scalar Lagrange multiplier λ with a simple gradient update.

    Parameters
    ----------
    cost_limit:
        The per-rollout cost budget J_C. λ increases when cost exceeds this.
    lagrangian_multiplier_init:
        Initial value of λ (default 0.0).
    lambda_lr:
        Learning rate for the multiplier optimizer.
    lambda_optimizer:
        Torch optimizer class name ("Adam", "SGD", "RMSprop"). Default "Adam".
    lagrangian_upper_bound:
        Hard cap on λ (default 1000.0, effectively uncapped for normal runs).
    softplus:
        Whether to parameterise λ via softplus(x). Keeps λ > 0 automatically.
    """

    def __init__(
        self,
        cost_limit: float = 25.0,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.035,
        lambda_optimizer: str = "Adam",
        lagrangian_upper_bound: float = 1000.0,
        softplus: bool = False,
    ) -> None:
        super().__init__()
        self.cost_limit = float(cost_limit)
        self.lagrangian_upper_bound = float(lagrangian_upper_bound)
        self._softplus = softplus

        init_val = float(lagrangian_multiplier_init)
        if softplus:
            # Invert softplus: x such that softplus(x) ≈ init_val
            # softplus⁻¹(y) = log(exp(y) - 1)  for y > 0
            if init_val > 0:
                x_init = float(th.log(th.expm1(th.tensor(init_val))))
            else:
                x_init = -10.0  # softplus(-10) ≈ 0
            self._lambda_param = nn.Parameter(th.tensor(x_init, dtype=th.float32))
        else:
            self._lambda_param = nn.Parameter(
                th.tensor(max(0.0, init_val), dtype=th.float32)
            )

        opt_cls = getattr(th.optim, lambda_optimizer)
        self.lambda_optimizer: th.optim.Optimizer = opt_cls(
            [self._lambda_param], lr=lambda_lr
        )

    # ------------------------------------------------------------------
    @property
    def lagrangian_multiplier(self) -> th.Tensor:
        """The current λ value as a non-negative scalar tensor."""
        if self._softplus:
            return F.softplus(self._lambda_param).clamp(
                max=self.lagrangian_upper_bound
            )
        return self._lambda_param.clamp(min=0.0, max=self.lagrangian_upper_bound)

    def update_lagrange_multiplier(self, ep_cost: float) -> None:
        """Gradient step: λ ← λ + lr × (ep_cost − cost_limit).

        The sign convention matches omnisafe: the Lagrangian loss with respect
        to λ is ``-λ × (ep_cost − cost_limit)`` so maximising w.r.t. λ means
        the *gradient* of that loss with respect to λ is ``-(ep_cost - cost_limit)``,
        and we *minimise* with the optimizer, so the net update is:

            λ ← λ − lr × (−(ep_cost − cost_limit))
              = λ + lr × (ep_cost − cost_limit)

        This increases λ when constraints are violated and decreases it when
        not (down to the non-negativity floor at 0).
        """
        self.lambda_optimizer.zero_grad()
        # Loss to minimise (w.r.t. λ): -λ × (ep_cost - cost_limit)
        # → gradient = -(ep_cost - cost_limit)  → update adds the deficit
        loss = -self.lagrangian_multiplier * (ep_cost - self.cost_limit)
        loss.backward()
        self.lambda_optimizer.step()
        # Enforce non-negativity for the direct (non-softplus) parameterisation
        if not self._softplus:
            with th.no_grad():
                self._lambda_param.clamp_(min=0.0)

    def state_dict_extra(self) -> dict:
        """Return serialisable state for save/load (called by PPOLag.save)."""
        return {
            "lambda_param": self._lambda_param.data.clone(),
            "optimizer": self.lambda_optimizer.state_dict(),
        }

    def load_state_dict_extra(self, state: dict) -> None:
        with th.no_grad():
            self._lambda_param.copy_(state["lambda_param"])
        self.lambda_optimizer.load_state_dict(state["optimizer"])


__all__ = ["Lagrange"]
