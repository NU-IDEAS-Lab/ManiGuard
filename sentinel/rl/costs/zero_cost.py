"""Stub cost functions for MVP plumbing.

``ZeroCost`` is the MVP source: every step returns 0.0 so the constrained-PPO /
CPO / FOCOPS stack can be exercised end-to-end before real cost signals
(LTL monitor, collision, drop-after-grasp) are wired in.

``ConstantCost`` is a non-zero counterpart used in unit tests so the Lagrange
update path / CPO line-search actually executes the cost-feasibility branches at
least once before we ship.
"""

from __future__ import annotations

from typing import Any

from sentinel.rl.costs.cost_function_base import BaseCostFunction


class ZeroCost(BaseCostFunction):
    """Always-zero cost — the MVP stub."""

    def _step(self, task: Any, env: Any, action: Any) -> tuple[float, dict[str, Any]]:
        return 0.0, {}


class ConstantCost(BaseCostFunction):
    """Constant per-step cost — exercises Lagrange / CPO code paths in tests."""

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)

    def _step(self, task: Any, env: Any, action: Any) -> tuple[float, dict[str, Any]]:
        return self.value, {"constant_cost": self.value}


__all__ = ["ZeroCost", "ConstantCost"]
