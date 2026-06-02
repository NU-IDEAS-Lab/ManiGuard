"""Abstract base class for cost functions.

Lifecycle mirrors ``omnigibson.reward_functions.reward_function_base.BaseRewardFunction``
so cost functions feel native to anyone who has touched OG reward functions, and so
the same task / env hooks can drive both streams. Reset is called once per episode
(at env.reset), step is called once per env.step with ``(task, env, action)`` and
must return a scalar cost plus an optional info dict for logging.

The interface is the one PKU-Alignment/omnisafe (Apache-2.0) uses internally: a
per-step scalar cost that the on-policy adapter sums into episode-level cost for
the Lagrangian / CPO line-search / FOCOPS constraint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


class BaseCostFunction(ABC):
    """Base class for safe-RL cost functions.

    Concrete subclasses implement ``_step``; ``step`` caches the returned values
    on the instance so external code (callbacks, tests) can inspect the most
    recent (cost, info) without recomputation.
    """

    def __init__(self) -> None:
        self._cost: float | None = None
        self._info: dict[str, Any] | None = None

    @abstractmethod
    def _step(self, task: Any, env: Any, action: Any) -> tuple[float, dict[str, Any]]:
        """Compute this cost at the current sim step.

        Args:
            task: The OG task instance (e.g. ``PickAndLiftPrivilegedTask``).
                May be ``None`` when the cost function does not need task state.
            env: The single OG ``Environment`` instance for this vec slot.
                May be ``None`` for purely action-based costs.
            action: The action that was just stepped. Type matches whatever the
                vec env wrapper hands us (numpy or torch).

        Returns:
            ``(cost, info)`` — ``cost`` is a non-negative scalar; ``info`` is a
            small dict of debug fields (subclass-specific). The convention from
            omnisafe / safety-gymnasium is that costs are non-negative; we don't
            enforce it but downstream Lagrangian math assumes it.
        """
        raise NotImplementedError

    def step(self, task: Any, env: Any, action: Any) -> tuple[float, dict[str, Any]]:
        self._cost, self._info = self._step(task=task, env=env, action=action)
        return float(self._cost), deepcopy(self._info)

    def reset(self, task: Any, env: Any) -> None:
        self._cost = None
        self._info = None

    @property
    def cost(self) -> float:
        if self._cost is None:
            raise RuntimeError("cost queried before first step()")
        return float(self._cost)

    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            raise RuntimeError("info queried before first step()")
        return deepcopy(self._info)


__all__ = ["BaseCostFunction"]
