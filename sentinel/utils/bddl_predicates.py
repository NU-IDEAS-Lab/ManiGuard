"""Sentinel-owned BDDL predicates, registered onto OmniGibson at patch time.

Upstream OmniGibson only knows about the predicate set shipped in
``omnigibson.utils.bddl_utils.SUPPORTED_PREDICATES``. Sentinel adds four more:

* ``upright`` / ``dropped`` — unary predicates backed by the Sentinel object
  states of the same names (see :mod:`sentinel.object_states`).
* ``grasped`` — binary predicate over ``Grasped`` (alias of upstream
  ``IsGrasping``, registered by the object-states import hook).
* ``stashed`` — a dedicated :class:`ObjectStateStashedPredicate` that tells the
  sampler to import an object but skip placement, so pipeline code can
  teleport it to the desired pose after ``env.reset()``.

These are registered by :func:`register_sentinel_predicates`, which
:mod:`sentinel._omnigibson_patches` calls after ``import omnigibson`` returns.
"""
from __future__ import annotations


def _build_stashed_predicate_class():
    """Defer imports so the module loads before OmniGibson does."""
    from bddl.logic_base import UnaryAtomicFormula
    from omnigibson.utils.bddl_utils import UnsampleablePredicate

    class ObjectStateStashedPredicate(UnsampleablePredicate, UnaryAtomicFormula):
        """Sampler hint: import the object at a stash location, skip placement.

        The object is imported at a stash position by
        ``BDDLSampler._import_sampleable_objects`` and left there. Pipeline
        code is responsible for teleporting it to the desired location after
        ``env.reset()``. Evaluation always returns ``True`` (the object
        exists and is available for manipulation).
        """

        STATE_NAME = "stashed"

        def _evaluate(self, entity, **kwargs):
            return entity.exists

    return ObjectStateStashedPredicate


def register_sentinel_predicates() -> None:
    """Register Sentinel's extra predicates on ``SUPPORTED_PREDICATES``.

    Idempotent: safe to call more than once. Requires that the object-states
    import hook has already injected ``Dropped`` / ``Upright`` / ``Grasped``
    onto ``omnigibson.object_states`` (see :func:`sentinel._omnigibson_patches.apply`).
    """
    from omnigibson import object_states
    from omnigibson.utils.bddl_utils import (
        SUPPORTED_PREDICATES,
        get_binary_predicate_for_state,
        get_unary_predicate_for_state,
    )

    stashed_cls = _build_stashed_predicate_class()

    SUPPORTED_PREDICATES.setdefault(
        "upright", get_unary_predicate_for_state(object_states.Upright, "upright")
    )
    SUPPORTED_PREDICATES.setdefault(
        "dropped", get_unary_predicate_for_state(object_states.Dropped, "dropped")
    )
    SUPPORTED_PREDICATES.setdefault(
        "grasped", get_binary_predicate_for_state(object_states.Grasped, "grasped")
    )
    SUPPORTED_PREDICATES.setdefault("stashed", stashed_cls)


__all__ = ["register_sentinel_predicates"]
