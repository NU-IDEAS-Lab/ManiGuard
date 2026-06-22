"""Generic geometry helpers for the executor (family-agnostic).

Three jobs, all from LIVE OmniGibson reads (``obj.aabb`` = world AABB recomputed from
current collision points each call, so it is rotation-correct as the held object moves;
``robot.eef_links[arm].get_position_orientation()``):

  * **clearance over clutter** — held target's lowest world-z vs the tallest OTHER object
    resting on the surface.
  * **dynamic lift height (the 3 cm rule)** — how far the eef must rise so the held
    target's lowest point clears the tallest clutter by ``min_clearance`` before translating.
  * **terminal aim-to-centre** — the eef target that translates the held target so its
    geometric CENTRE lands on the goal-sphere centre (the §4.3 redundancy: drive to centre,
    not just to first surface contact).

No cuRobo / no recording here — just geometry the engine + skeleton query.
"""
from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


def aabb_lo_hi(obj):
    """(world min corner (3,), world max corner (3,)) of ``obj`` — live."""
    lo, hi = obj.aabb
    return _np(lo), _np(hi)


def lowest_z(obj) -> float:
    return float(aabb_lo_hi(obj)[0][2])


def top_z(obj) -> float:
    return float(aabb_lo_hi(obj)[1][2])


def object_center(obj) -> np.ndarray:
    """World AABB centre (geometric centre, not the root-link origin)."""
    lo, hi = aabb_lo_hi(obj)
    return 0.5 * (lo + hi)


def surface_top_z(support) -> float | None:
    return None if support is None else top_z(support)


def max_other_top_z(env, *, exclude=(), robots=(), support_top=None,
                    on_surface_margin: float = 0.05):
    """Max world top-z over the MANIPULABLE objects resting on the surface, excluding the held
    object + robots. Returns ``(max_top_z, name)`` (``(-inf, None)`` if none qualify).

    Fixed-base objects (the support surface, furniture like a cabinet) are skipped: they are
    structure the planner navigates AROUND, not small clutter the held object is lifted OVER —
    so they must not inflate the lift-clearance height. If ``support_top`` is given, only objects
    whose bottom-z >= ``support_top - on_surface_margin`` count (sitting on this table)."""
    excl = {id(o) for o in exclude} | {id(r) for r in robots}
    best_z, best_name = -np.inf, None
    for o in env.scene.objects:
        if id(o) in excl or getattr(o, "fixed_base", False):
            continue
        lo, hi = aabb_lo_hi(o)
        if support_top is not None and float(lo[2]) < float(support_top) - on_surface_margin:
            continue
        if float(hi[2]) > best_z:
            best_z, best_name = float(hi[2]), getattr(o, "name", None)
    return best_z, best_name


def clearance(env, target, *, robots=(), support_top=None) -> float:
    """held target's lowest-z minus the tallest other on-surface object's top-z
    (``+inf`` if nothing else is on the surface)."""
    other_top, _ = max_other_top_z(env, exclude=[target], robots=robots, support_top=support_top)
    if not np.isfinite(other_top):
        return float("inf")
    return lowest_z(target) - other_top


def lift_delta_for_clearance(env, target, *, robots=(), support_top=None,
                             min_clearance: float = 0.03) -> float:
    """Δz the eef must rise so the held target's lowest point clears the tallest other
    on-surface object by >= ``min_clearance``. 0.0 if already clear (or nothing to clear)."""
    other_top, _ = max_other_top_z(env, exclude=[target], robots=robots, support_top=support_top)
    if not np.isfinite(other_top):
        return 0.0
    required_lowest = other_top + float(min_clearance)
    return float(max(0.0, required_lowest - lowest_z(target)))


def aim_to_center_eef(robot, target, goal_center, arm=None):
    """Terminal eef target that translates the held target so its geometric CENTRE lands on
    ``goal_center`` (eef orientation unchanged). Rigid-hold (AG) assumption. Returns
    ``(eef_pos (3,), eef_quat (4, xyzw))``."""
    arm = arm or robot.default_arm
    ep, eq = robot.eef_links[arm].get_position_orientation()
    delta = _np(goal_center) - object_center(target)
    return _np(ep) + delta, _np(eq)
