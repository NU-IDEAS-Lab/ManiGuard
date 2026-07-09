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


# --- place_across pinned-seed helpers (no-flare = elbow stays in the reach plane) ---

def arm_flare(q_arm) -> float:
    """Out-of-plane elbow-flare proxy = |panda_joint3| (arm index 2, the upper-arm-roll DOF).
    0 at reset (no flare); a real elbow-swivel branch flip moves this >= 1 rad. q_arm = 7-vec."""
    return abs(float(_np(q_arm)[2]))


def quat_yaw(quat_xyzw) -> float:
    """Yaw (rotation about world z) from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in _np(quat_xyzw))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def object_forward_extent(aabb_lo, aabb_hi, eef_pos, direction) -> float:
    """How far the AABB reaches past ``eef_pos`` along unit ``direction`` (max corner projection, >= 0).

    Used by the reach fallback to size how far the eef can be pulled back toward the robot while the held
    object still reaches into the goal sphere (bigger forward reach => larger pull-back budget)."""
    lo = _np(aabb_lo); hi = _np(aabb_hi); e = _np(eef_pos); d = _np(direction)
    d = d / (np.linalg.norm(d) + 1e-9)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return float(max(0.0, np.max((corners - e) @ d)))


def aabb_sphere_hit(aabb_lo, aabb_hi, center, radius, offset=(0.0, 0.0, 0.0)) -> bool:
    """True if the AABB rigidly shifted by ``offset`` comes within ``radius`` of ``center`` (closest-point
    test — the SAME criterion as ``utils.goal_region.object_intersects_goal_region``)."""
    lo = _np(aabb_lo) + _np(offset); hi = _np(aabb_hi) + _np(offset); c = _np(center)
    closest = np.minimum(np.maximum(c, lo), hi)
    return bool(float(np.dot(c - closest, c - closest)) <= float(radius) * float(radius))


def noflare_seed(q_arm_entry, base_pos, base_yaw: float, target_xy) -> np.ndarray:
    """Tailored no-flare IK warm-start seed (7,): aim j0 (panda_joint1) at the target azimuth in the
    base frame, zero j2 (panda_joint3 = the flare DOF), keep j1/j3/j4/j5/j6 from the live post-lift
    config (the in-plane reach + top-down wrist). A seed only — solve_ik refines it to the exact
    endpoint; j2=0 + cspace-toward-seed biases the solution onto the no-flare branch."""
    seed = _np(q_arm_entry).copy()
    d = _np(target_xy)[:2] - _np(base_pos)[:2]
    az = float(np.arctan2(d[1], d[0])) - float(base_yaw)          # target azimuth in the base frame
    seed[0] = float((az + np.pi) % (2.0 * np.pi) - np.pi)         # wrap to [-pi, pi]
    seed[2] = 0.0
    return seed


def elbow_lateral_offset(shoulder_xy, elbow_xy, eef_xy):
    """DIAGNOSTICS-ONLY Cartesian out-of-plane flare (m): |elbow offset from the vertical plane
    through the shoulder->eef reach direction|. None if the reach direction is degenerate (eef ~above
    shoulder). All args = world xy (2,) from LIVE link reads."""
    s, x, e = _np(shoulder_xy)[:2], _np(elbow_xy)[:2], _np(eef_xy)[:2]
    v = e - s
    nv = float(np.linalg.norm(v))
    if nv < 0.05:
        return None
    n = np.array([-v[1], v[0]]) / nv                              # horizontal normal to the reach plane
    return abs(float(np.dot(x - s, n)))


def servo_orient_waypoints(start_quat_xyzw, target_quat_xyzw, n: int):
    """n slerp'd orientations from start (exclusive) to target (inclusive), xyzw — the
    per-waypoint orientation ramp for a SERVO segment with ``orient_slerp`` (dusty pour)."""
    from scipy.spatial.transform import Rotation as R, Slerp
    rots = R.from_quat(np.stack([np.asarray(start_quat_xyzw, float),
                                 np.asarray(target_quat_xyzw, float)]))
    s = Slerp([0.0, 1.0], rots)
    return [s(i / n).as_quat() for i in range(1, n + 1)]
