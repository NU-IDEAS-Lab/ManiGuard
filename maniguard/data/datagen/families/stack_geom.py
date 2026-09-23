"""NumPy geometry helpers for stack retrieval.

Compute the initial stack footprint and top height, choose a destination pile
within surface and reach limits, compute live transfer heights, and determine
the top of the destination pile. Inputs are arrays of world AABBs and positions;
the family skeleton supplies the current simulation geometry.
"""
from __future__ import annotations

import numpy as np


def combined_xy_aabb(aabbs):
    """(lo_xy (2,), hi_xy (2,)) enclosing every object's world XY AABB.

    ``aabbs`` = list of ``(lo(3,), hi(3,))`` world AABB corners."""
    los = np.array([np.asarray(lo, float)[:2] for lo, _ in aabbs])
    his = np.array([np.asarray(hi, float)[:2] for _, hi in aabbs])
    return np.min(los, axis=0), np.max(his, axis=0)


def transfer_height(aabbs) -> float:
    """Max world top-z (``hi[2]``) over the objects = the initial tallest point of the stack pack."""
    return float(max(float(np.asarray(hi, float)[2]) for _, hi in aabbs))


def _max_onsurface_offset(pack_center, d, surf_lo, surf_hi, stack_half, slack):
    """Largest offset ``t`` such that ``pack_center + d*t`` keeps the ``±stack_half`` footprint inside
    ``[surf_lo, surf_hi]`` (relaxed outward by ``slack`` per edge), or ``None`` if no ``t`` works.

    The pile centre is constrained to the ray ``pack_center + d*t``; per axis the centre must lie in
    ``[surf_lo+stack_half-slack, surf_hi-stack_half+slack]``. Intersecting the per-axis ``t`` ranges
    gives ``[t_lo, t_hi]``; the caller wants the *largest* clearance that still fits, i.e. ``t_hi``."""
    lo_b = surf_lo + stack_half - slack           # per-axis min for the centre
    hi_b = surf_hi - stack_half + slack           # per-axis max for the centre
    t_lo, t_hi = -np.inf, np.inf
    for i in (0, 1):
        di, pc = float(d[i]), float(pack_center[i])
        if di > 1e-9:
            t_lo = max(t_lo, (lo_b[i] - pc) / di); t_hi = min(t_hi, (hi_b[i] - pc) / di)
        elif di < -1e-9:
            t_lo = max(t_lo, (hi_b[i] - pc) / di); t_hi = min(t_hi, (lo_b[i] - pc) / di)
        else:                                     # d[i]~0 -> centre_i fixed at pc; must already be inside
            if not (lo_b[i] <= pc <= hi_b[i]):
                return None
    if t_lo > t_hi:
        return None
    return (t_lo, t_hi)


def _max_reach_offset(pack_center, d, robot, reach_target):
    """Largest offset ``t`` such that ``|pack_center + d*t - robot| <= reach_target`` (``|d|==1``), or
    ``None`` if the pack is already beyond ``reach_target`` (no forward ``t`` stays in reach)."""
    v = np.asarray(pack_center, float) - np.asarray(robot, float)
    b = float(v @ d)
    c = float(v @ v) - float(reach_target) ** 2
    disc = b * b - c
    if disc < 0:
        return None
    t = -b + disc ** 0.5
    return t if t > 0 else None


def dest_center(pack_lo_xy, pack_hi_xy, right_dir_xy, stack_half, *, gap,
                surf_lo_xy, surf_hi_xy, robot_xy, reach_max, rail_half=0.0, eef_off=0.0,
                pull_robot_ward=0.0, min_gap=0.01, edge_slack=0.03,
                gap_max=None, reach_comfort=None):
    """Return a destination-pile center satisfying the configured footprint and reach bounds.

    The ideal offset clears both the object footprint and the source-facing finger
    rail. If necessary, reduce the gap to fit the surface, then allow edge_slack of
    footprint overhang while keeping the center on the surface. Return None when no
    candidate satisfies the minimum separation and reach checks. Physical stacking
    stability is assessed during execution."""
    lo = np.asarray(pack_lo_xy, float)
    hi = np.asarray(pack_hi_xy, float)
    d = np.asarray(right_dir_xy, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    surf_lo = np.asarray(surf_lo_xy, float)
    surf_hi = np.asarray(surf_hi_xy, float)
    robot = np.asarray(robot_xy, float)
    pack_center = 0.5 * (lo + hi)
    # pack half-extent projected onto right_dir = max over corners of |(corner - centre)·d|
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    pack_half_r = float(np.max(np.abs((corners - pack_center) @ d)))
    base_term = pack_half_r + max(float(stack_half), float(rail_half) + float(eef_off))
    offset = base_term + float(gap)
    # When gap_max and reach_comfort are set, widen the gap to leave room for
    # the bottom-target grasp. Bound it by the surface footprint and comfortable
    # reach, without reducing the initial minimum gap at this stage.
    if gap_max is not None and reach_comfort is not None and float(gap_max) > float(gap):
        rc = _max_reach_offset(pack_center, d, robot_xy, float(reach_comfort))
        # Bound the carry end-effector target, not only the destination pile center.
        # An edge grasp can offset the eef away from the robot by stack_half, so
        # reduce the center reach cap accordingly. Preserve the minimum-gap floor.
        if rc is not None:
            rc = rc - float(stack_half)
        iv0 = _max_onsurface_offset(pack_center, d, np.asarray(surf_lo_xy, float),
                                    np.asarray(surf_hi_xy, float), float(stack_half), 0.0)
        surf_cap = iv0[1] if iv0 is not None else offset
        cap = min(surf_cap, rc) if rc is not None else surf_cap
        offset = max(offset, min(base_term + float(gap_max), cap))

    def _reach_ok(c):
        return float(np.linalg.norm(c - robot)) <= float(reach_max)

    # --- Ideal offset with robot-ward pull ------------------------------------
    center = pack_center + d * offset
    if float(pull_robot_ward) > 0.0:
        to_robot = robot - center
        nr = float(np.linalg.norm(to_robot))
        if nr > 1e-9:
            pulled = center + (to_robot / nr) * float(pull_robot_ward)
            proj = float((pulled - pack_center) @ d)
            if proj < offset:                     # never below the group-gap floor along `right`
                pulled = pulled + d * (offset - proj)
            center = pulled
    foot_lo, foot_hi = center - float(stack_half), center + float(stack_half)
    if not (np.any(foot_lo < surf_lo) or np.any(foot_hi > surf_hi)) and _reach_ok(center):
        return center

    # --- graded fallback: clamp the pile back onto the surface (small-table recovery) ---
    min_offset = pack_half_r + float(stack_half) + float(min_gap)   # source & dest just clear
    for slack in (0.0, float(edge_slack)):
        iv = _max_onsurface_offset(pack_center, d, surf_lo, surf_hi, float(stack_half), slack)
        if iv is None:
            continue
        t_lo, t_hi = iv
        t = min(offset, t_hi)                     # keep the most source-clearance that fits
        if t >= max(min_offset, t_lo):
            c = pack_center + d * t
            if _reach_ok(c):
                return c
    return None


def live_h_safe(other_top_z, grasp_z, drop, *, clearance, finger_margin) -> float:
    """Per-phase transfer height clearing the LIVE tallest OTHER object AND the whole gripper's lowest
    point. ``= max(max(other_top_z, grasp_z) + clearance, other_top_z + drop + finger_margin)``.

    ``other_top_z`` = live max top-z over every object except the held one (drops as objects are removed,
    so H_safe drops per phase instead of staying pinned to the initial tallest object); ``grasp_z`` = the
    current phase's eef grasp z; ``drop`` = the gripper's lowest point below eef (``gripper_drop_below_eef``).
    """
    return max(max(float(other_top_z), float(grasp_z)) + float(clearance),
               float(other_top_z) + float(drop) + float(finger_margin))


def dest_pile_top(aabbs, dest_xy, footprint_half, *, support_top) -> float:
    """Max top-z over objects whose XY centre is within ``footprint_half`` (Chebyshev) of ``dest_xy``;
    ``support_top`` if none are there yet (the first object lands on the table). Scoped to the dest
    footprint so the shrinking source stack never inflates the place-descent depth."""
    dest = np.asarray(dest_xy, float)
    best = float(support_top)
    for lo, hi in aabbs:
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        cxy = 0.5 * (lo[:2] + hi[:2])
        if float(np.max(np.abs(cxy - dest))) <= float(footprint_half):
            best = max(best, float(hi[2]))
    return best
