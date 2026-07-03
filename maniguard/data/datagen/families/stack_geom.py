"""Pure-numpy stack-retrieve layout geometry (no OmniGibson / cuRobo).

Three jobs for the stack family's ``_prepare`` / ``resolve_compute``:

  * ``combined_xy_aabb`` / ``transfer_height`` — the 4-object pack's XY footprint + the FIXED safe
    transfer height (initial tallest point), captured once on the pristine scene.
  * ``dest_center`` — where to build the ONE right-side re-stack pile: pack-right edge + a ``gap``,
    clamped to the surface + the robot's reach; ``None`` if that footprint can't fit (report the task).
  * ``dest_pile_top`` — the live top-z of whatever already sits at the dest footprint, so each place
    descends onto the growing pile (mirror of the executor's ``max_other_top_z``, scoped to the dest).

All functions take plain numpy AABBs ``(lo(3,), hi(3,))`` / xy arrays — no sim reads, so they are
trivially unit-testable; the skeleton feeds them live OmniGibson AABBs.
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


def dest_center(pack_lo_xy, pack_hi_xy, right_dir_xy, stack_half, *, gap,
                surf_lo_xy, surf_hi_xy, robot_xy, reach_max, rail_half=0.0, eef_off=0.0,
                pull_robot_ward=0.0):
    """XY centre of the right-side re-stack pile, or ``None`` if infeasible.

    Ideal centre = ``pack_centre + right_dir*(pack_half_along_right + gap + max(stack_half,
    rail_half + eef_off))``. The dest clears the source by the WIDER of the object footprint
    (``stack_half``) and the finger-rail reach toward the source (``rail_half + eef_off``, where
    ``eef_off`` = the CHOSEN grasp's eef→object-centre offset projected toward the source: 0 for a
    centre / dest-side grasp, up to ``stack_half`` for a group1-facing rim grasp). This only pushes the
    dest when a grasp actually points the rail at the source (bowl rim grasp) — a wide / centre grasp
    keeps ``max = stack_half`` and is NOT over-pushed (else the far dest makes the pure-IK carry
    unreachable, task_0000). Returns ``None`` if the dest footprint (``centre ± stack_half`` — the object,
    not the transient rail) escapes ``[surf_lo, surf_hi]`` OR the centre lies beyond ``reach_max`` of
    ``robot_xy``.
    """
    lo = np.asarray(pack_lo_xy, float)
    hi = np.asarray(pack_hi_xy, float)
    d = np.asarray(right_dir_xy, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    pack_center = 0.5 * (lo + hi)
    # pack half-extent projected onto right_dir = max over corners of |(corner - centre)·d|
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    pack_half_r = float(np.max(np.abs((corners - pack_center) @ d)))
    offset = pack_half_r + float(gap) + max(float(stack_half), float(rail_half) + float(eef_off))
    center = pack_center + d * offset
    # Fix 4 (narrow / arc table): pull the dest TOWARD the robot to reduce off-table overhang, but never
    # below the group-gap floor along `right` (clamp the along-right projection >= offset) so the source
    # clearance is preserved. pull=0 (regular wide tables) => no change.
    if float(pull_robot_ward) > 0.0:
        to_robot = np.asarray(robot_xy, float) - center
        nr = float(np.linalg.norm(to_robot))
        if nr > 1e-9:
            pulled = center + (to_robot / nr) * float(pull_robot_ward)
            proj = float((pulled - pack_center) @ d)
            if proj < offset:
                pulled = pulled + d * (offset - proj)
            center = pulled
    foot_lo = center - float(stack_half)
    foot_hi = center + float(stack_half)
    if np.any(foot_lo < np.asarray(surf_lo_xy, float)) or np.any(foot_hi > np.asarray(surf_hi_xy, float)):
        return None
    if float(np.linalg.norm(center - np.asarray(robot_xy, float))) > float(reach_max):
        return None
    return center


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
