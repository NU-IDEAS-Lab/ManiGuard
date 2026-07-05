"""Pure-numpy stack-retrieve layout geometry (no OmniGibson / cuRobo).

Three jobs for the stack family's ``_prepare`` / ``resolve_compute``:

  * ``combined_xy_aabb`` / ``transfer_height`` — the 4-object pack's XY footprint + the FIXED safe
    transfer height (initial tallest point), captured once on the pristine scene.
  * ``dest_center`` — where to build the ONE right-side re-stack pile: pack-right edge + a ``gap``,
    graded-clamped onto the surface (ideal -> slide-to-edge -> small overhang), within the robot's
    reach; ``None`` only if even the tightest non-overlapping pile is off-surface (report / swap).
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
    """XY centre of the right-side re-stack pile, or ``None`` if genuinely infeasible.

    IDEAL centre = ``pack_centre + right_dir*(pack_half_along_right + gap + max(stack_half,
    rail_half + eef_off))``. The dest clears the source by the WIDER of the object footprint
    (``stack_half``) and the finger-rail reach toward the source (``rail_half + eef_off``). When the
    ideal centre (+ the Fix-4 robot-ward pull) keeps the footprint on-surface and in reach it is
    returned unchanged — the regular wide-table path.

    GRADED FALLBACK (small tables where the ideal pile overshoots the edge — the 10 sweep setup-crashes):
    instead of failing, slide the pile back along ``right`` toward the source, keeping the MOST clearance
    that still fits, and only fail if even the tightest non-overlapping pile is off-surface:
      1. slide so the footprint stays FULLY on-surface (sacrifice gap, never overlap the source);
      2. still short -> allow the footprint to overhang the edge by ``edge_slack`` (the pile's COM /
         centre stays on-table — physically it rests, but VALIDATE stacking stability in sim);
      3. even the source-touching pile (offset < ``pack_half_r + stack_half + min_gap``) is off-surface
         -> ``None`` (true hard limit: object too big for a 2nd pile on this surface — report / swap).
    Reach is re-checked on the chosen centre (always closer than the ideal, so it never regresses reach).
    """
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
    # ADAPTIVE GAP (opt-in via gap_max + reach_comfort): on a roomy table, EXPAND the source<->dest
    # clearance beyond the minimal ``gap`` so the exposed bottom target has room to be grasped without
    # the arm fouling the re-stack pile (task_0023 waffle collapse). Only widens (never below ``gap``),
    # capped by GAP_MAX and by whichever binds first — the footprint staying on-surface or the dest
    # staying within COMFORTABLE reach. Small tables keep the minimal offset (graded clamp then shrinks).
    if gap_max is not None and reach_comfort is not None and float(gap_max) > float(gap):
        rc = _max_reach_offset(pack_center, d, robot_xy, float(reach_comfort))
        # The reach cap must bound the CARRY target, not the pile CENTRE. The re-stack carry (``over_dest``)
        # drives the held object's centre to ``dest_centre`` -> the EEF lands at ``dest_centre + grasp_off``,
        # where an edge grasp on the pile's far side offsets the EEF by up to ``+stack_half`` ALONG ``d``
        # (away from the robot). Capping only the centre at ``reach_comfort`` lets a THICK pile's carry
        # target sit ~stack_half beyond comfort, so the pure-IK servo fails while the centre looks fine
        # (task_0000: centre reach 0.70 < 0.72 but over_dest 0.81 -> every s0_carry servo_ik_fail). Pull the
        # centre cap in by stack_half so the WORST-CASE carry stays within comfort. No-op for thin piles;
        # the ``max(offset, ...)`` floor below still guarantees the minimal-gap dest (old, reachable) behaviour.
        if rc is not None:
            rc = rc - float(stack_half)
        iv0 = _max_onsurface_offset(pack_center, d, np.asarray(surf_lo_xy, float),
                                    np.asarray(surf_hi_xy, float), float(stack_half), 0.0)
        surf_cap = iv0[1] if iv0 is not None else offset
        cap = min(surf_cap, rc) if rc is not None else surf_cap
        offset = max(offset, min(base_term + float(gap_max), cap))

    def _reach_ok(c):
        return float(np.linalg.norm(c - robot)) <= float(reach_max)

    # --- tier 1: ideal offset + Fix-4 robot-ward pull (UNCHANGED — wide tables keep the same dest) ---
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
