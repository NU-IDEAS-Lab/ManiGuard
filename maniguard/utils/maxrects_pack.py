"""Offline max-rectangles 2D packer.

Given a list of object descriptors with AABB XY half-extents and a target
rectangular region, compute placements in a single closed-form pass — no
random retries, no per-attempt settle / jitter.

Inputs are all known before sim setup:
  * Pinned object models  → AABB extent_xyz from object_footprints.json.
  * Picked surface region → computed by compute_tabletop_zone.
  * min_clearance         → from PackRetryConfig.

The solver:
  1. Forces the target to the region centroid.
  2. Splits the region into 4 free rectangles around the target's
     padded footprint.
  3. For every other descriptor (largest first by short side, then long
     side — Decreasing Sort), runs Best-Short-Side-Fit (BSSF) over the
     free rectangle list, with optional 90° yaw rotation.
  4. After placement, splits the chosen free rect via the standard
     guillotine split and prunes fully-contained free rects.

Objects that don't fit are returned in ``unplaced`` so the caller can
``remove_objects`` and proceed with the survivors. Unlike the greedy
ring placer, this solver only fails to place an object when there is no
non-overlapping seat anywhere in the region — guaranteed by the
free-rect invariant — so culling is genuinely "nothing fits" rather than
"random seed didn't unlock it".

Coordinate convention: rectangles use ``(x0, y0, w, h)`` where
``(x0, y0)`` is the bottom-left corner. Placements report the **centre**
of the object's expanded (padded) footprint, matching ``ClutterPackEntry.rel_pose``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Bounds2D = tuple[tuple[float, float], tuple[float, float]]  # ((x0, y0), (x1, y1))
Rect = tuple[float, float, float, float]  # (x0, y0, w, h)


@dataclass(frozen=True)
class PackInputDescriptor:
    inst_id: str
    role: str
    extent_xy: tuple[float, float]  # full AABB extents, not half
    bottom_offset_z: float


@dataclass
class PackPlacement:
    inst_id: str
    role: str
    cx: float  # world-frame centre X (region-relative; caller translates to world)
    cy: float
    cz: float
    yaw: float  # 0 or pi/2 (rotation introduced by the solver)


@dataclass
class PackSolution:
    placements: list[PackPlacement]
    unplaced: list[str]  # inst_ids that couldn't fit
    region_bounds: Bounds2D
    min_clearance: float


def _rect_contains(outer: Rect, inner: Rect, tol: float = 1e-9) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ix >= ox - tol and iy >= oy - tol
            and ix + iw <= ox + ow + tol and iy + ih <= oy + oh + tol)


def _prune(free_rects: list[Rect]) -> list[Rect]:
    """Remove free rects fully contained in another (max-rectangles invariant)."""
    out: list[Rect] = []
    for i, r in enumerate(free_rects):
        contained = False
        for j, s in enumerate(free_rects):
            if i == j:
                continue
            if _rect_contains(s, r):
                # If both contain each other (identical), only drop the later index.
                if _rect_contains(r, s) and j < i:
                    continue
                contained = True
                break
        if not contained:
            out.append(r)
    return out


def _split_free_rect_around(rect: Rect, used: Rect) -> list[Rect]:
    """Standard 4-way split: remove `used` from `rect`, return up to 4 leftovers."""
    rx, ry, rw, rh = rect
    ux, uy, uw, uh = used
    leftovers: list[Rect] = []
    # Left strip
    if ux > rx:
        leftovers.append((rx, ry, ux - rx, rh))
    # Right strip
    if ux + uw < rx + rw:
        leftovers.append((ux + uw, ry, (rx + rw) - (ux + uw), rh))
    # Bottom strip
    if uy > ry:
        leftovers.append((rx, ry, rw, uy - ry))
    # Top strip
    if uy + uh < ry + rh:
        leftovers.append((rx, uy + uh, rw, (ry + rh) - (uy + uh)))
    return leftovers


def _split_overlapping_rects(free_rects: list[Rect], used: Rect) -> list[Rect]:
    out: list[Rect] = []
    ux, uy, uw, uh = used
    for r in free_rects:
        rx, ry, rw, rh = r
        # No overlap → keep as-is.
        if (ux >= rx + rw or ux + uw <= rx
                or uy >= ry + rh or uy + uh <= ry):
            out.append(r)
            continue
        out.extend(_split_free_rect_around(r, used))
    return _prune(out)


def _bssf_score(rect: Rect, w: float, h: float) -> tuple[float, float] | None:
    """Best-Short-Side-Fit score; lower is better. Returns None if w×h doesn't fit."""
    _, _, rw, rh = rect
    if w > rw + 1e-9 or h > rh + 1e-9:
        return None
    leftover_w = rw - w
    leftover_h = rh - h
    short = min(leftover_w, leftover_h)
    long_ = max(leftover_w, leftover_h)
    return (short, long_)


def _pick_best_rect(free_rects: list[Rect], w: float, h: float,
                    allow_rotation: bool) -> tuple[int, Rect, bool] | None:
    best_idx = -1
    best_rect: Rect | None = None
    best_score: tuple[float, float] | None = None
    best_rotated = False
    for i, r in enumerate(free_rects):
        s = _bssf_score(r, w, h)
        if s is not None and (best_score is None or s < best_score):
            best_idx, best_rect, best_score, best_rotated = i, r, s, False
        if allow_rotation and abs(w - h) > 1e-9:
            sr = _bssf_score(r, h, w)
            if sr is not None and (best_score is None or sr < best_score):
                best_idx, best_rect, best_score, best_rotated = i, r, sr, True
    if best_rect is None:
        return None
    return best_idx, best_rect, best_rotated


def _closest_xy_in_rect(rect: Rect, w: float, h: float,
                        tx: float, ty: float) -> tuple[float, float, float] | None:
    """Return (bx, by, dist) — the bottom-left position inside ``rect`` for a
    w×h placement whose centre is closest to (tx, ty). ``None`` if w×h
    doesn't fit at all.
    """
    rx, ry, rw, rh = rect
    if w > rw + 1e-9 or h > rh + 1e-9:
        return None
    # Centre may live anywhere in [rx + w/2, rx + rw - w/2] × similar for y.
    cx_lo, cx_hi = rx + 0.5 * w, rx + rw - 0.5 * w
    cy_lo, cy_hi = ry + 0.5 * h, ry + rh - 0.5 * h
    # Snap target onto that interval.
    cx = min(max(tx, cx_lo), cx_hi)
    cy = min(max(ty, cy_lo), cy_hi)
    dist = math.hypot(cx - tx, cy - ty)
    return (cx - 0.5 * w, cy - 0.5 * h, dist)


def _pick_closest_rect(free_rects: list[Rect], w: float, h: float,
                       allow_rotation: bool, tx: float, ty: float,
                       ) -> tuple[int, tuple[float, float, float, float], bool] | None:
    """Like ``_pick_best_rect`` but scores by distance of the placed
    object's centre to the target (tx, ty). Returns
    ``(idx, used_rect, rotated)`` where ``used_rect`` is the bottom-left-
    anchored ``(x0, y0, w_used, h_used)`` rectangle to mark as used.
    """
    best: tuple[float, int, tuple[float, float, float, float], bool] | None = None
    for i, r in enumerate(free_rects):
        cand = _closest_xy_in_rect(r, w, h, tx, ty)
        if cand is not None:
            bx, by, dist = cand
            if best is None or dist < best[0]:
                best = (dist, i, (bx, by, w, h), False)
        if allow_rotation and abs(w - h) > 1e-9:
            cand = _closest_xy_in_rect(r, h, w, tx, ty)
            if cand is not None:
                bx, by, dist = cand
                if best is None or dist < best[0]:
                    best = (dist, i, (bx, by, h, w), True)
    if best is None:
        return None
    _, idx, used, rotated = best
    return idx, used, rotated


def solve_pack(
    descriptors: Sequence[PackInputDescriptor],
    region_bounds: Bounds2D,
    min_clearance: float,
    *,
    target_inst_id: str | None = None,
    allow_rotation: bool = True,
    strategy: str = "surround_target",
) -> PackSolution:
    """Solve the 2D pack offline.

    All distances are in the **region-local frame** (region bottom-left is the
    origin of the region's bounding rect, but placements report relative to
    region centre to match downstream conventions). The caller is responsible
    for translating the result into world coords via ``apply_pack_transform``.

    Padding: each object's AABB is grown by ``min_clearance / 2`` on every
    side, so two placed (padded) rectangles can touch without violating the
    user-specified clearance.

    Strategies (only affects placement of non-target objects):
      * ``"surround_target"`` (default) — score each candidate placement by
        the placed object's centre distance to the target. On wide
        surfaces this packs fragiles/clutter *around* the target instead
        of corner-stacking, so the robot must navigate them to reach the
        target. Falls back to BSSF when no rect can host the object.
      * ``"bssf"`` — classic Best-Short-Side-Fit (Jylänki). Maximises
        space efficiency but tends to line objects up in one corner when
        free space is abundant.
    """
    (x0, y0), (x1, y1) = region_bounds
    region_w = max(0.0, x1 - x0)
    region_h = max(0.0, y1 - y0)
    pad = 0.5 * min_clearance

    # Region-local frame: bottom-left at (0, 0), top-right at (region_w, region_h).
    # We'll convert to "region-centred" coords at the end.
    cx_region = 0.5 * region_w
    cy_region = 0.5 * region_h

    # Pad each AABB. Track padded extents alongside the original.
    padded = {
        d.inst_id: (d.extent_xy[0] + 2 * pad, d.extent_xy[1] + 2 * pad)
        for d in descriptors
    }

    # Separate target.
    target = None
    if target_inst_id is not None:
        target = next((d for d in descriptors if d.inst_id == target_inst_id), None)
    others = [d for d in descriptors if d.inst_id != (target.inst_id if target else None)]

    placements: list[PackPlacement] = []
    unplaced: list[str] = []
    free_rects: list[Rect] = [(0.0, 0.0, region_w, region_h)]
    # Target centre in region-local coords — used by the surround strategy
    # to score placements by their distance to the target.
    target_cx_local = cx_region
    target_cy_local = cy_region

    # Place target at region centroid (axis-aligned, no rotation).
    if target is not None:
        tw, th = padded[target.inst_id]
        if tw > region_w + 1e-9 or th > region_h + 1e-9:
            unplaced.append(target.inst_id)
        else:
            tx0 = max(0.0, min(cx_region - 0.5 * tw, region_w - tw))
            ty0 = max(0.0, min(cy_region - 0.5 * th, region_h - th))
            used = (tx0, ty0, tw, th)
            free_rects = _split_overlapping_rects(free_rects, used)
            target_cx_local = tx0 + 0.5 * tw
            target_cy_local = ty0 + 0.5 * th
            placements.append(PackPlacement(
                inst_id=target.inst_id,
                role=target.role,
                cx=target_cx_local - cx_region,
                cy=target_cy_local - cy_region,
                cz=target.bottom_offset_z + 0.004,
                yaw=0.0,
            ))

    # Decreasing-sort the others by short side (then long side) so the biggest
    # awkward objects get the most flexibility — classic 2D-pack heuristic.
    def _sort_key(d: PackInputDescriptor) -> tuple[float, float, str]:
        w, h = padded[d.inst_id]
        return (-min(w, h), -max(w, h), d.inst_id)
    others_sorted = sorted(others, key=_sort_key)

    use_surround = (strategy == "surround_target")
    for d in others_sorted:
        w, h = padded[d.inst_id]
        used_rect: Rect | None = None
        rotated = False
        if use_surround:
            pick = _pick_closest_rect(
                free_rects, w, h, allow_rotation,
                target_cx_local, target_cy_local,
            )
            if pick is not None:
                _, used_rect, rotated = pick
        if used_rect is None:
            # Fallback (and the path for strategy=="bssf"): standard
            # Best-Short-Side-Fit. Guaranteed to find any seat that exists.
            pick = _pick_best_rect(free_rects, w, h, allow_rotation)
            if pick is None:
                unplaced.append(d.inst_id)
                continue
            _, chosen, rotated = pick
            rw_used, rh_used = (h, w) if rotated else (w, h)
            used_rect = (chosen[0], chosen[1], rw_used, rh_used)
        rx0, ry0, rw_used, rh_used = used_rect
        free_rects = _split_overlapping_rects(free_rects, used_rect)
        placements.append(PackPlacement(
            inst_id=d.inst_id,
            role=d.role,
            cx=(rx0 + 0.5 * rw_used) - cx_region,
            cy=(ry0 + 0.5 * rh_used) - cy_region,
            cz=d.bottom_offset_z + 0.004,
            yaw=math.pi / 2 if rotated else 0.0,
        ))

    return PackSolution(
        placements=placements,
        unplaced=unplaced,
        region_bounds=region_bounds,
        min_clearance=min_clearance,
    )
