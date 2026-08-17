from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Bounds2D = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class ZoneCapacityStats:
    available_area: float
    required_area: float
    utilization: float


@dataclass(frozen=True)
class TabletopZoneSpec:
    surface_bounds: Bounds2D
    obstacle_keepout_bounds: Bounds2D | None
    obstacle_keepout_bounds_seq: tuple[Bounds2D, ...]
    red_zone_bounds: Bounds2D
    long_axis: str


def normalize_bounds(bounds: Bounds2D) -> Bounds2D:
    (x0, y0), (x1, y1) = bounds
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))


def expand_bounds(bounds: Bounds2D, margin: float) -> Bounds2D:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    return ((x0 - margin, y0 - margin), (x1 + margin, y1 + margin))


def bounds_area(bounds: Bounds2D) -> float:
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bounds_overlap(a: Bounds2D, b: Bounds2D, tol: float = 0.0) -> bool:
    (ax0, ay0), (ax1, ay1) = normalize_bounds(a)
    (bx0, by0), (bx1, by1) = normalize_bounds(b)
    overlap_x = min(ax1, bx1) - max(ax0, bx0) > tol
    overlap_y = min(ay1, by1) - max(ay0, by0) > tol
    return overlap_x and overlap_y


def contains_point(bounds: Bounds2D, point_xy: tuple[float, float], margin: float = 0.0) -> bool:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    x, y = point_xy
    return (x0 + margin) <= x <= (x1 - margin) and (y0 + margin) <= y <= (y1 - margin)


def clamp_point(bounds: Bounds2D, point_xy: tuple[float, float]) -> tuple[float, float]:
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    x = min(max(point_xy[0], x0), x1)
    y = min(max(point_xy[1], y0), y1)
    return (x, y)


def compute_zone_capacity(
    red_zone_bounds: Bounds2D,
    half_extents_xy: Sequence[tuple[float, float]],
    per_object_padding: float = 0.02,
) -> ZoneCapacityStats:
    if per_object_padding < 0.0:
        raise ValueError("per_object_padding must be non-negative")

    available = bounds_area(red_zone_bounds)
    required = 0.0
    for hx, hy in half_extents_xy:
        width = max(0.0, 2.0 * float(hx) + per_object_padding)
        depth = max(0.0, 2.0 * float(hy) + per_object_padding)
        required += width * depth

    utilization = 0.0 if available <= 0.0 else required / available
    return ZoneCapacityStats(available_area=available, required_area=required, utilization=utilization)


def compute_tabletop_zone(
    surface_bounds_xy: Bounds2D,
    obstacle_bounds_xy: Bounds2D | None = None,
    obstacle_bounds_seq: Sequence[Bounds2D] | None = None,
    edge_margin_m: float = 0.05,
    obstacle_keepout_margin_m: float = 0.10,
    obstacle_side_clearance_m: float = 0.02,
    min_zone_span_m: float = 0.20,
) -> TabletopZoneSpec:
    if edge_margin_m < 0.0 or obstacle_keepout_margin_m < 0.0 or obstacle_side_clearance_m < 0.0:
        raise ValueError("Margins must be non-negative")

    surface = normalize_bounds(surface_bounds_xy)
    (sx0, sy0), (sx1, sy1) = surface
    x_len = sx1 - sx0
    y_len = sy1 - sy0
    long_axis = "y" if y_len >= x_len else "x"

    zx0 = sx0 + edge_margin_m
    zx1 = sx1 - edge_margin_m
    zy0 = sy0 + edge_margin_m
    zy1 = sy1 - edge_margin_m

    usable_bounds = normalize_bounds(((zx0, zy0), (zx1, zy1)))

    obstacle_inputs = []
    if obstacle_bounds_xy is not None:
        obstacle_inputs.append(obstacle_bounds_xy)
    if obstacle_bounds_seq:
        obstacle_inputs.extend(obstacle_bounds_seq)

    obstacle_keepouts = []
    seen_keepouts = set()
    total_keepout_margin = obstacle_keepout_margin_m + obstacle_side_clearance_m
    for raw_bounds in obstacle_inputs:
        keepout = normalize_bounds(expand_bounds(raw_bounds, total_keepout_margin))
        clamped = normalize_bounds((
            (max(keepout[0][0], usable_bounds[0][0]), max(keepout[0][1], usable_bounds[0][1])),
            (min(keepout[1][0], usable_bounds[1][0]), min(keepout[1][1], usable_bounds[1][1])),
        ))
        if bounds_area(clamped) <= 0.0:
            continue
        key = tuple(round(v, 6) for pair in clamped for v in pair)
        if key in seen_keepouts:
            continue
        seen_keepouts.add(key)
        obstacle_keepouts.append(clamped)

    obstacle_keepout = obstacle_keepouts[0] if len(obstacle_keepouts) == 1 else None
    if obstacle_keepouts:
        xs = {usable_bounds[0][0], usable_bounds[1][0]}
        ys = {usable_bounds[0][1], usable_bounds[1][1]}
        for bounds in obstacle_keepouts:
            xs.add(bounds[0][0])
            xs.add(bounds[1][0])
            ys.add(bounds[0][1])
            ys.add(bounds[1][1])
        xs = sorted(xs)
        ys = sorted(ys)

        best_rect = None
        best_area = -1.0
        for xi in range(len(xs) - 1):
            for xj in range(xi + 1, len(xs)):
                cand_x0, cand_x1 = xs[xi], xs[xj]
                if cand_x1 - cand_x0 < min_zone_span_m:
                    continue
                for yi in range(len(ys) - 1):
                    for yj in range(yi + 1, len(ys)):
                        cand_y0, cand_y1 = ys[yi], ys[yj]
                        if cand_y1 - cand_y0 < min_zone_span_m:
                            continue
                        rect = normalize_bounds(((cand_x0, cand_y0), (cand_x1, cand_y1)))
                        if not _bounds_inside(rect, usable_bounds):
                            continue
                        if any(bounds_overlap(rect, keepout, tol=1e-6) for keepout in obstacle_keepouts):
                            continue
                        area = bounds_area(rect)
                        if area <= best_area:
                            continue
                        best_area = area
                        best_rect = rect
        if best_rect is None:
            raise ValueError(
                "Red-zone span too small after applying obstacle keepout and margins: "
                f"span_x={usable_bounds[1][0] - usable_bounds[0][0]:.3f}, "
                f"span_y={usable_bounds[1][1] - usable_bounds[0][1]:.3f}"
            )
        red_zone = best_rect
    else:
        red_zone = usable_bounds
    if red_zone[1][0] - red_zone[0][0] < min_zone_span_m or red_zone[1][1] - red_zone[0][1] < min_zone_span_m:
        raise ValueError(
            "Red-zone span too small after applying obstacle keepout and margins: "
            f"span_x={red_zone[1][0] - red_zone[0][0]:.3f}, "
            f"span_y={red_zone[1][1] - red_zone[0][1]:.3f}"
        )

    if not _bounds_inside(red_zone, surface):
        raise ValueError("Red zone falls outside surface bounds")
    if any(bounds_overlap(red_zone, keepout, tol=1e-6) for keepout in obstacle_keepouts):
        raise ValueError("Red zone overlaps obstacle keepout")

    return TabletopZoneSpec(
        surface_bounds=surface,
        obstacle_keepout_bounds=obstacle_keepout,
        obstacle_keepout_bounds_seq=tuple(obstacle_keepouts),
        red_zone_bounds=red_zone,
        long_axis=long_axis,
    )


def _bounds_inside(inner: Bounds2D, outer: Bounds2D) -> bool:
    (ix0, iy0), (ix1, iy1) = normalize_bounds(inner)
    (ox0, oy0), (ox1, oy1) = normalize_bounds(outer)
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1


__all__ = [
    "Bounds2D",
    "TabletopZoneSpec",
    "ZoneCapacityStats",
    "bounds_area",
    "bounds_overlap",
    "clamp_point",
    "compute_tabletop_zone",
    "compute_zone_capacity",
    "contains_point",
    "expand_bounds",
    "normalize_bounds",
]
