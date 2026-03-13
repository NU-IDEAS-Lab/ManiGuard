from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class ZoneCapacityStats:
    available_area: float
    required_area: float
    utilization: float


@dataclass(frozen=True)
class TabletopZoneSpec:
    surface_bounds: Bounds2D
    obstacle_keepout_bounds: Optional[Bounds2D]
    red_zone_bounds: Bounds2D
    long_axis: str


@dataclass(frozen=True)
class KitchenBarZoneSpec:
    workspace_preset: str
    bar_bounds: Bounds2D
    sink_keepout_bounds: Bounds2D
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


def contains_point(bounds: Bounds2D, point_xy: Tuple[float, float], margin: float = 0.0) -> bool:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    x, y = point_xy
    return (x0 + margin) <= x <= (x1 - margin) and (y0 + margin) <= y <= (y1 - margin)


def clamp_point(bounds: Bounds2D, point_xy: Tuple[float, float]) -> Tuple[float, float]:
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    x = min(max(point_xy[0], x0), x1)
    y = min(max(point_xy[1], y0), y1)
    return (x, y)


def compute_zone_capacity(
    red_zone_bounds: Bounds2D,
    half_extents_xy: Sequence[Tuple[float, float]],
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


def compute_kitchen_bar_zone(
    bar_bounds_xy: Bounds2D,
    sink_bounds_xy: Bounds2D,
    workspace_preset: str = "kitchen_bar_sink_left_v1",
    edge_margin_m: float = 0.05,
    sink_keepout_margin_m: float = 0.10,
    sink_side_clearance_m: float = 0.02,
    min_zone_span_m: float = 0.20,
) -> KitchenBarZoneSpec:
    if workspace_preset != "kitchen_bar_sink_left_v1":
        raise ValueError(f"Unsupported workspace_preset: {workspace_preset}")
    if edge_margin_m < 0.0 or sink_keepout_margin_m < 0.0 or sink_side_clearance_m < 0.0:
        raise ValueError("Margins must be non-negative")

    bar_bounds = normalize_bounds(bar_bounds_xy)
    sink_keepout = expand_bounds(sink_bounds_xy, sink_keepout_margin_m)

    (bx0, by0), (bx1, by1) = bar_bounds
    x_len = bx1 - bx0
    y_len = by1 - by0

    # For this hardcoded scene preset, "sink_left" is defined as the positive side
    # of the bar long axis, which corresponds to the user-requested spacious side.
    if y_len >= x_len:
        long_axis = "y"
        zx0 = bx0 + edge_margin_m
        zx1 = bx1 - edge_margin_m
        zy0 = max(sink_keepout[1][1] + sink_side_clearance_m, by0 + edge_margin_m)
        zy1 = by1 - edge_margin_m
    else:
        long_axis = "x"
        zx0 = max(sink_keepout[1][0] + sink_side_clearance_m, bx0 + edge_margin_m)
        zx1 = bx1 - edge_margin_m
        zy0 = by0 + edge_margin_m
        zy1 = by1 - edge_margin_m

    if zx1 - zx0 < min_zone_span_m or zy1 - zy0 < min_zone_span_m:
        raise ValueError(
            "Red-zone span too small after applying sink keepout and margins: "
            f"span_x={zx1 - zx0:.3f}, span_y={zy1 - zy0:.3f}"
        )

    red_zone = normalize_bounds(((zx0, zy0), (zx1, zy1)))

    # Red zone must remain inside bar and not overlap sink keepout.
    if not _bounds_inside(red_zone, bar_bounds):
        raise ValueError("Red zone falls outside bar top bounds")
    if bounds_overlap(red_zone, sink_keepout, tol=1e-6):
        raise ValueError("Red zone overlaps sink keepout")

    return KitchenBarZoneSpec(
        workspace_preset=workspace_preset,
        bar_bounds=bar_bounds,
        sink_keepout_bounds=sink_keepout,
        red_zone_bounds=red_zone,
        long_axis=long_axis,
    )


def compute_tabletop_zone(
    surface_bounds_xy: Bounds2D,
    obstacle_bounds_xy: Optional[Bounds2D] = None,
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

    if obstacle_bounds_xy is not None:
        obstacle_keepout = expand_bounds(obstacle_bounds_xy, obstacle_keepout_margin_m)

        if long_axis == "y":
            zx0 = sx0 + edge_margin_m
            zx1 = sx1 - edge_margin_m
            # Place red zone on whichever side of the obstacle has more room.
            obs_center_y = 0.5 * (obstacle_keepout[0][1] + obstacle_keepout[1][1])
            surf_center_y = 0.5 * (sy0 + sy1)
            if obs_center_y <= surf_center_y:
                # Obstacle on low-y side → red zone on high-y side.
                zy0 = max(obstacle_keepout[1][1] + obstacle_side_clearance_m, sy0 + edge_margin_m)
                zy1 = sy1 - edge_margin_m
            else:
                zy0 = sy0 + edge_margin_m
                zy1 = min(obstacle_keepout[0][1] - obstacle_side_clearance_m, sy1 - edge_margin_m)
        else:
            zy0 = sy0 + edge_margin_m
            zy1 = sy1 - edge_margin_m
            obs_center_x = 0.5 * (obstacle_keepout[0][0] + obstacle_keepout[1][0])
            surf_center_x = 0.5 * (sx0 + sx1)
            if obs_center_x <= surf_center_x:
                zx0 = max(obstacle_keepout[1][0] + obstacle_side_clearance_m, sx0 + edge_margin_m)
                zx1 = sx1 - edge_margin_m
            else:
                zx0 = sx0 + edge_margin_m
                zx1 = min(obstacle_keepout[0][0] - obstacle_side_clearance_m, sx1 - edge_margin_m)
    else:
        obstacle_keepout = None
        zx0 = sx0 + edge_margin_m
        zx1 = sx1 - edge_margin_m
        zy0 = sy0 + edge_margin_m
        zy1 = sy1 - edge_margin_m

    if zx1 - zx0 < min_zone_span_m or zy1 - zy0 < min_zone_span_m:
        raise ValueError(
            "Red-zone span too small after applying obstacle keepout and margins: "
            f"span_x={zx1 - zx0:.3f}, span_y={zy1 - zy0:.3f}"
        )

    red_zone = normalize_bounds(((zx0, zy0), (zx1, zy1)))

    if not _bounds_inside(red_zone, surface):
        raise ValueError("Red zone falls outside surface bounds")
    if obstacle_keepout is not None and bounds_overlap(red_zone, obstacle_keepout, tol=1e-6):
        raise ValueError("Red zone overlaps obstacle keepout")

    return TabletopZoneSpec(
        surface_bounds=surface,
        obstacle_keepout_bounds=obstacle_keepout,
        red_zone_bounds=red_zone,
        long_axis=long_axis,
    )


def _bounds_inside(inner: Bounds2D, outer: Bounds2D) -> bool:
    (ix0, iy0), (ix1, iy1) = normalize_bounds(inner)
    (ox0, oy0), (ox1, oy1) = normalize_bounds(outer)
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1
