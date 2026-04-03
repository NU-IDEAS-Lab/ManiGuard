from __future__ import annotations

from dataclasses import dataclass

from omnigibson.utils.tabletop_workspace import (
    Bounds2D,
    ZoneCapacityStats,
    bounds_overlap,
    clamp_point,
    compute_tabletop_zone,
    compute_zone_capacity,
    contains_point,
    expand_bounds,
    normalize_bounds,
)


@dataclass(frozen=True)
class KitchenBarZoneSpec:
    workspace_preset: str
    bar_bounds: Bounds2D
    sink_keepout_bounds: Bounds2D
    red_zone_bounds: Bounds2D
    long_axis: str
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
def _bounds_inside(inner: Bounds2D, outer: Bounds2D) -> bool:
    (ix0, iy0), (ix1, iy1) = normalize_bounds(inner)
    (ox0, oy0), (ox1, oy1) = normalize_bounds(outer)
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1


__all__ = [
    "Bounds2D",
    "ZoneCapacityStats",
    "KitchenBarZoneSpec",
    "normalize_bounds",
    "expand_bounds",
    "bounds_overlap",
    "contains_point",
    "clamp_point",
    "compute_zone_capacity",
    "compute_tabletop_zone",
    "compute_kitchen_bar_zone",
]
