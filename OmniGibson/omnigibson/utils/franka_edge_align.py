from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple


DEFAULT_ROLE_WEIGHTS = {
    "target": 3.0,
    "fragile": 2.0,
    "clutter": 1.0,
}


@dataclass(frozen=True)
class EdgeAlignObject:
    name: str
    role: str
    position_xy: Tuple[float, float]


@dataclass(frozen=True)
class EdgeAlignRequest:
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    pack_objects_world: Tuple[EdgeAlignObject, ...]
    role_weights: Dict[str, float]
    robot_half_extent_xy: Tuple[float, float]
    edge_gap_m: float
    edge_margin_m: float
    scan_offsets_m: Tuple[float, ...]
    collision_checker: Optional[Callable[[Tuple[float, float, float]], Sequence[str]]] = None
    preferred_edge: Optional[str] = None


@dataclass(frozen=True)
class EdgeAlignResult:
    edge_label: str
    base_pose: Dict[str, Tuple[float, ...]]
    anchor_s: float
    candidate_rank: int
    collision_hits: Tuple[str, ...]
    gap_actual: float
    failure_reason: Optional[str]


def select_best_table_edge(
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]],
    pack_objects_world: Sequence[EdgeAlignObject],
) -> str:
    if not pack_objects_world:
        raise ValueError("pack_objects_world must be non-empty")

    (x_min, y_min), (x_max, y_max) = _normalize_aabb(table_aabb_xy)
    cx = sum(obj.position_xy[0] for obj in pack_objects_world) / float(len(pack_objects_world))
    cy = sum(obj.position_xy[1] for obj in pack_objects_world) / float(len(pack_objects_world))
    distances = {
        "x_min": abs(cx - x_min),
        "x_max": abs(cx - x_max),
        "y_min": abs(cy - y_min),
        "y_max": abs(cy - y_max),
    }
    return min(distances, key=distances.get)


def compute_weighted_edge_anchor(
    edge_label: str,
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]],
    pack_objects_world: Sequence[EdgeAlignObject],
    role_weights: Optional[Dict[str, float]],
    robot_half_extent_xy: Tuple[float, float],
    edge_margin_m: float,
) -> float:
    if not pack_objects_world:
        raise ValueError("pack_objects_world must be non-empty")

    weights = DEFAULT_ROLE_WEIGHTS if role_weights is None else role_weights
    (x_min, y_min), (x_max, y_max) = _normalize_aabb(table_aabb_xy)

    if edge_label in {"x_min", "x_max"}:
        tangent_vals = [obj.position_xy[1] for obj in pack_objects_world]
        half_tangent = robot_half_extent_xy[1]
        lo = y_min + edge_margin_m + half_tangent
        hi = y_max - edge_margin_m - half_tangent
    elif edge_label in {"y_min", "y_max"}:
        tangent_vals = [obj.position_xy[0] for obj in pack_objects_world]
        half_tangent = robot_half_extent_xy[0]
        lo = x_min + edge_margin_m + half_tangent
        hi = x_max - edge_margin_m - half_tangent
    else:
        raise ValueError(f"Unsupported edge_label: {edge_label}")

    if lo > hi:
        lo, hi = (x_min + x_max) * 0.5, (x_min + x_max) * 0.5

    weighted_sum = 0.0
    weight_sum = 0.0
    for obj, proj in zip(pack_objects_world, tangent_vals):
        w = float(weights.get(obj.role, 1.0))
        weighted_sum += w * proj
        weight_sum += w
    anchor = weighted_sum / max(weight_sum, 1e-6)
    return _clamp(anchor, lo, hi)


def place_franka_edge_aligned(request: EdgeAlignRequest) -> EdgeAlignResult:
    if not request.pack_objects_world:
        raise ValueError("pack_objects_world must be non-empty")
    if request.edge_gap_m < 0.0:
        raise ValueError("edge_gap_m must be >= 0")
    if request.edge_margin_m < 0.0:
        raise ValueError("edge_margin_m must be >= 0")
    if not request.scan_offsets_m:
        raise ValueError("scan_offsets_m must be non-empty")

    (x_min, y_min), (x_max, y_max) = _normalize_aabb(request.table_aabb_xy)
    edge_label = request.preferred_edge or select_best_table_edge(
        request.table_aabb_xy, request.pack_objects_world
    )
    anchor_s = compute_weighted_edge_anchor(
        edge_label=edge_label,
        table_aabb_xy=request.table_aabb_xy,
        pack_objects_world=request.pack_objects_world,
        role_weights=request.role_weights,
        robot_half_extent_xy=request.robot_half_extent_xy,
        edge_margin_m=request.edge_margin_m,
    )

    center_xy = _pack_center_xy(request.pack_objects_world)
    candidates: List[Tuple[int, Tuple[float, float, float], Tuple[str, ...], float]] = []
    best_by_hits: Optional[Tuple[int, Tuple[float, float, float], Tuple[str, ...], float]] = None
    failure_reason = None

    for rank, offset in enumerate(request.scan_offsets_m):
        s = _clamp_anchor_with_offset(
            edge_label=edge_label,
            anchor_s=anchor_s,
            offset=offset,
            table_aabb_xy=((x_min, y_min), (x_max, y_max)),
            robot_half_extent_xy=request.robot_half_extent_xy,
            edge_margin_m=request.edge_margin_m,
        )
        x, y, gap_actual = _edge_pose_xy(
            edge_label=edge_label,
            tangent_s=s,
            table_aabb_xy=((x_min, y_min), (x_max, y_max)),
            robot_half_extent_xy=request.robot_half_extent_xy,
            edge_gap_m=request.edge_gap_m,
        )
        yaw = math.atan2(center_xy[1] - y, center_xy[0] - x)
        hits = ()
        if request.collision_checker is not None:
            hits = tuple(sorted(set(request.collision_checker((x, y, yaw)))))

        candidate = (rank, (x, y, yaw), hits, gap_actual)
        candidates.append(candidate)
        if len(hits) == 0:
            best_by_hits = candidate
            break
        if best_by_hits is None or len(hits) < len(best_by_hits[2]):
            best_by_hits = candidate

    if best_by_hits is None:
        raise RuntimeError("No edge-aligned candidates generated.")

    rank, pose_xyyaw, hits, gap_actual = best_by_hits
    if len(hits) > 0:
        failure_reason = "no_collision_free_candidate"

    x, y, yaw = pose_xyyaw
    return EdgeAlignResult(
        edge_label=edge_label,
        base_pose={
            "position": (float(x), float(y), 0.0),
            "orientation": _quat_from_yaw(yaw),
        },
        anchor_s=float(anchor_s),
        candidate_rank=int(rank),
        collision_hits=hits,
        gap_actual=float(gap_actual),
        failure_reason=failure_reason,
    )


def _normalize_aabb(
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (x0, y0), (x1, y1) = table_aabb_xy
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))


def _pack_center_xy(pack_objects_world: Sequence[EdgeAlignObject]) -> Tuple[float, float]:
    cx = sum(obj.position_xy[0] for obj in pack_objects_world) / float(len(pack_objects_world))
    cy = sum(obj.position_xy[1] for obj in pack_objects_world) / float(len(pack_objects_world))
    return (cx, cy)


def _clamp_anchor_with_offset(
    edge_label: str,
    anchor_s: float,
    offset: float,
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]],
    robot_half_extent_xy: Tuple[float, float],
    edge_margin_m: float,
) -> float:
    (x_min, y_min), (x_max, y_max) = table_aabb_xy
    if edge_label in {"x_min", "x_max"}:
        lo = y_min + edge_margin_m + robot_half_extent_xy[1]
        hi = y_max - edge_margin_m - robot_half_extent_xy[1]
    else:
        lo = x_min + edge_margin_m + robot_half_extent_xy[0]
        hi = x_max - edge_margin_m - robot_half_extent_xy[0]
    if lo > hi:
        lo = hi = 0.5 * (lo + hi)
    return _clamp(anchor_s + offset, lo, hi)


def _edge_pose_xy(
    edge_label: str,
    tangent_s: float,
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]],
    robot_half_extent_xy: Tuple[float, float],
    edge_gap_m: float,
) -> Tuple[float, float, float]:
    (x_min, y_min), (x_max, y_max) = table_aabb_xy
    hx, hy = robot_half_extent_xy

    if edge_label == "x_min":
        x = x_min - hx - edge_gap_m
        y = tangent_s
        gap_actual = x_min - (x + hx)
    elif edge_label == "x_max":
        x = x_max + hx + edge_gap_m
        y = tangent_s
        gap_actual = x - hx - x_max
    elif edge_label == "y_min":
        x = tangent_s
        y = y_min - hy - edge_gap_m
        gap_actual = y_min - (y + hy)
    elif edge_label == "y_max":
        x = tangent_s
        y = y_max + hy + edge_gap_m
        gap_actual = y - hy - y_max
    else:
        raise ValueError(f"Unsupported edge_label: {edge_label}")

    return x, y, float(gap_actual)


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
