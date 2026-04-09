from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TableEdgeMountObstacle:
    name: str
    center_xy: Tuple[float, float]
    radius: float


@dataclass(frozen=True)
class TableEdgeMountRequest:
    target_position: Tuple[float, float, float]
    table_aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    ground_z: float = 0.0
    desired_gap: float = 0.03
    gap_tolerance: float = 0.02
    reachable_distance_range: Tuple[float, float] = (0.45, 1.10)
    edge_margin: float = 0.08
    tangent_offsets: Tuple[float, ...] = (-0.22, -0.12, 0.0, 0.12, 0.22)
    # Robot footprint half-extent in XY. This is used to convert desired gap into center offset from table edge.
    robot_radius: float = 0.24
    min_clearance: float = 0.06
    obstacles: Tuple[TableEdgeMountObstacle, ...] = ()


@dataclass(frozen=True)
class TableEdgeMountCandidate:
    position: Tuple[float, float, float]
    yaw: float
    edge_label: str
    gap_actual: float
    gap_error: float
    distance_to_target_xy: float
    min_clearance: float
    gap_ok: bool
    reachable: bool
    collision_free: bool
    feasible: bool
    score: float


@dataclass(frozen=True)
class TableEdgeMountResult:
    best_pose: Optional[Dict[str, Tuple[float, ...]]]
    ranked_candidates: Tuple[TableEdgeMountCandidate, ...]
    failure_reason: Optional[str]
    debug_metrics: Dict[str, float]


def plan_table_edge_mount(request: TableEdgeMountRequest) -> TableEdgeMountResult:
    _validate_request(request)

    tx, ty, _ = request.target_position
    (x_min, y_min), (x_max, y_max) = _normalize_aabb(request.table_aabb_xy)

    edge_candidates = sorted(
        (
            ("x_min", abs(tx - x_min)),
            ("x_max", abs(tx - x_max)),
            ("y_min", abs(ty - y_min)),
            ("y_max", abs(ty - y_max)),
        ),
        key=lambda item: item[1],
    )

    candidates: List[TableEdgeMountCandidate] = []
    n_feasible = 0
    n_reachable = 0
    n_collision_free = 0
    n_gap_ok = 0

    for edge_label, _ in edge_candidates:
        for tangent_offset in request.tangent_offsets:
            position_xy, edge_coord = _edge_position_with_offset(
                edge_label=edge_label,
                tangent_offset=tangent_offset,
                target_xy=(tx, ty),
                aabb_xy=((x_min, y_min), (x_max, y_max)),
                desired_gap=request.desired_gap,
                robot_radius=request.robot_radius,
                edge_margin=request.edge_margin,
            )
            x, y = position_xy
            z = request.ground_z
            yaw = atan2(ty - y, tx - x)
            dist = _distance_xy((x, y), (tx, ty))
            # Gap is defined from robot footprint side to the table edge, not from robot center to edge.
            gap_actual = abs(edge_coord - (x if edge_label.startswith("x_") else y)) - request.robot_radius
            gap_error = abs(gap_actual - request.desired_gap)
            min_clearance = _compute_min_clearance(
                position_xy=(x, y),
                obstacles=request.obstacles,
                robot_radius=request.robot_radius,
            )

            min_reach, max_reach = request.reachable_distance_range
            reachable = min_reach <= dist <= max_reach
            collision_free = min_clearance >= request.min_clearance
            gap_ok = gap_error <= request.gap_tolerance
            feasible = reachable and collision_free and gap_ok

            if reachable:
                n_reachable += 1
            if collision_free:
                n_collision_free += 1
            if gap_ok:
                n_gap_ok += 1
            if feasible:
                n_feasible += 1

            score = _score_candidate(
                dist=dist,
                gap_error=gap_error,
                min_clearance=min_clearance,
                reachable=reachable,
                collision_free=collision_free,
                gap_ok=gap_ok,
                reachable_distance_range=request.reachable_distance_range,
                gap_tolerance=request.gap_tolerance,
            )
            candidates.append(
                TableEdgeMountCandidate(
                    position=(x, y, z),
                    yaw=yaw,
                    edge_label=edge_label,
                    gap_actual=gap_actual,
                    gap_error=gap_error,
                    distance_to_target_xy=dist,
                    min_clearance=min_clearance,
                    gap_ok=gap_ok,
                    reachable=reachable,
                    collision_free=collision_free,
                    feasible=feasible,
                    score=score,
                )
            )

    ranked = tuple(sorted(candidates, key=lambda cand: cand.score, reverse=True))
    feasible = [cand for cand in ranked if cand.feasible]
    best_pose = None
    failure_reason = None

    if feasible:
        best = feasible[0]
        best_pose = {
            "position": best.position,
            "orientation": _quat_from_yaw(best.yaw),
        }
    else:
        failure_reason = _infer_failure_reason(ranked)

    debug_metrics = {
        "num_candidates": float(len(ranked)),
        "num_gap_ok": float(n_gap_ok),
        "num_reachable": float(n_reachable),
        "num_collision_free": float(n_collision_free),
        "num_feasible": float(n_feasible),
    }
    return TableEdgeMountResult(
        best_pose=best_pose,
        ranked_candidates=ranked,
        failure_reason=failure_reason,
        debug_metrics=debug_metrics,
    )


def _validate_request(request: TableEdgeMountRequest) -> None:
    if request.desired_gap <= 0.0:
        raise ValueError("desired_gap must be > 0")
    if request.gap_tolerance < 0.0:
        raise ValueError("gap_tolerance must be >= 0")
    min_reach, max_reach = request.reachable_distance_range
    if min_reach <= 0.0 or max_reach <= min_reach:
        raise ValueError("reachable_distance_range must be valid")
    if request.edge_margin < 0.0:
        raise ValueError("edge_margin must be >= 0")
    if request.robot_radius <= 0.0:
        raise ValueError("robot_radius must be > 0")
    if request.min_clearance < 0.0:
        raise ValueError("min_clearance must be >= 0")
    if len(request.tangent_offsets) == 0:
        raise ValueError("tangent_offsets must be non-empty")


def _normalize_aabb(
    aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (x0, y0), (x1, y1) = aabb_xy
    x_min = min(x0, x1)
    y_min = min(y0, y1)
    x_max = max(x0, x1)
    y_max = max(y0, y1)
    return (x_min, y_min), (x_max, y_max)


def _edge_position_with_offset(
    edge_label: str,
    tangent_offset: float,
    target_xy: Tuple[float, float],
    aabb_xy: Tuple[Tuple[float, float], Tuple[float, float]],
    desired_gap: float,
    robot_radius: float,
    edge_margin: float,
) -> Tuple[Tuple[float, float], float]:
    tx, ty = target_xy
    (x_min, y_min), (x_max, y_max) = aabb_xy

    y_low = y_min + edge_margin
    y_high = y_max - edge_margin
    x_low = x_min + edge_margin
    x_high = x_max - edge_margin

    if y_low > y_high:
        y_low, y_high = y_min, y_max
    if x_low > x_high:
        x_low, x_high = x_min, x_max

    if edge_label == "x_min":
        x = x_min - (desired_gap + robot_radius)
        y = _clamp(ty + tangent_offset, y_low, y_high)
        return (x, y), x_min
    if edge_label == "x_max":
        x = x_max + (desired_gap + robot_radius)
        y = _clamp(ty + tangent_offset, y_low, y_high)
        return (x, y), x_max
    if edge_label == "y_min":
        x = _clamp(tx + tangent_offset, x_low, x_high)
        y = y_min - (desired_gap + robot_radius)
        return (x, y), y_min
    if edge_label == "y_max":
        x = _clamp(tx + tangent_offset, x_low, x_high)
        y = y_max + (desired_gap + robot_radius)
        return (x, y), y_max
    raise ValueError(f"Unknown edge label: {edge_label}")


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def _compute_min_clearance(
    position_xy: Tuple[float, float],
    obstacles: Sequence[TableEdgeMountObstacle],
    robot_radius: float,
) -> float:
    if not obstacles:
        return 10.0
    x, y = position_xy
    clearances = []
    for obstacle in obstacles:
        ox, oy = obstacle.center_xy
        dist = _distance_xy((x, y), (ox, oy))
        clearances.append(dist - (robot_radius + max(obstacle.radius, 0.0)))
    return min(clearances)


def _score_candidate(
    dist: float,
    gap_error: float,
    min_clearance: float,
    reachable: bool,
    collision_free: bool,
    gap_ok: bool,
    reachable_distance_range: Tuple[float, float],
    gap_tolerance: float,
) -> float:
    min_reach, max_reach = reachable_distance_range
    ideal_reach = 0.5 * (min_reach + max_reach)
    dist_penalty = abs(dist - ideal_reach)
    gap_norm = gap_error / max(gap_tolerance, 1e-6)
    clearance_score = max(-1.0, min(1.0, min_clearance))

    score = 0.0
    score += 2.0 if reachable else -2.0
    score += 2.0 if collision_free else -2.0
    score += 1.8 if gap_ok else -1.8
    score += 0.5 * clearance_score
    score -= 0.8 * dist_penalty
    score -= 0.5 * gap_norm
    return score


def _infer_failure_reason(ranked: Sequence[TableEdgeMountCandidate]) -> str:
    if not ranked:
        return "no_candidates_generated"
    any_gap_ok = any(candidate.gap_ok for candidate in ranked)
    any_reachable = any(candidate.reachable for candidate in ranked)
    any_collision_free = any(candidate.collision_free for candidate in ranked)
    if not any_gap_ok:
        return "no_candidates_meet_gap_constraint"
    if not any_reachable:
        return "no_candidates_meet_reachability"
    if not any_collision_free:
        return "all_candidates_in_collision"
    return "no_feasible_candidates_after_scoring"


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, sin(half), cos(half))
