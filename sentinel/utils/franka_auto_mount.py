from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from random import Random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AutoMountObstacle:
    name: str
    position_xy: Tuple[float, float]
    radius: float


@dataclass(frozen=True)
class AutoMountRequest:
    scene: str
    target_object: str
    target_position: Tuple[float, float, float]
    search_radius: Tuple[float, float] = (0.45, 0.75)
    height_bounds: Tuple[float, float] = (0.0, 0.0)
    max_candidates: int = 64
    ring_count: int = 4
    angle_samples: int = 16
    seed: int = 0
    obstacles: Tuple[AutoMountObstacle, ...] = ()


@dataclass(frozen=True)
class AutoMountCandidate:
    position: Tuple[float, float, float]
    yaw: float
    score: float
    reachable: bool
    collision_free: bool
    min_clearance: float
    visibility_score: float
    distance_to_target_xy: float


@dataclass(frozen=True)
class AutoMountResult:
    best_pose: Optional[Dict[str, Tuple[float, ...]]]
    ranked_candidates: Tuple[AutoMountCandidate, ...]
    failure_reason: Optional[str]
    debug_metrics: Dict[str, float]


ReachabilityFn = Callable[[Tuple[float, float, float], float], bool]
CollisionFreeFn = Callable[[Tuple[float, float, float], float], bool]
VisibilityFn = Callable[[Tuple[float, float, float], float], float]


def plan_franka_auto_mount(
    request: AutoMountRequest,
    reachability_fn: Optional[ReachabilityFn] = None,
    collision_free_fn: Optional[CollisionFreeFn] = None,
    visibility_fn: Optional[VisibilityFn] = None,
) -> AutoMountResult:
    """
    Deterministic base pose planner around a target object.

    The planner is intentionally simulator-agnostic:
    - If callback(s) are provided, they are used for high-fidelity feasibility checks.
    - Without callbacks, it falls back to geometry-based heuristics.
    """
    tx, ty, tz = request.target_position
    candidates = _generate_candidates(request=request)

    scored: List[AutoMountCandidate] = []
    n_reachable = 0
    n_collision_free = 0

    for position, yaw in candidates:
        dist = _distance_xy(position, (tx, ty, tz))
        min_clearance = _compute_min_clearance(position=position, obstacles=request.obstacles)

        reachable = (
            reachability_fn(position, yaw)
            if reachability_fn is not None
            else _default_reachability(dist=dist, radius_range=request.search_radius)
        )
        collision_free = (
            collision_free_fn(position, yaw)
            if collision_free_fn is not None
            else _default_collision_free(min_clearance=min_clearance)
        )
        visibility = (
            visibility_fn(position, yaw)
            if visibility_fn is not None
            else _default_visibility(position=position, target_position=request.target_position)
        )

        if reachable:
            n_reachable += 1
        if collision_free:
            n_collision_free += 1

        score = _score_candidate(
            dist=dist,
            min_clearance=min_clearance,
            reachable=reachable,
            collision_free=collision_free,
            visibility=visibility,
            radius_range=request.search_radius,
        )
        scored.append(
            AutoMountCandidate(
                position=position,
                yaw=yaw,
                score=score,
                reachable=reachable,
                collision_free=collision_free,
                min_clearance=min_clearance,
                visibility_score=visibility,
                distance_to_target_xy=dist,
            )
        )

    ranked = tuple(sorted(scored, key=lambda c: c.score, reverse=True))
    feasible = [c for c in ranked if c.reachable and c.collision_free]

    if feasible:
        best = feasible[0]
        best_pose = {
            "position": best.position,
            "orientation": _quat_from_yaw(best.yaw),
        }
        failure_reason = None
    else:
        best_pose = None
        failure_reason = _infer_failure_reason(ranked)

    debug_metrics = {
        "num_generated": float(len(candidates)),
        "num_ranked": float(len(ranked)),
        "num_reachable": float(n_reachable),
        "num_collision_free": float(n_collision_free),
        "num_feasible": float(len(feasible)),
    }
    return AutoMountResult(
        best_pose=best_pose,
        ranked_candidates=ranked,
        failure_reason=failure_reason,
        debug_metrics=debug_metrics,
    )


def _generate_candidates(request: AutoMountRequest) -> List[Tuple[Tuple[float, float, float], float]]:
    min_radius, max_radius = request.search_radius
    if min_radius <= 0.0 or max_radius <= min_radius:
        raise ValueError(f"Invalid search_radius: {request.search_radius}")
    if request.ring_count <= 0 or request.angle_samples <= 0:
        raise ValueError("ring_count and angle_samples must be > 0")

    rng = Random(request.seed)
    radius_step = (max_radius - min_radius) / request.ring_count
    tx, ty, _ = request.target_position

    out: List[Tuple[Tuple[float, float, float], float]] = []
    for ring_idx in range(request.ring_count):
        radius = min_radius + (ring_idx + 0.5) * radius_step
        angle_offset = rng.random() * (2.0 * pi)
        for angle_idx in range(request.angle_samples):
            angle = angle_offset + angle_idx * (2.0 * pi / request.angle_samples)
            x = tx + radius * cos(angle)
            y = ty + radius * sin(angle)
            z = request.height_bounds[0]
            # Base yaw faces target for the mounted arm setup.
            yaw = atan2(ty - y, tx - x)
            out.append(((x, y, z), yaw))
            if len(out) >= request.max_candidates:
                return out
    return out


def _distance_xy(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def _compute_min_clearance(position: Tuple[float, float, float], obstacles: Sequence[AutoMountObstacle]) -> float:
    if not obstacles:
        return 1.0
    x, y, _ = position
    clearances = []
    for obs in obstacles:
        ox, oy = obs.position_xy
        dist = sqrt((x - ox) ** 2 + (y - oy) ** 2)
        clearances.append(dist - obs.radius)
    return min(clearances)


def _default_reachability(dist: float, radius_range: Tuple[float, float]) -> bool:
    # Slightly conservative margin to avoid extreme annulus edge cases.
    min_r, max_r = radius_range
    margin = 0.03
    return (min_r + margin) <= dist <= (max_r - margin)


def _default_collision_free(min_clearance: float) -> bool:
    # Require a small positive clearance from obstacle envelopes.
    return min_clearance > 0.05


def _default_visibility(position: Tuple[float, float, float], target_position: Tuple[float, float, float]) -> float:
    # A simple bounded heuristic: closer (while still feasible) gets a mild visibility bonus.
    dist = _distance_xy(position, target_position)
    return max(0.0, min(1.0, 1.0 - (dist - 0.45) / 0.4))


def _score_candidate(
    dist: float,
    min_clearance: float,
    reachable: bool,
    collision_free: bool,
    visibility: float,
    radius_range: Tuple[float, float],
) -> float:
    min_r, max_r = radius_range
    ideal_r = 0.5 * (min_r + max_r)
    # Smaller penalty near annulus midpoint.
    dist_penalty = abs(dist - ideal_r)
    clearance_score = max(-1.0, min(1.0, min_clearance))

    score = 0.0
    score += 2.0 if reachable else -2.0
    score += 2.0 if collision_free else -2.0
    score += 0.8 * clearance_score
    score += 0.6 * max(0.0, min(1.0, visibility))
    score -= 0.5 * dist_penalty
    return score


def _infer_failure_reason(ranked: Sequence[AutoMountCandidate]) -> str:
    if not ranked:
        return "no_candidates_generated"
    any_reachable = any(c.reachable for c in ranked)
    any_collision_free = any(c.collision_free for c in ranked)
    if not any_reachable:
        return "no_reachable_candidates"
    if not any_collision_free:
        return "all_candidates_in_collision"
    return "no_feasible_candidates_after_scoring"


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    # Quaternion around z-axis.
    return (0.0, 0.0, sin(half), cos(half))
