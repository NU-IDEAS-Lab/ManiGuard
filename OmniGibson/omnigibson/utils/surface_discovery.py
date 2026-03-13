"""Pure-geometry surface scoring and obstacle detection.

No simulator dependency — operates only on AABBs and category strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]

_TABLE_LIKE_CATEGORIES = frozenset({
    "bar", "breakfast_table", "coffee_table", "console_table",
    "counter", "countertop", "desk", "dining_table", "kitchen_table",
    "nightstand", "side_table", "table", "workbench",
})

_OBSTACLE_CATEGORIES: Dict[str, str] = {
    "sink": "sink",
    "drop_in_sink": "sink",
    "kitchen_sink": "sink",
    "stove": "stove",
    "cooktop": "stove",
    "burner": "stove",
    "range": "stove",
}


@dataclass(frozen=True)
class SurfaceCandidate:
    name: str
    category: str
    aabb_xy: Bounds2D
    top_z: float
    area: float
    score: float


@dataclass(frozen=True)
class SurfaceObstacle:
    name: str
    category: str
    aabb_xy: Bounds2D
    obstacle_type: str


@dataclass(frozen=True)
class SurfaceAnalysis:
    surface: SurfaceCandidate
    obstacles: Tuple[SurfaceObstacle, ...]
    free_area: float
    approach_edges: Tuple[str, ...]


def is_table_like(category: str) -> bool:
    text = str(category).lower().replace("-", "_").replace(" ", "_")
    if text in _TABLE_LIKE_CATEGORIES:
        return True
    return any(tok in text for tok in ("table", "counter", "desk", "bar", "workbench"))


def is_obstacle_like(category: str) -> str:
    text = str(category).lower().replace("-", "_").replace(" ", "_")
    for key, obs_type in _OBSTACLE_CATEGORIES.items():
        if key in text:
            return obs_type
    return ""


def _normalize(bounds: Bounds2D) -> Bounds2D:
    (x0, y0), (x1, y1) = bounds
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))


def _area(bounds: Bounds2D) -> float:
    (x0, y0), (x1, y1) = _normalize(bounds)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def score_surface(
    aabb_xy: Bounds2D,
    top_z: float,
    category: str,
    min_area: float = 0.15,
    min_short_side: float = 0.25,
    max_height: float = 1.5,
) -> float:
    (x0, y0), (x1, y1) = _normalize(aabb_xy)
    dx = x1 - x0
    dy = y1 - y0
    area = dx * dy
    short_side = min(dx, dy)

    if area < min_area or short_side < min_short_side or top_z > max_height:
        return 0.0

    area_score = min(area / 1.0, 1.0)

    long_side = max(dx, dy)
    aspect = short_side / max(long_side, 1e-6)
    aspect_score = min(aspect / 0.5, 1.0)

    if 0.6 <= top_z <= 1.2:
        height_score = 1.0
    elif 0.3 <= top_z <= 1.5:
        height_score = 0.6
    else:
        height_score = 0.2

    category_bonus = 1.3 if is_table_like(category) else 1.0

    return area_score * 0.4 + aspect_score * 0.2 + height_score * 0.3 + 0.1 * category_bonus


def detect_obstacles_on_surface(
    surface_aabb_xy: Bounds2D,
    surface_top_z: float,
    candidates: Sequence[Dict],
    z_tolerance: float = 0.15,
) -> List[SurfaceObstacle]:
    (sx0, sy0), (sx1, sy1) = _normalize(surface_aabb_xy)
    obstacles = []
    for c in candidates:
        obs_type = is_obstacle_like(c["category"])
        if not obs_type:
            continue
        (cx0, cy0), (cx1, cy1) = _normalize(c["aabb_xy"])
        # Check XY overlap with surface.
        if cx1 <= sx0 or cx0 >= sx1 or cy1 <= sy0 or cy0 >= sy1:
            continue
        # Check Z proximity (obstacle must be near surface top).
        c_top_z = c.get("top_z", surface_top_z)
        if abs(c_top_z - surface_top_z) > z_tolerance:
            continue
        obstacles.append(SurfaceObstacle(
            name=c["name"],
            category=c["category"],
            aabb_xy=_normalize(c["aabb_xy"]),
            obstacle_type=obs_type,
        ))
    return obstacles


def rank_approach_edges(
    surface_aabb_xy: Bounds2D,
    obstacle_aabbs: Sequence[Bounds2D] = (),
    wall_aabbs: Sequence[Bounds2D] = (),
) -> List[str]:
    (sx0, sy0), (sx1, sy1) = _normalize(surface_aabb_xy)
    cx = 0.5 * (sx0 + sx1)
    cy = 0.5 * (sy0 + sy1)

    edges = {
        "x_min": (sx0, cy),
        "x_max": (sx1, cy),
        "y_min": (cx, sy0),
        "y_max": (cx, sy1),
    }

    def _min_clearance(edge_point: Tuple[float, float], blockers: Sequence[Bounds2D]) -> float:
        ex, ey = edge_point
        min_d = float("inf")
        for b in blockers:
            (bx0, by0), (bx1, by1) = _normalize(b)
            dx = max(bx0 - ex, 0.0, ex - bx1)
            dy = max(by0 - ey, 0.0, ey - by1)
            d = (dx * dx + dy * dy) ** 0.5
            min_d = min(min_d, d)
        return min_d

    all_blockers = list(obstacle_aabbs) + list(wall_aabbs)
    if not all_blockers:
        dx = sx1 - sx0
        dy = sy1 - sy0
        if dy >= dx:
            return ["x_min", "x_max", "y_min", "y_max"]
        else:
            return ["y_min", "y_max", "x_min", "x_max"]

    scored = []
    for label, point in edges.items():
        clearance = _min_clearance(point, all_blockers)
        scored.append((label, clearance))

    scored.sort(key=lambda x: -x[1])
    return [label for label, _ in scored]


def analyze_surface(
    name: str,
    category: str,
    aabb_xy: Bounds2D,
    top_z: float,
    scene_objects: Sequence[Dict],
) -> SurfaceAnalysis:
    score = score_surface(aabb_xy, top_z, category)
    surface = SurfaceCandidate(
        name=name,
        category=category,
        aabb_xy=_normalize(aabb_xy),
        top_z=top_z,
        area=_area(aabb_xy),
        score=score,
    )
    obstacles = detect_obstacles_on_surface(aabb_xy, top_z, scene_objects)
    obstacle_area = sum(_area(o.aabb_xy) for o in obstacles)
    free_area = max(0.0, surface.area - obstacle_area)
    obstacle_aabbs = [o.aabb_xy for o in obstacles]
    edges = rank_approach_edges(aabb_xy, obstacle_aabbs=obstacle_aabbs)

    return SurfaceAnalysis(
        surface=surface,
        obstacles=tuple(obstacles),
        free_area=free_area,
        approach_edges=tuple(edges),
    )
