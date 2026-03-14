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


def _overlaps(a: Bounds2D, b: Bounds2D) -> bool:
    """Check if two 2D AABBs overlap (strict inequality — touching doesn't count)."""
    (a0x, a0y), (a1x, a1y) = _normalize(a)
    (b0x, b0y), (b1x, b1y) = _normalize(b)
    return a0x < b1x and a1x > b0x and a0y < b1y and a1y > b0y


def compute_robot_placement_box(
    edge_label: str,
    surface_aabb_xy: Bounds2D,
    robot_footprint_xy: Tuple[float, float] = (0.35, 0.35),
    edge_gap_m: float = 0.03,
    tangent_offset: float = 0.0,
) -> Bounds2D:
    """Compute the 2D AABB where the robot base would be placed for a given edge.

    The robot is a robot-sized box placed just outside the table edge (normal
    direction) with a gap, centered along the edge's tangent direction plus
    an optional offset.

    Args:
        edge_label: One of "x_min", "x_max", "y_min", "y_max".
        surface_aabb_xy: The table's 2D bounding box.
        robot_footprint_xy: (width_x, width_y) of the robot base footprint.
        edge_gap_m: Gap between robot and table edge.
        tangent_offset: Offset along the edge from the center (for scanning).

    Returns:
        2D AABB of the robot placement region.
    """
    (sx0, sy0), (sx1, sy1) = _normalize(surface_aabb_xy)
    half_rx = robot_footprint_xy[0] / 2.0
    half_ry = robot_footprint_xy[1] / 2.0
    cx = 0.5 * (sx0 + sx1)
    cy = 0.5 * (sy0 + sy1)

    if edge_label == "x_min":
        robot_cx = sx0 - half_rx - edge_gap_m
        robot_cy = cy + tangent_offset
        return ((robot_cx - half_rx, robot_cy - half_ry), (robot_cx + half_rx, robot_cy + half_ry))
    elif edge_label == "x_max":
        robot_cx = sx1 + half_rx + edge_gap_m
        robot_cy = cy + tangent_offset
        return ((robot_cx - half_rx, robot_cy - half_ry), (robot_cx + half_rx, robot_cy + half_ry))
    elif edge_label == "y_min":
        robot_cx = cx + tangent_offset
        robot_cy = sy0 - half_ry - edge_gap_m
        return ((robot_cx - half_rx, robot_cy - half_ry), (robot_cx + half_rx, robot_cy + half_ry))
    elif edge_label == "y_max":
        robot_cx = cx + tangent_offset
        robot_cy = sy1 + half_ry + edge_gap_m
        return ((robot_cx - half_rx, robot_cy - half_ry), (robot_cx + half_rx, robot_cy + half_ry))
    else:
        raise ValueError(f"Unsupported edge_label: {edge_label}")


def _build_scan_offsets(edge_length: float, step: float = 0.15) -> Tuple[float, ...]:
    """Build scan offsets that cover the full edge length, centered at 0."""
    offsets = [0.0]
    half = edge_length / 2.0
    d = step
    while d < half:
        offsets.append(d)
        offsets.append(-d)
        d += step
    return tuple(offsets)


def check_edge_reachability(
    surface_aabb_xy: Bounds2D,
    scene_object_aabbs: Sequence[Bounds2D],
    surface_name: str = "",
    robot_footprint_xy: Tuple[float, float] = (0.35, 0.35),
    edge_gap_m: float = 0.03,
) -> List[str]:
    """Return edges where the robot can be placed without colliding with scene objects.

    For each of the four edges, samples several positions along the edge and
    checks whether a robot-sized box at that position overlaps any scene object.
    An edge is reachable if at least one position is collision-free.

    Args:
        surface_aabb_xy: The table surface AABB.
        scene_object_aabbs: AABBs of all other scene objects (walls, furniture, etc.).
        surface_name: Name of this surface (for logging only).
        robot_footprint_xy: (width_x, width_y) of the robot base.
        edge_gap_m: Gap between robot and table edge.

    Returns:
        List of reachable edge labels, ordered by preference (long-side edges first).
    """
    (sx0, sy0), (sx1, sy1) = _normalize(surface_aabb_xy)
    dx = sx1 - sx0
    dy = sy1 - sy0
    # Prefer approaching along the long side (robot faces the short side).
    if dy >= dx:
        edge_order = ["x_min", "x_max", "y_min", "y_max"]
    else:
        edge_order = ["y_min", "y_max", "x_min", "x_max"]

    # Build scan offsets scaled to each edge's length.
    edge_lengths = {
        "x_min": dy, "x_max": dy,
        "y_min": dx, "y_max": dx,
    }

    reachable = []
    for edge in edge_order:
        offsets = _build_scan_offsets(edge_lengths[edge])
        found_clear = False
        for offset in offsets:
            robot_box = compute_robot_placement_box(
                edge, surface_aabb_xy, robot_footprint_xy, edge_gap_m,
                tangent_offset=offset,
            )
            blocked = False
            for obj_aabb in scene_object_aabbs:
                if _overlaps(robot_box, obj_aabb):
                    blocked = True
                    break
            if not blocked:
                found_clear = True
                break
        if found_clear:
            reachable.append(edge)
    return reachable


def rank_approach_edges(
    surface_aabb_xy: Bounds2D,
    obstacle_aabbs: Sequence[Bounds2D] = (),
    wall_aabbs: Sequence[Bounds2D] = (),
    reachable_edges: Optional[Sequence[str]] = None,
) -> List[str]:
    """Rank approach edges by clearance, optionally filtering to reachable ones.

    If *reachable_edges* is provided, only those edges are considered.
    """
    (sx0, sy0), (sx1, sy1) = _normalize(surface_aabb_xy)
    cx = 0.5 * (sx0 + sx1)
    cy = 0.5 * (sy0 + sy1)

    all_edge_labels = ["x_min", "x_max", "y_min", "y_max"]
    if reachable_edges is not None:
        all_edge_labels = [e for e in all_edge_labels if e in reachable_edges]
    if not all_edge_labels:
        return []

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
            default = ["x_min", "x_max", "y_min", "y_max"]
        else:
            default = ["y_min", "y_max", "x_min", "x_max"]
        return [e for e in default if e in all_edge_labels]

    scored = []
    for label in all_edge_labels:
        clearance = _min_clearance(edges[label], all_blockers)
        scored.append((label, clearance))

    scored.sort(key=lambda x: -x[1])
    return [label for label, _ in scored]


def analyze_surface(
    name: str,
    category: str,
    aabb_xy: Bounds2D,
    top_z: float,
    scene_objects: Sequence[Dict],
    scene_object_aabbs: Optional[Sequence[Bounds2D]] = None,
    robot_footprint_xy: Tuple[float, float] = (0.35, 0.35),
    edge_gap_m: float = 0.03,
) -> SurfaceAnalysis:
    """Analyze a surface for suitability, including robot reachability.

    Args:
        name: Surface object name.
        category: Surface category string.
        aabb_xy: 2D bounding box of the surface.
        top_z: Z-height of the surface top.
        scene_objects: List of dicts with name/category/aabb_xy/top_z for obstacle detection.
        scene_object_aabbs: AABBs of *other* scene objects (not this surface) for
            robot reachability checks.  If None, reachability is not checked (all
            edges considered reachable).
        robot_footprint_xy: Robot base footprint (width_x, width_y).
        edge_gap_m: Gap between robot and table edge.
    """
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

    # Determine which edges are reachable by the robot.
    reachable = None
    if scene_object_aabbs is not None:
        reachable = check_edge_reachability(
            surface_aabb_xy=aabb_xy,
            scene_object_aabbs=scene_object_aabbs,
            surface_name=name,
            robot_footprint_xy=robot_footprint_xy,
            edge_gap_m=edge_gap_m,
        )
        if not reachable:
            # No reachable edge → surface is unusable, override score to 0.
            surface = SurfaceCandidate(
                name=name,
                category=category,
                aabb_xy=_normalize(aabb_xy),
                top_z=top_z,
                area=_area(aabb_xy),
                score=0.0,
            )

    edges = rank_approach_edges(
        aabb_xy,
        obstacle_aabbs=obstacle_aabbs,
        reachable_edges=reachable,
    )

    return SurfaceAnalysis(
        surface=surface,
        obstacles=tuple(obstacles),
        free_area=free_area,
        approach_edges=tuple(edges),
    )
