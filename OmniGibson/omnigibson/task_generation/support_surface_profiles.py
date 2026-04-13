from __future__ import annotations

import copy
import json
import math
import os
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]

_PROFILE_FILENAME = "support_surface_profiles_v1.json"
DEFAULT_PROFILE_PATH = os.path.join(os.path.dirname(__file__), _PROFILE_FILENAME)
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    "outputs",
    "support_surface_profiles",
)

_PROFILE_CACHE: Dict[str, dict] = {}


def _as_numpy_array(values: Sequence[float]) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy(), dtype=float)
    return np.asarray(values, dtype=float)


def make_empty_support_surface_profiles_document() -> dict:
    return {
        "version": "support_surface_profiles_v1",
        "frame_convention": {
            "type": "object_local_top_plane_xy",
            "xy_units": "m",
            "quat_convention": "xyzw",
            "notes": (
                "All usable_regions are expressed in the support object's local "
                "root frame. top_plane_z_local is measured in the same frame."
            ),
        },
        "generator_defaults": {
            "region_shape": "rect",
            "review_status": "auto_pending",
        },
        "profiles": {},
    }


def ensure_support_surface_profiles_document(document: Optional[dict]) -> dict:
    base = make_empty_support_surface_profiles_document()
    if document is None:
        return base

    merged = copy.deepcopy(document)
    for key in ("version", "frame_convention", "generator_defaults", "profiles"):
        if key not in merged:
            merged[key] = copy.deepcopy(base[key])
    for nested_key in ("frame_convention", "generator_defaults"):
        if not isinstance(merged[nested_key], dict):
            merged[nested_key] = copy.deepcopy(base[nested_key])
            continue
        for child_key, child_value in base[nested_key].items():
            if child_key not in merged[nested_key]:
                merged[nested_key][child_key] = copy.deepcopy(child_value)
    if not isinstance(merged["profiles"], dict):
        raise ValueError("support surface profiles document must contain dict 'profiles'")
    return merged


def load_support_surface_profiles(path: Optional[str] = None, use_cache: bool = True) -> dict:
    resolved = os.path.abspath(path or DEFAULT_PROFILE_PATH)
    if use_cache and resolved in _PROFILE_CACHE:
        return copy.deepcopy(_PROFILE_CACHE[resolved])

    if not os.path.exists(resolved):
        doc = make_empty_support_surface_profiles_document()
    else:
        with open(resolved, "r", encoding="utf-8") as f:
            doc = ensure_support_surface_profiles_document(json.load(f))

    _PROFILE_CACHE[resolved] = copy.deepcopy(doc)
    return doc


def save_support_surface_profiles(document: dict, path: Optional[str] = None) -> str:
    resolved = os.path.abspath(path or DEFAULT_PROFILE_PATH)
    doc = ensure_support_surface_profiles_document(document)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=True, sort_keys=False)
        f.write("\n")
    _PROFILE_CACHE[resolved] = copy.deepcopy(doc)
    return resolved


def get_support_surface_profile(document: Optional[dict], category: Optional[str], model: Optional[str]) -> Optional[dict]:
    if document is None or not category or not model:
        return None
    profiles = ensure_support_surface_profiles_document(document)["profiles"]
    entry = (profiles.get(category) or {}).get(model)
    return copy.deepcopy(entry) if entry is not None else None


def set_support_surface_profile(document: dict, category: str, model: str, entry: dict) -> dict:
    if not category or not model:
        raise ValueError("category and model are required")
    doc = ensure_support_surface_profiles_document(document)
    doc["profiles"].setdefault(category, {})[model] = copy.deepcopy(entry)
    return doc


def delete_support_surface_profile(document: dict, category: str, model: str) -> dict:
    doc = ensure_support_surface_profiles_document(document)
    category_profiles = doc["profiles"].get(category)
    if not category_profiles:
        return doc
    category_profiles.pop(model, None)
    if not category_profiles:
        doc["profiles"].pop(category, None)
    return doc


def iter_support_surface_profiles(document: Optional[dict]) -> Iterator[Tuple[str, str, dict]]:
    profiles = ensure_support_surface_profiles_document(document)["profiles"]
    for category in sorted(profiles):
        for model in sorted(profiles[category]):
            yield category, model, copy.deepcopy(profiles[category][model])


def make_profile_status(entry: Optional[dict]) -> str:
    if entry is None:
        return "missing"
    if entry.get("review_status") == "rejected":
        return "rejected"
    if not entry.get("candidate_for_generation", False):
        return "non_candidate"
    if not entry.get("usable_regions"):
        return "no_regions"
    return "available"


def normalize_bounds(bounds: Bounds2D) -> Bounds2D:
    (x0, y0), (x1, y1) = bounds
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))


def bounds_area(bounds: Bounds2D) -> float:
    (x0, y0), (x1, y1) = normalize_bounds(bounds)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def region_bounds_xy(region: Optional[dict], *, world: bool = False) -> Optional[Bounds2D]:
    if not region:
        return None
    if world:
        raw = region.get("world_bounds_xy")
        if not raw or len(raw) != 2:
            return None
        return normalize_bounds(
            (
                tuple(float(v) for v in raw[0]),
                tuple(float(v) for v in raw[1]),
            )
        )

    if "xy_min" not in region or "xy_max" not in region:
        return None
    return normalize_bounds(
        (
            tuple(float(v) for v in region["xy_min"]),
            tuple(float(v) for v in region["xy_max"]),
        )
    )


def region_area_m2(region: Optional[dict], *, world: bool = False) -> Optional[float]:
    bounds = region_bounds_xy(region, world=world)
    if bounds is None:
        return None
    raw = region.get("area_m2") if region else None
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return bounds_area(bounds)


def region_span_xy(region: Optional[dict], *, world: bool = False) -> Optional[Tuple[float, float]]:
    if region:
        explicit_key = "world_span_xy_m" if world else "span_xy_m"
        raw = region.get(explicit_key)
        if raw is not None and len(raw) == 2:
            return (max(0.0, float(raw[0])), max(0.0, float(raw[1])))
    bounds = region_bounds_xy(region, world=world)
    if bounds is None:
        return None
    (x0, y0), (x1, y1) = bounds
    return (max(0.0, float(x1) - float(x0)), max(0.0, float(y1) - float(y0)))


def _normalize_min_span_xy(min_span_xy_m: Optional[Sequence[float]]) -> Tuple[float, float]:
    if min_span_xy_m is None:
        return (0.0, 0.0)
    if len(min_span_xy_m) != 2:
        raise ValueError("min_span_xy_m must contain exactly two values")
    return (max(0.0, float(min_span_xy_m[0])), max(0.0, float(min_span_xy_m[1])))


def region_satisfies_min_span(
    region: Optional[dict],
    *,
    world: bool = False,
    min_span_xy_m: Optional[Sequence[float]] = None,
    allow_rotation: bool = True,
) -> bool:
    span = region_span_xy(region, world=world)
    if span is None:
        return False
    min_span_x, min_span_y = _normalize_min_span_xy(min_span_xy_m)
    if allow_rotation:
        span = tuple(sorted(span))
        min_span_x, min_span_y = tuple(sorted((min_span_x, min_span_y)))
    return span[0] + 1e-9 >= min_span_x and span[1] + 1e-9 >= min_span_y


def current_timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def select_dominant_top_plane_height(
    local_hit_heights: Sequence[float],
    band_size_m: float = 0.02,
    min_count_ratio: float = 0.2,
) -> Tuple[Optional[float], int]:
    if band_size_m <= 0.0:
        raise ValueError("band_size_m must be > 0")
    if len(local_hit_heights) == 0:
        return None, 0

    heights = np.asarray(local_hit_heights, dtype=float)
    bucket_counts: Dict[int, int] = {}
    bucket_values: Dict[int, List[float]] = {}
    for value in heights:
        bucket = int(round(value / band_size_m))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        bucket_values.setdefault(bucket, []).append(float(value))

    max_count = max(bucket_counts.values())
    threshold = max(1, int(math.ceil(max_count * float(min_count_ratio))))
    eligible = [bucket for bucket, count in bucket_counts.items() if count >= threshold]
    best_bucket = max(eligible)
    vals = bucket_values[best_bucket]
    return float(sum(vals) / len(vals)), int(bucket_counts[best_bucket])


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []

    for row in range(rows):
        for col in range(cols):
            if not bool(mask[row, col]) or visited[row, col]:
                continue
            q = deque([(row, col)])
            visited[row, col] = True
            coords: List[Tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if visited[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def remove_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1:
        return mask.astype(bool, copy=True)
    cleaned = np.zeros_like(mask, dtype=bool)
    for coords in connected_components(mask):
        if len(coords) >= min_cells:
            cleaned[coords[:, 0], coords[:, 1]] = True
    return cleaned


def largest_true_rectangle(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    rows, cols = mask.shape
    if rows == 0 or cols == 0 or not np.any(mask):
        return None

    heights = np.zeros(cols, dtype=int)
    best = None
    best_area = 0
    best_width = 0
    best_height = 0

    for row in range(rows):
        heights = np.where(mask[row], heights + 1, 0)
        stack: List[Tuple[int, int]] = []
        for col in range(cols + 1):
            current_height = int(heights[col]) if col < cols else 0
            start = col
            while stack and stack[-1][1] > current_height:
                start_idx, height = stack.pop()
                width = col - start_idx
                area = height * width
                if area <= 0:
                    start = start_idx
                    continue
                if (
                    area > best_area
                    or (area == best_area and width > best_width)
                    or (area == best_area and width == best_width and height > best_height)
                ):
                    best_area = area
                    best_width = width
                    best_height = height
                    best = (row + 1 - height, row + 1, start_idx, col)
                start = start_idx
            if current_height > 0 and (not stack or stack[-1][1] < current_height):
                stack.append((start, current_height))
    return best


def maximal_rectangle_cover(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    working = mask.astype(bool, copy=True)
    rectangles: List[Tuple[int, int, int, int]] = []
    while True:
        rect = largest_true_rectangle(working)
        if rect is None:
            break
        r0, r1, c0, c1 = rect
        if r1 <= r0 or c1 <= c0:
            break
        rectangles.append(rect)
        working[r0:r1, c0:c1] = False
    return rectangles


def mask_to_usable_regions(
    mask: np.ndarray,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    *,
    region_id_prefix: str = "region",
    min_component_cells: int = 1,
    min_region_area_m2: float = 0.0,
    source: str = "raycast_maximal_rectangles",
) -> List[dict]:
    cleaned = remove_small_components(mask, min_component_cells)
    x_edges_arr = np.asarray(x_edges, dtype=float)
    y_edges_arr = np.asarray(y_edges, dtype=float)
    if cleaned.shape != (len(y_edges_arr) - 1, len(x_edges_arr) - 1):
        raise ValueError("mask shape must match edge arrays")

    regions: List[dict] = []
    region_idx = 0
    component_labels = np.zeros_like(cleaned, dtype=int)
    for component_idx, coords in enumerate(connected_components(cleaned), start=1):
        component_labels[coords[:, 0], coords[:, 1]] = component_idx

    for r0, r1, c0, c1 in maximal_rectangle_cover(cleaned):
        xy_min = (float(x_edges_arr[c0]), float(y_edges_arr[r0]))
        xy_max = (float(x_edges_arr[c1]), float(y_edges_arr[r1]))
        area_m2 = bounds_area((xy_min, xy_max))
        if area_m2 < float(min_region_area_m2):
            continue
        region_mask = cleaned[r0:r1, c0:c1]
        coverage_ratio = float(region_mask.mean()) if region_mask.size else 0.0
        span_xy = (float(xy_max[0] - xy_min[0]), float(xy_max[1] - xy_min[1]))
        label_patch = component_labels[r0:r1, c0:c1]
        component_ids = [int(v) for v in np.unique(label_patch) if int(v) > 0]
        regions.append(
            {
                "region_id": f"{region_id_prefix}_{region_idx:02d}",
                "shape": "rect",
                "xy_min": [xy_min[0], xy_min[1]],
                "xy_max": [xy_max[0], xy_max[1]],
                "span_xy_m": [round(span_xy[0], 6), round(span_xy[1], 6)],
                "area_m2": round(area_m2, 6),
                "coverage_ratio": round(coverage_ratio, 6),
                "cell_count": int(region_mask.sum()),
                "component_ids": component_ids,
                "confidence": "auto",
                "source": source,
            }
        )
        region_idx += 1
    return regions


def profile_generation_area_m2(
    profile_entry: Optional[dict],
    *,
    min_span_xy_m: Optional[Sequence[float]] = None,
) -> Optional[float]:
    if not profile_entry:
        return None
    regions = list(profile_entry.get("usable_regions", []) or [])
    if not regions:
        return None

    min_span_xy = _normalize_min_span_xy(min_span_xy_m)
    regions = [
        region for region in regions
        if region_satisfies_min_span(region, min_span_xy_m=min_span_xy)
    ]
    if not regions:
        return None

    areas = [region_area_m2(region) for region in regions]
    areas = [float(area) for area in areas if area is not None and float(area) > 0.0]
    if not areas:
        return None
    return max(areas)


def choose_profile_region(
    profile_entry: Optional[dict],
    *,
    world_regions: Optional[Sequence[dict]] = None,
    required_area_m2: float = 0.0,
    min_span_xy_m: Optional[Sequence[float]] = None,
    rng=None,
) -> Optional[dict]:
    if not profile_entry:
        return None

    local_regions = list(profile_entry.get("usable_regions", []) or [])
    if not local_regions:
        return None

    selected_regions = list(world_regions) if world_regions is not None else local_regions
    if len(selected_regions) != len(local_regions):
        raise ValueError("world_regions must align 1:1 with profile usable_regions")

    required = max(0.0, float(required_area_m2))
    min_span_xy = _normalize_min_span_xy(min_span_xy_m)
    candidates = []
    for local_region, selected_region in zip(local_regions, selected_regions):
        if not region_satisfies_min_span(
            selected_region,
            world=world_regions is not None,
            min_span_xy_m=min_span_xy,
        ):
            continue
        region = copy.deepcopy(selected_region)
        area = region_area_m2(region, world=world_regions is not None)
        if area is None:
            continue
        region["_selection_area_m2"] = float(area)
        region["_region_id"] = local_region.get("region_id")
        candidates.append(region)

    if not candidates:
        return None

    sized = [item for item in candidates if item["_selection_area_m2"] + 1e-9 >= required]
    pool = sized if sized else list(candidates)
    if not pool:
        return None

    pool = sorted(pool, key=lambda item: (-item["_selection_area_m2"], item.get("_region_id") or ""))
    if rng is None:
        chosen = pool[0]
    else:
        chosen = pool[int(rng.integers(len(pool)))]

    chosen = copy.deepcopy(chosen)
    chosen.pop("_selection_area_m2", None)
    chosen.pop("_region_id", None)
    return chosen


def _normalize_interval(lo: float, hi: float) -> Tuple[float, float]:
    return (min(lo, hi), max(lo, hi))


def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    a0, a1 = _normalize_interval(*a)
    b0, b1 = _normalize_interval(*b)
    return max(a0, b0) <= min(a1, b1)


def evaluate_region_reachability(
    region_bounds_xy: Bounds2D,
    table_aabb_xy: Bounds2D,
    *,
    reachable_distance_range: Tuple[float, float] = (0.45, 1.10),
    desired_gap_m: float = 0.03,
    robot_radius_m: float = 0.24,
    edge_margin_m: float = 0.08,
    robot_half_extent_xy: Tuple[float, float] = (0.15, 0.15),
) -> dict:
    region = normalize_bounds(region_bounds_xy)
    table = normalize_bounds(table_aabb_xy)
    (rx0, ry0), (rx1, ry1) = region
    (tx0, ty0), (tx1, ty1) = table
    min_reach, max_reach = reachable_distance_range
    normal_offset = float(desired_gap_m) + float(robot_radius_m)
    depth_band = (
        max(0.0, float(min_reach) - normal_offset),
        max(0.0, float(max_reach) - normal_offset),
    )

    tangent_limits = {
        "x_min": (ty0 + edge_margin_m + robot_half_extent_xy[1], ty1 - edge_margin_m - robot_half_extent_xy[1]),
        "x_max": (ty0 + edge_margin_m + robot_half_extent_xy[1], ty1 - edge_margin_m - robot_half_extent_xy[1]),
        "y_min": (tx0 + edge_margin_m + robot_half_extent_xy[0], tx1 - edge_margin_m - robot_half_extent_xy[0]),
        "y_max": (tx0 + edge_margin_m + robot_half_extent_xy[0], tx1 - edge_margin_m - robot_half_extent_xy[0]),
    }

    tangent_spans = {
        "x_min": (ry0, ry1),
        "x_max": (ry0, ry1),
        "y_min": (rx0, rx1),
        "y_max": (rx0, rx1),
    }
    depth_spans = {
        "x_min": (rx0 - tx0, rx1 - tx0),
        "x_max": (tx1 - rx1, tx1 - rx0),
        "y_min": (ry0 - ty0, ry1 - ty0),
        "y_max": (ty1 - ry1, ty1 - ry0),
    }

    reachable_edges: List[str] = []
    for edge_label in ("x_min", "x_max", "y_min", "y_max"):
        tangent_ok = _interval_overlap(tangent_spans[edge_label], tangent_limits[edge_label])
        depth_ok = _interval_overlap(depth_spans[edge_label], depth_band)
        if tangent_ok and depth_ok:
            reachable_edges.append(edge_label)

    return {
        "reachable": bool(reachable_edges),
        "edge_labels": tuple(reachable_edges),
        "reachable_distance_range": [float(min_reach), float(max_reach)],
        "normal_offset_m": round(normal_offset, 6),
        "depth_band_m": [round(depth_band[0], 6), round(depth_band[1], 6)],
    }


def _quat_apply_xyzw(quat_xyzw: Sequence[float], vec_xyz: Sequence[float]) -> np.ndarray:
    quat_arr = _as_numpy_array(quat_xyzw).reshape(-1)
    vec_arr = _as_numpy_array(vec_xyz).reshape(-1)
    x, y, z, w = (float(v) for v in quat_arr[:4])
    vx, vy, vz = (float(v) for v in vec_arr[:3])
    qvec = np.asarray([x, y, z], dtype=float)
    vec = np.asarray([vx, vy, vz], dtype=float)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (w * uv + uuv)


def _quat_conjugate_xyzw(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in _as_numpy_array(quat_xyzw).reshape(-1)[:4])
    return np.asarray([-x, -y, -z, w], dtype=float)


def transform_local_point_to_world(
    point_xyz_local: Sequence[float],
    *,
    position_xyz_world: Sequence[float],
    orientation_xyzw_world: Sequence[float],
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    scaled = _as_numpy_array(point_xyz_local) * _as_numpy_array(scale_xyz)
    rotated = _quat_apply_xyzw(orientation_xyzw_world, scaled)
    return rotated + _as_numpy_array(position_xyz_world)


def transform_world_point_to_local(
    point_xyz_world: Sequence[float],
    *,
    position_xyz_world: Sequence[float],
    orientation_xyzw_world: Sequence[float],
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    translated = _as_numpy_array(point_xyz_world) - _as_numpy_array(position_xyz_world)
    unrotated = _quat_apply_xyzw(_quat_conjugate_xyzw(orientation_xyzw_world), translated)
    scale = _as_numpy_array(scale_xyz)
    if np.any(np.isclose(scale, 0.0)):
        raise ValueError("scale_xyz must be non-zero")
    return unrotated / scale


def transform_region_rect_to_world(
    region: dict,
    *,
    top_plane_z_local: float,
    position_xyz_world: Sequence[float],
    orientation_xyzw_world: Sequence[float],
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict:
    (x0, y0), (x1, y1) = normalize_bounds(
        (
            tuple(float(v) for v in region["xy_min"]),
            tuple(float(v) for v in region["xy_max"]),
        )
    )
    corners_local = (
        (x0, y0, top_plane_z_local),
        (x0, y1, top_plane_z_local),
        (x1, y0, top_plane_z_local),
        (x1, y1, top_plane_z_local),
    )
    corners_world = np.asarray(
        [
            transform_local_point_to_world(
                corner,
                position_xyz_world=position_xyz_world,
                orientation_xyzw_world=orientation_xyzw_world,
                scale_xyz=scale_xyz,
            )
            for corner in corners_local
        ],
        dtype=float,
    )
    x_min = float(np.min(corners_world[:, 0]))
    y_min = float(np.min(corners_world[:, 1]))
    x_max = float(np.max(corners_world[:, 0]))
    y_max = float(np.max(corners_world[:, 1]))
    center = corners_world.mean(axis=0)
    world_bounds = ((x_min, y_min), (x_max, y_max))
    world_span_xy = (abs(float(x_max - x_min)), abs(float(y_max - y_min)))
    world_area_m2 = bounds_area(world_bounds)
    return {
        "region_id": region.get("region_id"),
        "shape": region.get("shape", "rect"),
        "span_xy_m": [
            round(float(v), 6) for v in (region.get("span_xy_m") or region_span_xy(region) or (0.0, 0.0))
        ],
        "world_span_xy_m": [
            round(world_span_xy[0], 6),
            round(world_span_xy[1], 6),
        ],
        "coverage_ratio": float(region.get("coverage_ratio", 1.0)),
        "reachable_edge_labels": list(region.get("reachable_edge_labels", ()) or ()),
        "world_bounds_xy": [[x_min, y_min], [x_max, y_max]],
        "world_center_xyz": [float(center[0]), float(center[1]), float(center[2])],
        "top_plane_z_world": float(np.mean(corners_world[:, 2])),
        "area_m2": float(world_area_m2),
    }


def profile_regions_to_world(
    profile_entry: Optional[dict],
    *,
    position_xyz_world: Sequence[float],
    orientation_xyzw_world: Sequence[float],
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> List[dict]:
    if not profile_entry:
        return []
    top_plane_z_local = float(profile_entry.get("top_plane_z_local", 0.0))
    regions = []
    for region in profile_entry.get("usable_regions", []) or []:
        regions.append(
            transform_region_rect_to_world(
                region,
                top_plane_z_local=top_plane_z_local,
                position_xyz_world=position_xyz_world,
                orientation_xyzw_world=orientation_xyzw_world,
                scale_xyz=scale_xyz,
            )
        )
    return regions


__all__ = [
    "Bounds2D",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PROFILE_PATH",
    "bounds_area",
    "connected_components",
    "current_timestamp_utc",
    "delete_support_surface_profile",
    "choose_profile_region",
    "ensure_support_surface_profiles_document",
    "evaluate_region_reachability",
    "largest_true_rectangle",
    "get_support_surface_profile",
    "iter_support_surface_profiles",
    "load_support_surface_profiles",
    "maximal_rectangle_cover",
    "make_empty_support_surface_profiles_document",
    "make_profile_status",
    "mask_to_usable_regions",
    "normalize_bounds",
    "profile_generation_area_m2",
    "profile_regions_to_world",
    "region_area_m2",
    "region_bounds_xy",
    "region_satisfies_min_span",
    "region_span_xy",
    "remove_small_components",
    "save_support_surface_profiles",
    "select_dominant_top_plane_height",
    "set_support_surface_profile",
    "transform_local_point_to_world",
    "transform_world_point_to_local",
    "transform_region_rect_to_world",
]
