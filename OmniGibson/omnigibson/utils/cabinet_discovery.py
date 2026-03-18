"""Cabinet discovery and manipulation utilities.

Finds cabinets in scenes, estimates interior volumes, opens doors,
and computes packing zones for interior placement tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]

_CABINET_CATEGORIES = frozenset({
    "bottom_cabinet", "bottom_cabinet_no_top", "top_cabinet",
    "metal_bottom_cabinet", "cabinet", "cabinet_base",
    "wardrobe", "locker",
})

_CABINET_CATEGORY_PRIORITY = {
    "bottom_cabinet": 3,
    "bottom_cabinet_no_top": 3,
    "metal_bottom_cabinet": 3,
    "cabinet": 3,
    "cabinet_base": 2,
    "wardrobe": 1,
    "top_cabinet": 3,
    "locker": 1,
}


def is_cabinet_like(category: str) -> bool:
    return category.lower() in _CABINET_CATEGORIES


@dataclass(frozen=True)
class CabinetCompartment:
    """A single openable compartment (drawer or door section) within a cabinet."""
    joint_name: str
    joint_type: str  # "revolute" or "prismatic"
    link_name: str
    interior_bounds_xy: Bounds2D
    interior_bottom_z: float
    interior_top_z: float


@dataclass(frozen=True)
class CabinetCandidate:
    name: str
    category: str
    aabb_xy: Bounds2D
    interior_bounds_xy: Bounds2D
    interior_bottom_z: float
    interior_top_z: float
    exterior_aabb_min: Tuple[float, float, float]
    exterior_aabb_max: Tuple[float, float, float]
    score: float
    compartments: List[CabinetCompartment] = field(default_factory=list)


@dataclass(frozen=True)
class CabinetAnalysis:
    cabinet: CabinetCandidate
    approach_edges: List[str]
    selected_compartment: Optional[CabinetCompartment] = None
    opened_joints: List[str] = field(default_factory=list)


def estimate_interior_bounds(cabinet_obj, wall_thickness=0.03, bottom_offset=0.02):
    """Estimate interior volume of a cabinet from its exterior AABB.

    Shrinks the exterior AABB by wall_thickness on each side and bottom_offset
    from the bottom. The top is left as-is (open top or shelf ceiling).

    Returns:
        (interior_bounds_xy, interior_bottom_z, interior_top_z)
    """
    aabb_min, aabb_max = cabinet_obj.aabb
    ext_min = [float(aabb_min[i]) for i in range(3)]
    ext_max = [float(aabb_max[i]) for i in range(3)]

    interior_xy = (
        (ext_min[0] + wall_thickness, ext_min[1] + wall_thickness),
        (ext_max[0] - wall_thickness, ext_max[1] - wall_thickness),
    )
    interior_bottom_z = ext_min[2] + bottom_offset
    interior_top_z = ext_max[2] - wall_thickness

    return interior_xy, interior_bottom_z, interior_top_z


def discover_compartments(cabinet_obj, wall_thickness=0.03):
    """Detect individual compartments by inspecting openable joints and their child links.

    Each openable joint (revolute door or prismatic drawer) corresponds to one
    compartment.  The compartment bounds are derived from the child link's AABB,
    shrunk by wall_thickness to approximate the usable interior.

    Returns a list of CabinetCompartment, one per openable joint.
    """
    joints = getattr(cabinet_obj, "joints", {}) or {}
    links = getattr(cabinet_obj, "links", {}) or {}
    if not joints or not links:
        return []

    compartments = []
    for joint_name, joint in joints.items():
        try:
            jtype = getattr(joint, "joint_type", "")
            jtype_str = str(jtype).lower()
            if "revolute" not in jtype_str and "prismatic" not in jtype_str:
                continue

            lower = float(joint.lower_limit)
            upper = float(joint.upper_limit)
            if abs(upper - lower) < 0.01:
                continue

            # Find child link.
            child_path = joint.body1
            child_link_name = child_path.split("/")[-1] if child_path else ""
            if child_link_name not in links:
                continue

            child_link = links[child_link_name]
            link_min, link_max = child_link.aabb
            link_x_min, link_y_min, link_z_min = (
                float(link_min[0]), float(link_min[1]), float(link_min[2]),
            )
            link_x_max, link_y_max, link_z_max = (
                float(link_max[0]), float(link_max[1]), float(link_max[2]),
            )

            # Shrink by wall_thickness to get usable interior.
            comp_xy = (
                (link_x_min + wall_thickness, link_y_min + wall_thickness),
                (link_x_max - wall_thickness, link_y_max - wall_thickness),
            )
            comp_bottom_z = link_z_min + wall_thickness
            comp_top_z = link_z_max - wall_thickness

            # Ensure positive dimensions.
            if (comp_xy[1][0] <= comp_xy[0][0]
                    or comp_xy[1][1] <= comp_xy[0][1]
                    or comp_top_z <= comp_bottom_z):
                continue

            compartments.append(CabinetCompartment(
                joint_name=joint_name,
                joint_type="prismatic" if "prismatic" in jtype_str else "revolute",
                link_name=child_link_name,
                interior_bounds_xy=comp_xy,
                interior_bottom_z=comp_bottom_z,
                interior_top_z=comp_top_z,
            ))
        except Exception:
            continue

    return compartments


def open_cabinet_doors(cabinet_obj, og_mod, open_fraction=0.90, joint_names=None):
    """Open revolute/prismatic joints on a cabinet.

    If joint_names is provided, only those joints are opened. Otherwise all
    openable joints are opened.

    Returns list of opened joint names.
    """
    opened = []
    joints = getattr(cabinet_obj, "joints", {}) or {}
    for joint_name, joint in joints.items():
        if joint_names is not None and joint_name not in joint_names:
            continue
        try:
            jtype = getattr(joint, "joint_type", "")
            jtype_str = str(jtype).lower()
            # Only open revolute (doors) and prismatic (drawers) joints.
            if "revolute" not in jtype_str and "prismatic" not in jtype_str:
                continue

            lower = float(joint.lower_limit)
            upper = float(joint.upper_limit)
            if abs(upper - lower) < 0.01:
                continue

            target = lower + open_fraction * (upper - lower)
            joint.set_pos(target)
            opened.append(joint_name)
        except Exception:
            continue

    if opened:
        for _ in range(5):
            og_mod.sim.step()
        print(f"[Pipeline] Opened {len(opened)} cabinet joints: {opened}")
    return opened


def compute_cabinet_packing_zone(interior_bounds_xy, edge_margin_m=0.02):
    """Compute a 2D rectangular packing zone from cabinet interior bounds.

    Returns (red_zone_bounds, surface_bounds) compatible with run_pack_retry_loop.
    """
    (x0, y0), (x1, y1) = interior_bounds_xy
    red_zone = (
        (x0 + edge_margin_m, y0 + edge_margin_m),
        (x1 - edge_margin_m, y1 - edge_margin_m),
    )
    return red_zone, interior_bounds_xy


def _determine_opening_edge(cabinet_obj, scene_objects):
    """Heuristic: the cabinet opening faces the direction with most clearance.

    Checks which side of the cabinet AABB has the fewest nearby objects.
    """
    aabb_min, aabb_max = cabinet_obj.aabb
    cx = 0.5 * (float(aabb_min[0]) + float(aabb_max[0]))
    cy = 0.5 * (float(aabb_min[1]) + float(aabb_max[1]))
    sx0, sy0 = float(aabb_min[0]), float(aabb_min[1])
    sx1, sy1 = float(aabb_max[0]), float(aabb_max[1])

    # Probe distance for checking clearance.
    probe = 0.5
    edge_counts = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}

    cab_name = getattr(cabinet_obj, "name", "")
    for obj in scene_objects:
        name = getattr(obj, "name", "")
        if name == cab_name:
            continue
        try:
            o_min, o_max = obj.aabb
            ox0, oy0, oz0 = float(o_min[0]), float(o_min[1]), float(o_min[2])
            ox1, oy1, oz1 = float(o_max[0]), float(o_max[1]), float(o_max[2])
        except Exception:
            continue

        # Only consider objects at roughly the same height.
        if oz1 < float(aabb_min[2]) or oz0 > float(aabb_max[2]):
            continue

        # Check proximity to each edge.
        if ox1 >= sx0 - probe and ox0 <= sx0 and oy1 >= sy0 and oy0 <= sy1:
            edge_counts["x_min"] += 1
        if ox0 <= sx1 + probe and ox1 >= sx1 and oy1 >= sy0 and oy0 <= sy1:
            edge_counts["x_max"] += 1
        if oy1 >= sy0 - probe and oy0 <= sy0 and ox1 >= sx0 and ox0 <= sx1:
            edge_counts["y_min"] += 1
        if oy0 <= sy1 + probe and oy1 >= sy1 and ox1 >= sx0 and ox0 <= sx1:
            edge_counts["y_max"] += 1

    # Opening is on the side with least obstructions.
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1])
    return [e[0] for e in sorted_edges]


def discover_best_cabinet(env):
    """Find the best cabinet in the loaded scene.

    Prefers bottom_cabinet (reachable by robot) over top_cabinet.
    Scores by interior volume.

    Returns (CabinetAnalysis, cabinet_obj).
    """
    best_candidate = None
    best_obj = None

    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        cat = str(getattr(obj, "category", ""))
        if not is_cabinet_like(cat):
            continue

        try:
            aabb_min, aabb_max = obj.aabb
        except Exception:
            continue

        ext_min = tuple(float(aabb_min[i]) for i in range(3))
        ext_max = tuple(float(aabb_max[i]) for i in range(3))

        # Skip cabinets that are too high for the robot to reach.
        if ext_min[2] > 1.2:
            continue

        interior_xy, interior_bottom_z, interior_top_z = estimate_interior_bounds(obj)

        # Ensure interior has positive dimensions.
        (ix0, iy0), (ix1, iy1) = interior_xy
        if ix1 <= ix0 or iy1 <= iy0 or interior_top_z <= interior_bottom_z:
            continue

        interior_area = (ix1 - ix0) * (iy1 - iy0)
        interior_height = interior_top_z - interior_bottom_z

        # Score: interior volume * category priority.
        priority = _CABINET_CATEGORY_PRIORITY.get(cat, 0)
        score = interior_area * interior_height * (1.0 + 0.5 * priority)

        compartments = discover_compartments(obj)

        candidate = CabinetCandidate(
            name=name, category=cat,
            aabb_xy=((ext_min[0], ext_min[1]), (ext_max[0], ext_max[1])),
            interior_bounds_xy=interior_xy,
            interior_bottom_z=interior_bottom_z,
            interior_top_z=interior_top_z,
            exterior_aabb_min=ext_min,
            exterior_aabb_max=ext_max,
            score=score,
            compartments=compartments,
        )

        if best_candidate is None or score > best_candidate.score:
            best_candidate = candidate
            best_obj = obj

    if best_candidate is None:
        raise RuntimeError("No suitable cabinet found in scene.")

    approach_edges = _determine_opening_edge(best_obj, env.scene.objects)

    analysis = CabinetAnalysis(
        cabinet=best_candidate,
        approach_edges=approach_edges,
    )
    return analysis, best_obj


def place_robot_facing_cabinet(robot, cabinet_analysis, floor_z, standoff_m=0.75):
    """Place the robot facing the cabinet opening.

    Positions the robot at standoff distance from the opening edge,
    centered on the cabinet, facing inward.

    Returns (position, orientation, edge_label).
    """
    import omnigibson.utils.transform_utils as T
    import torch as th

    cab = cabinet_analysis.cabinet
    (sx0, sy0), (sx1, sy1) = cab.aabb_xy
    cx = 0.5 * (sx0 + sx1)
    cy = 0.5 * (sy0 + sy1)

    opening_edge = cabinet_analysis.approach_edges[0] if cabinet_analysis.approach_edges else "y_min"

    if opening_edge == "x_min":
        pos = (sx0 - standoff_m, cy, floor_z)
        yaw = 0.0  # Face +x
    elif opening_edge == "x_max":
        pos = (sx1 + standoff_m, cy, floor_z)
        yaw = math.pi  # Face -x
    elif opening_edge == "y_min":
        pos = (cx, sy0 - standoff_m, floor_z)
        yaw = math.pi / 2  # Face +y
    elif opening_edge == "y_max":
        pos = (cx, sy1 + standoff_m, floor_z)
        yaw = -math.pi / 2  # Face -y
    else:
        pos = (cx, sy0 - standoff_m, floor_z)
        yaw = math.pi / 2

    quat = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
    orientation = tuple(float(q) for q in quat)

    robot.set_position_orientation(position=pos, orientation=orientation)
    return pos, orientation, opening_edge
