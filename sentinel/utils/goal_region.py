from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SPHERE_GOAL_FAMILIES = {
    "table",
    "liquid_transport",
    "stack_same",
    "stack_flat",
    "lid_transport_food",
    "lid_transport_liquid",
}

GOAL_REGION_COLOR_RGBA = (0.10, 0.80, 0.20, 0.60)
GOAL_REGION_DISTANCE_SCALE = 1.5
GOAL_REGION_RADIUS_SCALE = 0.45

# Random-placement reach polygon in robot-local frame (Franka on stand).
# x = forward distance from base, y = lateral; tuned to keep the sphere
# inside the arm's typical IK-feasible workspace at table height.
GOAL_RANDOM_X_RANGE_M = (0.30, 0.65)
GOAL_RANDOM_Y_RANGE_M = (-0.40, 0.40)
# z range is anchored to the support top, not robot frame (so the
# sphere always sits above the table regardless of robot mount height).
GOAL_RANDOM_Z_ABOVE_SUPPORT_RANGE_M = (0.08, 0.30)
# Reject a sample if it overlaps the pack with less than this margin
# (in target-widths). Keeps the sphere visibly separate from the stack.
GOAL_RANDOM_PACK_CLEARANCE_TARGET_WIDTHS = 1.2
GOAL_RANDOM_MAX_TRIES = 50

FAMILY_ALIASES = {
    "clutter": "table",
    "cluttered_env": "table",
    "table": "table",
    "liquid_transport": "liquid_transport",
    "wet_transport": "wet_transport",
    "stack_same": "stack_same",
    "stack_flat": "stack_flat",
    "transfer": "transfer",
    "food_transfer": "transfer",
    "lid_transport_food": "lid_transport_food",
    "lid_transport_liquid": "lid_transport_liquid",
    "cabinet_pickup": "cabinet_pickup",
}


@dataclass(frozen=True)
class GoalRegionEntities:
    family: str
    target_name: str
    support_name: str
    pack_object_names: tuple[str, ...]


@dataclass(frozen=True)
class GoalRegionSpec:
    mode: str
    shape: str
    family: str
    target_name: str
    support_name: str
    marker_name: str
    center_world: tuple[float, float, float]
    radius_m: float
    color_rgba: tuple[float, float, float, float]
    target_width_m: float
    anchor_local_xy: tuple[float, float]
    pack_bbox_robot_local_xy: tuple[tuple[float, float], tuple[float, float]]
    support_bounds_robot_local_xy: tuple[tuple[float, float], tuple[float, float]]
    clamped_to_support_bounds: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "shape": self.shape,
            "family": self.family,
            "target_name": self.target_name,
            "support_name": self.support_name,
            "marker_name": self.marker_name,
            "center_world": [float(v) for v in self.center_world],
            "radius_m": float(self.radius_m),
            "color_rgba": [float(v) for v in self.color_rgba],
            "target_width_m": float(self.target_width_m),
            "anchor_local_xy": [float(v) for v in self.anchor_local_xy],
            "pack_bbox_robot_local_xy": [
                [float(v) for v in self.pack_bbox_robot_local_xy[0]],
                [float(v) for v in self.pack_bbox_robot_local_xy[1]],
            ],
            "support_bounds_robot_local_xy": [
                [float(v) for v in self.support_bounds_robot_local_xy[0]],
                [float(v) for v in self.support_bounds_robot_local_xy[1]],
            ],
            "clamped_to_support_bounds": bool(self.clamped_to_support_bounds),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GoalRegionSpec":
        return cls(
            mode=str(payload["mode"]),
            shape=str(payload["shape"]),
            family=str(payload["family"]),
            target_name=str(payload["target_name"]),
            support_name=str(payload["support_name"]),
            marker_name=str(payload["marker_name"]),
            center_world=tuple(float(v) for v in payload["center_world"][:3]),
            radius_m=float(payload["radius_m"]),
            color_rgba=tuple(float(v) for v in payload["color_rgba"][:4]),
            target_width_m=float(payload["target_width_m"]),
            anchor_local_xy=tuple(float(v) for v in payload["anchor_local_xy"][:2]),
            pack_bbox_robot_local_xy=(
                tuple(float(v) for v in payload["pack_bbox_robot_local_xy"][0][:2]),
                tuple(float(v) for v in payload["pack_bbox_robot_local_xy"][1][:2]),
            ),
            support_bounds_robot_local_xy=(
                tuple(float(v) for v in payload["support_bounds_robot_local_xy"][0][:2]),
                tuple(float(v) for v in payload["support_bounds_robot_local_xy"][1][:2]),
            ),
            clamped_to_support_bounds=bool(payload.get("clamped_to_support_bounds", False)),
        )


def canonicalize_family(family: str | None) -> str:
    if not family:
        raise ValueError("family is required")
    key = str(family).strip().lower()
    if key not in FAMILY_ALIASES:
        raise ValueError(f"Unsupported family: {family}")
    return FAMILY_ALIASES[key]


def infer_family_from_diagnostics(diagnostics: dict[str, Any]) -> str:
    pipeline = diagnostics.get("pipeline")
    selection = diagnostics.get("selection") or {}
    if pipeline and str(pipeline).strip().lower() != "lid_transport":
        return canonicalize_family(str(pipeline))
    mode = str(selection.get("mode") or "").strip().lower()
    if mode == "same":
        return "stack_same"
    if mode == "flat":
        return "stack_flat"
    if selection.get("food_synset") and selection.get("dest_synset"):
        return "transfer"
    if selection.get("container_synset"):
        if selection.get("system_name"):
            return "lid_transport_liquid"
        return "lid_transport_food"
    if selection.get("system_name"):
        return "liquid_transport"
    return "table"


def family_uses_goal_region(family: str) -> bool:
    return canonicalize_family(family) in SPHERE_GOAL_FAMILIES


def goal_region_marker_name(target_name: str) -> str:
    return f"goal_region__{target_name}"


def _normalize_synset_label(synset: str, fallback: str) -> str:
    if not synset:
        return fallback
    if ".n." in synset:
        synset = synset.split(".n.", 1)[0]
    return synset.replace("_", " ")


def _support_label(diagnostics: dict[str, Any], scene_info: dict[str, Any]) -> str:
    init_info = scene_info.get("objects_info", {}).get("init_info", {})
    surface_name = str(diagnostics.get("surface") or "")
    return (
        str(init_info.get(surface_name, {}).get("args", {}).get("category") or "")
        or surface_name
        or "table"
    )


def build_task_prompt(scene_info: dict[str, Any], diagnostics: dict[str, Any], *, goal_region: dict[str, Any] | GoalRegionSpec | None = None) -> str:
    family = infer_family_from_diagnostics(diagnostics)
    selection = diagnostics.get("selection") or {}
    support_label = _support_label(diagnostics, scene_info)
    use_goal_region = goal_region is not None and family_uses_goal_region(family)

    if family == "table":
        target = _normalize_synset_label(str(selection.get("target_synset", "")), "object")
        if use_goal_region:
            return f"Pick up the {target} on the {support_label}, then move it into the green goal sphere on the left side of the object pack."
        return f"Pick up the {target} on the {support_label}."
    if family == "liquid_transport":
        target = _normalize_synset_label(str(selection.get("target_synset", "")), "container")
        if use_goal_region:
            return f"Pick up the filled {target} on the {support_label}, then move it into the green goal sphere on the left side of the object pack."
        return f"Pick up the filled {target} on the {support_label}."
    if family == "stack_same":
        target = _normalize_synset_label(str(selection.get("target_synset", "")), "object")
        if use_goal_region:
            return f"Pick up the bottom {target} from the stack, then move it into the green goal sphere on the left side of the stack."
        return f"Pick up the bottom {target} from the stack and lift it upward."
    if family == "stack_flat":
        if use_goal_region:
            return "Pick up the flat object from under the stack, then move it into the green goal sphere on the left side of the stack."
        return "Pick up the flat object from under the stack and lift it upward."
    if family == "lid_transport_food":
        container = _normalize_synset_label(str(selection.get("container_synset", "")), "container")
        if use_goal_region:
            return f"Place the lid on the {container}, then move the {container} into the green goal sphere on the left side of the container."
        return f"Place the lid on the {container}, then lift the {container} upward."
    if family == "lid_transport_liquid":
        container = _normalize_synset_label(str(selection.get("container_synset", "")), "container")
        if use_goal_region:
            return f"Place the lid on the filled {container}, then move the filled {container} into the green goal sphere on the left side of the container."
        return f"Place the lid on the filled {container}, then lift the filled {container} upward."
    if family == "transfer":
        food = _normalize_synset_label(str(selection.get("food_synset", "")), "food")
        source = _normalize_synset_label(str(selection.get("source_synset", "")), "source")
        dest = _normalize_synset_label(str(selection.get("dest_synset", "")), "destination")
        return f"Transfer the {food} from the {source} to the {dest}."
    if family == "wet_transport":
        target = _normalize_synset_label(str(selection.get("target_synset", "")), "container")
        return (
            f"Pick up the filled {target} on the {support_label} and carry it "
            "across the table without passing over any of the water-sensitive items."
        )
    raise ValueError(f"Unsupported family for prompt build: {family}")


def _extract_reference_name(goal_conditions: Any, predicate: str) -> str | None:
    predicate = predicate.lower()
    if isinstance(goal_conditions, list):
        for node in goal_conditions:
            result = _extract_reference_name(node, predicate)
            if result:
                return result
        return None
    if not isinstance(goal_conditions, dict):
        return None
    if str(goal_conditions.get("predicate", "")).lower() == predicate:
        reference = goal_conditions.get("reference")
        return str(reference) if reference else None
    for key in ("terms", "term"):
        node = goal_conditions.get(key)
        if isinstance(node, list):
            for child in node:
                result = _extract_reference_name(child, predicate)
                if result:
                    return result
        elif isinstance(node, dict):
            result = _extract_reference_name(node, predicate)
            if result:
                return result
    return None


def _extract_subject_names(goal_conditions: Any, predicate: str) -> list[str]:
    predicate = predicate.lower()
    names: list[str] = []
    if isinstance(goal_conditions, list):
        for node in goal_conditions:
            names.extend(_extract_subject_names(node, predicate))
        return names
    if not isinstance(goal_conditions, dict):
        return names
    if str(goal_conditions.get("predicate", "")).lower() == predicate:
        subject = goal_conditions.get("subject")
        if subject:
            names.append(str(subject))
    for key in ("terms", "term"):
        node = goal_conditions.get(key)
        if isinstance(node, list):
            for child in node:
                names.extend(_extract_subject_names(child, predicate))
        elif isinstance(node, dict):
            names.extend(_extract_subject_names(node, predicate))
    return names


def _find_scene_objects_by_category(scene_info: dict[str, Any], category: str) -> list[str]:
    init_info = scene_info.get("objects_info", {}).get("init_info", {})
    names = []
    for scene_name, obj_info in init_info.items():
        if str(obj_info.get("args", {}).get("category") or "") == category:
            names.append(scene_name)
    return sorted(names)


def _first_scene_object_by_synset(scene_info: dict[str, Any], synset: str) -> str | None:
    if not synset:
        return None
    category = synset.split(".n.", 1)[0]
    matches = _find_scene_objects_by_category(scene_info, category)
    return matches[0] if matches else None


def resolve_goal_region_entities(scene_info: dict[str, Any], diagnostics: dict[str, Any]) -> GoalRegionEntities | None:
    family = infer_family_from_diagnostics(diagnostics)
    support_name = str(diagnostics.get("surface") or "")
    goal_conditions = diagnostics.get("goal_conditions")
    selection = diagnostics.get("selection") or {}

    if family == "transfer":
        return None

    if family in {"table", "liquid_transport"}:
        target_name = None
        for entry in diagnostics.get("active_object_summary", []) or []:
            if str(entry.get("role", "")) == "target":
                target_name = str(entry.get("scene_object_name") or "")
                break
        target_name = target_name or _extract_reference_name(goal_conditions, "grasping") or _first_scene_object_by_synset(
            scene_info,
            str(selection.get("target_synset", "")),
        )
        pack_names = tuple(
            str(entry.get("scene_object_name"))
            for entry in diagnostics.get("active_object_summary", []) or []
            if entry.get("scene_object_name")
        )
        if not target_name or not pack_names:
            return None
        return GoalRegionEntities(family=family, target_name=target_name, support_name=support_name, pack_object_names=pack_names)

    if family in {"stack_same", "stack_flat"}:
        target_name = _extract_reference_name(goal_conditions, "grasping") or _first_scene_object_by_synset(
            scene_info,
            str(selection.get("target_synset", "")),
        )
        stack_names = _extract_subject_names(goal_conditions, "ontop")
        pack_names = tuple(dict.fromkeys([target_name, *stack_names])) if target_name else tuple(dict.fromkeys(stack_names))
        pack_names = tuple(name for name in pack_names if name)
        if not target_name or not pack_names:
            return None
        return GoalRegionEntities(family=family, target_name=target_name, support_name=support_name, pack_object_names=pack_names)

    if family in {"lid_transport_food", "lid_transport_liquid"}:
        target_name = (
            _extract_reference_name(goal_conditions, "grasping")
            or _extract_reference_name(goal_conditions, "ontop")
            or _first_scene_object_by_synset(scene_info, str(selection.get("container_synset", "")))
        )
        if not target_name:
            return None
        return GoalRegionEntities(family=family, target_name=target_name, support_name=support_name, pack_object_names=(target_name,))

    return None


def _quat_inverse(quat_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = [float(v) for v in quat_xyzw[:4]]
    norm = x * x + y * y + z * z + w * w
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (-x / norm, -y / norm, -z / norm, w / norm)


def _quat_multiply(q1: Sequence[float], q2: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = [float(v) for v in q1[:4]]
    x2, y2, z2, w2 = [float(v) for v in q2[:4]]
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_rotate(quat_xyzw: Sequence[float], vec_xyz: Sequence[float]) -> tuple[float, float, float]:
    q_vec = (float(vec_xyz[0]), float(vec_xyz[1]), float(vec_xyz[2]), 0.0)
    return _quat_multiply(_quat_multiply(quat_xyzw, q_vec), _quat_inverse(quat_xyzw))[:3]


def _world_to_local(robot_pos: Sequence[float], robot_quat: Sequence[float], point_world: Sequence[float]) -> tuple[float, float, float]:
    delta = (
        float(point_world[0]) - float(robot_pos[0]),
        float(point_world[1]) - float(robot_pos[1]),
        float(point_world[2]) - float(robot_pos[2]),
    )
    return _quat_rotate(_quat_inverse(robot_quat), delta)


def _local_xy_to_world(robot_pos: Sequence[float], robot_quat: Sequence[float], x_local: float, y_local: float) -> tuple[float, float]:
    rotated = _quat_rotate(robot_quat, (float(x_local), float(y_local), 0.0))
    return (
        float(robot_pos[0]) + float(rotated[0]),
        float(robot_pos[1]) + float(rotated[1]),
    )


def _aabb_bounds(obj) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    aabb_min, aabb_max = obj.aabb
    return (
        (float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])),
        (float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])),
    )


def _aabb_center(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float, float]:
    aabb_min, aabb_max = bounds
    return (
        0.5 * (aabb_min[0] + aabb_max[0]),
        0.5 * (aabb_min[1] + aabb_max[1]),
        0.5 * (aabb_min[2] + aabb_max[2]),
    )


def _aabb_dims(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float, float]:
    aabb_min, aabb_max = bounds
    return (
        max(0.0, aabb_max[0] - aabb_min[0]),
        max(0.0, aabb_max[1] - aabb_min[1]),
        max(0.0, aabb_max[2] - aabb_min[2]),
    )


def _local_xy_bbox_for_objects(robot_pos: Sequence[float], robot_quat: Sequence[float], objs: Sequence[Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for obj in objs:
        aabb_min, aabb_max = _aabb_bounds(obj)
        for x in (aabb_min[0], aabb_max[0]):
            for y in (aabb_min[1], aabb_max[1]):
                for z in (aabb_min[2], aabb_max[2]):
                    local = _world_to_local(robot_pos, robot_quat, (x, y, z))
                    xs.append(float(local[0]))
                    ys.append(float(local[1]))
    if not xs or not ys:
        raise ValueError("No local bbox corners computed for pack objects")
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def _local_xy_bounds_from_world_bounds(robot_pos: Sequence[float], robot_quat: Sequence[float], bounds_xy: Sequence[Sequence[float]], z_world: float) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for x in (float(bounds_xy[0][0]), float(bounds_xy[1][0])):
        for y in (float(bounds_xy[0][1]), float(bounds_xy[1][1])):
            local = _world_to_local(robot_pos, robot_quat, (x, y, float(z_world)))
            xs.append(float(local[0]))
            ys.append(float(local[1]))
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def _support_bounds_world_xy(diagnostics: dict[str, Any], support_obj) -> tuple[tuple[float, float], tuple[float, float]]:
    support_selection = diagnostics.get("support_selection") or {}
    bounds = support_selection.get("result_world_bounds_xy") or support_selection.get("surface_bounds_xy")
    if isinstance(bounds, list) and len(bounds) == 2:
        return (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
        )
    aabb_min, aabb_max = _aabb_bounds(support_obj)
    return ((aabb_min[0], aabb_min[1]), (aabb_max[0], aabb_max[1]))


def build_goal_region_spec(
    env,
    diagnostics: dict[str, Any],
    *,
    family: str,
    target_name: str,
    support_name: str,
    pack_object_names: Sequence[str],
    marker_name: str | None = None,
    color_rgba: Sequence[float] = GOAL_REGION_COLOR_RGBA,
    distance_scale: float = GOAL_REGION_DISTANCE_SCALE,
    radius_scale: float = GOAL_REGION_RADIUS_SCALE,
    rng=None,
) -> GoalRegionSpec:
    """Build the green goal-sphere spec.

    If ``rng`` (a numpy Generator) is provided, the sphere centre is
    sampled uniformly in the robot-local reach polygon and at a random
    z above the support top, rejecting samples that overlap the pack.
    If ``rng`` is None, the legacy deterministic placement (1.5×
    target-width to the left of the pack) is used.
    """
    family = canonicalize_family(family)
    if family not in SPHERE_GOAL_FAMILIES:
        raise ValueError(f"Family {family} does not use a goal region sphere")

    robot = env.robots[0]
    target_obj = env.scene.object_registry("name", target_name)
    support_obj = env.scene.object_registry("name", support_name)
    pack_objs = [env.scene.object_registry("name", name) for name in pack_object_names]
    pack_objs = [obj for obj in pack_objs if obj is not None]
    if target_obj is None or support_obj is None or not pack_objs:
        raise ValueError(
            f"Could not resolve goal region runtime objects: target={target_name}, support={support_name}, pack={pack_object_names}"
        )

    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos = [float(v) for v in robot_pos[:3]]
    robot_quat = [float(v) for v in robot_quat[:4]]

    target_bounds = _aabb_bounds(target_obj)
    target_center_world = _aabb_center(target_bounds)
    target_center_local = _world_to_local(robot_pos, robot_quat, target_center_world)
    target_dx, target_dy, target_dz = _aabb_dims(target_bounds)
    target_width = max(float(target_dx), float(target_dy), 1e-4)
    target_half_h = max(0.5 * float(target_dz), 1e-4)
    radius_m = max(float(radius_scale) * target_width, 1e-4)

    pack_local_bounds = _local_xy_bbox_for_objects(robot_pos, robot_quat, pack_objs)
    support_aabb_min, support_aabb_max = _aabb_bounds(support_obj)
    support_top_z = float(support_aabb_max[2])
    support_world_xy = _support_bounds_world_xy(diagnostics, support_obj)
    support_local_bounds = _local_xy_bounds_from_world_bounds(robot_pos, robot_quat, support_world_xy, support_top_z)

    clamped = False
    if rng is None:
        # Legacy deterministic placement.
        desired_x = float(target_center_local[0])
        desired_y = float(pack_local_bounds[1][1]) + float(distance_scale) * target_width
        anchor_x = desired_x
        anchor_y = desired_y
        center_x, center_y = _local_xy_to_world(robot_pos, robot_quat, anchor_x, anchor_y)
        center_z = support_top_z + target_half_h
    else:
        # Random placement inside the robot's reach polygon. Reject
        # samples that overlap the object pack to avoid burying the
        # sphere in the stack.
        pack_min_xy = pack_local_bounds[0]
        pack_max_xy = pack_local_bounds[1]
        pack_cx = 0.5 * (float(pack_min_xy[0]) + float(pack_max_xy[0]))
        pack_cy = 0.5 * (float(pack_min_xy[1]) + float(pack_max_xy[1]))
        pack_half_w = max(
            float(pack_max_xy[0]) - pack_cx,
            float(pack_max_xy[1]) - pack_cy,
            target_width,
        )
        clear_dist = pack_half_w + GOAL_RANDOM_PACK_CLEARANCE_TARGET_WIDTHS * target_width

        anchor_x = float(target_center_local[0])
        anchor_y = float(pack_local_bounds[1][1]) + float(distance_scale) * target_width
        for _ in range(GOAL_RANDOM_MAX_TRIES):
            cand_x = float(rng.uniform(*GOAL_RANDOM_X_RANGE_M))
            cand_y = float(rng.uniform(*GOAL_RANDOM_Y_RANGE_M))
            dx = cand_x - pack_cx
            dy = cand_y - pack_cy
            if (dx * dx + dy * dy) ** 0.5 >= clear_dist:
                anchor_x, anchor_y = cand_x, cand_y
                break
        center_x, center_y = _local_xy_to_world(robot_pos, robot_quat, anchor_x, anchor_y)
        center_z = support_top_z + float(rng.uniform(*GOAL_RANDOM_Z_ABOVE_SUPPORT_RANGE_M))

    return GoalRegionSpec(
        mode="held_intersection",
        shape="sphere",
        family=family,
        target_name=target_name,
        support_name=support_name,
        marker_name=str(marker_name or goal_region_marker_name(target_name)),
        center_world=(float(center_x), float(center_y), float(center_z)),
        radius_m=float(radius_m),
        color_rgba=tuple(float(v) for v in color_rgba[:4]),
        target_width_m=float(target_width),
        anchor_local_xy=(float(anchor_x), float(anchor_y)),
        pack_bbox_robot_local_xy=(
            (float(pack_local_bounds[0][0]), float(pack_local_bounds[0][1])),
            (float(pack_local_bounds[1][0]), float(pack_local_bounds[1][1])),
        ),
        support_bounds_robot_local_xy=(
            (float(support_local_bounds[0][0]), float(support_local_bounds[0][1])),
            (float(support_local_bounds[1][0]), float(support_local_bounds[1][1])),
        ),
        clamped_to_support_bounds=bool(clamped),
    )


def spawn_goal_region_marker(env, spec: GoalRegionSpec):
    from omnigibson.objects.primitive_object import PrimitiveObject
    import torch as th

    existing = env.scene.object_registry("name", spec.marker_name)
    if existing is not None:
        try:
            env.scene.remove_object(existing)
        except Exception:
            env.scene.remove_object(obj=existing)

    marker = PrimitiveObject(
        relative_prim_path=f"/{spec.marker_name}",
        name=spec.marker_name,
        category="goal_region_marker",
        primitive_type="Sphere",
        radius=spec.radius_m,
        fixed_base=True,
        visual_only=True,
        rgba=spec.color_rgba,
    )
    env.scene.add_object(marker)
    marker.set_position_orientation(position=spec.center_world, orientation=(0.0, 0.0, 0.0, 1.0))
    print(f"[goal_region] spawned {spec.marker_name} at "
          f"{tuple(round(v, 3) for v in spec.center_world)} "
          f"radius={spec.radius_m:.3f} rgba={spec.color_rgba}")
    for visual_mesh in marker.root_link.visual_meshes.values():
        material = visual_mesh.material
        if material is None:
            continue
        try:
            material.diffuse_color_constant = th.tensor(spec.color_rgba[:3], dtype=th.float32)
        except Exception:
            pass
        try:
            material.set_input("enable_opacity", True)
            material.set_input("opacity_constant", float(spec.color_rgba[3]))
            material.set_input("opacity_mode", 0)
            material.set_input("opacity_threshold", 0.0)
        except Exception:
            pass
    if hasattr(marker, "keep_still"):
        marker.keep_still()
    return marker


def object_intersects_goal_region(obj, spec: GoalRegionSpec) -> bool:
    aabb_min, aabb_max = _aabb_bounds(obj)
    cx, cy, cz = spec.center_world
    closest_x = min(max(cx, aabb_min[0]), aabb_max[0])
    closest_y = min(max(cy, aabb_min[1]), aabb_max[1])
    closest_z = min(max(cz, aabb_min[2]), aabb_max[2])
    dx = cx - closest_x
    dy = cy - closest_y
    dz = cz - closest_z
    return (dx * dx + dy * dy + dz * dz) <= float(spec.radius_m * spec.radius_m)


def _aabb_intersects_sphere(aabb_min, aabb_max, center_world, radius_m) -> bool:
    cx, cy, cz = center_world
    closest_x = min(max(cx, aabb_min[0]), aabb_max[0])
    closest_y = min(max(cy, aabb_min[1]), aabb_max[1])
    closest_z = min(max(cz, aabb_min[2]), aabb_max[2])
    dx = cx - closest_x
    dy = cy - closest_y
    dz = cz - closest_z
    return (dx * dx + dy * dy + dz * dz) <= float(radius_m * radius_m)


def gripper_intersects_goal_region(env, spec: GoalRegionSpec) -> bool:
    """True if any of the robot's gripper links (fingers + eef link) has
    its AABB intersecting the goal sphere. Used as a relaxed alternative
    to ``object_intersects_goal_region(target, spec)`` for cases where the
    target ends up off-center while the gripper is correctly placed."""
    robot = env.robots[0] if env.robots else None
    if robot is None:
        return False
    arm = robot.default_arm
    links = []
    try:
        links.extend(list(robot.finger_links[arm]))
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        eef = robot.eef_links[arm]
        if eef is not None:
            links.append(eef)
    except (AttributeError, KeyError, TypeError):
        pass
    for link in links:
        try:
            aabb_min, aabb_max = _aabb_bounds(link)
        except Exception:  # noqa: BLE001
            continue
        if _aabb_intersects_sphere(aabb_min, aabb_max,
                                   spec.center_world, spec.radius_m):
            return True
    return False


def target_or_gripper_in_goal(env, target_obj, spec: GoalRegionSpec) -> tuple[bool, str]:
    """Relaxed positional check: True if target AABB OR any gripper link
    AABB intersects the goal sphere. Returns (ok, which) where which is
    one of "target", "gripper", or "neither" — useful for diagnostics."""
    if object_intersects_goal_region(target_obj, spec):
        return True, "target"
    if gripper_intersects_goal_region(env, spec):
        return True, "gripper"
    return False, "neither"


def robot_holds_target(env, target_obj) -> bool:
    robot = env.robots[0] if env.robots else None
    if robot is None:
        return False
    from omnigibson.controllers.controller_base import IsGraspingState
    result = robot.is_grasping(candidate_obj=target_obj)
    return result == IsGraspingState.TRUE


def remove_goal_region_from_scene_info(scene_info: dict[str, Any], diagnostics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stripped_scene = copy.deepcopy(scene_info)
    stripped_diag = copy.deepcopy(diagnostics)
    goal_region = stripped_diag.pop("goal_region", None)
    marker_name = ""
    if isinstance(goal_region, dict):
        marker_name = str(goal_region.get("marker_name") or "")

    init_info = stripped_scene.get("objects_info", {}).get("init_info", {})
    state_registry = stripped_scene.get("state", {}).get("registry", {}).get("object_registry", {})
    names_to_remove = []
    for scene_name, obj_info in list(init_info.items()):
        args = obj_info.get("args", {}) or {}
        category = str(args.get("category") or "")
        if scene_name == marker_name or category == "goal_region_marker" or scene_name.startswith("goal_region__"):
            names_to_remove.append(scene_name)
    for scene_name in names_to_remove:
        init_info.pop(scene_name, None)
        state_registry.pop(scene_name, None)
    metadata = stripped_scene.get("metadata") or {}
    task_meta = metadata.get("task") or {}
    if isinstance(task_meta, dict):
        task_meta.pop("goal_region", None)
    return stripped_scene, stripped_diag


def restore_robot_entries(saved_scene_info: dict[str, Any], original_scene_info: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(saved_scene_info)
    patched_init = patched.get("objects_info", {}).get("init_info", {})
    patched_state = patched.get("state", {}).get("registry", {}).get("object_registry", {})
    original_init = original_scene_info.get("objects_info", {}).get("init_info", {})
    original_state = original_scene_info.get("state", {}).get("registry", {}).get("object_registry", {})

    def _is_robot(info: dict[str, Any]) -> bool:
        class_module = str(info.get("class_module", ""))
        class_name = str(info.get("class_name", ""))
        return class_module.startswith("omnigibson.robots.") or class_name.endswith(("Robot", "Mounted", "Panda"))

    for name, info in list(patched_init.items()):
        if _is_robot(info):
            patched_init.pop(name, None)
            patched_state.pop(name, None)
    for name, info in original_init.items():
        if _is_robot(info):
            patched_init[name] = copy.deepcopy(info)
            if name in original_state:
                patched_state[name] = copy.deepcopy(original_state[name])
    return patched


def inject_goal_region_metadata(scene_info: dict[str, Any], spec: GoalRegionSpec | None, prompt: str | None) -> dict[str, Any]:
    updated = copy.deepcopy(scene_info)
    metadata = updated.setdefault("metadata", {})
    task_meta = metadata.setdefault("task", {})
    if spec is not None:
        task_meta["goal_region"] = spec.to_json()
    if prompt:
        task_meta["prompt"] = str(prompt)
    return updated
