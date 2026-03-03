import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import yaml
from bddl.activity import Conditions
from bddl.object_taxonomy import ObjectTaxonomy

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_scene_path
from omnigibson.utils.clutter_pack_layout import (
    ClutterObjectDescriptor,
    ClutterPackEntry,
    ClutterPackSpec,
    PackIntegrityReport,
    apply_pack_transform,
    build_clutter_pack,
    validate_pack_integrity,
)
from omnigibson.utils.franka_edge_align import (
    DEFAULT_ROLE_WEIGHTS,
    EdgeAlignObject,
    EdgeAlignRequest,
    EdgeAlignResult,
    place_franka_edge_aligned,
)
from omnigibson.utils.kitchen_bar_workspace import (
    KitchenBarZoneSpec,
    ZoneCapacityStats,
    bounds_overlap,
    compute_kitchen_bar_zone,
    compute_zone_capacity,
    contains_point,
)
from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec
import omnigibson.utils.transform_utils as T


# Stable defaults for local GUI + MVP workflow.
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False

_OBJECT_TAXONOMY = ObjectTaxonomy()

HARDCODE_SCENE_MODEL = "house_double_floor_lower"
HARDCODE_SUPPORT_NAME = "bar_egwapq_0"
HARDCODE_SUPPORT_CATEGORY = "bar"
HARDCODE_SINK_NAME = "drop_in_sink_lkklqs_0"
WORKSPACE_PRESET = "kitchen_bar_sink_left_v1"
FIXED_EDGE_LABEL = "x_min"
FIXED_EDGE_SCAN_OFFSETS = (0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20)


@dataclass(frozen=True)
class MVPGateReport:
    scene_sane: bool
    base_on_ground: bool
    base_collision_free: bool
    gap_ok: bool
    target_in_reach_band: bool
    pack_integrity_ok: bool
    all_objects_in_red_zone: bool
    all_objects_outside_sink_keepout: bool
    pass_gate: bool
    failure_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ObjectZoneReport:
    all_in_red_zone: bool
    all_outside_sink_keepout: bool
    out_of_zone_instances: Tuple[str, ...]
    in_keepout_instances: Tuple[str, ...]


def parse_args():
    parser = argparse.ArgumentParser(description="Kitchen-bar hardcoded cup-first FrankaMounted MVP runner")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--activity-name", default=None, help="BehaviorTask activity_name override.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--mount-gap-m", type=float, default=0.03, help="Desired Franka base gap from fixed edge.")
    parser.add_argument("--jitter-scale", type=float, default=0.01, help="Action jitter sigma for zero_jitter mode.")
    parser.add_argument("--showcase-gui", action="store_true", help="Enable manual GUI camera teleoperation.")
    parser.add_argument("--strict-gate", dest="strict_gate", action="store_true", help="Enable strict gate.")
    parser.add_argument("--no-strict-gate", dest="strict_gate", action="store_false", help="Disable strict gate.")
    parser.set_defaults(strict_gate=True)
    parser.add_argument("--debug-jsonl", default=None, help="Optional JSONL path for reset diagnostics.")
    return parser.parse_args()


def _default_config_path():
    return os.path.join(og.root_path, "configs", "franka_mounted_behavior_cached_kitchen_bar.yaml")


def _append_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_config(args):
    cfg_path = args.config or _default_config_path()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.activity_name:
        cfg["task"]["activity_name"] = args.activity_name

    scene_model = cfg["scene"].get("scene_model")
    if scene_model != HARDCODE_SCENE_MODEL:
        raise RuntimeError(
            f"Kitchen-bar runner is hardcoded for scene_model='{HARDCODE_SCENE_MODEL}', got '{scene_model}'."
        )

    cfg["scene"]["scene_instance"] = None
    cfg["scene"]["scene_file"] = os.path.join(get_scene_path(scene_model), "json", f"{scene_model}_best.json")

    # MVP path uses BDDL object sampling + explicit post-reset layout.
    cfg["task"]["online_object_sampling"] = True
    cfg["task"]["use_presampled_robot_pose"] = False
    return cfg


def _task_uses_substance_system(activity_name, activity_definition_id=0):
    try:
        cond = Conditions(
            behavior_activity=activity_name,
            activity_definition=activity_definition_id,
            simulator_name="omnigibson",
            predefined_problem=None,
        )
    except Exception:
        return False, None

    for synset in cond.parsed_objects.keys():
        try:
            if "substance" in _OBJECT_TAXONOMY.get_abilities(synset):
                return True, synset
        except Exception:
            continue
    return False, None


def _configure_dynamics(cfg):
    activity_name = cfg["task"]["activity_name"]
    activity_definition_id = cfg["task"].get("activity_definition_id", 0)
    needs_substance, synset = _task_uses_substance_system(activity_name, activity_definition_id)
    if needs_substance:
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        print(f"[MVP] GPU dynamics enabled (substance detected: {synset}).")
    else:
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = True
        print("[MVP] GPU dynamics disabled (no substance objects detected).")


def _to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _to_float3(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return [float(x[0]), float(x[1]), float(x[2])]


def _quat_from_yaw(yaw):
    half = 0.5 * yaw
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _distance_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_finite_pose(pos3):
    return len(pos3) >= 3 and all(math.isfinite(v) for v in pos3[:3])


def _is_floor_like(category):
    text = str(category).lower()
    return any(tok in text for tok in ("floor", "carpet", "rug", "mat", "paver", "tile"))


def _normalize_bounds_2d(bounds):
    (x0, y0), (x1, y1) = bounds
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))


def _bounds_inside(inner, outer, margin=0.0):
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    (ix0, iy0), (ix1, iy1) = _normalize_bounds_2d(inner)
    (ox0, oy0), (ox1, oy1) = _normalize_bounds_2d(outer)
    return (
        ix0 >= (ox0 + margin)
        and iy0 >= (oy0 + margin)
        and ix1 <= (ox1 - margin)
        and iy1 <= (oy1 - margin)
    )


def _resolve_support_and_sink(env):
    try:
        support = env.scene.object_registry("name", HARDCODE_SUPPORT_NAME)
    except Exception:
        support = None
    if support is None:
        raise RuntimeError(f"Hardcoded support '{HARDCODE_SUPPORT_NAME}' not found in scene objects.")

    support_category = str(getattr(support, "category", ""))
    if support_category != HARDCODE_SUPPORT_CATEGORY:
        raise RuntimeError(
            f"Hardcoded support '{HARDCODE_SUPPORT_NAME}' category mismatch: "
            f"expected '{HARDCODE_SUPPORT_CATEGORY}', got '{support_category}'."
        )

    try:
        sink = env.scene.object_registry("name", HARDCODE_SINK_NAME)
    except Exception:
        sink = None
    if sink is None:
        raise RuntimeError(f"Hardcoded sink '{HARDCODE_SINK_NAME}' not found in scene objects.")

    return support, sink


def _iter_scope_objects(env):
    scope = getattr(env.task, "object_scope", {}) or {}
    for inst, ent in scope.items():
        if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
            continue
        obj = getattr(ent, "wrapped_obj", None)
        if obj is None:
            continue
        yield inst, obj


def _get_scope_obj(env, inst):
    scope = getattr(env.task, "object_scope", {}) or {}
    ent = scope.get(inst, None)
    if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
        return None
    return getattr(ent, "wrapped_obj", None)


def _build_task_object_sets(env, task_spec):
    available = {inst for inst, _ in _iter_scope_objects(env)}
    target_ids = [inst for inst in task_spec.target_ids if inst in available]
    fragile_ids = [inst for inst in task_spec.fragile_ids if inst in available and inst not in target_ids]
    support_ids = [inst for inst in task_spec.support_ids if inst in available]

    excluded_prefixes = ("agent.", "floor.")
    clutter_ids = []
    for inst in available:
        if inst in target_ids or inst in fragile_ids or inst in support_ids:
            continue
        if inst.startswith(excluded_prefixes):
            continue
        clutter_ids.append(inst)

    clutter_ids = sorted(clutter_ids)
    if not target_ids:
        for inst, _ in _iter_scope_objects(env):
            if inst.startswith(("coffee_cup.", "cup.", "mug.")):
                target_ids = [inst]
                break

    return {
        "target_ids": tuple(target_ids),
        "fragile_ids": tuple(sorted(fragile_ids)),
        "support_ids": tuple(sorted(support_ids)),
        "clutter_ids": tuple(clutter_ids),
    }


def _descriptor_from_obj(inst, role, obj):
    aabb_min, aabb_max = obj.aabb
    mn = _to_float3(aabb_min)
    mx = _to_float3(aabb_max)
    dx = max(0.01, mx[0] - mn[0])
    dy = max(0.01, mx[1] - mn[1])
    dz = max(0.01, mx[2] - mn[2])
    return ClutterObjectDescriptor(
        instance_id=inst,
        role=role,
        half_extent_xy=(0.5 * dx, 0.5 * dy),
        height=dz,
    )


def _compute_pack_relative_bounds(pack_spec: ClutterPackSpec, descriptor_by_inst):
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    for entry in pack_spec.object_entries:
        d = descriptor_by_inst[entry.inst_id]
        rel_x, rel_y = entry.rel_pose[0], entry.rel_pose[1]
        x_min = min(x_min, rel_x - d.half_extent_xy[0])
        x_max = max(x_max, rel_x + d.half_extent_xy[0])
        y_min = min(y_min, rel_y - d.half_extent_xy[1])
        y_max = max(y_max, rel_y + d.half_extent_xy[1])

    if not math.isfinite(x_min):
        raise RuntimeError("Failed to compute clutter-pack bounds.")

    return ((x_min, y_min), (x_max, y_max))


def _choose_pack_origin_in_zone(zone_bounds, rel_bounds):
    (zx0, zy0), (zx1, zy1) = _normalize_bounds_2d(zone_bounds)
    (rx0, ry0), (rx1, ry1) = _normalize_bounds_2d(rel_bounds)

    ox_lo = zx0 - rx0
    ox_hi = zx1 - rx1
    oy_lo = zy0 - ry0
    oy_hi = zy1 - ry1

    if ox_lo > ox_hi or oy_lo > oy_hi:
        raise RuntimeError(
            "zone_capacity_exceeded: clutter pack does not fit inside red zone "
            f"(ox=[{ox_lo:.3f},{ox_hi:.3f}], oy=[{oy_lo:.3f},{oy_hi:.3f}])"
        )

    desired_x = 0.5 * (zx0 + zx1)
    desired_y = 0.5 * (zy0 + zy1)
    origin_x = min(max(desired_x, ox_lo), ox_hi)
    origin_y = min(max(desired_y, oy_lo), oy_hi)
    return (origin_x, origin_y)


def _scale_pack_spec_xy(pack_spec: ClutterPackSpec, scale_xy: float) -> ClutterPackSpec:
    if scale_xy <= 0.0:
        raise ValueError("scale_xy must be > 0")
    if scale_xy >= 0.999:
        return pack_spec

    scaled_entries = []
    for entry in pack_spec.object_entries:
        rel_x, rel_y, rel_z, qx, qy, qz, qw = entry.rel_pose
        scaled_entries.append(
            ClutterPackEntry(
                inst_id=entry.inst_id,
                role=entry.role,
                rel_pose=(rel_x * scale_xy, rel_y * scale_xy, rel_z, qx, qy, qz, qw),
            )
        )
    return ClutterPackSpec(
        table_obj_name=pack_spec.table_obj_name,
        pack_origin_world=pack_spec.pack_origin_world,
        object_entries=tuple(scaled_entries),
        seed=pack_spec.seed,
        template_id=pack_spec.template_id,
    )


def _fit_pack_to_zone(pack_spec, descriptor_by_inst, red_zone_bounds, min_scale=0.40):
    if min_scale <= 0.0 or min_scale > 1.0:
        raise ValueError("min_scale must be in (0, 1]")

    scale = 1.0
    last_error = None
    while scale >= min_scale - 1e-6:
        candidate_pack = _scale_pack_spec_xy(pack_spec, scale)
        rel_bounds = _compute_pack_relative_bounds(candidate_pack, descriptor_by_inst)
        try:
            origin_xy = _choose_pack_origin_in_zone(red_zone_bounds, rel_bounds)
            return candidate_pack, rel_bounds, origin_xy, scale
        except RuntimeError as e:
            last_error = e
            scale = round(scale * 0.90, 3)

    raise RuntimeError(
        "zone_capacity_exceeded: clutter pack cannot fit in fixed red zone after compaction. "
        f"min_scale={min_scale:.2f}, last_error={last_error}"
    )


def _aabb_overlap_3d(aabb_a, aabb_b, tol=0.0):
    a_min = _to_float3(aabb_a[0])
    a_max = _to_float3(aabb_a[1])
    b_min = _to_float3(aabb_b[0])
    b_max = _to_float3(aabb_b[1])
    overlap_x = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0]) > tol
    overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1]) > tol
    overlap_z = min(a_max[2], b_max[2]) - max(a_min[2], b_min[2]) > tol
    return overlap_x and overlap_y and overlap_z


def _get_robot_base_aabb(robot):
    links = getattr(robot, "links", {}) or {}
    for key in ("base_link", "base", "base_footprint", "chassis"):
        if key in links:
            return links[key].aabb

    best = None
    best_z = float("inf")
    for link in links.values():
        try:
            aabb = link.aabb
            mn = _to_float3(aabb[0])
            mx = _to_float3(aabb[1])
            z = 0.5 * (mn[2] + mx[2])
            if z < best_z:
                best = aabb
                best_z = z
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Could not resolve robot base AABB.")
    return best


def _get_robot_base_link(robot):
    links = getattr(robot, "links", {}) or {}
    for key in ("base_link", "base", "base_footprint", "chassis"):
        if key in links:
            return links[key], key

    best_link = None
    best_name = None
    best_z = float("inf")
    for name, link in links.items():
        try:
            aabb = link.aabb
            mn = _to_float3(aabb[0])
            mx = _to_float3(aabb[1])
            z = 0.5 * (mn[2] + mx[2])
            if z < best_z:
                best_z = z
                best_link = link
                best_name = name
        except Exception:
            continue
    if best_link is None:
        raise RuntimeError("Could not resolve robot base link.")
    return best_link, best_name


def _robot_half_extent_xy(robot):
    aabb = _get_robot_base_aabb(robot)
    mn = _to_float3(aabb[0])
    mx = _to_float3(aabb[1])
    return ((mx[0] - mn[0]) * 0.5, (mx[1] - mn[1]) * 0.5)


def _collect_robot_penetration_hits(env):
    robot = env.robots[0]
    hits = set()
    try:
        robot_paths = set(getattr(robot, "link_prim_paths", []))
        for link in (getattr(robot, "links", {}) or {}).values():
            if not hasattr(link, "contact_list"):
                continue
            for contact in link.contact_list():
                for body in (contact.body0, contact.body1):
                    if body in robot_paths:
                        continue
                    tokens = body.split("/")
                    if len(tokens) < 2:
                        continue
                    obj_path = "/".join(tokens[:-1])
                    try:
                        obj = env.scene.object_registry("prim_path", obj_path)
                    except Exception:
                        obj = None
                    if obj is None or obj is robot:
                        continue
                    if _is_floor_like(getattr(obj, "category", "")):
                        continue
                    hits.add(getattr(obj, "name", "unknown_obj"))
    except Exception:
        pass

    # Conservative AABB fallback / supplement:
    # check all low-height robot links against scene objects to catch teleport-overlap cases
    # where contact_list may miss immediate deep interpenetration.
    robot_link_aabbs = []
    for link in (getattr(robot, "links", {}) or {}).values():
        try:
            aabb = link.aabb
            mn = _to_float3(aabb[0])
            mx = _to_float3(aabb[1])
            robot_link_aabbs.append((mn, mx))
        except Exception:
            continue

    for obj in getattr(env.scene, "objects", []):
        if obj is None or obj is robot:
            continue
        if _is_floor_like(getattr(obj, "category", "")):
            continue
        try:
            aabb = obj.aabb
            obj_min = _to_float3(aabb[0])
            obj_max = _to_float3(aabb[1])
        except Exception:
            continue

        for link_min, link_max in robot_link_aabbs:
            # Ignore very high-only overlaps for low-base placement checks.
            if min(link_min[2], link_max[2]) > (obj_max[2] + 0.30):
                continue
            if _aabb_overlap_3d((link_min, link_max), (obj_min, obj_max), tol=0.002):
                hits.add(getattr(obj, "name", "unknown_obj"))
                break

    return sorted(hits)


def _compute_floor_z(env):
    floor_z = 0.0
    for inst, obj in _iter_scope_objects(env):
        if not inst.startswith("floor."):
            continue
        try:
            _, aabb_max = obj.aabb
            floor_z = max(floor_z, _to_float3(aabb_max)[2])
        except Exception:
            continue
    return floor_z


def _print_object_inventory(env, obj_sets):
    scope = getattr(env.task, "object_scope", {}) or {}
    print("[MVP] Object inventory:")
    print(f"[MVP]   target_set={list(obj_sets['target_ids'])}")
    print(f"[MVP]   fragile_set={list(obj_sets['fragile_ids'])}")
    print(f"[MVP]   clutter_set={list(obj_sets['clutter_ids'])}")
    tracked = list(obj_sets["target_ids"]) + list(obj_sets["fragile_ids"]) + list(obj_sets["clutter_ids"])
    for inst in tracked:
        ent = scope.get(inst, None)
        if ent is None or not getattr(ent, "exists", False):
            print(f"[MVP]   {inst} -> MISSING")
            continue
        obj = getattr(ent, "wrapped_obj", None)
        if obj is None:
            print(f"[MVP]   {inst} -> MISSING_WRAPPED_OBJ")
            continue
        pos = _to_float3(obj.get_position_orientation()[0])
        print(f"[MVP]   {inst} -> {getattr(obj, 'name', 'unknown')} @ {pos}")


def _set_showcase_camera_manual(env, target_obj):
    if target_obj is None:
        return
    robot = env.robots[0]
    robot_pos = _to_float3(robot.get_position_orientation()[0])
    target_pos = _to_float3(target_obj.get_position_orientation()[0])
    center = [
        0.5 * (robot_pos[0] + target_pos[0]),
        0.5 * (robot_pos[1] + target_pos[1]),
        max(robot_pos[2] + 0.7, target_pos[2] + 0.25),
    ]
    cam_pos = [center[0] - 1.0, center[1] - 1.1, center[2] + 0.5]
    direction = np.asarray([center[0] - cam_pos[0], center[1] - cam_pos[1], center[2] - cam_pos[2]], dtype=np.float32)
    direction /= max(1e-6, np.linalg.norm(direction))
    pan = float(np.arctan2(-direction[0], direction[1]))
    tilt = float(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    cam_quat = T.euler2quat(th.tensor([math.pi / 2 + tilt, 0.0, pan], dtype=th.float32))
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat.tolist())
    og.sim.enable_viewer_camera_teleoperation()
    print("[MVP] Manual GUI mode: camera teleoperation enabled.")


def _make_zero_jitter_action(robot, rng, jitter_scale):
    action = np.zeros_like(np.asarray(robot.action_space.sample(), dtype=np.float32))
    jitter = rng.normal(0.0, jitter_scale, size=action.shape).astype(np.float32)
    action = action + jitter
    if hasattr(robot.action_space, "low") and hasattr(robot.action_space, "high"):
        low = np.asarray(robot.action_space.low, dtype=np.float32)
        high = np.asarray(robot.action_space.high, dtype=np.float32)
        action = np.clip(action, low, high)
    return action


def _set_demo_arm_pose(robot):
    try:
        arm_idx = robot.arm_control_idx[robot.default_arm]
        arm_target = np.asarray([0.0, -0.55, 0.0, -2.05, 0.0, 1.65, 0.75], dtype=np.float32)
        if len(arm_idx) == len(arm_target):
            robot.set_joint_positions(positions=arm_target, indices=arm_idx, drive=False)
    except Exception as e:
        print(f"[MVP] Demo arm pose set skipped: {e}")
    try:
        gripper_idx = robot.gripper_control_idx[robot.default_arm]
        robot.set_joint_positions(
            positions=np.asarray([0.04] * len(gripper_idx), dtype=np.float32),
            indices=gripper_idx,
            drive=False,
        )
    except Exception:
        pass


def _evaluate_object_zone_constraints(objects_by_inst, red_zone_bounds, sink_keepout_bounds):
    out_of_zone = []
    in_keepout = []

    for inst, obj in objects_by_inst.items():
        try:
            aabb_min, aabb_max = obj.aabb
            mn = _to_float3(aabb_min)
            mx = _to_float3(aabb_max)
            obj_bounds = ((mn[0], mn[1]), (mx[0], mx[1]))
        except Exception:
            pos = _to_float3(obj.get_position_orientation()[0])
            obj_bounds = ((pos[0], pos[1]), (pos[0], pos[1]))

        center_in_zone = contains_point(
            red_zone_bounds,
            (
                0.5 * (obj_bounds[0][0] + obj_bounds[1][0]),
                0.5 * (obj_bounds[0][1] + obj_bounds[1][1]),
            ),
        )
        if (not _bounds_inside(obj_bounds, red_zone_bounds)) and (not center_in_zone):
            out_of_zone.append(inst)
        if bounds_overlap(obj_bounds, sink_keepout_bounds, tol=1e-6):
            in_keepout.append(inst)

    return ObjectZoneReport(
        all_in_red_zone=len(out_of_zone) == 0,
        all_outside_sink_keepout=len(in_keepout) == 0,
        out_of_zone_instances=tuple(sorted(out_of_zone)),
        in_keepout_instances=tuple(sorted(in_keepout)),
    )


def _evaluate_gate(
    robot_pos,
    target_pos,
    floor_z,
    penetration_hits,
    edge_align_result: EdgeAlignResult,
    target_gap_m: float,
    integrity_report: PackIntegrityReport,
    zone_report: ObjectZoneReport,
):
    scene_sane = _is_finite_pose(robot_pos) and _is_finite_pose(target_pos) and max(abs(target_pos[0]), abs(target_pos[1])) < 100.0
    base_on_ground = abs(robot_pos[2] - floor_z) <= 0.03
    base_collision_free = len(penetration_hits) == 0
    gap_ok = edge_align_result.gap_actual >= 0.0 and abs(edge_align_result.gap_actual - target_gap_m) <= 0.04
    target_dist = _distance_xy(robot_pos[:2], target_pos[:2])
    target_in_reach_band = 0.20 <= target_dist <= 1.10
    pack_integrity_ok = integrity_report.ok
    all_objects_in_red_zone = zone_report.all_in_red_zone
    all_objects_outside_sink_keepout = zone_report.all_outside_sink_keepout

    failures = []
    if not scene_sane:
        failures.append("scene_not_sane")
    if not base_on_ground:
        failures.append("robot_base_not_on_ground")
    if not base_collision_free:
        failures.append("robot_base_penetration")
    if not gap_ok:
        failures.append("table_gap_not_ok")
    if not target_in_reach_band:
        failures.append("target_not_in_reach_band")
    if not pack_integrity_ok:
        failures.append("pack_integrity_broken")
    if not all_objects_in_red_zone:
        failures.append("objects_outside_red_zone")
    if not all_objects_outside_sink_keepout:
        failures.append("objects_inside_sink_keepout")

    return MVPGateReport(
        scene_sane=scene_sane,
        base_on_ground=base_on_ground,
        base_collision_free=base_collision_free,
        gap_ok=gap_ok,
        target_in_reach_band=target_in_reach_band,
        pack_integrity_ok=pack_integrity_ok,
        all_objects_in_red_zone=all_objects_in_red_zone,
        all_objects_outside_sink_keepout=all_objects_outside_sink_keepout,
        pass_gate=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def _log_workspace_geometry(zone: KitchenBarZoneSpec, support, sink):
    print(f"[MVP] workspace_preset={zone.workspace_preset}")
    print(
        f"[MVP] support={HARDCODE_SUPPORT_NAME} (cat={getattr(support, 'category', 'unknown')}) "
        f"sink={HARDCODE_SINK_NAME} (cat={getattr(sink, 'category', 'unknown')})"
    )
    print(f"[MVP] bar_bounds={zone.bar_bounds}")
    print(f"[MVP] sink_keepout_bounds={zone.sink_keepout_bounds}")
    print(f"[MVP] red_zone_bounds={zone.red_zone_bounds}, long_axis={zone.long_axis}")


def main():
    args = parse_args()
    cfg = _load_config(args)
    _configure_dynamics(cfg)

    task_name = cfg["task"]["activity_name"]
    task_spec = build_manipulation_task_spec(task_name)

    print(
        f"[MVP] Configured: task={task_name}, scene={cfg['scene']['scene_model']}, "
        f"workspace={WORKSPACE_PRESET}, strict_gate={args.strict_gate}, steps={args.steps}"
    )
    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"[MVP] Episode {ep + 1}/{args.episodes} reset")
            env.reset()
            og.sim.step()

            support, sink = _resolve_support_and_sink(env)
            support_aabb_min, support_aabb_max = support.aabb
            sink_aabb_min, sink_aabb_max = sink.aabb
            bar_bounds_xy = (
                (float(support_aabb_min[0]), float(support_aabb_min[1])),
                (float(support_aabb_max[0]), float(support_aabb_max[1])),
            )
            sink_bounds_xy = (
                (float(sink_aabb_min[0]), float(sink_aabb_min[1])),
                (float(sink_aabb_max[0]), float(sink_aabb_max[1])),
            )
            table_top_z = float(support_aabb_max[2])

            zone = compute_kitchen_bar_zone(
                bar_bounds_xy=bar_bounds_xy,
                sink_bounds_xy=sink_bounds_xy,
                workspace_preset=WORKSPACE_PRESET,
                edge_margin_m=0.05,
                sink_keepout_margin_m=0.10,
                sink_side_clearance_m=0.02,
                min_zone_span_m=0.20,
            )
            _log_workspace_geometry(zone, support, sink)

            obj_sets = _build_task_object_sets(env, task_spec)
            if len(obj_sets["target_ids"]) == 0:
                raise RuntimeError("No target objects found in object_scope.")
            target_inst = obj_sets["target_ids"][0]
            target_obj = _get_scope_obj(env, target_inst)

            descriptors: List[ClutterObjectDescriptor] = []
            for inst in obj_sets["target_ids"]:
                obj = _get_scope_obj(env, inst)
                if obj is None:
                    continue
                descriptors.append(_descriptor_from_obj(inst, "target", obj))
            for inst in obj_sets["fragile_ids"]:
                obj = _get_scope_obj(env, inst)
                if obj is None:
                    continue
                descriptors.append(_descriptor_from_obj(inst, "fragile", obj))
            for inst in obj_sets["clutter_ids"]:
                obj = _get_scope_obj(env, inst)
                if obj is None:
                    continue
                descriptors.append(_descriptor_from_obj(inst, "clutter", obj))

            if len(descriptors) == 0:
                raise RuntimeError("No clutter-pack descriptors were created.")

            descriptor_by_inst = {d.instance_id: d for d in descriptors}
            zone_capacity: ZoneCapacityStats = compute_zone_capacity(
                red_zone_bounds=zone.red_zone_bounds,
                half_extents_xy=[d.half_extent_xy for d in descriptors],
                per_object_padding=0.02,
            )
            print(
                "[MVP] zone_capacity: "
                f"required={zone_capacity.required_area:.4f}, "
                f"available={zone_capacity.available_area:.4f}, "
                f"utilization={zone_capacity.utilization:.3f}"
            )

            if zone_capacity.utilization > 0.85:
                raise RuntimeError(
                    "zone_capacity_exceeded: object footprint too dense for fixed red zone "
                    f"(utilization={zone_capacity.utilization:.3f})"
                )

            pack_spec = build_clutter_pack(
                table_obj_name=getattr(support, "name", HARDCODE_SUPPORT_NAME),
                descriptors=descriptors,
                seed=args.seed + ep,
            )
            pack_spec, rel_bounds, origin_xy, compact_scale = _fit_pack_to_zone(
                pack_spec=pack_spec,
                descriptor_by_inst=descriptor_by_inst,
                red_zone_bounds=zone.red_zone_bounds,
                min_scale=0.40,
            )
            pack_origin = (origin_xy[0], origin_xy[1], table_top_z)
            print(f"[MVP] pack_origin={pack_origin}, compact_scale={compact_scale:.3f}")

            objects_by_inst = {}
            for d in descriptors:
                obj = _get_scope_obj(env, d.instance_id)
                if obj is not None:
                    objects_by_inst[d.instance_id] = obj

            world_positions = apply_pack_transform(
                pack_spec=pack_spec,
                objects_by_inst=objects_by_inst,
                pack_origin_world=pack_origin,
                pack_yaw=0.0,
                table_top_z=table_top_z,
            )
            for _ in range(3):
                og.sim.step()

            integrity = validate_pack_integrity(
                pack_spec=pack_spec,
                world_positions=world_positions,
                pack_origin_world=pack_origin,
                pack_yaw=0.0,
                tol_xy=0.035,
            )

            zone_report = _evaluate_object_zone_constraints(
                objects_by_inst=objects_by_inst,
                red_zone_bounds=zone.red_zone_bounds,
                sink_keepout_bounds=zone.sink_keepout_bounds,
            )

            robot = env.robots[0]
            floor_z = _compute_floor_z(env)
            half_extent_xy = _robot_half_extent_xy(robot)
            pack_objects_world = []
            for entry in pack_spec.object_entries:
                if entry.inst_id not in world_positions:
                    continue
                wx, wy, _ = world_positions[entry.inst_id]
                pack_objects_world.append(
                    EdgeAlignObject(
                        name=entry.inst_id,
                        role=entry.role,
                        position_xy=(wx, wy),
                    )
                )

            if len(pack_objects_world) == 0:
                raise RuntimeError("No pack objects available for fixed-edge alignment.")

            def collision_checker(base_pose_xyyaw):
                x, y, yaw = base_pose_xyyaw
                robot.set_position_orientation(position=(x, y, floor_z), orientation=_quat_from_yaw(yaw))
                og.sim.step()
                return _collect_robot_penetration_hits(env)

            edge_result = place_franka_edge_aligned(
                EdgeAlignRequest(
                    table_aabb_xy=zone.bar_bounds,
                    pack_objects_world=tuple(pack_objects_world),
                    role_weights=DEFAULT_ROLE_WEIGHTS,
                    robot_half_extent_xy=(half_extent_xy[0], half_extent_xy[1]),
                    edge_gap_m=args.mount_gap_m,
                    edge_margin_m=0.05,
                    scan_offsets_m=FIXED_EDGE_SCAN_OFFSETS,
                    collision_checker=collision_checker,
                    preferred_edge=FIXED_EDGE_LABEL,
                )
            )

            final_pos = (
                edge_result.base_pose["position"][0],
                edge_result.base_pose["position"][1],
                floor_z,
            )
            robot.set_position_orientation(position=final_pos, orientation=edge_result.base_pose["orientation"])
            og.sim.step()

            penetration_hits = _collect_robot_penetration_hits(env)
            target_pos = _to_float3(target_obj.get_position_orientation()[0])
            robot_pos = _to_float3(robot.get_position_orientation()[0])
            gate = _evaluate_gate(
                robot_pos=robot_pos,
                target_pos=target_pos,
                floor_z=floor_z,
                penetration_hits=penetration_hits,
                edge_align_result=edge_result,
                target_gap_m=args.mount_gap_m,
                integrity_report=integrity,
                zone_report=zone_report,
            )

            print(
                f"[MVP] Mount: edge={edge_result.edge_label}, rank={edge_result.candidate_rank}, "
                f"gap={edge_result.gap_actual:.3f}, hits={len(edge_result.collision_hits)}"
            )
            if penetration_hits:
                print(f"[MVP] Penetration objects: {penetration_hits}")
            if zone_report.out_of_zone_instances:
                print(f"[MVP] Out-of-zone objects: {list(zone_report.out_of_zone_instances)}")
            if zone_report.in_keepout_instances:
                print(f"[MVP] Sink-keepout objects: {list(zone_report.in_keepout_instances)}")

            print(f"[MVP] Gate: pass={gate.pass_gate}, reasons={list(gate.failure_reasons)}")
            _print_object_inventory(env, obj_sets)

            _append_jsonl(
                args.debug_jsonl,
                {
                    "episode": ep + 1,
                    "workspace_preset": WORKSPACE_PRESET,
                    "support_name": HARDCODE_SUPPORT_NAME,
                    "sink_name": HARDCODE_SINK_NAME,
                    "bar_bounds": zone.bar_bounds,
                    "red_zone_bounds": zone.red_zone_bounds,
                    "sink_keepout_bounds": zone.sink_keepout_bounds,
                    "zone_capacity_stats": asdict(zone_capacity),
                    "fixed_edge_mount": {
                        "edge_label": FIXED_EDGE_LABEL,
                        "scan_offsets": list(FIXED_EDGE_SCAN_OFFSETS),
                    },
                    "pack_origin_world": list(pack_origin),
                    "pack_compact_scale": compact_scale,
                    "edge_result": asdict(edge_result),
                    "integrity": asdict(integrity),
                    "zone_report": asdict(zone_report),
                    "gate": asdict(gate),
                },
            )

            if args.showcase_gui:
                _set_showcase_camera_manual(env, target_obj)

            if args.strict_gate and not gate.pass_gate:
                raise RuntimeError(f"Strict gate failed: {list(gate.failure_reasons)}")

            _set_demo_arm_pose(robot)
            og.sim.step()

            executed = 0
            terminated = False
            truncated = False
            for _ in range(args.steps):
                print(f"[MVP] Step {executed + 1}/{args.steps}")
                action = _make_zero_jitter_action(robot, rng, args.jitter_scale)
                _, _, terminated, truncated, _ = env.step(action)
                executed += 1
                if terminated or truncated:
                    break
            print(f"[MVP] Episode done: steps={executed}, terminated={terminated}, truncated={truncated}")
    finally:
        print("[MVP] Shutdown simulator.")
        try:
            og.clear()
        except Exception as e:
            print(f"[MVP] og.clear warning (ignored): {e}")


if __name__ == "__main__":
    main()
