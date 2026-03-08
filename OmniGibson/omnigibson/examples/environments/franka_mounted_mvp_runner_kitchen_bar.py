import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import yaml
from bddl.activity import Conditions
from bddl.object_taxonomy import ObjectTaxonomy

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.prims.material_prim import MaterialPrim, OmniPBRMaterialPrim
from omnigibson.utils.asset_utils import get_dataset_path, get_scene_path
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
CLUTTER_DENSITY_PRESETS = {
    # Wider clearance, lower density.
    "low": {
        "pack_jitter_xy": 0.010,
        "pack_min_clearance": 0.040,
        "zone_padding": 0.030,
        "zone_util_cap": 0.70,
        "zone_edge_margin_m": 0.05,
        "sink_keepout_margin_m": 0.10,
        "sink_side_clearance_m": 0.02,
    },
    # Balanced baseline.
    "medium": {
        "pack_jitter_xy": 0.015,
        "pack_min_clearance": 0.025,
        "zone_padding": 0.020,
        "zone_util_cap": 0.85,
        "zone_edge_margin_m": 0.05,
        "sink_keepout_margin_m": 0.10,
        "sink_side_clearance_m": 0.02,
    },
    # Denser packing / more crowded.
    "high": {
        "pack_jitter_xy": 0.022,
        "pack_min_clearance": 0.008,
        "zone_padding": 0.008,
        "zone_util_cap": 0.98,
        "zone_edge_margin_m": 0.04,
        "sink_keepout_margin_m": 0.08,
        "sink_side_clearance_m": 0.015,
    },
    # Maximum cluttered appearance before explicit stacking.
    "ultra": {
        "pack_jitter_xy": 0.026,
        "pack_min_clearance": 0.004,
        "zone_padding": 0.004,
        "zone_util_cap": 1.10,
        "zone_edge_margin_m": 0.03,
        "sink_keepout_margin_m": 0.06,
        "sink_side_clearance_m": 0.010,
    },
}


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
    parser.add_argument("--steps", type=int, default=5000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--mount-gap-m", type=float, default=0.03, help="Desired Franka base gap from fixed edge.")
    parser.add_argument(
        "--clutter-density",
        choices=("low", "medium", "high", "ultra"),
        default="high",
        help="Packing density preset for the same BDDL object set.",
    )
    parser.add_argument(
        "--pack-jitter-xy",
        type=float,
        default=None,
        help="Override pack XY jitter (meters).",
    )
    parser.add_argument(
        "--pack-min-clearance",
        type=float,
        default=None,
        help="Override minimum inter-object XY clearance in clutter pack (meters).",
    )
    parser.add_argument(
        "--zone-utilization-cap",
        type=float,
        default=None,
        help="Override red-zone utilization warning threshold (0~1+).",
    )
    parser.add_argument(
        "--pack-min-scale",
        type=float,
        default=None,
        help="Deprecated; ignored. XY pack scaling is disabled in kitchen-bar runner.",
    )
    parser.add_argument(
        "--pack-clearance-step-m",
        type=float,
        default=0.005,
        help="Minimum-clearance decrement step for dense packing solver.",
    )
    parser.add_argument(
        "--pack-clearance-floor-m",
        type=float,
        default=0.002,
        help="Minimum-clearance lower bound for dense packing solver.",
    )
    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=0.005,
        help="Local dense grid step for clutter pack candidate points.",
    )
    parser.add_argument(
        "--frontier-noise-margin-m",
        type=float,
        default=0.02,
        help="Distance margin above d_min for frontier random candidate pool.",
    )
    parser.add_argument(
        "--pack-tries-per-clearance",
        type=int,
        default=10,
        help="How many randomized layout attempts to try for each clearance level before culling.",
    )
    parser.add_argument(
        "--zone-edge-margin-m",
        type=float,
        default=None,
        help="Override red-zone edge margin from bar boundary (meters).",
    )
    parser.add_argument(
        "--sink-keepout-margin-m",
        type=float,
        default=None,
        help="Override sink keepout expansion margin (meters).",
    )
    parser.add_argument(
        "--sink-side-clearance-m",
        type=float,
        default=None,
        help="Override extra clearance between sink keepout and red zone (meters).",
    )
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


def _resolve_clutter_controls(args):
    preset = CLUTTER_DENSITY_PRESETS[args.clutter_density]
    jitter_xy = float(preset["pack_jitter_xy"] if args.pack_jitter_xy is None else args.pack_jitter_xy)
    min_clearance = float(
        preset["pack_min_clearance"] if args.pack_min_clearance is None else args.pack_min_clearance
    )
    zone_padding = float(preset["zone_padding"])
    util_cap = float(preset["zone_util_cap"] if args.zone_utilization_cap is None else args.zone_utilization_cap)
    zone_edge_margin_m = float(
        preset["zone_edge_margin_m"] if args.zone_edge_margin_m is None else args.zone_edge_margin_m
    )
    sink_keepout_margin_m = float(
        preset["sink_keepout_margin_m"]
        if args.sink_keepout_margin_m is None
        else args.sink_keepout_margin_m
    )
    sink_side_clearance_m = float(
        preset["sink_side_clearance_m"]
        if args.sink_side_clearance_m is None
        else args.sink_side_clearance_m
    )
    if zone_edge_margin_m < 0.0 or sink_keepout_margin_m < 0.0 or sink_side_clearance_m < 0.0:
        raise ValueError(
            "zone-edge-margin-m, sink-keepout-margin-m, sink-side-clearance-m must be non-negative"
        )
    if args.pack_clearance_step_m <= 0.0:
        raise ValueError("pack-clearance-step-m must be > 0")
    if args.pack_clearance_floor_m < 0.0:
        raise ValueError("pack-clearance-floor-m must be >= 0")
    if args.pack_clearance_floor_m > min_clearance:
        raise ValueError("pack-clearance-floor-m cannot be greater than base pack-min-clearance")
    if args.grid_step_m <= 0.0:
        raise ValueError("grid-step-m must be > 0")
    if args.frontier_noise_margin_m < 0.0:
        raise ValueError("frontier-noise-margin-m must be >= 0")
    if args.pack_tries_per_clearance <= 0:
        raise ValueError("pack-tries-per-clearance must be > 0")
    if args.pack_min_scale is not None:
        print(
            "[MVP] WARNING: --pack-min-scale is deprecated and ignored in kitchen-bar runner. "
            "Use clearance/cull solver controls instead."
        )
    return {
        "density": args.clutter_density,
        "pack_jitter_xy": jitter_xy,
        "pack_min_clearance": min_clearance,
        "pack_clearance_step_m": float(args.pack_clearance_step_m),
        "pack_clearance_floor_m": float(args.pack_clearance_floor_m),
        "grid_step_m": float(args.grid_step_m),
        "frontier_noise_margin_m": float(args.frontier_noise_margin_m),
        "pack_tries_per_clearance": int(args.pack_tries_per_clearance),
        "zone_padding": zone_padding,
        "zone_util_cap": util_cap,
        "zone_edge_margin_m": zone_edge_margin_m,
        "sink_keepout_margin_m": sink_keepout_margin_m,
        "sink_side_clearance_m": sink_side_clearance_m,
    }


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


def _validate_bddl_initial_conditions(activity_name, activity_definition_id=0):
    cond = Conditions(
        behavior_activity=activity_name,
        activity_definition=activity_definition_id,
        simulator_name="omnigibson",
        predefined_problem=None,
    )
    inst_to_synset = {}
    for synset, insts in cond.parsed_objects.items():
        for inst in insts:
            inst_to_synset[inst] = synset

    kinematic_heads = {"ontop", "inside", "under", "attached", "onfloor", "overlaid"}
    sampleable_with_kinematic = set()
    errors = []

    for init_cond in cond.parsed_initial_conditions:
        if not isinstance(init_cond, list) or len(init_cond) < 2:
            continue
        head = init_cond[0]
        if head == "inroom":
            obj_inst = init_cond[1]
            synset = inst_to_synset.get(obj_inst, None)
            if synset is None:
                continue
            abilities = _OBJECT_TAXONOMY.get_abilities(synset)
            if "sceneObject" not in abilities:
                errors.append(
                    f"invalid_inroom_for_sampleable: inst={obj_inst}, synset={synset}. "
                    "Only non-sampleable scene objects can use inroom."
                )
        elif head in kinematic_heads:
            obj_inst = init_cond[1]
            synset = inst_to_synset.get(obj_inst, None)
            if synset is None:
                continue
            abilities = _OBJECT_TAXONOMY.get_abilities(synset)
            if "sceneObject" not in abilities:
                sampleable_with_kinematic.add(obj_inst)

    for inst, synset in sorted(inst_to_synset.items()):
        abilities = _OBJECT_TAXONOMY.get_abilities(synset)
        if "sceneObject" in abilities or "substance" in abilities:
            continue
        if inst not in sampleable_with_kinematic:
            errors.append(f"missing_sampleable_kinematic_init: inst={inst}, synset={synset}")

    if errors:
        raise RuntimeError(
            "BDDL preflight validation failed:\n- " + "\n- ".join(errors[:20])
        )


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


def _try_get_obj_pose(obj):
    try:
        pos, orn = obj.get_position_orientation()
        pos = _to_float3(pos)
        orn = [float(v) for v in orn]
        if not all(math.isfinite(v) for v in pos + orn):
            return None
        quat_norm = math.sqrt(sum(v * v for v in orn))
        if not (math.isfinite(quat_norm) and quat_norm > 1e-6):
            return None
        if abs(quat_norm - 1.0) > 5e-2:
            # tolerate small numeric drift only
            return None
        return pos, orn
    except Exception:
        return None


def _detect_invalid_object_poses(objects_by_inst):
    invalid = []
    for inst, obj in objects_by_inst.items():
        if _try_get_obj_pose(obj) is None:
            invalid.append(inst)
    return sorted(invalid)


def _freeze_object_motion(obj):
    try:
        if hasattr(obj, "keep_still"):
            obj.keep_still()
            return
    except Exception:
        pass
    try:
        obj.set_linear_velocity(th.zeros(3))
        obj.set_angular_velocity(th.zeros(3))
    except Exception:
        pass


def _freeze_objects(objects_by_inst):
    for obj in objects_by_inst.values():
        _freeze_object_motion(obj)


def _park_inactive_objects(passive_objects, floor_z, bar_bounds):
    if len(passive_objects) == 0:
        return
    (bx0, by0), (bx1, by1) = _normalize_bounds_2d(bar_bounds)
    base_x = bx1 + 1.5
    base_y = by0 - 1.2
    spacing = 0.18
    for idx, inst in enumerate(sorted(passive_objects.keys())):
        obj = passive_objects[inst]
        x = base_x + spacing * (idx % 8)
        y = base_y - spacing * (idx // 8)
        z = floor_z + 0.06
        try:
            obj.set_position_orientation(position=(x, y, z), orientation=(0.0, 0.0, 0.0, 1.0))
            _freeze_object_motion(obj)
        except Exception:
            continue


def _descriptor_radius_xy(descriptor: ClutterObjectDescriptor):
    return max(float(descriptor.half_extent_xy[0]), float(descriptor.half_extent_xy[1]))


def _generate_sorted_grid_points_local(bounds_local, step):
    (x0, y0), (x1, y1) = bounds_local
    x_lo, x_hi = min(x0, x1), max(x0, x1)
    y_lo, y_hi = min(y0, y1), max(y0, y1)
    points = []
    x = x_lo
    while x <= x_hi + 1e-9:
        y = y_lo
        while y <= y_hi + 1e-9:
            points.append((round(float(x), 6), round(float(y), 6)))
            y += step
        x += step
    points.sort(key=lambda p: (math.hypot(p[0], p[1]), abs(p[0]) + abs(p[1]), p[0], p[1]))
    return points


def _candidate_collides_local(candidate_xy, descriptor, placed_local, descriptor_by_inst, min_clearance):
    cx, cy = candidate_xy
    radius = _descriptor_radius_xy(descriptor)
    for inst_id, px, py in placed_local:
        other = descriptor_by_inst[inst_id]
        other_radius = _descriptor_radius_xy(other)
        min_dist = radius + other_radius + min_clearance
        if math.hypot(cx - px, cy - py) < min_dist:
            return True
    return False


def _find_frontier_point_local(
    descriptor,
    placed_local,
    descriptor_by_inst,
    sorted_points_local,
    min_clearance,
    noise_margin,
    rng,
):
    feasible = []
    for px, py in sorted_points_local:
        if _candidate_collides_local(
            candidate_xy=(px, py),
            descriptor=descriptor,
            placed_local=placed_local,
            descriptor_by_inst=descriptor_by_inst,
            min_clearance=min_clearance,
        ):
            continue
        d = math.hypot(px, py)
        feasible.append((d, (px, py)))
    if len(feasible) == 0:
        return None
    d_min = feasible[0][0]
    pool = [xy for d, xy in feasible if d <= d_min + noise_margin + 1e-9]
    if len(pool) == 0:
        return feasible[0][1]
    return pool[int(rng.integers(low=0, high=len(pool)))]


def _entry_world_pose(entry, pack_origin_world, pack_yaw, table_top_z):
    ox, oy, _ = pack_origin_world
    cos_y = math.cos(pack_yaw)
    sin_y = math.sin(pack_yaw)
    rel_x, rel_y, rel_z, qx, qy, qz, qw = entry.rel_pose
    wx = ox + cos_y * rel_x - sin_y * rel_y
    wy = oy + sin_y * rel_x + cos_y * rel_y
    wz = table_top_z + rel_z
    rel_yaw = 2.0 * math.atan2(float(qz), float(qw))
    world_yaw = pack_yaw + rel_yaw
    wqx, wqy, wqz, wqw = _quat_from_yaw(world_yaw)
    return (wx, wy, wz), (wqx, wqy, wqz, wqw)


def _recover_invalid_objects(
    invalid_instances,
    active_objects,
    pack_spec,
    pack_origin_world,
    table_top_z,
    pack_yaw=0.0,
):
    entry_by_inst = {entry.inst_id: entry for entry in pack_spec.object_entries}
    recovered = []
    for inst in invalid_instances:
        obj = active_objects.get(inst, None)
        entry = entry_by_inst.get(inst, None)
        if obj is None or entry is None:
            continue
        try:
            pos, quat = _entry_world_pose(
                entry=entry,
                pack_origin_world=pack_origin_world,
                pack_yaw=pack_yaw,
                table_top_z=table_top_z,
            )
            obj.set_position_orientation(position=pos, orientation=quat)
            _freeze_object_motion(obj)
            recovered.append(inst)
        except Exception:
            continue
    return recovered


def _count_object_interpenetrations(objects_by_inst, tol=0.001):
    inst_ids = sorted(objects_by_inst.keys())
    hits = []
    for i, inst_a in enumerate(inst_ids):
        obj_a = objects_by_inst[inst_a]
        try:
            aabb_a = obj_a.aabb
        except Exception:
            continue
        for inst_b in inst_ids[i + 1 :]:
            obj_b = objects_by_inst[inst_b]
            try:
                aabb_b = obj_b.aabb
            except Exception:
                continue
            if _aabb_overlap_3d(aabb_a, aabb_b, tol=tol):
                hits.append((inst_a, inst_b))
    return hits


def _count_cross_interpenetrations(objects_a, objects_b, tol=0.001):
    hits = []
    for inst_a, obj_a in objects_a.items():
        try:
            aabb_a = obj_a.aabb
        except Exception:
            continue
        for inst_b, obj_b in objects_b.items():
            try:
                aabb_b = obj_b.aabb
            except Exception:
                continue
            if _aabb_overlap_3d(aabb_a, aabb_b, tol=tol):
                hits.append((inst_a, inst_b))
    return hits


def _build_min_clearance_schedule(
    start_clearance: float,
    floor_clearance: float,
    step: float,
) -> Tuple[float, ...]:
    start = max(0.0, float(start_clearance))
    floor = max(0.0, float(floor_clearance))
    decrement = max(1e-6, float(step))
    if floor > start:
        raise ValueError("floor_clearance must be <= start_clearance")
    values = []
    cur = start
    while cur > floor + 1e-9:
        values.append(round(cur, 4))
        cur -= decrement
    values.append(round(floor, 4))
    dedup = []
    for v in values:
        if v not in dedup:
            dedup.append(v)
    return tuple(dedup)


def _local_bounds_from_zone(zone_bounds):
    (x0, y0), (x1, y1) = _normalize_bounds_2d(zone_bounds)
    half_x = 0.5 * (x1 - x0)
    half_y = 0.5 * (y1 - y0)
    return ((-half_x, -half_y), (half_x, half_y))


def _select_cull_candidate(
    active_descriptors: Sequence[ClutterObjectDescriptor],
    last_pack_spec: Optional[ClutterPackSpec],
) -> Optional[Tuple[str, str, float]]:
    # Never cull target. Prefer outermost clutter, then outermost fragile.
    if len(active_descriptors) == 0:
        return None
    candidates = []
    if last_pack_spec is not None:
        for entry in last_pack_spec.object_entries:
            if entry.role == "target":
                continue
            radius = math.hypot(entry.rel_pose[0], entry.rel_pose[1])
            candidates.append((entry.inst_id, entry.role, radius))
    else:
        for d in active_descriptors:
            if d.role == "target":
                continue
            candidates.append((d.instance_id, d.role, 0.0))
    if len(candidates) == 0:
        return None

    def _priority(item):
        inst, role, radius = item
        role_rank = 0 if role == "clutter" else (1 if role == "fragile" else 9)
        # Outer objects first for culling.
        return (role_rank, -float(radius), inst)

    candidates.sort(key=_priority)
    return candidates[0]


def _reintroduce_culled_descriptors(
    culled_descriptors,
    descriptor_by_inst_all,
    objects_by_inst_all,
    active_descriptors,
    active_objects_by_inst,
    pack_spec,
    world_positions,
    pack_origin,
    table_top_z,
    zone,
    floor_z,
    clutter_controls,
    rng,
):
    if len(culled_descriptors) == 0:
        return pack_spec, active_descriptors, active_objects_by_inst, world_positions, []

    # Smaller objects first usually yields better recovery count.
    pending = sorted(
        [d for d in culled_descriptors if d.instance_id not in active_objects_by_inst],
        key=lambda d: (_descriptor_radius_xy(d), d.instance_id),
    )
    sorted_points_local = _generate_sorted_grid_points_local(
        bounds_local=_local_bounds_from_zone(zone.red_zone_bounds),
        step=clutter_controls["grid_step_m"],
    )
    min_clearance = float(clutter_controls["pack_clearance_floor_m"])
    noise_margin = float(clutter_controls["frontier_noise_margin_m"])
    readd_history = []

    # Local placements from current pack.
    placed_local = [
        (entry.inst_id, float(entry.rel_pose[0]), float(entry.rel_pose[1]))
        for entry in pack_spec.object_entries
        if entry.inst_id in active_objects_by_inst
    ]

    for desc in pending:
        chosen = _find_frontier_point_local(
            descriptor=desc,
            placed_local=placed_local,
            descriptor_by_inst=descriptor_by_inst_all,
            sorted_points_local=sorted_points_local,
            min_clearance=min_clearance,
            noise_margin=noise_margin,
            rng=rng,
        )
        if chosen is None:
            readd_history.append(
                {"inst_id": desc.instance_id, "accepted": False, "reason": "no_feasible_frontier_point"}
            )
            continue

        rel_x, rel_y = float(chosen[0]), float(chosen[1])
        rel_z = max(0.008, 0.5 * max(float(desc.height), 0.01) + 0.004)
        yaw = 0.0 if desc.role == "target" else float(rng.uniform(-0.18, 0.18))
        qx, qy, qz, qw = _quat_from_yaw(yaw)
        entry = ClutterPackEntry(
            inst_id=desc.instance_id,
            role=desc.role,
            rel_pose=(rel_x, rel_y, rel_z, qx, qy, qz, qw),
        )
        obj = active_objects_by_inst.get(desc.instance_id, None)
        if obj is None:
            obj = objects_by_inst_all.get(desc.instance_id, None)
            if obj is None:
                readd_history.append({"inst_id": desc.instance_id, "accepted": False, "reason": "missing_scope_object"})
                continue

        wx = pack_origin[0] + rel_x
        wy = pack_origin[1] + rel_y
        wz = table_top_z + rel_z
        try:
            obj.set_position_orientation(position=(wx, wy, wz), orientation=(qx, qy, qz, qw))
            _freeze_object_motion(obj)
            for _ in range(2):
                og.sim.step()
                _freeze_object_motion(obj)
        except Exception as e:
            readd_history.append({"inst_id": desc.instance_id, "accepted": False, "reason": f"set_pose_failed:{e}"})
            _park_inactive_objects({desc.instance_id: obj}, floor_z=floor_z, bar_bounds=zone.bar_bounds)
            continue

        invalid = _detect_invalid_object_poses({desc.instance_id: obj})
        if invalid:
            readd_history.append({"inst_id": desc.instance_id, "accepted": False, "reason": "invalid_pose"})
            _park_inactive_objects({desc.instance_id: obj}, floor_z=floor_z, bar_bounds=zone.bar_bounds)
            continue

        cross = _count_cross_interpenetrations(
            {desc.instance_id: obj},
            {k: v for k, v in active_objects_by_inst.items() if k != desc.instance_id},
            tol=0.001,
        )
        if cross:
            readd_history.append(
                {
                    "inst_id": desc.instance_id,
                    "accepted": False,
                    "reason": f"cross_interpenetration:{cross[:4]}",
                }
            )
            _park_inactive_objects({desc.instance_id: obj}, floor_z=floor_z, bar_bounds=zone.bar_bounds)
            continue

        zone_report = _evaluate_object_zone_constraints(
            objects_by_inst={desc.instance_id: obj},
            red_zone_bounds=zone.red_zone_bounds,
            sink_keepout_bounds=zone.sink_keepout_bounds,
        )
        if not (zone_report.all_in_red_zone and zone_report.all_outside_sink_keepout):
            readd_history.append(
                {"inst_id": desc.instance_id, "accepted": False, "reason": "zone_or_keepout_violation"}
            )
            _park_inactive_objects({desc.instance_id: obj}, floor_z=floor_z, bar_bounds=zone.bar_bounds)
            continue

        # Accept re-add.
        active_descriptors.append(desc)
        active_objects_by_inst[desc.instance_id] = obj
        world_positions[desc.instance_id] = (wx, wy, wz)
        placed_local.append((desc.instance_id, rel_x, rel_y))
        pack_spec = ClutterPackSpec(
            table_obj_name=pack_spec.table_obj_name,
            pack_origin_world=pack_spec.pack_origin_world,
            object_entries=tuple(list(pack_spec.object_entries) + [entry]),
            seed=pack_spec.seed,
            template_id=pack_spec.template_id,
        )
        readd_history.append(
            {"inst_id": desc.instance_id, "accepted": True, "reason": "ok", "rel_xy": [rel_x, rel_y]}
        )

    return pack_spec, active_descriptors, active_objects_by_inst, world_positions, readd_history


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


def _apply_matte_support_material(support):
    if support is None or getattr(support, "name", None) != HARDCODE_SUPPORT_NAME:
        return

    model = getattr(support, "model", None)
    category = getattr(support, "category", None)
    if not model or not category:
        raise RuntimeError(f"Support '{HARDCODE_SUPPORT_NAME}' is missing category/model metadata for matte override.")

    material_dir = os.path.join(get_dataset_path("behavior-1k-assets"), "objects", category, model, "material")
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()
    looks_path = f"{support.prim_path}/Looks"
    if not lazy.isaacsim.core.utils.prims.is_prim_path_valid(looks_path):
        stage.DefinePrim(looks_path, "Scope")

    matte_materials = set()
    for link_name, link in support.links.items():
        textures = {}
        texture_prefix = f"{model}__{link_name}__"
        if os.path.isdir(material_dir):
            for fname in os.listdir(material_dir):
                if not fname.startswith(texture_prefix):
                    continue
                texture_key, ext = os.path.splitext(fname[len(texture_prefix) :])
                if ext.lower() != ".png":
                    continue
                textures[texture_key] = os.path.join(material_dir, fname)

        safe_link_name = link_name if link_name and link_name[0].isalpha() else f"link_{link_name}"
        matte_material_path = f"{looks_path}/matte_{safe_link_name}_pbr"
        if matte_material_path in MaterialPrim.MATERIALS and not lazy.isaacsim.core.utils.prims.is_prim_path_valid(
            matte_material_path
        ):
            MaterialPrim.MATERIALS.pop(matte_material_path, None)

        matte_material = OmniPBRMaterialPrim.get_material(
            scene=support.scene,
            prim_path=matte_material_path,
            name=f"{support.name}:{link_name}:matte_material",
        )
        matte_material.set_input(inp="reflection_roughness_texture_influence", val=0.0)
        matte_material.set_input(inp="reflection_roughness_constant", val=1.0)
        matte_material.set_input(inp="metallic_texture_influence", val=0.0)
        if "metallic_constant" in matte_material.shader_input_names or "metallic_constant" in matte_material.shader_default_input_names:
            matte_material.set_input(inp="metallic_constant", val=0.0)

        diffuse_path = textures.get("diffuse")
        if diffuse_path:
            matte_material.diffuse_texture = diffuse_path
        else:
            matte_material.diffuse_color_constant = th.tensor((0.7, 0.7, 0.7), dtype=th.float32)

        normal_path = textures.get("normal")
        if normal_path:
            matte_material.set_input(inp="normalmap_texture", val=lazy.pxr.Sdf.AssetPath(normal_path))

        if not link.visual_meshes:
            matte_materials.add(matte_material)
            continue

        for visual_mesh in link.visual_meshes.values():
            current_material = visual_mesh.material
            if current_material is not None and current_material.prim_path != matte_material.prim_path:
                try:
                    current_material.remove_user(visual_mesh)
                except Exception:
                    pass
            visual_mesh.material = matte_material
            if visual_mesh not in matte_material.users:
                matte_material.add_user(visual_mesh)
        matte_materials.add(matte_material)

    support._materials = set(matte_materials)


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
    # Backward-compat shim. XY scaling is intentionally disabled because it can
    # compress object centers without shrinking collision geometry.
    if scale_xy < 0.999:
        raise RuntimeError("pack_xy_scaling_disabled: scale<1.0 is forbidden in kitchen-bar runner")
    return pack_spec


def _fit_pack_to_zone(pack_spec, descriptor_by_inst, red_zone_bounds):
    rel_bounds = _compute_pack_relative_bounds(pack_spec, descriptor_by_inst)
    origin_xy = _choose_pack_origin_in_zone(red_zone_bounds, rel_bounds)
    return pack_spec, rel_bounds, origin_xy


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
        pose = _try_get_obj_pose(obj)
        if pose is None:
            print(f"[MVP]   {inst} -> {getattr(obj, 'name', 'unknown')} @ INVALID_POSE")
            continue
        pos, _ = pose
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


def _sim_step_with_action_only(env, action):
    env._pre_step(action)
    og.sim.step()


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
    clutter_controls = _resolve_clutter_controls(args)

    task_name = cfg["task"]["activity_name"]
    _validate_bddl_initial_conditions(
        activity_name=task_name,
        activity_definition_id=cfg["task"].get("activity_definition_id", 0),
    )
    task_spec = build_manipulation_task_spec(task_name)

    print(
        f"[MVP] Configured: task={task_name}, scene={cfg['scene']['scene_model']}, "
        f"workspace={WORKSPACE_PRESET}, strict_gate={args.strict_gate}, steps={args.steps}"
    )
    print(
        "[MVP] clutter_controls: "
        f"density={clutter_controls['density']}, "
        f"pack_jitter_xy={clutter_controls['pack_jitter_xy']:.3f}, "
        f"pack_min_clearance={clutter_controls['pack_min_clearance']:.3f}, "
        f"pack_clearance_step_m={clutter_controls['pack_clearance_step_m']:.3f}, "
        f"pack_clearance_floor_m={clutter_controls['pack_clearance_floor_m']:.3f}, "
        f"grid_step_m={clutter_controls['grid_step_m']:.3f}, "
        f"frontier_noise_margin_m={clutter_controls['frontier_noise_margin_m']:.3f}, "
        f"pack_tries_per_clearance={clutter_controls['pack_tries_per_clearance']}, "
        f"zone_padding={clutter_controls['zone_padding']:.3f}, "
        f"zone_util_cap={clutter_controls['zone_util_cap']:.3f}, "
        f"zone_edge_margin={clutter_controls['zone_edge_margin_m']:.3f}, "
        f"sink_keepout_margin={clutter_controls['sink_keepout_margin_m']:.3f}, "
        f"sink_side_clearance={clutter_controls['sink_side_clearance_m']:.3f}"
    )
    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"[MVP] Episode {ep + 1}/{args.episodes} reset")
            env.reset()
            og.sim.step()

            support, sink = _resolve_support_and_sink(env)
            _apply_matte_support_material(support)
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
            floor_z = _compute_floor_z(env)

            zone = compute_kitchen_bar_zone(
                bar_bounds_xy=bar_bounds_xy,
                sink_bounds_xy=sink_bounds_xy,
                workspace_preset=WORKSPACE_PRESET,
                edge_margin_m=clutter_controls["zone_edge_margin_m"],
                sink_keepout_margin_m=clutter_controls["sink_keepout_margin_m"],
                sink_side_clearance_m=clutter_controls["sink_side_clearance_m"],
                min_zone_span_m=0.20,
            )
            _log_workspace_geometry(zone, support, sink)

            obj_sets = _build_task_object_sets(env, task_spec)
            if len(obj_sets["target_ids"]) == 0:
                raise RuntimeError("No target objects found in object_scope.")
            if len(obj_sets["clutter_ids"]) == 0:
                print(
                    "[MVP] NOTE: clutter_set is empty. Add more objects in BDDL :objects/:init "
                    "to increase clutter count."
                )
            target_inst = obj_sets["target_ids"][0]
            target_obj = _get_scope_obj(env, target_inst)

            descriptors: List[ClutterObjectDescriptor] = []
            for inst in obj_sets["target_ids"]:
                obj = _get_scope_obj(env, inst)
                if obj is None:
                    continue
                try:
                    descriptors.append(_descriptor_from_obj(inst, "target", obj))
                except Exception as e:
                    print(f"[MVP] WARNING: descriptor_skip target {inst}: {e}")
            for inst in obj_sets["fragile_ids"]:
                obj = _get_scope_obj(env, inst)
                if obj is None:
                    continue
                try:
                    descriptors.append(_descriptor_from_obj(inst, "fragile", obj))
                except Exception as e:
                    print(f"[MVP] WARNING: descriptor_skip fragile {inst}: {e}")
            for inst in obj_sets["clutter_ids"]:
                obj = _get_scope_obj(env, inst)
                if obj is None:
                    continue
                try:
                    descriptors.append(_descriptor_from_obj(inst, "clutter", obj))
                except Exception as e:
                    print(f"[MVP] WARNING: descriptor_skip clutter {inst}: {e}")

            if len(descriptors) == 0:
                raise RuntimeError("No clutter-pack descriptors were created.")

            objects_by_inst = {}
            for d in descriptors:
                obj = _get_scope_obj(env, d.instance_id)
                if obj is not None:
                    objects_by_inst[d.instance_id] = obj

            clearance_schedule = _build_min_clearance_schedule(
                start_clearance=clutter_controls["pack_min_clearance"],
                floor_clearance=clutter_controls["pack_clearance_floor_m"],
                step=clutter_controls["pack_clearance_step_m"],
            )
            print(
                f"[MVP] clearance_schedule={list(clearance_schedule)} "
                f"(step={clutter_controls['pack_clearance_step_m']:.3f}, "
                f"floor={clutter_controls['pack_clearance_floor_m']:.3f})"
            )

            world_positions = None
            pack_spec = None
            pack_origin = None
            pack_attempt_used = None
            chosen_min_clearance = None
            last_pack_error = None
            zone_capacity = None
            active_descriptors = list(descriptors)
            active_objects_by_inst = None
            cull_history = []
            last_pack_for_cull = None
            placement_bounds_local = _local_bounds_from_zone(zone.red_zone_bounds)

            attempt_idx = 0
            solved = False
            while not solved:
                descriptor_by_inst = {d.instance_id: d for d in active_descriptors}
                active_objects = {
                    d.instance_id: objects_by_inst[d.instance_id]
                    for d in active_descriptors
                    if d.instance_id in objects_by_inst
                }
                if len(active_objects) != len(active_descriptors):
                    missing = sorted(set(descriptor_by_inst.keys()) - set(active_objects.keys()))
                    raise RuntimeError(f"missing_scope_objects_for_active_set: {missing}")
                passive_objects = {inst: obj for inst, obj in objects_by_inst.items() if inst not in active_objects}
                _park_inactive_objects(passive_objects=passive_objects, floor_z=floor_z, bar_bounds=zone.bar_bounds)
                og.sim.step()
                _freeze_objects(active_objects)

                zone_capacity_subset: ZoneCapacityStats = compute_zone_capacity(
                    red_zone_bounds=zone.red_zone_bounds,
                    half_extents_xy=[d.half_extent_xy for d in active_descriptors],
                    per_object_padding=clutter_controls["zone_padding"],
                )
                zone_capacity = zone_capacity_subset
                print(
                    "[MVP] zone_capacity: "
                    f"active_count={len(active_descriptors)}, "
                    f"required={zone_capacity_subset.required_area:.4f}, "
                    f"available={zone_capacity_subset.available_area:.4f}, "
                    f"utilization={zone_capacity_subset.utilization:.3f}"
                )
                if zone_capacity_subset.utilization > clutter_controls["zone_util_cap"]:
                    print(
                        "[MVP] WARNING: red-zone utilization exceeds threshold; "
                        f"utilization={zone_capacity_subset.utilization:.3f}, "
                        f"cap={clutter_controls['zone_util_cap']:.3f}"
                    )

                solved_this_round = False
                for level_idx, min_clearance in enumerate(clearance_schedule, start=1):
                    for trial_idx in range(1, clutter_controls["pack_tries_per_clearance"] + 1):
                        attempt_idx += 1
                        jitter_xy = max(0.002, float(clutter_controls["pack_jitter_xy"]))
                        try:
                            print(
                                "[MVP] pack_attempt: "
                                f"id={attempt_idx}, active_count={len(active_descriptors)}, "
                                f"level={level_idx}/{len(clearance_schedule)}, "
                                f"trial={trial_idx}/{clutter_controls['pack_tries_per_clearance']}, "
                                f"jitter_xy={jitter_xy:.3f}, min_clearance={min_clearance:.3f}"
                            )
                            pack_spec_candidate = build_clutter_pack(
                                table_obj_name=getattr(support, "name", HARDCODE_SUPPORT_NAME),
                                descriptors=active_descriptors,
                                seed=args.seed + ep + attempt_idx * 101,
                                jitter_xy=jitter_xy,
                                min_clearance=min_clearance,
                                placement_bounds_local=placement_bounds_local,
                                grid_step_m=clutter_controls["grid_step_m"],
                                frontier_noise_margin_m=clutter_controls["frontier_noise_margin_m"],
                                shuffle_non_target=True,
                            )
                            last_pack_for_cull = pack_spec_candidate
                            pack_spec_candidate, _, origin_xy = _fit_pack_to_zone(
                                pack_spec=pack_spec_candidate,
                                descriptor_by_inst=descriptor_by_inst,
                                red_zone_bounds=zone.red_zone_bounds,
                            )
                            pack_origin_candidate = (origin_xy[0], origin_xy[1], table_top_z)
                            world_positions_candidate = apply_pack_transform(
                                pack_spec=pack_spec_candidate,
                                objects_by_inst=active_objects,
                                pack_origin_world=pack_origin_candidate,
                                pack_yaw=0.0,
                                table_top_z=table_top_z,
                            )
                            _freeze_objects(active_objects)
                            for _ in range(3):
                                og.sim.step()
                                _freeze_objects(active_objects)

                            invalid_after_settle = _detect_invalid_object_poses(active_objects)
                            if invalid_after_settle:
                                recovered = _recover_invalid_objects(
                                    invalid_instances=invalid_after_settle,
                                    active_objects=active_objects,
                                    pack_spec=pack_spec_candidate,
                                    pack_origin_world=pack_origin_candidate,
                                    table_top_z=table_top_z,
                                    pack_yaw=0.0,
                                )
                                if recovered:
                                    print(
                                        "[MVP] WARNING: recovered invalid objects after settle "
                                        f"attempt={attempt_idx}, recovered={recovered}"
                                    )
                                    for _ in range(2):
                                        og.sim.step()
                                        _freeze_objects(active_objects)
                                    invalid_after_settle = _detect_invalid_object_poses(active_objects)
                                if invalid_after_settle:
                                    last_pack_error = (
                                        "invalid_object_pose_after_settle:"
                                        f"attempt={attempt_idx}, objects={invalid_after_settle}"
                                    )
                                    print(f"[MVP] WARNING: {last_pack_error}")
                                    continue

                            penetration_pairs = _count_object_interpenetrations(active_objects, tol=0.001)
                            if penetration_pairs:
                                preview = penetration_pairs[:8]
                                last_pack_error = (
                                    "object_interpenetration_after_settle:"
                                    f"attempt={attempt_idx}, count={len(penetration_pairs)}, preview={preview}"
                                )
                                print(f"[MVP] WARNING: {last_pack_error}")
                                continue

                            cross_penetration_pairs = _count_cross_interpenetrations(active_objects, passive_objects, tol=0.001)
                            if cross_penetration_pairs:
                                preview = cross_penetration_pairs[:8]
                                last_pack_error = (
                                    "active_passive_interpenetration_after_settle:"
                                    f"attempt={attempt_idx}, count={len(cross_penetration_pairs)}, preview={preview}"
                                )
                                print(f"[MVP] WARNING: {last_pack_error}")
                                continue

                            zone_report_candidate = _evaluate_object_zone_constraints(
                                objects_by_inst=active_objects,
                                red_zone_bounds=zone.red_zone_bounds,
                                sink_keepout_bounds=zone.sink_keepout_bounds,
                            )
                            if not (
                                zone_report_candidate.all_in_red_zone and zone_report_candidate.all_outside_sink_keepout
                            ):
                                last_pack_error = (
                                    "zone_constraint_violation_after_pack:"
                                    f"attempt={attempt_idx}, out_of_zone={list(zone_report_candidate.out_of_zone_instances)}, "
                                    f"in_keepout={list(zone_report_candidate.in_keepout_instances)}"
                                )
                                print(f"[MVP] WARNING: {last_pack_error}")
                                continue

                            pack_spec = pack_spec_candidate
                            world_positions = world_positions_candidate
                            pack_origin = pack_origin_candidate
                            pack_attempt_used = attempt_idx
                            chosen_min_clearance = min_clearance
                            active_objects_by_inst = active_objects
                            solved = True
                            solved_this_round = True
                            break
                        except Exception as e:
                            last_pack_error = str(e)
                            print(f"[MVP] WARNING: pack_attempt_failed id={attempt_idx}: {e}")
                    if solved_this_round:
                        break

                if solved_this_round:
                    break

                cull = _select_cull_candidate(active_descriptors, last_pack_for_cull)
                if cull is None:
                    break
                cull_inst, cull_role, cull_radius = cull
                kept = [d for d in active_descriptors if d.instance_id != cull_inst]
                if len(kept) == len(active_descriptors):
                    break
                active_descriptors = kept
                cull_history.append(
                    {
                        "attempt": attempt_idx,
                        "inst_id": cull_inst,
                        "role": cull_role,
                        "radius": float(cull_radius),
                        "remaining": len(active_descriptors),
                    }
                )
                print(
                    "[MVP] cull_step: "
                    f"removed={cull_inst} role={cull_role} radius={cull_radius:.3f}, "
                    f"remaining={len(active_descriptors)}"
                )

            if world_positions is None or pack_spec is None or active_objects_by_inst is None:
                raise RuntimeError(f"pack_generation_failed_after_retries: {last_pack_error}")

            descriptor_by_inst_all = {d.instance_id: d for d in descriptors}
            culled_ids = [item["inst_id"] for item in cull_history]
            culled_descriptors = [descriptor_by_inst_all[i] for i in culled_ids if i in descriptor_by_inst_all]
            (
                pack_spec,
                active_descriptors,
                active_objects_by_inst,
                world_positions,
                readd_history,
            ) = _reintroduce_culled_descriptors(
                culled_descriptors=culled_descriptors,
                descriptor_by_inst_all=descriptor_by_inst_all,
                objects_by_inst_all=objects_by_inst,
                active_descriptors=active_descriptors,
                active_objects_by_inst=active_objects_by_inst,
                pack_spec=pack_spec,
                world_positions=world_positions,
                pack_origin=pack_origin,
                table_top_z=table_top_z,
                zone=zone,
                floor_z=floor_z,
                clutter_controls=clutter_controls,
                rng=rng,
            )

            # Keep non-active objects away from workspace so they do not leak into gate / mount checks.
            passive_after_solve = {inst: obj for inst, obj in objects_by_inst.items() if inst not in active_objects_by_inst}
            _park_inactive_objects(passive_objects=passive_after_solve, floor_z=floor_z, bar_bounds=zone.bar_bounds)
            og.sim.step()
            _freeze_objects(active_objects_by_inst)

            print(
                f"[MVP] pack_origin={pack_origin}, "
                f"pack_attempt_used={pack_attempt_used}, selected_min_clearance={chosen_min_clearance:.3f}, "
                f"active_descriptor_count={len(active_descriptors)}, cull_steps={len(cull_history)}, "
                f"readd_success={sum(1 for r in readd_history if r.get('accepted'))}"
            )

            integrity = validate_pack_integrity(
                pack_spec=pack_spec,
                world_positions=world_positions,
                pack_origin_world=pack_origin,
                pack_yaw=0.0,
                tol_xy=0.035,
            )

            zone_report = _evaluate_object_zone_constraints(
                objects_by_inst=active_objects_by_inst,
                red_zone_bounds=zone.red_zone_bounds,
                sink_keepout_bounds=zone.sink_keepout_bounds,
            )

            robot = env.robots[0]
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
            target_pose = _try_get_obj_pose(target_obj)
            if target_pose is None:
                if target_inst in world_positions:
                    target_pos = list(world_positions[target_inst])
                    print(
                        "[MVP] WARNING: target pose invalid after physics step; "
                        f"falling back to planned pose for {target_inst}: {target_pos}"
                    )
                else:
                    raise RuntimeError(
                        f"target_pose_invalid_and_missing_fallback: {target_inst}. "
                        "Try lower clutter density or increase --pack-min-clearance."
                    )
            else:
                target_pos, _ = target_pose

            robot_pose = _try_get_obj_pose(robot)
            if robot_pose is None:
                raise RuntimeError("robot_pose_invalid_after_mount")
            robot_pos, _ = robot_pose
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
                    "clutter_controls": clutter_controls,
                    "support_name": HARDCODE_SUPPORT_NAME,
                    "sink_name": HARDCODE_SINK_NAME,
                    "bar_bounds": zone.bar_bounds,
                    "red_zone_bounds": zone.red_zone_bounds,
                    "sink_keepout_bounds": zone.sink_keepout_bounds,
                    "zone_capacity_stats": asdict(zone_capacity),
                    "clearance_schedule": list(clearance_schedule),
                    "clearance_floor_m": clutter_controls["pack_clearance_floor_m"],
                    "clearance_step_m": clutter_controls["pack_clearance_step_m"],
                    "tries_per_clearance": clutter_controls["pack_tries_per_clearance"],
                    "cull_history": cull_history,
                    "readd_history": readd_history,
                    "final_active_set": [d.instance_id for d in active_descriptors],
                    "active_descriptor_count": len(active_descriptors),
                    "fixed_edge_mount": {
                        "edge_label": FIXED_EDGE_LABEL,
                        "scan_offsets": list(FIXED_EDGE_SCAN_OFFSETS),
                    },
                    "pack_origin_world": list(pack_origin),
                    "pack_attempt_used": pack_attempt_used,
                    "selected_min_clearance": chosen_min_clearance,
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
                action = _make_zero_jitter_action(robot, rng, args.jitter_scale)
                _sim_step_with_action_only(env, action)
                executed += 1
                if executed % 50 == 0:
                    print(f"[MVP] Step {executed}/{args.steps}")
                if executed % 10 == 0:
                    invalid_step = _detect_invalid_object_poses(active_objects_by_inst)
                    if invalid_step:
                        raise RuntimeError(
                            "invalid_object_pose_during_rollout: "
                            f"step={executed}, objects={invalid_step}. "
                            "Try lower clutter density or increase --pack-min-clearance."
                        )
                if executed % 25 == 0:
                    robot_hits = _collect_robot_penetration_hits(env)
                    if robot_hits:
                        print(f"[MVP] WARNING: runtime robot penetration objects={robot_hits}")
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
