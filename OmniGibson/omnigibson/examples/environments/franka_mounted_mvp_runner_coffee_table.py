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
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_scene_path
from omnigibson.utils.clutter_pack_layout import (
    ClutterObjectDescriptor,
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
    select_best_table_edge,
)
from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec
import omnigibson.utils.transform_utils as T


# Stable defaults for local GUI + MVP workflow.
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False

_OBJECT_TAXONOMY = ObjectTaxonomy()

HARDCODE_SCENE_MODEL = "house_double_floor_lower"
HARDCODE_TABLE_NAME = "coffee_table_koagbh_0"
HARDCODE_TABLE_CATEGORY = "coffee_table"


@dataclass(frozen=True)
class MVPGateReport:
    scene_sane: bool
    base_on_ground: bool
    base_collision_free: bool
    gap_ok: bool
    target_in_reach_band: bool
    pack_integrity_ok: bool
    pass_gate: bool
    failure_reasons: Tuple[str, ...]


def parse_args():
    parser = argparse.ArgumentParser(description="Cup-first FrankaMounted MVP runner (simple, strict, reproducible).")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--scene-model", default=None, help="Scene model override.")
    parser.add_argument("--activity-name", default=None, help="BehaviorTask activity_name override.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("--steps", type=int, default=5000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--mount-gap-m", type=float, default=0.03, help="Desired Franka base gap from table edge.")
    parser.add_argument(
        "--placement-table-name",
        default=HARDCODE_TABLE_NAME,
        help="Hardcoded table object name for this cup-first inspect runner.",
    )
    parser.add_argument("--jitter-scale", type=float, default=0.01, help="Action jitter sigma for zero_jitter mode.")
    parser.add_argument("--showcase-gui", action="store_true", help="Enable manual GUI camera teleoperation.")
    parser.add_argument("--strict-gate", dest="strict_gate", action="store_true", help="Enable strict gate.")
    parser.add_argument("--no-strict-gate", dest="strict_gate", action="store_false", help="Disable strict gate.")
    parser.set_defaults(strict_gate=True)
    parser.add_argument("--debug-jsonl", default=None, help="Optional JSONL path for reset diagnostics.")
    return parser.parse_args()


def _default_config_path():
    return os.path.join(og.root_path, "configs", "franka_mounted_behavior_cached.yaml")


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

    if args.scene_model:
        cfg["scene"]["scene_model"] = args.scene_model
    if args.activity_name:
        cfg["task"]["activity_name"] = args.activity_name

    scene_model = cfg["scene"]["scene_model"]
    if scene_model != HARDCODE_SCENE_MODEL:
        raise RuntimeError(
            f"This runner is hardcoded for scene_model='{HARDCODE_SCENE_MODEL}', got '{scene_model}'. "
            "Switch scene or use another runner."
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


def _is_table_like(category):
    text = str(category).lower()
    return any(tok in text for tok in ("table", "counter", "desk"))


def _resolve_hardcoded_table(env, table_name):
    try:
        obj = env.scene.object_registry("name", table_name)
    except Exception:
        obj = None
    if obj is None:
        raise RuntimeError(f"Hardcoded table '{table_name}' not found in scene objects.")
    category = str(getattr(obj, "category", ""))
    if category != HARDCODE_TABLE_CATEGORY:
        raise RuntimeError(
            f"Hardcoded table '{table_name}' category mismatch: expected '{HARDCODE_TABLE_CATEGORY}', got '{category}'."
        )
    return table_name, obj


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
        # Hard fallback: cup-like objects in scope.
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


def _score_support_candidate(target_obj, support_obj):
    try:
        target_pos = _to_float3(target_obj.get_position_orientation()[0])
        aabb_min, aabb_max = support_obj.aabb
        aabb_min = _to_float3(aabb_min)
        aabb_max = _to_float3(aabb_max)
    except Exception:
        return -1e9

    center_xy = [(aabb_min[0] + aabb_max[0]) * 0.5, (aabb_min[1] + aabb_max[1]) * 0.5]
    dist_xy = _distance_xy(target_pos[:2], center_xy)
    top_z = aabb_max[2]
    dz = abs(target_pos[2] - top_z)
    inside_xy = (
        (aabb_min[0] - 0.05) <= target_pos[0] <= (aabb_max[0] + 0.05)
        and (aabb_min[1] - 0.05) <= target_pos[1] <= (aabb_max[1] + 0.05)
    )

    on_top = False
    if object_states.OnTop in target_obj.states:
        try:
            on_top = bool(target_obj.states[object_states.OnTop].get_value(support_obj))
        except Exception:
            on_top = False

    score = 0.0
    score += 3.0 if on_top else 0.0
    score += 2.0 if inside_xy else 0.0
    score += 1.0 if _is_table_like(getattr(support_obj, "category", "")) else 0.0
    score -= 0.25 * dist_xy
    score -= 1.2 * dz
    return score


def _select_support_table(env, target_inst, support_ids, preferred_table_name=None):
    target_obj = _get_scope_obj(env, target_inst)
    if target_obj is None:
        raise RuntimeError(f"Target object missing from scope: {target_inst}")

    if preferred_table_name:
        pref = preferred_table_name.lower().strip()
        preferred = []
        for obj in getattr(env.scene, "objects", []):
            if obj is None or not _is_table_like(getattr(obj, "category", "")):
                continue
            name = str(getattr(obj, "name", "")).lower()
            cat = str(getattr(obj, "category", "")).lower()
            if pref in name or pref in cat:
                preferred.append((getattr(obj, "name", "unknown_table"), obj))
        if preferred:
            best_pref = None
            best_pref_score = -1e9
            for inst, obj in preferred:
                score = _score_support_candidate(target_obj, obj)
                if score > best_pref_score:
                    best_pref = (inst, obj)
                    best_pref_score = score
            if best_pref is not None:
                return best_pref

    best = None
    best_score = -1e9
    for inst in support_ids:
        obj = _get_scope_obj(env, inst)
        if obj is None:
            continue
        score = _score_support_candidate(target_obj, obj)
        if score > best_score:
            best = (inst, obj)
            best_score = score

    if best is None:
        for obj in getattr(env.scene, "objects", []):
            if obj is None or not _is_table_like(getattr(obj, "category", "")):
                continue
            score = _score_support_candidate(target_obj, obj)
            if score > best_score:
                best = (getattr(obj, "name", "unknown_table"), obj)
                best_score = score

    if best is None:
        raise RuntimeError("Unable to resolve tabletop support object for clutter pack.")
    return best


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


def _compute_pack_radius(pack_spec: ClutterPackSpec, descriptor_by_inst):
    radius = 0.15
    for entry in pack_spec.object_entries:
        d = descriptor_by_inst[entry.inst_id]
        x, y = entry.rel_pose[0], entry.rel_pose[1]
        r = math.hypot(x, y) + max(d.half_extent_xy[0], d.half_extent_xy[1]) + 0.02
        radius = max(radius, r)
    return radius


def _clamp_pack_origin(origin_xy, table_aabb_xy, pack_radius, margin=0.04):
    (x_min, y_min), (x_max, y_max) = table_aabb_xy
    x_lo = x_min + pack_radius + margin
    x_hi = x_max - pack_radius - margin
    y_lo = y_min + pack_radius + margin
    y_hi = y_max - pack_radius - margin
    if x_lo > x_hi:
        x_lo = x_hi = 0.5 * (x_min + x_max)
    if y_lo > y_hi:
        y_lo = y_hi = 0.5 * (y_min + y_max)
    x = min(max(origin_xy[0], x_lo), x_hi)
    y = min(max(origin_xy[1], y_lo), y_hi)
    return x, y


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
        base_link, _ = _get_robot_base_link(robot)
        robot_paths = set(getattr(robot, "link_prim_paths", []))
        for contact in base_link.contact_list():
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
        return sorted(hits)
    except Exception:
        # Fallback to conservative AABB only if contact path resolution fails.
        base_aabb = _get_robot_base_aabb(robot)
        base_min = _to_float3(base_aabb[0])
        base_max = _to_float3(base_aabb[1])
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
            if obj_min[2] > base_max[2] + 0.12:
                continue
            if _aabb_overlap_3d((base_min, base_max), (obj_min, obj_max), tol=0.002):
                hits.add(getattr(obj, "name", "unknown_obj"))
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


def _print_table_candidates(env):
    print("[MVP] Table candidates in scene:")
    count = 0
    for obj in getattr(env.scene, "objects", []):
        if obj is None or not _is_table_like(getattr(obj, "category", "")):
            continue
        try:
            pos = _to_float3(obj.get_position_orientation()[0])
        except Exception:
            pos = [float("nan"), float("nan"), float("nan")]
        print(
            f"[MVP]   table={getattr(obj, 'name', 'unknown_table')} "
            f"cat={getattr(obj, 'category', 'unknown')} pos={pos}"
        )
        count += 1
    if count == 0:
        print("[MVP]   <none>")


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
    cam_pos = [center[0] - 1.2, center[1] - 1.0, center[2] + 0.6]
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
    # High-visibility upright-ish arm posture for manual scene inspection.
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


def _edge_try_order(table_aabb_xy, pack_objects_world):
    nearest = select_best_table_edge(table_aabb_xy, pack_objects_world)
    all_edges = ("x_min", "x_max", "y_min", "y_max")
    return [nearest] + [edge for edge in all_edges if edge != nearest]


def _solve_edge_alignment_with_fallback(
    table_aabb_xy,
    pack_objects_world,
    half_extent_xy,
    mount_gap_m,
    collision_checker,
):
    best = None
    for edge in _edge_try_order(table_aabb_xy, pack_objects_world):
        result = place_franka_edge_aligned(
            EdgeAlignRequest(
                table_aabb_xy=table_aabb_xy,
                pack_objects_world=tuple(pack_objects_world),
                role_weights=DEFAULT_ROLE_WEIGHTS,
                robot_half_extent_xy=(half_extent_xy[0], half_extent_xy[1]),
                edge_gap_m=mount_gap_m,
                edge_margin_m=0.05,
                scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.22, -0.22, 0.30, -0.30),
                collision_checker=collision_checker,
                preferred_edge=edge,
            )
        )
        if best is None or len(result.collision_hits) < len(best.collision_hits):
            best = result
        if len(result.collision_hits) == 0:
            break
    return best


def _evaluate_gate(
    robot_pos,
    target_pos,
    floor_z,
    penetration_hits,
    edge_align_result: EdgeAlignResult,
    target_gap_m: float,
    integrity_report: PackIntegrityReport,
):
    scene_sane = _is_finite_pose(robot_pos) and _is_finite_pose(target_pos) and max(abs(target_pos[0]), abs(target_pos[1])) < 100.0
    base_on_ground = abs(robot_pos[2] - floor_z) <= 0.03
    base_collision_free = len(penetration_hits) == 0
    gap_ok = edge_align_result.gap_actual >= 0.0 and abs(edge_align_result.gap_actual - target_gap_m) <= 0.04
    target_dist = _distance_xy(robot_pos[:2], target_pos[:2])
    target_in_reach_band = 0.25 <= target_dist <= 1.05
    pack_integrity_ok = integrity_report.ok

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

    return MVPGateReport(
        scene_sane=scene_sane,
        base_on_ground=base_on_ground,
        base_collision_free=base_collision_free,
        gap_ok=gap_ok,
        target_in_reach_band=target_in_reach_band,
        pack_integrity_ok=pack_integrity_ok,
        pass_gate=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def main():
    args = parse_args()
    cfg = _load_config(args)
    _configure_dynamics(cfg)

    task_name = cfg["task"]["activity_name"]
    task_spec = build_manipulation_task_spec(task_name)

    print(
        f"[MVP] Configured: task={task_name}, scene={cfg['scene']['scene_model']}, "
        f"strict_gate={args.strict_gate}, steps={args.steps}"
    )
    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"[MVP] Episode {ep + 1}/{args.episodes} reset")
            env.reset()
            og.sim.step()
            _print_table_candidates(env)

            obj_sets = _build_task_object_sets(env, task_spec)
            if len(obj_sets["target_ids"]) == 0:
                raise RuntimeError("No target objects found in object_scope.")
            target_inst = obj_sets["target_ids"][0]
            target_obj = _get_scope_obj(env, target_inst)
            support_inst, support_obj = _resolve_hardcoded_table(env, args.placement_table_name)
            support_aabb_min, support_aabb_max = support_obj.aabb
            support_aabb_min = _to_float3(support_aabb_min)
            support_aabb_max = _to_float3(support_aabb_max)
            table_aabb_xy = (
                (support_aabb_min[0], support_aabb_min[1]),
                (support_aabb_max[0], support_aabb_max[1]),
            )
            table_top_z = support_aabb_max[2]
            table_center_xy = (
                0.5 * (support_aabb_min[0] + support_aabb_max[0]),
                0.5 * (support_aabb_min[1] + support_aabb_max[1]),
            )

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
            pack_spec = build_clutter_pack(
                table_obj_name=getattr(support_obj, "name", support_inst),
                descriptors=descriptors,
                seed=args.seed + ep,
            )
            pack_radius = _compute_pack_radius(pack_spec, descriptor_by_inst)
            origin_xy = _clamp_pack_origin(table_center_xy, table_aabb_xy, pack_radius)
            pack_origin = (origin_xy[0], origin_xy[1], table_top_z)

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
                raise RuntimeError("No pack objects available for edge alignment.")

            def collision_checker(base_pose_xyyaw):
                x, y, yaw = base_pose_xyyaw
                robot.set_position_orientation(position=(x, y, floor_z), orientation=_quat_from_yaw(yaw))
                og.sim.step()
                return _collect_robot_penetration_hits(env)

            edge_result = _solve_edge_alignment_with_fallback(
                table_aabb_xy=table_aabb_xy,
                pack_objects_world=pack_objects_world,
                half_extent_xy=half_extent_xy,
                mount_gap_m=args.mount_gap_m,
                collision_checker=collision_checker,
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
            )

            print(
                f"[MVP] Mount: edge={edge_result.edge_label}, rank={edge_result.candidate_rank}, "
                f"gap={edge_result.gap_actual:.3f}, hits={len(edge_result.collision_hits)}"
            )
            if penetration_hits:
                print(f"[MVP] Penetration objects: {penetration_hits}")
            print(f"[MVP] Gate: pass={gate.pass_gate}, reasons={list(gate.failure_reasons)}")
            print(f"[MVP] Support table: {support_inst} ({getattr(support_obj, 'name', support_inst)})")
            _print_object_inventory(env, obj_sets)

            _append_jsonl(
                args.debug_jsonl,
                {
                    "episode": ep + 1,
                    "support_inst": support_inst,
                    "pack_origin_world": list(pack_origin),
                    "edge_result": asdict(edge_result),
                    "integrity": asdict(integrity),
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
            for step_idx in range(args.steps):
                print(f"=== [MVP] Step {step_idx + 1}/{args.steps} ===")
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
