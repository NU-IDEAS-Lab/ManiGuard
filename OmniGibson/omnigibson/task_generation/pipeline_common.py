"""Shared infrastructure for task generation pipelines.

Contains helpers for BDDL management, sim interaction, video recording,
pack callbacks, and other utilities reused across different pipeline types
(e.g., table clutter, cabinet clutter).
"""

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
import torch as th

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")


# Pool constants and activity generators live in omnigibson.utils.bddl_generator.
# Re-exported here for backward compatibility with pinch_point / cabinet pipelines.
from omnigibson.utils.bddl_generator import DENSITY_PRESETS  # noqa: F401, E402
from omnigibson.utils.bddl_generator import generate_clutter_activity as generate_activity  # noqa: F401, E402

STRUCTURAL_CATEGORY_KEYWORDS = (
    "wall", "walls", "floor", "ceiling", "roof", "window", "door",
    "stairs", "stair", "railing", "beam", "column", "pillar",
)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def make_base_arg_parser(description="Task generation pipeline"):
    """Create an argument parser with args common to all pipelines."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--scene-model", default=None,
                   help="Scene to use. If omitted, auto-selects based on object footprint.")
    p.add_argument("--activity-name", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mount-gap-m", type=float, default=0.10)
    p.add_argument("--jitter-scale", type=float, default=0.01)
    p.add_argument("--showcase-gui", action="store_true")
    p.add_argument("--strict-gate", dest="strict_gate", action="store_true")
    p.add_argument("--no-strict-gate", dest="strict_gate", action="store_false")
    p.set_defaults(strict_gate=True)
    p.add_argument("--debug-jsonl", default=None)
    p.add_argument("--clutter-density", default="medium", choices=list(DENSITY_PRESETS))
    p.add_argument("--pack-jitter-xy", type=float, default=None)
    p.add_argument("--pack-min-clearance", type=float, default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--curation-manifest", default=None)
    p.add_argument("--allow-deferred", action="store_true")
    p.add_argument("--zone-edge-margin-m", type=float, default=None)
    p.add_argument("--obstacle-keepout-margin-m", type=float, default=None)
    p.add_argument("--obstacle-side-clearance-m", type=float, default=None)
    p.add_argument("--perimeter-clear-margin-m", type=float, default=None)
    return p


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def append_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=_json_default) + "\n")


def _json_default(obj):
    """Fallback serializer for Tensor / ndarray values in diagnostics."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def strip_room_suffix(room: str) -> str:
    if room and room[-1].isdigit() and "_" in room:
        room = "_".join(room.rsplit("_", 1)[:-1])
    return room


def get_taxonomy():
    from bddl.object_taxonomy import ObjectTaxonomy
    if not hasattr(get_taxonomy, "_cache"):
        get_taxonomy._cache = ObjectTaxonomy()
    return get_taxonomy._cache


def resolve_synset(category):
    try:
        return get_taxonomy().get_synset_from_category(category)
    except Exception:
        return f"{category}.n.01"


def refresh_activity_cache():
    import bddl
    from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES
    refreshed = sorted(os.listdir(
        os.path.join(os.path.dirname(bddl.__file__), "activity_definitions")
    ))
    BEHAVIOR_ACTIVITIES.clear()
    BEHAVIOR_ACTIVITIES.extend(refreshed)


def needs_gpu_dynamics(activity_name):
    try:
        from bddl.activity import Conditions
        taxonomy = get_taxonomy()
        cond = Conditions(behavior_activity=activity_name, activity_definition=0,
                          simulator_name="omnigibson", predefined_problem=None)
        for synset in cond.parsed_objects:
            try:
                if "substance" in taxonomy.get_abilities(synset):
                    print(f"[Pipeline] GPU dynamics enabled (substance: {synset})")
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def get_scene_json_path(scene_model):
    from omnigibson.utils.asset_utils import get_scene_path
    return os.path.join(
        get_scene_path(scene_model), "json", f"{scene_model}_best.json",
    )


def discover_from_scene_json(scene_json_path, category_filter_fn, priority_map=None):
    """Find (category, room_type, room_instance) of best matching object from scene JSON.

    Returns a 3-tuple: (category, room_type, room_instance) where room_type
    is the stripped semantic name (for BDDL ``inroom``) and room_instance is
    the full instance name (for ``load_room_instances``).  Returns None if no
    match is found.
    """
    with open(scene_json_path, "r", encoding="utf-8") as f:
        init_infos = json.load(f).get("objects_info", {}).get("init_info", {})

    candidates = []
    for info in init_infos.values():
        args = info.get("args", {})
        cat = args.get("category", "")
        if not category_filter_fn(cat):
            continue
        rooms = args.get("in_rooms", [])
        room_instance = rooms[0] if rooms else "living_room_0"
        room_type = strip_room_suffix(room_instance)
        candidates.append((cat, room_type, room_instance))

    if not candidates:
        return None
    if priority_map:
        candidates.sort(key=lambda c: priority_map.get(c[0], 0), reverse=True)
    return candidates[0]


# Placeable-driven scene selection helpers live in utils/placeable.py.
from omnigibson.task_generation.utils.placeable import pick_scene_from_placeable


# ---------------------------------------------------------------------------
# Sim-dependent helpers
# ---------------------------------------------------------------------------

def _robot_config():
    return {
        "type": "FrankaMounted", "obs_modalities": ["rgb"],
        "action_type": "continuous", "action_normalize": True,
        "controller_config": {
            "arm_0": {"name": "OperationalSpaceController"},
            "gripper_0": {"name": "MultiFingerGripperController"},
        },
    }


def build_task_config(scene_model, activity_name):
    return {
        "scene": {"type": "InteractiveTraversableScene", "scene_model": scene_model},
        "task": {
            "type": "BehaviorTask", "activity_name": activity_name,
            "activity_definition_id": 0, "activity_conditions_met": False,
            "online_object_sampling": True,
        },
        "robots": [_robot_config()],
    }


def iter_scope_objects(env):
    for inst, ent in (getattr(env.task, "object_scope", {}) or {}).items():
        if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
            continue
        obj = getattr(ent, "wrapped_obj", None)
        if obj is not None:
            yield inst, obj


def get_scope_obj(env, inst):
    ent = (getattr(env.task, "object_scope", {}) or {}).get(inst)
    if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
        return None
    return getattr(ent, "wrapped_obj", None)


def is_structural_object(obj):
    name = str(getattr(obj, "name", "") or "").lower()
    cat = str(getattr(obj, "category", "") or "").lower()
    if getattr(obj, "is_system", False):
        return True
    return any(token in name or token in cat for token in STRUCTURAL_CATEGORY_KEYWORDS)


def _object_bounds_xy(obj):
    obj_min, obj_max = obj.aabb
    return (
        (float(obj_min[0]), float(obj_min[1])),
        (float(obj_max[0]), float(obj_max[1])),
    )


def _bounds_overlap_xy(bounds_a, bounds_b):
    (ax0, ay0), (ax1, ay1) = bounds_a
    (bx0, by0), (bx1, by1) = bounds_b
    return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)


def _expanded_bounds(bounds_xy, margin_m):
    (x0, y0), (x1, y1) = bounds_xy
    return ((x0 - margin_m, y0 - margin_m), (x1 + margin_m, y1 + margin_m))


def _yaw_from_quat(quat):
    x, y, z, w = (float(v) for v in quat)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def region_bounds_to_world_xy(region, support_obj):
    """Axis-aligned world xy bounds of a placeable region, given the live support pose.

    The region dict carries object-local xy_min/xy_max (scale-normalized).
    We apply support_obj's world scale + yaw + position to the four corners
    and return the axis-aligned xy bounds that enclose them. Pitch/roll are
    ignored; tables are assumed upright.
    """
    pos, quat = support_obj.get_position_orientation()
    pos = [float(v) for v in pos]
    quat = [float(v) for v in quat]
    scale_vec = support_obj.scale
    sx, sy = float(scale_vec[0]), float(scale_vec[1])
    yaw = _yaw_from_quat(quat)
    c, s = math.cos(yaw), math.sin(yaw)

    xy_min = region["xy_min"]
    xy_max = region["xy_max"]
    corners = []
    for (lx, ly) in ((xy_min[0], xy_min[1]), (xy_min[0], xy_max[1]),
                     (xy_max[0], xy_min[1]), (xy_max[0], xy_max[1])):
        ex, ey = lx * sx, ly * sy
        wx = pos[0] + c * ex - s * ey
        wy = pos[1] + s * ex + c * ey
        corners.append((wx, wy))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def _oriented_keepout_bounds_xy(base_xy, yaw, x_min, x_max, y_min, y_max):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for lx, ly in ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)):
        wx = float(base_xy[0]) + c * lx - s * ly
        wy = float(base_xy[1]) + s * lx + c * ly
        corners.append((wx, wy))
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def clear_support_area(env, support_obj, surface_bounds_xy, margin_m=0.60):
    """Remove all non-structural preset objects on and around the support surface.

    Removes every object whose xy bounding box overlaps the support surface
    bounds expanded by ``margin_m``, except the support itself, BDDL scope
    objects, and structural objects (walls, floors, etc.).  No z-filtering
    or category filtering — everything in the area goes.
    """
    import omnigibson as og

    expanded_bounds = _expanded_bounds(surface_bounds_xy, margin_m)
    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for _, obj in iter_scope_objects(env)}

    to_remove = []
    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        if not name or name == support_name or name in scope_names:
            continue
        if is_structural_object(obj):
            continue
        try:
            obj_bounds = _object_bounds_xy(obj)
        except Exception:
            continue
        if not _bounds_overlap_xy(obj_bounds, expanded_bounds):
            continue
        to_remove.append(obj)

    if to_remove:
        names = [getattr(o, "name", "?") for o in to_remove]
        og.sim.batch_remove_objects(to_remove)
        print(f"[Pipeline] Cleared {len(to_remove)} objects from support area: {names}")
    return [getattr(o, "name", "") for o in to_remove]


def clear_robot_base_region(env, support_obj, base_xy, robot_half_extent_xy,
                            margin_m=0.05, base_yaw=0.0,
                            workspace_front_m=0.0, workspace_side_m=0.0,
                            workspace_rear_m=0.0):
    """Remove preset objects overlapping the chosen Franka/base keepout region."""
    import omnigibson as og

    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for _, obj in iter_scope_objects(env)}
    hx = float(robot_half_extent_xy[0]) + margin_m
    hy = float(robot_half_extent_xy[1]) + margin_m
    keepout_bounds = _oriented_keepout_bounds_xy(
        base_xy=base_xy,
        yaw=float(base_yaw),
        x_min=-(hx + float(workspace_rear_m)),
        x_max=hx + float(workspace_front_m),
        y_min=-(hy + float(workspace_side_m)),
        y_max=hy + float(workspace_side_m),
    )

    to_remove = []
    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        if not name or name == support_name or name in scope_names:
            continue
        if is_structural_object(obj):
            continue
        try:
            obj_bounds = _object_bounds_xy(obj)
        except Exception:
            continue
        if not _bounds_overlap_xy(obj_bounds, keepout_bounds):
            continue
        to_remove.append(obj)

    if to_remove:
        names = [getattr(o, "name", "?") for o in to_remove]
        og.sim.batch_remove_objects(to_remove)
        print(
            f"[Pipeline] Removed {len(to_remove)} robot-mount keepout objects: {names} "
            f"(front={workspace_front_m:.2f}, side={workspace_side_m:.2f}, rear={workspace_rear_m:.2f})"
        )
    return [getattr(o, "name", "") for o in to_remove]


def build_task_object_sets(env, task_spec):
    available = {inst for inst, _ in iter_scope_objects(env)}
    target_ids = [i for i in task_spec.target_ids if i in available]
    support_ids = [i for i in task_spec.support_ids if i in available]
    fragile_ids = [
        i for i in task_spec.fragile_ids
        if i in available and i not in target_ids and i not in support_ids
    ]
    assigned = set(target_ids) | set(fragile_ids) | set(support_ids)
    clutter_ids = sorted(
        i for i in available
        if i not in assigned and not i.startswith(("agent.", "floor."))
    )
    if not target_ids:
        for inst, _ in iter_scope_objects(env):
            if inst.startswith(("coffee_cup.", "cup.", "mug.")):
                target_ids = [inst]
                break
    return {
        "target_ids": tuple(target_ids),
        "fragile_ids": tuple(sorted(fragile_ids)),
        "support_ids": tuple(sorted(support_ids)),
        "clutter_ids": tuple(clutter_ids),
    }


def object_aabb_dims(obj):
    """Return (dx, dy, dz) AABB dimensions in meters, each floored at 0.01.

    Returns None if obj.aabb raises (e.g., object not yet loaded). Callers
    building clutter / stack descriptors share this so the dimensions are
    computed identically across pipelines.
    """
    try:
        a_min, a_max = obj.aabb
    except Exception:
        return None
    return (
        max(0.01, float(a_max[0] - a_min[0])),
        max(0.01, float(a_max[1] - a_min[1])),
        max(0.01, float(a_max[2] - a_min[2])),
    )


def build_descriptors(env, obj_sets):
    from omnigibson.utils.clutter_pack_layout import ClutterObjectDescriptor

    descriptors, objects_by_inst = [], {}
    for role, id_key in [("target", "target_ids"), ("fragile", "fragile_ids"), ("clutter", "clutter_ids")]:
        for inst in obj_sets[id_key]:
            obj = get_scope_obj(env, inst)
            if obj is None:
                continue
            dims = object_aabb_dims(obj)
            if dims is None:
                continue
            dx, dy, dz = dims
            try:
                obj_pos = obj.get_position_orientation()[0]
                aabb_min = obj.aabb[0]
                root_to_bottom_z = max(0.0, float(obj_pos[2]) - float(aabb_min[2]))
            except Exception:
                continue
            descriptors.append(ClutterObjectDescriptor(
                instance_id=inst, role=role,
                half_extent_xy=(0.5 * dx, 0.5 * dy),
                height=dz,
                root_to_bottom_z=root_to_bottom_z,
            ))
            objects_by_inst[inst] = obj
    return descriptors, objects_by_inst


def robot_half_extent_xy(robot):
    for key in ("base_link", "base", "base_footprint", "chassis"):
        link = (getattr(robot, "links", {}) or {}).get(key)
        if link is not None:
            try:
                mn, mx = link.aabb
                return ((float(mx[0] - mn[0])) * 0.5, (float(mx[1] - mn[1])) * 0.5)
            except Exception:
                pass
    return (0.15, 0.15)


# Categories whose AABB physically blocks the robot base at floor level.
# Probed on Rs_int: walls (7), doors (2) — all axis-aligned boxes with
# valid .aabb on DatasetObject. Kept narrow so is_structural_object()'s
# broader keyword filter (which catches floors, ceilings, floor_lamp) is
# not reused here.
BASE_BLOCKING_CATEGORIES = frozenset({"walls", "wall", "door"})


def make_base_collision_checker(env, robot_half_extent_xy, min_clearance_m=0.05):
    """Return a callable (x, y, yaw) -> list of colliding blocker names.

    Snapshots walls / doors from the loaded scene at call time. Each frame:
    axis-aligned expansion of the robot footprint by robot half-extent +
    min_clearance_m, then rectangle-rectangle overlap against each blocker's
    precomputed XY AABB. Pose yaw is ignored (conservative; safe upper bound
    on footprint). No try/except: if .aabb fails on a blocker, the build
    crashes so the caller sees it immediately.
    """
    blockers = []
    for obj in env.scene.objects:
        cat = str(getattr(obj, "category", "") or "").lower()
        if cat not in BASE_BLOCKING_CATEGORIES:
            continue
        mn, mx = obj.aabb
        blockers.append((
            str(getattr(obj, "name", "?")),
            (float(mn[0]), float(mn[1])),
            (float(mx[0]), float(mx[1])),
        ))

    hx, hy = float(robot_half_extent_xy[0]), float(robot_half_extent_xy[1])
    pad = float(min_clearance_m)

    def check(pose_xyyaw):
        x, y, _yaw = pose_xyyaw
        lo_x, hi_x = x - hx - pad, x + hx + pad
        lo_y, hi_y = y - hy - pad, y + hy + pad
        hits = []
        for name, (bx0, by0), (bx1, by1) in blockers:
            if lo_x <= bx1 and hi_x >= bx0 and lo_y <= by1 and hi_y >= by0:
                hits.append(name)
        return hits

    check.blocker_count = len(blockers)
    return check


# ---------------------------------------------------------------------------
# Pack callback factories
# ---------------------------------------------------------------------------

def make_settle_fn(og_mod, th_mod):
    def settle(objs):
        for _ in range(3):
            og_mod.sim.step()
        for _ in range(7):
            og_mod.sim.step()
            for obj in objs.values():
                try:
                    vel = obj.get_linear_velocity()
                    vz = float(vel[2]) if hasattr(vel, '__getitem__') else 0.0
                    obj.set_linear_velocity(th_mod.tensor([0.0, 0.0, min(0.0, vz)]))
                    obj.set_angular_velocity(th_mod.zeros(3))
                except Exception:
                    pass
        for obj in objs.values():
            try:
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            except Exception:
                pass
        og_mod.sim.step()
    return settle


def stabilize_active_objects(og_mod, objs, steps, support_obj=None):
    if not objs or steps <= 0:
        return
    for _ in range(int(steps)):
        if support_obj is not None:
            try:
                if hasattr(support_obj, "set_linear_velocity"):
                    support_obj.set_linear_velocity(th.zeros(3))
                if hasattr(support_obj, "set_angular_velocity"):
                    support_obj.set_angular_velocity(th.zeros(3))
                if hasattr(support_obj, "keep_still"):
                    support_obj.keep_still()
            except Exception:
                pass
        for obj in objs.values():
            try:
                if hasattr(obj, "set_linear_velocity"):
                    obj.set_linear_velocity(th.zeros(3))
                if hasattr(obj, "set_angular_velocity"):
                    obj.set_angular_velocity(th.zeros(3))
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            except Exception:
                pass
        og_mod.sim.step()



def pin_support_object_to_world(support_obj):
    if support_obj is None:
        return False
    if bool(getattr(support_obj, "fixed_base", False)):
        return False
    try:
        joint_path = f"{support_obj.prim_path}/curationFixedJoint"
        root_link_path = getattr(getattr(support_obj, "root_link", None), "prim_path", None)
        if not root_link_path:
            root_name = getattr(support_obj, "_root_link_name", None)
            if root_name:
                root_link_path = f"{support_obj.prim_path}/{root_name}"
        if not root_link_path:
            return False
        from omnigibson.utils.usd_utils import create_joint

        create_joint(
            prim_path=joint_path,
            joint_type="FixedJoint",
            body1=root_link_path,
        )
        support_obj.fixed_base = True
        return True
    except Exception:
        return False


def make_park_fn(og_mod, zone_surface_bounds, floor_z):
    """Return a callback that parks passive objects off to the side.

    Used inside the pack retry loop where objects may be parked/un-parked
    across retry iterations.  For final cleanup after the loop, use
    ``remove_objects`` instead.
    """
    def park(passive_objs):
        if not passive_objs:
            return
        (_, by0), (bx1, _) = zone_surface_bounds
        base_x, base_y = bx1 + 1.5, by0 - 1.2
        for idx, inst in enumerate(sorted(passive_objs)):
            x = base_x + 0.18 * (idx % 8)
            y = base_y - 0.18 * (idx // 8)
            try:
                passive_objs[inst].set_position_orientation(
                    position=(x, y, floor_z + 0.06), orientation=(0, 0, 0, 1),
                )
                if hasattr(passive_objs[inst], "keep_still"):
                    passive_objs[inst].keep_still()
            except Exception:
                pass
        og_mod.sim.step()
    return park


def remove_objects(og_mod, objs_by_inst):
    """Remove objects from the scene permanently (post-pack cleanup)."""
    if not objs_by_inst:
        return
    og_mod.sim.batch_remove_objects(list(objs_by_inst.values()))
    print(f"[Pipeline] Removed {len(objs_by_inst)} objects: {sorted(objs_by_inst.keys())}")


def validate_poses(objs):
    invalid = []
    for inst, obj in objs.items():
        try:
            pos = obj.get_position_orientation()[0]
            if not all(math.isfinite(float(pos[i])) for i in range(3)):
                invalid.append(inst)
        except Exception:
            invalid.append(inst)
    return invalid


def check_interpenetration(objs, tol):
    inst_ids = sorted(objs.keys())
    hits = []
    for i, a in enumerate(inst_ids):
        try:
            aabb_a = objs[a].aabb
        except Exception:
            continue
        for b in inst_ids[i + 1:]:
            try:
                aabb_b = objs[b].aabb
                if all(
                    min(float(aabb_a[1][d]), float(aabb_b[1][d]))
                    - max(float(aabb_a[0][d]), float(aabb_b[0][d])) > tol
                    for d in range(3)
                ):
                    hits.append((a, b))
            except Exception:
                continue
    return hits


# Camera + video helpers live in utils/video.py; re-export the ones
# run_ltl_rollout and external callers import from pipeline_common.
from omnigibson.task_generation.utils.video import (  # noqa: F401
    build_video_view_specs,
    close_video_writer,
    expected_video_path,
    init_video_writer,
    setup_cameras,
)


# ---------------------------------------------------------------------------
# Pre-rollout stabilisation and LTL step-0 validation
# ---------------------------------------------------------------------------

def _try_upright_objects(og_mod, objects_by_inst):
    """Re-set any tipped objects to upright orientation, preserving position."""
    from omnigibson.object_states import Upright
    fixed = []
    for inst, obj in objects_by_inst.items():
        try:
            if Upright not in obj.states:
                continue
            if not obj.states[Upright].get_value():
                pos = obj.get_position_orientation()[0]
                obj.set_position_orientation(
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    orientation=(0, 0, 0, 1),
                )
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
                fixed.append(inst)
        except Exception:
            continue
    if fixed:
        og_mod.sim.step()
        print(f"[Pipeline] Re-uprighted {len(fixed)} objects: {fixed}")
    return fixed


def validate_ltl_step0(env, activity_name, scene_model, active_objects_by_inst):
    """Evaluate LTL propositions at step 0 and return (ok, label_dict).

    Creates a temporary LTL monitor, runs one evaluation, and checks
    whether the initial state would immediately violate any safety
    constraint.  Returns ``(True, labels)`` if clean.
    """
    from omnigibson.utils.safety_monitor import TaskLTLMonitor

    try:
        monitor = TaskLTLMonitor(
            env=env, activity_name=activity_name,
            scene_model=scene_model,
            active_objects_by_inst=active_objects_by_inst,
        )
        monitor.reset()
        info = monitor.step(0)
        labels = info.get("ap", {})
        doomed = bool(info.get("doomed", False))
        return not doomed, labels
    except Exception as exc:
        print(f"[Pipeline] WARNING: LTL step-0 validation failed: {exc}")
        return True, {}


def stabilize_and_validate(
    env, og_mod, activity_name, scene_model,
    active_objects_by_inst, max_attempts=3,
):
    """Stabilise objects and validate LTL step 0.

    Runs up to *max_attempts* rounds of: re-upright tipped objects →
    settle physics → evaluate LTL step 0.  Returns ``(ok, labels)``
    where *ok* is True if a clean initial state was achieved.
    """
    ok = False
    labels = {}
    for attempt in range(max_attempts):
        # Fix tipped objects.
        _try_upright_objects(og_mod, active_objects_by_inst)

        # Physics settle (reuse shared helper).
        stabilize_active_objects(og_mod, active_objects_by_inst, steps=3)

        # Evaluate LTL step 0.
        ok, labels = validate_ltl_step0(
            env, activity_name, scene_model, active_objects_by_inst,
        )
        if ok:
            if attempt > 0:
                print(f"[Pipeline] LTL step-0 clean after {attempt + 1} stabilisation rounds")
            return True, labels

        print(f"[Pipeline] LTL step-0 violation (attempt {attempt + 1}/{max_attempts}): "
              f"{labels}")

    return False, labels


# ---------------------------------------------------------------------------
# LTL rollout (shared by all pipelines)
# ---------------------------------------------------------------------------

def run_ltl_rollout(env, activity_name, scene_model, active_objects_by_inst,
                    robot, target_obj, args, episode, rng,
                    support_obj=None, camera_override=None):
    """Run jitter-action rollout with LTL monitoring and video recording.

    Returns the LTL summary dict.
    """
    import omnigibson as og
    from omnigibson.utils.safety_monitor import TaskLTLMonitor

    ltl_monitor = TaskLTLMonitor(
        env=env, activity_name=activity_name,
        scene_model=scene_model,
        active_objects_by_inst=active_objects_by_inst,
    )
    ltl_monitor.reset()
    ltl_monitor.step(0)

    video_writers = []
    if args.save_video:
        video_views = build_video_view_specs(
            args,
            robot,
            target_obj,
            support_obj=support_obj,
            active_objects_by_inst=active_objects_by_inst,
            camera_override=camera_override,
        )
        # Position external cameras and set viewer to opposite side
        video_views = setup_cameras(env, video_views)
        args._resolved_video_views = tuple(
            {
                "label": view["label"],
                "eye": view["position"],
                "lookat": [float(v) for v in view["lookat"]],
                "orientation": view["orientation"],
                "sensor_name": view["sensor_name"],
                "canonical": bool(view["canonical"]),
            }
            for view in video_views
        )
        for _ in range(3):
            og.sim.step()
        og.sim.render()
        og.sim.render()

        for view in video_views:
            stem = args.save_video[:-4] if args.save_video.endswith(".mp4") else args.save_video
            base_path = f"{stem}_{view['label']}.mp4"
            video_path = expected_video_path(base_path, episode)
            print(f"[Pipeline] Video: {video_path} (sensor={view['sensor_name']})")
            # Pin the MP4 stream size to the sensor's actual render size so
            # PyAV doesn't silently upscale. `sensor.image_height` reads the
            # Kit viewport's current texture resolution, not whatever we
            # asked for in sensor_kwargs.
            sensor = env.external_sensors.get(view["sensor_name"])
            frame_hw = (
                (int(sensor.image_height), int(sensor.image_width))
                if sensor is not None else None
            )
            writer = init_video_writer(
                base_path, episode, args.video_fps, robot=None, frame_hw=frame_hw,
            )
            if writer is None:
                raise RuntimeError(f"Failed to initialize video writer for {video_path}.")
            video_writers.append({"view": view, "writer": writer, "path": video_path, "cam": view["sensor_name"]})

    executed = 0
    for _ in range(args.steps):
        action = rng.normal(0.0, args.jitter_scale,
                            size=robot.action_space.shape).astype(np.float32)
        if hasattr(robot.action_space, "low"):
            action = np.clip(action, robot.action_space.low, robot.action_space.high)
        env._pre_step(action)
        og.sim.step()
        executed += 1

        # Record from all external cameras simultaneously (one render pass)
        if video_writers:
            og.sim.render()
            raw_obs, _ = env.get_obs()
            external = raw_obs.get("external", {})
            for writer_info in video_writers:
                cam_obs = external.get(writer_info["cam"], {})
                rgb = cam_obs.get("rgb")
                if rgb is not None:
                    frame = rgb[..., :3].cpu().numpy().astype(np.uint8)
                    try:
                        import av
                        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                        for packet in writer_info["writer"]["stream"].encode(video_frame):
                            writer_info["writer"]["container"].mux(packet)
                    except Exception:
                        pass

        ltl_monitor.step(executed)
        if executed % 50 == 0:
            print(f"[Pipeline] Step {executed}/{args.steps}")

    for writer_info in video_writers:
        close_video_writer(writer_info["writer"])

    summary = ltl_monitor.summary()
    print(f"[Pipeline] Episode done: steps={executed}, violated={summary['violated']}")
    return summary, executed


# ---------------------------------------------------------------------------
# Run directory setup
# ---------------------------------------------------------------------------

def init_run_dir(args, default_label):
    """Set up args.run_dir / debug_jsonl / save_video with a default dir name.

    default_label: the `{label}_{timestamp}` directory name to use when
    args.run_dir is None. Callers pass whatever identifier fits their
    pipeline (scene_model for scene-based; "empty_<surface>_<setup>" for
    empty-scene). Timestamp is appended here.
    """
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(_DEFAULT_RUNS_DIR, f"{default_label}_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.debug_jsonl is None:
        args.debug_jsonl = os.path.join(args.run_dir, "diagnostics.jsonl")
    if args.save_video is True:
        args.save_video = os.path.join(args.run_dir, "rollout.mp4")
    elif args.save_video is False:
        args.save_video = None
    print(f"[Pipeline] Run directory: {args.run_dir}")


def setup_run_dir(args):
    """BasePipeline entrypoint: run dir labeled by scene_model."""
    init_run_dir(args, args.scene_model or "auto")


def load_scene_curation(args):
    if not getattr(args, "curation_manifest", None):
        return None
    from omnigibson.task_generation.curation.curation_manifest import (
        apply_scene_entry_to_args,
        load_curation_manifest,
    )

    manifest = load_curation_manifest(args.curation_manifest)
    entry = manifest.get_scene_entry(args.scene_model)
    if entry.status == "defer" and not args.allow_deferred:
        reason = entry.defer_reason or "scene marked as deferred in curation manifest"
        raise RuntimeError(f"Scene '{args.scene_model}' is deferred: {reason}")
    apply_scene_entry_to_args(args, entry)
    args._scene_curation = entry
    args._scene_curation_manifest = manifest
    print(
        f"[Curation] scene={entry.scene_model}, status={entry.status}, "
        f"repair_mode={entry.repair_mode}, issue_tags={entry.issue_tags}"
    )
    if entry.surface_name or entry.support_category:
        print(
            f"[Curation] forced_surface name={entry.surface_name} "
            f"category={entry.support_category}"
        )
    if (
        entry.clutter_density is not None
        or entry.pack_jitter_xy is not None
        or entry.pack_min_clearance is not None
        or entry.perimeter_clear_margin_m is not None
        or entry.support_clear_mode is not None
        or entry.perimeter_clear_mode is not None
    ):
        print(
            "[Curation] overrides: "
            f"density={entry.clutter_density}, "
            f"pack_jitter_xy={entry.pack_jitter_xy}, "
            f"pack_min_clearance={entry.pack_min_clearance}, "
            f"perimeter_clear_margin_m={entry.perimeter_clear_margin_m}, "
            f"support_clear_mode={entry.support_clear_mode}, "
            f"perimeter_clear_mode={entry.perimeter_clear_mode}"
        )
    print(
        "[Curation] video: "
        f"eye={entry.video_camera_eye}, "
        f"lookat={entry.video_camera_lookat}"
    )
    return entry


def pipeline_exit(code=0):
    """Clean exit to avoid Isaac Sim shutdown segfault."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# ---------------------------------------------------------------------------
# BasePipeline — shared skeleton for table-based task generation
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class EpisodeContext:
    """Mutable bag of per-episode state shared between pipeline stages."""
    env: Any = None
    og: Any = None                     # omnigibson module
    args: Any = None
    rng: Any = None

    # Surface
    support_obj: Any = None
    surface_info: Any = None           # SurfaceAnalysis
    surface_name: str = ""
    surface_bounds_xy: Optional[Tuple] = None
    table_top_z: float = 0.0
    floor_z: float = 0.0
    removed_area_objects: list[str] = field(default_factory=list)
    removed_robot_base_objects: list[str] = field(default_factory=list)
    resolved_video_views: Tuple = field(default_factory=tuple)

    # Activity
    activity_name: str = ""
    selection: Dict = field(default_factory=dict)
    curation: Any = None

    # Objects (populated by identify_objects)
    target_obj: Any = None
    active_objects: Dict[str, Any] = field(default_factory=dict)

    # Robot
    robot: Any = None
    edge_result: Any = None

    # Gate
    gate_pass: bool = False

    # Episode index
    episode: int = 0


class BasePipeline(ABC):
    """Base class for table-based task generation pipelines.

    Subclasses implement the pipeline-specific hooks:
      - add_args()          — register CLI flags
      - activity_prefix()   — default activity name prefix
      - generate_activity() — produce BDDL + LTL + selection dict
      - configure_task()    — tweak the env config (e.g. sampling_whitelist)
      - identify_objects()  — partition scope objects into roles
      - place_objects()     — arrange objects on the table
      - make_edge_objects() — build EdgeAlignObject list for robot placement
      - extra_gate_checks() — additional gate conditions (default: True)
      - diagnostics_extra() — extra fields for the diagnostics JSONL
    """

    # -- Subclass hooks (override these) ------------------------------------

    @classmethod
    @abstractmethod
    def add_args(cls, parser):
        """Register pipeline-specific CLI arguments on *parser*."""

    @abstractmethod
    def activity_prefix(self):
        """Return default activity name prefix, e.g. 'auto_stack_on'."""

    @abstractmethod
    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        """Generate BDDL + LTL files.

        Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
        """

    def configure_task(self, cfg, selection):
        """Optional hook to modify the env config before loading.

        For example, inject sampling_whitelist.  Default: no-op.
        """

    @abstractmethod
    def identify_objects(self, ctx):
        """Identify and group task objects from the BDDL scope.

        Must populate ``ctx.target_obj`` and ``ctx.active_objects``.
        """

    @abstractmethod
    def place_objects(self, ctx):
        """Arrange objects on the support surface.

        Called after identify_objects().  May use ctx.surface_bounds_xy,
        ctx.table_top_z, ctx.support_obj, etc.
        """

    @abstractmethod
    def make_edge_objects(self, ctx):
        """Return a tuple of EdgeAlignObject for robot placement."""

    @abstractmethod
    def select_objects(self, args, rng):
        """Pre-select objects and estimate required surface area.

        Must return a dict with at minimum ``"required_area_m2"`` plus
        pipeline-specific object selections.
        """

    def extra_gate_checks(self, ctx):
        """Additional gate conditions beyond the shared ones.  Default: True."""
        return True

    def diagnostics_extra(self, ctx):
        """Return a dict of extra fields for the diagnostics JSONL."""
        return {}

    # -- Shared machinery (not intended for override) -----------------------

    @classmethod
    def make_parser(cls, description="Task generation pipeline"):
        parser = make_base_arg_parser(description=description)
        cls.add_args(parser)
        return parser

    def run(self):
        parser = self.make_parser()
        args = parser.parse_args()
        setup_run_dir(args)
        load_scene_curation(args)
        if args.dry_run:
            self._run_dry_run(args)
        else:
            self._run_sim(args)

    def _run_dry_run(self, args):
        # Object-first scene selection for dry-run too.
        rng_pre = np.random.default_rng(args.seed)
        pre_selection = self.select_objects(args, rng_pre)
        args._pre_selection = pre_selection

        required = pre_selection["required_area_m2"]
        pick = pick_scene_from_placeable(rng_pre, required, scene_model=args.scene_model)
        args.scene_model = pick["scene_model"]
        args._picked_surface = pick
        print(f"[Pipeline] Scene={pick['scene_model']}, "
              f"surface={pick['category']}/{pick['model']} in {pick['room_instance']}, "
              f"area={pick['area_m2']:.3f} m² (required={required:.3f} m²)")

        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"
        curation = getattr(args, "_scene_curation", None)

        support_synset = resolve_synset(pick["category"])
        room_instance = pick["room_instance"]
        support_room = strip_room_suffix(room_instance)
        if curation and curation.support_category:
            support_synset = resolve_synset(curation.support_category)
        if curation and curation.support_room:
            support_room = curation.support_room

        rng = np.random.default_rng(args.seed)
        bddl_text, ltl_safety, bddl_path, json_path, selection = \
            self.generate_activity(activity_name, support_synset, support_room, args, rng)

        print(f"[Pipeline] Dry-run complete:")
        print(f"  BDDL:       {bddl_path}")
        print(f"  ltl_safety: {json_path}")
        print(f"  activity:   {activity_name}")
        print(f"\nGenerated BDDL:\n{bddl_text}")
        print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

        append_jsonl(args.debug_jsonl, {
            "event": "dry_run", "activity_name": activity_name,
            "scene_model": scene_label,
            "selection": selection,
            **self.diagnostics_extra(EpisodeContext(selection=selection, args=args)),
        })

    def _run_sim(self, args):
        import omnigibson as og
        from omnigibson.macros import gm

        gm.ENABLE_OBJECT_STATES = True

        # -- Object-first scene selection -----------------------------------
        rng_pre = np.random.default_rng(args.seed)
        pre_selection = self.select_objects(args, rng_pre)
        args._pre_selection = pre_selection

        required = pre_selection["required_area_m2"]
        pick = pick_scene_from_placeable(rng_pre, required, scene_model=args.scene_model)
        args.scene_model = pick["scene_model"]
        args._picked_surface = pick
        print(f"[Pipeline] Scene={pick['scene_model']}, "
              f"surface={pick['category']}/{pick['model']} in {pick['room_instance']}, "
              f"area={pick['area_m2']:.3f} m² (required={required:.3f} m²)")

        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"
        curation = getattr(args, "_scene_curation", None)

        # -- Resolve support surface ----------------------------------------
        scene_json = get_scene_json_path(args.scene_model)
        if not os.path.isfile(scene_json):
            raise RuntimeError(f"Scene JSON not found: {scene_json}")
        surface_category = pick["category"]
        room_instance = pick["room_instance"]
        support_room = strip_room_suffix(room_instance)
        support_synset = resolve_synset(surface_category)
        if curation and curation.support_category:
            surface_category = curation.support_category
            support_synset = resolve_synset(surface_category)
        if curation and curation.support_room:
            support_room = curation.support_room
        print(f"[Pipeline] Discovered: category={surface_category} "
              f"synset={support_synset} room={support_room}")

        # Store surface info in pre_selection for _run_episode.
        if pre_selection is None:
            pre_selection = {}
            args._pre_selection = pre_selection
        pre_selection.setdefault("_surface_category", surface_category)
        pre_selection.setdefault("_room_type", support_room)
        pre_selection.setdefault("_room_instance", room_instance)

        # -- Generate BDDL --------------------------------------------------
        rng = np.random.default_rng(args.seed)
        _, _, bddl_path, _, selection = self.generate_activity(
            activity_name, support_synset, support_room, args, rng,
        )
        print(f"[Pipeline] Generated BDDL: {bddl_path}")
        refresh_activity_cache()

        # -- GPU dynamics ----------------------------------------------------
        gpu = needs_gpu_dynamics(activity_name)
        gm.USE_GPU_DYNAMICS = gpu
        gm.ENABLE_FLATCACHE = not gpu

        # -- Load environment ------------------------------------------------
        cfg = build_task_config(args.scene_model, activity_name)
        cfg["scene"]["scene_file"] = scene_json
        cfg["scene"]["scene_instance"] = None
        if room_instance:
            cfg["scene"]["load_room_instances"] = [room_instance]
            print(f"[Pipeline] Partial load: room={room_instance}")
        cfg["task"]["online_object_sampling"] = True
        cfg["task"]["use_presampled_robot_pose"] = False
        if curation and getattr(curation, "surface_name", None):
            cfg["task"]["inroom_object_name_whitelist"] = {
                f"{support_synset}_1": [curation.surface_name],
            }

        self.configure_task(cfg, selection)

        # 3 external cameras (canonical names + resolution shared across
        # task-generation, teleop, training, eval).
        from omnigibson.utils.camera_setup import build_external_camera_configs
        cfg.setdefault("env", {})["external_sensors"] = build_external_camera_configs()

        print(f"[Pipeline] scene={scene_label}, activity={activity_name}, "
              f"strict_gate={args.strict_gate}")
        env = og.Environment(configs=cfg)
        exit_code = 0

        try:
            for ep in range(args.episodes):
                ctx = EpisodeContext(
                    env=env, og=og, args=args, rng=rng,
                    activity_name=activity_name,
                    selection=selection, episode=ep, curation=curation,
                )
                print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
                self._run_episode(ctx)

                payload = {
                    "episode": ep + 1,
                    "scene_model": scene_label,
                    "activity_name": activity_name,
                    "surface": ctx.surface_name,
                    "gate_pass": ctx.gate_pass,
                    "ltl_violated": ctx.ltl_summary.get("violated") if hasattr(ctx, "ltl_summary") else None,
                    "steps_executed": ctx.steps_executed if hasattr(ctx, "steps_executed") else 0,
                    "selection": selection,
                    "cameras": list(getattr(args, "_resolved_video_views", ())),
                    **self.diagnostics_extra(ctx),
                }
                if curation:
                    payload.update({
                        "curation_status": curation.status,
                        "repair_mode": curation.repair_mode,
                        "issue_tags": list(curation.issue_tags),
                    })
                append_jsonl(args.debug_jsonl, payload)
        except Exception:
            exit_code = 1
            print("[Pipeline] ERROR: pipeline execution failed.")
            traceback.print_exc()
        finally:
            print("[Pipeline] Shutdown simulator.")
            pipeline_exit(exit_code)

    def _run_episode(self, ctx):
        env, og, args = ctx.env, ctx.og, ctx.args
        env.reset()
        og.sim.step()

        # -- Find support surface object ------------------------------------
        # Prefer exact (category, model) match from placeable pick for
        # instance-level precision; fall back to curation surface_name/
        # category overrides.
        picked = getattr(args, "_picked_surface", None)
        forced_name = getattr(ctx.curation, "surface_name", None) if ctx.curation else None
        target_category = (
            getattr(ctx.curation, "support_category", None) if ctx.curation else None
        ) or (picked["category"] if picked else args._pre_selection["_surface_category"])
        target_model = picked["model"] if picked and not forced_name else None

        support_obj = None
        for obj in env.scene.objects:
            name = getattr(obj, "name", "")
            if forced_name:
                if name == forced_name:
                    support_obj = obj
                    break
                continue
            if getattr(obj, "category", "") != target_category:
                continue
            if target_model and getattr(obj, "model", "") != target_model:
                continue
            support_obj = obj
            break

        if support_obj is None:
            detail = forced_name or (f"{target_category}/{target_model}" if target_model else target_category)
            raise RuntimeError(f"Support surface '{detail}' not found in scene objects.")

        ctx.support_obj = support_obj
        ctx.surface_name = getattr(support_obj, "name", "")

        # Analyze the surface for obstacle/approach info.
        from omnigibson.utils.surface_discovery import analyze_surface
        try:
            aabb_min, aabb_max = support_obj.aabb
            surface_aabb_xy = (
                (float(aabb_min[0]), float(aabb_min[1])),
                (float(aabb_max[0]), float(aabb_max[1])),
            )
            top_z = float(aabb_max[2])
            scene_data = []
            for obj in env.scene.objects:
                try:
                    o_min, o_max = obj.aabb
                    scene_data.append({
                        "name": getattr(obj, "name", ""),
                        "category": str(getattr(obj, "category", "")),
                        "aabb_xy": ((float(o_min[0]), float(o_min[1])),
                                    (float(o_max[0]), float(o_max[1]))),
                        "top_z": float(o_max[2]),
                        "bottom_z": float(o_min[2]),
                    })
                except Exception:
                    continue
            other_aabbs = [
                d["aabb_xy"] for d in scene_data
                if d["name"] != ctx.surface_name
                and d["top_z"] >= 0.15
                and d.get("bottom_z", 0) <= top_z + 0.3
            ]
            ctx.surface_info = analyze_surface(
                ctx.surface_name, target_category, surface_aabb_xy, top_z,
                scene_data, scene_object_aabbs=other_aabbs,
            )
        except Exception:
            ctx.surface_info = None

        print(f"[Pipeline] Support surface: {ctx.surface_name} ({target_category})")
        # Pin support first so it cannot move, then clear and compute geometry once.
        if pin_support_object_to_world(support_obj):
            print(f"[Pipeline] Pinned support to world: {support_obj.name}")
        og.sim.step()

        aabb_min, aabb_max = support_obj.aabb
        # surface_bounds_xy comes from the raycast-validated placeable region
        # (object-local, scale-invariant) carried in picked, transformed into
        # world via support_obj's live pose. NOT the full object world AABB:
        # this respects L/U shaped tables where the AABB includes empty
        # corners, and picks the specific region (region_00 or region_01) the
        # picker selected -- so 2-region models can expose their smaller
        # region on its own merit.
        ctx.surface_bounds_xy = region_bounds_to_world_xy(picked, support_obj)
        ctx.picked_region = picked
        if ctx.curation and getattr(ctx.curation, "surface_bounds_override_xy", None):
            ctx.surface_bounds_xy = ctx.curation.surface_bounds_override_xy
            print(f"[Pipeline] Surface bounds override: {ctx.surface_bounds_xy}")
        ctx.table_top_z = float(aabb_max[2])
        ctx.floor_z = float(aabb_min[2])

        clear_margin = args.perimeter_clear_margin_m if args.perimeter_clear_margin_m is not None else 0.60
        ctx.removed_area_objects = clear_support_area(
            env, support_obj, ctx.surface_bounds_xy, margin_m=clear_margin,
        )
        if ctx.removed_area_objects:
            og.sim.step()

        # -- Pipeline-specific: identify & place objects --------------------
        self.identify_objects(ctx)
        self.place_objects(ctx)

        # -- Robot placement ------------------------------------------------
        from omnigibson.utils.franka_edge_align import (
            DEFAULT_ROLE_WEIGHTS, EdgeAlignRequest, EdgeAlignResult, _quat_from_yaw, place_franka_edge_aligned,
        )
        from omnigibson.utils.tabletop_workspace import compute_tabletop_zone

        ctx.robot = env.robots[0]

        if hasattr(ctx, "_zone") and ctx._zone is not None:
            zone = ctx._zone
        else:
            obstacle_bounds_xy = None
            obstacle_bounds_seq = []
            if ctx.curation and getattr(ctx.curation, "obstacle_bounds_override_xy", None):
                obstacle_bounds_xy = ctx.curation.obstacle_bounds_override_xy
                print(f"[Pipeline] Obstacle bounds override: {obstacle_bounds_xy}")
                obstacle_bounds_seq.append(obstacle_bounds_xy)
            elif ctx.surface_info and ctx.surface_info.obstacles:
                obstacle_bounds_seq.extend(obstacle.aabb_xy for obstacle in ctx.surface_info.obstacles)

            zone = compute_tabletop_zone(
                surface_bounds_xy=ctx.surface_bounds_xy,
                obstacle_bounds_xy=obstacle_bounds_xy,
                obstacle_bounds_seq=tuple(obstacle_bounds_seq),
                edge_margin_m=args.zone_edge_margin_m or 0.04,
                obstacle_keepout_margin_m=args.obstacle_keepout_margin_m or 0.08,
                obstacle_side_clearance_m=args.obstacle_side_clearance_m or 0.015,
            )

        pack_objects_world = self.make_edge_objects(ctx)

        preferred_edge = None
        if ctx.curation and ctx.curation.preferred_edge:
            preferred_edge = ctx.curation.preferred_edge
        if ctx.surface_info and ctx.surface_info.approach_edges:
            preferred_edge = preferred_edge or ctx.surface_info.approach_edges[0]

        robot_half_xy = robot_half_extent_xy(ctx.robot)
        base_checker = make_base_collision_checker(env, robot_half_xy, min_clearance_m=0.05)
        print(f"[Pipeline] Base collision checker: tracking {base_checker.blocker_count} walls/doors")

        edge_request = EdgeAlignRequest(
            table_aabb_xy=zone.surface_bounds,
            pack_objects_world=tuple(pack_objects_world),
            role_weights=DEFAULT_ROLE_WEIGHTS,
            robot_half_extent_xy=robot_half_xy,
            edge_gap_m=args.mount_gap_m, edge_margin_m=0.05,
            scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10,
                            0.15, -0.15, 0.20, -0.20),
            preferred_edge=preferred_edge,
            anchor_offset_m=getattr(args, "mount_anchor_offset_m", 0.0) or 0.0,
            collision_checker=base_checker,
        )
        override_pose = getattr(args, "mount_base_pose_xyyaw", None)
        if override_pose is None:
            ctx.edge_result = place_franka_edge_aligned(edge_request)
        else:
            override_x, override_y, override_yaw = (float(v) for v in override_pose)
            ctx.edge_result = EdgeAlignResult(
                edge_label="override",
                base_pose={
                    "position": (override_x, override_y, 0.0),
                    "orientation": _quat_from_yaw(override_yaw),
                },
                anchor_s=0.0,
                candidate_rank=0,
                collision_hits=(),
                gap_actual=float("nan"),
                failure_reason=None,
            )
        base_yaw = _yaw_from_quat(ctx.edge_result.base_pose["orientation"])
        ctx.removed_robot_base_objects = clear_robot_base_region(
                env,
                support_obj,
                ctx.edge_result.base_pose["position"][:2],
                edge_request.robot_half_extent_xy,
                margin_m=0.05,
                base_yaw=base_yaw,
                workspace_front_m=getattr(args, "mount_workspace_front_m", 0.0) or 0.0,
                workspace_side_m=getattr(args, "mount_workspace_side_m", 0.0) or 0.0,
                workspace_rear_m=getattr(args, "mount_workspace_rear_m", 0.0) or 0.0,
            )
        if ctx.removed_robot_base_objects:
            og.sim.step()
            if override_pose is None:
                ctx.edge_result = place_franka_edge_aligned(edge_request)
        ctx.robot.set_position_orientation(
            position=(ctx.edge_result.base_pose["position"][0],
                      ctx.edge_result.base_pose["position"][1], ctx.floor_z),
            orientation=ctx.edge_result.base_pose["orientation"],
        )
        post_mount_settle_steps = int(getattr(args, "post_mount_settle_steps", 0) or 0)
        if post_mount_settle_steps > 0:
            stabilize_active_objects(
                og,
                ctx.active_objects,
                post_mount_settle_steps,
                support_obj=ctx.support_obj,
            )
        else:
            og.sim.step()
        print(f"[Pipeline] Robot: edge={ctx.edge_result.edge_label}, "
              f"gap={ctx.edge_result.gap_actual:.3f}, "
              f"collision_hits={list(ctx.edge_result.collision_hits)}, "
              f"failure_reason={ctx.edge_result.failure_reason}")

        # -- Gate -----------------------------------------------------------
        rp = [float(v) for v in ctx.robot.get_position_orientation()[0][:3]]
        tp = [float(v) for v in ctx.target_obj.get_position_orientation()[0][:3]]
        target_dist = math.hypot(rp[0] - tp[0], rp[1] - tp[1])
        ctx.gate_pass = (
            all(math.isfinite(v) for v in rp + tp)
            and abs(rp[2] - ctx.floor_z) <= 0.03
            and not ctx.edge_result.collision_hits
            and 0.20 <= target_dist <= 1.10
            and self.extra_gate_checks(ctx)
        )

        # -- LTL step-0 validation (stabilise objects first) ----------------
        if ctx.gate_pass and ctx.active_objects:
            ltl_ok, ltl_labels = stabilize_and_validate(
                env=env, og_mod=og,
                activity_name=ctx.activity_name,
                scene_model=args.scene_model,
                active_objects_by_inst=ctx.active_objects,
            )
            if not ltl_ok:
                ctx.gate_pass = False
                print(f"[Pipeline] Gate failed: LTL step-0 violations persist: "
                      f"{ltl_labels}")

        print(f"[Pipeline] Gate: pass={ctx.gate_pass}, dist={target_dist:.3f}")
        if args.strict_gate and not ctx.gate_pass:
            raise RuntimeError("Strict gate failed.")

        # -- Save scene snapshot --------------------------------------------
        save_scene = ctx.gate_pass or bool(
            ctx.curation and ctx.curation.save_scene_even_if_gate_fails
        )
        if save_scene:
            # Assign in_rooms to pipeline-spawned objects (task objects +
            # robot) so the saved snapshot supports partial room loading
            # at eval time. Without this, objects have empty in_rooms and
            # OmniGibson's room filter drops them on restore.
            _room = ctx.selection.get("_room_instance", "")
            if _room:
                for obj in list(ctx.active_objects.values()) + ([ctx.robot] if ctx.robot else []):
                    if hasattr(obj, "in_rooms") and _room not in (obj.in_rooms or []):
                        obj.in_rooms = list(set((obj.in_rooms or []) + [_room]))
            scene_save_path = os.path.join(args.run_dir, f"scene_ep{ctx.episode + 1}.json")
            og.sim.save(json_paths=[scene_save_path])
            print(f"[Pipeline] Scene saved: {scene_save_path}")

        # -- LTL rollout ----------------------------------------------------
        camera_override = None
        if ctx.curation and (
            ctx.curation.video_camera_eye is not None or ctx.curation.video_camera_lookat is not None
        ):
            camera_override = {
                "eye": ctx.curation.video_camera_eye,
                "lookat": ctx.curation.video_camera_lookat,
            }
        ctx.ltl_summary, ctx.steps_executed = run_ltl_rollout(
            env=env, activity_name=ctx.activity_name,
            scene_model=args.scene_model,
            active_objects_by_inst=ctx.active_objects,
            robot=ctx.robot, target_obj=ctx.target_obj,
            args=args, episode=ctx.episode, rng=ctx.rng,
            support_obj=ctx.support_obj,
            camera_override=camera_override,
        )
        ctx.resolved_video_views = tuple(getattr(args, "_resolved_video_views", ()))
