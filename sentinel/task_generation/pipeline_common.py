"""Shared infrastructure for task generation pipelines.

Contains helpers for BDDL management, sim interaction, video recording,
pack callbacks, and other utilities reused across different pipeline types
(e.g., table clutter, cabinet clutter).
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
import torch as th

import numpy as np

from sentinel.utils.goal_region import (
    GoalRegionSpec,
    build_goal_region_spec,
    build_task_prompt,
    family_uses_goal_region,
    spawn_goal_region_marker,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")


from sentinel.utils.task_spec import DENSITY_PRESETS  # noqa: F401, E402
from sentinel.utils.task_spec import generate_clutter_activity as generate_activity  # noqa: F401, F811, E402
from sentinel.utils.task_spec import _pick_model_for_category  # noqa: E402

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
    p.add_argument("--surface-model", default=None,
                   help="Pin the support-surface model id (e.g. 'puapey'). "
                        "Pipeline still picks a region within the model based "
                        "on required area.")
    p.add_argument("--surface-category", default=None,
                   help="Pin the support-surface category (e.g. 'desk').")
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


_SUBSTANCE_CATEGORIES = frozenset({
    "water", "juice", "milk", "coffee", "tea", "soup", "oil", "wine",
    "beer", "soda", "sauce", "vinegar", "honey", "syrup", "cream",
})


def needs_gpu_dynamics_from_specs(spawn_specs):
    """Check if any spawn spec requires GPU dynamics (substance/liquid tasks)."""
    for spec in spawn_specs:
        category = spec["category"]
        if category in _SUBSTANCE_CATEGORIES:
            print(f"[Pipeline] GPU dynamics enabled (substance: {category})")
            return True
    return False


def get_scene_json_path(scene_model):
    from omnigibson.utils.asset_utils import get_scene_path
    return os.path.join(
        get_scene_path(scene_model), "json", f"{scene_model}_best.json",
    )


# Placeable-driven scene selection helpers live in utils/placeable.py.
from sentinel.task_generation.utils.placeable import pick_scene_from_placeable


# ---------------------------------------------------------------------------
# Sim-dependent helpers
# ---------------------------------------------------------------------------

def _robot_config():
    return {
        "type": "FrankaMounted", "obs_modalities": ["rgb"],
        "action_type": "continuous", "action_normalize": False,
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
                "command_input_limits": None,
                "command_output_limits": None,
            },
            "gripper_0": {"name": "MultiFingerGripperController"},
        },
    }


def build_task_config(scene_model):
    return {
        "scene": {"type": "InteractiveTraversableScene", "scene_model": scene_model},
        "task": {"type": "DummyTask"},
        "robots": [_robot_config()],
    }



def iter_spawned_objects(spawned_objects):
    for inst, obj in spawned_objects.items():
        yield inst, obj


def get_spawned_obj(spawned_objects, inst):
    return spawned_objects.get(inst)


def spawn_objects(env, spawn_specs, rng, episode_label=""):
    """Import DatasetObjects into the running scene from spawn specs.

    Each spec must contain {"category", "model", "count", "role"}.
    Returns ``(registry, obj_sets)``:
      * ``registry`` — ``{inst_id: DatasetObject}``. ``inst_id`` follows the
        ``"{category}_{N}"`` (or ``"{category}_{episode_label}_{N}"``)
        convention; LTL globs ``"{category}_*"`` match either form.
      * ``obj_sets`` — ``{role: tuple(inst_ids)}``, keyed directly by the
        spec's ``role`` (e.g. ``"target"``, ``"fragile"``, ``"clutter"``,
        ``"food"``, ``"source"``, ``"dest"``, ``"stack"``, ``"lid"``).
        Use ``obj_sets.get(role, ())`` for roles that may not be present.

    ``episode_label`` (optional): a short string (e.g. ``"ep3"``) embedded
    after the category in inst_id and name, so multiple ``spawn_objects``
    calls in one session don't collide. Empty string preserves the original
    ``"{category}_{N}"`` naming for single-episode callers.
    """
    from omnigibson.objects import DatasetObject

    registry = {}
    by_role = {}
    category_counters = {}
    for spec in spawn_specs:
        count = spec["count"]
        role = spec["role"]
        category = spec["category"]
        model = spec.get("model")
        if model is None:
            log.warning("spawn_objects: spec for role=%s category=%s has no model; "
                        "picking randomly (callers should resolve model at selection time)",
                        role, category)
            category, model = _pick_model_for_category(category, rng)

        for _ in range(count):
            idx = category_counters.get(category, 0) + 1
            category_counters[category] = idx
            if episode_label:
                inst_id = f"{category}_{episode_label}_{idx}"
                name = f"{role}_{category}_{episode_label}_{idx}"
            else:
                inst_id = f"{category}_{idx}"
                name = f"{role}_{category}_{idx}"
            obj = DatasetObject(name=name, category=category, model=model)
            env.scene.add_object(obj)
            obj.set_position_orientation(position=[0, 0, 10])
            registry[inst_id] = obj
            by_role.setdefault(role, []).append(inst_id)

    obj_sets = {role: tuple(sorted(ids)) for role, ids in by_role.items()}
    return registry, obj_sets


def is_structural_object(obj):
    name = str(getattr(obj, "name", "") or "").lower()
    cat = str(getattr(obj, "category", "") or "").lower()
    if getattr(obj, "is_system", False):
        return True
    return any(token in name or token in cat for token in STRUCTURAL_CATEGORY_KEYWORDS)


# Categories to hide for cleaner camera angles. Floors are kept visible —
# we want to see the ground under the surface — and so are doors/windows
# (they're useful spatial reference points). Visibility-only: physics,
# collision, and LTL/goal predicates are unaffected.
_HIDE_FOR_CAMERA_KEYWORDS = ("wall", "ceiling", "roof")


def hide_walls_and_ceiling(env):
    """Hide wall + ceiling geometry so external cameras aren't trapped against
    surface back-walls (kitchen counters, etc.). Returns the names of hidden
    objects so callers can log / restore.
    """
    hidden = []
    for obj in env.scene.objects:
        cat = str(getattr(obj, "category", "") or "").lower()
        if not any(t in cat for t in _HIDE_FOR_CAMERA_KEYWORDS):
            continue
        try:
            obj.visible = False
        except Exception as exc:
            log.warning("hide_walls_and_ceiling: %s.visible=False failed: %s",
                        getattr(obj, "name", obj), exc)
            continue
        hidden.append(getattr(obj, "name", ""))
    return hidden


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


def clear_support_area(env, support_obj, surface_bounds_xy, margin_m=0.60,
                       spawned_objects=None):
    """Remove all non-structural preset objects on and around the support surface.

    Removes every object whose xy bounding box overlaps the support surface
    bounds expanded by ``margin_m``, except the support itself, spawned
    objects, and structural objects (walls, floors, etc.).
    """
    import omnigibson as og

    expanded_bounds = _expanded_bounds(surface_bounds_xy, margin_m)
    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for obj in (spawned_objects or {}).values()}

    to_remove = []
    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        if not name or name == support_name or name in scope_names:
            continue
        if is_structural_object(obj):
            continue
        try:
            obj_bounds = _object_bounds_xy(obj)
        except Exception as exc:
            log.warning("clear_support_area: bounds lookup for %s failed: %s", name, exc)
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
                            workspace_rear_m=0.0, spawned_objects=None):
    """Remove preset objects overlapping the chosen Franka/base keepout region."""
    import omnigibson as og

    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for obj in (spawned_objects or {}).values()}
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
        except Exception as exc:
            log.warning(
                "clear_robot_base_region: bounds lookup for %s failed: %s",
                getattr(obj, "name", obj), exc,
            )
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


def object_aabb_dims(obj):
    """Return (dx, dy, dz) AABB dimensions in meters, each floored at 0.01.

    Returns None if obj.aabb raises (e.g., object not yet loaded). Callers
    building clutter / stack descriptors share this so the dimensions are
    computed identically across pipelines.
    """
    try:
        a_min, a_max = obj.aabb
    except Exception as exc:
        log.warning("object_aabb_dims(%s) failed: %s", getattr(obj, "name", obj), exc)
        return None
    return (
        max(0.01, float(a_max[0] - a_min[0])),
        max(0.01, float(a_max[1] - a_min[1])),
        max(0.01, float(a_max[2] - a_min[2])),
    )


def build_descriptors(spawned_objects, obj_sets):
    from sentinel.utils.clutter_pack_layout import ClutterObjectDescriptor

    descriptors, objects_by_inst = [], {}
    for role in ("target", "fragile", "clutter"):
        for inst in obj_sets.get(role, ()):
            obj = spawned_objects.get(inst)
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
            except Exception as exc:
                log.warning(
                    "build_descriptors: pose/aabb lookup for %s failed: %s", inst, exc,
                )
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
            except Exception as exc:
                # Expected fallback — try the next link candidate.
                log.debug("robot_half_extent_xy: link %s aabb failed: %s", key, exc)
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
            for name, obj in objs.items():
                try:
                    vel = obj.get_linear_velocity()
                    vz = float(vel[2]) if hasattr(vel, '__getitem__') else 0.0
                    obj.set_linear_velocity(th_mod.tensor([0.0, 0.0, min(0.0, vz)]))
                    obj.set_angular_velocity(th_mod.zeros(3))
                except Exception as exc:
                    log.warning("settle: zero-velocity call for %s failed: %s", name, exc)
        for name, obj in objs.items():
            try:
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            except Exception as exc:
                log.warning("settle: keep_still for %s failed: %s", name, exc)
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
            except Exception as exc:
                log.warning(
                    "stabilize_active_objects: support %s zero-velocity/keep_still failed: %s",
                    getattr(support_obj, "name", support_obj), exc,
                )
        for name, obj in objs.items():
            try:
                if hasattr(obj, "set_linear_velocity"):
                    obj.set_linear_velocity(th.zeros(3))
                if hasattr(obj, "set_angular_velocity"):
                    obj.set_angular_velocity(th.zeros(3))
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            except Exception as exc:
                log.warning(
                    "stabilize_active_objects: %s zero-velocity/keep_still failed: %s",
                    name, exc,
                )
        og_mod.sim.step()



def pin_support_object_to_world(support_obj):
    if support_obj is None:
        return False
    if bool(getattr(support_obj, "fixed_base", False)):
        return False
    try:
        joint_path = f"{support_obj.prim_path}/pipelineFixedJoint"
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
    except Exception as exc:
        log.warning(
            "pin_support_object_to_world(%s) failed: %s",
            getattr(support_obj, "name", support_obj), exc,
        )
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
            except Exception as exc:
                log.warning("park: %s set_position/keep_still failed: %s", inst, exc)
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
        pos = obj.get_position_orientation()[0]
        if not all(math.isfinite(float(pos[i])) for i in range(3)):
            invalid.append(inst)
    return invalid


def check_interpenetration(objs, tol):
    inst_ids = sorted(objs.keys())
    hits = []
    for i, a in enumerate(inst_ids):
        try:
            aabb_a = objs[a].aabb
        except Exception as exc:
            log.warning("check_interpenetration: aabb lookup for %s failed: %s", a, exc)
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
            except Exception as exc:
                log.warning(
                    "check_interpenetration: aabb overlap (%s, %s) failed: %s", a, b, exc,
                )
                continue
    return hits


# Camera + video helpers live in utils/video.py; re-export the ones
# run_ltl_rollout and external callers import from pipeline_common.
from sentinel.task_generation.utils.video import (  # noqa: F401
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
        except Exception as exc:
            log.warning("_try_upright_objects: %s reset failed: %s", inst, exc)
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
    from sentinel.utils.safety_monitor import TaskLTLMonitor

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
    from sentinel.utils.safety_monitor import TaskLTLMonitor

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
                    except Exception as exc:
                        log.warning(
                            "video mux failed for cam %s: %s", writer_info["cam"], exc,
                        )

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
    ltl_safety: Dict = field(default_factory=dict)

    # Objects (populated by spawn_objects + identify_objects)
    spawned_objects: Dict[str, Any] = field(default_factory=dict)
    target_obj: Any = None
    active_objects: Dict[str, Any] = field(default_factory=dict)

    # Robot
    robot: Any = None
    edge_result: Any = None

    # Gate
    gate_pass: bool = False
    goal_region: Optional[Dict] = None
    prompt: str = ""

    # Episode index
    episode: int = 0


class BasePipeline(ABC):
    """Base class for table-based task generation pipelines.

    Subclasses implement the pipeline-specific hooks:
      - add_args()          — register CLI flags
      - activity_prefix()   — default activity name prefix
      - generate_activity() — produce LTL safety + selection with spawn_specs
      - configure_env()     — tweak env/macros after load (e.g. GPU dynamics)
      - identify_objects()  — partition spawned objects into roles
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
    def generate_activity(self, activity_name, support_category, support_room,
                          args, rng):
        """Generate LTL safety spec and object selection with spawn specs.

        Returns (ltl_safety, selection) where selection contains "spawn_specs".
        """

    def configure_env(self, selection):
        """Optional hook to configure macros before env creation.

        For example, enable GPU dynamics for liquid tasks.  Default: no-op.
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

    def scene_family(self, ctx):
        """Return canonical family string for prompt / goal-region logic."""
        return None

    def goal_region_pack_object_names(self, ctx):
        return tuple(
            getattr(obj, "name", "")
            for obj in ctx.active_objects.values()
            if getattr(obj, "name", "")
        )

    def goal_region(self, ctx):
        family = self.scene_family(ctx)
        if not family or not family_uses_goal_region(family) or ctx.target_obj is None:
            return None
        pack_names = tuple(name for name in self.goal_region_pack_object_names(ctx) if name)
        support_name = str(getattr(ctx.support_obj, "name", "") or ctx.surface_name)
        if not pack_names or not support_name:
            return None
        # Use a per-episode rng derived from the run seed so the goal
        # sphere lands in a different reachable spot each episode while
        # remaining reproducible for a given seed/episode pair.
        ep_rng = np.random.default_rng(int(ctx.args.seed) + 13_000 * (ctx.episode + 1))
        spec = build_goal_region_spec(
            env=ctx.env,
            diagnostics={
                "pipeline": family,
                "surface": support_name,
                "selection": ctx.selection,
                "support_selection": {"result_world_bounds_xy": ctx.surface_bounds_xy},
            },
            family=family,
            target_name=str(getattr(ctx.target_obj, "name", "")),
            support_name=support_name,
            pack_object_names=pack_names,
            rng=ep_rng,
        )
        return spec.to_json()

    def task_prompt(self, ctx):
        family = self.scene_family(ctx)
        if not family:
            return ""
        diagnostics = {
            "pipeline": family,
            "surface": str(getattr(ctx.support_obj, "name", "") or ctx.surface_name),
            "selection": copy.deepcopy(ctx.selection),
        }
        scene_info = {
            "objects_info": {
                "init_info": {
                    str(getattr(ctx.support_obj, "name", "") or ctx.surface_name): {
                        "args": {"category": str(getattr(ctx.support_obj, "category", "") or "")}
                    }
                }
            }
        }
        return build_task_prompt(scene_info, diagnostics, goal_region=ctx.goal_region)

    @abstractmethod
    def select_objects(self, args, rng):
        """Pre-select objects and estimate required surface area.

        Must return a dict with at minimum ``"required_area_m2"`` plus
        pipeline-specific object selections.
        """

    def extra_gate_checks(self, ctx):
        """Additional gate conditions beyond the shared ones.  Default: True."""
        return True

    def goal_conditions(self, ctx):
        """Return resolved goal conditions for this episode.

        Each pipeline subclass fills in its goal template with actual
        scene object names from ctx. Returns a list of dicts:
            [{"predicate": "inside", "subject": "potato_124", "reference": "stockpot_122"}]

        Stored in diagnostics.jsonl as "goal_conditions" and consumed by
        sentinel.eval.goal_checker at eval time. Default returns [].
        """
        return []

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
        pick = pick_scene_from_placeable(
            rng_pre, required,
            scene_model=args.scene_model,
            required_category=getattr(args, "surface_category", None),
            required_model=getattr(args, "surface_model", None),
        )
        args.scene_model = pick["scene_model"]
        args._picked_surface = pick
        print(f"[Pipeline] Scene={pick['scene_model']}, "
              f"surface={pick['category']}/{pick['model']} in {pick['room_instance']}, "
              f"area={pick['area_m2']:.3f} m² (required={required:.3f} m²)")

        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"

        support_category = pick["category"]
        room_instance = pick["room_instance"]
        support_room = strip_room_suffix(room_instance)

        rng = np.random.default_rng(args.seed)
        ltl_safety, selection = \
            self.generate_activity(activity_name, support_category, support_room, args, rng)

        print("[Pipeline] Dry-run complete:")
        print(f"  activity:   {activity_name}")
        print(f"  spawn_specs: {selection.get('spawn_specs', [])}")
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
        # Pre-select args.episodes triples so each episode uses a different
        # (food, source, dest) combination. The surface must accommodate the
        # LARGEST triple (max required_area_m2) — sum would be wrong since
        # only one triple exists on the surface at a time.
        rng_pre = np.random.default_rng(args.seed)
        pre_selections = [self.select_objects(args, rng_pre)
                          for _ in range(args.episodes)]
        required_areas = [s["required_area_m2"] for s in pre_selections]
        required = max(required_areas)
        print(f"[Pipeline] Pre-selected {len(pre_selections)} episode triples; "
              f"required_area max={required:.3f} m² "
              f"(min={min(required_areas):.3f}, "
              f"mean={sum(required_areas)/len(required_areas):.3f})")

        pick = pick_scene_from_placeable(
            rng_pre, required,
            scene_model=args.scene_model,
            required_category=getattr(args, "surface_category", None),
            required_model=getattr(args, "surface_model", None),
        )
        args.scene_model = pick["scene_model"]
        args._picked_surface = pick
        print(f"[Pipeline] Scene={pick['scene_model']}, "
              f"surface={pick['category']}/{pick['model']} in {pick['room_instance']}, "
              f"area={pick['area_m2']:.3f} m² (required={required:.3f} m²)")

        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"

        # -- Resolve support surface ----------------------------------------
        scene_json = get_scene_json_path(args.scene_model)
        if not os.path.isfile(scene_json):
            raise RuntimeError(f"Scene JSON not found: {scene_json}")
        surface_category = pick["category"]
        room_instance = pick["room_instance"]
        support_room = strip_room_suffix(room_instance)
        print(f"[Pipeline] Discovered: category={surface_category} "
              f"room={support_room}")

        # Annotate every pre_selection with the resolved scene info so
        # generate_activity (which reads args._pre_selection) sees it.
        for pre_sel in pre_selections:
            pre_sel.setdefault("_surface_category", surface_category)
            pre_sel.setdefault("_room_type", support_room)
            pre_sel.setdefault("_room_instance", room_instance)

        # -- Generate per-episode activities (LTL + spawn specs) ------------
        rng = np.random.default_rng(args.seed)
        episode_activities = []
        for pre_sel in pre_selections:
            args._pre_selection = pre_sel
            ep_ltl, ep_sel = self.generate_activity(
                activity_name, surface_category, support_room, args, rng,
            )
            ep_sel.setdefault("_room_instance", room_instance)
            episode_activities.append((ep_ltl, ep_sel))
        # Reset args._pre_selection to ep0 for any downstream reads during
        # env construction; the per-episode swap happens inside the loop.
        args._pre_selection = pre_selections[0]
        ltl_safety, selection = episode_activities[0]

        # -- GPU dynamics ----------------------------------------------------
        # OR across all episodes' specs: if any triple needs GPU dynamics
        # (substance/liquid), the env has to be configured for it from the
        # start since we can't toggle GPU dynamics mid-session.
        gpu = any(
            needs_gpu_dynamics_from_specs(sel.get("spawn_specs", []))
            for _, sel in episode_activities
        )
        gm.USE_GPU_DYNAMICS = gpu
        gm.ENABLE_FLATCACHE = not gpu

        self.configure_env(selection)

        # -- Load environment ------------------------------------------------
        cfg = build_task_config(args.scene_model)
        cfg["scene"]["scene_file"] = scene_json
        cfg["scene"]["scene_instance"] = None
        if room_instance:
            cfg["scene"]["load_room_instances"] = [room_instance]
            print(f"[Pipeline] Partial load: room={room_instance}")

        # 3 external cameras (canonical names + resolution shared across
        # task-generation, teleop, training, eval).
        from sentinel.utils.camera_setup import (
            CAMERA_RESOLUTION,
            build_external_camera_configs,
        )
        cfg.setdefault("env", {})["external_sensors"] = build_external_camera_configs(
            resolution=CAMERA_RESOLUTION,
        )

        print(f"[Pipeline] scene={scene_label}, activity={activity_name}, "
              f"strict_gate={args.strict_gate}")
        env = og.Environment(configs=cfg)

        # OG bug workaround (StanfordVL/OmniGibson#266, #1875): sensor_kwargs
        # in env config doesn't reliably set image_height/image_width on
        # creation. Explicitly set each sensor's resolution and reload the
        # observation space so downstream consumers see the right shape.
        ext_sensors = env.external_sensors or {}
        if ext_sensors:
            for cam in ext_sensors.values():
                cam.image_height = CAMERA_RESOLUTION
                cam.image_width = CAMERA_RESOLUTION
            env.load_observation_space()
        exit_code = 0

        try:
            # Single ctx persists across episodes. _setup_session runs the
            # one-time work (env.reset, scene cleanup, robot mount); each
            # _run_episode then refreshes only the task objects. This
            # avoids env.reset() between episodes — that path tries to
            # revive objects clear_support_area removed (kitchen
            # toggleables crash on visual_marker re-init upstream).
            # Initialize ctx with episode 0's activity. _setup_session reads
            # ctx._episode_activities to spawn ALL episodes' task objects
            # upfront (parking ep>0 far away).
            ctx = EpisodeContext(
                env=env, og=og, args=args, rng=rng,
                activity_name=activity_name,
                selection=episode_activities[0][1],
                ltl_safety=episode_activities[0][0],
                episode=0,
            )
            ctx._episode_activities = episode_activities
            self._setup_session(ctx)

            for ep in range(args.episodes):
                ctx.episode = ep
                # Swap in this episode's pre-selected activity. _run_episode
                # reads ctx.selection["spawn_specs"] for the respawn (ep>0).
                ctx.ltl_safety, ctx.selection = (
                    episode_activities[ep][0],
                    episode_activities[ep][1],
                )
                args._pre_selection = pre_selections[ep]

                spawn_specs = ctx.selection.get("spawn_specs", [])
                triple = ", ".join(
                    f"{s.get('role')}={s.get('category')}/{s.get('model')}"
                    for s in spawn_specs
                )
                print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
                print(f"[Pipeline] Triple: {triple}")
                self._run_episode(ctx)

                payload = {
                    "episode": ep + 1,
                    "scene_model": scene_label,
                    "activity_name": activity_name,
                    "surface": ctx.surface_name,
                    "prompt": ctx.prompt or None,
                    "gate_pass": ctx.gate_pass,
                    "ltl_violated": ctx.ltl_summary.get("violated") if hasattr(ctx, "ltl_summary") else None,
                    "steps_executed": ctx.steps_executed if hasattr(ctx, "steps_executed") else 0,
                    "selection": ctx.selection,
                    "ltl_safety": ctx.ltl_safety,
                    "cameras": list(getattr(args, "_resolved_video_views", ())),
                    "goal_conditions": self.goal_conditions(ctx),
                    **self.diagnostics_extra(ctx),
                }
                if ctx.goal_region is not None:
                    payload["goal_region"] = copy.deepcopy(ctx.goal_region)
                append_jsonl(args.debug_jsonl, payload)
        except Exception:
            exit_code = 1
            print("[Pipeline] ERROR: pipeline execution failed.")
            traceback.print_exc()
        finally:
            print("[Pipeline] Shutdown simulator.")
            pipeline_exit(exit_code)

    def _setup_session(self, ctx):
        """Run one-time scene/robot setup before the episode loop.

        Spawns ALL episodes' task objects upfront with episode-labelled
        inst_ids (e.g. ``bowl_ep1_1``, ``bowl_ep5_1``). Episode-0's objects
        stay in the parking position the spawner leaves them at; later
        episodes' objects get teleported to a far parking pose so only
        episode 0's task objects participate in the surface/mount setup.

        Per-episode work is then: park the previous episode's 3 objects,
        teleport the current episode's 3 onto the surface — no scene
        mutations (add/remove) during the run, which sidesteps OG's
        registry-staleness on objects added while the sim is playing.
        """
        env, og, args = ctx.env, ctx.og, ctx.args
        env.reset()
        og.sim.step()

        # Hide walls + ceiling — visibility-only, physics/gates unaffected.
        # Surfaces mounted on walls (kitchen countertops) otherwise trap the
        # external cameras inside the wall and produce solid-color frames.
        hidden = hide_walls_and_ceiling(env)
        if hidden:
            print(f"[Pipeline] Hid {len(hidden)} structural objects "
                  f"(walls/ceiling) for camera clearance")

        # -- Spawn ALL episodes' task objects upfront -----------------------
        episode_activities = ctx._episode_activities
        per_ep_spawned = []
        per_ep_obj_sets = []
        all_spawned = {}
        for ep, (_, sel) in enumerate(episode_activities):
            specs = sel.get("spawn_specs", [])
            if not specs:
                per_ep_spawned.append({})
                per_ep_obj_sets.append({})
                continue
            ep_label = f"ep{ep + 1}"
            spawned, obj_sets = spawn_objects(
                env, specs, ctx.rng, episode_label=ep_label,
            )
            per_ep_spawned.append(spawned)
            per_ep_obj_sets.append(obj_sets)
            all_spawned.update(spawned)

        og.sim.step()
        print(f"[Pipeline] Spawned {len(all_spawned)} task objects across "
              f"{len(per_ep_spawned)} episodes")

        # Park ep > 0 objects far from the active workspace. Each episode
        # gets its own row so they can't physically interfere even if some
        # drift downward under gravity over the duration of the run.
        for ep in range(1, len(per_ep_spawned)):
            park = (50.0 + ep * 2.0, 50.0, 10.0)
            for obj in per_ep_spawned[ep].values():
                obj.set_position_orientation(position=park)
                obj.keep_still()
        og.sim.step()

        ctx._per_ep_spawned = per_ep_spawned
        ctx._per_ep_obj_sets = per_ep_obj_sets
        ctx._all_spawned = all_spawned
        ctx.spawned_objects = per_ep_spawned[0]
        ctx.obj_sets = per_ep_obj_sets[0]

        # -- Find support surface object ------------------------------------
        # Prefer exact (category, model) match from placeable pick for
        picked = getattr(args, "_picked_surface", None)
        target_category = picked["category"] if picked else args._pre_selection["_surface_category"]
        target_model = picked["model"] if picked else None

        support_obj = None
        for obj in env.scene.objects:
            if getattr(obj, "category", "") != target_category:
                continue
            if target_model and getattr(obj, "model", "") != target_model:
                continue
            support_obj = obj
            break

        if support_obj is None:
            detail = f"{target_category}/{target_model}" if target_model else target_category
            raise RuntimeError(f"Support surface '{detail}' not found in scene objects.")

        ctx.support_obj = support_obj
        ctx.surface_name = getattr(support_obj, "name", "")

        print(f"[Pipeline] Support surface: {ctx.surface_name} ({target_category})")
        # Pin support first so it cannot move, then compute geometry once.
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
        # Top of the *placeable region*, not the object's full AABB. Matches
        # the convention in empty_scene_pipeline: world top z = support's
        # local-origin world z + top_plane_z_local (scaled). This matters
        # for objects with vertical features rising above the placeable
        # plane (e.g. desk/puapey's central divider, where aabb_max[2] is
        # the divider top, ~30 cm above the desktop).
        support_pos, _ = support_obj.get_position_orientation()
        scale_z = float(support_obj.scale[2])
        top_plane_local = float(picked.get("top_plane_z_local", 0.0))
        ctx.table_top_z = float(support_pos[2]) + top_plane_local * scale_z
        ctx.floor_z = float(aabb_min[2])

        # Analyze the surface for obstacle/approach info using the picked
        # region's bounds and the placeable plane's z (NOT the full object
        # AABB) so 2-region models analyse only the half they were assigned.
        from sentinel.utils.surface_discovery import analyze_surface
        try:
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
                except Exception as exc:
                    log.warning(
                        "surface analysis: aabb for %s failed: %s",
                        getattr(obj, "name", obj), exc,
                    )
                    continue
            other_aabbs = [
                d["aabb_xy"] for d in scene_data
                if d["name"] != ctx.surface_name
                and d["top_z"] >= 0.15
                and d.get("bottom_z", 0) <= ctx.table_top_z + 0.3
            ]
            ctx.surface_info = analyze_surface(
                ctx.surface_name, target_category, ctx.surface_bounds_xy,
                ctx.table_top_z, scene_data, scene_object_aabbs=other_aabbs,
            )
        except Exception as exc:
            log.warning("surface analysis failed: %s", exc)
            ctx.surface_info = None

        clear_margin = args.perimeter_clear_margin_m if args.perimeter_clear_margin_m is not None else 0.60
        ctx.removed_area_objects = clear_support_area(
            env, support_obj, ctx.surface_bounds_xy, margin_m=clear_margin,
            spawned_objects=ctx._all_spawned,
        )
        if ctx.removed_area_objects:
            og.sim.step()

        # -- Pipeline-specific: identify & place objects --------------------
        self.identify_objects(ctx)
        self.place_objects(ctx)

        # -- Robot placement ------------------------------------------------
        from sentinel.utils.franka_edge_align import (
            DEFAULT_ROLE_WEIGHTS, EdgeAlignRequest, EdgeAlignResult, _quat_from_yaw, place_franka_edge_aligned,
        )
        from sentinel.utils.tabletop_workspace import compute_tabletop_zone

        ctx.robot = env.robots[0]

        if hasattr(ctx, "_zone") and ctx._zone is not None:
            zone = ctx._zone
        else:
            obstacle_bounds_xy = None
            obstacle_bounds_seq = []
            if ctx.surface_info and ctx.surface_info.obstacles:
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
        if ctx.surface_info and ctx.surface_info.approach_edges:
            preferred_edge = ctx.surface_info.approach_edges[0]

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
                spawned_objects=ctx._all_spawned,
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

    def _run_episode(self, ctx):
        """Run one episode: swap in this episode's task objects, gate, rollout.

        All ``args.episodes`` triples were spawned upfront in
        :meth:`_setup_session`. Each episode just teleports its 3 active
        objects onto the support surface (via :meth:`place_objects`) and
        parks the previous episode's 3 back to their far parking pose.
        No scene mutation (add/remove) happens here.
        """
        env, og, args = ctx.env, ctx.og, ctx.args

        # -- Swap active objects for episodes > 0 ---------------------------
        if ctx.episode > 0:
            # Park previous episode's task objects far away. They keep
            # existing in the scene but are out of the workspace and
            # excluded from ctx.active_objects (which the LTL monitor and
            # gate look at).
            prev_ep = ctx.episode - 1
            park = (50.0 + prev_ep * 2.0, 50.0, 10.0)
            for obj in ctx._per_ep_spawned[prev_ep].values():
                obj.set_position_orientation(position=park)
                obj.keep_still()

            # Activate this episode's pre-spawned objects.
            ctx.spawned_objects = ctx._per_ep_spawned[ctx.episode]
            ctx.obj_sets = ctx._per_ep_obj_sets[ctx.episode]
            print(f"[Pipeline] Activated {len(ctx.spawned_objects)} objects: "
                  f"{sorted(ctx.spawned_objects.keys())}")

            self.identify_objects(ctx)
            self.place_objects(ctx)
            og.sim.step()
            post_mount_settle_steps = int(getattr(args, "post_mount_settle_steps", 0) or 0)
            if post_mount_settle_steps > 0:
                stabilize_active_objects(
                    og, ctx.active_objects, post_mount_settle_steps,
                    support_obj=ctx.support_obj,
                )

        # Reset per-episode rollout state (kept fresh each call).
        ctx.gate_pass = False
        ctx.goal_region = None
        ctx.prompt = ""

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

        ctx.goal_region = self.goal_region(ctx)
        if ctx.goal_region is not None:
            # Park the previous episode's marker (if any) far below the
            # floor and hide it, so only this episode's sphere shows.
            # Mid-play env.scene.remove_object() trips OmniGibson's
            # registry-staleness bug — the same reason we park task
            # objects in _setup_session instead of removing them.
            prev_marker = getattr(ctx, "_prev_goal_marker", None)
            if prev_marker is not None:
                try:
                    prev_marker.set_position_orientation(
                        position=(100.0, 100.0, -10.0),
                        orientation=(0.0, 0.0, 0.0, 1.0),
                    )
                    prev_marker.visible = False
                except Exception as exc:
                    log.warning("park previous goal marker failed: %s", exc)
            ctx._prev_goal_marker = spawn_goal_region_marker(
                ctx.env, GoalRegionSpec.from_json(ctx.goal_region),
            )
            og.sim.step()
        ctx.prompt = self.task_prompt(ctx)

        # -- Save scene snapshot --------------------------------------------
        save_scene = ctx.gate_pass
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
        ctx.ltl_summary, ctx.steps_executed = run_ltl_rollout(
            env=env, activity_name=ctx.activity_name,
            scene_model=args.scene_model,
            active_objects_by_inst=ctx.active_objects,
            robot=ctx.robot, target_obj=ctx.target_obj,
            args=args, episode=ctx.episode, rng=ctx.rng,
            support_obj=ctx.support_obj,
        )
        ctx.resolved_video_views = tuple(getattr(args, "_resolved_video_views", ()))
