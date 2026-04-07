"""Shared infrastructure for task generation pipelines.

Contains helpers for BDDL management, sim interaction, video recording,
pack callbacks, and other utilities reused across different pipeline types
(e.g., table clutter, cabinet clutter).
"""

import argparse
import json
import math
import os
import shutil
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

DEFAULT_VIDEO_CANDIDATE_MODE = "support_relative_v1"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def make_base_arg_parser(description="Task generation pipeline"):
    """Create an argument parser with args common to all pipelines."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--scene-model", required=True)
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
    p.add_argument("--video-viewer-only", action="store_true")
    p.add_argument("--video-candidate-mode", default=DEFAULT_VIDEO_CANDIDATE_MODE)
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
    """Find (category, room) of best matching object from scene JSON. No sim needed."""
    with open(scene_json_path, "r", encoding="utf-8") as f:
        init_infos = json.load(f).get("objects_info", {}).get("init_info", {})

    candidates = []
    for info in init_infos.values():
        args = info.get("args", {})
        cat = args.get("category", "")
        if not category_filter_fn(cat):
            continue
        rooms = args.get("in_rooms", [])
        room = strip_room_suffix(rooms[0]) if rooms else "living_room"
        candidates.append((cat, room))

    if not candidates:
        return None
    if priority_map:
        candidates.sort(key=lambda c: priority_map.get(c[0], 0), reverse=True)
    return candidates[0]


def estimate_surface_area_from_scene_json(scene_json_path, surface_category):
    """Estimate the XY surface area (m²) of a table from the scene JSON.

    Reads the model's base AABB from asset metadata, applies the scene scale,
    and returns the XY footprint.  Returns None if the data is unavailable.
    """
    import glob as globmod

    with open(scene_json_path, "r", encoding="utf-8") as f:
        init_infos = json.load(f).get("objects_info", {}).get("init_info", {})

    # Find the first object matching the surface category.
    for info in init_infos.values():
        obj_args = info.get("args", {})
        if obj_args.get("category", "") != surface_category:
            continue
        model = obj_args.get("model", "")
        scale = obj_args.get("scale", [1.0, 1.0, 1.0])
        if not model:
            continue

        # Look up the model's base AABB extent from asset metadata.
        asset_base = os.path.join(
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            "datasets", "behavior-1k-assets", "objects", surface_category,
        )
        meta_paths = globmod.glob(os.path.join(asset_base, model, "misc", "metadata.json"))
        if not meta_paths:
            continue
        with open(meta_paths[0], "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        try:
            ext = meta["link_bounding_boxes"]["base_link"]["collision"]["axis_aligned"]["extent"]
        except (KeyError, TypeError):
            continue
        area = (ext[0] * scale[0]) * (ext[1] * scale[1])
        return area
    return None


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


def build_descriptors(env, obj_sets):
    from omnigibson.utils.clutter_pack_layout import ClutterObjectDescriptor

    descriptors, objects_by_inst = [], {}
    for role, id_key in [("target", "target_ids"), ("fragile", "fragile_ids"), ("clutter", "clutter_ids")]:
        for inst in obj_sets[id_key]:
            obj = get_scope_obj(env, inst)
            if obj is None:
                continue
            try:
                aabb_min, aabb_max = obj.aabb
                obj_pos = obj.get_position_orientation()[0]
                dx = max(0.01, float(aabb_max[0] - aabb_min[0]))
                dy = max(0.01, float(aabb_max[1] - aabb_min[1]))
                dz = max(0.01, float(aabb_max[2] - aabb_min[2]))
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


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def init_video_writer(base_path, episode, fps, robot=None):
    try:
        import av
    except ImportError:
        print("[Pipeline] WARNING: PyAV not available — video recording disabled.")
        return None
    import omnigibson as og

    try:
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        vh, vw = int(rgb.shape[0]), int(rgb.shape[1])
    except Exception:
        vh, vw = 720, 1280

    # Find wrist camera for picture-in-picture overlay.
    wrist = None
    wrist_name = None
    sensor_names = []
    if robot:
        from omnigibson.sensors import VisionSensor
        for name, sensor in robot.sensors.items():
            sensor_names.append(name)
            if isinstance(sensor, VisionSensor) and "hand" in name.lower():
                wrist = sensor
                wrist_name = name
                break
        if wrist is None:
            for name, sensor in robot.sensors.items():
                if isinstance(sensor, VisionSensor):
                    wrist = sensor
                    wrist_name = name
                    break
    wh, ww = 0, 0
    if wrist:
        try:
            wrist_rgb = wrist.get_obs()[0].get("rgb")
            if wrist_rgb is not None:
                wh, ww = int(wrist_rgb.shape[0]), int(wrist_rgb.shape[1])
        except Exception:
            pass

    if wh > 0 and ww > 0:
        scale = vh / wh
        scaled_ww = int(ww * scale)
        total_w = vw + scaled_ww
    else:
        total_w = vw

    stem = base_path[:-4] if base_path.endswith(".mp4") else base_path
    fpath = f"{stem}_ep{episode + 1}.mp4"
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)

    container = av.open(fpath, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = total_w
    stream.height = vh
    stream.pix_fmt = "yuv420p"

    if robot:
        print(
            "[Pipeline] Robot sensors: "
            + (", ".join(sensor_names) if sensor_names else "<none>")
        )
    if wrist_name:
        print(f"[Pipeline] Wrist sensor selected: {wrist_name} ({ww}x{wh})")
    elif robot:
        print("[Pipeline] Wrist sensor selected: <none>; falling back to viewer-only frames")

    return {"container": container, "stream": stream, "wrist": wrist,
            "wrist_name": wrist_name, "viewer_hw": (vh, vw), "wrist_hw": (wh, ww)}


def expected_video_path(base_path, episode):
    stem = base_path[:-4] if base_path.endswith(".mp4") else base_path
    return f"{stem}_ep{episode + 1}.mp4"


def expected_labeled_video_path(base_path, episode, label):
    stem = base_path[:-4] if base_path.endswith(".mp4") else base_path
    return f"{stem}_{label}_ep{episode + 1}.mp4"


def _sanitize_view_label(label):
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(label).strip())
    return cleaned or "view"


def _object_world_position(obj):
    return np.asarray([float(v) for v in obj.get_position_orientation()[0][:3]], dtype=np.float32)


def _support_relative_video_views(robot, target_obj, support_obj=None, active_objects_by_inst=None):
    rp = _object_world_position(robot)
    tp = _object_world_position(target_obj)

    if support_obj is not None:
        try:
            aabb_min, aabb_max = support_obj.aabb
            x0, y0, z0 = [float(v) for v in aabb_min[:3]]
            x1, y1, z1 = [float(v) for v in aabb_max[:3]]
            support_center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z1], dtype=np.float32)
            hx = max(0.10, 0.5 * abs(x1 - x0))
            hy = max(0.10, 0.5 * abs(y1 - y0))
            table_top_z = z1
        except Exception:
            support_center = np.asarray([(rp[0] + tp[0]) * 0.5, (rp[1] + tp[1]) * 0.5, max(rp[2], tp[2])], dtype=np.float32)
            hx, hy = 0.35, 0.35
            table_top_z = float(max(rp[2], tp[2]))
    else:
        support_center = np.asarray([(rp[0] + tp[0]) * 0.5, (rp[1] + tp[1]) * 0.5, max(rp[2], tp[2])], dtype=np.float32)
        hx, hy = 0.35, 0.35
        table_top_z = float(max(rp[2], tp[2]))

    cluster_positions = []
    if active_objects_by_inst:
        for obj in active_objects_by_inst.values():
            try:
                cluster_positions.append(_object_world_position(obj))
            except Exception:
                continue
    if cluster_positions:
        cluster_center = np.mean(np.stack(cluster_positions, axis=0), axis=0)
    else:
        cluster_center = tp

    lookat = np.asarray([
        float(0.40 * support_center[0] + 0.60 * cluster_center[0]),
        float(0.40 * support_center[1] + 0.60 * cluster_center[1]),
        float(max(table_top_z + 0.15, cluster_center[2], tp[2])),
    ], dtype=np.float32)

    toward_support = support_center[:2] - rp[:2]
    norm = float(np.linalg.norm(toward_support))
    if norm < 1e-6:
        toward_support = cluster_center[:2] - rp[:2]
        norm = float(np.linalg.norm(toward_support))
    if norm < 1e-6:
        toward_support = np.asarray([0.0, 1.0], dtype=np.float32)
        norm = 1.0
    forward = toward_support / norm
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float32)

    primary_extent = abs(forward[0]) * hx + abs(forward[1]) * hy
    lateral_extent = abs(lateral[0]) * hx + abs(lateral[1]) * hy
    front_dist = max(primary_extent + 0.55, 0.80)
    side_dist = max(lateral_extent + 0.65, 0.95)
    low_z = max(table_top_z + 0.40, tp[2] + 0.30, rp[2] + 0.70)
    high_z = low_z + 0.28
    topdown_z = max(table_top_z + 1.25, high_z + 0.25)

    return [
        {
            "label": "opposite_side_front",
            "eye": (
                float(lookat[0] + forward[0] * front_dist),
                float(lookat[1] + forward[1] * front_dist),
                float(low_z),
            ),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": True,
        },
        {
            "label": "left_overview",
            "eye": (
                float(lookat[0] + forward[0] * (front_dist * 0.45) + lateral[0] * side_dist),
                float(lookat[1] + forward[1] * (front_dist * 0.45) + lateral[1] * side_dist),
                float(high_z),
            ),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": False,
        },
        {
            "label": "right_overview",
            "eye": (
                float(lookat[0] + forward[0] * (front_dist * 0.45) - lateral[0] * side_dist),
                float(lookat[1] + forward[1] * (front_dist * 0.45) - lateral[1] * side_dist),
                float(high_z),
            ),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": False,
        },
    ]


def build_video_view_specs(args, robot, target_obj, support_obj=None,
                           active_objects_by_inst=None, camera_override=None):
    rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
    tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
    default_center = [0.5 * (rp[0] + tp[0]), 0.5 * (rp[1] + tp[1]), max(rp[2] + 0.7, tp[2] + 0.25)]
    default_eye = [default_center[0] - 1.0, default_center[1] - 1.1, default_center[2] + 0.5]
    if camera_override and camera_override.get("lookat") is not None:
        default_center = [float(v) for v in camera_override["lookat"]]
    if camera_override and camera_override.get("eye") is not None:
        default_eye = [float(v) for v in camera_override["eye"]]

    candidate_views = tuple(getattr(args, "video_candidate_views", ()) or ())
    candidate_mode = str(getattr(args, "video_candidate_mode", "") or "").strip() or DEFAULT_VIDEO_CANDIDATE_MODE
    if not candidate_views:
        if candidate_mode == "support_relative_v1":
            views = _support_relative_video_views(
                robot=robot,
                target_obj=target_obj,
                support_obj=support_obj,
                active_objects_by_inst=active_objects_by_inst,
            )
            final_label = getattr(args, "video_final_view", None)
            if final_label:
                normalized = _sanitize_view_label(final_label)
                for view in views:
                    view["canonical"] = view["label"] == normalized
                if not any(view["canonical"] for view in views):
                    views[0]["canonical"] = True
            return views
        scene_entry = getattr(args, "_scene_curation", None)
        issue_tags = set(getattr(scene_entry, "issue_tags", ()) or ())
        if issue_tags & {"missing_third_person_video", "bad_camera_framing"}:
            dx = default_eye[0] - default_center[0]
            dy = default_eye[1] - default_center[1]
            dz = default_eye[2] - default_center[2]
            return [
                {
                    "label": "diag_left_far",
                    "eye": tuple(default_eye),
                    "lookat": tuple(default_center),
                    "canonical": True,
                },
                {
                    "label": "diag_left_mid",
                    "eye": (
                        default_center[0] + dx * 0.78,
                        default_center[1] + dy * 0.78,
                        default_center[2] + dz * 0.82,
                    ),
                    "lookat": tuple(default_center),
                    "canonical": False,
                },
                {
                    "label": "diag_right_mid",
                    "eye": (
                        default_center[0] - dx * 0.72,
                        default_center[1] + dy * 0.72,
                        default_center[2] + dz * 0.82,
                    ),
                    "lookat": tuple(default_center),
                    "canonical": False,
                },
            ]
        return [{
            "label": "default",
            "eye": tuple(default_eye),
            "lookat": tuple(default_center),
            "canonical": True,
        }]

    final_label = getattr(args, "video_final_view", None)
    views = []
    for idx, raw in enumerate(candidate_views):
        label = _sanitize_view_label(raw["label"])
        views.append({
            "label": label,
            "eye": tuple(float(v) for v in raw["eye"]),
            "lookat": tuple(float(v) for v in raw["lookat"]),
            "canonical": label == _sanitize_view_label(final_label) if final_label else idx == 0,
        })
    if not any(view["canonical"] for view in views):
        views[0]["canonical"] = True
    return views


def set_viewer_camera_pose(eye, lookat):
    import omnigibson as og
    import omnigibson.utils.transform_utils as T

    cam_pos = [float(v) for v in eye]
    center = [float(v) for v in lookat]
    d = np.asarray([center[i] - cam_pos[i] for i in range(3)], dtype=np.float32)
    d /= max(1e-6, np.linalg.norm(d))
    cam_quat = T.euler2quat(th.tensor(
        [math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))), 0.0, float(np.arctan2(-d[0], d[1]))],
        dtype=th.float32,
    ))
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat.tolist())


def record_frame(vw):
    import omnigibson as og
    try:
        import av
        viewer_rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        viewer_np = viewer_rgb[..., :3].cpu().numpy().astype(np.uint8)
        vh, vw_px = vw["viewer_hw"]

        wrist_np = None
        if vw["wrist"] and vw["wrist_hw"][0] > 0:
            try:
                wrist_obs = vw["wrist"].get_obs()[0].get("rgb")
                if wrist_obs is not None:
                    wrist_raw = wrist_obs[..., :3].cpu().numpy().astype(np.uint8)
                    from PIL import Image
                    wrist_img = Image.fromarray(wrist_raw)
                    scale = vh / wrist_raw.shape[0]
                    new_w = int(wrist_raw.shape[1] * scale)
                    wrist_np = np.array(wrist_img.resize((new_w, vh), Image.BILINEAR))
            except Exception:
                pass

        if wrist_np is not None:
            composite = np.concatenate([viewer_np, wrist_np], axis=1)
        else:
            composite = viewer_np

        frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
        for packet in vw["stream"].encode(frame):
            vw["container"].mux(packet)
    except Exception:
        pass


def render_recorded_frame(vw, eye=None, lookat=None):
    import omnigibson as og

    if eye is not None and lookat is not None:
        set_viewer_camera_pose(eye, lookat)
        # The viewer camera can lag one render behind pose updates; flush twice so
        # each candidate writer captures its own intended viewpoint consistently.
        og.sim.render()
        og.sim.render()
    record_frame(vw)


def close_video_writer(vw):
    try:
        for packet in vw["stream"].encode():
            vw["container"].mux(packet)
        vw["container"].close()
    except Exception:
        pass


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
        args._resolved_video_views = tuple(
            {
                "label": view["label"],
                "eye": tuple(float(v) for v in view["eye"]),
                "lookat": tuple(float(v) for v in view["lookat"]),
                "canonical": bool(view["canonical"]),
            }
            for view in video_views
        )
        set_viewer_camera_pose(video_views[0]["eye"], video_views[0]["lookat"])
        for _ in range(3):
            og.sim.step()
        writer_robot = None if getattr(args, "video_viewer_only", False) else robot
        mode = "viewer_only" if writer_robot is None else "viewer_plus_wrist"
        use_labeled_outputs = len(video_views) > 1 or bool(getattr(args, "video_candidate_views", ()))
        for view in video_views:
            if use_labeled_outputs:
                stem = args.save_video[:-4] if args.save_video.endswith(".mp4") else args.save_video
                base_path = f"{stem}_{view['label']}.mp4"
            else:
                base_path = args.save_video
            video_path = expected_video_path(base_path, episode)
            print(
                f"[Pipeline] Video output: {video_path} ({mode}, label={view['label']}, canonical={view['canonical']})"
            )
            writer = init_video_writer(base_path, episode, args.video_fps, robot=writer_robot)
            if writer is None:
                raise RuntimeError(
                    f"Failed to initialize video writer for {video_path}. "
                    "Ensure PyAV is installed and the viewer camera can render frames."
                )
            video_writers.append({"view": view, "writer": writer, "path": video_path})

        # Prime each fixed candidate pose before recording starts so the
        # first encoded frames do not capture stale viewer state.
        for writer_info in video_writers:
            set_viewer_camera_pose(writer_info["view"]["eye"], writer_info["view"]["lookat"])
            og.sim.render()
            og.sim.render()
        if video_writers:
            set_viewer_camera_pose(video_writers[0]["view"]["eye"], video_writers[0]["view"]["lookat"])
            og.sim.render()
            og.sim.render()

    executed = 0
    for _ in range(args.steps):
        action = rng.normal(0.0, args.jitter_scale,
                            size=robot.action_space.shape).astype(np.float32)
        if hasattr(robot.action_space, "low"):
            action = np.clip(action, robot.action_space.low, robot.action_space.high)
        env._pre_step(action)
        og.sim.step()
        executed += 1
        for writer_info in video_writers:
            render_recorded_frame(
                writer_info["writer"],
                eye=writer_info["view"]["eye"],
                lookat=writer_info["view"]["lookat"],
            )
        ltl_monitor.step(executed)
        if executed % 50 == 0:
            print(f"[Pipeline] Step {executed}/{args.steps}")

    for writer_info in video_writers:
        close_video_writer(writer_info["writer"])

    canonical_writer = next((w for w in video_writers if w["view"]["canonical"]), None)
    if canonical_writer is not None:
        canonical_path = expected_video_path(args.save_video, episode)
        if canonical_writer["path"] != canonical_path:
            shutil.copyfile(canonical_writer["path"], canonical_path)
            print(f"[Pipeline] Canonical video selected: {canonical_path} <- {canonical_writer['path']}")

    summary = ltl_monitor.summary()
    print(f"[Pipeline] Episode done: steps={executed}, violated={summary['violated']}")
    return summary, executed


# ---------------------------------------------------------------------------
# Run directory setup
# ---------------------------------------------------------------------------

def setup_run_dir(args):
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = args.scene_model
        args.run_dir = os.path.join(_DEFAULT_RUNS_DIR, f"{label}_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.debug_jsonl is None:
        args.debug_jsonl = os.path.join(args.run_dir, "diagnostics.jsonl")
    if args.save_video is True:
        args.save_video = os.path.join(args.run_dir, "rollout.mp4")
    elif args.save_video is False:
        args.save_video = None
    print(f"[Pipeline] Run directory: {args.run_dir}")


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
        f"viewer_only={entry.video_viewer_only}, "
        f"candidate_mode={entry.video_candidate_mode}, "
        f"eye={entry.video_camera_eye}, "
        f"lookat={entry.video_camera_lookat}, "
        f"candidate_views={len(entry.video_candidate_views)}, "
        f"final_view={entry.video_final_view}"
    )
    return entry


def pipeline_exit(code=0):
    """Clean exit to avoid Isaac Sim shutdown segfault."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# ---------------------------------------------------------------------------
# Surface discovery (shared across table-based pipelines)
# ---------------------------------------------------------------------------

SURFACE_CATEGORY_PRIORITY = {
    "breakfast_table": 3, "dining_table": 3, "conference_table": 3,
    "commercial_kitchen_table": 3, "lab_table": 3,
    "coffee_table": 2, "garden_coffee_table": 2, "pedestal_table": 2,
    "pool_table": 2, "flat_bench": 2,
    "desk": 1, "reception_desk": 1, "counter": 1, "countertop": 1,
    "checkout_counter": 1, "console_table": 1, "nightstand": 1,
}


def discover_surface_from_scene_json(scene_json_path):
    """Find (category, room) of the best table-like surface from scene JSON."""
    from omnigibson.utils.surface_discovery import is_table_like
    return discover_from_scene_json(scene_json_path, is_table_like, SURFACE_CATEGORY_PRIORITY)


def discover_best_surface(env, forced_name=None, forced_category=None):
    """Find the best table-like surface in a loaded scene (sim-dependent)."""
    from omnigibson.utils.surface_discovery import analyze_surface, is_table_like

    scene_data, obj_map = [], {}
    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        cat = str(getattr(obj, "category", ""))
        try:
            aabb_min, aabb_max = obj.aabb
        except Exception:
            continue
        scene_data.append({
            "name": name, "category": cat,
            "aabb_xy": ((float(aabb_min[0]), float(aabb_min[1])),
                        (float(aabb_max[0]), float(aabb_max[1]))),
            "top_z": float(aabb_max[2]),
            "bottom_z": float(aabb_min[2]),
        })
        obj_map[name] = obj

    best_analysis, best_obj = None, None
    for data in scene_data:
        if not is_table_like(data["category"]):
            continue
        if forced_name and data["name"] != forced_name:
            continue
        if forced_category and data["category"] != forced_category:
            continue
        other_aabbs = [
            d["aabb_xy"] for d in scene_data
            if d["name"] != data["name"]
            and d["top_z"] >= 0.15
            and d.get("bottom_z", 0) <= data["top_z"] + 0.3
        ]
        analysis = analyze_surface(
            data["name"], data["category"], data["aabb_xy"], data["top_z"],
            scene_data, scene_object_aabbs=other_aabbs,
        )
        if forced_name or forced_category:
            return analysis, obj_map[data["name"]]
        if analysis.surface.score <= 0:
            continue
        if best_analysis is None or analysis.surface.score > best_analysis.surface.score:
            best_analysis, best_obj = analysis, obj_map[data["name"]]

    if best_analysis is None:
        if forced_name or forced_category:
            detail = forced_name or forced_category
            raise RuntimeError(f"Forced surface '{detail}' not found in scene.")
        raise RuntimeError("No suitable table-like surface found in scene.")
    return best_analysis, best_obj


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
        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"
        curation = getattr(args, "_scene_curation", None)

        support_synset, support_room = "breakfast_table.n.01", "living_room"
        try:
            scene_json = get_scene_json_path(args.scene_model)
            discovery = discover_surface_from_scene_json(scene_json)
            if discovery:
                support_synset = resolve_synset(discovery[0])
                support_room = discovery[1]
                print(f"[Pipeline] Discovered: {discovery[0]} in {support_room}")
        except Exception as e:
            print(f"[Pipeline] Surface discovery failed: {e}")
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

        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"
        curation = getattr(args, "_scene_curation", None)

        # -- Resolve support surface ----------------------------------------
        scene_json = get_scene_json_path(args.scene_model)
        if not os.path.isfile(scene_json):
            raise RuntimeError(f"Scene JSON not found: {scene_json}")
        discovery = discover_surface_from_scene_json(scene_json)
        if discovery is None:
            raise RuntimeError(f"No table-like surface in scene '{args.scene_model}'.")
        surface_category = discovery[0]
        support_synset = resolve_synset(surface_category)
        support_room = discovery[1]
        if curation and curation.support_category:
            surface_category = curation.support_category
            support_synset = resolve_synset(surface_category)
        if curation and curation.support_room:
            support_room = curation.support_room
        print(f"[Pipeline] Discovered: category={surface_category} "
              f"synset={support_synset} room={support_room}")

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
        cfg["task"]["online_object_sampling"] = True
        cfg["task"]["use_presampled_robot_pose"] = False
        if curation and getattr(curation, "surface_name", None):
            cfg["task"]["inroom_object_name_whitelist"] = {
                f"{support_synset}_1": [curation.surface_name],
            }

        self.configure_task(cfg, selection)

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

        # -- Surface discovery ----------------------------------------------
        forced_name = getattr(ctx.curation, "surface_name", None) if ctx.curation else None
        forced_category = getattr(ctx.curation, "support_category", None) if ctx.curation else None
        surface_info, support_obj = discover_best_surface(
            env, forced_name=forced_name, forced_category=forced_category,
        )
        ctx.surface_info = surface_info
        ctx.support_obj = support_obj
        ctx.surface_name = surface_info.surface.name
        print(f"[Pipeline] Best surface: {surface_info.surface.name} "
              f"(score={surface_info.surface.score:.3f})")
        # Pin support first so it cannot move, then clear and compute geometry once.
        if pin_support_object_to_world(support_obj):
            print(f"[Pipeline] Pinned support to world: {support_obj.name}")
        og.sim.step()

        aabb_min, aabb_max = support_obj.aabb
        ctx.surface_bounds_xy = (
            (float(aabb_min[0]), float(aabb_min[1])),
            (float(aabb_max[0]), float(aabb_max[1])),
        )
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

        edge_request = EdgeAlignRequest(
            table_aabb_xy=zone.surface_bounds,
            pack_objects_world=tuple(pack_objects_world),
            role_weights=DEFAULT_ROLE_WEIGHTS,
            robot_half_extent_xy=robot_half_extent_xy(ctx.robot),
            edge_gap_m=args.mount_gap_m, edge_margin_m=0.05,
            scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10,
                            0.15, -0.15, 0.20, -0.20),
            preferred_edge=preferred_edge,
            anchor_offset_m=getattr(args, "mount_anchor_offset_m", 0.0) or 0.0,
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
              f"gap={ctx.edge_result.gap_actual:.3f}")

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
