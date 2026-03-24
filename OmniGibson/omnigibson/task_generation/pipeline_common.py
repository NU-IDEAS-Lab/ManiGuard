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
from datetime import datetime

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")


# Pool constants and activity generators live in omnigibson.utils.bddl_generator.
# Re-exported here for backward compatibility with pinch_point / cabinet pipelines.
from omnigibson.utils.bddl_generator import DENSITY_PRESETS  # noqa: F401, E402
from omnigibson.utils.bddl_generator import generate_clutter_activity as generate_activity  # noqa: F401, E402

# Categories of movable furniture that can block robot placement.
CLEARABLE_CATEGORIES = {
    "chair", "straight_chair", "armchair", "swivel_chair", "folding_chair",
    "highchair", "rocking_chair", "barber_chair", "wheelchair",
    "stool", "bar_stool", "bench", "ottoman", "hassock",
    "pot_plant", "plant", "stand", "pedestal", "trash_can", "wastebasket",
    "side_table", "end_table", "coffee_table", "tray",
}

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
    return p


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def append_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


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


def compute_floor_z(env):
    floor_z = 0.0
    for inst, obj in iter_scope_objects(env):
        if inst.startswith("floor."):
            try:
                floor_z = max(floor_z, float(obj.aabb[1][2]))
            except Exception:
                pass
    return floor_z


def is_clearable(category):
    cat = category.lower()
    if cat in CLEARABLE_CATEGORIES:
        return True
    for prefix in ("chair", "stool", "bench", "ottoman", "hassock"):
        if prefix in cat:
            return True
    return False


def clear_perimeter(env, support_obj, surface_bounds_xy, top_z, floor_z,
                    margin_m=0.60):
    """Remove movable furniture near the support surface from the scene."""
    import omnigibson as og

    (x0, y0), (x1, y1) = surface_bounds_xy
    ex0, ey0 = x0 - margin_m, y0 - margin_m
    ex1, ey1 = x1 + margin_m, y1 + margin_m

    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for _, obj in iter_scope_objects(env)}

    to_remove = []
    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        cat = str(getattr(obj, "category", ""))
        if name == support_name or name in scope_names:
            continue
        if not is_clearable(cat):
            continue
        try:
            obj_min, obj_max = obj.aabb
            ox0, oy0 = float(obj_min[0]), float(obj_min[1])
            ox1, oy1 = float(obj_max[0]), float(obj_max[1])
        except Exception:
            continue
        if ox1 < ex0 or ox0 > ex1 or oy1 < ey0 or oy0 > ey1:
            continue
        to_remove.append(obj)

    if to_remove:
        names = [getattr(o, "name", "?") for o in to_remove]
        og.sim.batch_remove_objects(to_remove)
        print(f"[Pipeline] Removed {len(to_remove)} perimeter objects: {names}")
    return [getattr(o, "name", "") for o in to_remove]


def build_task_object_sets(env, task_spec):
    available = {inst for inst, _ in iter_scope_objects(env)}
    target_ids = [i for i in task_spec.target_ids if i in available]
    fragile_ids = [i for i in task_spec.fragile_ids if i in available and i not in target_ids]
    support_ids = [i for i in task_spec.support_ids if i in available]
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
                dx = max(0.01, float(aabb_max[0] - aabb_min[0]))
                dy = max(0.01, float(aabb_max[1] - aabb_min[1]))
                dz = max(0.01, float(aabb_max[2] - aabb_min[2]))
            except Exception:
                continue
            descriptors.append(ClutterObjectDescriptor(
                instance_id=inst, role=role,
                half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
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
    if robot:
        from omnigibson.sensors import VisionSensor
        for name, sensor in robot.sensors.items():
            if isinstance(sensor, VisionSensor) and "hand" in name.lower():
                wrist = sensor
                break
        if wrist is None:
            for name, sensor in robot.sensors.items():
                if isinstance(sensor, VisionSensor):
                    wrist = sensor
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

    return {"container": container, "stream": stream, "wrist": wrist,
            "viewer_hw": (vh, vw), "wrist_hw": (wh, ww)}


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


def close_video_writer(vw):
    try:
        for packet in vw["stream"].encode():
            vw["container"].mux(packet)
        vw["container"].close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LTL rollout (shared by all pipelines)
# ---------------------------------------------------------------------------

def run_ltl_rollout(env, activity_name, scene_model, active_objects_by_inst,
                    robot, target_obj, args, episode, rng):
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

    video_writer = None
    if args.save_video:
        # Position viewer camera looking at workspace from a diagonal offset.
        import omnigibson.utils.transform_utils as T
        import torch as th
        rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
        tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
        center = [0.5 * (rp[0] + tp[0]), 0.5 * (rp[1] + tp[1]),
                  max(rp[2] + 0.7, tp[2] + 0.25)]
        cam_pos = [center[0] - 1.0, center[1] - 1.1, center[2] + 0.5]
        d = np.asarray([center[i] - cam_pos[i] for i in range(3)], dtype=np.float32)
        d /= max(1e-6, np.linalg.norm(d))
        cam_quat = T.euler2quat(th.tensor(
            [math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))),
             0.0,
             float(np.arctan2(-d[0], d[1]))],
            dtype=th.float32,
        ))
        og.sim.viewer_camera.set_position_orientation(
            position=cam_pos, orientation=cam_quat.tolist())
        og.sim.enable_viewer_camera_teleoperation()
        for _ in range(3):
            og.sim.step()
        video_writer = init_video_writer(args.save_video, episode, args.video_fps, robot=robot)

    executed = 0
    for _ in range(args.steps):
        action = rng.normal(0.0, args.jitter_scale,
                            size=robot.action_space.shape).astype(np.float32)
        if hasattr(robot.action_space, "low"):
            action = np.clip(action, robot.action_space.low, robot.action_space.high)
        env._pre_step(action)
        og.sim.step()
        executed += 1
        if video_writer:
            record_frame(video_writer)
        ltl_monitor.step(executed)
        if executed % 50 == 0:
            print(f"[Pipeline] Step {executed}/{args.steps}")

    if video_writer:
        close_video_writer(video_writer)

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


def pipeline_exit():
    """Clean exit to avoid Isaac Sim shutdown segfault."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


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


def discover_best_surface(env):
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
        if analysis.surface.score <= 0:
            continue
        if best_analysis is None or analysis.surface.score > best_analysis.surface.score:
            best_analysis, best_obj = analysis, obj_map[data["name"]]

    if best_analysis is None:
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

    # Activity
    activity_name: str = ""
    selection: Dict = field(default_factory=dict)

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
        if args.dry_run:
            self._run_dry_run(args)
        else:
            self._run_sim(args)

    def _run_dry_run(self, args):
        scene_label = args.scene_model
        activity_name = args.activity_name or f"{self.activity_prefix()}_{scene_label}"

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

        self.configure_task(cfg, selection)

        print(f"[Pipeline] scene={scene_label}, activity={activity_name}, "
              f"strict_gate={args.strict_gate}")
        env = og.Environment(configs=cfg)

        try:
            for ep in range(args.episodes):
                ctx = EpisodeContext(
                    env=env, og=og, args=args, rng=rng,
                    activity_name=activity_name,
                    selection=selection, episode=ep,
                )
                print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
                self._run_episode(ctx)

                append_jsonl(args.debug_jsonl, {
                    "episode": ep + 1,
                    "scene_model": scene_label,
                    "activity_name": activity_name,
                    "surface": ctx.surface_name,
                    "gate_pass": ctx.gate_pass,
                    "ltl_violated": ctx.ltl_summary.get("violated") if hasattr(ctx, "ltl_summary") else None,
                    "steps_executed": ctx.steps_executed if hasattr(ctx, "steps_executed") else 0,
                    "selection": selection,
                    **self.diagnostics_extra(ctx),
                })
        finally:
            print("[Pipeline] Shutdown simulator.")
            pipeline_exit()

    def _run_episode(self, ctx):
        env, og, args = ctx.env, ctx.og, ctx.args
        env.reset()
        og.sim.step()

        # -- Surface discovery ----------------------------------------------
        surface_info, support_obj = discover_best_surface(env)
        ctx.surface_info = surface_info
        ctx.support_obj = support_obj
        ctx.surface_name = surface_info.surface.name
        print(f"[Pipeline] Best surface: {surface_info.surface.name} "
              f"(score={surface_info.surface.score:.3f})")
        aabb_min, aabb_max = support_obj.aabb
        ctx.surface_bounds_xy = (
            (float(aabb_min[0]), float(aabb_min[1])),
            (float(aabb_max[0]), float(aabb_max[1])),
        )
        ctx.table_top_z = float(aabb_max[2])
        ctx.floor_z = compute_floor_z(env)
        clear_perimeter(env, support_obj, ctx.surface_bounds_xy,
                        ctx.table_top_z, ctx.floor_z)

        # -- Pipeline-specific: identify & place objects --------------------
        self.identify_objects(ctx)
        self.place_objects(ctx)

        # -- Robot placement ------------------------------------------------
        from omnigibson.utils.franka_edge_align import (
            DEFAULT_ROLE_WEIGHTS, EdgeAlignRequest, place_franka_edge_aligned,
        )
        from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone

        ctx.robot = env.robots[0]

        obstacle_bounds_xy = None
        if ctx.surface_info and ctx.surface_info.obstacles:
            obstacle_bounds_xy = ctx.surface_info.obstacles[0].aabb_xy

        zone = compute_tabletop_zone(
            surface_bounds_xy=ctx.surface_bounds_xy,
            obstacle_bounds_xy=obstacle_bounds_xy,
            edge_margin_m=0.04,
            obstacle_keepout_margin_m=0.08,
            obstacle_side_clearance_m=0.015,
        )

        pack_objects_world = self.make_edge_objects(ctx)

        preferred_edge = None
        if ctx.surface_info and ctx.surface_info.approach_edges:
            preferred_edge = ctx.surface_info.approach_edges[0]

        ctx.edge_result = place_franka_edge_aligned(EdgeAlignRequest(
            table_aabb_xy=zone.surface_bounds,
            pack_objects_world=tuple(pack_objects_world),
            role_weights=DEFAULT_ROLE_WEIGHTS,
            robot_half_extent_xy=robot_half_extent_xy(ctx.robot),
            edge_gap_m=args.mount_gap_m, edge_margin_m=0.05,
            scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10,
                            0.15, -0.15, 0.20, -0.20),
            preferred_edge=preferred_edge,
        ))
        ctx.robot.set_position_orientation(
            position=(ctx.edge_result.base_pose["position"][0],
                      ctx.edge_result.base_pose["position"][1], ctx.floor_z),
            orientation=ctx.edge_result.base_pose["orientation"],
        )
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
        print(f"[Pipeline] Gate: pass={ctx.gate_pass}, dist={target_dist:.3f}")
        if args.strict_gate and not ctx.gate_pass:
            raise RuntimeError("Strict gate failed.")

        # -- Save scene snapshot --------------------------------------------
        if ctx.gate_pass:
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
        )
