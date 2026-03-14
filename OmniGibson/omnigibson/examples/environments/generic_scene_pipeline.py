"""Generic scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json,
packs clutter objects, places robot, and runs LTL-monitored rollouts.

Usage:
    python generic_scene_pipeline.py --scene-model Benevolence_1_int --dry-run
    python generic_scene_pipeline.py --scene-model Benevolence_1_int --episodes 1 --steps 300
"""

import argparse
from datetime import datetime
import json
import math
import os

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")

_DENSITY_PRESETS = {
    "low": {"fragile_count": 2, "clutter_count": 1},
    "medium": {"fragile_count": 4, "clutter_count": 2},
    "high": {"fragile_count": 6, "clutter_count": 4},
    "ultra": {"fragile_count": 8, "clutter_count": 6},
}

_SURFACE_CATEGORY_PRIORITY = {
    "breakfast_table": 3, "dining_table": 3, "coffee_table": 2,
    "desk": 1, "counter": 1, "countertop": 1,
}


def parse_args():
    p = argparse.ArgumentParser(description="Generic scene generation pipeline")
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
    p.add_argument("--clutter-density", default="medium", choices=list(_DENSITY_PRESETS))
    p.add_argument("--pack-jitter-xy", type=float, default=None)
    p.add_argument("--pack-min-clearance", type=float, default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _strip_room_suffix(room: str) -> str:
    if room and room[-1].isdigit() and "_" in room:
        room = "_".join(room.rsplit("_", 1)[:-1])
    return room


def _discover_surface_from_scene_json(scene_json_path: str):
    """Find (category, room) of best table-like surface from scene JSON. No sim needed."""
    from omnigibson.utils.surface_discovery import is_table_like

    with open(scene_json_path, "r", encoding="utf-8") as f:
        init_infos = json.load(f).get("objects_info", {}).get("init_info", {})

    candidates = []
    for info in init_infos.values():
        args = info.get("args", {})
        cat = args.get("category", "")
        if not is_table_like(cat):
            continue
        rooms = args.get("in_rooms", [])
        room = _strip_room_suffix(rooms[0]) if rooms else "living_room"
        candidates.append((cat, room))

    if not candidates:
        return None
    candidates.sort(key=lambda c: _SURFACE_CATEGORY_PRIORITY.get(c[0], 0), reverse=True)
    return candidates[0]


def _generate_activity(activity_name, support_synset, support_room, density_key):
    """Generate BDDL + LTL safety files and return (bddl_text, ltl_safety, bddl_path, json_path)."""
    import bddl
    from omnigibson.utils.bddl_generator import (
        BDDLGenConfig, ObjectSpec, generate_bddl_problem,
        generate_ltl_safety_json, write_activity_files,
    )

    density = _DENSITY_PRESETS[density_key]
    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        objects=[
            ObjectSpec(synset="coffee_cup.n.01", count=1, role="target"),
            ObjectSpec(synset="wineglass.n.01", count=density["fragile_count"], role="fragile"),
            ObjectSpec(synset="plate.n.04", count=density["clutter_count"], role="clutter"),
        ],
    )
    bddl_text = generate_bddl_problem(config)
    ltl_safety = generate_ltl_safety_json(
        activity_name=activity_name,
        fragile_synsets=["wineglass.n.01", "plate.n.04"],
        target_synsets=["coffee_cup.n.01"],
    )
    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)
    return bddl_text, ltl_safety, bddl_path, json_path


def _get_taxonomy():
    from bddl.object_taxonomy import ObjectTaxonomy
    if not hasattr(_get_taxonomy, "_cache"):
        _get_taxonomy._cache = ObjectTaxonomy()
    return _get_taxonomy._cache


def _resolve_synset(category):
    try:
        return _get_taxonomy().get_synset_from_category(category)
    except Exception:
        return f"{category}.n.01"


def _refresh_activity_cache():
    import bddl
    from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES
    refreshed = sorted(os.listdir(
        os.path.join(os.path.dirname(bddl.__file__), "activity_definitions")
    ))
    BEHAVIOR_ACTIVITIES.clear()
    BEHAVIOR_ACTIVITIES.extend(refreshed)


# ---------------------------------------------------------------------------
# Sim-dependent helpers
# ---------------------------------------------------------------------------

def _build_task_config(scene_model, activity_name):
    return {
        "scene": {"type": "InteractiveTraversableScene", "scene_model": scene_model},
        "task": {
            "type": "BehaviorTask", "activity_name": activity_name,
            "activity_definition_id": 0, "activity_conditions_met": False,
            "online_object_sampling": True,
        },
        "robots": [{
            "type": "FrankaMounted", "obs_modalities": ["rgb"],
            "action_type": "continuous", "action_normalize": True,
            "controller_config": {
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
    }


def _discover_best_surface(env):
    """Find best table-like surface in loaded scene, considering robot reachability."""
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
        # Blocker AABBs: skip ground-level and overhead objects.
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
        raise RuntimeError(f"No suitable table-like surface found in scene.")
    return best_analysis, best_obj


def _iter_scope_objects(env):
    for inst, ent in (getattr(env.task, "object_scope", {}) or {}).items():
        if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
            continue
        obj = getattr(ent, "wrapped_obj", None)
        if obj is not None:
            yield inst, obj


def _get_scope_obj(env, inst):
    ent = (getattr(env.task, "object_scope", {}) or {}).get(inst)
    if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
        return None
    return getattr(ent, "wrapped_obj", None)


# Categories of movable furniture that can block robot placement near a table.
_CLEARABLE_CATEGORIES = {
    "chair", "straight_chair", "armchair", "swivel_chair", "folding_chair",
    "highchair", "rocking_chair", "barber_chair", "wheelchair",
    "stool", "bar_stool", "bench", "ottoman", "hassock",
    "pot_plant", "plant", "stand", "pedestal", "trash_can", "wastebasket",
    "side_table", "end_table", "coffee_table", "tray",
}


def _is_clearable(category):
    """Check if a category is movable furniture we should clear from the perimeter."""
    cat = category.lower()
    if cat in _CLEARABLE_CATEGORIES:
        return True
    # Catch variants like "straight_chair_abc123" via substring.
    for prefix in ("chair", "stool", "bench", "ottoman", "hassock"):
        if prefix in cat:
            return True
    return False


def _clear_perimeter(env, support_obj, surface_bounds_xy, table_top_z, floor_z,
                     margin_m=0.60, stash_offset=3.0):
    """Move movable furniture (chairs, stools, etc.) near the table out of the way.

    Only clears objects whose category matches _CLEARABLE_CATEGORIES.
    Walls, floors, large furniture, and structural elements are never moved.
    """
    import omnigibson as og

    (x0, y0), (x1, y1) = surface_bounds_xy
    ex0, ey0 = x0 - margin_m, y0 - margin_m
    ex1, ey1 = x1 + margin_m, y1 + margin_m

    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for _, obj in _iter_scope_objects(env)}

    stash_x = x1 + stash_offset
    stash_y = y0 - stash_offset
    cleared = []

    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        cat = str(getattr(obj, "category", ""))

        if name == support_name or name in scope_names:
            continue
        if not _is_clearable(cat):
            continue

        try:
            obj_min, obj_max = obj.aabb
            ox0, oy0 = float(obj_min[0]), float(obj_min[1])
            ox1, oy1 = float(obj_max[0]), float(obj_max[1])
        except Exception:
            continue

        # Skip objects outside the expanded perimeter.
        if ox1 < ex0 or ox0 > ex1 or oy1 < ey0 or oy0 > ey1:
            continue

        sx = stash_x + 0.3 * len(cleared)
        try:
            obj.set_position_orientation(
                position=(sx, stash_y, floor_z + 0.05),
                orientation=(0, 0, 0, 1),
            )
            if hasattr(obj, "keep_still"):
                obj.keep_still()
            cleared.append(name)
        except Exception:
            pass

    if cleared:
        og.sim.step()
        print(f"[Pipeline] Cleared {len(cleared)} perimeter objects: {cleared}")
    return cleared


def _compute_floor_z(env):
    floor_z = 0.0
    for inst, obj in _iter_scope_objects(env):
        if inst.startswith("floor."):
            try:
                floor_z = max(floor_z, float(obj.aabb[1][2]))
            except Exception:
                pass
    return floor_z


def _build_task_object_sets(env, task_spec):
    available = {inst for inst, _ in _iter_scope_objects(env)}
    target_ids = [i for i in task_spec.target_ids if i in available]
    fragile_ids = [i for i in task_spec.fragile_ids if i in available and i not in target_ids]
    support_ids = [i for i in task_spec.support_ids if i in available]
    assigned = set(target_ids) | set(fragile_ids) | set(support_ids)
    clutter_ids = sorted(
        i for i in available
        if i not in assigned and not i.startswith(("agent.", "floor."))
    )
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


def _build_descriptors(env, obj_sets):
    from omnigibson.utils.clutter_pack_layout import ClutterObjectDescriptor

    descriptors, objects_by_inst = [], {}
    for role, id_key in [("target", "target_ids"), ("fragile", "fragile_ids"), ("clutter", "clutter_ids")]:
        for inst in obj_sets[id_key]:
            obj = _get_scope_obj(env, inst)
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


def _robot_half_extent_xy(robot):
    for key in ("base_link", "base", "base_footprint", "chassis"):
        link = (getattr(robot, "links", {}) or {}).get(key)
        if link is not None:
            try:
                mn, mx = link.aabb
                return ((float(mx[0] - mn[0])) * 0.5, (float(mx[1] - mn[1])) * 0.5)
            except Exception:
                pass
    return (0.15, 0.15)


# -- Pack callback factories ------------------------------------------------

def _make_settle_fn(og_mod, th_mod):
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


def _make_park_fn(og_mod, zone_surface_bounds, floor_z):
    def park(passive_objs):
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


def _validate_poses(objs):
    invalid = []
    for inst, obj in objs.items():
        try:
            pos = obj.get_position_orientation()[0]
            if not all(math.isfinite(float(pos[i])) for i in range(3)):
                invalid.append(inst)
        except Exception:
            invalid.append(inst)
    return invalid


def _check_interpenetration(objs, tol):
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


# -- Video helpers -----------------------------------------------------------

def _get_wrist_sensor(robot):
    """Find the wrist/hand camera sensor on the robot."""
    from omnigibson.sensors import VisionSensor
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor) and "hand" in name.lower():
            return sensor
    # Fallback: return the first VisionSensor found.
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor):
            return sensor
    return None


def _init_video_writer(base_path, episode, fps, robot=None):
    try:
        import av
    except ImportError:
        print("[Pipeline] WARNING: PyAV not available — video recording disabled.")
        return None
    import omnigibson as og

    # Probe viewer resolution.
    try:
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        vh, vw = int(rgb.shape[0]), int(rgb.shape[1])
    except Exception:
        vh, vw = 720, 1280

    # Probe wrist camera resolution.
    wrist = _get_wrist_sensor(robot) if robot else None
    wh, ww = 0, 0
    if wrist:
        try:
            wrist_rgb = wrist.get_obs()[0].get("rgb")
            if wrist_rgb is not None:
                wh, ww = int(wrist_rgb.shape[0]), int(wrist_rgb.shape[1])
        except Exception:
            pass

    # Composite: viewer on left, wrist scaled to same height on right.
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


def _record_frame(vw):
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
                    # Scale wrist image to match viewer height.
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


def _close_video_writer(vw):
    try:
        for packet in vw["stream"].encode():
            vw["container"].mux(packet)
        vw["container"].close()
    except Exception:
        pass


def _set_showcase_camera(env, target_obj, robot):
    """Position viewer camera looking at workspace from a diagonal offset."""
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    import torch as th

    rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
    tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
    center = [0.5 * (rp[0] + tp[0]), 0.5 * (rp[1] + tp[1]), max(rp[2] + 0.7, tp[2] + 0.25)]
    cam_pos = [center[0] - 1.0, center[1] - 1.1, center[2] + 0.5]
    d = np.asarray([center[i] - cam_pos[i] for i in range(3)], dtype=np.float32)
    d /= max(1e-6, np.linalg.norm(d))
    cam_quat = T.euler2quat(th.tensor(
        [math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))),
         0.0,
         float(np.arctan2(-d[0], d[1]))],
        dtype=th.float32,
    ))
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat.tolist())
    og.sim.enable_viewer_camera_teleoperation()


# ---------------------------------------------------------------------------
# GPU dynamics detection
# ---------------------------------------------------------------------------

def _needs_gpu_dynamics(activity_name):
    try:
        from bddl.activity import Conditions
        taxonomy = _get_taxonomy()
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


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    """Generate BDDL + ltl_safety.json without starting the simulator."""
    activity_name = args.activity_name or f"auto_clutter_on_{args.scene_model}"

    # Try JSON-based surface discovery if scene exists.
    support_synset, support_room = "breakfast_table.n.01", "living_room"
    try:
        from omnigibson.utils.asset_utils import get_scene_path
        scene_json = os.path.join(
            get_scene_path(args.scene_model), "json", f"{args.scene_model}_best.json",
        )
        discovery = _discover_surface_from_scene_json(scene_json)
        if discovery:
            support_synset = _resolve_synset(discovery[0])
            support_room = discovery[1]
            print(f"[Pipeline] Discovered: {discovery[0]} in {support_room}")
    except Exception:
        pass

    bddl_text, ltl_safety, bddl_path, json_path = _generate_activity(
        activity_name, support_synset, support_room, args.clutter_density,
    )
    print(f"[Pipeline] Dry-run complete:")
    print(f"  BDDL:       {bddl_path}")
    print(f"  ltl_safety: {json_path}")
    print(f"  activity:   {activity_name}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    _append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "scene_model": args.scene_model, "density": args.clutter_density,
    })
    return activity_name, bddl_path, json_path


def run_sim(args, activity_name=None):
    """Full sim-validation path: surface discovery, pack, robot, gate, LTL."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import get_scene_path
    from omnigibson.utils.clutter_pack_layout import validate_pack_integrity
    from omnigibson.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS, EdgeAlignObject, EdgeAlignRequest, place_franka_edge_aligned,
    )
    from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone
    from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec
    from omnigibson.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop
    from omnigibson.utils.safety_monitor import TaskLTLMonitor

    gm.ENABLE_OBJECT_STATES = True

    if activity_name is None:
        activity_name = args.activity_name or f"auto_clutter_on_{args.scene_model}"

    # -- Discover surface from scene JSON -----------------------------------
    scene_json = os.path.join(
        get_scene_path(args.scene_model), "json", f"{args.scene_model}_best.json",
    )
    if not os.path.isfile(scene_json):
        raise RuntimeError(f"Scene JSON not found: {scene_json}")

    discovery = _discover_surface_from_scene_json(scene_json)
    if discovery is None:
        raise RuntimeError(f"No table-like surface in scene '{args.scene_model}'.")

    support_synset = _resolve_synset(discovery[0])
    support_room = discovery[1]
    print(f"[Pipeline] Discovered: category={discovery[0]} synset={support_synset} room={support_room}")

    # -- Generate BDDL ------------------------------------------------------
    _, _, bddl_path, _ = _generate_activity(
        activity_name, support_synset, support_room, args.clutter_density,
    )
    print(f"[Pipeline] Generated BDDL: {bddl_path}")
    _refresh_activity_cache()

    # -- GPU dynamics --------------------------------------------------------
    gpu = _needs_gpu_dynamics(activity_name)
    gm.USE_GPU_DYNAMICS = gpu
    gm.ENABLE_FLATCACHE = not gpu

    # -- Load environment ----------------------------------------------------
    cfg = _build_task_config(args.scene_model, activity_name)
    cfg["scene"]["scene_file"] = scene_json
    cfg["scene"]["scene_instance"] = None
    cfg["task"]["online_object_sampling"] = True
    cfg["task"]["use_presampled_robot_pose"] = False

    print(f"[Pipeline] scene={args.scene_model}, activity={activity_name}, strict_gate={args.strict_gate}")
    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
            env.reset()
            og.sim.step()

            # -- Surface discovery (sim) ------------------------------------
            surface_info, support_obj = _discover_best_surface(env)
            print(f"[Pipeline] Best surface: {surface_info.surface.name} "
                  f"(score={surface_info.surface.score:.3f}, area={surface_info.surface.area:.3f})")

            aabb_min, aabb_max = support_obj.aabb
            surface_bounds_xy = (
                (float(aabb_min[0]), float(aabb_min[1])),
                (float(aabb_max[0]), float(aabb_max[1])),
            )
            table_top_z = float(aabb_max[2])

            obstacle_bounds_xy = None
            if surface_info.obstacles:
                obstacle_bounds_xy = surface_info.obstacles[0].aabb_xy

            # -- Zone computation -------------------------------------------
            zone = compute_tabletop_zone(
                surface_bounds_xy=surface_bounds_xy,
                obstacle_bounds_xy=obstacle_bounds_xy,
                edge_margin_m=0.04,
                obstacle_keepout_margin_m=0.08,
                obstacle_side_clearance_m=0.015,
            )
            print(f"[Pipeline] Zone: red_zone={zone.red_zone_bounds}, long_axis={zone.long_axis}")

            # -- Clear perimeter objects (chairs, etc.) ---------------------
            floor_z = _compute_floor_z(env)
            _clear_perimeter(env, support_obj, surface_bounds_xy, table_top_z, floor_z)

            # -- Build object sets ------------------------------------------
            task_spec = build_manipulation_task_spec(activity_name)
            obj_sets = _build_task_object_sets(env, task_spec)

            if not obj_sets["target_ids"]:
                raise RuntimeError("No target objects found.")
            target_obj = _get_scope_obj(env, obj_sets["target_ids"][0])

            descriptors, objects_by_inst = _build_descriptors(env, obj_sets)
            if not descriptors:
                raise RuntimeError("No clutter-pack descriptors created.")

            # -- Pack retry loop --------------------------------------------
            pack_config = PackRetryConfig(
                pack_jitter_xy=args.pack_jitter_xy or 0.022,
                pack_min_clearance=args.pack_min_clearance or 0.008,
            )
            settle_fn = _make_settle_fn(og, th)
            park_fn = _make_park_fn(og, zone.surface_bounds, floor_z)

            pack_result = run_pack_retry_loop(
                support_name=getattr(support_obj, "name", "support"),
                descriptors=descriptors, objects_by_inst=objects_by_inst,
                red_zone_bounds=zone.red_zone_bounds, table_top_z=table_top_z,
                floor_z=floor_z, config=pack_config, base_seed=args.seed, episode=ep,
                settle_fn=settle_fn, park_fn=park_fn,
                validate_poses_fn=_validate_poses,
                check_interpenetration_fn=_check_interpenetration,
                obstacle_keepout_bounds=zone.obstacle_keepout_bounds,
            )
            print(f"[Pipeline] Pack solved: attempt={pack_result.attempt_used}, "
                  f"active={len(pack_result.active_descriptors)}")

            # Park inactive objects.
            passive = {i: o for i, o in objects_by_inst.items()
                       if i not in pack_result.active_objects_by_inst}
            park_fn(passive)

            # -- Integrity check --------------------------------------------
            integrity = validate_pack_integrity(
                pack_spec=pack_result.pack_spec,
                world_positions=pack_result.world_positions,
                pack_origin_world=pack_result.pack_origin,
                pack_yaw=0.0, tol_xy=pack_config.integrity_tol_xy,
            )

            # -- Robot placement --------------------------------------------
            robot = env.robots[0]
            pack_objects_world = tuple(
                EdgeAlignObject(
                    name=e.inst_id, role=e.role,
                    position_xy=(pack_result.world_positions[e.inst_id][0],
                                 pack_result.world_positions[e.inst_id][1]),
                )
                for e in pack_result.pack_spec.object_entries
                if e.inst_id in pack_result.world_positions
            )
            if not pack_objects_world:
                raise RuntimeError("No pack objects for edge alignment.")

            preferred_edge = surface_info.approach_edges[0] if surface_info.approach_edges else None
            edge_result = place_franka_edge_aligned(EdgeAlignRequest(
                table_aabb_xy=zone.surface_bounds,
                pack_objects_world=pack_objects_world,
                role_weights=DEFAULT_ROLE_WEIGHTS,
                robot_half_extent_xy=_robot_half_extent_xy(robot),
                edge_gap_m=args.mount_gap_m, edge_margin_m=0.05,
                scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20),
                preferred_edge=preferred_edge,
            ))
            robot.set_position_orientation(
                position=(edge_result.base_pose["position"][0],
                          edge_result.base_pose["position"][1], floor_z),
                orientation=edge_result.base_pose["orientation"],
            )
            og.sim.step()
            print(f"[Pipeline] Robot: edge={edge_result.edge_label}, gap={edge_result.gap_actual:.3f}")

            # -- Gate -------------------------------------------------------
            rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
            tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
            target_dist = math.hypot(rp[0] - tp[0], rp[1] - tp[1])
            gate_pass = (
                all(math.isfinite(v) for v in rp + tp)
                and abs(rp[2] - floor_z) <= 0.03
                and not edge_result.collision_hits
                and 0.20 <= target_dist <= 1.10
                and integrity.ok
            )
            print(f"[Pipeline] Gate: pass={gate_pass}")
            if args.strict_gate and not gate_pass:
                raise RuntimeError("Strict gate failed.")

            # -- Save scene snapshot ----------------------------------------
            if gate_pass:
                scene_save_path = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
                og.sim.save(json_paths=[scene_save_path])
                print(f"[Pipeline] Scene saved: {scene_save_path}")

            # -- LTL rollout ------------------------------------------------
            ltl_monitor = TaskLTLMonitor(
                env=env, activity_name=activity_name,
                scene_model=args.scene_model,
                active_objects_by_inst=pack_result.active_objects_by_inst,
            )
            ltl_monitor.reset()
            ltl_monitor.step(0)

            video_writer = None
            if args.save_video:
                _set_showcase_camera(env, target_obj, robot)
                for _ in range(3):
                    og.sim.step()
                video_writer = _init_video_writer(args.save_video, ep, args.video_fps, robot=robot)

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
                    _record_frame(video_writer)
                ltl_monitor.step(executed)
                if executed % 50 == 0:
                    print(f"[Pipeline] Step {executed}/{args.steps}")

            if video_writer:
                _close_video_writer(video_writer)

            summary = ltl_monitor.summary()
            print(f"[Pipeline] Episode done: steps={executed}, violated={summary['violated']}")

            _append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "scene_model": args.scene_model,
                "activity_name": activity_name, "surface": surface_info.surface.name,
                "pack_attempt_used": pack_result.attempt_used,
                "gate_pass": gate_pass, "ltl_violated": summary["violated"],
                "steps_executed": executed,
            })

    finally:
        print("[Pipeline] Shutdown simulator.")
        try:
            og.clear()
        except Exception as e:
            print(f"[Pipeline] og.clear warning: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_run_dir(args):
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(_DEFAULT_RUNS_DIR, f"{args.scene_model}_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.debug_jsonl is None:
        args.debug_jsonl = os.path.join(args.run_dir, "diagnostics.jsonl")
    if args.save_video is True:
        args.save_video = os.path.join(args.run_dir, "rollout.mp4")
    elif args.save_video is False:
        args.save_video = None
    print(f"[Pipeline] Run directory: {args.run_dir}")


def main():
    args = parse_args()
    _setup_run_dir(args)
    if args.dry_run:
        run_dry_run(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main()
