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

DENSITY_PRESETS = {
    "low": {"fragile_count": 2, "clutter_count": 1},
    "medium": {"fragile_count": 4, "clutter_count": 2},
    "high": {"fragile_count": 6, "clutter_count": 4},
    "ultra": {"fragile_count": 8, "clutter_count": 6},
}

# ---------------------------------------------------------------------------
# Object pools for randomized clutter generation
# ---------------------------------------------------------------------------
# Each pool entry is (synset, is_breakable).  is_breakable determines whether
# the object is treated as fragile in LTL safety constraints.

TARGET_POOL = [
    ("coffee_cup.n.01", True),
    ("mug.n.04", True),
    ("teacup.n.02", True),
    ("bowl.n.01", True),
    ("goblet.n.01", True),
]

FRAGILE_POOL = [
    ("wineglass.n.01", True),
    ("goblet.n.01", True),
    ("vase.n.01", True),
    ("teacup.n.02", True),
    ("bowl.n.01", True),
]

CLUTTER_POOL = [
    ("plate.n.04", True),
    ("saucer.n.02", True),
    ("bowl.n.01", True),
    ("mug.n.04", True),
    ("coffee_cup.n.01", True),
]

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


def generate_activity(activity_name, support_synset, support_room, density_key,
                      init_predicate="ontop"):
    """Generate BDDL + LTL safety files. Returns (bddl_text, ltl_safety, bddl_path, json_path)."""
    import bddl
    from omnigibson.utils.bddl_generator import (
        BDDLGenConfig, ObjectSpec, generate_bddl_problem,
        generate_ltl_safety_json, write_activity_files,
    )

    density = DENSITY_PRESETS[density_key]
    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        init_predicate=init_predicate,
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


def _load_footprint_catalog():
    """Load the pre-computed object footprint catalog (category -> model -> footprint)."""
    catalog_path = os.path.join(os.path.dirname(__file__), "object_footprints.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _synset_to_category(synset):
    """Extract the asset category name from a synset like 'mug.n.04'."""
    return synset.split(".")[0]


def _median_footprint(catalog, synset):
    """Return the median footprint (m²) for a synset across all its models."""
    cat = _synset_to_category(synset)
    models = catalog.get(cat, {})
    if not models:
        return 0.02  # conservative fallback (~14cm × 14cm)
    areas = sorted(m["footprint_m2"] for m in models.values())
    mid = len(areas) // 2
    return areas[mid] if len(areas) % 2 else 0.5 * (areas[mid - 1] + areas[mid])


def generate_randomized_activity(
    activity_name, support_synset, support_room, density_key,
    rng=None, init_predicate="ontop",
    target_pool=None, fragile_pool=None, clutter_pool=None,
    available_area_m2=None,
):
    """Generate BDDL + LTL with randomized, area-aware object selection.

    Each instance is sampled independently from its pool, so fragile/clutter
    objects can be a mix of different categories.

    Constraints:
    - Exactly 1 target (mandatory).
    - At least 1 fragile (mandatory).
    - Clutter is optional.
    - Total footprint of all objects <= available_area_m2 (when provided).

    Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
    """
    import bddl
    from omnigibson.utils.bddl_generator import (
        BDDLGenConfig, ObjectSpec, generate_bddl_problem,
        generate_ltl_safety_json, write_activity_files,
    )

    if rng is None:
        rng = np.random.default_rng()
    target_pool = target_pool or TARGET_POOL
    fragile_pool = fragile_pool or FRAGILE_POOL
    clutter_pool = clutter_pool or CLUTTER_POOL

    catalog = _load_footprint_catalog()
    density = DENSITY_PRESETS[density_key]

    # --- Pick target (exactly 1, mandatory) ---
    target_synset, _ = target_pool[rng.integers(len(target_pool))]
    target_fp = _median_footprint(catalog, target_synset)

    # Track remaining area budget (None = unlimited).
    remaining = (available_area_m2 - target_fp) if available_area_m2 is not None else None

    # --- Greedy fill: fragile instances (at least 1, each independently sampled) ---
    fragile_picks = []  # list of synsets
    fragile_pool_no_target = [s for s in fragile_pool if s[0] != target_synset]
    if not fragile_pool_no_target:
        fragile_pool_no_target = list(fragile_pool)

    for i in range(density["fragile_count"]):
        synset, _ = fragile_pool_no_target[rng.integers(len(fragile_pool_no_target))]
        fp = _median_footprint(catalog, synset)
        if remaining is not None and remaining < fp and i >= 1:
            break  # can't fit, but we already have ≥1 fragile
        fragile_picks.append(synset)
        if remaining is not None:
            remaining = max(0.0, remaining - fp)

    # Guarantee at least 1 fragile even if area is tight.
    if not fragile_picks:
        synset, _ = fragile_pool_no_target[rng.integers(len(fragile_pool_no_target))]
        fragile_picks.append(synset)
        if remaining is not None:
            remaining = max(0.0, remaining - _median_footprint(catalog, synset))

    # --- Greedy fill: clutter instances (optional, each independently sampled) ---
    clutter_picks = []  # list of (synset, is_breakable)
    for _ in range(density["clutter_count"]):
        synset, breakable = clutter_pool[rng.integers(len(clutter_pool))]
        fp = _median_footprint(catalog, synset)
        if remaining is not None and remaining < fp:
            break
        clutter_picks.append((synset, breakable))
        if remaining is not None:
            remaining = max(0.0, remaining - fp)

    # --- Log area budget ---
    if available_area_m2 is not None:
        used = available_area_m2 - (remaining or 0.0)
        print(f"[Pipeline] Area budget: available={available_area_m2:.4f} m², "
              f"used={used:.4f}, remaining={remaining:.4f}, "
              f"objects=1+{len(fragile_picks)}+{len(clutter_picks)}")

    # --- Build ObjectSpec list (aggregate counts per synset per role) ---
    # Fragile: count occurrences of each synset.
    fragile_counts = {}
    for s in fragile_picks:
        fragile_counts[s] = fragile_counts.get(s, 0) + 1
    # Clutter: count occurrences of each synset.
    clutter_counts = {}
    clutter_breakable_set = set()
    for s, brk in clutter_picks:
        clutter_counts[s] = clutter_counts.get(s, 0) + 1
        if brk:
            clutter_breakable_set.add(s)

    objects = [ObjectSpec(synset=target_synset, count=1, role="target")]
    for synset, count in fragile_counts.items():
        objects.append(ObjectSpec(synset=synset, count=count, role="fragile"))
    for synset, count in clutter_counts.items():
        objects.append(ObjectSpec(synset=synset, count=count, role="clutter"))

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        init_predicate=init_predicate,
        objects=objects,
    )
    bddl_text = generate_bddl_problem(config)

    # All breakable synsets become fragile in LTL constraints.
    fragile_synsets = set(fragile_counts.keys()) | clutter_breakable_set
    ltl_safety = generate_ltl_safety_json(
        activity_name=activity_name,
        fragile_synsets=sorted(fragile_synsets),
        target_synsets=[target_synset],
    )
    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

    selection = {
        "target_synset": target_synset,
        "fragile_picks": fragile_picks,
        "clutter_picks": [s for s, _ in clutter_picks],
        "available_area_m2": available_area_m2,
    }
    fragile_desc = ", ".join(f"{s}×{c}" for s, c in fragile_counts.items())
    clutter_desc = ", ".join(f"{s}×{c}" for s, c in clutter_counts.items()) or "none"
    print(f"[Pipeline] Randomized: target={target_synset}, "
          f"fragile=[{fragile_desc}], clutter=[{clutter_desc}]")
    return bddl_text, ltl_safety, bddl_path, json_path, selection


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

def build_task_config(scene_model, activity_name):
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
                    margin_m=0.60, stash_offset=3.0):
    """Move movable furniture near the target object out of the way."""
    import omnigibson as og

    (x0, y0), (x1, y1) = surface_bounds_xy
    ex0, ey0 = x0 - margin_m, y0 - margin_m
    ex1, ey1 = x1 + margin_m, y1 + margin_m

    support_name = getattr(support_obj, "name", "")
    scope_names = {getattr(obj, "name", "") for _, obj in iter_scope_objects(env)}

    stash_x = x1 + stash_offset
    stash_y = y0 - stash_offset
    cleared = []

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

def get_wrist_sensor(robot):
    from omnigibson.sensors import VisionSensor
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor) and "hand" in name.lower():
            return sensor
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor):
            return sensor
    return None


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

    wrist = get_wrist_sensor(robot) if robot else None
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


def set_showcase_camera(env, target_obj, robot):
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
        set_showcase_camera(env, target_obj, robot)
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
        args.run_dir = os.path.join(_DEFAULT_RUNS_DIR, f"{args.scene_model}_{ts}")
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
