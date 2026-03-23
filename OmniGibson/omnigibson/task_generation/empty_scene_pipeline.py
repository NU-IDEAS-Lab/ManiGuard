"""Empty-scene task generation pipeline.

Starts from a bare Scene (floor plane only), spawns a randomized support
surface and task objects via the env config ``objects`` list (following the
grasp_task_demo pattern), then runs the standard clutter / stack / transfer
placement + LTL-monitored rollouts.

Domain randomization: surface category/model, target, fragile, and clutter
types are all randomized per episode from the pools in pipeline_common.

Usage:
    # Clutter on empty scene (random surface + objects)
    python -m omnigibson.task_generation.empty_scene_pipeline \\
        --setup clutter --episodes 1 --steps 300 --save-video

    # Stack on a specific desk
    python -m omnigibson.task_generation.empty_scene_pipeline \\
        --setup stack --surface-category desk --stack-height medium \\
        --episodes 1 --steps 300 --save-video

    # Food transfer
    python -m omnigibson.task_generation.empty_scene_pipeline \\
        --setup transfer --episodes 1 --steps 300 --save-video

    # Dry-run (generate BDDL only, no sim)
    python -m omnigibson.task_generation.empty_scene_pipeline \\
        --setup clutter --dry-run
"""

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    CLUTTER_POOL,
    DENSITY_PRESETS,
    FRAGILE_POOL,
    STACK_HEIGHT_PRESETS,
    STACK_ITEM_POOL,
    STACK_TARGET_POOL,
    TARGET_POOL,
    TRANSFER_DEST_POOL,
    TRANSFER_FOOD_POOL,
    TRANSFER_SOURCE_POOL,
    append_jsonl,
    generate_activity,
    generate_randomized_activity,
    generate_stack_activity,
    generate_transfer_activity,
    make_settle_fn,
    pipeline_exit,
    refresh_activity_cache,
    robot_half_extent_xy,
    run_ltl_rollout,
)

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")

# Surface categories suitable for the empty-scene pipeline.
# Criteria: adequate height for FrankaMounted (>0.5m), reasonable surface area
# for multi-object tasks (>0.4 m²).
# See /tmp/surface_catalog.py for the full catalog with dimensions.
SURFACE_CATEGORY_POOL = [
    "breakfast_table",      # 32 models, avg 1.28 m², avg H 0.64m
    "desk",                 # 44 models, avg 1.31 m², avg H 0.76m
    "conference_table",     #  4 models, avg 5.03 m², avg H 0.72m
    "lab_table",            #  3 models, avg 4.06 m², avg H 1.27m
    "commercial_kitchen_table",  # 5 models, avg 1.92 m², avg H 0.94m
    "pedestal_table",       # 21 models, avg 0.74 m², avg H 0.65m
    "coffee_table",         # 40 models, avg 0.70 m², avg H 0.42m — short but many models
]

# Minimum usable surface area (m²).  Tables smaller than this are skipped
# because most task objects won't fit.
_MIN_SURFACE_AREA_M2 = 0.35


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Empty-scene task generation pipeline")
    p.add_argument("--setup", required=True, choices=["clutter", "stack", "transfer"])
    p.add_argument("--surface-category", default=None,
                   help="Surface category (random from pool if omitted)")
    p.add_argument("--surface-model", default=None,
                   help="Specific model ID (random if omitted)")
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
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    # Clutter.
    p.add_argument("--clutter-density", default="medium", choices=list(DENSITY_PRESETS))
    p.add_argument("--pack-jitter-xy", type=float, default=None)
    p.add_argument("--pack-min-clearance", type=float, default=None)
    # Stack.
    p.add_argument("--stack-height", default="medium", choices=list(STACK_HEIGHT_PRESETS))
    p.add_argument("--target-synset", default=None)
    p.add_argument("--stack-synset", default=None)
    # Transfer.
    p.add_argument("--food-synset", default=None)
    p.add_argument("--source-synset", default=None)
    p.add_argument("--dest-synset", default=None)
    p.add_argument("--goal-predicate", default=None, choices=["inside", "ontop"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synset_to_category(synset):
    return synset.split(".")[0]


def _resolve_synset(category):
    try:
        from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
        return OBJECT_TAXONOMY.get_synset_from_category(category)
    except Exception:
        return f"{category}.n.01"


def _pick_random_model(category, rng):
    from omnigibson.utils.asset_utils import get_all_object_category_models
    models = get_all_object_category_models(category)
    if not models:
        return None
    return models[rng.integers(len(models))]


def _pick_synset_with_model(pool, rng, exclude=None):
    """Pick a (synset, ...) entry from *pool* whose category has models.

    Shuffles the pool and returns the first entry with available assets.
    Returns None if nothing in the pool is spawnable.
    *exclude*: set of synsets to skip (e.g. to avoid duplicating the target).
    """
    from omnigibson.utils.asset_utils import get_all_object_category_models
    exclude = exclude or set()
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    for i in indices:
        entry = pool[i]
        synset = entry[0]
        if synset in exclude:
            continue
        cat = _synset_to_category(synset)
        if get_all_object_category_models(cat):
            return entry
    return None


def _load_surface_catalog():
    """Load the pre-computed surface catalog JSON."""
    import json
    catalog_path = os.path.join(os.path.dirname(__file__), "surface_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_surface(rng, category=None, model=None, min_area=None):
    """Pick a surface category + model, filtering by minimum area.

    If *category* and *model* are both given they are used as-is (no filtering).
    Otherwise candidates are drawn from the catalog and rejected if their
    surface area is below *min_area*.
    """
    if min_area is None:
        min_area = _MIN_SURFACE_AREA_M2

    if category is not None and model is not None:
        synset = _resolve_synset(category)
        return category, model, synset

    catalog = _load_surface_catalog()

    # Build a list of (category, model, area) candidates.
    candidates = []
    cats_to_try = [category] if category else list(SURFACE_CATEGORY_POOL)
    for cat in cats_to_try:
        cat_models = catalog.get(cat, {})
        for m, info in cat_models.items():
            area = info["surface_area_m2"]
            if area >= min_area:
                candidates.append((cat, m, area))

    if not candidates:
        raise RuntimeError(
            f"No surface models found with area >= {min_area:.2f} m² "
            f"(categories searched: {cats_to_try})"
        )

    # Weighted random: prefer larger surfaces (weight = area).
    areas = np.array([c[2] for c in candidates])
    probs = areas / areas.sum()
    idx = rng.choice(len(candidates), p=probs)
    cat, m, area = candidates[idx]
    synset = _resolve_synset(cat)
    print(f"[Pipeline] Picked surface: {cat}/{m} (area={area:.4f} m²)")
    return cat, m, synset


def _make_obj_cfg(name, category, model, position, fixed_base=False, bounding_box=None):
    """Build a DatasetObject config dict for the env ``objects`` list."""
    cfg = dict(
        type="DatasetObject",
        name=name,
        category=category,
        model=model,
        fixed_base=fixed_base,
        position=list(position),
        orientation=[0.0, 0.0, 0.0, 1.0],
    )
    if bounding_box is not None:
        cfg["bounding_box"] = list(bounding_box)
    return cfg


# ---------------------------------------------------------------------------
# Object config builders (domain randomization)
# ---------------------------------------------------------------------------

def _build_clutter_objects(rng, density_key):
    """Randomize and return (obj_cfgs, role_map, selection).

    obj_cfgs: list of DatasetObject config dicts (parked far away).
    role_map: {obj_name: role_str}.
    selection: metadata dict for diagnostics.
    """
    density = DENSITY_PRESETS[density_key]
    cfgs, roles = [], {}
    idx = 0

    # Target (1) — must have a model in the asset catalog.
    entry = _pick_synset_with_model(TARGET_POOL, rng)
    target_synset = entry[0] if entry else "coffee_cup.n.01"
    cat = _synset_to_category(target_synset)
    model = _pick_random_model(cat, rng)
    if model:
        name = f"target_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
        roles[name] = "target"
        idx += 1

    # Fragile.
    fragile_synsets = []
    for i in range(density["fragile_count"]):
        entry = _pick_synset_with_model(FRAGILE_POOL, rng, exclude={target_synset})
        if entry is None:
            continue
        synset = entry[0]
        cat = _synset_to_category(synset)
        model = _pick_random_model(cat, rng)
        if model:
            name = f"fragile_{cat}_{idx}"
            cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
            roles[name] = "fragile"
            fragile_synsets.append(synset)
            idx += 1

    # Clutter.
    for i in range(density["clutter_count"]):
        entry = _pick_synset_with_model(CLUTTER_POOL, rng)
        if entry is None:
            continue
        synset = entry[0]
        cat = _synset_to_category(synset)
        model = _pick_random_model(cat, rng)
        if model:
            name = f"clutter_{cat}_{idx}"
            cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
            roles[name] = "clutter"
            idx += 1

    selection = {
        "target_synset": target_synset,
        "fragile_synsets": fragile_synsets,
    }
    return cfgs, roles, selection


def _build_stack_objects(rng, stack_height_key, target_synset=None, stack_synset=None):
    preset = STACK_HEIGHT_PRESETS[stack_height_key]
    stack_above = preset["stack_above"]

    if target_synset is None:
        entry = _pick_synset_with_model(STACK_TARGET_POOL, rng)
        target_synset = entry[0] if entry else "plate.n.04"
    if stack_synset is None:
        entry = _pick_synset_with_model(STACK_ITEM_POOL, rng)
        stack_synset = entry[0] if entry else "plate.n.04"

    cfgs, roles = [], {}
    idx = 0

    cat = _synset_to_category(target_synset)
    model = _pick_random_model(cat, rng)
    if model:
        name = f"target_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
        roles[name] = "target"
        idx += 1

    stack_cat = _synset_to_category(stack_synset)
    for i in range(stack_above):
        model = _pick_random_model(stack_cat, rng)
        if model:
            name = f"stack_{stack_cat}_{idx}"
            cfgs.append(_make_obj_cfg(name, stack_cat, model, position=(100 + idx, 100, -100)))
            roles[name] = "stack"
            idx += 1

    selection = {
        "target_synset": target_synset,
        "stack_synset": stack_synset,
        "stack_above": stack_above,
    }
    return cfgs, roles, selection


def _build_transfer_objects(rng, food_synset=None, source_synset=None,
                            dest_synset=None, goal_predicate=None):
    # Pick synsets that actually have models in the asset catalog.
    if food_synset is None:
        entry = _pick_synset_with_model(TRANSFER_FOOD_POOL, rng)
        food_synset = entry[0] if entry else "cookie.n.01"
    if source_synset is None:
        entry = _pick_synset_with_model(TRANSFER_SOURCE_POOL, rng)
        source_synset = entry[0] if entry else "plate.n.04"
    if dest_synset is None:
        entry = _pick_synset_with_model(TRANSFER_DEST_POOL, rng)
        if entry:
            dest_synset = entry[0]
            if goal_predicate is None:
                goal_predicate = entry[1]
        else:
            dest_synset = "bowl.n.01"
    if goal_predicate is None:
        goal_predicate = "inside"

    cfgs, roles = [], {}
    idx = 0
    for synset, role in [(food_synset, "food"), (source_synset, "source"), (dest_synset, "dest")]:
        cat = _synset_to_category(synset)
        model = _pick_random_model(cat, rng)
        if model:
            name = f"{role}_{cat}_{idx}"
            cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
            roles[name] = role
            idx += 1

    selection = {
        "food_synset": food_synset,
        "source_synset": source_synset,
        "dest_synset": dest_synset,
        "goal_predicate": goal_predicate,
    }
    return cfgs, roles, selection


# ---------------------------------------------------------------------------
# BDDL generation (for LTL safety files — sampler is bypassed)
# ---------------------------------------------------------------------------

def _generate_bddl(args, activity_name, support_synset, rng):
    support_room = None  # No rooms in empty Scene.
    if args.setup == "clutter":
        result = generate_activity(
            activity_name, support_synset, support_room, args.clutter_density,
        )
        return (*result, None)
    elif args.setup == "stack":
        return generate_stack_activity(
            activity_name, support_synset, support_room, args.stack_height,
            target_synset=args.target_synset, stack_synset=args.stack_synset, rng=rng,
        )
    elif args.setup == "transfer":
        return generate_transfer_activity(
            activity_name, support_synset, support_room,
            food_synset=args.food_synset, source_synset=args.source_synset,
            dest_synset=args.dest_synset, goal_predicate=args.goal_predicate, rng=rng,
        )
    raise ValueError(f"Unknown setup: {args.setup}")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def setup_run_dir(args):
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = args.surface_category or "random"
        args.run_dir = os.path.join(_DEFAULT_RUNS_DIR, f"empty_{label}_{args.setup}_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.debug_jsonl is None:
        args.debug_jsonl = os.path.join(args.run_dir, "diagnostics.jsonl")
    if args.save_video is True:
        args.save_video = os.path.join(args.run_dir, "rollout.mp4")
    elif args.save_video is False:
        args.save_video = None
    args.scene_model = f"empty_{args.surface_category or 'random'}"
    print(f"[Pipeline] Run directory: {args.run_dir}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    rng = np.random.default_rng(args.seed)
    surface_cat, surface_model, support_synset = _pick_surface(
        rng, args.surface_category, args.surface_model,
    )
    args.surface_category = surface_cat
    activity_name = args.activity_name or f"auto_{args.setup}_empty_{surface_cat}"

    bddl_text, ltl_safety, bddl_path, json_path, selection = _generate_bddl(
        args, activity_name, support_synset, rng,
    )
    print(f"[Pipeline] Dry-run (empty scene, setup={args.setup}):")
    print(f"  Surface:    {surface_cat} / {surface_model}")
    print(f"  BDDL:       {bddl_path}")
    print(f"  ltl_safety: {json_path}")
    print(f"  activity:   {activity_name}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "setup": args.setup, "surface": f"{surface_cat}/{surface_model}",
        **({"selection": selection} if selection else {}),
    })


def run_sim(args):
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS, EdgeAlignObject, EdgeAlignRequest,
        place_franka_edge_aligned,
    )
    from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True

    _MAX_SURFACE_RETRIES = 3

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 1000
        min_area = _MIN_SURFACE_AREA_M2
        episode_done = False

        # Retry loop: if packing fails, close env and try a larger surface.
        for surface_attempt in range(_MAX_SURFACE_RETRIES):
            rng = np.random.default_rng(ep_seed + surface_attempt * 100)

            # -- Domain randomization: pick surface + objects ---------------
            surface_cat, surface_model, support_synset = _pick_surface(
                rng, args.surface_category, args.surface_model,
                min_area=min_area,
            )
            activity_name = args.activity_name or f"auto_{args.setup}_empty_{surface_cat}"

            # Generate BDDL + LTL safety files (for LTL monitor, not sampler).
            _, _, bddl_path, _, bddl_selection = _generate_bddl(
                args, activity_name, support_synset, rng,
            )
            refresh_activity_cache()

            if args.setup == "clutter":
                obj_cfgs, roles, selection = _build_clutter_objects(rng, args.clutter_density)
            elif args.setup == "stack":
                obj_cfgs, roles, selection = _build_stack_objects(
                    rng, args.stack_height,
                    target_synset=args.target_synset, stack_synset=args.stack_synset,
                )
            elif args.setup == "transfer":
                obj_cfgs, roles, selection = _build_transfer_objects(
                    rng, food_synset=args.food_synset, source_synset=args.source_synset,
                    dest_synset=args.dest_synset, goal_predicate=args.goal_predicate,
                )

            # Surface config: placed at origin, fixed.
            # Look up the surface height from the catalog so we can place it
            # with the bottom of its legs on the floor (z=0).
            catalog = _load_surface_catalog()
            surface_height = catalog.get(surface_cat, {}).get(
                surface_model, {}).get("height_m", 0.75)
            surface_z = surface_height / 2.0  # center of bbox above floor

            surface_cfg = _make_obj_cfg(
                name="support_surface",
                category=surface_cat,
                model=surface_model,
                position=[0.0, 0.0, surface_z],
                fixed_base=True,
            )

            # -- Build env config (grasp_task_demo pattern) -----------------
            all_objects = [surface_cfg] + obj_cfgs
            cfg = dict(
                scene=dict(type="Scene"),
                robots=[dict(
                    type="FrankaMounted",
                    obs_modalities=["rgb"],
                    action_type="continuous",
                    action_normalize=True,
                    controller_config={
                        "arm_0": {"name": "OperationalSpaceController"},
                        "gripper_0": {"name": "MultiFingerGripperController"},
                    },
                )],
                objects=all_objects,
                task=dict(type="DummyTask"),
            )

            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes} "
                  f"(surface attempt {surface_attempt + 1}/{_MAX_SURFACE_RETRIES})")
            print(f"[Pipeline] Surface: {surface_cat}/{surface_model}, "
                  f"objects: {len(obj_cfgs)}, setup: {args.setup}")
            sys.stdout.flush()

            env = og.Environment(configs=cfg)
            try:
                env.reset()

                # Park robot far from origin so it doesn't interfere with
                # object placement and physics settling on the table.
                robot = env.robots[0]
                robot.set_position_orientation(
                    position=(50.0, 50.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
                og.sim.step()

                # -- Locate objects in the scene ----------------------------
                support_obj = env.scene.object_registry("name", "support_surface")
                if support_obj is None:
                    raise RuntimeError("Support surface not found in scene.")

                aabb_min, aabb_max = support_obj.aabb
                surface_bounds_xy = (
                    (float(aabb_min[0]), float(aabb_min[1])),
                    (float(aabb_max[0]), float(aabb_max[1])),
                )
                table_top_z = float(aabb_max[2])
                floor_z = 0.0
                print(f"[Pipeline] Surface bounds: {surface_bounds_xy}, "
                      f"top_z={table_top_z:.3f}")
                sys.stdout.flush()

                # Build objects_by_inst + roles lookup.
                objects_by_inst = {}
                roles_by_inst = {}
                for obj_cfg in obj_cfgs:
                    name = obj_cfg["name"]
                    obj = env.scene.object_registry("name", name)
                    if obj is not None:
                        objects_by_inst[name] = obj
                        roles_by_inst[name] = roles[name]

                print(f"[Pipeline] Task objects found: {len(objects_by_inst)}")
                sys.stdout.flush()

                # -- Place objects on the surface via pack layout -----------
                if args.setup in ("clutter", "transfer"):
                    from omnigibson.utils.clutter_pack_layout import (
                        ClutterObjectDescriptor, build_clutter_pack,
                        apply_pack_transform,
                    )

                    descriptors = []
                    for inst, obj in objects_by_inst.items():
                        try:
                            a_min, a_max = obj.aabb
                            dx = max(0.01, float(a_max[0] - a_min[0]))
                            dy = max(0.01, float(a_max[1] - a_min[1]))
                            dz = max(0.01, float(a_max[2] - a_min[2]))
                        except Exception:
                            continue
                        descriptors.append(ClutterObjectDescriptor(
                            instance_id=inst, role=roles_by_inst[inst],
                            half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
                        ))

                    zone = compute_tabletop_zone(
                        surface_bounds_xy=surface_bounds_xy,
                        obstacle_bounds_xy=None,
                        edge_margin_m=0.04,
                    )
                    half_w = 0.5 * (zone.red_zone_bounds[1][0]
                                    - zone.red_zone_bounds[0][0])
                    half_h = 0.5 * (zone.red_zone_bounds[1][1]
                                    - zone.red_zone_bounds[0][1])
                    cx = 0.5 * (surface_bounds_xy[0][0] + surface_bounds_xy[1][0])
                    cy = 0.5 * (surface_bounds_xy[0][1] + surface_bounds_xy[1][1])
                    pack_origin = (cx, cy, table_top_z)

                    # Retry with decreasing clearance.
                    bounds_local = ((-half_w, -half_h), (half_w, half_h))
                    pack_spec = None
                    for clearance in (0.025, 0.015, 0.008, 0.003):
                        try:
                            pack_spec = build_clutter_pack(
                                table_obj_name="support_surface",
                                descriptors=descriptors,
                                seed=ep_seed,
                                min_clearance=clearance,
                                placement_bounds_local=bounds_local,
                            )
                            break
                        except RuntimeError as e:
                            print(f"[Pipeline] Pack clearance={clearance:.3f}: {e}")
                            sys.stdout.flush()
                    if pack_spec is None:
                        raise RuntimeError(
                            "Could not pack objects on surface at any clearance.")
                    apply_pack_transform(
                        pack_spec, objects_by_inst, pack_origin, pack_yaw=0.0)
                    print(f"[Pipeline] Pack placed: {len(descriptors)} objects")

                elif args.setup == "stack":
                    from omnigibson.utils.clutter_pack_layout import (
                        StackObjectDescriptor, build_stack_layout,
                        apply_stack_transform,
                    )

                    stack_descs = []
                    for inst, obj in objects_by_inst.items():
                        try:
                            a_min, a_max = obj.aabb
                            dx = max(0.01, float(a_max[0] - a_min[0]))
                            dy = max(0.01, float(a_max[1] - a_min[1]))
                            dz = max(0.01, float(a_max[2] - a_min[2]))
                        except Exception:
                            continue
                        stack_descs.append(StackObjectDescriptor(
                            instance_id=inst, role=roles_by_inst[inst],
                            half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
                        ))

                    cx = 0.5 * (surface_bounds_xy[0][0] + surface_bounds_xy[1][0])
                    cy = 0.5 * (surface_bounds_xy[0][1] + surface_bounds_xy[1][1])
                    stack_origin = (cx, cy, table_top_z)
                    stack_spec = build_stack_layout(
                        support_obj_name="support_surface",
                        descriptors=stack_descs, seed=ep_seed,
                    )
                    apply_stack_transform(stack_spec, objects_by_inst, stack_origin)
                    print(f"[Pipeline] Stack placed: {len(stack_descs)} objects")

                # Pack succeeded — mark episode as ready to continue.
                episode_done = True

            except RuntimeError as pack_err:
                # Pack or surface error — close this env and retry with a
                # larger minimum area so _pick_surface selects a bigger table.
                print(f"[Pipeline] Surface attempt {surface_attempt + 1} failed: "
                      f"{pack_err}")
                sys.stdout.flush()
                env.close()
                min_area *= 2.0  # Double the minimum area for next attempt.
                continue

            # If we reach here, packing succeeded — break the retry loop.
            break
        else:
            # All surface retries exhausted.
            raise RuntimeError(
                f"Episode {ep + 1}: could not find a surface large enough "
                f"after {_MAX_SURFACE_RETRIES} attempts."
            )

        # -- From here on, env is live and packing succeeded. ---------------
        try:

            sys.stdout.flush()

            # -- Settle physics ---------------------------------------------
            for obj in objects_by_inst.values():
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            settle_fn = make_settle_fn(og, th)
            settle_fn(objects_by_inst)

            # -- Transfer: teleport food onto source ------------------------
            if args.setup == "transfer":
                food_obj, source_obj = None, None
                for name, role in roles_by_inst.items():
                    obj = objects_by_inst.get(name)
                    if obj is None:
                        continue
                    if role == "food" and food_obj is None:
                        food_obj = obj
                    elif role == "source" and source_obj is None:
                        source_obj = obj
                if food_obj and source_obj:
                    src_pos = source_obj.get_position_orientation()[0]
                    try:
                        src_top_z = float(source_obj.aabb[1][2])
                    except Exception:
                        src_top_z = float(src_pos[2]) + 0.03
                    try:
                        f_half_h = 0.5 * max(0.01, float(food_obj.aabb[1][2] - food_obj.aabb[0][2]))
                    except Exception:
                        f_half_h = 0.02
                    food_obj.set_position_orientation(
                        position=(float(src_pos[0]), float(src_pos[1]),
                                  src_top_z + f_half_h + 0.005),
                    )
                    if hasattr(food_obj, "keep_still"):
                        food_obj.keep_still()
                    og.sim.step()
                    print(f"[Pipeline] Food teleported onto source")

            # -- Robot placement --------------------------------------------
            robot = env.robots[0]
            zone = compute_tabletop_zone(
                surface_bounds_xy=surface_bounds_xy, obstacle_bounds_xy=None,
                edge_margin_m=0.04,
            )

            pack_objects_world = []
            for inst, obj in objects_by_inst.items():
                try:
                    pos = obj.get_position_orientation()[0]
                    pack_objects_world.append(EdgeAlignObject(
                        name=inst, role=roles_by_inst[inst],
                        position_xy=(float(pos[0]), float(pos[1])),
                    ))
                except Exception:
                    continue

            edge_result = None
            if pack_objects_world:
                edge_result = place_franka_edge_aligned(EdgeAlignRequest(
                    table_aabb_xy=zone.surface_bounds,
                    pack_objects_world=tuple(pack_objects_world),
                    role_weights=DEFAULT_ROLE_WEIGHTS,
                    robot_half_extent_xy=robot_half_extent_xy(robot),
                    edge_gap_m=args.mount_gap_m, edge_margin_m=0.05,
                    scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10,
                                    0.15, -0.15, 0.20, -0.20),
                ))
                robot.set_position_orientation(
                    position=(edge_result.base_pose["position"][0],
                              edge_result.base_pose["position"][1], floor_z),
                    orientation=edge_result.base_pose["orientation"],
                )
                og.sim.step()
                print(f"[Pipeline] Robot: edge={edge_result.edge_label}, "
                      f"gap={edge_result.gap_actual:.3f}")
            else:
                cx = 0.5 * (surface_bounds_xy[0][0] + surface_bounds_xy[1][0])
                robot.set_position_orientation(
                    position=(cx, surface_bounds_xy[0][1] - 0.3, floor_z))
                og.sim.step()
                print("[Pipeline] Robot placed at fallback position")

            sys.stdout.flush()

            # -- Gate -------------------------------------------------------
            target_obj = None
            for name, role in roles_by_inst.items():
                if role in ("target", "food"):
                    target_obj = objects_by_inst.get(name)
                    break
            if target_obj is None and objects_by_inst:
                target_obj = next(iter(objects_by_inst.values()))

            rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
            if target_obj is not None:
                tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
            else:
                tp = rp
            target_dist = math.hypot(rp[0] - tp[0], rp[1] - tp[1])
            gate_pass = (
                all(math.isfinite(v) for v in rp + tp)
                and abs(rp[2] - floor_z) <= 0.03
                and (edge_result is None or not edge_result.collision_hits)
                and 0.20 <= target_dist <= 1.10
            )
            print(f"[Pipeline] Gate: pass={gate_pass}, dist={target_dist:.3f}")
            if args.strict_gate and not gate_pass:
                raise RuntimeError("Strict gate failed.")

            # -- Save scene snapshot ----------------------------------------
            if gate_pass:
                scene_save = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
                og.sim.save(json_paths=[scene_save])
                print(f"[Pipeline] Scene saved: {scene_save}")

            # -- LTL rollout ------------------------------------------------
            summary, executed = run_ltl_rollout(
                env=env, activity_name=activity_name,
                scene_model=args.scene_model,
                active_objects_by_inst=objects_by_inst,
                robot=robot, target_obj=target_obj,
                args=args, episode=ep, rng=rng,
            )

            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "setup": args.setup,
                "surface": f"{surface_cat}/{surface_model}",
                "activity_name": activity_name,
                "gate_pass": gate_pass,
                "ltl_violated": summary["violated"],
                "steps_executed": executed,
                "selection": selection,
            })

        finally:
            env.close()

    print("[Pipeline] Shutdown simulator.")
    pipeline_exit()


def main():
    args = parse_args()
    setup_run_dir(args)
    if args.dry_run:
        run_dry_run(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main()
