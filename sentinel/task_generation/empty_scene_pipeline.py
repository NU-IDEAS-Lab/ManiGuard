"""Empty-scene task generation pipeline.

Starts from a bare Scene (floor plane only), spawns a randomized support
surface and task objects via the env config ``objects`` list (following the
grasp_task_demo pattern), then runs the standard clutter / stack / transfer
placement + LTL-monitored rollouts.

Domain randomization: surface category/model, target, fragile, and clutter
types are all randomized per episode from the pools in pipeline_common.

Usage:
    # Clutter on empty scene (random surface + objects)
    python -m sentinel.task_generation.empty_scene_pipeline \\
        --setup clutter --episodes 1 --steps 300 --save-video

    # Stack on a specific desk
    python -m sentinel.task_generation.empty_scene_pipeline \\
        --setup stack --surface-category desk --stack-height medium \\
        --episodes 1 --steps 300 --save-video

    # Food transfer
    python -m sentinel.task_generation.empty_scene_pipeline \\
        --setup transfer --episodes 1 --steps 300 --save-video

    # Dry-run (generate BDDL only, no sim)
    python -m sentinel.task_generation.empty_scene_pipeline \\
        --setup clutter --dry-run
"""

import argparse
import copy
import math
import os
import sys
from datetime import datetime

import numpy as np

from sentinel.task_generation.pipeline_common import (
    append_jsonl,
    make_settle_fn,
    object_aabb_dims,
    pipeline_exit,
    refresh_activity_cache,
    robot_half_extent_xy,
    run_ltl_rollout,
)
from sentinel.utils.bddl_generator import (
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
    generate_clutter_activity as generate_activity,
    generate_stack_activity,
    generate_transfer_activity,
)
import logging

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")
_PARK_POS = (100.0, 100.0, -100.0)

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
    # Batch mode.
    p.add_argument("--batch-size", type=int, default=None,
                   help="Episodes per env load (default: all episodes in one batch)")
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


def _pick_surface(rng, category=None, model=None, min_area=None):
    """Pick a surface region from placeable, filtered by minimum region area.

    Operates at region granularity: 2-region models contribute both regions
    as independent candidates, so the smaller one is also eligible on its
    own merit. Area-weighted random: larger regions preferred. No height
    filter -- true object height is read from live world AABB post-reset.

    Returns (pick, synset) where pick is a self-contained dict carrying
    category / model / region_id / xy_min / xy_max / top_plane_z_local /
    area_m2 / reachable_edge_labels / height_m.
    """
    from sentinel.task_generation.utils.placeable import pick_surface_from_placeable
    if min_area is None:
        min_area = _MIN_SURFACE_AREA_M2
    pick = pick_surface_from_placeable(
        rng, required_area_m2=min_area,
        required_category=category, required_model=model,
        weighted_by_area=True,
    )
    synset = _resolve_synset(pick["category"])
    print(f"[Pipeline] Picked surface: {pick['category']}/{pick['model']} "
          f"(region={pick['region_id']}, area={pick['area_m2']:.4f} m²)")
    return pick, synset


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

def _generate_bddl(args, activity_name, support_synset, rng, selection=None):
    support_room = None  # No rooms in empty Scene.
    if args.setup == "clutter":
        # Build pre_selection from the already-resolved synsets.
        pre = {
            "target_synset": args.target_synset or (selection or {}).get("target_synset", "coffee_cup.n.01"),
            "fragile_picks": (selection or {}).get("fragile_synsets", []),
            "clutter_picks": [],
        }
        return generate_activity(
            activity_name, support_synset, support_room, args.clutter_density,
            rng=rng, pre_selection=pre,
        )
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
    from sentinel.task_generation.pipeline_common import init_run_dir
    label = f"empty_{args.surface_category or 'random'}_{args.setup}"
    init_run_dir(args, label)
    args.scene_model = f"empty_{args.surface_category or 'random'}"
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    rng = np.random.default_rng(args.seed)
    pick, support_synset = _pick_surface(
        rng, args.surface_category, args.surface_model,
    )
    surface_cat = pick["category"]
    args.surface_category = surface_cat
    activity_name = args.activity_name or f"auto_{args.setup}_empty_{surface_cat}"

    # Build a pre_selection for the dry-run BDDL generation.
    if args.setup == "clutter":
        dry_cfgs, dry_roles, dry_sel = _build_clutter_objects(rng, args.clutter_density)
    elif args.setup == "stack":
        dry_cfgs, dry_roles, dry_sel = _build_stack_objects(
            rng, args.stack_height,
            target_synset=args.target_synset, stack_synset=args.stack_synset,
        )
    elif args.setup == "transfer":
        dry_cfgs, dry_roles, dry_sel = _build_transfer_objects(
            rng, food_synset=args.food_synset, source_synset=args.source_synset,
            dest_synset=args.dest_synset, goal_predicate=args.goal_predicate,
        )
    else:
        dry_sel = {}

    # Patch args with resolved synsets for BDDL gen.
    if args.setup == "transfer":
        args.food_synset = dry_sel.get("food_synset")
        args.source_synset = dry_sel.get("source_synset")
        args.dest_synset = dry_sel.get("dest_synset")
        args.goal_predicate = dry_sel.get("goal_predicate")
    elif args.setup == "stack":
        args.target_synset = dry_sel.get("target_synset")
        args.stack_synset = dry_sel.get("stack_synset")

    bddl_text, ltl_safety, bddl_path, json_path, selection = _generate_bddl(
        args, activity_name, support_synset, rng, selection=dry_sel,
    )
    print(f"[Pipeline] Dry-run (empty scene, setup={args.setup}):")
    print(f"  Surface:    {pick['category']} / {pick['model']}")
    print(f"  BDDL:       {bddl_path}")
    print(f"  ltl_safety: {json_path}")
    print(f"  activity:   {activity_name}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "setup": args.setup, "surface": f"{pick['category']}/{pick['model']}",
        **({"selection": selection} if selection else {}),
    })


def _park_objects(objects_by_inst, og_mod):
    """Move all objects to the park position."""
    for obj in objects_by_inst.values():
        obj.set_position_orientation(position=_PARK_POS)
        if hasattr(obj, "keep_still"):
            obj.keep_still()
    og_mod.sim.step()


def _run_episode_inner(ep, ep_seed, args, env, og, th, robot, support_obj,
                       surface_bounds_xy, table_top_z, floor_z,
                       obj_cfgs, roles, selection, activity_name, support_synset,
                       surface_cat, surface_model):
    """Run a single episode (placement + rollout) on a live env.

    Objects are already in the scene (parked). This function unparks them,
    places them, runs the rollout, then parks them back.
    """
    from sentinel.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS, EdgeAlignObject, EdgeAlignRequest,
        place_franka_edge_aligned,
    )
    from sentinel.utils.tabletop_workspace import compute_tabletop_zone

    # -- Resolve scene objects from configs --------------------------------
    objects_by_inst = {}
    roles_by_inst = {}
    for obj_cfg in obj_cfgs:
        name = obj_cfg["name"]
        obj = env.scene.object_registry("name", name)
        if obj is not None:
            objects_by_inst[name] = obj
            roles_by_inst[name] = roles[name]

    # -- Place objects on the surface --------------------------------------
    if args.setup in ("clutter", "transfer"):
        from sentinel.utils.clutter_pack_layout import (
            ClutterObjectDescriptor, build_clutter_pack, apply_pack_transform,
        )

        descriptors = []
        for inst, obj in objects_by_inst.items():
            dims = object_aabb_dims(obj)
            if dims is None:
                continue
            dx, dy, dz = dims
            descriptors.append(ClutterObjectDescriptor(
                instance_id=inst, role=roles_by_inst[inst],
                half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
            ))

        zone = compute_tabletop_zone(
            surface_bounds_xy=surface_bounds_xy, obstacle_bounds_xy=None,
            edge_margin_m=0.04,
        )
        half_w = 0.5 * (zone.red_zone_bounds[1][0] - zone.red_zone_bounds[0][0])
        half_h = 0.5 * (zone.red_zone_bounds[1][1] - zone.red_zone_bounds[0][1])
        cx = 0.5 * (surface_bounds_xy[0][0] + surface_bounds_xy[1][0])
        cy = 0.5 * (surface_bounds_xy[0][1] + surface_bounds_xy[1][1])
        pack_origin = (cx, cy, table_top_z)

        bounds_local = ((-half_w, -half_h), (half_w, half_h))
        pack_spec = None
        for clearance in (0.025, 0.015, 0.008, 0.003):
            try:
                pack_spec = build_clutter_pack(
                    table_obj_name="support_surface",
                    descriptors=descriptors, seed=ep_seed,
                    min_clearance=clearance,
                    placement_bounds_local=bounds_local,
                )
                break
            except RuntimeError as e:
                print(f"[Pipeline] Pack clearance={clearance:.3f}: {e}")
        if pack_spec is None:
            raise RuntimeError("Could not pack objects on surface at any clearance.")
        apply_pack_transform(pack_spec, objects_by_inst, pack_origin, pack_yaw=0.0)
        print(f"[Pipeline] Pack placed: {len(descriptors)} objects")

    elif args.setup == "stack":
        from sentinel.utils.clutter_pack_layout import (
            StackObjectDescriptor, build_stack_layout, apply_stack_transform,
        )

        stack_descs = []
        for inst, obj in objects_by_inst.items():
            dims = object_aabb_dims(obj)
            if dims is None:
                continue
            dx, dy, dz = dims
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

    # -- Settle physics ----------------------------------------------------
    for obj in objects_by_inst.values():
        if hasattr(obj, "keep_still"):
            obj.keep_still()
    settle_fn = make_settle_fn(og, th)
    settle_fn(objects_by_inst)

    # -- Transfer: teleport food onto source -------------------------------
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
            print("[Pipeline] Food teleported onto source")

    # -- Robot placement ---------------------------------------------------
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
        except Exception as exc:
            log.warning("empty_scene _run_episode_inner: pose read for %s failed: %s", getattr(obj, "name", obj), exc)
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

    # -- Gate --------------------------------------------------------------
    target_obj = None
    for name, role in roles_by_inst.items():
        if role in ("target", "food"):
            target_obj = objects_by_inst.get(name)
            break
    if target_obj is None and objects_by_inst:
        target_obj = next(iter(objects_by_inst.values()))

    rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
    tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]] if target_obj else rp
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

    # Save full Omniverse scene snapshot (same format BasePipeline uses):
    # objects_info.init_info + state.registry.object_registry, replayable by
    # og.sim.load() and by so101_franka_teleop's _build_from_snapshot.
    if gate_pass:
        scene_save = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
        og.sim.save(json_paths=[scene_save])
        print(f"[Pipeline] Scene saved: {scene_save}")

    # -- LTL rollout -------------------------------------------------------
    rng = np.random.default_rng(ep_seed)
    summary, executed = run_ltl_rollout(
        env=env, activity_name=activity_name,
        scene_model=args.scene_model,
        active_objects_by_inst=objects_by_inst,
        robot=robot, target_obj=target_obj,
        args=args, episode=ep, rng=rng,
        support_obj=support_obj,
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

    # -- Park objects back -------------------------------------------------
    _park_objects(objects_by_inst, og)
    robot.set_position_orientation(position=(50.0, 50.0, 0.0))
    og.sim.step()


def run_sim(args):
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True

    batch_size = args.batch_size or args.episodes

    for batch_start in range(0, args.episodes, batch_size):
        batch_end = min(batch_start + batch_size, args.episodes)
        batch_eps = list(range(batch_start, batch_end))

        # -- Pre-select objects for all episodes in batch ------------------
        episode_data = []  # (obj_cfgs, roles, selection, ep_seed)
        for ep in batch_eps:
            ep_seed = args.seed + ep * 1000
            rng = np.random.default_rng(ep_seed)
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
            else:
                raise ValueError(f"Unknown setup: {args.setup}")
            episode_data.append((obj_cfgs, roles, selection, ep_seed))

        # -- Compute max footprint across episodes -------------------------
        from sentinel.utils.bddl_generator import _load_footprint_catalog, _median_footprint
        fp_catalog = _load_footprint_catalog()
        max_footprint = _MIN_SURFACE_AREA_M2
        for obj_cfgs, _, _, _ in episode_data:
            ep_fp = sum(
                _median_footprint(fp_catalog, _resolve_synset(c["category"]))
                for c in obj_cfgs
            )
            max_footprint = max(max_footprint, ep_fp * 1.3)

        # -- Pick surface for the batch (area >= max footprint) ------------
        rng_surface = np.random.default_rng(args.seed + batch_start)
        surface_pick, support_synset = _pick_surface(
            rng_surface, args.surface_category, args.surface_model,
            min_area=max_footprint,
        )
        surface_cat = surface_pick["category"]
        surface_model = surface_pick["model"]
        activity_name = args.activity_name or f"auto_{args.setup}_empty_{surface_cat}"
        print(f"[Pipeline] Max footprint across batch: {max_footprint:.4f} m²")

        # -- Collect all unique objects ------------------------------------
        all_obj_cfgs_map = {}  # name -> cfg (deduplicated)
        for obj_cfgs, roles, _, _ in episode_data:
            for cfg in obj_cfgs:
                name = cfg["name"]
                if name not in all_obj_cfgs_map:
                    parked = dict(cfg)
                    parked["position"] = list(_PARK_POS)
                    all_obj_cfgs_map[name] = parked

        # -- Build env config with surface + all objects -------------------
        # Spawn the support with its origin at z = height_m/2 so the bottom
        # sits on the floor (z=0) under the B1K center-origin convention.
        # Placement bounds come from the SAME region surface_pick selected so
        # 2-region models stay self-consistent across picker and spawn.
        surface_height_m = float(surface_pick["height_m"])
        surface_top_plane_z_local = float(surface_pick["top_plane_z_local"])
        surface_region = surface_pick
        surface_spawn_xyz = (0.0, 0.0, surface_height_m / 2.0)
        surface_cfg = _make_obj_cfg(
            name="support_surface", category=surface_cat,
            model=surface_model, position=list(surface_spawn_xyz),
            fixed_base=True,
        )

        all_objects = [surface_cfg] + list(all_obj_cfgs_map.values())
        cfg = dict(
            scene=dict(type="Scene"),
            robots=[dict(
                type="FrankaMounted", obs_modalities=["rgb"],
                action_type="continuous", action_normalize=True,
                controller_config={
                    "arm_0": {"name": "OperationalSpaceController"},
                    "gripper_0": {"name": "MultiFingerGripperController"},
                },
            )],
            objects=all_objects,
            task=dict(type="DummyTask"),
            # 3 external cameras for multi-view recording (same layout as
            # BasePipeline's scene-based pipelines).
            env={"external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "name": "cam_opposite",
                    "relative_prim_path": "/cam_opposite",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {"image_height": 720, "image_width": 1280},
                },
                {
                    "sensor_type": "VisionSensor",
                    "name": "cam_left",
                    "relative_prim_path": "/cam_left",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {"image_height": 720, "image_width": 1280},
                },
                {
                    "sensor_type": "VisionSensor",
                    "name": "cam_right",
                    "relative_prim_path": "/cam_right",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {"image_height": 720, "image_width": 1280},
                },
            ]},
        )

        print(f"\n[Pipeline] Batch {batch_start//batch_size + 1}: "
              f"episodes {batch_start+1}-{batch_end}, "
              f"surface={surface_cat}/{surface_model}, "
              f"objects={len(all_obj_cfgs_map)}")
        sys.stdout.flush()

        import time as _time
        _t_env_start = _time.time()
        env = og.Environment(configs=cfg)
        try:
            env.reset()
            robot = env.robots[0]
            robot.set_position_orientation(position=(50.0, 50.0, 0.0))
            og.sim.step()
            print(f"[Pipeline] Env init: {_time.time() - _t_env_start:.1f}s")

            support_obj = env.scene.object_registry("name", "support_surface")
            if support_obj is None:
                raise RuntimeError("Support surface not found in scene.")

            # Placement bounds come from the placeable region (object-local),
            # translated by the spawn pose. Identity orientation + center-origin
            # convention -> world = spawn + local. This is the raycast-validated
            # usable zone (may be a strict subset of the full object AABB, e.g.
            # excludes overhangs and occluded areas).
            region_xy_min = surface_region["xy_min"]
            region_xy_max = surface_region["xy_max"]
            surface_bounds_xy = (
                (surface_spawn_xyz[0] + float(region_xy_min[0]),
                 surface_spawn_xyz[1] + float(region_xy_min[1])),
                (surface_spawn_xyz[0] + float(region_xy_max[0]),
                 surface_spawn_xyz[1] + float(region_xy_max[1])),
            )
            table_top_z = surface_spawn_xyz[2] + surface_top_plane_z_local
            floor_z = 0.0
            print(f"[Pipeline] Surface bounds: {surface_bounds_xy}, "
                  f"top_z={table_top_z:.3f}  "
                  f"(region={surface_region['region_id']}, "
                  f"area={surface_region['area_m2']:.3f} m^2)")

            # -- Run each episode in the batch -----------------------------
            for idx, (obj_cfgs, roles, selection, ep_seed) in enumerate(episode_data):
                ep = batch_start + idx

                # Generate BDDL + LTL for this episode's synsets.
                saved_args = copy.copy(args)
                if args.setup == "transfer":
                    args.food_synset = selection["food_synset"]
                    args.source_synset = selection["source_synset"]
                    args.dest_synset = selection["dest_synset"]
                    args.goal_predicate = selection["goal_predicate"]
                elif args.setup == "stack":
                    args.target_synset = selection["target_synset"]
                    args.stack_synset = selection["stack_synset"]

                rng_bddl = np.random.default_rng(ep_seed)
                _, _, bddl_path, _, _ = _generate_bddl(
                    args, activity_name, support_synset, rng_bddl,
                    selection=selection,
                )
                refresh_activity_cache()

                # Restore args.
                for attr in ("food_synset", "source_synset", "dest_synset",
                             "goal_predicate", "target_synset"):
                    setattr(args, attr, getattr(saved_args, attr, None))
                if hasattr(saved_args, "stack_synset"):
                    args.stack_synset = saved_args.stack_synset

                print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
                sys.stdout.flush()
                _t_ep_start = _time.time()

                try:
                    _run_episode_inner(
                        ep=ep, ep_seed=ep_seed, args=args,
                        env=env, og=og, th=th, robot=robot,
                        support_obj=support_obj,
                        surface_bounds_xy=surface_bounds_xy,
                        table_top_z=table_top_z, floor_z=floor_z,
                        obj_cfgs=obj_cfgs, roles=roles,
                        selection=selection,
                        activity_name=activity_name,
                        support_synset=support_synset,
                        surface_cat=surface_cat,
                        surface_model=surface_model,
                    )
                    print(f"[Pipeline] Episode {ep + 1} took {_time.time() - _t_ep_start:.1f}s")
                except RuntimeError as e:
                    print(f"[Pipeline] Episode {ep + 1} failed: {e} "
                          f"({_time.time() - _t_ep_start:.1f}s)")
                    # Park everything and continue to next episode.
                    _park_objects(
                        {c["name"]: env.scene.object_registry("name", c["name"])
                         for c in obj_cfgs
                         if env.scene.object_registry("name", c["name"]) is not None},
                        og,
                    )
                    robot.set_position_orientation(position=(50.0, 50.0, 0.0))
                    og.sim.step()

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
