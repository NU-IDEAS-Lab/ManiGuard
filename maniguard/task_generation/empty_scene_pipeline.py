"""Empty-scene task generation pipeline.

Starts from a bare Scene (floor plane only), spawns a randomized support
surface and task objects via the env config ``objects`` list (following the
grasp_task_demo pattern), then runs the standard clutter / stack / transfer
placement + LTL-monitored rollouts.

Domain randomization: surface category/model, target, fragile, and clutter
types are all randomized per episode from the pools in pipeline_common.

Usage:
    # Clutter on empty scene (random surface + objects)
    python -m maniguard.task_generation.empty_scene_pipeline \\
        --setup clutter --episodes 1 --steps 300 --save-video

    # Stack on a specific desk
    python -m maniguard.task_generation.empty_scene_pipeline \\
        --setup stack --surface-category desk --stack-height medium \\
        --episodes 1 --steps 300 --save-video

    # Food transfer
    python -m maniguard.task_generation.empty_scene_pipeline \\
        --setup transfer --episodes 1 --steps 300 --save-video

    # Dry-run (generate BDDL only, no sim)
    python -m maniguard.task_generation.empty_scene_pipeline \\
        --setup clutter --dry-run
"""

import argparse
import copy
import logging
import math
import os
import sys

import numpy as np

from maniguard.task_generation.pipeline_common import (
    append_jsonl,
    make_settle_fn,
    object_aabb_dims,
    pipeline_exit,
    robot_half_extent_xy,
    run_ltl_rollout,
)
from maniguard.task_generation.transfer_scene_pipeline import build_transfer_objects
from maniguard.task_generation.utils.clutter_pipeline.select import (
    select_fillable_container,
    select_fragile,
)
from maniguard.task_generation.utils.clutter_pipeline.select import (
    select_obstacle as select_table_obstacle,
)
from maniguard.task_generation.utils.clutter_pipeline.select import (
    select_target as select_clutter_target,
)
from maniguard.task_generation.utils.liquid_transport.select import (
    select_liquid_fragile,
)
from maniguard.task_generation.utils.stack_pipeline.select import select_stack_objects
from maniguard.utils.goal_region import (
    GoalRegionSpec,
    build_goal_region_spec,
    family_uses_goal_region,
    spawn_goal_region_marker,
)
from maniguard.utils.task_spec import (
    DENSITY_PRESETS,
    LIQUID_PRESETS,
    STACK_HEIGHT_PRESETS,
    _pick_model_for_category,
    generate_liquid_transport_ltl_safety_json,
    generate_stack_activity,
    generate_transfer_activity,
)
from maniguard.utils.task_spec import (
    generate_clutter_activity as generate_activity,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
    p.add_argument("--setup", required=True,
                   choices=["clutter", "stack", "transfer", "liquid",
                            "dusty_transfer"])
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
    p.add_argument("--stack-mode", default="same",
                   choices=["same", "flat", "receptacle"],
                   help="Stack variant: same target/item, flat target, or "
                        "concave receptacle target.")
    p.add_argument("--stack-height", default="medium", choices=list(STACK_HEIGHT_PRESETS))
    p.add_argument("--target-model", default=None,
                   help="Override target model id; category resolved from the "
                        "stack-mode's pool.")
    p.add_argument("--stack-model", default=None,
                   help="Override stack-item model id; category resolved from "
                        "the stack-mode's pool (or target pool in same-mode).")
    # Transfer.
    p.add_argument("--food-model", default=None,
                   help="Override food model id; category is inferred from compat matrix.")
    p.add_argument("--source-model", default=None,
                   help="Override source container model id.")
    p.add_argument("--dest-model", default=None,
                   help="Override destination container model id.")
    p.add_argument("--goal-predicate", default=None, choices=["inside", "ontop"])
    # Dusty transfer (--setup dusty_transfer)
    p.add_argument("--sponge-model", default=None,
                   help="dusty_transfer: pin a sponge model "
                        "(choices: aewrov, klwueh, qewotb).")
    # Liquid.
    p.add_argument("--difficulty", default="medium", choices=list(LIQUID_PRESETS),
                   help="Liquid difficulty (spill threshold and tilt limit only)")
    p.add_argument("--container-category", default=None,
                   help="Liquid setup: pin fillable container category "
                        "(random from fillable_container_pool.json if omitted)")
    p.add_argument("--system-name", default="water",
                   help="Liquid particle system name")
    # Batch mode.
    p.add_argument("--batch-size", type=int, default=None,
                   help="Episodes per env load (default: all episodes in one batch)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_synset(category):
    from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
    synset = OBJECT_TAXONOMY.get_synset_from_category(category)
    if synset is None:
        raise RuntimeError(
            f"_resolve_synset: BEHAVIOR taxonomy has no synset for "
            f"category {category!r}. Surface picker handed a category "
            "not registered in OBJECT_TAXONOMY — regenerate "
            "placeable_surfaces_v1.json against the current taxonomy."
        )
    return synset


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
    from maniguard.task_generation.utils.placeable import pick_surface_from_placeable
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


def _make_obj_cfg(name, category, model, position, fixed_base=False):
    """Build a DatasetObject config dict for the env ``objects`` list."""
    return dict(
        type="DatasetObject",
        name=name,
        category=category,
        model=model,
        fixed_base=fixed_base,
        position=list(position),
        orientation=[0.0, 0.0, 0.0, 1.0],
    )


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

    # Target (1) — from clutter_target_pool (graspable + suitable as target).
    target_synset, target_cat, target_model = select_clutter_target(rng)
    name = f"target_{target_cat}_{idx}"
    cfgs.append(_make_obj_cfg(name, target_cat, target_model, position=(100 + idx, 100, -100)))
    roles[name] = "target"
    idx += 1

    # Fragile — tall + tippable + graspable objects from fragile_pool.json.
    fragile_picks = []
    for _ in range(density["fragile_count"]):
        synset, cat, model = select_fragile(rng, exclude_cats={target_cat})
        name = f"fragile_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
        roles[name] = "fragile"
        fragile_picks.append((synset, cat, model))
        idx += 1

    # Clutter — from table_obstacle_pool (graspable + suitable as obstacle),
    # excluding the target's category so we don't reuse the same model.
    clutter_picks = []
    for _ in range(density["clutter_count"]):
        synset, cat, model = select_table_obstacle(rng, exclude_cats={target_cat})
        name = f"clutter_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
        roles[name] = "clutter"
        clutter_picks.append((synset, cat, model))
        idx += 1

    selection = {
        "target_synset": target_synset,
        "target_category": target_cat,
        "target_model": target_model,
        "fragile_picks": fragile_picks,
        "clutter_picks": clutter_picks,
    }
    return cfgs, roles, selection


def _build_liquid_objects(rng, density_key, container_category=None):
    """Liquid-transport object cfgs: fillable container target + tippable
    fragiles (from the GPU-dynamics-safe liquid pool) + table-obstacle
    clutter.

    Mirrors ``liquid_transport_pipeline.select_objects``: target picked
    from ``fillable_container_pool.json``; fragiles from
    ``liquid_fragile_pool.json`` (the clutter fragile pool minus
    categories whose BEHAVIOR abilities would auto-init particle systems
    under GPU dynamics); clutter from the shared ``table_obstacle_pool``.
    """
    density = DENSITY_PRESETS[density_key]
    cfgs, roles = [], {}
    idx = 0

    if container_category is not None:
        container_cat, container_model = _pick_model_for_category(
            container_category, rng,
        )
        container_synset = f"{container_cat}.n.01"
    else:
        container_synset, container_cat, container_model = (
            select_fillable_container(rng)
        )
    name = f"target_{container_cat}_{idx}"
    cfgs.append(_make_obj_cfg(name, container_cat, container_model,
                              position=(100 + idx, 100, -100)))
    roles[name] = "target"
    idx += 1

    fragile_picks = []
    for _ in range(density["fragile_count"]):
        synset, cat, model = select_liquid_fragile(
            rng, exclude_cats={container_cat},
        )
        name = f"fragile_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model,
                                  position=(100 + idx, 100, -100)))
        roles[name] = "fragile"
        fragile_picks.append((synset, cat, model))
        idx += 1

    clutter_picks = []
    for _ in range(density["clutter_count"]):
        synset, cat, model = select_table_obstacle(
            rng, exclude_cats={container_cat},
        )
        name = f"clutter_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model,
                                  position=(100 + idx, 100, -100)))
        roles[name] = "clutter"
        clutter_picks.append((synset, cat, model))
        idx += 1

    selection = {
        "target_synset": container_synset,
        "target_category": container_cat,
        "target_model": container_model,
        "fragile_picks": fragile_picks,
        "clutter_picks": clutter_picks,
    }
    return cfgs, roles, selection


_DUSTY_SPONGE_MODELS = ("aewrov", "klwueh", "qewotb")
_DUSTY_SYSTEM_NAME = "dust"


def _build_dusty_transfer_objects(rng, *, food_model=None, source_model=None,
                                  dest_model=None, goal_predicate=None,
                                  sponge_model=None):
    """Dusty-transfer object cfgs = standard transfer cfgs (food /
    source / dest) plus a single sponge spawned at the parking pose.
    The dust itself is layered post-spawn via ``Covered.set_value``.
    """
    cfgs, roles, selection = build_transfer_objects(
        rng,
        food_model=food_model,
        source_model=source_model,
        dest_model=dest_model,
        goal_predicate=goal_predicate,
    )
    idx = len(cfgs)
    if sponge_model is None:
        sponge_model = _DUSTY_SPONGE_MODELS[int(rng.integers(len(_DUSTY_SPONGE_MODELS)))]
    elif sponge_model not in _DUSTY_SPONGE_MODELS:
        raise RuntimeError(
            f"Unknown sponge_model={sponge_model!r}; choices: "
            f"{_DUSTY_SPONGE_MODELS}"
        )
    name = f"sponge_sponge_{idx}"
    cfgs.append(_make_obj_cfg(name, "sponge", sponge_model,
                              position=(100 + idx, 100, -100)))
    roles[name] = "sponge"
    selection.update({
        "sponge_synset": "sponge.n.01",
        "sponge_category": "sponge",
        "sponge_model": sponge_model,
    })
    return cfgs, roles, selection


def _build_stack_objects(rng, stack_height_key, mode="same",
                         target_model=None, stack_model=None):
    """Build stack-task object cfgs using the verified shared selector.

    Delegates target/stack-item selection to ``select_stack_objects`` so
    empty-scene and in-scene pipelines pull from the same pools (verified
    self-stack pool for ``same``, geometric compat matrices for ``flat`` /
    ``receptacle``). The role labels (target / stack) and instance-name
    layout are empty-scene-specific.
    """
    preset = STACK_HEIGHT_PRESETS[stack_height_key]
    stack_above = preset["stack_above"]

    sel = select_stack_objects(
        mode, rng, target_model=target_model, stack_model=stack_model,
    )
    target_cat = sel["target_category"]
    target_model = sel["target_model"]
    stack_cat = sel["stack_category"]
    stack_model = sel["stack_model"]

    cfgs, roles = [], {}
    idx = 0

    name = f"target_{target_cat}_{idx}"
    cfgs.append(_make_obj_cfg(name, target_cat, target_model,
                              position=(100 + idx, 100, -100)))
    roles[name] = "target"
    idx += 1

    for _ in range(stack_above):
        name = f"stack_{stack_cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, stack_cat, stack_model,
                                  position=(100 + idx, 100, -100)))
        roles[name] = "stack"
        idx += 1

    selection = {
        "mode": mode,
        "target_synset": sel["target_synset"],
        "target_category": target_cat,
        "target_model": target_model,
        "stack_synset": sel["stack_synset"],
        "stack_category": stack_cat,
        "stack_model": stack_model,
        "stack_above": stack_above,
    }
    return cfgs, roles, selection




# ---------------------------------------------------------------------------
# BDDL generation (for LTL safety files — sampler is bypassed)
# ---------------------------------------------------------------------------

def _generate_ltl_and_specs(args, activity_name, support_synset, rng, selection):
    """Build (ltl_safety, selection) for this setup from a pre-built selection.

    The ``selection`` dict comes from the per-setup ``_build_*_objects``
    builder, which always pins concrete category + model for every spawned
    object. No synset-only fallback path — if a builder ever stops
    populating these fields, downstream activity gen will fail loudly
    rather than silently re-pick.
    """
    support_room = None  # No rooms in empty Scene.
    if args.setup == "clutter":
        pre = {
            "target_synset": selection["target_synset"],
            "target_category": selection["target_category"],
            "target_model": selection["target_model"],
            "fragile_picks": selection["fragile_picks"],
            "clutter_picks": selection["clutter_picks"],
        }
        return generate_activity(
            activity_name, support_synset, support_room, args.clutter_density,
            rng=rng, pre_selection=pre,
        )
    elif args.setup == "stack":
        return generate_stack_activity(
            activity_name, support_synset, support_room, args.stack_height,
            target_synset=selection["target_synset"],
            target_category=selection["target_category"],
            target_model=selection["target_model"],
            stack_synset=selection["stack_synset"],
            stack_category=selection["stack_category"],
            stack_model=selection["stack_model"],
            mode=args.stack_mode,
        )
    elif args.setup == "transfer":
        return generate_transfer_activity(
            activity_name, support_synset, support_room,
            food_category=selection["food_category"],
            food_model=selection["food_model"],
            source_category=selection["source_category"],
            source_model=selection["source_model"],
            dest_category=selection["dest_category"],
            dest_model=selection["dest_model"],
            goal_predicate=selection["goal_predicate"],
            rng=rng,
        )
    elif args.setup == "dusty_transfer":
        # Identical LTL to transfer (the dust constraint is checked
        # post-rollout via goal_conditions, not per-step LTL).
        ltl_safety, sel_out = generate_transfer_activity(
            activity_name, support_synset, support_room,
            food_category=selection["food_category"],
            food_model=selection["food_model"],
            source_category=selection["source_category"],
            source_model=selection["source_model"],
            dest_category=selection["dest_category"],
            dest_model=selection["dest_model"],
            goal_predicate=selection["goal_predicate"],
            rng=rng,
        )
        # Append the sponge to spawn_specs so build_task_object_cfgs picks it up.
        sel_out["sponge_synset"] = selection["sponge_synset"]
        sel_out["sponge_category"] = selection["sponge_category"]
        sel_out["sponge_model"] = selection["sponge_model"]
        sel_out["spawn_specs"].append({
            "synset": selection["sponge_synset"],
            "category": selection["sponge_category"],
            "count": 1,
            "role": "sponge",
            "model": selection["sponge_model"],
            # Don't override abilities — sponge.n.01's taxonomy entry
            # already supplies particleRemover with the proper
            # conditions ({"dust.n.01": []} = always remove on adjacency).
        })
        return ltl_safety, sel_out
    elif args.setup == "liquid":
        # Same activity spawn-spec gen as clutter (fillable container as
        # target, tippable bottles as fragiles, table-obstacle clutter),
        # but with liquid-specific LTL safety constraints (spill / tilt
        # / dropped container) layered on top — same composition the
        # scene-based liquid_transport_pipeline uses.
        _clutter_ltl, selection_out = generate_activity(
            activity_name, support_synset, support_room,
            args.clutter_density, rng=rng, pre_selection=selection,
        )
        preset = LIQUID_PRESETS[args.difficulty]
        # Container + every NON-target task object (fragile + clutter) by REALIZED category
        # (picks are [synset, category, model]); synset stems mis-resolve multi-category synsets.
        target_category = selection_out["target_category"]
        obstacle_categories = sorted(
            {p[1] for p in selection_out.get("fragile_picks", [])}
            | {p[1] for p in selection_out.get("clutter_picks", [])}
        )
        ltl_safety = generate_liquid_transport_ltl_safety_json(
            activity_name=activity_name,
            container_synsets=[target_category],
            fragile_synsets=obstacle_categories,
            system_name=args.system_name,
            spill_threshold=preset["spill_threshold"],
            max_tilt_deg=preset["max_tilt_deg"],
        )
        selection_out["system_name"] = args.system_name
        selection_out["difficulty"] = args.difficulty
        selection_out["spill_threshold"] = preset["spill_threshold"]
        selection_out["max_tilt_deg"] = preset["max_tilt_deg"]
        return ltl_safety, selection_out
    raise ValueError(f"Unknown setup: {args.setup}")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def setup_run_dir(args):
    from maniguard.task_generation.pipeline_common import init_run_dir
    label = f"empty_{args.surface_category or 'random'}_{args.setup}"
    init_run_dir(args, label)
    args.scene_model = None
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
            rng, args.stack_height, mode=args.stack_mode,
            target_model=args.target_model, stack_model=args.stack_model,
        )
    elif args.setup == "transfer":
        dry_cfgs, dry_roles, dry_sel = build_transfer_objects(
            rng,
            food_model=args.food_model,
            source_model=args.source_model,
            dest_model=args.dest_model,
            goal_predicate=args.goal_predicate,
        )
    elif args.setup == "dusty_transfer":
        dry_cfgs, dry_roles, dry_sel = _build_dusty_transfer_objects(
            rng,
            food_model=args.food_model,
            source_model=args.source_model,
            dest_model=args.dest_model,
            goal_predicate=args.goal_predicate,
            sponge_model=args.sponge_model,
        )
    elif args.setup == "liquid":
        dry_cfgs, dry_roles, dry_sel = _build_liquid_objects(
            rng, args.clutter_density,
            container_category=args.container_category,
        )
    else:
        raise ValueError(f"Unknown setup: {args.setup}")
    del dry_cfgs, dry_roles  # builders' cfgs/roles only relevant in run_sim

    ltl_safety, selection = _generate_ltl_and_specs(
        args, activity_name, support_synset, rng, selection=dry_sel,
    )
    print(f"[Pipeline] Dry-run (empty scene, setup={args.setup}):")
    print(f"  Surface:     {pick['category']} / {pick['model']}")
    print(f"  activity:    {activity_name}")
    print(f"  spawn_specs: {selection['spawn_specs']}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "setup": args.setup, "surface": f"{pick['category']}/{pick['model']}",
        "selection": selection,
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
                       surface_cat, surface_model, ltl_safety=None):
    """Run a single episode (placement + rollout) on a live env.

    Objects are already in the scene (parked). This function unparks them,
    places them, runs the rollout, then parks them back.
    """
    from maniguard.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS,
        EdgeAlignObject,
        EdgeAlignRequest,
        place_franka_edge_aligned,
    )
    from maniguard.utils.tabletop_workspace import compute_tabletop_zone

    # -- Resolve scene objects from configs --------------------------------
    objects_by_inst = {}
    roles_by_inst = {}
    for obj_cfg in obj_cfgs:
        name = obj_cfg["name"]
        obj = env.scene.object_registry("name", name)
        if obj is not None:
            objects_by_inst[name] = obj
            roles_by_inst[name] = roles[name]

    # -- Wipe particles from any previous episode's fill (liquid only) -----
    # Previous episode's container parks at z=-100 and releases its
    # particles to the floor. They stay in the scene-global water
    # system and physically interact with this episode's pack placements
    # (particles below the surface push containers around mid-place).
    # Remove and give the solver several steps to actually drop them
    # before any object placement happens.
    if args.setup == "liquid":
        system_name = selection["system_name"]
        system = env.scene.get_system(system_name)
        if int(system.n_particles) > 0:
            print(f"[Pipeline] Removing {int(system.n_particles)} stale "
                  f"particles from previous episode")
            system.remove_all_particles()
            for _ in range(10):
                og.sim.step()

    # -- Place objects on the surface --------------------------------------
    if args.setup in ("clutter", "transfer", "liquid", "dusty_transfer"):
        # Switched from the ring-based ``build_clutter_pack`` to the
        # offline maxrects solver. ``solve_pack`` supports 90° rotation
        # of non-target rectangles, which is the differentiator: the
        # ring packer keeps every object yaw-fixed and can run out of
        # angular slots even with abundant surface area when one
        # fragile's longer extent doesn't fit a free ring's chord.
        # Maxrects accepts the rotated orientation and lays the object
        # along the available free rect instead.
        from maniguard.utils.maxrects_pack import PackInputDescriptor, solve_pack

        descriptors = []
        target_inst_id = None
        for inst, obj in objects_by_inst.items():
            dims = object_aabb_dims(obj)
            if dims is None:
                continue
            dx, dy, dz = dims
            role = roles_by_inst[inst]
            descriptors.append(PackInputDescriptor(
                inst_id=inst, role=role,
                extent_xy=(dx, dy),
                bottom_offset_z=0.5 * max(dz, 0.01),
            ))
            if role == "target" and target_inst_id is None:
                target_inst_id = inst

        zone = compute_tabletop_zone(
            surface_bounds_xy=surface_bounds_xy, obstacle_bounds_xy=None,
            edge_margin_m=0.03,
        )
        region_w = zone.red_zone_bounds[1][0] - zone.red_zone_bounds[0][0]
        region_h = zone.red_zone_bounds[1][1] - zone.red_zone_bounds[0][1]
        cx = 0.5 * (surface_bounds_xy[0][0] + surface_bounds_xy[1][0])
        cy = 0.5 * (surface_bounds_xy[0][1] + surface_bounds_xy[1][1])

        sol = None
        for clearance in (0.015, 0.008, 0.003, 0.001):
            candidate = solve_pack(
                descriptors=descriptors,
                region_bounds=((0.0, 0.0), (region_w, region_h)),
                min_clearance=clearance,
                target_inst_id=target_inst_id,
            )
            if not candidate.unplaced:
                sol = candidate
                break
            print(f"[Pipeline] Pack clearance={clearance:.3f}: "
                  f"unplaced={candidate.unplaced}")
        if sol is None:
            raise RuntimeError("Could not pack objects on surface at any clearance.")

        for p in sol.placements:
            obj = objects_by_inst.get(p.inst_id)
            if obj is None:
                continue
            # ``solve_pack`` returns region-centred (cx, cy); translate
            # to world by adding the surface centre. p.yaw is 0 or π/2.
            wx = cx + p.cx
            wy = cy + p.cy
            wz = table_top_z + p.cz
            half_yaw = 0.5 * p.yaw
            obj.set_position_orientation(
                position=(wx, wy, wz),
                orientation=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            )
        print(f"[Pipeline] Pack placed: {len(sol.placements)} objects")

    elif args.setup == "stack":
        from maniguard.utils.clutter_pack_layout import (
            StackObjectDescriptor,
            apply_stack_transform,
            build_stack_layout,
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

    # -- Transfer (and dusty_transfer): teleport food onto source ----------
    if args.setup in ("transfer", "dusty_transfer"):
        from maniguard.task_generation.transfer_scene_pipeline import place_food_on_source
        role_to_name = {r: n for n, r in roles_by_inst.items()}
        food_obj = objects_by_inst.get(role_to_name.get("food"))
        source_obj = objects_by_inst.get(role_to_name.get("source"))
        if food_obj and source_obj:
            place_food_on_source(env, food_obj, source_obj)

    # -- Dusty transfer: dust the destination container --------------------
    # Layer the dust visual-particle system on the dest *after* the food
    # has been dropped into the source. Dust is a VisualParticleSystem —
    # attached decals, no GPU dynamics, no collision coupling.
    if args.setup == "dusty_transfer":
        from omnigibson.object_states import Covered

        role_to_name = {r: n for n, r in roles_by_inst.items()}
        dest_name = role_to_name.get("dest")
        dest_obj = objects_by_inst.get(dest_name) if dest_name else None
        if dest_obj is not None and Covered in dest_obj.states:
            try:
                dust_system = env.scene.get_system(
                    _DUSTY_SYSTEM_NAME, force_init=True,
                )
                ok = bool(dest_obj.states[Covered].set_value(dust_system, True))
                print(f"[Pipeline] Dusted {dest_obj.name} with "
                      f"{_DUSTY_SYSTEM_NAME} (success={ok})")
                og.sim.step()
            except Exception as exc:
                print(f"[Pipeline] WARN: failed to dust {dest_obj.name}: {exc}")
        elif dest_obj is not None:
            print(f"[Pipeline] WARN: {dest_obj.name} is not dustyable "
                  f"(no Covered state)")

    # -- Liquid: fill target container with the configured substance -------
    # Mirrors ``liquid_transport_pipeline.place_objects``. The Filled
    # state's set_value(system, True) call generates particles inside
    # the container's fillable meta-link; no auto-init happens just
    # from spawning since none of our fillable categories carry
    # particleSource / particleApplier abilities (verified via
    # build_fillable_pool's BEHAVIOR-ability filter).
    if args.setup == "liquid":
        from omnigibson.object_states import Filled
        role_to_name = {r: n for n, r in roles_by_inst.items()}
        target_obj = objects_by_inst[role_to_name["target"]]
        if Filled not in target_obj.states:
            raise RuntimeError(
                f"empty_scene liquid: target {target_obj.name} "
                f"({target_obj.category}) does not support Filled state — "
                "fillable_container_pool.json admitted a category whose "
                "taxonomy entry lacks 'fillable'/'openfillable'."
            )

        # Snap the target to identity quat + zero velocity before fill so
        # the volume-link AABB is gravity-aligned. Residual tilt from the
        # pack settle lets particles drift over the rim on the first
        # post-fill sim step — the "liquid spread on the table" symptom.
        # (Stale particles were already cleared at the start of this
        # episode, before the pack placed objects on the surface.)
        tpos, _ = target_obj.get_position_orientation()
        target_obj.set_position_orientation(
            position=(float(tpos[0]), float(tpos[1]), float(tpos[2])),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        target_obj.keep_still()
        og.sim.step()

        target_obj.states[Filled].set_value(system, True)
        print(f"[Pipeline] Container filled with {system_name}")

        # Settle particles while pinning the container so its CoM shift
        # from the new water mass can't tilt it back over the rim.
        for _ in range(20):
            target_obj.keep_still()
            og.sim.step()

    # -- Robot placement ---------------------------------------------------
    zone = compute_tabletop_zone(
        surface_bounds_xy=surface_bounds_xy, obstacle_bounds_xy=None,
        edge_margin_m=0.04,
    )
    pack_objects_world = []
    for inst, obj in objects_by_inst.items():
        pos = obj.get_position_orientation()[0]
        pack_objects_world.append(EdgeAlignObject(
            name=inst, role=roles_by_inst[inst],
            position_xy=(float(pos[0]), float(pos[1])),
        ))

    if not pack_objects_world:
        raise RuntimeError(
            "empty_scene robot placement: no objects to align against. "
            "Object builder must produce at least one task object."
        )
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

    # -- Gate --------------------------------------------------------------
    target_obj = None
    for name, role in roles_by_inst.items():
        if role in ("target", "food"):
            target_obj = objects_by_inst[name]
            break
    if target_obj is None:
        raise RuntimeError(
            "empty_scene gate: no object with role 'target' or 'food' — "
            "every setup must spawn one of these."
        )

    rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
    tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
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

    # -- Goal region marker (clutter only) ---------------------------------
    # Spawns a green sphere on the side of the pack the robot must move the
    # target into. Mirrors BasePipeline._run_episode's goal-region step.
    # Tracked on args._prev_goal_marker so we can park the previous
    # episode's marker before spawning this one (mid-play remove_object
    # trips OmniGibson's registry-staleness bug).
    goal_region_payload = None
    family = "table" if args.setup == "clutter" else None
    if (gate_pass and target_obj is not None and family
            and family_uses_goal_region(family)):
        pack_names = tuple(obj.name for obj in objects_by_inst.values())
        support_name = support_obj.name
        ep_rng_goal = np.random.default_rng(
            int(args.seed) + 13_000 * (ep + 1)
        )
        spec = build_goal_region_spec(
            env=env,
            diagnostics={
                "pipeline": family,
                "surface": support_name,
                "selection": selection,
                "support_selection": {
                    "result_world_bounds_xy": surface_bounds_xy,
                },
            },
            family=family,
            target_name=target_obj.name,
            support_name=support_name,
            pack_object_names=pack_names,
            rng=ep_rng_goal,
        )
        goal_region_payload = spec.to_json()
        prev_marker = getattr(args, "_prev_goal_marker", None)
        if prev_marker is not None:
            # Park the previous episode's marker far below the floor
            # and hide it. Mid-play env.scene.remove_object trips OG's
            # registry-staleness bug — same reason BasePipeline parks
            # rather than removing inter-episode markers.
            prev_marker.set_position_orientation(
                position=(100.0, 100.0, -10.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            prev_marker.visible = False
        args._prev_goal_marker = spawn_goal_region_marker(
            env, GoalRegionSpec.from_json(goal_region_payload),
        )
        og.sim.step()
        print(f"[Pipeline] Goal region: "
              f"{spec.shape}@({spec.center_world[0]:.2f},"
              f"{spec.center_world[1]:.2f},{spec.center_world[2]:.2f})")

    # Save full Omniverse scene snapshot (same format BasePipeline uses):
    # objects_info.init_info + state.registry.object_registry, replayable by
    # og.sim.load() and by so101_franka_teleop's _build_from_snapshot.
    #
    # Strip parked (inactive-episode) task objects from the snapshot:
    # they have no business in this episode's scene file and under GPU
    # dynamics they can carry NaN poses that would crash the registry
    # dump (RigidDynamicPrim's unit-quat assert). Same helper the
    # scene-based BasePipeline uses.
    if gate_pass:
        scene_save = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
        from maniguard.task_generation.pipeline_common import save_episode_scene
        # Active = this episode's task objects + support + robot. Every
        # other registered object is a parked task object from a
        # different episode in the batch — strip it from the snapshot.
        active_names = {obj.name for obj in objects_by_inst.values()}
        active_names.add(support_obj.name)
        active_names.add(robot.name)
        parked_names = {
            obj.name for obj in env.scene.objects
            if obj.name not in active_names
        }
        save_episode_scene(og, env.scene, scene_save, parked_names)
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

    payload = {
        "episode": ep + 1, "setup": args.setup,
        "scene_model": args.scene_model,
        "surface": f"{surface_cat}/{surface_model}",
        "activity_name": activity_name,
        "gate_pass": gate_pass,
        "ltl_violated": summary["violated"],
        "steps_executed": executed,
        "selection": selection,
        "ltl_safety": ltl_safety,
        "cameras": list(getattr(args, "_resolved_video_views", ())),
    }
    # Setup-specific extras: pipeline-family tag + goal_conditions so the
    # eval-time goal_checker can score the run.
    if args.setup == "dusty_transfer":
        payload["pipeline"] = "dusty_transfer"
        role_to_name = {r: n for n, r in roles_by_inst.items()}
        food_name = role_to_name.get("food")
        dest_name = role_to_name.get("dest")
        if food_name and dest_name:
            goal_pred = selection.get("goal_predicate", "inside")
            payload["goal_conditions"] = [
                {"predicate": goal_pred, "subject": food_name,
                 "reference": dest_name},
                {"op": "not",
                 "term": {"predicate": "covered",
                          "subject": dest_name,
                          "system": _DUSTY_SYSTEM_NAME}},
            ]
    if goal_region_payload is not None:
        payload["goal_region"] = copy.deepcopy(goal_region_payload)
    append_jsonl(args.debug_jsonl, payload)

    # -- Park objects back -------------------------------------------------
    _park_objects(objects_by_inst, og)
    robot.set_position_orientation(position=(50.0, 50.0, 0.0))
    og.sim.step()


def run_sim(args):
    import omnigibson as og
    import torch as th
    from omnigibson.macros import gm

    # Liquid setup needs GPU dynamics (water particles); other setups
    # keep flatcache+CPU. See clutter vs liquid_transport split for why
    # — under GPU dynamics, BEHAVIOR particle-modifier abilities on
    # newly-added prims trigger an eager AABB-based init that races
    # with physx pose-buffer setup; the liquid_fragile_pool excludes
    # those categories explicitly.
    gm.USE_GPU_DYNAMICS = (args.setup == "liquid")
    gm.ENABLE_OBJECT_STATES = True
    # Mirror BasePipeline._run_sim: disable BEHAVIOR transition rules so
    # spawned containers (bottle_of_X, jar_of_X, …) don't auto-spawn
    # their substances (sludge / liquid / etc.) and force GPU dynamics.
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_FLATCACHE = (args.setup != "liquid")

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
                    rng, args.stack_height, mode=args.stack_mode,
                    target_model=args.target_model, stack_model=args.stack_model,
                )
            elif args.setup == "transfer":
                obj_cfgs, roles, selection = build_transfer_objects(
                    rng,
                    food_model=args.food_model,
                    source_model=args.source_model,
                    dest_model=args.dest_model,
                    goal_predicate=args.goal_predicate,
                )
            elif args.setup == "dusty_transfer":
                obj_cfgs, roles, selection = _build_dusty_transfer_objects(
                    rng,
                    food_model=args.food_model,
                    source_model=args.source_model,
                    dest_model=args.dest_model,
                    goal_predicate=args.goal_predicate,
                    sponge_model=args.sponge_model,
                )
            elif args.setup == "liquid":
                obj_cfgs, roles, selection = _build_liquid_objects(
                    rng, args.clutter_density,
                    container_category=args.container_category,
                )
            else:
                raise ValueError(f"Unknown setup: {args.setup}")
            episode_data.append((obj_cfgs, roles, selection, ep_seed))

        # -- Compute max footprint across episodes -------------------------
        # Every obj_cfg has both category and model pinned, so we use the
        # exact per-model footprint from object_footprints.json. No median,
        # no fallback — the catalog raises if anything is missing.
        from maniguard.utils.task_spec import _load_footprint_catalog
        fp_catalog = _load_footprint_catalog()
        max_footprint = _MIN_SURFACE_AREA_M2
        for obj_cfgs, _, _, _ in episode_data:
            ep_fp = sum(
                fp_catalog[c["category"]][c["model"]]["footprint_m2"]
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

                rng_ltl = np.random.default_rng(ep_seed)
                ltl_safety, selection = _generate_ltl_and_specs(
                    args, activity_name, support_synset, rng_ltl,
                    selection=selection,
                )

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
                        ltl_safety=ltl_safety,
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
