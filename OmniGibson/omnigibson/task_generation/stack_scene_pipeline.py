"""Stack retrieval scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json
for a stack-retrieval task (get the target from under a stack), teleports objects
into a vertical stack, settles physics, validates OnTop state, and runs
LTL-monitored rollouts.

Supports ``--empty-scene`` mode: skips scene-JSON discovery, spawns a support
surface via the BDDL object sampler on a bare floor plane.

Usage:
    python -m omnigibson.task_generation.stack_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.stack_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 \
        --stack-height medium --save-video

    python -m omnigibson.task_generation.stack_scene_pipeline \
        --empty-scene --episodes 1 --steps 300 --stack-height medium --save-video
"""

import math
import os

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    STACK_HEIGHT_PRESETS,
    append_jsonl,
    build_empty_scene_config,
    build_task_config,
    clear_perimeter,
    compute_floor_z,
    discover_from_scene_json,
    find_spawned_support,
    generate_stack_activity,
    get_scene_json_path,
    get_scope_obj,
    get_support_bounds,
    iter_scope_objects,
    make_base_arg_parser,
    make_settle_fn,
    needs_gpu_dynamics,
    pipeline_exit,
    refresh_activity_cache,
    resolve_synset,
    robot_half_extent_xy,
    run_ltl_rollout,
    setup_run_dir,
)

_SURFACE_CATEGORY_PRIORITY = {
    "breakfast_table": 3, "dining_table": 3, "conference_table": 3,
    "commercial_kitchen_table": 3, "lab_table": 3,
    "coffee_table": 2, "garden_coffee_table": 2, "pedestal_table": 2,
    "pool_table": 2, "flat_bench": 2,
    "desk": 1, "reception_desk": 1, "counter": 1, "countertop": 1,
    "checkout_counter": 1, "console_table": 1, "nightstand": 1,
}


def parse_args():
    p = make_base_arg_parser(description="Stack retrieval scene generation pipeline")
    p.add_argument("--stack-height", default="medium",
                   choices=list(STACK_HEIGHT_PRESETS),
                   help="Number of objects stacked on top of the target")
    p.add_argument("--target-synset", default=None,
                   help="Override target object synset")
    p.add_argument("--stack-synset", default=None,
                   help="Override stack object synset")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Table-specific discovery (shared with clutter pipeline)
# ---------------------------------------------------------------------------

def _discover_surface_from_scene_json(scene_json_path):
    from omnigibson.utils.surface_discovery import is_table_like
    return discover_from_scene_json(scene_json_path, is_table_like, _SURFACE_CATEGORY_PRIORITY)


def _discover_best_surface(env):
    """Find best table-like surface in loaded scene."""
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
# Stack layout helpers
# ---------------------------------------------------------------------------

def _build_stack_descriptors(env, target_ids, stack_ids):
    """Build StackObjectDescriptors from live env objects, ordered bottom-to-top.

    Order: target (bottom of the "retrieve" stack), then stack objects above it.
    """
    from omnigibson.utils.clutter_pack_layout import StackObjectDescriptor

    descriptors = []
    for inst, role in [(tid, "target") for tid in target_ids] + \
                      [(sid, "stack") for sid in stack_ids]:
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
        descriptors.append(StackObjectDescriptor(
            instance_id=inst, role=role,
            half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
        ))
    return descriptors


def _identify_stack_objects(env, selection):
    """Partition scope objects into target / stack / support using selection info.

    The *selection* dict (from generate_stack_activity) contains:
        target_synset, stack_synset, stack_above
    Instance IDs in the BDDL follow the pattern ``<synset>_<N>``.
    """
    target_synset = selection["target_synset"]
    stack_synset = selection["stack_synset"]

    target_ids, stack_ids = [], []
    for inst, obj in iter_scope_objects(env):
        if inst.startswith(("agent.", "floor.")):
            continue
        # Match by synset prefix in instance id.
        if inst.startswith(target_synset + "_"):
            target_ids.append(inst)
        elif inst.startswith(stack_synset + "_"):
            # When target and stack share the same synset, the target is
            # always _1 (first instance).  Skip it here to avoid double-counting.
            if stack_synset == target_synset and inst == f"{target_synset}_1":
                continue
            stack_ids.append(inst)

    # Sort stack by instance number so ordering is deterministic.
    stack_ids.sort(key=lambda s: int(s.rsplit("_", 1)[-1]))
    return target_ids, stack_ids


def _validate_ontop_state(env, stack_descriptors, support_obj, objects_by_inst):
    """Check that each object in the stack is OnTop of the one below it."""
    from omnigibson.object_states.on_top import OnTop

    chain = []
    for desc in stack_descriptors:
        obj = objects_by_inst.get(desc.instance_id)
        if obj is not None:
            chain.append((desc.instance_id, obj))

    if not chain:
        return False, "empty stack"

    # Bottom object should be on the support surface.
    bottom_inst, bottom_obj = chain[0]
    try:
        on_support = bottom_obj.states[OnTop].get_value(support_obj)
    except Exception:
        on_support = False
    if not on_support:
        return False, f"{bottom_inst} not OnTop support"

    # Each subsequent object should be on the one below.
    for i in range(1, len(chain)):
        upper_inst, upper_obj = chain[i]
        lower_inst, lower_obj = chain[i - 1]
        try:
            on_lower = upper_obj.states[OnTop].get_value(lower_obj)
        except Exception:
            on_lower = False
        if not on_lower:
            return False, f"{upper_inst} not OnTop {lower_inst}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    """Generate BDDL + ltl_safety.json without starting the simulator."""
    empty = getattr(args, "empty_scene", False)
    scene_label = args.scene_model or "empty"
    activity_name = args.activity_name or f"auto_stack_on_{scene_label}"

    if empty:
        support_synset = args.surface_synset
        support_room = None  # No rooms in empty Scene — skip inroom predicate.
        print(f"[Pipeline] Empty-scene mode: surface={support_synset}")
    else:
        support_synset, support_room = "breakfast_table.n.01", "living_room"
        try:
            scene_json = get_scene_json_path(args.scene_model)
            discovery = _discover_surface_from_scene_json(scene_json)
            if discovery:
                support_synset = resolve_synset(discovery[0])
                support_room = discovery[1]
                print(f"[Pipeline] Discovered: {discovery[0]} in {support_room}")
        except Exception as e:
            print(f"[Pipeline] Surface discovery failed: {e}")

    rng = np.random.default_rng(args.seed)
    bddl_text, ltl_safety, bddl_path, json_path, selection = generate_stack_activity(
        activity_name, support_synset, support_room, args.stack_height,
        target_synset=args.target_synset, stack_synset=args.stack_synset,
        rng=rng,
    )
    print(f"[Pipeline] Dry-run complete:")
    print(f"  BDDL:       {bddl_path}")
    print(f"  ltl_safety: {json_path}")
    print(f"  activity:   {activity_name}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "scene_model": scene_label, "stack_height": args.stack_height,
        "empty_scene": empty, "selection": selection,
    })
    return activity_name, bddl_path, json_path


def run_sim(args, activity_name=None):
    """Full sim-validation path: surface discovery, stack, robot, gate, LTL."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.clutter_pack_layout import (
        build_stack_layout, apply_stack_transform,
    )
    from omnigibson.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS, EdgeAlignObject, EdgeAlignRequest,
        place_franka_edge_aligned,
    )
    from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone

    gm.ENABLE_OBJECT_STATES = True

    empty = getattr(args, "empty_scene", False)
    scene_label = args.scene_model or "empty"

    if activity_name is None:
        activity_name = args.activity_name or f"auto_stack_on_{scene_label}"

    # -- Resolve support surface --------------------------------------------
    if empty:
        support_synset = args.surface_synset
        support_room = None  # No rooms in empty Scene — skip inroom predicate.
        print(f"[Pipeline] Empty-scene mode: surface={support_synset}")
    else:
        scene_json = get_scene_json_path(args.scene_model)
        if not os.path.isfile(scene_json):
            raise RuntimeError(f"Scene JSON not found: {scene_json}")
        discovery = _discover_surface_from_scene_json(scene_json)
        if discovery is None:
            raise RuntimeError(f"No table-like surface in scene '{args.scene_model}'.")
        surface_category = discovery[0]
        support_synset = resolve_synset(surface_category)
        support_room = discovery[1]
        print(f"[Pipeline] Discovered: category={surface_category} synset={support_synset} "
              f"room={support_room}")

    # -- Generate BDDL ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    _, _, bddl_path, _, selection = generate_stack_activity(
        activity_name, support_synset, support_room, args.stack_height,
        target_synset=args.target_synset, stack_synset=args.stack_synset,
        rng=rng,
    )
    print(f"[Pipeline] Generated BDDL: {bddl_path}")
    refresh_activity_cache()

    # -- GPU dynamics --------------------------------------------------------
    gpu = needs_gpu_dynamics(activity_name)
    gm.USE_GPU_DYNAMICS = gpu
    gm.ENABLE_FLATCACHE = not gpu

    # -- Load environment ----------------------------------------------------
    if empty:
        cfg = build_empty_scene_config(activity_name)
    else:
        cfg = build_task_config(args.scene_model, activity_name)
        cfg["scene"]["scene_file"] = scene_json
        cfg["scene"]["scene_instance"] = None
    cfg["task"]["online_object_sampling"] = True
    cfg["task"]["use_presampled_robot_pose"] = False

    # Pin all stack/target instances to the same model so they stack reliably.
    if selection.get("sampling_whitelist"):
        cfg["task"]["sampling_whitelist"] = selection["sampling_whitelist"]

    print(f"[Pipeline] scene={scene_label}, activity={activity_name}, "
          f"empty={empty}, strict_gate={args.strict_gate}")
    env = og.Environment(configs=cfg)

    try:
        for ep in range(args.episodes):
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
            env.reset()
            og.sim.step()

            # -- Surface discovery ------------------------------------------
            if empty:
                support_inst, support_obj = find_spawned_support(env, support_synset)
                if support_obj is None:
                    raise RuntimeError(
                        f"Support surface '{support_synset}' was not spawned by BDDL sampler.")
                surface_bounds_xy, table_top_z = get_support_bounds(support_obj)
                floor_z = 0.0
                surface_name = support_inst
                print(f"[Pipeline] Spawned surface: {support_inst}, top_z={table_top_z:.3f}")
            else:
                surface_info, support_obj = _discover_best_surface(env)
                print(f"[Pipeline] Best surface: {surface_info.surface.name} "
                      f"(score={surface_info.surface.score:.3f})")
                aabb_min, aabb_max = support_obj.aabb
                surface_bounds_xy = (
                    (float(aabb_min[0]), float(aabb_min[1])),
                    (float(aabb_max[0]), float(aabb_max[1])),
                )
                table_top_z = float(aabb_max[2])
                floor_z = compute_floor_z(env)
                surface_name = surface_info.surface.name
                clear_perimeter(env, support_obj, surface_bounds_xy, table_top_z, floor_z)

            # -- Identify stack objects -------------------------------------
            target_ids, stack_ids = _identify_stack_objects(env, selection)
            if not target_ids:
                raise RuntimeError("No target objects found in scope.")
            print(f"[Pipeline] Objects: target={target_ids}, stack={stack_ids}")

            target_obj = get_scope_obj(env, target_ids[0])
            objects_by_inst = {}
            for inst in target_ids + stack_ids:
                obj = get_scope_obj(env, inst)
                if obj is not None:
                    objects_by_inst[inst] = obj

            # -- Build stack descriptors (bottom-to-top) --------------------
            stack_descriptors = _build_stack_descriptors(env, target_ids, stack_ids)
            if len(stack_descriptors) < 2:
                raise RuntimeError(f"Need at least 2 objects for a stack, "
                                   f"got {len(stack_descriptors)}.")

            # -- Compute stack origin (centre of table surface) -------------
            cx = 0.5 * (surface_bounds_xy[0][0] + surface_bounds_xy[1][0])
            cy = 0.5 * (surface_bounds_xy[0][1] + surface_bounds_xy[1][1])
            stack_origin = (cx, cy, table_top_z)

            # -- Build and apply stack layout -------------------------------
            ep_seed = args.seed + ep * 1000
            stack_spec = build_stack_layout(
                support_obj_name=getattr(support_obj, "name", "support"),
                descriptors=stack_descriptors,
                seed=ep_seed,
            )
            placements = apply_stack_transform(
                stack_spec, objects_by_inst, stack_origin,
            )
            print(f"[Pipeline] Stack placed: {len(placements)} objects at "
                  f"origin=({cx:.3f}, {cy:.3f}, {table_top_z:.3f})")

            # -- Keep still + settle physics --------------------------------
            for obj in objects_by_inst.values():
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            settle_fn = make_settle_fn(og, th)
            settle_fn(objects_by_inst)

            # -- Validate OnTop chain ---------------------------------------
            max_ontop_attempts = 3
            ontop_ok = False
            for attempt in range(max_ontop_attempts):
                ontop_ok, ontop_msg = _validate_ontop_state(
                    env, stack_descriptors, support_obj, objects_by_inst,
                )
                if ontop_ok:
                    print(f"[Pipeline] OnTop validation: OK (attempt {attempt + 1})")
                    break
                print(f"[Pipeline] OnTop validation failed (attempt {attempt + 1}): "
                      f"{ontop_msg}")
                # Re-place with a different seed and settle again.
                ep_seed += 1
                stack_spec = build_stack_layout(
                    support_obj_name=getattr(support_obj, "name", "support"),
                    descriptors=stack_descriptors,
                    seed=ep_seed,
                )
                placements = apply_stack_transform(
                    stack_spec, objects_by_inst, stack_origin,
                )
                for obj in objects_by_inst.values():
                    if hasattr(obj, "keep_still"):
                        obj.keep_still()
                settle_fn(objects_by_inst)

            # -- Robot placement --------------------------------------------
            robot = env.robots[0]

            obstacle_bounds_xy = None
            if not empty and surface_info.obstacles:
                obstacle_bounds_xy = surface_info.obstacles[0].aabb_xy

            zone = compute_tabletop_zone(
                surface_bounds_xy=surface_bounds_xy,
                obstacle_bounds_xy=obstacle_bounds_xy,
                edge_margin_m=0.04,
                obstacle_keepout_margin_m=0.08 if not empty else 0.0,
                obstacle_side_clearance_m=0.015 if not empty else 0.0,
            )

            pack_objects_world = tuple(
                EdgeAlignObject(
                    name=inst, role="target" if inst in target_ids else "stack",
                    position_xy=(placements[inst][0], placements[inst][1]),
                )
                for inst in placements
            )
            preferred_edge = None
            if not empty and surface_info.approach_edges:
                preferred_edge = surface_info.approach_edges[0]
            edge_result = place_franka_edge_aligned(EdgeAlignRequest(
                table_aabb_xy=zone.surface_bounds,
                pack_objects_world=pack_objects_world,
                role_weights=DEFAULT_ROLE_WEIGHTS,
                robot_half_extent_xy=robot_half_extent_xy(robot),
                edge_gap_m=args.mount_gap_m, edge_margin_m=0.05,
                scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10,
                                0.15, -0.15, 0.20, -0.20),
                preferred_edge=preferred_edge,
            ))
            robot.set_position_orientation(
                position=(edge_result.base_pose["position"][0],
                          edge_result.base_pose["position"][1], floor_z),
                orientation=edge_result.base_pose["orientation"],
            )
            og.sim.step()
            print(f"[Pipeline] Robot: edge={edge_result.edge_label}, "
                  f"gap={edge_result.gap_actual:.3f}")

            # -- Gate -------------------------------------------------------
            rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
            tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
            target_dist = math.hypot(rp[0] - tp[0], rp[1] - tp[1])
            gate_pass = (
                all(math.isfinite(v) for v in rp + tp)
                and abs(rp[2] - floor_z) <= 0.03
                and not edge_result.collision_hits
                and 0.20 <= target_dist <= 1.10
                and ontop_ok
            )
            print(f"[Pipeline] Gate: pass={gate_pass}, ontop={ontop_ok}, "
                  f"dist={target_dist:.3f}")
            if args.strict_gate and not gate_pass:
                raise RuntimeError("Strict gate failed.")

            # -- Save scene snapshot ----------------------------------------
            if gate_pass:
                scene_save_path = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
                og.sim.save(json_paths=[scene_save_path])
                print(f"[Pipeline] Scene saved: {scene_save_path}")

            # -- LTL rollout ------------------------------------------------
            summary, executed = run_ltl_rollout(
                env=env, activity_name=activity_name,
                scene_model=scene_label,
                active_objects_by_inst=objects_by_inst,
                robot=robot, target_obj=target_obj,
                args=args, episode=ep, rng=rng,
            )

            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "scene_model": scene_label,
                "activity_name": activity_name,
                "surface": surface_name,
                "stack_height": args.stack_height,
                "ontop_valid": ontop_ok,
                "gate_pass": gate_pass, "ltl_violated": summary["violated"],
                "steps_executed": executed, "empty_scene": empty,
                "selection": selection,
            })

    finally:
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
