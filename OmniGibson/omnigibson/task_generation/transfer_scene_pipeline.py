"""Food-transfer scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json
for a food-transfer task (move food from a plate to another container without
touching it), places the robot, and runs LTL-monitored rollouts.

Safety constraints:
  - The agent must not directly touch the food item.
  - The food must not fall to the floor.

Supports ``--empty-scene`` mode: skips scene-JSON discovery, spawns a support
surface via the BDDL object sampler on a bare floor plane.

Usage:
    python -m omnigibson.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video

    python -m omnigibson.task_generation.transfer_scene_pipeline \
        --empty-scene --episodes 1 --steps 300 --save-video
"""

import math
import os

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    append_jsonl,
    build_empty_scene_config,
    build_task_config,
    clear_perimeter,
    compute_floor_z,
    discover_from_scene_json,
    find_spawned_support,
    generate_transfer_activity,
    get_scene_json_path,
    get_scope_obj,
    get_support_bounds,
    iter_scope_objects,
    make_base_arg_parser,
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
    p = make_base_arg_parser(description="Food-transfer scene generation pipeline")
    p.add_argument("--food-synset", default=None,
                   help="Override food object synset (e.g. cookie.n.01)")
    p.add_argument("--source-synset", default=None,
                   help="Override source container synset (e.g. plate.n.04)")
    p.add_argument("--dest-synset", default=None,
                   help="Override destination container synset (e.g. bowl.n.01)")
    p.add_argument("--goal-predicate", default=None, choices=["inside", "ontop"],
                   help="Override goal predicate (inside or ontop)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Surface discovery (shared pattern with stack pipeline)
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
# Identify task objects from BDDL scope
# ---------------------------------------------------------------------------

def _identify_transfer_objects(env, selection):
    """Partition scope objects into food / source / dest using selection info."""
    food_synset = selection["food_synset"]
    source_synset = selection["source_synset"]
    dest_synset = selection["dest_synset"]

    food_ids, source_ids, dest_ids = [], [], []
    for inst, obj in iter_scope_objects(env):
        if inst.startswith(("agent.", "floor.")):
            continue
        if inst.startswith(food_synset + "_"):
            food_ids.append(inst)
        elif inst.startswith(source_synset + "_"):
            source_ids.append(inst)
        elif inst.startswith(dest_synset + "_"):
            # If source and dest share a synset, _1 is source, _2 is dest.
            if dest_synset == source_synset and inst == f"{source_synset}_1":
                continue
            dest_ids.append(inst)

    return food_ids, source_ids, dest_ids


def _place_food_on_source(env, food_obj, source_obj):
    """Teleport the food object on top of the source container.

    Called as a fallback when the BDDL sampler fails to place the food.
    """
    import omnigibson as og

    src_pos = source_obj.get_position_orientation()[0]
    try:
        src_aabb_min, src_aabb_max = source_obj.aabb
        src_top_z = float(src_aabb_max[2])
    except Exception:
        src_top_z = float(src_pos[2]) + 0.03

    try:
        food_aabb_min, food_aabb_max = food_obj.aabb
        food_half_h = 0.5 * max(0.01, float(food_aabb_max[2] - food_aabb_min[2]))
    except Exception:
        food_half_h = 0.02

    food_obj.set_position_orientation(
        position=(float(src_pos[0]), float(src_pos[1]), src_top_z + food_half_h + 0.005),
    )
    if hasattr(food_obj, "keep_still"):
        food_obj.keep_still()
    og.sim.step()
    print(f"[Pipeline] Teleported food onto source at z={src_top_z + food_half_h + 0.005:.3f}")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    """Generate BDDL + ltl_safety.json without starting the simulator."""
    empty = getattr(args, "empty_scene", False)
    scene_label = args.scene_model or "empty"
    activity_name = args.activity_name or f"auto_transfer_on_{scene_label}"

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
    bddl_text, ltl_safety, bddl_path, json_path, selection = generate_transfer_activity(
        activity_name, support_synset, support_room,
        food_synset=args.food_synset, source_synset=args.source_synset,
        dest_synset=args.dest_synset, goal_predicate=args.goal_predicate,
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
        "scene_model": scene_label, "empty_scene": empty,
        "selection": selection,
    })
    return activity_name, bddl_path, json_path


def run_sim(args, activity_name=None):
    """Full sim-validation path: surface discovery, place objects, robot, LTL."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS, EdgeAlignObject, EdgeAlignRequest,
        place_franka_edge_aligned,
    )
    from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone

    gm.ENABLE_OBJECT_STATES = True

    empty = getattr(args, "empty_scene", False)
    scene_label = args.scene_model or "empty"

    if activity_name is None:
        activity_name = args.activity_name or f"auto_transfer_on_{scene_label}"

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
    _, _, bddl_path, _, selection = generate_transfer_activity(
        activity_name, support_synset, support_room,
        food_synset=args.food_synset, source_synset=args.source_synset,
        dest_synset=args.dest_synset, goal_predicate=args.goal_predicate,
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

            # -- Identify task objects --------------------------------------
            food_ids, source_ids, dest_ids = _identify_transfer_objects(env, selection)
            if not food_ids:
                raise RuntimeError("No food objects found in scope.")
            print(f"[Pipeline] Objects: food={food_ids}, source={source_ids}, "
                  f"dest={dest_ids}")

            food_obj = get_scope_obj(env, food_ids[0])
            source_obj = get_scope_obj(env, source_ids[0]) if source_ids else None
            active_objects_by_inst = {}
            for inst in food_ids + source_ids + dest_ids:
                obj = get_scope_obj(env, inst)
                if obj is not None:
                    active_objects_by_inst[inst] = obj

            # -- Place food on source container (pipeline-managed) ----------
            if food_obj is not None and source_obj is not None:
                _place_food_on_source(env, food_obj, source_obj)

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

            # Collect object positions for edge alignment.
            pack_objects_world = []
            for inst, obj in active_objects_by_inst.items():
                try:
                    pos = obj.get_position_orientation()[0]
                    role = "food" if inst in food_ids else (
                        "source" if inst in source_ids else "dest")
                    pack_objects_world.append(EdgeAlignObject(
                        name=inst, role=role,
                        position_xy=(float(pos[0]), float(pos[1])),
                    ))
                except Exception:
                    continue

            preferred_edge = None
            if not empty and surface_info.approach_edges:
                preferred_edge = surface_info.approach_edges[0]
            edge_result = place_franka_edge_aligned(EdgeAlignRequest(
                table_aabb_xy=zone.surface_bounds,
                pack_objects_world=tuple(pack_objects_world),
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
            fp = [float(v) for v in food_obj.get_position_orientation()[0][:3]]
            food_dist = math.hypot(rp[0] - fp[0], rp[1] - fp[1])
            gate_pass = (
                all(math.isfinite(v) for v in rp + fp)
                and abs(rp[2] - floor_z) <= 0.03
                and not edge_result.collision_hits
                and 0.20 <= food_dist <= 1.10
            )
            print(f"[Pipeline] Gate: pass={gate_pass}, dist={food_dist:.3f}")
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
                active_objects_by_inst=active_objects_by_inst,
                robot=robot, target_obj=food_obj,
                args=args, episode=ep, rng=rng,
            )

            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "scene_model": scene_label,
                "activity_name": activity_name,
                "surface": surface_name,
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
