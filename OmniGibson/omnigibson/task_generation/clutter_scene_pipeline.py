"""Table clutter scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json,
packs clutter objects, places robot, and runs LTL-monitored rollouts.

Usage:
    python -m omnigibson.task_generation.clutter_scene_pipeline --scene-model Benevolence_1_int --dry-run
    python -m omnigibson.task_generation.clutter_scene_pipeline --scene-model Benevolence_1_int --episodes 1 --steps 300
"""

import math
import os

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    append_jsonl,
    build_descriptors,
    build_task_config,
    build_task_object_sets,
    check_interpenetration,
    clear_perimeter,
    compute_floor_z,
    discover_from_scene_json,
    generate_activity,
    get_scene_json_path,
    get_scope_obj,
    make_base_arg_parser,
    make_park_fn,
    make_settle_fn,
    needs_gpu_dynamics,
    pipeline_exit,
    refresh_activity_cache,
    resolve_synset,
    robot_half_extent_xy,
    run_ltl_rollout,
    setup_run_dir,
    validate_poses,
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
    p = make_base_arg_parser(description="Table clutter scene generation pipeline")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Table-specific discovery
# ---------------------------------------------------------------------------

def _discover_surface_from_scene_json(scene_json_path):
    from omnigibson.utils.surface_discovery import is_table_like
    return discover_from_scene_json(scene_json_path, is_table_like, _SURFACE_CATEGORY_PRIORITY)


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
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    """Generate BDDL + ltl_safety.json without starting the simulator."""
    activity_name = args.activity_name or f"auto_clutter_on_{args.scene_model}"

    support_synset, support_room = "breakfast_table.n.01", "living_room"
    try:
        scene_json = get_scene_json_path(args.scene_model)
        discovery = _discover_surface_from_scene_json(scene_json)
        if discovery:
            support_synset = resolve_synset(discovery[0])
            support_room = discovery[1]
            print(f"[Pipeline] Discovered: {discovery[0]} in {support_room}")
    except Exception:
        pass

    bddl_text, ltl_safety, bddl_path, json_path = generate_activity(
        activity_name, support_synset, support_room, args.clutter_density,
    )
    print(f"[Pipeline] Dry-run complete:")
    print(f"  BDDL:       {bddl_path}")
    print(f"  ltl_safety: {json_path}")
    print(f"  activity:   {activity_name}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "scene_model": args.scene_model, "density": args.clutter_density,
    })
    return activity_name, bddl_path, json_path


def run_sim(args, activity_name=None):
    """Full sim-validation path: surface discovery, pack, robot, gate, LTL."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.clutter_pack_layout import validate_pack_integrity
    from omnigibson.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS, EdgeAlignObject, EdgeAlignRequest, place_franka_edge_aligned,
    )
    from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone
    from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec
    from omnigibson.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop

    gm.ENABLE_OBJECT_STATES = True

    if activity_name is None:
        activity_name = args.activity_name or f"auto_clutter_on_{args.scene_model}"

    # -- Discover surface from scene JSON -----------------------------------
    scene_json = get_scene_json_path(args.scene_model)
    if not os.path.isfile(scene_json):
        raise RuntimeError(f"Scene JSON not found: {scene_json}")

    discovery = _discover_surface_from_scene_json(scene_json)
    if discovery is None:
        raise RuntimeError(f"No table-like surface in scene '{args.scene_model}'.")

    support_synset = resolve_synset(discovery[0])
    support_room = discovery[1]
    print(f"[Pipeline] Discovered: category={discovery[0]} synset={support_synset} room={support_room}")

    # -- Generate BDDL ------------------------------------------------------
    _, _, bddl_path, _ = generate_activity(
        activity_name, support_synset, support_room, args.clutter_density,
    )
    print(f"[Pipeline] Generated BDDL: {bddl_path}")
    refresh_activity_cache()

    # -- GPU dynamics --------------------------------------------------------
    gpu = needs_gpu_dynamics(activity_name)
    gm.USE_GPU_DYNAMICS = gpu
    gm.ENABLE_FLATCACHE = not gpu

    # -- Load environment ----------------------------------------------------
    cfg = build_task_config(args.scene_model, activity_name)
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
            floor_z = compute_floor_z(env)
            clear_perimeter(env, support_obj, surface_bounds_xy, table_top_z, floor_z)

            # -- Build object sets ------------------------------------------
            task_spec = build_manipulation_task_spec(activity_name)
            obj_sets = build_task_object_sets(env, task_spec)

            if not obj_sets["target_ids"]:
                raise RuntimeError("No target objects found.")
            target_obj = get_scope_obj(env, obj_sets["target_ids"][0])

            descriptors, objects_by_inst = build_descriptors(env, obj_sets)
            if not descriptors:
                raise RuntimeError("No clutter-pack descriptors created.")

            # -- Pack retry loop --------------------------------------------
            pack_config = PackRetryConfig(
                pack_jitter_xy=args.pack_jitter_xy or 0.022,
                pack_min_clearance=args.pack_min_clearance or 0.008,
            )
            settle_fn = make_settle_fn(og, th)
            park_fn = make_park_fn(og, zone.surface_bounds, floor_z)

            pack_result = run_pack_retry_loop(
                support_name=getattr(support_obj, "name", "support"),
                descriptors=descriptors, objects_by_inst=objects_by_inst,
                red_zone_bounds=zone.red_zone_bounds, table_top_z=table_top_z,
                floor_z=floor_z, config=pack_config, base_seed=args.seed, episode=ep,
                settle_fn=settle_fn, park_fn=park_fn,
                validate_poses_fn=validate_poses,
                check_interpenetration_fn=check_interpenetration,
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
                robot_half_extent_xy=robot_half_extent_xy(robot),
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
            summary, executed = run_ltl_rollout(
                env=env, activity_name=activity_name, scene_model=args.scene_model,
                active_objects_by_inst=pack_result.active_objects_by_inst,
                robot=robot, target_obj=target_obj,
                args=args, episode=ep, rng=rng,
            )

            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "scene_model": args.scene_model,
                "activity_name": activity_name, "surface": surface_info.surface.name,
                "pack_attempt_used": pack_result.attempt_used,
                "gate_pass": gate_pass, "ltl_violated": summary["violated"],
                "steps_executed": executed,
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
