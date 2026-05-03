"""Cabinet clutter scene generation pipeline.

Auto-discovers a cabinet in any scene, opens its doors, packs clutter objects
inside, places robot facing the opening, and runs LTL-monitored rollouts.

Usage:
    python -m sentinel.task_generation.cabinet_clutter_pipeline --scene-model Rs_int --dry-run
    python -m sentinel.task_generation.cabinet_clutter_pipeline --scene-model Rs_int --episodes 1 --steps 300
"""

import math
import os

import numpy as np

from sentinel.task_generation.pipeline_common import (
    append_jsonl,
    build_descriptors,
    build_task_config,
    build_task_object_sets,
    check_interpenetration,
    clear_support_area,
    discover_from_scene_json,
    get_scene_json_path,
    get_spawned_obj,
    make_base_arg_parser,
    make_park_fn,
    make_settle_fn,
    needs_gpu_dynamics_from_specs,
    pipeline_exit,
    resolve_synset,
    run_ltl_rollout,
    setup_run_dir,
    spawn_objects,
    validate_poses,
)
from sentinel.utils.cabinet_discovery import (
    _CABINET_CATEGORY_PRIORITY,
    compute_cabinet_packing_zone,
    discover_best_cabinet,
    is_cabinet_like,
    open_cabinet_doors,
    place_robot_facing_cabinet,
)
import logging

log = logging.getLogger(__name__)


def parse_args():
    p = make_base_arg_parser(description="Cabinet clutter scene generation pipeline")
    p.add_argument("--open-fraction", type=float, default=0.90,
                   help="How far to open cabinet doors (0=closed, 1=fully open)")
    p.add_argument("--standoff-m", type=float, default=0.75,
                   help="Robot standoff distance from cabinet opening")
    return p.parse_args()


def run_dry_run(args):
    """Generate LTL safety spec for cabinet task without starting the simulator."""
    from sentinel.utils.task_spec import generate_clutter_activity

    activity_name = args.activity_name or f"auto_cabinet_on_{args.scene_model}"

    support_synset, support_room = "bottom_cabinet.n.01", "kitchen"
    try:
        scene_json = get_scene_json_path(args.scene_model)
        discovery = discover_from_scene_json(
            scene_json, is_cabinet_like, _CABINET_CATEGORY_PRIORITY,
        )
        if discovery:
            support_synset = resolve_synset(discovery[0])
            support_room = discovery[1]
            print(f"[Pipeline] Discovered cabinet: {discovery[0]} in {support_room}")
    except Exception as exc:
        log.warning("cabinet_clutter_pipeline dry-run: scene path lookup for %s failed: %s", args.scene_model, exc)
        pass

    ltl_safety, selection = generate_clutter_activity(
        activity_name, support_synset, support_room, args.clutter_density,
        init_predicate="inside",
    )
    print(f"[Pipeline] Dry-run complete:")
    print(f"  activity:    {activity_name}")
    print(f"  spawn_specs: {selection.get('spawn_specs', [])}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "scene_model": args.scene_model, "density": args.clutter_density,
        "pipeline": "cabinet",
    })
    return activity_name


def run_sim(args, activity_name=None):
    """Full sim path: cabinet discovery, door opening, interior packing, robot placement, LTL."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from sentinel.utils.clutter_pack_layout import validate_pack_integrity
    from sentinel.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop
    from sentinel.utils.task_spec import generate_clutter_activity

    gm.ENABLE_OBJECT_STATES = True

    if activity_name is None:
        activity_name = args.activity_name or f"auto_cabinet_on_{args.scene_model}"

    # -- Discover cabinet from scene JSON -----------------------------------
    scene_json = get_scene_json_path(args.scene_model)
    if not os.path.isfile(scene_json):
        raise RuntimeError(f"Scene JSON not found: {scene_json}")

    discovery = discover_from_scene_json(
        scene_json, is_cabinet_like, _CABINET_CATEGORY_PRIORITY,
    )
    if discovery is None:
        raise RuntimeError(f"No cabinet-like object in scene '{args.scene_model}'.")

    support_synset = resolve_synset(discovery[0])
    support_room = discovery[1]
    print(f"[Pipeline] Discovered cabinet: category={discovery[0]} synset={support_synset} room={support_room}")

    # -- Generate activity (LTL + spawn specs) ------------------------------
    ltl_safety, selection = generate_clutter_activity(
        activity_name, support_synset, support_room, args.clutter_density,
        init_predicate="inside",
    )
    spawn_specs = selection.get("spawn_specs", [])

    # -- GPU dynamics --------------------------------------------------------
    gpu = needs_gpu_dynamics_from_specs(spawn_specs)
    gm.USE_GPU_DYNAMICS = gpu
    gm.ENABLE_FLATCACHE = not gpu

    # -- Load environment ----------------------------------------------------
    cfg = build_task_config(args.scene_model)
    cfg["scene"]["scene_file"] = scene_json
    cfg["scene"]["scene_instance"] = None

    print(f"[Pipeline] scene={args.scene_model}, activity={activity_name}, strict_gate={args.strict_gate}")
    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
            env.reset()
            og.sim.step()

            # -- Spawn task objects -----------------------------------------
            spawned_objects = spawn_objects(env, spawn_specs, rng)
            og.sim.step()
            print(f"[Pipeline] Spawned {len(spawned_objects)} objects: "
                  f"{sorted(spawned_objects.keys())}")

            # -- Cabinet discovery (sim) ------------------------------------
            cabinet_info, cabinet_obj = discover_best_cabinet(env)
            cab = cabinet_info.cabinet
            print(f"[Pipeline] Best cabinet: {cab.name} ({cab.category}), "
                  f"score={cab.score:.3f}, interior_z=[{cab.interior_bottom_z:.2f}, {cab.interior_top_z:.2f}]")

            # -- Select compartment and open its door -----------------------
            if cab.compartments:
                comp = cab.compartments[rng.integers(len(cab.compartments))]
                print(f"[Pipeline] Selected compartment: joint={comp.joint_name} "
                      f"({comp.joint_type}), link={comp.link_name}, "
                      f"z=[{comp.interior_bottom_z:.2f}, {comp.interior_top_z:.2f}]")
                opened = open_cabinet_doors(
                    cabinet_obj, og, open_fraction=args.open_fraction,
                    joint_names=[comp.joint_name],
                )
                packing_xy = comp.interior_bounds_xy
                placement_z = comp.interior_bottom_z
            else:
                print("[Pipeline] No compartments detected, opening all joints")
                opened = open_cabinet_doors(cabinet_obj, og, open_fraction=args.open_fraction)
                packing_xy = cab.interior_bounds_xy
                placement_z = cab.interior_bottom_z

            # -- Compute packing zone (interior) ----------------------------
            red_zone, surface_bounds = compute_cabinet_packing_zone(
                packing_xy, edge_margin_m=0.02,
            )
            print(f"[Pipeline] Cabinet zone: red_zone={red_zone}, placement_z={placement_z:.3f}")

            # -- Clear perimeter objects ------------------------------------
            floor_z = float(cabinet_obj.aabb[0][2])
            clear_support_area(env, cabinet_obj, cab.aabb_xy,
                               spawned_objects=spawned_objects)

            # -- Build object sets ------------------------------------------
            obj_sets = build_task_object_sets(spawned_objects)

            if not obj_sets["target_ids"]:
                raise RuntimeError("No target objects found.")
            target_obj = get_spawned_obj(spawned_objects, obj_sets["target_ids"][0])

            descriptors, objects_by_inst = build_descriptors(spawned_objects, obj_sets)
            if not descriptors:
                raise RuntimeError("No clutter-pack descriptors created.")

            # -- Filter objects that don't fit in the compartment --------------
            if cab.compartments:
                comp_height = comp.interior_top_z - comp.interior_bottom_z
                fitting = [d for d in descriptors if d.height <= comp_height]
                n_parked = len(descriptors) - len(fitting)
                if n_parked:
                    print(f"[Pipeline] Parking {n_parked} objects too tall "
                          f"for compartment (height={comp_height:.3f}m)")
                descriptors = fitting
                if not descriptors:
                    raise RuntimeError("No objects fit inside the selected compartment.")

            # -- Pack retry loop (inside cabinet) ---------------------------
            pack_config = PackRetryConfig(
                pack_jitter_xy=args.pack_jitter_xy or 0.015,
                pack_min_clearance=args.pack_min_clearance or 0.005,
            )
            settle_fn = make_settle_fn(og, th)
            park_fn = make_park_fn(og, surface_bounds, floor_z)

            pack_result = run_pack_retry_loop(
                support_name=getattr(cabinet_obj, "name", "cabinet"),
                descriptors=descriptors, objects_by_inst=objects_by_inst,
                red_zone_bounds=red_zone, table_top_z=placement_z,
                floor_z=floor_z, config=pack_config, base_seed=args.seed, episode=ep,
                settle_fn=settle_fn, park_fn=park_fn,
                validate_poses_fn=validate_poses,
                check_interpenetration_fn=check_interpenetration,
                obstacle_keepout_bounds=None,
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

            # -- Robot placement (facing cabinet opening) -------------------
            robot = env.robots[0]
            pos, orientation, edge_label = place_robot_facing_cabinet(
                robot, cabinet_info, floor_z, standoff_m=args.standoff_m,
            )
            og.sim.step()
            print(f"[Pipeline] Robot placed facing {edge_label}, standoff={args.standoff_m}")

            # -- Gate -------------------------------------------------------
            rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
            tp = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
            target_dist = math.hypot(rp[0] - tp[0], rp[1] - tp[1])
            gate_pass = (
                all(math.isfinite(v) for v in rp + tp)
                and abs(rp[2] - floor_z) <= 0.03
                and 0.15 <= target_dist <= 1.30
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
                "activity_name": activity_name, "cabinet": cab.name,
                "cabinet_category": cab.category,
                "compartment_joint": comp.joint_name if cab.compartments else None,
                "compartment_link": comp.link_name if cab.compartments else None,
                "pack_attempt_used": pack_result.attempt_used,
                "gate_pass": gate_pass, "ltl_violated": summary["violated"],
                "steps_executed": executed, "pipeline": "cabinet",
                "ltl_safety": ltl_safety,
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
