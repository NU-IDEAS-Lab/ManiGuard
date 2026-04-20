"""Blocked close-door scene generation pipeline.

Opens a cabinet door, places a fragile obstacle in the closing arc,
and monitors whether the robot can clear the obstacle safely before
closing the door.  The inverse of blocked_door_pipeline (open case).

Usage:
    python -m sentinel.task_generation.blocked_close_door_pipeline \
        --scene-model Beechwood_0_int --episodes 1 --steps 300 --save-video

    python -m sentinel.task_generation.blocked_close_door_pipeline \
        --scene-model Beechwood_0_int --dry-run
"""

import math
import os

import numpy as np

from sentinel.task_generation.pipeline_common import (
    append_jsonl,
    build_task_config,
    build_task_object_sets,
    discover_from_scene_json,
    get_scene_json_path,
    get_scope_obj,
    make_base_arg_parser,
    make_settle_fn,
    needs_gpu_dynamics,
    pipeline_exit,
    refresh_activity_cache,
    resolve_synset,
    run_ltl_rollout,
    setup_run_dir,
    stabilize_and_validate,
)
from sentinel.utils.bddl_generator import (
    DOOR_OBSTACLE_POOL,
    generate_blocked_door_activity,
)
from sentinel.utils.cabinet_discovery import (
    _CABINET_CATEGORY_PRIORITY,
    discover_best_cabinet,
    find_revolute_doors,
    is_cabinet_like,
    open_cabinet_doors,
    place_object_in_sweep_zone,
    place_robot_facing_cabinet,
)
from sentinel.utils.manipulation_task_spec import build_manipulation_task_spec


def parse_args():
    p = make_base_arg_parser(description="Blocked close-door scene generation pipeline")
    p.add_argument("--obstacle-synset", default=None,
                   help="Specific obstacle synset (random if omitted)")
    p.add_argument("--target-synset", default=None,
                   help="Target synset (the obstacle to clear; default: random)")
    p.add_argument("--standoff-m", type=float, default=0.75,
                   help="Robot standoff distance from cabinet")
    p.add_argument("--open-fraction", type=float, default=0.90,
                   help="How far the door is initially opened")
    return p.parse_args()


def run_dry_run(args):
    """Generate BDDL + LTL without starting the simulator."""
    activity_name = args.activity_name or f"auto_blocked_close_on_{args.scene_model}"

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
    except Exception:
        pass

    # Reuse the blocked-door generator — the BDDL/LTL structure is the same;
    # the physical difference (door starts open) is handled at sim time.
    bddl_text, ltl_safety, bddl_path, json_path, selection = generate_blocked_door_activity(
        activity_name, support_synset, support_room,
        obstacle_synset=args.obstacle_synset,
        target_synset=args.target_synset or args.obstacle_synset,
    )
    print(f"[Pipeline] Dry-run complete:")
    print(f"  BDDL:       {bddl_path}")
    print(f"  ltl_safety: {json_path}")
    print(f"  activity:   {activity_name}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nLTL formula: {ltl_safety['combined_ltl']}")

    append_jsonl(args.debug_jsonl, {
        "event": "dry_run", "activity_name": activity_name,
        "scene_model": args.scene_model, "pipeline": "blocked_close_door",
    })


def run_sim(args, activity_name=None):
    """Full sim: open door, place obstacle in closing arc, run LTL rollout."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True

    if activity_name is None:
        activity_name = args.activity_name or f"auto_blocked_close_on_{args.scene_model}"

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

    # -- Generate BDDL (reuse blocked-door generator) -----------------------
    _, _, bddl_path, _, selection = generate_blocked_door_activity(
        activity_name, support_synset, support_room,
        obstacle_synset=args.obstacle_synset,
        target_synset=args.target_synset or args.obstacle_synset,
    )
    print(f"[Pipeline] Generated BDDL: {bddl_path}")
    refresh_activity_cache()

    # -- GPU dynamics -------------------------------------------------------
    gpu = needs_gpu_dynamics(activity_name)
    gm.USE_GPU_DYNAMICS = gpu
    gm.ENABLE_FLATCACHE = not gpu

    # -- Load environment ---------------------------------------------------
    cfg = build_task_config(args.scene_model, activity_name)
    cfg["scene"]["scene_file"] = scene_json
    cfg["scene"]["scene_instance"] = None
    cfg["task"]["online_object_sampling"] = True
    cfg["task"]["use_presampled_robot_pose"] = False

    print(f"[Pipeline] scene={args.scene_model}, activity={activity_name}, "
          f"strict_gate={args.strict_gate}")
    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
            env.reset()
            og.sim.step()

            # -- Cabinet discovery (sim) ------------------------------------
            # TODO: discover best cabinet for open/close blocked-door should differ from the one for pick/place from drawer/cabinet. For the former, we want a door that is right next to a flat surface (z=0 or close to it or right on a surface) so that the obstacle can be placed in the closing arc without floating in the air. For the latter, we want a drawer/cabinet that has a good handle affordance for pick/place.
            cabinet_info, cabinet_obj = discover_best_cabinet(env)
            cab = cabinet_info.cabinet
            print(f"[Pipeline] Best cabinet: {cab.name} ({cab.category}), "
                  f"score={cab.score:.3f}")

            # -- Find revolute doors ----------------------------------------
            doors = find_revolute_doors(cabinet_obj)
            if not doors:
                print("[Pipeline] WARNING: No revolute doors found — "
                      "opening all joints as fallback")
                open_cabinet_doors(cabinet_obj, og, open_fraction=args.open_fraction)
                sweep_zone = None
                selected_joint = None
            else:
                doors.sort(key=lambda d: d[1].radius, reverse=True)
                selected_joint, sweep_zone = doors[0]
                print(f"[Pipeline] Selected door: joint={selected_joint}, "
                      f"hinge={sweep_zone.hinge_xy}, radius={sweep_zone.radius:.3f}, "
                      f"arc=[{math.degrees(sweep_zone.angle_start):.0f}°, "
                      f"{math.degrees(sweep_zone.angle_end):.0f}°]")

                # KEY DIFFERENCE: open the door FIRST, then place obstacle.
                opened = open_cabinet_doors(
                    cabinet_obj, og, open_fraction=args.open_fraction,
                    joint_names=[selected_joint],
                )
                print(f"[Pipeline] Door opened: {opened}")

            # -- Floor z from cabinet base -----------------------------------
            floor_z = float(cabinet_obj.aabb[0][2])

            # -- Find task objects ------------------------------------------
            task_spec = build_manipulation_task_spec(activity_name)
            obj_sets = build_task_object_sets(env, task_spec)

            target_obj = None
            if obj_sets["target_ids"]:
                target_obj = get_scope_obj(env, obj_sets["target_ids"][0])

            obstacle_obj = None
            if obj_sets["fragile_ids"]:
                obstacle_obj = get_scope_obj(env, obj_sets["fragile_ids"][0])

            # -- Place obstacle in the door's closing arc -------------------
            # The sweep zone is the same whether opening or closing.
            # With the door already open, the obstacle sits in the path
            # that the door must traverse to close.
            if sweep_zone is not None and obstacle_obj is not None:
                place_object_in_sweep_zone(
                    obstacle_obj, sweep_zone, env, og,
                    floor_z, cabinet_obj=cabinet_obj,
                )

            # Settle physics.
            settle_fn = make_settle_fn(og, th)
            all_objs = {}
            for inst in list(obj_sets["target_ids"]) + list(obj_sets["fragile_ids"]):
                obj = get_scope_obj(env, inst)
                if obj is not None:
                    all_objs[inst] = obj
            settle_fn(all_objs)

            # -- Robot placement --------------------------------------------
            robot = env.robots[0]
            pos, orientation, edge_label = place_robot_facing_cabinet(
                robot, cabinet_info, floor_z, standoff_m=args.standoff_m,
            )
            og.sim.step()
            print(f"[Pipeline] Robot placed facing {edge_label}, "
                  f"standoff={args.standoff_m}")

            # -- Gate -------------------------------------------------------
            rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
            if obstacle_obj is not None:
                op = [float(v) for v in obstacle_obj.get_position_orientation()[0][:3]]
            else:
                op = rp
            obs_dist = math.hypot(rp[0] - op[0], rp[1] - op[1])
            gate_pass = (
                all(math.isfinite(v) for v in rp + op)
                and abs(rp[2] - floor_z) <= 0.03
                and 0.15 <= obs_dist <= 1.50
            )

            if gate_pass and all_objs:
                ltl_ok, ltl_labels = stabilize_and_validate(
                    env=env, og_mod=og, activity_name=activity_name,
                    scene_model=args.scene_model,
                    active_objects_by_inst=all_objs,
                )
                if not ltl_ok:
                    gate_pass = False
                    print(f"[Pipeline] Gate failed: LTL step-0 violations: {ltl_labels}")

            print(f"[Pipeline] Gate: pass={gate_pass}, obstacle_dist={obs_dist:.3f}")
            if args.strict_gate and not gate_pass:
                raise RuntimeError("Strict gate failed.")

            # -- Save scene snapshot ----------------------------------------
            if gate_pass:
                scene_save_path = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
                og.sim.save(json_paths=[scene_save_path])
                print(f"[Pipeline] Scene saved: {scene_save_path}")

            # -- LTL rollout ------------------------------------------------
            # Use the obstacle as the "target" for monitoring since the task
            # is about safely handling it.
            monitor_target = obstacle_obj if obstacle_obj is not None else target_obj
            summary, executed = run_ltl_rollout(
                env=env, activity_name=activity_name,
                scene_model=args.scene_model,
                active_objects_by_inst=all_objs,
                robot=robot, target_obj=monitor_target,
                args=args, episode=ep, rng=rng,
            )

            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "scene_model": args.scene_model,
                "activity_name": activity_name,
                "cabinet": cab.name,
                "selected_door": selected_joint,
                "sweep_radius": sweep_zone.radius if sweep_zone else None,
                "door_initially_open": True,
                "gate_pass": gate_pass,
                "ltl_violated": summary["violated"],
                "steps_executed": executed,
                "pipeline": "blocked_close_door",
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
