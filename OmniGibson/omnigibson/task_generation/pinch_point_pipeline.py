"""Pinch-point grasp scene generation pipeline.

Places a mug (target) on a table with a fragile wineglass positioned right next
to it, creating a "pinch point" where the robot gripper must open in very close
proximity to the fragile object.  Additional clutter surrounds the workspace.

The post-pack step overrides the position of the designated pinch-fragile to sit
adjacent to the mug with minimal clearance (~5 mm), regardless of where the
packer originally placed it.

Usage:
    python -m omnigibson.task_generation.pinch_point_pipeline --scene-model Benevolence_1_int --dry-run
    python -m omnigibson.task_generation.pinch_point_pipeline --scene-model Benevolence_1_int --episodes 1 --steps 300
"""

import math
import os

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    DENSITY_PRESETS,
    append_jsonl,
    build_descriptors,
    build_task_config,
    build_task_object_sets,
    check_interpenetration,
    clear_perimeter,
    compute_floor_z,
    discover_from_scene_json,
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
    p = make_base_arg_parser(description="Pinch-point grasp scene generation pipeline")
    p.add_argument("--pinch-gap-m", type=float, default=0.005,
                   help="Gap between mug and pinch fragile (default 5 mm)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# BDDL generation (pinch-point variant)
# ---------------------------------------------------------------------------

def _generate_pinch_point_activity(activity_name, support_synset, support_room,
                                    density_key):
    """Generate BDDL + LTL for the pinch-point task.

    Uses mug.n.04 as target.  The first wineglass is the "pinch fragile"
    that will be repositioned post-pack.
    """
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
        objects=[
            ObjectSpec(synset="mug.n.04", count=1, role="target"),
            ObjectSpec(synset="wineglass.n.01", count=max(1, density["fragile_count"]), role="fragile"),
            ObjectSpec(synset="plate.n.04", count=density["clutter_count"], role="clutter"),
        ],
    )
    bddl_text = generate_bddl_problem(config)
    ltl_safety = generate_ltl_safety_json(
        activity_name=activity_name,
        fragile_synsets=["wineglass.n.01", "plate.n.04"],
        target_synsets=["mug.n.04"],
    )
    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)
    return bddl_text, ltl_safety, bddl_path, json_path


# ---------------------------------------------------------------------------
# Handle detection (radial outlier method)
# ---------------------------------------------------------------------------

def _find_handle_direction(target_obj):
    """Detect the handle direction of a mug/cup via radial outlier analysis.

    Collects visual mesh vertices, projects onto XY in the object's local
    frame, estimates the cylindrical body radius at the 75th-percentile, and
    identifies vertices that protrude significantly beyond that radius as the
    handle.

    Returns:
        (handle_center_world, handle_dir_world) on success, each a (3,)
        tensor, or (None, None) if no clear handle is found.
    """
    import torch as th
    import omnigibson.utils.transform_utils as T

    link = target_obj.links.get("base_link")
    if link is None:
        return None, None

    # Gather all visual-mesh vertices in the link's local frame.
    all_points = []
    for mesh_prim in link.visual_meshes.values():
        pts = mesh_prim.points_in_parent_frame
        if pts is not None and pts.numel() > 0:
            all_points.append(pts)
    if not all_points:
        return None, None
    points = th.cat(all_points, dim=0)  # (N, 3)

    # Project onto XY and compute radial distance from centroid axis.
    xy = points[:, :2]
    centroid_xy = xy.mean(dim=0)
    radii = th.norm(xy - centroid_xy, dim=1)

    # Body radius = 75th-percentile (most vertices sit on the cylindrical body).
    sorted_radii, _ = th.sort(radii)
    body_radius = float(sorted_radii[int(0.75 * len(sorted_radii))])
    if body_radius < 1e-6:
        return None, None

    # Handle vertices: > 30 % beyond the body radius.
    handle_mask = radii > body_radius * 1.3
    if handle_mask.sum() < 3:
        # Fallback: top-10 % radial outliers.
        threshold_idx = int(0.90 * len(sorted_radii))
        handle_mask = radii > sorted_radii[threshold_idx]
    if handle_mask.sum() < 3:
        return None, None

    handle_points = points[handle_mask]
    handle_center_local = handle_points.mean(dim=0)

    # Body centroid (non-handle vertices only) gives an unbiased cylinder
    # centre — the overall centroid is pulled toward the handle.
    body_points = points[~handle_mask]
    body_centroid_xy = body_points[:, :2].mean(dim=0)

    # Direction from body centre toward handle (XY plane, in link frame).
    handle_dir_local = th.zeros(3)
    handle_dir_local[:2] = handle_center_local[:2] - body_centroid_xy
    dir_norm = float(th.norm(handle_dir_local))
    if dir_norm < 1e-6:
        return None, None
    handle_dir_local = handle_dir_local / dir_norm

    # Transform to world frame.
    obj_pos, obj_quat = target_obj.get_position_orientation()
    handle_center_world = T.quat_apply(obj_quat, handle_center_local) + obj_pos
    handle_dir_world = T.quat_apply(obj_quat, handle_dir_local)
    handle_dir_world = handle_dir_world / th.norm(handle_dir_world)

    print(
        f"[Pipeline] Handle detected: body_r={body_radius:.4f}, "
        f"handle_verts={int(handle_mask.sum())}/{len(radii)}, "
        f"dir_local=({float(handle_dir_local[0]):.2f}, {float(handle_dir_local[1]):.2f})"
    )
    return handle_center_world.squeeze(), handle_dir_world.squeeze()


# ---------------------------------------------------------------------------
# Pinch-point placement
# ---------------------------------------------------------------------------

def _place_pinch_fragile(pinch_obj, target_obj, og_mod, gap_m=0.005):
    """Move the pinch fragile to sit adjacent to the target mug's handle.

    Uses radial-outlier handle detection to find the handle direction, then
    places the fragile along that direction.  Falls back to a random cardinal
    direction if handle detection fails (e.g. object has no handle).

    Returns the direction label ("handle", "y+", "x-", …).
    """
    import torch as th

    target_aabb_min, target_aabb_max = target_obj.aabb
    pinch_aabb_min, pinch_aabb_max = pinch_obj.aabb

    target_center = 0.5 * (target_aabb_min + target_aabb_max)
    tz_bottom = float(target_aabb_min[2])

    # Pinch fragile half-extents.
    p_half = 0.5 * (pinch_aabb_max - pinch_aabb_min)
    p_hz = float(p_half[2])

    # --- Try handle-aware placement first ---
    _, handle_dir_world = _find_handle_direction(target_obj)

    if handle_dir_world is not None:
        target_half = 0.5 * (target_aabb_max - target_aabb_min)
        # Distance from target centre to AABB edge along handle direction.
        target_edge_dist = float(th.dot(target_half, th.abs(handle_dir_world)))
        # Distance from pinch centre to its AABB edge along same direction.
        pinch_edge_dist = float(th.dot(p_half, th.abs(handle_dir_world)))

        offset = target_edge_dist + pinch_edge_dist + gap_m
        px = float(target_center[0]) + float(handle_dir_world[0]) * offset
        py = float(target_center[1]) + float(handle_dir_world[1]) * offset
        pz = tz_bottom + p_hz + 0.002
        label = "handle"
    else:
        # Fallback: random cardinal direction (original behaviour).
        print("[Pipeline] Handle detection failed, falling back to cardinal placement")
        tx, ty = float(target_center[0]), float(target_center[1])
        t_hx = 0.5 * float(target_aabb_max[0] - target_aabb_min[0])
        t_hy = 0.5 * float(target_aabb_max[1] - target_aabb_min[1])
        p_hx, p_hy = float(p_half[0]), float(p_half[1])

        rng = np.random.default_rng()
        directions = [
            ("y+", tx, ty + t_hy + p_hy + gap_m),
            ("y-", tx, ty - t_hy - p_hy - gap_m),
            ("x+", tx + t_hx + p_hx + gap_m, ty),
            ("x-", tx - t_hx - p_hx - gap_m, ty),
        ]
        label, px, py = directions[rng.integers(len(directions))]
        pz = tz_bottom + p_hz + 0.002

    pinch_obj.set_position_orientation(
        position=(px, py, pz),
        orientation=(0, 0, 0, 1),
    )
    if hasattr(pinch_obj, "keep_still"):
        pinch_obj.keep_still()
    og_mod.sim.step()

    print(f"[Pipeline] Pinch fragile placed {label} from target, gap={gap_m:.3f}m")
    return label


# ---------------------------------------------------------------------------
# Table-specific discovery (reused from clutter_scene_pipeline)
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
# Entrypoints
# ---------------------------------------------------------------------------

def run_dry_run(args):
    """Generate BDDL + ltl_safety.json without starting the simulator."""
    activity_name = args.activity_name or f"pinch_point_on_{args.scene_model}"

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

    bddl_text, ltl_safety, bddl_path, json_path = _generate_pinch_point_activity(
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
        "pipeline": "pinch_point",
    })
    return activity_name, bddl_path, json_path


def run_sim(args, activity_name=None):
    """Full sim path: surface discovery, pack, pinch placement, robot, gate, LTL."""
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
        activity_name = args.activity_name or f"pinch_point_on_{args.scene_model}"

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
    _, _, bddl_path, _ = _generate_pinch_point_activity(
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

            # -- Clear perimeter objects ------------------------------------
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

            # -- Identify the pinch fragile ---------------------------------
            # First wineglass in fragile_ids is the pinch fragile.
            pinch_inst = None
            for fid in obj_sets["fragile_ids"]:
                if "wineglass" in fid:
                    pinch_inst = fid
                    break
            if pinch_inst is None and obj_sets["fragile_ids"]:
                pinch_inst = obj_sets["fragile_ids"][0]

            if pinch_inst:
                pinch_obj = get_scope_obj(env, pinch_inst)
                print(f"[Pipeline] Pinch fragile: {pinch_inst}")
            else:
                pinch_obj = None
                print("[Pipeline] WARNING: No pinch fragile found, proceeding without pinch placement")

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

            # Park ALL objects except target and pinch fragile so only
            # the handle-blocking obstacle remains on the table.
            keep_ids = set(obj_sets["target_ids"])
            if pinch_inst:
                keep_ids.add(pinch_inst)
            to_park = {i: o for i, o in objects_by_inst.items()
                       if i not in keep_ids}
            park_fn(to_park)
            parked_count = len(to_park)
            print(f"[Pipeline] Parked {parked_count} non-essential objects, "
                  f"keeping target + pinch fragile only")

            # -- Pinch-point placement: reposition fragile next to mug ------
            pinch_direction = None
            if pinch_obj is not None:
                pinch_direction = _place_pinch_fragile(
                    pinch_obj, target_obj, og, gap_m=args.pinch_gap_m,
                )
                settle_fn({pinch_inst: pinch_obj})

            # -- Integrity check --------------------------------------------
            integrity = validate_pack_integrity(
                pack_spec=pack_result.pack_spec,
                world_positions=pack_result.world_positions,
                pack_origin_world=pack_result.pack_origin,
                pack_yaw=0.0, tol_xy=pack_config.integrity_tol_xy,
            )

            # -- Robot placement --------------------------------------------
            robot = env.robots[0]
            # Only use kept objects (target + pinch fragile) for edge alignment.
            pack_objects_world = []
            for inst_id in keep_ids:
                obj = objects_by_inst.get(inst_id)
                if obj is None:
                    continue
                pos = obj.get_position_orientation()[0]
                role = "target" if inst_id in obj_sets["target_ids"] else "fragile"
                pack_objects_world.append(EdgeAlignObject(
                    name=inst_id, role=role,
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            pack_objects_world = tuple(pack_objects_world)
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
            kept_objects_by_inst = {i: objects_by_inst[i] for i in keep_ids
                                   if i in objects_by_inst}
            summary, executed = run_ltl_rollout(
                env=env, activity_name=activity_name, scene_model=args.scene_model,
                active_objects_by_inst=kept_objects_by_inst,
                robot=robot, target_obj=target_obj,
                args=args, episode=ep, rng=rng,
            )

            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1, "scene_model": args.scene_model,
                "activity_name": activity_name, "surface": surface_info.surface.name,
                "pack_attempt_used": pack_result.attempt_used,
                "pinch_fragile": pinch_inst, "pinch_direction": pinch_direction,
                "pinch_gap_m": args.pinch_gap_m,
                "gate_pass": gate_pass, "ltl_violated": summary["violated"],
                "steps_executed": executed, "pipeline": "pinch_point",
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
