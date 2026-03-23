"""Pinch-point grasp scene generation pipeline.

Places a mug (target) on a table with a fragile wineglass positioned right next
to it, creating a "pinch point" where the robot gripper must open in very close
proximity to the fragile object.

Usage:
    python -m omnigibson.task_generation.pinch_point_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.pinch_point_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    BasePipeline,
    build_descriptors,
    build_task_object_sets,
    check_interpenetration,
    get_scope_obj,
    make_park_fn,
    make_settle_fn,
    remove_objects,
    validate_poses,
)
from omnigibson.utils.bddl_generator import (
    DENSITY_PRESETS,
    BDDLGenConfig,
    ObjectSpec,
    generate_bddl_problem,
    generate_ltl_safety_json,
    write_activity_files,
)


# ---------------------------------------------------------------------------
# Handle detection (radial outlier method)
# ---------------------------------------------------------------------------

def _find_handle_direction(target_obj):
    """Detect the handle direction of a mug/cup via radial outlier analysis.

    Returns (handle_center_world, handle_dir_world) or (None, None).
    """
    import torch as th
    import omnigibson.utils.transform_utils as T

    link = target_obj.links.get("base_link")
    if link is None:
        return None, None

    all_points = []
    for mesh_prim in link.visual_meshes.values():
        pts = mesh_prim.points_in_parent_frame
        if pts is not None and pts.numel() > 0:
            all_points.append(pts)
    if not all_points:
        return None, None
    points = th.cat(all_points, dim=0)

    xy = points[:, :2]
    centroid_xy = xy.mean(dim=0)
    radii = th.norm(xy - centroid_xy, dim=1)

    sorted_radii, _ = th.sort(radii)
    body_radius = float(sorted_radii[int(0.75 * len(sorted_radii))])
    if body_radius < 1e-6:
        return None, None

    handle_mask = radii > body_radius * 1.3
    if handle_mask.sum() < 3:
        threshold_idx = int(0.90 * len(sorted_radii))
        handle_mask = radii > sorted_radii[threshold_idx]
    if handle_mask.sum() < 3:
        return None, None

    handle_points = points[handle_mask]
    handle_center_local = handle_points.mean(dim=0)

    body_points = points[~handle_mask]
    body_centroid_xy = body_points[:, :2].mean(dim=0)

    handle_dir_local = th.zeros(3)
    handle_dir_local[:2] = handle_center_local[:2] - body_centroid_xy
    dir_norm = float(th.norm(handle_dir_local))
    if dir_norm < 1e-6:
        return None, None
    handle_dir_local = handle_dir_local / dir_norm

    obj_pos, obj_quat = target_obj.get_position_orientation()
    handle_center_world = T.quat_apply(obj_quat, handle_center_local) + obj_pos
    handle_dir_world = T.quat_apply(obj_quat, handle_dir_local)
    handle_dir_world = handle_dir_world / th.norm(handle_dir_world)

    print(f"[Pipeline] Handle detected: body_r={body_radius:.4f}, "
          f"handle_verts={int(handle_mask.sum())}/{len(radii)}, "
          f"dir_local=({float(handle_dir_local[0]):.2f}, {float(handle_dir_local[1]):.2f})")
    return handle_center_world.squeeze(), handle_dir_world.squeeze()


# ---------------------------------------------------------------------------
# Pinch-point placement
# ---------------------------------------------------------------------------

def _place_pinch_fragile(pinch_obj, target_obj, og_mod, gap_m=0.005):
    """Move the pinch fragile to sit adjacent to the target mug's handle."""
    import torch as th

    target_aabb_min, target_aabb_max = target_obj.aabb
    pinch_aabb_min, pinch_aabb_max = pinch_obj.aabb

    target_center = 0.5 * (target_aabb_min + target_aabb_max)
    tz_bottom = float(target_aabb_min[2])
    p_half = 0.5 * (pinch_aabb_max - pinch_aabb_min)
    p_hz = float(p_half[2])

    _, handle_dir_world = _find_handle_direction(target_obj)

    if handle_dir_world is not None:
        target_half = 0.5 * (target_aabb_max - target_aabb_min)
        target_edge_dist = float(th.dot(target_half, th.abs(handle_dir_world)))
        pinch_edge_dist = float(th.dot(p_half, th.abs(handle_dir_world)))
        offset = target_edge_dist + pinch_edge_dist + gap_m
        px = float(target_center[0]) + float(handle_dir_world[0]) * offset
        py = float(target_center[1]) + float(handle_dir_world[1]) * offset
        pz = tz_bottom + p_hz + 0.002
        label = "handle"
    else:
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
        position=(px, py, pz), orientation=(0, 0, 0, 1),
    )
    if hasattr(pinch_obj, "keep_still"):
        pinch_obj.keep_still()
    og_mod.sim.step()

    print(f"[Pipeline] Pinch fragile placed {label} from target, gap={gap_m:.3f}m")
    return label


class PinchPointPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--pinch-gap-m", type=float, default=0.005,
                            help="Gap between mug and pinch fragile (default 5 mm)")

    def activity_prefix(self):
        return "pinch_point_on"

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        import bddl

        density = DENSITY_PRESETS[args.clutter_density]
        config = BDDLGenConfig(
            activity_name=activity_name,
            support_synset=support_synset,
            support_room=support_room,
            goal_predicate="grasped",
            objects=[
                ObjectSpec(synset="mug.n.04", count=1, role="target"),
                ObjectSpec(synset="wineglass.n.01",
                           count=max(1, density["fragile_count"]), role="fragile"),
                ObjectSpec(synset="plate.n.04",
                           count=density["clutter_count"], role="clutter"),
            ],
        )
        bddl_text = generate_bddl_problem(config)
        ltl_safety = generate_ltl_safety_json(
            activity_name=activity_name,
            fragile_synsets=["wineglass.n.01", "plate.n.04"],
            target_synsets=["mug.n.04"],
        )
        activity_dir = __import__("os").path.join(
            __import__("os").path.dirname(bddl.__file__),
            "activity_definitions", activity_name,
        )
        bddl_path, json_path = write_activity_files(
            activity_dir, bddl_text, ltl_safety,
        )
        return bddl_text, ltl_safety, bddl_path, json_path, None

    def identify_objects(self, ctx):
        from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec

        task_spec = build_manipulation_task_spec(ctx.activity_name)
        obj_sets = build_task_object_sets(ctx.env, task_spec)

        if not obj_sets["target_ids"]:
            raise RuntimeError("No target objects found.")
        print(f"[Pipeline] Objects: target={obj_sets['target_ids']}, "
              f"fragile={obj_sets.get('fragile_ids', [])}, "
              f"clutter={obj_sets.get('clutter_ids', [])}")

        ctx.target_obj = get_scope_obj(ctx.env, obj_sets["target_ids"][0])
        ctx._obj_sets = obj_sets

        # Identify the pinch fragile (first wineglass).
        ctx._pinch_inst = None
        for fid in obj_sets["fragile_ids"]:
            if "wineglass" in fid:
                ctx._pinch_inst = fid
                break
        if ctx._pinch_inst is None and obj_sets["fragile_ids"]:
            ctx._pinch_inst = obj_sets["fragile_ids"][0]

        if ctx._pinch_inst:
            ctx._pinch_obj = get_scope_obj(ctx.env, ctx._pinch_inst)
            print(f"[Pipeline] Pinch fragile: {ctx._pinch_inst}")
        else:
            ctx._pinch_obj = None
            print("[Pipeline] WARNING: No pinch fragile found")

        descriptors, objects_by_inst = build_descriptors(ctx.env, obj_sets)
        if not descriptors:
            raise RuntimeError("No clutter-pack descriptors created.")
        ctx._descriptors = descriptors
        ctx._objects_by_inst = objects_by_inst

    def place_objects(self, ctx):
        import torch as th
        from omnigibson.utils.clutter_pack_layout import validate_pack_integrity
        from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone
        from omnigibson.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop

        args = ctx.args

        obstacle_bounds_xy = None
        if ctx.surface_info and ctx.surface_info.obstacles:
            obstacle_bounds_xy = ctx.surface_info.obstacles[0].aabb_xy

        zone = compute_tabletop_zone(
            surface_bounds_xy=ctx.surface_bounds_xy,
            obstacle_bounds_xy=obstacle_bounds_xy,
            edge_margin_m=0.04,
            obstacle_keepout_margin_m=0.08,
            obstacle_side_clearance_m=0.015,
        )
        ctx._zone = zone

        pack_config = PackRetryConfig(
            pack_jitter_xy=args.pack_jitter_xy or 0.022,
            pack_min_clearance=args.pack_min_clearance or 0.008,
        )
        settle_fn = make_settle_fn(ctx.og, th)
        park_fn = make_park_fn(ctx.og, zone.surface_bounds, ctx.floor_z)

        pack_result = run_pack_retry_loop(
            support_name=getattr(ctx.support_obj, "name", "support"),
            descriptors=ctx._descriptors,
            objects_by_inst=ctx._objects_by_inst,
            red_zone_bounds=zone.red_zone_bounds,
            table_top_z=ctx.table_top_z,
            floor_z=ctx.floor_z,
            config=pack_config,
            base_seed=args.seed,
            episode=ctx.episode,
            settle_fn=settle_fn,
            park_fn=park_fn,
            validate_poses_fn=validate_poses,
            check_interpenetration_fn=check_interpenetration,
            obstacle_keepout_bounds=zone.obstacle_keepout_bounds,
        )
        ctx._pack_result = pack_result
        print(f"[Pipeline] Pack solved: attempt={pack_result.attempt_used}, "
              f"active={len(pack_result.active_descriptors)}")

        # Remove ALL objects except target and pinch fragile.
        keep_ids = set(ctx._obj_sets["target_ids"])
        if ctx._pinch_inst:
            keep_ids.add(ctx._pinch_inst)
        to_remove = {i: o for i, o in ctx._objects_by_inst.items()
                     if i not in keep_ids}
        remove_objects(ctx.og, to_remove)

        # Place pinch fragile next to target handle.
        ctx._pinch_direction = None
        if ctx._pinch_obj is not None:
            ctx._pinch_direction = _place_pinch_fragile(
                ctx._pinch_obj, ctx.target_obj, ctx.og,
                gap_m=args.pinch_gap_m,
            )
            settle_fn({ctx._pinch_inst: ctx._pinch_obj})

        # Integrity check.
        ctx._integrity = validate_pack_integrity(
            pack_spec=pack_result.pack_spec,
            world_positions=pack_result.world_positions,
            pack_origin_world=pack_result.pack_origin,
            pack_yaw=0.0, tol_xy=pack_config.integrity_tol_xy,
        )

        # Only kept objects participate in LTL rollout.
        ctx.active_objects = {i: ctx._objects_by_inst[i] for i in keep_ids
                              if i in ctx._objects_by_inst}

    def make_edge_objects(self, ctx):
        from omnigibson.utils.franka_edge_align import EdgeAlignObject

        result = []
        for inst, obj in ctx.active_objects.items():
            try:
                pos = obj.get_position_orientation()[0]
                role = ("target" if inst in ctx._obj_sets["target_ids"]
                        else "fragile")
                result.append(EdgeAlignObject(
                    name=inst, role=role,
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            except Exception:
                continue
        if not result:
            raise RuntimeError("No pack objects for edge alignment.")
        return tuple(result)

    def extra_gate_checks(self, ctx):
        return getattr(ctx, "_integrity", None) is not None and ctx._integrity.ok

    def diagnostics_extra(self, ctx):
        return {
            "density": getattr(ctx.args, "clutter_density", None),
            "pinch_fragile": getattr(ctx, "_pinch_inst", None),
            "pinch_direction": getattr(ctx, "_pinch_direction", None),
            "pinch_gap_m": getattr(ctx.args, "pinch_gap_m", None),
            "pack_attempt_used": getattr(ctx._pack_result, "attempt_used", None)
            if hasattr(ctx, "_pack_result") else None,
        }


def main():
    PinchPointPipeline().run()


if __name__ == "__main__":
    main()
