"""Table clutter scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json,
packs clutter objects, places robot, and runs LTL-monitored rollouts.

Usage:
    python -m omnigibson.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

import math
import numpy as np

from omnigibson.task_generation.pipeline_common import (
    BasePipeline,
    build_descriptors,
    build_task_object_sets,
    check_interpenetration,
    estimate_surface_area_from_scene_json,
    get_scene_json_path,
    get_scope_obj,
    iter_scope_objects,
    make_park_fn,
    make_settle_fn,
    remove_objects,
    validate_poses,
)
from omnigibson.utils.tabletop_workspace import bounds_overlap
from omnigibson.utils.bddl_generator import generate_clutter_activity


def _quat_tilt_deg(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    zz = 1.0 - 2.0 * (x * x + y * y)
    zz = max(-1.0, min(1.0, zz))
    return math.degrees(math.acos(zz))


def _collect_resident_surface_objects(ctx):
    scope_names = {getattr(obj, "name", "") for _, obj in iter_scope_objects(ctx.env)}
    support_name = getattr(ctx.support_obj, "name", "")
    residents = []
    for obj in ctx.env.scene.objects:
        name = getattr(obj, "name", "")
        if not name or name == support_name or name in scope_names:
            continue
        try:
            aabb_min, aabb_max = obj.aabb
            bounds_xy = (
                (float(aabb_min[0]), float(aabb_min[1])),
                (float(aabb_max[0]), float(aabb_max[1])),
            )
            bottom_z = float(aabb_min[2])
            top_z = float(aabb_max[2])
            position, orientation = obj.get_position_orientation()
        except Exception:
            continue
        if not bounds_overlap(bounds_xy, ctx.surface_bounds_xy, tol=1e-6):
            continue
        if abs(bottom_z - ctx.table_top_z) > 0.12:
            continue
        residents.append({
            "name": name,
            "category": str(getattr(obj, "category", "")),
            "obj": obj,
            "aabb_xy": bounds_xy,
            "position": tuple(float(v) for v in position[:3]),
            "orientation": tuple(float(v) for v in orientation[:4]),
            "tilt_deg": _quat_tilt_deg(orientation[:4]),
            "top_z": top_z,
            "bottom_z": bottom_z,
        })
    return residents


def _resident_stability_error(entries, *, max_xy_shift_m=0.03, max_z_shift_m=0.03, max_tilt_delta_deg=12.0):
    for entry in entries:
        obj = entry["obj"]
        try:
            position, orientation = obj.get_position_orientation()
        except Exception:
            return f"resident_pose_error:{entry['name']}"
        dx = float(position[0]) - entry["position"][0]
        dy = float(position[1]) - entry["position"][1]
        dz = float(position[2]) - entry["position"][2]
        xy_shift = math.hypot(dx, dy)
        tilt_deg = _quat_tilt_deg(orientation[:4])
        if xy_shift > max_xy_shift_m:
            return f"resident_xy_shift:{entry['name']}:{xy_shift:.3f}"
        if abs(dz) > max_z_shift_m:
            return f"resident_z_shift:{entry['name']}:{dz:.3f}"
        if tilt_deg - entry["tilt_deg"] > max_tilt_delta_deg:
            return f"resident_tilt_delta:{entry['name']}:{tilt_deg - entry['tilt_deg']:.1f}"
    return None


class ClutterPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--randomize", action="store_true",
                            help="Randomize target, fragile, and clutter object "
                                 "types each episode")

    def activity_prefix(self):
        return "auto_clutter_on"

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        # Estimate surface area for area-aware object budgeting.
        surface_area = None
        if args.scene_model:
            try:
                scene_json = get_scene_json_path(args.scene_model)
                from omnigibson.task_generation.pipeline_common import (
                    discover_surface_from_scene_json,
                )
                discovery = discover_surface_from_scene_json(scene_json)
                if discovery:
                    surface_area = estimate_surface_area_from_scene_json(
                        scene_json, discovery[0],
                    )
            except Exception:
                pass

        return generate_clutter_activity(
            activity_name, support_synset, support_room,
            args.clutter_density, rng=rng,
            available_area_m2=surface_area,
        )

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

        descriptors, objects_by_inst = build_descriptors(ctx.env, obj_sets)
        if not descriptors:
            raise RuntimeError("No clutter-pack descriptors created.")
        ctx._descriptors = descriptors
        ctx._objects_by_inst = objects_by_inst

    def place_objects(self, ctx):
        import torch as th
        from omnigibson.object_states import OnTop, Upright
        from omnigibson.utils.clutter_pack_layout import validate_pack_integrity
        from omnigibson.utils.tabletop_workspace import compute_tabletop_zone
        from omnigibson.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop

        args = ctx.args

        # Compute zone (needed for pack layout).
        obstacle_bounds_xy = None
        obstacle_bounds_seq = []
        if ctx.curation and getattr(ctx.curation, "obstacle_bounds_override_xy", None):
            obstacle_bounds_xy = ctx.curation.obstacle_bounds_override_xy
            print(f"[Pipeline] Obstacle bounds override: {obstacle_bounds_xy}")
            obstacle_bounds_seq.append(obstacle_bounds_xy)
        elif ctx.surface_info and ctx.surface_info.obstacles:
            obstacle_bounds_seq.extend(obstacle.aabb_xy for obstacle in ctx.surface_info.obstacles)

        ctx._resident_surface_entries = []
        if ctx.curation and getattr(ctx.curation, "use_resident_surface_obstacles", False):
            ctx._resident_surface_entries = _collect_resident_surface_objects(ctx)
            obstacle_bounds_seq.extend(entry["aabb_xy"] for entry in ctx._resident_surface_entries)
            print(
                "[Pipeline] Resident surface objects: "
                + (
                    ", ".join(f"{entry['name']}<{entry['category']}>" for entry in ctx._resident_surface_entries)
                    if ctx._resident_surface_entries else "<none>"
                )
            )

        zone = compute_tabletop_zone(
            surface_bounds_xy=ctx.surface_bounds_xy,
            obstacle_bounds_xy=obstacle_bounds_xy,
            obstacle_bounds_seq=tuple(obstacle_bounds_seq),
            edge_margin_m=args.zone_edge_margin_m or 0.04,
            obstacle_keepout_margin_m=args.obstacle_keepout_margin_m or 0.08,
            obstacle_side_clearance_m=args.obstacle_side_clearance_m or 0.015,
        )
        ctx._zone = zone
        print(f"[Pipeline] Zone: red_zone={zone.red_zone_bounds}, "
              f"long_axis={zone.long_axis}")

        pack_config = PackRetryConfig(
            pack_jitter_xy=args.pack_jitter_xy or 0.022,
            pack_min_clearance=args.pack_min_clearance or 0.008,
            pack_clearance_step_m=getattr(args, "pack_clearance_step_m", None) or 0.005,
            pack_clearance_floor_m=getattr(args, "pack_clearance_floor_m", None) or 0.002,
            pack_clearance_search_mode=getattr(args, "pack_clearance_search_mode", None) or "shrink_from_start",
        )
        settle_fn = make_settle_fn(ctx.og, th)
        park_fn = make_park_fn(ctx.og, zone.surface_bounds, ctx.floor_z)

        extra_validation_fn = None
        if ctx.curation and ctx.curation.require_support_and_upright_after_pack:
            role_by_inst = {d.instance_id: d.role for d in ctx._descriptors}

            def _validate_support_and_upright(active_objects):
                for inst_id, obj in active_objects.items():
                    try:
                        if not bool(obj.states[OnTop].get_value(ctx.support_obj)):
                            return f"support_state_failed:{inst_id}"
                    except Exception:
                        return f"support_state_error:{inst_id}"
                    if role_by_inst.get(inst_id) not in {"target", "fragile"}:
                        continue
                    try:
                        if not bool(obj.states[Upright].get_value()):
                            return f"upright_state_failed:{inst_id}"
                    except Exception:
                        return f"upright_state_error:{inst_id}"
                    if getattr(zone, "obstacle_keepout_bounds_seq", ()):
                        try:
                            obj_aabb_min, obj_aabb_max = obj.aabb
                            obj_bounds = (
                                (float(obj_aabb_min[0]), float(obj_aabb_min[1])),
                                (float(obj_aabb_max[0]), float(obj_aabb_max[1])),
                            )
                        except Exception:
                            return f"object_aabb_error:{inst_id}"
                        for keepout in zone.obstacle_keepout_bounds_seq:
                            if bounds_overlap(obj_bounds, keepout, tol=1e-6):
                                return f"resident_keepout_overlap:{inst_id}"
                if ctx.curation and getattr(ctx.curation, "require_resident_surface_stability", False):
                    resident_error = _resident_stability_error(ctx._resident_surface_entries)
                    if resident_error:
                        return resident_error
                return None

            extra_validation_fn = _validate_support_and_upright

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
            obstacle_keepout_bounds_seq=getattr(zone, "obstacle_keepout_bounds_seq", ()),
            extra_validation_fn=extra_validation_fn,
        )
        ctx._pack_result = pack_result
        print(f"[Pipeline] Pack solved: attempt={pack_result.attempt_used}, "
              f"active={len(pack_result.active_descriptors)}")

        # Remove inactive objects from the scene (they were parked during retries).
        passive = {i: o for i, o in ctx._objects_by_inst.items()
                   if i not in pack_result.active_objects_by_inst}
        remove_objects(ctx.og, passive)

        # Integrity check.
        ctx._integrity = validate_pack_integrity(
            pack_spec=pack_result.pack_spec,
            world_positions=pack_result.world_positions,
            pack_origin_world=pack_result.pack_origin,
            pack_yaw=0.0, tol_xy=pack_config.integrity_tol_xy,
        )

        # active_objects for the LTL rollout.
        ctx.active_objects = pack_result.active_objects_by_inst
        ctx._active_object_summary = [
            {
                "inst_id": inst_id,
                "scene_object_name": getattr(obj, "name", None),
                "category": str(getattr(obj, "category", "")),
                "role": next(
                    (d.role for d in pack_result.active_descriptors if d.instance_id == inst_id),
                    None,
                ),
            }
            for inst_id, obj in sorted(ctx.active_objects.items())
        ]
        print(f"[Pipeline] Active object summary: {ctx._active_object_summary}")

    def make_edge_objects(self, ctx):
        from omnigibson.utils.franka_edge_align import EdgeAlignObject

        pack_result = ctx._pack_result
        objects = tuple(
            EdgeAlignObject(
                name=e.inst_id, role=e.role,
                position_xy=(pack_result.world_positions[e.inst_id][0],
                             pack_result.world_positions[e.inst_id][1]),
            )
            for e in pack_result.pack_spec.object_entries
            if e.inst_id in pack_result.world_positions
        )
        if not objects:
            raise RuntimeError("No pack objects for edge alignment.")
        return objects

    def extra_gate_checks(self, ctx):
        if getattr(ctx, "_integrity", None) is None or not ctx._integrity.ok:
            return False
        if ctx.curation and getattr(ctx.curation, "require_resident_surface_stability", False):
            resident_error = _resident_stability_error(getattr(ctx, "_resident_surface_entries", ()))
            if resident_error:
                ctx._resident_gate_error = resident_error
                print(f"[Pipeline] Resident stability gate failed: {resident_error}")
                return False
        return True

    def diagnostics_extra(self, ctx):
        extra = {"density": getattr(ctx.args, "clutter_density", None)}
        if hasattr(ctx, "_pack_result"):
            extra["pack_attempt_used"] = ctx._pack_result.attempt_used
            extra["pack_clearance_used"] = ctx._pack_result.clearance_used
        if hasattr(ctx, "_active_object_summary"):
            extra["active_object_summary"] = ctx._active_object_summary
        sel = ctx.selection
        if sel is not None:
            extra["selection"] = sel
        if hasattr(ctx, "_resident_surface_entries"):
            extra["resident_surface_objects"] = [
                {
                    "name": entry["name"],
                    "category": entry["category"],
                }
                for entry in ctx._resident_surface_entries
            ]
        if hasattr(ctx, "_resident_gate_error"):
            extra["resident_gate_error"] = ctx._resident_gate_error
        if getattr(ctx, "removed_area_objects", None):
            extra["removed_area_objects"] = list(ctx.removed_area_objects)
        if getattr(ctx, "removed_robot_base_objects", None):
            extra["removed_robot_base_objects"] = list(ctx.removed_robot_base_objects)
        if getattr(ctx, "resolved_video_views", None):
            extra["resolved_video_views"] = list(ctx.resolved_video_views)
        if getattr(ctx.args, "video_candidate_mode", None) is not None:
            extra["video_candidate_mode"] = ctx.args.video_candidate_mode
        return extra


def main():
    ClutterPipeline().run()


if __name__ == "__main__":
    main()
