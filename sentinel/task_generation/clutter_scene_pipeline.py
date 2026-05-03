"""Table clutter scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json,
packs clutter objects, places robot, and runs LTL-monitored rollouts.

Usage:
    python -m sentinel.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m sentinel.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    build_descriptors,
    build_task_object_sets,
    check_interpenetration,
    get_spawned_obj,
    make_park_fn,
    make_settle_fn,
    remove_objects,
    validate_poses,
)
from sentinel.utils.task_spec import generate_clutter_activity
import logging

log = logging.getLogger(__name__)


class ClutterPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--randomize", action="store_true",
                            help="Randomize target, fragile, and clutter object "
                                 "types each episode")

    def activity_prefix(self):
        return "auto_clutter_on"

    def scene_family(self, ctx):
        return "table"

    def select_objects(self, args, rng):
        from sentinel.utils.task_spec import (
            TARGET_POOL, FRAGILE_POOL, CLUTTER_POOL, DENSITY_PRESETS,
            estimate_object_set_footprint,
        )
        density = DENSITY_PRESETS[args.clutter_density]
        target_synset = TARGET_POOL[rng.integers(len(TARGET_POOL))][0]

        fragile_pool = [s for s in FRAGILE_POOL if s[0] != target_synset] or list(FRAGILE_POOL)
        fragile_picks = [fragile_pool[rng.integers(len(fragile_pool))][0]
                         for _ in range(density["fragile_count"])]

        clutter_picks = [CLUTTER_POOL[rng.integers(len(CLUTTER_POOL))][0]
                         for _ in range(density["clutter_count"])]

        synset_counts = [(target_synset, 1)]
        for s in set(fragile_picks):
            synset_counts.append((s, fragile_picks.count(s)))
        for s in set(clutter_picks):
            synset_counts.append((s, clutter_picks.count(s)))

        return {
            "required_area_m2": estimate_object_set_footprint(synset_counts),
            "target_synset": target_synset,
            "fragile_picks": fragile_picks,
            "clutter_picks": clutter_picks,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        return generate_clutter_activity(
            activity_name, support_synset, support_room,
            args.clutter_density, rng=rng,
            pre_selection=args._pre_selection,
        )

    def identify_objects(self, ctx):
        obj_sets = build_task_object_sets(ctx.spawned_objects)

        if not obj_sets["target_ids"]:
            raise RuntimeError("No target objects found.")
        print(f"[Pipeline] Objects: target={obj_sets['target_ids']}, "
              f"fragile={obj_sets.get('fragile_ids', [])}, "
              f"clutter={obj_sets.get('clutter_ids', [])}")

        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, obj_sets["target_ids"][0])
        ctx._obj_sets = obj_sets

        descriptors, objects_by_inst = build_descriptors(ctx.spawned_objects, obj_sets)
        if not descriptors:
            raise RuntimeError("No clutter-pack descriptors created.")
        ctx._descriptors = descriptors
        ctx._objects_by_inst = objects_by_inst

    def place_objects(self, ctx):
        import torch as th
        from sentinel.utils.clutter_pack_layout import validate_pack_integrity
        from sentinel.utils.tabletop_workspace import compute_tabletop_zone
        from sentinel.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop

        args = ctx.args

        # Compute zone (needed for pack layout).
        obstacle_bounds_xy = None
        obstacle_bounds_seq = []
        if ctx.surface_info and ctx.surface_info.obstacles:
            obstacle_bounds_seq.extend(obstacle.aabb_xy for obstacle in ctx.surface_info.obstacles)

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
        from sentinel.utils.franka_edge_align import EdgeAlignObject

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

    def goal_conditions(self, ctx):
        if ctx.target_obj:
            return [{"predicate": "grasping", "subject": "robot", "reference": ctx.target_obj.name}]
        return []

    def extra_gate_checks(self, ctx):
        if getattr(ctx, "_integrity", None) is None or not ctx._integrity.ok:
            return False
        return True

    def diagnostics_extra(self, ctx):
        extra = {"density": getattr(ctx.args, "clutter_density", None), "pipeline": "table"}
        if hasattr(ctx, "_pack_result"):
            extra["pack_attempt_used"] = ctx._pack_result.attempt_used
            extra["pack_clearance_used"] = ctx._pack_result.clearance_used
        if hasattr(ctx, "_active_object_summary"):
            extra["active_object_summary"] = ctx._active_object_summary
        sel = ctx.selection
        if sel is not None:
            extra["selection"] = sel
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
