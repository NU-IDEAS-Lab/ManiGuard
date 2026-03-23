"""Table clutter scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json,
packs clutter objects, places robot, and runs LTL-monitored rollouts.

Usage:
    python -m omnigibson.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    BasePipeline,
    build_descriptors,
    build_task_object_sets,
    check_interpenetration,
    estimate_surface_area_from_scene_json,
    get_scene_json_path,
    get_scope_obj,
    make_park_fn,
    make_settle_fn,
    remove_objects,
    validate_poses,
)
from omnigibson.utils.bddl_generator import generate_clutter_activity


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
        from omnigibson.utils.clutter_pack_layout import validate_pack_integrity
        from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone
        from omnigibson.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop

        args = ctx.args

        # Compute zone (needed for pack layout).
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
        print(f"[Pipeline] Zone: red_zone={zone.red_zone_bounds}, "
              f"long_axis={zone.long_axis}")

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
        return getattr(ctx, "_integrity", None) is not None and ctx._integrity.ok

    def diagnostics_extra(self, ctx):
        extra = {"density": getattr(ctx.args, "clutter_density", None)}
        if hasattr(ctx, "_pack_result"):
            extra["pack_attempt_used"] = ctx._pack_result.attempt_used
        sel = ctx.selection
        if sel is not None:
            extra["selection"] = sel
        return extra


def main():
    ClutterPipeline().run()


if __name__ == "__main__":
    main()
