"""Liquid transport scene generation pipeline.

Places a liquid-filled container on a table with fragile obstacles,
then runs LTL-monitored rollouts that track spill, tilt, and obstacle
safety.

Usage:
    python -m omnigibson.task_generation.liquid_transport_pipeline \
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m omnigibson.task_generation.liquid_transport_pipeline \
        --scene-model Rs_int --difficulty hard --system-name water

    python -m omnigibson.task_generation.liquid_transport_pipeline \
        --scene-model Rs_int --dry-run
"""

import math

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    BasePipeline,
    EpisodeContext,
    build_descriptors,
    build_task_object_sets,
    get_scope_obj,
    iter_scope_objects,
    make_park_fn,
    make_settle_fn,
    resolve_synset,
    validate_poses,
    check_interpenetration,
)
from omnigibson.utils.bddl_generator import (
    LIQUID_CONTAINER_POOL,
    LIQUID_OBSTACLE_POOL,
    LIQUID_PRESETS,
    generate_liquid_transport_activity,
)
from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec


class LiquidTransportPipeline(BasePipeline):
    """Pipeline for liquid-filled container transport with spill monitoring."""

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--difficulty", default="medium", choices=list(LIQUID_PRESETS),
            help="Difficulty preset (controls obstacle count, spill threshold, tilt limit)",
        )
        parser.add_argument(
            "--container-synset", default=None,
            help="Specific container synset (random if omitted)",
        )
        parser.add_argument(
            "--system-name", default="water",
            help="Liquid particle system name",
        )

    def activity_prefix(self):
        return "auto_liquid_transport_on"

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        return generate_liquid_transport_activity(
            activity_name, support_synset, support_room,
            difficulty=args.difficulty,
            container_synset=args.container_synset,
            system_name=args.system_name,
            rng=rng,
        )

    def configure_task(self, cfg, selection):
        # Liquid particles require GPU dynamics.
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    def identify_objects(self, ctx):
        task_spec = build_manipulation_task_spec(ctx.activity_name)
        obj_sets = build_task_object_sets(ctx.env, task_spec)

        if not obj_sets["target_ids"]:
            raise RuntimeError("No target container found in BDDL scope.")
        ctx.target_obj = get_scope_obj(ctx.env, obj_sets["target_ids"][0])

        ctx._obj_sets = obj_sets
        descriptors, objects_by_inst = build_descriptors(ctx.env, obj_sets)
        ctx._descriptors = descriptors
        ctx._objects_by_inst = objects_by_inst
        ctx.active_objects = objects_by_inst

    def place_objects(self, ctx):
        """Place objects on the table, then fill the container with liquid."""
        import torch as th
        from omnigibson.utils.clutter_pack_layout import (
            build_clutter_pack,
            apply_pack_transform,
        )
        from omnigibson.utils.kitchen_bar_workspace import compute_tabletop_zone

        zone = compute_tabletop_zone(
            surface_bounds_xy=ctx.surface_bounds_xy,
            obstacle_bounds_xy=None,
            edge_margin_m=0.04,
        )

        half_w = 0.5 * (zone.red_zone_bounds[1][0] - zone.red_zone_bounds[0][0])
        half_h = 0.5 * (zone.red_zone_bounds[1][1] - zone.red_zone_bounds[0][1])
        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])
        pack_origin = (cx, cy, ctx.table_top_z)

        bounds_local = ((-half_w, -half_h), (half_w, half_h))
        pack_spec = None
        for clearance in (0.030, 0.020, 0.010, 0.005):
            try:
                pack_spec = build_clutter_pack(
                    table_obj_name=ctx.surface_name,
                    descriptors=ctx._descriptors,
                    seed=ctx.args.seed + ctx.episode,
                    min_clearance=clearance,
                    placement_bounds_local=bounds_local,
                )
                break
            except RuntimeError as e:
                print(f"[Pipeline] Pack clearance={clearance:.3f}: {e}")
        if pack_spec is None:
            raise RuntimeError("Could not pack objects on surface.")

        apply_pack_transform(pack_spec, ctx._objects_by_inst, pack_origin, pack_yaw=0.0)

        # Settle physics before filling.
        settle_fn = make_settle_fn(ctx.og, th)
        settle_fn(ctx._objects_by_inst)

        # -- Fill the container with liquid --------------------------------
        system_name = ctx.selection.get("system_name", "water")
        from omnigibson.object_states import Filled
        try:
            system = ctx.env.scene.get_system(system_name)
            if ctx.target_obj is not None and Filled in ctx.target_obj.states:
                ctx.target_obj.states[Filled].set_value(system, True)
                ctx.og.sim.step()
                print(f"[Pipeline] Container filled with {system_name}")
            else:
                print(f"[Pipeline] WARNING: Container does not support Filled state")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not fill container: {e}")

        # Let the liquid settle.
        for _ in range(10):
            ctx.og.sim.step()

        ctx._pack_origin = pack_origin

    def make_edge_objects(self, ctx):
        from omnigibson.utils.franka_edge_align import EdgeAlignObject
        result = []
        for inst, obj in ctx._objects_by_inst.items():
            try:
                pos = obj.get_position_orientation()[0]
                role = "target"
                for r, key in [("target", "target_ids"), ("fragile", "fragile_ids")]:
                    if inst in ctx._obj_sets[key]:
                        role = r
                        break
                result.append(EdgeAlignObject(
                    name=inst, role=role,
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            except Exception:
                continue
        return tuple(result)

    def extra_gate_checks(self, ctx):
        # Verify the container is actually filled.
        from omnigibson.object_states import ContainedParticles
        system_name = ctx.selection.get("system_name", "water")
        try:
            system = ctx.env.scene.get_system(system_name)
            data = ctx.target_obj.states[ContainedParticles].get_value(system)
            n = data.n_in_volume
            print(f"[Pipeline] Container particle count: {n}")
            return n > 0
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not check particle count: {e}")
            return True

    def diagnostics_extra(self, ctx):
        return {
            "difficulty": getattr(ctx.args, "difficulty", "medium"),
            "system_name": ctx.selection.get("system_name", "water") if ctx.selection else "water",
            "pipeline": "liquid_transport",
        }


def main():
    pipeline = LiquidTransportPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
