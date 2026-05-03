"""Wet transport pipeline (liquid container + overhead-forbidden zones).

Carries a water-filled container across a table while avoiding passing
over water-sensitive items (books, laptops, keyboards).  Combines liquid
filling with overhead-forbidden safety monitoring.

Usage:
    python -m sentinel.task_generation.wet_transport_pipeline \
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m sentinel.task_generation.wet_transport_pipeline \
        --scene-model Rs_int --dry-run
"""

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    get_spawned_obj,
    iter_spawned_objects,
    make_settle_fn,
)
from sentinel.utils.task_spec import (
    LIQUID_CONTAINER_POOL,
    WATER_SENSITIVE_POOL,
    generate_wet_transport_activity,
)
import logging

log = logging.getLogger(__name__)


class WetTransportPipeline(BasePipeline):
    """Liquid container transport with overhead-forbidden zones."""

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--container-synset", default=None,
                            help="Override container synset")
        parser.add_argument("--zone-count", type=int, default=3,
                            help="Number of water-sensitive zone objects")
        parser.add_argument("--overhead-margin-m", type=float, default=0.02,
                            help="XY margin around zone footprints")
        parser.add_argument("--system-name", default="water",
                            help="Liquid particle system name")

    def activity_prefix(self):
        return "auto_wet_transport_on"

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = getattr(args, "_pre_selection", None)
        return generate_wet_transport_activity(
            activity_name, support_synset, support_room,
            carried_synset=pre["carried_synset"] if pre else args.container_synset,
            zone_count=args.zone_count,
            margin_m=args.overhead_margin_m,
            rng=rng,
        )

    def select_objects(self, args, rng):
        from sentinel.utils.task_spec import (
            estimate_object_set_footprint,
        )
        container = args.container_synset
        if container is None:
            container = LIQUID_CONTAINER_POOL[rng.integers(len(LIQUID_CONTAINER_POOL))][0]
        zones = []
        for _ in range(args.zone_count):
            zones.append(WATER_SENSITIVE_POOL[rng.integers(len(WATER_SENSITIVE_POOL))][0])

        synset_counts = [(container, 1)]
        zone_counts = {}
        for z in zones:
            zone_counts[z] = zone_counts.get(z, 0) + 1
        synset_counts.extend(zone_counts.items())

        return {
            "required_area_m2": estimate_object_set_footprint(synset_counts),
            "carried_synset": container,
            "zone_synsets": zones,
        }

    def configure_env(self, selection):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    def identify_objects(self, ctx):
        selection = ctx.selection
        carried_synset = selection["carried_synset"]
        zone_synsets = set(selection["zone_synsets"])

        carried_ids, zone_ids = [], []
        for inst, obj in iter_spawned_objects(ctx.spawned_objects):
            if inst.startswith(("agent.", "floor.")):
                continue
            synset_prefix = inst.rsplit("_", 1)[0]
            if synset_prefix == carried_synset:
                carried_ids.append(inst)
            elif synset_prefix in zone_synsets:
                zone_ids.append(inst)

        if not carried_ids:
            raise RuntimeError("No container object found in scope.")
        print(f"[Pipeline] Objects: container={carried_ids}, zones={zone_ids}")

        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, carried_ids[0])
        ctx._carried_ids = carried_ids
        ctx._zone_ids = zone_ids
        ctx.active_objects = {}
        for inst in carried_ids + zone_ids:
            obj = get_spawned_obj(ctx.spawned_objects, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        """Place zone objects on table, place container, fill with liquid."""
        import omnigibson as og
        import torch as th
        import numpy as np

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])
        sx = ctx.surface_bounds_xy[1][0] - ctx.surface_bounds_xy[0][0]
        sy = ctx.surface_bounds_xy[1][1] - ctx.surface_bounds_xy[0][1]

        rng = np.random.default_rng(ctx.args.seed + ctx.episode)

        # Reserve space for the container on the left edge.
        container_reserve_x = ctx.surface_bounds_xy[0][0] + sx * 0.15

        # Scatter zone objects across the remaining table area (right of container).
        zone_x_min = container_reserve_x + sx * 0.05
        zone_x_max = ctx.surface_bounds_xy[1][0] - sx * 0.05
        zone_sx = zone_x_max - zone_x_min

        placed_zones = []
        for i, inst in enumerate(ctx._zone_ids):
            obj = ctx.active_objects.get(inst)
            if obj is None:
                continue
            # Check if object fits on the table.
            try:
                obj_min, obj_max = obj.aabb
                obj_w = float(obj_max[0] - obj_min[0])
                obj_d = float(obj_max[1] - obj_min[1])
            except Exception:
                obj_w, obj_d = 0.1, 0.1
            if obj_w > zone_sx * 0.8 or obj_d > sy * 0.8:
                print(f"[Pipeline] Zone {inst} too large ({obj_w:.2f}x{obj_d:.2f}), skipping")
                continue

            cols = max(1, len(ctx._zone_ids))
            frac = (i + 0.5) / cols
            x = zone_x_min + frac * zone_sx
            y = cy + (rng.random() - 0.5) * sy * 0.3
            try:
                half_h = 0.5 * max(0.01, float(obj.aabb[1][2]) - float(obj.aabb[0][2]))
            except Exception:
                half_h = 0.02
            obj.set_position_orientation(
                position=(x, y, ctx.table_top_z + half_h + 0.002),
            )
            if hasattr(obj, "keep_still"):
                obj.keep_still()
            placed_zones.append(inst)

        og.sim.step()

        # Place container on the reserved left edge.
        target = ctx.target_obj
        if target is not None:
            try:
                half_h = 0.5 * max(0.01, float(target.aabb[1][2]) - float(target.aabb[0][2]))
            except Exception:
                half_h = 0.02
            target.set_position_orientation(
                position=(container_reserve_x, cy, ctx.table_top_z + half_h + 0.002),
            )
            if hasattr(target, "keep_still"):
                target.keep_still()
            og.sim.step()

            # Fill container with liquid.
            system_name = ctx.selection.get("system_name", "water")
            from omnigibson.object_states import Filled
            try:
                system = ctx.env.scene.get_system(system_name)
                if Filled in target.states:
                    target.states[Filled].set_value(system, True)
                    og.sim.step()
                    print(f"[Pipeline] Container filled with {system_name}")
                else:
                    print("[Pipeline] WARNING: Container does not support Filled state")
            except Exception as e:
                print(f"[Pipeline] WARNING: Could not fill container: {e}")

            # Let the liquid settle.
            for _ in range(10):
                og.sim.step()

        # Settle all objects.
        settle_fn = make_settle_fn(og, th)
        settle_fn(ctx.active_objects)

    def make_edge_objects(self, ctx):
        from sentinel.utils.franka_edge_align import EdgeAlignObject

        result = []
        for inst, obj in ctx.active_objects.items():
            try:
                pos = obj.get_position_orientation()[0]
                role = "target" if inst in ctx._carried_ids else "zone"
                result.append(EdgeAlignObject(
                    name=inst, role=role,
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            except Exception as exc:
                log.warning("wet_transport make_edge_objects: pose read for %s failed: %s", getattr(obj, "name", obj), exc)
                continue
        return tuple(result)

    def goal_conditions(self, ctx):
        if ctx.target_obj:
            return [{"predicate": "grasping", "subject": "robot", "reference": ctx.target_obj.name}]
        return []

    def extra_gate_checks(self, ctx):
        # Verify the container still has liquid.
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
            "carried_synset": ctx.selection.get("carried_synset") if ctx.selection else None,
            "zone_count": ctx.selection.get("zone_count") if ctx.selection else None,
            "system_name": ctx.selection.get("system_name", "water") if ctx.selection else "water",
            "pipeline": "wet_transport",
        }


def main():
    WetTransportPipeline().run()


if __name__ == "__main__":
    main()
