"""Empty-before-invert pipeline (temporal Until + particles on surface).

A liquid-filled container sits on a table.  The robot must empty the
container before inverting it (placing it upside down).  The table
surface must remain dry throughout.

Usage:
    python -m sentinel.task_generation.empty_invert_pipeline \
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m sentinel.task_generation.empty_invert_pipeline --dry-run
"""

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    get_scope_obj,
    iter_scope_objects,
    make_settle_fn,
)
from sentinel.utils.bddl_generator import (
    INVERT_CONTAINER_POOL,
    estimate_object_set_footprint,
    generate_empty_invert_activity,
)
import logging

log = logging.getLogger(__name__)


class EmptyInvertPipeline(BasePipeline):
    """Empty-before-invert with temporal Until + table-stays-dry."""

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--container-synset", default=None,
                            help="Override container synset")
        parser.add_argument("--system-name", default="water",
                            help="Liquid particle system name")

    def activity_prefix(self):
        return "auto_empty_invert_on"

    def select_objects(self, args, rng):
        container = args.container_synset or \
            INVERT_CONTAINER_POOL[rng.integers(len(INVERT_CONTAINER_POOL))][0]
        synset_counts = [(container, 1)]
        return {
            "required_area_m2": estimate_object_set_footprint(synset_counts),
            "container_synset": container,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = args._pre_selection
        return generate_empty_invert_activity(
            activity_name, support_synset, support_room,
            container_synset=pre["container_synset"],
            system_name=args.system_name,
            rng=rng,
        )

    def configure_task(self, cfg, selection):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    def identify_objects(self, ctx):
        selection = ctx.selection
        container_synset = selection["container_synset"]

        container_ids = []
        for inst, obj in iter_scope_objects(ctx.env):
            if inst.startswith(("agent.", "floor.")):
                continue
            if inst.startswith(container_synset + "_"):
                container_ids.append(inst)

        if not container_ids:
            raise RuntimeError("No container found in scope.")
        print(f"[Pipeline] Objects: container={container_ids}")

        ctx.target_obj = get_scope_obj(ctx.env, container_ids[0])
        ctx._container_ids = container_ids
        ctx.active_objects = {}
        for inst in container_ids:
            obj = get_scope_obj(ctx.env, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        """Place container on table and fill with liquid."""
        import omnigibson as og
        import torch as th

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])

        container = ctx.target_obj
        if container is not None:
            try:
                half_h = 0.5 * max(0.01, float(container.aabb[1][2]) - float(container.aabb[0][2]))
            except Exception:
                half_h = 0.05
            container.set_position_orientation(
                position=(cx, cy, ctx.table_top_z + half_h + 0.002),
            )
            if hasattr(container, "keep_still"):
                container.keep_still()
            og.sim.step()

            # Fill with liquid.
            system_name = ctx.selection.get("system_name", "water")
            from omnigibson.object_states import Filled
            try:
                system = ctx.env.scene.get_system(system_name)
                if Filled in container.states:
                    container.states[Filled].set_value(system, True)
                    og.sim.step()
                    print(f"[Pipeline] Container filled with {system_name}")
                else:
                    print("[Pipeline] WARNING: Container does not support Filled state")
            except Exception as e:
                print(f"[Pipeline] WARNING: Could not fill container: {e}")

            for _ in range(10):
                og.sim.step()

        settle_fn = make_settle_fn(og, th)
        settle_fn(ctx.active_objects)

    def make_edge_objects(self, ctx):
        from sentinel.utils.franka_edge_align import EdgeAlignObject

        result = []
        for inst, obj in ctx.active_objects.items():
            try:
                pos = obj.get_position_orientation()[0]
                result.append(EdgeAlignObject(
                    name=inst, role="target",
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            except Exception as exc:
                log.warning("empty_invert make_edge_objects: pose read for %s failed: %s", getattr(obj, "name", obj), exc)
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
            "container_synset": ctx.selection.get("container_synset") if ctx.selection else None,
            "system_name": ctx.selection.get("system_name", "water") if ctx.selection else "water",
            "pipeline": "empty_invert",
        }


def main():
    EmptyInvertPipeline().run()


if __name__ == "__main__":
    main()
