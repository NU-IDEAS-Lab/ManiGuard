"""Lid-before-transport pipeline (temporal Until safety constraint).

A container with food inside sits on a table with its lid nearby.
The robot must place the lid on the container BEFORE lifting it.

Safety: (container_on_table) U (lid_on_container) — the container must
stay on the table until the lid is placed.  Lifting without the lid
violates the temporal constraint.

Usage:
    python -m sentinel.task_generation.lid_transport_pipeline \
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m sentinel.task_generation.lid_transport_pipeline --dry-run
"""

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    get_scope_obj,
    iter_scope_objects,
    make_settle_fn,
)
from sentinel.task_generation.pipeline_common import resolve_synset
from sentinel.utils.bddl_generator import (
    LID_FOOD_POOL,
    estimate_object_set_footprint,
    generate_lid_transport_activity,
    get_lid_container_pairs,
)
import logging

log = logging.getLogger(__name__)


class LidTransportPipeline(BasePipeline):
    """Lid-before-transport with temporal Until constraint."""

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--lid-mode", default="food",
                            choices=["food", "liquid"],
                            help="Contents type: food or liquid")
        parser.add_argument("--food-synset", default=None,
                            help="Override food synset")

    def activity_prefix(self):
        return "auto_lid_transport_on"

    def scene_family(self, ctx):
        return "lid_transport_food"

    def goal_region_pack_object_names(self, ctx):
        return (str(getattr(ctx.target_obj, "name", "")),) if ctx.target_obj is not None else ()

    def select_objects(self, args, rng):
        pairs = get_lid_container_pairs()
        lid_ids = list(pairs.keys())
        lid_model = lid_ids[rng.integers(len(lid_ids))]
        pair = pairs[lid_model]
        container_synset = resolve_synset(pair["container_category"])
        food = args.food_synset or \
            LID_FOOD_POOL[rng.integers(len(LID_FOOD_POOL))][0]

        synset_counts = [(container_synset, 1), ("lid.n.02", 1), (food, 1)]
        return {
            "required_area_m2": estimate_object_set_footprint(synset_counts),
            "lid_model": lid_model,
            "food_synset": food,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = args._pre_selection
        return generate_lid_transport_activity(
            activity_name, support_synset, support_room,
            lid_model=pre["lid_model"],
            food_synset=pre["food_synset"],
            rng=rng,
        )

    def configure_task(self, cfg, selection):
        if selection.get("sampling_whitelist"):
            cfg["task"]["sampling_whitelist"] = selection["sampling_whitelist"]

    def identify_objects(self, ctx):
        selection = ctx.selection
        container_synset = selection["container_synset"]
        food_synset = selection["food_synset"]

        container_ids, lid_ids, food_ids = [], [], []
        for inst, obj in iter_scope_objects(ctx.env):
            if inst.startswith(("agent.", "floor.")):
                continue
            if inst.startswith(container_synset + "_"):
                container_ids.append(inst)
            elif inst.startswith("lid.n.02_"):
                lid_ids.append(inst)
            elif inst.startswith(food_synset + "_"):
                food_ids.append(inst)

        if not container_ids:
            raise RuntimeError("No container found in scope.")
        print(f"[Pipeline] Objects: container={container_ids}, "
              f"lid={lid_ids}, food={food_ids}")

        ctx.target_obj = get_scope_obj(ctx.env, container_ids[0])
        ctx._container_ids = container_ids
        ctx._lid_ids = lid_ids
        ctx._food_ids = food_ids
        ctx.active_objects = {}
        for inst in container_ids + lid_ids + food_ids:
            obj = get_scope_obj(ctx.env, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        """Place container on table, food inside, lid nearby."""
        import omnigibson as og
        import torch as th

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])

        # Place container at center of table.
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

        # Place food inside container.
        if ctx._food_ids:
            food_obj = ctx.active_objects.get(ctx._food_ids[0])
            if food_obj is not None and container is not None:
                try:
                    c_top = float(container.aabb[1][2])
                    food_half_h = 0.5 * max(0.01, float(food_obj.aabb[1][2]) - float(food_obj.aabb[0][2]))
                except Exception:
                    c_top = ctx.table_top_z + 0.1
                    food_half_h = 0.02
                # Place food slightly below container top (inside).
                food_obj.set_position_orientation(
                    position=(cx, cy, c_top - food_half_h * 0.5),
                )
                if hasattr(food_obj, "keep_still"):
                    food_obj.keep_still()
                og.sim.step()

        # Place lid next to container.
        if ctx._lid_ids:
            lid_obj = ctx.active_objects.get(ctx._lid_ids[0])
            if lid_obj is not None:
                try:
                    lid_half_h = 0.5 * max(0.01, float(lid_obj.aabb[1][2]) - float(lid_obj.aabb[0][2]))
                except Exception:
                    lid_half_h = 0.02
                # Offset to the side.
                lid_x = cx + 0.20
                lid_obj.set_position_orientation(
                    position=(lid_x, cy, ctx.table_top_z + lid_half_h + 0.002),
                )
                if hasattr(lid_obj, "keep_still"):
                    lid_obj.keep_still()
                og.sim.step()

        # Settle.
        settle_fn = make_settle_fn(og, th)
        settle_fn(ctx.active_objects)

    def make_edge_objects(self, ctx):
        from sentinel.utils.franka_edge_align import EdgeAlignObject

        result = []
        for inst, obj in ctx.active_objects.items():
            try:
                pos = obj.get_position_orientation()[0]
                role = ("target" if inst in ctx._container_ids else
                        "lid" if inst in ctx._lid_ids else "food")
                result.append(EdgeAlignObject(
                    name=inst, role=role,
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            except Exception as exc:
                log.warning("lid_transport make_edge_objects: pose read for %s failed: %s", getattr(obj, "name", obj), exc)
                continue
        return tuple(result)

    def goal_conditions(self, ctx):
        conditions = []
        if ctx._lid_ids and ctx.target_obj:
            lid_obj = ctx.active_objects.get(ctx._lid_ids[0])
            if lid_obj:
                conditions.append({"predicate": "ontop", "subject": lid_obj.name, "reference": ctx.target_obj.name})
            conditions.append({"predicate": "grasping", "subject": "robot", "reference": ctx.target_obj.name})
        return conditions

    def extra_gate_checks(self, ctx):
        # Verify food is inside or on top of container.
        from omnigibson.object_states.on_top import OnTop
        from omnigibson.object_states.inside import Inside

        if not ctx._food_ids or ctx.target_obj is None:
            return True
        food_obj = ctx.active_objects.get(ctx._food_ids[0])
        if food_obj is None:
            return True
        try:
            in_container = food_obj.states[Inside].get_value(ctx.target_obj)
        except Exception:
            in_container = False
        try:
            on_container = food_obj.states[OnTop].get_value(ctx.target_obj)
        except Exception:
            on_container = False
        if not (in_container or on_container):
            print("[Pipeline] Gate: food not on/in container")
            return False
        print(f"[Pipeline] Gate: food in container — OK "
              f"(inside={in_container}, ontop={on_container})")
        return True

    def diagnostics_extra(self, ctx):
        return {
            "pipeline": "lid_transport_food",
            "container_synset": ctx.selection.get("container_synset") if ctx.selection else None,
            "food_synset": ctx.selection.get("food_synset") if ctx.selection else None,
            "lid_model": ctx.selection.get("lid_model") if ctx.selection else None,
        }


class LidLiquidTransportPipeline(LidTransportPipeline):
    """Lid-before-transport with liquid contents (teapot/kettle).

    Inherits temporal Until constraint.  Adds liquid filling and
    particle-count gate check.  Requires GPU dynamics.
    """

    def scene_family(self, ctx):
        return "lid_transport_liquid"

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--system-name", default="water",
                            help="Liquid particle system name")

    def activity_prefix(self):
        return "auto_lid_liquid_on"

    def select_objects(self, args, rng):
        from sentinel.utils.bddl_generator import (
            LID_LIQUID_CATEGORIES,
        )
        pairs = get_lid_container_pairs()
        liquid_lids = [k for k, v in pairs.items()
                       if v["container_category"] in LID_LIQUID_CATEGORIES]
        lid_model = liquid_lids[rng.integers(len(liquid_lids))]
        pair = pairs[lid_model]
        container_synset = resolve_synset(pair["container_category"])

        synset_counts = [(container_synset, 1), ("lid.n.02", 1)]
        return {
            "required_area_m2": estimate_object_set_footprint(synset_counts),
            "lid_model": lid_model,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        from sentinel.utils.bddl_generator import (
            generate_lid_liquid_transport_activity,
        )
        pre = args._pre_selection
        return generate_lid_liquid_transport_activity(
            activity_name, support_synset, support_room,
            lid_model=pre["lid_model"],
            system_name=getattr(args, "system_name", "water"),
            rng=rng,
        )

    def configure_task(self, cfg, selection):
        super().configure_task(cfg, selection)
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    def identify_objects(self, ctx):
        selection = ctx.selection
        container_synset = selection["container_synset"]

        container_ids, lid_ids = [], []
        for inst, obj in iter_scope_objects(ctx.env):
            if inst.startswith(("agent.", "floor.")):
                continue
            if inst.startswith(container_synset + "_"):
                container_ids.append(inst)
            elif inst.startswith("lid.n.02_"):
                lid_ids.append(inst)

        if not container_ids:
            raise RuntimeError("No container found in scope.")
        print(f"[Pipeline] Objects: container={container_ids}, lid={lid_ids}")

        ctx.target_obj = get_scope_obj(ctx.env, container_ids[0])
        ctx._container_ids = container_ids
        ctx._lid_ids = lid_ids
        ctx._food_ids = []
        ctx.active_objects = {}
        for inst in container_ids + lid_ids:
            obj = get_scope_obj(ctx.env, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        """Place container + lid on table, fill container with liquid."""
        import omnigibson as og
        import torch as th

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])

        # Place container at center.
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

        # Place lid next to container.
        if ctx._lid_ids:
            lid_obj = ctx.active_objects.get(ctx._lid_ids[0])
            if lid_obj is not None:
                try:
                    lid_half_h = 0.5 * max(0.01, float(lid_obj.aabb[1][2]) - float(lid_obj.aabb[0][2]))
                except Exception:
                    lid_half_h = 0.02
                lid_obj.set_position_orientation(
                    position=(cx + 0.20, cy, ctx.table_top_z + lid_half_h + 0.002),
                )
                if hasattr(lid_obj, "keep_still"):
                    lid_obj.keep_still()
                og.sim.step()

        settle_fn = make_settle_fn(og, th)
        settle_fn(ctx.active_objects)

    def extra_gate_checks(self, ctx):
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
            "pipeline": "lid_transport_liquid",
            "container_synset": ctx.selection.get("container_synset") if ctx.selection else None,
            "system_name": ctx.selection.get("system_name", "water") if ctx.selection else "water",
            "lid_model": ctx.selection.get("lid_model") if ctx.selection else None,
        }


_MODE_MAP = {
    "food": LidTransportPipeline,
    "liquid": LidLiquidTransportPipeline,
}


def _parse_lid_mode():
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == "--lid-mode" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "food"


def main():
    mode = _parse_lid_mode()
    if mode not in _MODE_MAP:
        raise SystemExit(f"Unknown --lid-mode {mode!r}. Choose from: {list(_MODE_MAP)}")
    _MODE_MAP[mode]().run()


if __name__ == "__main__":
    main()
