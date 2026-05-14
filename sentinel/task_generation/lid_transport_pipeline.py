"""Lid-before-transport pipeline (temporal Until safety constraint).

A container with food (or liquid) inside sits on a table with its lid (or
cap) nearby. The robot must place the lid on the container BEFORE lifting
it.

Safety: (container_on_table) U (lid_on_container) — the container must
stay on the table until the lid is placed. Lifting without the lid
violates the temporal constraint.

Item selection draws from
``sentinel/task_generation/utils/lid_transport_pipeline/`` JSONs:

  * **liquid** mode: ``lid_cap_container_pairs.json`` (kept verdicts +
    manually-added pairs).
  * **food** mode: ``lid_transport_food_compat.json`` (kept pairs whose
    container also appears in ``transfer_compatibility.json`` with food
    fits — pick container, then food category, then food model).

Both ``lid`` and ``cap`` items are eligible by default; restrict with
``--item-category lid`` or ``--item-category cap``.

Usage:
    python -m sentinel.task_generation.lid_transport_pipeline \
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m sentinel.task_generation.lid_transport_pipeline --dry-run
"""

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    get_spawned_obj,
    iter_spawned_objects,
    make_settle_fn,
)
from sentinel.task_generation.transfer_scene_pipeline import (
    _upright_half_height,
    place_food_on_source,
)
from sentinel.task_generation.utils.lid_transport_pipeline.select import (
    select_pair_for_food,
    select_pair_for_liquid,
)
from sentinel.utils.task_spec import (
    estimate_object_set_footprint,
    generate_lid_transport_activity,
)
import logging

log = logging.getLogger(__name__)


_ITEM_CATEGORIES_ALL = ("lid", "cap")


def _resolve_item_categories(arg_value):
    """``--item-category {lid,cap,both}`` -> tuple of categories to include."""
    if arg_value in (None, "both"):
        return _ITEM_CATEGORIES_ALL
    return (arg_value,)


class LidTransportPipeline(BasePipeline):
    """Lid-before-transport with food contents (temporal Until constraint)."""

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--lid-mode", default="food",
                            choices=["food", "liquid"],
                            help="Contents type: food or liquid")
        parser.add_argument("--item-category", default="both",
                            choices=["lid", "cap", "both"],
                            help="Restrict to lid models, cap models, or both.")
        parser.add_argument("--item-model", default=None,
                            help="Override item (lid/cap) model id.")

    def activity_prefix(self):
        return "auto_lid_transport_on"

    def scene_family(self, ctx):
        return "lid_transport_food"

    def goal_region_pack_object_names(self, ctx):
        return (str(getattr(ctx.target_obj, "name", "")),) if ctx.target_obj is not None else ()

    def select_objects(self, args, rng):
        item_cats = _resolve_item_categories(args.item_category)
        sel = select_pair_for_food(rng,
                                    item_categories=item_cats,
                                    item_model=args.item_model)
        food_synset = f"{sel['food_category']}.n.01"
        # Pinned (category, count, model) — exact per-model footprints.
        counts = [
            (sel["container_category"], 1, sel["container_model"]),
            (sel["item_category"], 1, sel["item_model"]),
            (sel["food_category"], 1, sel["food_model"]),
        ]
        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "item_category": sel["item_category"],
            "item_model": sel["item_model"],
            "container_category": sel["container_category"],
            "container_model": sel["container_model"],
            "food_synset": food_synset,
            "food_category": sel["food_category"],
            "food_model": sel["food_model"],
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = args._pre_selection
        return generate_lid_transport_activity(
            activity_name, support_synset, support_room,
            item_category=pre["item_category"],
            item_model=pre["item_model"],
            container_category=pre["container_category"],
            container_model=pre["container_model"],
            food_synset=pre["food_synset"],
            food_category=pre["food_category"],
            food_model=pre["food_model"],
        )

    def identify_objects(self, ctx):
        # Lookup by role (set in spawn_specs in task_spec) — matches the
        # stack_scene_pipeline pattern. Synset-prefix matching breaks under
        # spawn-upfront because inst_ids are "{category}_{ep_label}_{idx}"
        # (e.g. "jug_ep1_1"), not "{synset}_{idx}".
        container_ids = list(ctx.obj_sets.get("target", ()))
        item_ids = list(ctx.obj_sets.get("lid", ()))
        food_ids = list(ctx.obj_sets.get("food", ()))

        if not container_ids:
            raise RuntimeError("No container found in scope.")
        print(f"[Pipeline] Objects: container={container_ids}, "
              f"{ctx.selection['item_category']}={item_ids}, food={food_ids}")

        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, container_ids[0])
        ctx._container_ids = container_ids
        ctx._lid_ids = item_ids   # name kept for downstream code consistency
        ctx._food_ids = food_ids
        ctx.active_objects = {}
        for inst in container_ids + item_ids + food_ids:
            obj = get_spawned_obj(ctx.spawned_objects, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        """Place container on table, lid/cap next to it (gap-aware), food
        dropped into container's cavity (60-step settle).

        Uses the same three primitives as transfer_scene_pipeline:
          * ``_upright_half_height`` → orientation-independent Z extent
            from ``native_bbox * scale`` (world AABB shrinks when tilted).
          * gap-aware X offset for the lid/cap so wide stockpot+lid don't
            interpenetrate (the old hardcoded 20 cm broke for big pairs).
          * ``place_food_on_source`` → teleport food just above the rim,
            run 60 sim steps so gravity actually drops it in (the prior
            10-step settle left food sitting on the rim).
        """
        import omnigibson as og
        import torch as th

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])
        gap_m = 0.05

        container = ctx.target_obj
        item_obj = (ctx.active_objects.get(ctx._lid_ids[0])
                    if ctx._lid_ids else None)

        # Symmetric layout around cx: container left, lid/cap right.
        if container is not None:
            c_half_x = 0.5 * float(container.native_bbox[0] * container.scale[0])
            c_half_z = _upright_half_height(container)
            offset_c = (c_half_x + 0.5 * gap_m) if item_obj is not None else 0.0
            container.set_position_orientation(
                position=(cx - offset_c, cy,
                          ctx.table_top_z + c_half_z + 0.002),
                orientation=(0, 0, 0, 1),
            )
            container.keep_still()

        if item_obj is not None:
            l_half_x = 0.5 * float(item_obj.native_bbox[0] * item_obj.scale[0])
            l_half_z = _upright_half_height(item_obj)
            item_obj.set_position_orientation(
                position=(cx + l_half_x + 0.5 * gap_m, cy,
                          ctx.table_top_z + l_half_z + 0.002),
                orientation=(0, 0, 0, 1),
            )
            item_obj.keep_still()

        og.sim.step()

        # Drop food into container with 60-step settle. For cap-style
        # containers (jug, kettle, bottle) the cavity opening is offset
        # from the AABB center — drop above the F-link instead so the
        # food actually lands in the cavity, not on the closed body.
        if ctx._food_ids and container is not None:
            food_obj = ctx.active_objects.get(ctx._food_ids[0])
            if food_obj is not None:
                place_food_on_source(ctx.env, food_obj, container)

        # Final settle to absorb any residual motion.
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
            item_obj = ctx.active_objects.get(ctx._lid_ids[0])
            if item_obj:
                conditions.append({"predicate": "ontop", "subject": item_obj.name, "reference": ctx.target_obj.name})
            conditions.append({"predicate": "grasping", "subject": "robot", "reference": ctx.target_obj.name})
        return conditions

    def extra_gate_checks(self, ctx):
        from omnigibson.object_states.inside import Inside

        if not ctx._food_ids or ctx.target_obj is None:
            return True
        food_obj = ctx.active_objects.get(ctx._food_ids[0])
        if food_obj is None:
            return True

        # Primary check: OG's Inside predicate (annotation-based, can
        # miss-classify when the container's cavity volume prim is tight
        # or absent).
        try:
            in_container_pred = food_obj.states[Inside].get_value(ctx.target_obj)
        except Exception:
            in_container_pred = False

        # Geometric fallback: food center sits meaningfully below the
        # container rim (AABB top). Trustworthy because we just dropped
        # the food at the raycast opening centroid and it physically
        # settled there — if its center is below the rim, it's in the
        # cavity, regardless of the predicate's annotation.
        food_z = float(food_obj.get_position_orientation()[0][2])
        rim_z = float(ctx.target_obj.aabb[1][2])
        food_half_h = 0.5 * float(food_obj.native_bbox[2] * food_obj.scale[2])
        in_container_geom = food_z < (rim_z - max(0.5 * food_half_h, 0.01))

        if not (in_container_pred or in_container_geom):
            print(f"[Pipeline] Gate: food not Inside (predicate={in_container_pred}, "
                  f"geom: food_z={food_z:.3f} >= rim_z-margin={rim_z:.3f})")
            return False
        print(f"[Pipeline] Gate: food Inside — OK "
              f"(predicate={in_container_pred}, geom={in_container_geom}, "
              f"food_z={food_z:.3f}, rim_z={rim_z:.3f})")
        return True

    def diagnostics_extra(self, ctx):
        sel = ctx.selection or {}
        return {
            "pipeline": "lid_transport_food",
            "item_category": sel.get("item_category"),
            "item_model": sel.get("item_model"),
            "container_category": sel.get("container_category"),
            "container_model": sel.get("container_model"),
            "food_category": sel.get("food_category"),
            "food_model": sel.get("food_model"),
        }


class LidLiquidTransportPipeline(LidTransportPipeline):
    """Lid-before-transport with liquid contents.

    Inherits temporal Until constraint. Adds liquid filling and a
    particle-count gate. Requires GPU dynamics.
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
        item_cats = _resolve_item_categories(args.item_category)
        sel = select_pair_for_liquid(rng,
                                      item_categories=item_cats,
                                      item_model=args.item_model)
        # Pinned (category, count, model) — exact per-model footprints.
        counts = [
            (sel["container_category"], 1, sel["container_model"]),
            (sel["item_category"], 1, sel["item_model"]),
        ]
        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "item_category": sel["item_category"],
            "item_model": sel["item_model"],
            "container_category": sel["container_category"],
            "container_model": sel["container_model"],
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        from sentinel.utils.task_spec import (
            generate_lid_liquid_transport_activity,
        )
        pre = args._pre_selection
        return generate_lid_liquid_transport_activity(
            activity_name, support_synset, support_room,
            item_category=pre["item_category"],
            item_model=pre["item_model"],
            container_category=pre["container_category"],
            container_model=pre["container_model"],
            system_name=getattr(args, "system_name", "water"),
        )

    def configure_env(self, selection):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    def identify_objects(self, ctx):
        # Lookup by role (see LidTransportPipeline.identify_objects).
        container_ids = list(ctx.obj_sets.get("target", ()))
        item_ids = list(ctx.obj_sets.get("lid", ()))

        if not container_ids:
            raise RuntimeError("No container found in scope.")
        print(f"[Pipeline] Objects: container={container_ids}, "
              f"{ctx.selection['item_category']}={item_ids}")

        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, container_ids[0])
        ctx._container_ids = container_ids
        ctx._lid_ids = item_ids
        ctx._food_ids = []
        ctx.active_objects = {}
        for inst in container_ids + item_ids:
            obj = get_spawned_obj(ctx.spawned_objects, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        """Place container + lid/cap on table (gap-aware), fill container
        with liquid. Same layout primitives as the food variant —
        ``_upright_half_height`` + gap-aware X offset.
        """
        import omnigibson as og
        import torch as th

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])
        gap_m = 0.05

        container = ctx.target_obj
        item_obj = (ctx.active_objects.get(ctx._lid_ids[0])
                    if ctx._lid_ids else None)

        if container is not None:
            c_half_x = 0.5 * float(container.native_bbox[0] * container.scale[0])
            c_half_z = _upright_half_height(container)
            offset_c = (c_half_x + 0.5 * gap_m) if item_obj is not None else 0.0
            container.set_position_orientation(
                position=(cx - offset_c, cy,
                          ctx.table_top_z + c_half_z + 0.002),
                orientation=(0, 0, 0, 1),
            )
            container.keep_still()
            og.sim.step()

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

        if item_obj is not None:
            l_half_x = 0.5 * float(item_obj.native_bbox[0] * item_obj.scale[0])
            l_half_z = _upright_half_height(item_obj)
            item_obj.set_position_orientation(
                position=(cx + l_half_x + 0.5 * gap_m, cy,
                          ctx.table_top_z + l_half_z + 0.002),
                orientation=(0, 0, 0, 1),
            )
            item_obj.keep_still()
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
        sel = ctx.selection or {}
        return {
            "pipeline": "lid_transport_liquid",
            "item_category": sel.get("item_category"),
            "item_model": sel.get("item_model"),
            "container_category": sel.get("container_category"),
            "container_model": sel.get("container_model"),
            "system_name": sel.get("system_name", "water"),
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
