"""Food-transfer scene generation pipeline.

Moves a food item from a source container to a destination container
without the agent touching the food or letting it fall to the floor.

Usage:
    python -m sentinel.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m sentinel.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    get_scope_obj,
    iter_scope_objects,
)
from sentinel.utils.bddl_generator import generate_transfer_activity


def _place_food_on_source(env, food_obj, source_obj):
    """Teleport the food object on top of the source container."""
    import omnigibson as og

    src_pos = source_obj.get_position_orientation()[0]
    try:
        _, src_aabb_max = source_obj.aabb
        src_top_z = float(src_aabb_max[2])
    except Exception:
        src_top_z = float(src_pos[2]) + 0.03

    try:
        food_aabb_min, food_aabb_max = food_obj.aabb
        food_half_h = 0.5 * max(0.01, float(food_aabb_max[2] - food_aabb_min[2]))
    except Exception:
        food_half_h = 0.02

    food_obj.set_position_orientation(
        position=(float(src_pos[0]), float(src_pos[1]),
                  src_top_z + food_half_h + 0.005),
    )
    if hasattr(food_obj, "keep_still"):
        food_obj.keep_still()
    og.sim.step()
    print(f"[Pipeline] Teleported food onto source at "
          f"z={src_top_z + food_half_h + 0.005:.3f}")


class TransferPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--food-synset", default=None,
                            help="Override food object synset (e.g. cookie.n.01)")
        parser.add_argument("--source-synset", default=None,
                            help="Override source container synset (e.g. plate.n.04)")
        parser.add_argument("--dest-synset", default=None,
                            help="Override destination container synset (e.g. bowl.n.01)")
        parser.add_argument("--goal-predicate", default=None,
                            choices=["inside", "ontop"],
                            help="Override goal predicate (inside or ontop)")

    def activity_prefix(self):
        return "auto_transfer_on"

    def select_objects(self, args, rng):
        from sentinel.utils.bddl_generator import (
            TRANSFER_FOOD_POOL, TRANSFER_SOURCE_POOL, TRANSFER_DEST_POOL,
            estimate_object_set_footprint,
        )
        food = args.food_synset or TRANSFER_FOOD_POOL[rng.integers(len(TRANSFER_FOOD_POOL))][0]
        source = args.source_synset or TRANSFER_SOURCE_POOL[rng.integers(len(TRANSFER_SOURCE_POOL))][0]
        dest_entry = TRANSFER_DEST_POOL[rng.integers(len(TRANSFER_DEST_POOL))]
        dest = args.dest_synset or dest_entry[0]

        synset_counts = [(food, 1), (source, 1), (dest, 1)]
        return {
            "required_area_m2": estimate_object_set_footprint(synset_counts),
            "food_synset": food,
            "source_synset": source,
            "dest_synset": dest,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = getattr(args, "_pre_selection", None)
        return generate_transfer_activity(
            activity_name, support_synset, support_room,
            food_synset=pre["food_synset"] if pre else args.food_synset,
            source_synset=pre["source_synset"] if pre else args.source_synset,
            dest_synset=pre["dest_synset"] if pre else args.dest_synset,
            goal_predicate=args.goal_predicate,
            rng=rng,
        )

    def identify_objects(self, ctx):
        selection = ctx.selection
        food_synset = selection["food_synset"]
        source_synset = selection["source_synset"]
        dest_synset = selection["dest_synset"]

        food_ids, source_ids, dest_ids = [], [], []
        for inst, obj in iter_scope_objects(ctx.env):
            if inst.startswith(("agent.", "floor.")):
                continue
            if inst.startswith(food_synset + "_"):
                food_ids.append(inst)
            elif inst.startswith(source_synset + "_"):
                source_ids.append(inst)
            elif inst.startswith(dest_synset + "_"):
                if dest_synset == source_synset and inst == f"{source_synset}_1":
                    continue
                dest_ids.append(inst)

        if not food_ids:
            raise RuntimeError("No food objects found in scope.")
        print(f"[Pipeline] Objects: food={food_ids}, source={source_ids}, "
              f"dest={dest_ids}")

        ctx.target_obj = get_scope_obj(ctx.env, food_ids[0])
        ctx._source_obj = get_scope_obj(ctx.env, source_ids[0]) if source_ids else None
        ctx._food_ids = food_ids
        ctx._source_ids = source_ids
        ctx._dest_ids = dest_ids
        ctx.active_objects = {}
        for inst in food_ids + source_ids + dest_ids:
            obj = get_scope_obj(ctx.env, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        import omnigibson as og

        # Place source and dest on the table surface, side by side.
        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])
        spread = 0.15  # half the gap between source and dest

        if ctx._source_obj is not None:
            try:
                _, src_aabb_max = ctx._source_obj.aabb
                src_half_h = 0.5 * max(0.01, float(src_aabb_max[2]) - float(ctx._source_obj.aabb[0][2]))
            except Exception:
                src_half_h = 0.02
            ctx._source_obj.set_position_orientation(
                position=(cx - spread, cy, ctx.table_top_z + src_half_h + 0.002),
            )
            if hasattr(ctx._source_obj, "keep_still"):
                ctx._source_obj.keep_still()

        dest_obj = get_scope_obj(ctx.env, ctx._dest_ids[0]) if ctx._dest_ids else None
        if dest_obj is not None:
            try:
                _, dest_aabb_max = dest_obj.aabb
                dest_half_h = 0.5 * max(0.01, float(dest_aabb_max[2]) - float(dest_obj.aabb[0][2]))
            except Exception:
                dest_half_h = 0.02
            dest_obj.set_position_orientation(
                position=(cx + spread, cy, ctx.table_top_z + dest_half_h + 0.002),
            )
            if hasattr(dest_obj, "keep_still"):
                dest_obj.keep_still()

        og.sim.step()

        # Place food on source.
        if ctx.target_obj is not None and ctx._source_obj is not None:
            _place_food_on_source(ctx.env, ctx.target_obj, ctx._source_obj)

    def goal_conditions(self, ctx):
        goal_pred = ctx.selection.get("goal_predicate", "inside")
        dest_obj = get_scope_obj(ctx.env, ctx._dest_ids[0]) if ctx._dest_ids else None
        if ctx.target_obj and dest_obj:
            return [{"predicate": goal_pred, "subject": ctx.target_obj.name, "reference": dest_obj.name}]
        return []

    def extra_gate_checks(self, ctx):
        from omnigibson.object_states.on_top import OnTop

        if ctx.target_obj is None or ctx._source_obj is None:
            return True
        try:
            on_source = ctx.target_obj.states[OnTop].get_value(ctx._source_obj)
        except Exception:
            on_source = False
        if not on_source:
            print("[Pipeline] Gate: food is NOT on source")
            return False
        print("[Pipeline] Gate: food is on source — OK")
        return True

    def make_edge_objects(self, ctx):
        from sentinel.utils.franka_edge_align import EdgeAlignObject

        result = []
        for inst, obj in ctx.active_objects.items():
            try:
                pos = obj.get_position_orientation()[0]
                role = ("food" if inst in ctx._food_ids else
                        "source" if inst in ctx._source_ids else "dest")
                result.append(EdgeAlignObject(
                    name=inst, role=role,
                    position_xy=(float(pos[0]), float(pos[1])),
                ))
            except Exception:
                continue
        return tuple(result)


def main():
    TransferPipeline().run()


if __name__ == "__main__":
    main()
