"""Food-transfer scene generation pipeline.

Moves a food item from a source container to a destination container
without the agent touching the food or letting it fall to the floor.

Usage:
    python -m omnigibson.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

from omnigibson.task_generation.pipeline_common import (
    BasePipeline,
    get_scope_obj,
    iter_scope_objects,
)
from omnigibson.utils.bddl_generator import generate_transfer_activity


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

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        return generate_transfer_activity(
            activity_name, support_synset, support_room,
            food_synset=args.food_synset, source_synset=args.source_synset,
            dest_synset=args.dest_synset, goal_predicate=args.goal_predicate,
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
        if ctx.target_obj is not None and ctx._source_obj is not None:
            _place_food_on_source(ctx.env, ctx.target_obj, ctx._source_obj)

    def make_edge_objects(self, ctx):
        from omnigibson.utils.franka_edge_align import EdgeAlignObject

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
