"""Stack retrieval scene generation pipeline.

Retrieves a target object from under a vertical stack on a table.

Usage:
    python -m omnigibson.task_generation.stack_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.stack_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 \
        --stack-height medium --save-video
"""

import math

from omnigibson.task_generation.pipeline_common import (
    BasePipeline,
    get_scope_obj,
    iter_scope_objects,
    make_settle_fn,
)
from omnigibson.utils.bddl_generator import (
    STACK_HEIGHT_PRESETS,
    generate_stack_activity,
)


def _build_stack_descriptors(env, target_ids, stack_ids):
    """Build StackObjectDescriptors from live env objects, ordered bottom-to-top."""
    from omnigibson.utils.clutter_pack_layout import StackObjectDescriptor

    descriptors = []
    for inst, role in ([(tid, "target") for tid in target_ids] +
                       [(sid, "stack") for sid in stack_ids]):
        obj = get_scope_obj(env, inst)
        if obj is None:
            continue
        try:
            aabb_min, aabb_max = obj.aabb
            dx = max(0.01, float(aabb_max[0] - aabb_min[0]))
            dy = max(0.01, float(aabb_max[1] - aabb_min[1]))
            dz = max(0.01, float(aabb_max[2] - aabb_min[2]))
        except Exception:
            continue
        descriptors.append(StackObjectDescriptor(
            instance_id=inst, role=role,
            half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
        ))
    return descriptors


def _validate_ontop_state(env, stack_descriptors, support_obj, objects_by_inst):
    """Check that each object in the stack is OnTop of the one below it."""
    from omnigibson.object_states.on_top import OnTop

    chain = []
    for desc in stack_descriptors:
        obj = objects_by_inst.get(desc.instance_id)
        if obj is not None:
            chain.append((desc.instance_id, obj))

    if not chain:
        return False, "empty stack"

    bottom_inst, bottom_obj = chain[0]
    try:
        on_support = bottom_obj.states[OnTop].get_value(support_obj)
    except Exception:
        on_support = False
    if not on_support:
        return False, f"{bottom_inst} not OnTop support"

    for i in range(1, len(chain)):
        upper_inst, upper_obj = chain[i]
        lower_inst, lower_obj = chain[i - 1]
        try:
            on_lower = upper_obj.states[OnTop].get_value(lower_obj)
        except Exception:
            on_lower = False
        if not on_lower:
            return False, f"{upper_inst} not OnTop {lower_inst}"

    return True, "ok"


class StackPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--stack-height", default="medium",
                            choices=list(STACK_HEIGHT_PRESETS),
                            help="Number of objects stacked on top of the target")
        parser.add_argument("--target-synset", default=None,
                            help="Override target object synset")
        parser.add_argument("--stack-synset", default=None,
                            help="Override stack object synset")

    def activity_prefix(self):
        return "auto_stack_on"

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        return generate_stack_activity(
            activity_name, support_synset, support_room, args.stack_height,
            target_synset=args.target_synset, stack_synset=args.stack_synset,
            rng=rng,
        )

    def configure_task(self, cfg, selection):
        if selection.get("sampling_whitelist"):
            cfg["task"]["sampling_whitelist"] = selection["sampling_whitelist"]

    def identify_objects(self, ctx):
        selection = ctx.selection
        target_synset = selection["target_synset"]
        stack_synset = selection["stack_synset"]

        target_ids, stack_ids = [], []
        for inst, obj in iter_scope_objects(ctx.env):
            if inst.startswith(("agent.", "floor.")):
                continue
            if inst.startswith(target_synset + "_"):
                target_ids.append(inst)
            elif inst.startswith(stack_synset + "_"):
                if stack_synset == target_synset and inst == f"{target_synset}_1":
                    continue
                stack_ids.append(inst)

        stack_ids.sort(key=lambda s: int(s.rsplit("_", 1)[-1]))

        if not target_ids:
            raise RuntimeError("No target objects found in scope.")
        print(f"[Pipeline] Objects: target={target_ids}, stack={stack_ids}")

        ctx.target_obj = get_scope_obj(ctx.env, target_ids[0])
        ctx._target_ids = target_ids
        ctx._stack_ids = stack_ids
        ctx.active_objects = {}
        for inst in target_ids + stack_ids:
            obj = get_scope_obj(ctx.env, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        import torch as th
        from omnigibson.utils.clutter_pack_layout import (
            build_stack_layout, apply_stack_transform,
        )

        stack_descriptors = _build_stack_descriptors(
            ctx.env, ctx._target_ids, ctx._stack_ids,
        )
        if len(stack_descriptors) < 2:
            raise RuntimeError(f"Need at least 2 objects for a stack, "
                               f"got {len(stack_descriptors)}.")

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])
        stack_origin = (cx, cy, ctx.table_top_z)

        ep_seed = ctx.args.seed + ctx.episode * 1000
        stack_spec = build_stack_layout(
            support_obj_name=getattr(ctx.support_obj, "name", "support"),
            descriptors=stack_descriptors, seed=ep_seed,
        )
        ctx._placements = apply_stack_transform(
            stack_spec, ctx.active_objects, stack_origin,
        )
        print(f"[Pipeline] Stack placed: {len(ctx._placements)} objects at "
              f"origin=({cx:.3f}, {cy:.3f}, {ctx.table_top_z:.3f})")

        # Settle physics.
        for obj in ctx.active_objects.values():
            if hasattr(obj, "keep_still"):
                obj.keep_still()
        settle_fn = make_settle_fn(ctx.og, th)
        settle_fn(ctx.active_objects)

        # Validate OnTop chain with retries.
        ctx._ontop_ok = False
        for attempt in range(3):
            ctx._ontop_ok, ontop_msg = _validate_ontop_state(
                ctx.env, stack_descriptors, ctx.support_obj, ctx.active_objects,
            )
            if ctx._ontop_ok:
                print(f"[Pipeline] OnTop validation: OK (attempt {attempt + 1})")
                break
            print(f"[Pipeline] OnTop validation failed (attempt {attempt + 1}): "
                  f"{ontop_msg}")
            ep_seed += 1
            stack_spec = build_stack_layout(
                support_obj_name=getattr(ctx.support_obj, "name", "support"),
                descriptors=stack_descriptors, seed=ep_seed,
            )
            ctx._placements = apply_stack_transform(
                stack_spec, ctx.active_objects, stack_origin,
            )
            for obj in ctx.active_objects.values():
                if hasattr(obj, "keep_still"):
                    obj.keep_still()
            settle_fn(ctx.active_objects)

    def make_edge_objects(self, ctx):
        from omnigibson.utils.franka_edge_align import EdgeAlignObject

        return tuple(
            EdgeAlignObject(
                name=inst,
                role="target" if inst in ctx._target_ids else "stack",
                position_xy=(ctx._placements[inst][0], ctx._placements[inst][1]),
            )
            for inst in ctx._placements
        )

    def extra_gate_checks(self, ctx):
        return getattr(ctx, "_ontop_ok", False)

    def diagnostics_extra(self, ctx):
        return {
            "stack_height": getattr(ctx.args, "stack_height", None),
            "ontop_valid": getattr(ctx, "_ontop_ok", None),
        }


def main():
    StackPipeline().run()


if __name__ == "__main__":
    main()
