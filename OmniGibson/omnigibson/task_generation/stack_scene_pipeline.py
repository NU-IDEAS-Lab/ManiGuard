"""Stack retrieval scene generation pipeline.

Three variants based on the target (bottom) object type:
  - **same**:       target is the same type as the stack items
  - **flat**:       target is a flat object (tray, chopping board, …)
  - **receptacle**: target is a concave container (bowl, stockpot, …)

Usage:
    python -m omnigibson.task_generation.stack_scene_pipeline \
        --stack-mode same --scene-model Benevolence_1_int --dry-run

    python -m omnigibson.task_generation.stack_scene_pipeline \
        --stack-mode flat --scene-model Benevolence_1_int --episodes 1 \
        --steps 300 --save-video

    python -m omnigibson.task_generation.stack_scene_pipeline \
        --stack-mode receptacle --scene-model Benevolence_1_int --episodes 1 \
        --steps 300 --save-video
"""

import sys

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _StackBase(BasePipeline):
    """Shared logic for all stack-retrieval pipeline variants."""

    # Subclasses must set this.
    _stack_mode = None

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--stack-mode", default="same",
                            choices=["same", "flat", "receptacle"],
                            help="Stack variant: same, flat, or receptacle")
        parser.add_argument("--stack-height", default="medium",
                            choices=list(STACK_HEIGHT_PRESETS),
                            help="Number of objects stacked on top of the target")
        parser.add_argument("--target-synset", default=None,
                            help="Override target (bottom) object synset")
        parser.add_argument("--stack-synset", default=None,
                            help="Override stack object synset")

    def select_objects(self, args, rng):
        from omnigibson.utils.bddl_generator import (
            STACK_SAME_POOL, STACK_FLAT_TARGET_POOL, STACK_RECEPTACLE_TARGET_POOL,
            STACK_ITEM_POOL
        )
        mode = self._stack_mode

        if mode == "same":
            pool = STACK_SAME_POOL
            target = args.target_synset or pool[rng.integers(len(pool))][0]
            stack = target
        elif mode == "flat":
            target = args.target_synset or STACK_FLAT_TARGET_POOL[rng.integers(len(STACK_FLAT_TARGET_POOL))][0]
            stack = args.stack_synset or STACK_ITEM_POOL[rng.integers(len(STACK_ITEM_POOL))][0]
        elif mode == "receptacle":
            target = args.target_synset or STACK_RECEPTACLE_TARGET_POOL[rng.integers(len(STACK_RECEPTACLE_TARGET_POOL))][0]
            stack = args.stack_synset or STACK_ITEM_POOL[rng.integers(len(STACK_ITEM_POOL))][0]
        else:
            return None

        # Stack is vertical — footprint is just the larger of target vs stack item.
        from omnigibson.utils.bddl_generator import _load_footprint_catalog, _median_footprint
        catalog = _load_footprint_catalog()
        required = max(_median_footprint(catalog, target), _median_footprint(catalog, stack))
        return {
            "required_area_m2": required,
            "target_synset": target,
            "stack_synset": stack,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = getattr(args, "_pre_selection", None)
        return generate_stack_activity(
            activity_name, support_synset, support_room, args.stack_height,
            target_synset=pre["target_synset"] if pre else args.target_synset,
            stack_synset=pre["stack_synset"] if pre else args.stack_synset,
            mode=self._stack_mode,
            rng=rng,
        )

    def configure_task(self, cfg, selection):
        if selection.get("sampling_whitelist"):
            cfg["task"]["sampling_whitelist"] = selection["sampling_whitelist"]

    def identify_objects(self, ctx):
        selection = ctx.selection
        target_synset = selection["target_synset"]
        stack_synset = selection["stack_synset"]
        same_synset = target_synset == stack_synset

        target_ids, stack_ids = [], []
        for inst, obj in iter_scope_objects(ctx.env):
            if inst.startswith(("agent.", "floor.")):
                continue
            if same_synset and inst.startswith(target_synset + "_"):
                # Same synset for both: _1 is target, rest are stack.
                if inst == f"{target_synset}_1":
                    target_ids.append(inst)
                else:
                    stack_ids.append(inst)
            elif inst.startswith(target_synset + "_"):
                target_ids.append(inst)
            elif inst.startswith(stack_synset + "_"):
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
            "stack_mode": self._stack_mode,
            "stack_height": getattr(ctx.args, "stack_height", None),
            "ontop_valid": getattr(ctx, "_ontop_ok", None),
        }


# ---------------------------------------------------------------------------
# Concrete pipeline classes
# ---------------------------------------------------------------------------

class StackSamePipeline(_StackBase):
    """Target is the same object type as the stack items."""
    _stack_mode = "same"

    def activity_prefix(self):
        return "auto_stack_same_on"


class StackFlatPipeline(_StackBase):
    """Target is a flat object (tray, chopping board, etc.) under the stack."""
    _stack_mode = "flat"

    def activity_prefix(self):
        return "auto_stack_flat_on"


class StackReceptaclePipeline(_StackBase):
    """Target is a concave receptacle (bowl, stockpot, etc.) under the stack."""
    _stack_mode = "receptacle"

    def activity_prefix(self):
        return "auto_stack_recep_on"


# Backward compatibility alias.
StackPipeline = StackSamePipeline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_MODE_MAP = {
    "same": StackSamePipeline,
    "flat": StackFlatPipeline,
    "receptacle": StackReceptaclePipeline,
}


def _parse_stack_mode():
    """Peek at sys.argv for --stack-mode before full argparse runs."""
    for i, arg in enumerate(sys.argv):
        if arg == "--stack-mode" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "same"


def main():
    mode = _parse_stack_mode()
    if mode not in _MODE_MAP:
        raise SystemExit(f"Unknown --stack-mode {mode!r}. Choose from: {list(_MODE_MAP)}")
    _MODE_MAP[mode]().run()


if __name__ == "__main__":
    main()
