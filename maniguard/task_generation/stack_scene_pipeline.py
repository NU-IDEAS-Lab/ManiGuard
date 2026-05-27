"""Stack retrieval scene generation pipeline.

Three variants based on the target (bottom) object type:
  - **same**:       target is the same type as the stack items
  - **flat**:       target is a flat object (tray, chopping board, …)
  - **receptacle**: target is a concave container (bowl, stockpot, …)

Usage:
    python -m maniguard.task_generation.stack_scene_pipeline \
        --stack-mode same --scene-model Benevolence_1_int --dry-run

    python -m maniguard.task_generation.stack_scene_pipeline \
        --stack-mode flat --scene-model Benevolence_1_int --episodes 1 \
        --steps 300 --save-video

    python -m maniguard.task_generation.stack_scene_pipeline \
        --stack-mode receptacle --scene-model Benevolence_1_int --episodes 1 \
        --steps 300 --save-video
"""

import sys

from maniguard.task_generation.pipeline_common import (
    BasePipeline,
    get_spawned_obj,
    make_settle_fn,
)
from maniguard.task_generation.utils.stack_pipeline.select import select_stack_objects
from maniguard.utils.task_spec import (
    STACK_HEIGHT_PRESETS,
    generate_stack_activity,
)
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_stack_descriptors(spawned_objects, target_ids, stack_ids):
    """Build StackObjectDescriptors from spawned objects, ordered bottom-to-top.

    Uses ``native_bbox * scale`` (asset's authored upright extents) instead of
    ``obj.aabb`` (current world AABB). The world AABB shrinks if the object
    is currently tilted from spawning at z=10 with random orientation, which
    would underestimate height and cause stack penetration once objects are
    re-oriented upright by the layout transform.
    """
    from maniguard.utils.clutter_pack_layout import StackObjectDescriptor

    descriptors = []
    for inst, role in ([(tid, "target") for tid in target_ids] +
                       [(sid, "stack") for sid in stack_ids]):
        obj = get_spawned_obj(spawned_objects, inst)
        if obj is None:
            continue
        nbb = obj.native_bbox
        scale = obj.scale
        dx = max(0.01, float(nbb[0] * scale[0]))
        dy = max(0.01, float(nbb[1] * scale[1]))
        dz = max(0.01, float(nbb[2] * scale[2]))
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

    def scene_family(self, ctx):
        if self._stack_mode == "same":
            return "stack_same"
        if self._stack_mode == "flat":
            return "stack_flat"
        return None

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--stack-mode", default="same",
                            choices=["same", "flat", "receptacle"],
                            help="Stack variant: same, flat, or receptacle")
        parser.add_argument("--stack-height", default="medium",
                            choices=list(STACK_HEIGHT_PRESETS),
                            help="Number of objects stacked on top of the target")
        parser.add_argument("--target-model", default=None,
                            help="Override target (bottom) model id (e.g. qkjrwt). "
                                 "Synset/category is inferred by scanning the "
                                 "stack-mode's target pool.")
        parser.add_argument("--stack-model", default=None,
                            help="Override stack-item model id. Inferred from "
                                 "the stack-item pool (or target pool in "
                                 "--stack-mode same).")

    def select_objects(self, args, rng):
        return select_stack_objects(
            self._stack_mode, rng,
            target_model=args.target_model,
            stack_model=args.stack_model,
        )

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        pre = getattr(args, "_pre_selection", None) or {}
        return generate_stack_activity(
            activity_name, support_synset, support_room, args.stack_height,
            target_synset=pre.get("target_synset"),
            target_category=pre.get("target_category"),
            target_model=pre.get("target_model") or args.target_model,
            stack_synset=pre.get("stack_synset"),
            stack_category=pre.get("stack_category"),
            stack_model=pre.get("stack_model") or args.stack_model,
            mode=self._stack_mode,
        )

    def identify_objects(self, ctx):
        target_ids = list(ctx.obj_sets.get("target", ()))
        stack_ids = list(ctx.obj_sets.get("stack", ()))

        if not target_ids:
            raise RuntimeError("No target objects found in scope.")
        print(f"[Pipeline] Objects: target={target_ids}, stack={stack_ids}")

        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, target_ids[0])
        ctx._target_ids = target_ids
        ctx._stack_ids = stack_ids
        ctx.active_objects = {}
        for inst in target_ids + stack_ids:
            obj = get_spawned_obj(ctx.spawned_objects, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        import torch as th
        from maniguard.utils.clutter_pack_layout import (
            build_stack_layout, apply_stack_transform,
        )

        stack_descriptors = _build_stack_descriptors(
            ctx.spawned_objects, ctx._target_ids, ctx._stack_ids,
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
        from maniguard.utils.franka_edge_align import EdgeAlignObject

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
            "pipeline": self.scene_family(ctx),
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
