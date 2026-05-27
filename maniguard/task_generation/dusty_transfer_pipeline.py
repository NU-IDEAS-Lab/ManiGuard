"""Dusty food-transfer pipeline with a cleaning sponge.

Extends :class:`TransferPipeline` with two additions:

1. **The destination container starts dusty** — at spawn the destination
   gets ``Covered = True`` via OmniGibson's ``dust`` visual-particle
   system, so the agent must clean it before placing the food.
2. **A sponge is spawned on the side** — the sponge carries the
   ``particleRemover`` ability, so dragging it over the dusty
   destination removes dust particles in adjacency.

Task: pick up the sponge, wipe the destination clean of dust, then
transfer the food from the source into the now-clean destination.
Success requires both:

* ``inside(food, destination)`` — parent's transfer predicate
* ``NOT covered(destination, dust)`` — destination cleaned first

LTL safety is unchanged from the parent transfer pipeline (food can't
be dropped, agent can't touch the food directly). The "must clean
first" requirement is a goal-condition check at episode end, not a
per-step LTL.

Usage::

    python -m maniguard.task_generation.dusty_transfer_pipeline \\
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m maniguard.task_generation.dusty_transfer_pipeline --dry-run
"""
from __future__ import annotations

import logging

from maniguard.task_generation.transfer_scene_pipeline import (
    TransferPipeline,
    _upright_half_height,
    get_spawned_obj,
)

log = logging.getLogger(__name__)

_DUST_SYSTEM_NAME = "dust"
_SPONGE_CATEGORY = "sponge"
_SPONGE_SYNSET = "sponge.n.01"
# Three sponge models ship with BEHAVIOR-1K; all carry the
# ``particleRemover`` ability via taxonomy.
_SPONGE_MODELS = ("aewrov", "klwueh", "qewotb")


class DustyTransferPipeline(TransferPipeline):
    """Food-transfer with a dusty destination + a sponge to clean it.

    Dust is a ``VisualParticleSystem`` — pure visual attachment decals
    on the container surface, no GPU dynamics, no collision coupling.
    The sponge's ``ParticleRemover`` state removes those decals on
    adjacency (the sponge has to physically contact the surface).
    """

    # -- CLI ---------------------------------------------------------------

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            "--dust-dest", dest="dust_dest",
            action="store_true", default=True,
            help="Dust the destination container at spawn (default).",
        )
        parser.add_argument(
            "--no-dust-dest", dest="dust_dest", action="store_false",
            help="Skip dusting the destination (turns the task back into "
                 "a plain transfer with a sponge spawned but unused).",
        )
        parser.add_argument(
            "--sponge-model", default=None,
            help=f"Pin the sponge model id (choices: "
                 f"{', '.join(_SPONGE_MODELS)}). Random pick if omitted.",
        )

    # -- Family identity ---------------------------------------------------

    def activity_prefix(self):
        return "auto_dusty_transfer_on"

    def scene_family(self, ctx):
        return "dusty_transfer"

    def diagnostics_extra(self, ctx):
        base = super().diagnostics_extra(ctx) or {}
        base["pipeline"] = "dusty_transfer"
        base["dust_system"] = _DUST_SYSTEM_NAME
        base["dusted_dest"] = bool(getattr(ctx.args, "dust_dest", True))
        sel = ctx.selection or {}
        base["sponge_category"] = sel.get("sponge_category")
        base["sponge_model"] = sel.get("sponge_model")
        return base

    # -- Object selection: append sponge to the spawn list -----------------

    def select_objects(self, args, rng):
        selection = super().select_objects(args, rng)
        # Pick sponge model.
        if args.sponge_model:
            if args.sponge_model not in _SPONGE_MODELS:
                raise RuntimeError(
                    f"Unknown sponge model {args.sponge_model!r}; "
                    f"choices: {_SPONGE_MODELS}"
                )
            sponge_model = args.sponge_model
        else:
            sponge_model = _SPONGE_MODELS[int(rng.integers(len(_SPONGE_MODELS)))]
        selection["sponge_synset"] = _SPONGE_SYNSET
        selection["sponge_category"] = _SPONGE_CATEGORY
        selection["sponge_model"] = sponge_model
        # Bump required area to also fit the sponge on the surface.
        from maniguard.utils.task_spec import estimate_object_set_footprint
        counts = [
            (selection["food_category"], 1, selection["food_model"]),
            (selection["source_category"], 1, selection["source_model"]),
            (selection["dest_category"], 1, selection["dest_model"]),
            (_SPONGE_CATEGORY, 1, sponge_model),
        ]
        selection["required_area_m2"] = estimate_object_set_footprint(counts)
        return selection

    def generate_activity(self, activity_name, support_category, support_room,
                          args, rng):
        ltl_safety, selection = super().generate_activity(
            activity_name, support_category, support_room, args, rng,
        )
        pre = getattr(args, "_pre_selection", {}) or {}
        sponge_model = pre.get("sponge_model") or selection.get("sponge_model")
        if sponge_model is None:
            raise RuntimeError(
                "dusty_transfer: sponge model missing from selection; "
                "select_objects must have run first."
            )
        # Don't override abilities — sponge.n.01's taxonomy entry
        # already carries particleRemover with the proper
        # ``conditions`` dict ({"dust.n.01": []} = always remove on
        # adjacency). Passing an empty {"particleRemover": {}} would
        # MERGE OVER that and lose the conditions, crashing during
        # postprocess_ability_params.
        selection.setdefault("sponge_synset", _SPONGE_SYNSET)
        selection.setdefault("sponge_category", _SPONGE_CATEGORY)
        selection.setdefault("sponge_model", sponge_model)
        selection["spawn_specs"].append({
            "synset": _SPONGE_SYNSET,
            "category": _SPONGE_CATEGORY,
            "count": 1,
            "role": "sponge",
            "model": sponge_model,
        })
        return ltl_safety, selection

    # -- Identify spawned objects -----------------------------------------

    def identify_objects(self, ctx):
        super().identify_objects(ctx)
        sponge_ids = list(ctx.obj_sets.get("sponge", ()))
        ctx._sponge_ids = sponge_ids
        if sponge_ids:
            sponge = get_spawned_obj(ctx.spawned_objects, sponge_ids[0])
            ctx._sponge_obj = sponge
            if sponge is not None:
                ctx.active_objects[sponge_ids[0]] = sponge
        else:
            ctx._sponge_obj = None

    # -- Placement: standard transfer + sponge + dust ----------------------

    def _dust_object(self, ctx, obj):
        """Set ``Covered=True`` for ``obj`` with the dust visual-particle
        system. Returns True iff at least one visual particle landed.
        """
        if obj is None:
            return False
        from omnigibson.object_states import Covered

        try:
            system = ctx.env.scene.get_system(
                _DUST_SYSTEM_NAME, force_init=True,
            )
        except Exception as exc:
            log.warning("dust system lookup failed: %s", exc)
            return False
        if Covered not in obj.states:
            log.warning("dust: %s is not dustyable (no Covered state); "
                        "skipping", getattr(obj, "name", obj))
            return False
        try:
            ok = bool(obj.states[Covered].set_value(system, True))
        except Exception as exc:
            log.warning("dust: Covered.set_value(%s) failed: %s",
                        getattr(obj, "name", obj), exc)
            return False
        print(f"[Pipeline] Dusted {obj.name} with {_DUST_SYSTEM_NAME} "
              f"(success={ok})")
        return ok

    def _place_sponge_next_to_layout(self, ctx):
        """Place the sponge on the surface, off to one side of the
        source / dest pair (centered along x). Anchored against the
        surface region's +y edge so it doesn't collide with the
        source/dest centerline placement.
        """
        if ctx._sponge_obj is None:
            return
        sponge = ctx._sponge_obj
        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        # +y edge of the placeable region, inset by the sponge half + gap.
        sponge_half_y = 0.5 * float(sponge.native_bbox[1] * sponge.scale[1])
        sponge_half_z = _upright_half_height(sponge)
        y_max = ctx.surface_bounds_xy[1][1]
        gap_m = 0.05
        sy = y_max - sponge_half_y - gap_m
        sponge.set_position_orientation(
            position=(cx, sy, ctx.table_top_z + sponge_half_z + 0.002),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        sponge.keep_still()
        print(f"[Pipeline] Sponge {sponge.name} placed at "
              f"({cx:.3f}, {sy:.3f}, {ctx.table_top_z + sponge_half_z:.3f})")

    def place_objects(self, ctx):
        # Standard transfer placement (food in source, source/dest on surface).
        super().place_objects(ctx)

        # Sponge on the side.
        self._place_sponge_next_to_layout(ctx)

        # Dust the destination container.
        if getattr(ctx.args, "dust_dest", True) and ctx._dest_ids:
            dest_obj = get_spawned_obj(ctx.spawned_objects, ctx._dest_ids[0])
            self._dust_object(ctx, dest_obj)

        # Step once so the visual particles + sponge state register with
        # the contact-particles cache before downstream gates evaluate.
        ctx.og.sim.step()

    def make_edge_objects(self, ctx):
        """Include the sponge in the pack centroid so the Franka stays
        within reach of the sponge as well as the source/dest.
        """
        from maniguard.utils.franka_edge_align import EdgeAlignObject

        parent = list(super().make_edge_objects(ctx))
        if ctx._sponge_obj is not None:
            pos = ctx._sponge_obj.get_position_orientation()[0]
            parent.append(EdgeAlignObject(
                name=ctx._sponge_ids[0],
                role="sponge",
                position_xy=(float(pos[0]), float(pos[1])),
            ))
        return tuple(parent)

    # -- Success criteria --------------------------------------------------

    def goal_conditions(self, ctx):
        """Parent transfer condition + ``NOT covered(dest, dust)``.

        The destination must be wiped clean of dust before / by the time
        the food is placed inside it. The agent uses the sponge's
        ``ParticleRemover`` ability (adjacency-based) to clear the dust.
        """
        conditions = list(super().goal_conditions(ctx))
        if ctx._dest_ids:
            dest_obj = get_spawned_obj(ctx.spawned_objects, ctx._dest_ids[0])
            if dest_obj is not None:
                conditions.append({
                    "op": "not",
                    "term": {
                        "predicate": "covered",
                        "subject": dest_obj.name,
                        "system": _DUST_SYSTEM_NAME,
                    },
                })
        return conditions


def main():
    DustyTransferPipeline().run()


if __name__ == "__main__":
    main()
