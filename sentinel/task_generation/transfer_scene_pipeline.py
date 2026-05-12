"""Food-transfer scene generation pipeline.

Moves a food item from a source container to a destination container
without the agent touching the food or letting it fall to the floor.

Usage:
    python -m sentinel.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m sentinel.task_generation.transfer_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""
from __future__ import annotations

import json
import logging
import os

from sentinel.task_generation.pipeline_common import (
    BasePipeline,
    get_spawned_obj,
)
from sentinel.utils.task_spec import generate_transfer_activity

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transfer object selection: data paths, constants, and compatibility logic
# ---------------------------------------------------------------------------

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "utils", "food_transfer_pipeline")
_COMPAT_PATH = os.path.join(_UTILS_DIR, "transfer_compatibility.json")

# Categories excluded by ROLE (matrix is keyed by individual model; these
# remove whole categories regardless of which models passed the geometry +
# readiness filters at matrix-build time).
TRANSFER_EXCLUDE_CATS = {"coaster", "pencil_holder", "plant_pot", "vase", "outline_vase"}
TRANSFER_DEST_EXCLUDE_CATS = TRANSFER_EXCLUDE_CATS | {
    "baking_powder_jar", "basil_jar", "clove_jar", "coconut_oil_jar",
    "coffee_bean_jar", "cornstarch_jar", "frosting_jar", "granulated_sugar_jar",
    "hinged_jar", "hingeless_jar", "honey_jar", "instant_coffee_jar",
    "jar_of_clove", "jar_of_cocoa", "jar_of_coffee", "jar_of_cumin",
    "jar_of_curry_powder", "jar_of_dill_seed", "jar_of_honey", "jar_of_jam",
    "jar_of_jelly", "jar_of_kidney_beans", "jar_of_mayonnaise",
    "jar_of_orange_sauce", "jar_of_pepper_seasoning", "jar_of_peppercorns",
    "jar_of_puree", "jar_of_sesame_seed", "jar_of_spaghetti_sauce",
    "jar_of_strawberry_jam", "jar_of_sugar", "jar_of_tumeric",
    "jelly_bean_jar", "jelly_jar", "jimmies_jar", "marinara_jar",
    "noodle_jar", "peanut_butter_jar", "sodium_carbonate_jar",
    "tomato_sauce_jar", "yeast_jar",
}

_transfer_cache = None


def load_transfer_data():
    """Load the precomputed transfer compatibility matrix.

    The matrix at sentinel/task_generation/utils/transfer_compatibility.json
    is built by tools/build_transfer_compatibility.py and already filters by:
      * geometry: max(food.bbox_dims) <= container.opening_minor
      * readiness: status=graspable, role-specific suitability
    so the runtime work here is just inverting the matrix into the
    food→containers indexes that build_transfer_objects iterates.

    Returns ``{"compat", "food_to_sources", "food_to_dests"}``.
    """
    global _transfer_cache
    if _transfer_cache is not None:
        return _transfer_cache

    compat = {}
    if os.path.isfile(_COMPAT_PATH):
        with open(_COMPAT_PATH) as f:
            compat = json.load(f)
    else:
        log.warning("transfer_compatibility.json not found at %s", _COMPAT_PATH)

    food_to_sources = {}
    food_to_dests = {}
    for k, v in compat.items():
        c, _ = k.split("/", 1)
        valid_source = c not in TRANSFER_EXCLUDE_CATS
        valid_dest = c not in TRANSFER_DEST_EXCLUDE_CATS
        if not valid_source and not valid_dest:
            continue
        for fm in v.get("food_models", []):
            fkey = (fm["category"], fm["model"])
            if valid_dest:
                food_to_dests.setdefault(fkey, set()).add(k)
            if valid_source:
                food_to_sources.setdefault(fkey, set()).add(k)

    _transfer_cache = {
        "compat": compat,
        "food_to_sources": food_to_sources,
        "food_to_dests": food_to_dests,
    }
    return _transfer_cache


def build_transfer_objects(rng, food_model=None, source_model=None,
                           dest_model=None, goal_predicate=None):
    """Pick a compatible (source, food, dest) triple using the compatibility matrix.

    Optional ``*_model`` overrides pin a specific model id (e.g. ``"qkjrwt"``).
    The category is recovered from the matrix automatically. With no overrides,
    a random compatible triple is picked.

    Returns ``(obj_cfgs, roles, selection)``.
    """
    from omnigibson.utils.asset_utils import get_all_object_category_models

    data = load_transfer_data()
    compat = data["compat"]
    food_to_sources = data["food_to_sources"]
    food_to_dests = data["food_to_dests"]

    source_cat, source_mid = None, None
    food_cat, food_mid = None, None
    dest_cat, dest_mid = None, None

    all_source_keys = set()
    for fkey_sources in food_to_sources.values():
        all_source_keys |= fkey_sources
    source_keys = list(all_source_keys)
    if source_model:
        source_keys = [k for k in source_keys if k.endswith(f"/{source_model}")]
    rng.shuffle(source_keys)

    for s_key in source_keys:
        s_entry = compat[s_key]
        s_cat, s_mid = s_key.split("/", 1)
        if not get_all_object_category_models(s_cat):
            continue

        eligible = []
        for fm in s_entry["food_models"]:
            if food_model and fm["model"] != food_model:
                continue
            fkey = (fm["category"], fm["model"])
            dests = food_to_dests.get(fkey, set()) - {s_key}
            if dest_model:
                dests = {k for k in dests if k.endswith(f"/{dest_model}")}
            if dests:
                eligible.append((fm, list(dests)))
        if not eligible:
            continue

        f_pick, dest_keys = eligible[rng.integers(len(eligible))]
        f_cat, f_mid = f_pick["category"], f_pick["model"]
        d_key = dest_keys[rng.integers(len(dest_keys))]
        d_cat, d_mid = d_key.split("/", 1)

        source_cat, source_mid = s_cat, s_mid
        food_cat, food_mid = f_cat, f_mid
        dest_cat, dest_mid = d_cat, d_mid
        break

    if source_cat is None or food_cat is None or dest_cat is None:
        raise RuntimeError(
            "Could not find a compatible (source, food, dest) triple from "
            "the compatibility matrix. Check transfer_compatibility.json "
            "or your --*-model overrides."
        )

    if goal_predicate is None:
        goal_predicate = "inside"

    cfgs, roles = [], {}
    for idx, (cat, model, role) in enumerate([
        (food_cat, food_mid, "food"),
        (source_cat, source_mid, "source"),
        (dest_cat, dest_mid, "dest"),
    ]):
        name = f"{role}_{cat}_{idx}"
        cfgs.append(_make_obj_cfg(name, cat, model, position=(100 + idx, 100, -100)))
        roles[name] = role

    selection = {
        "food_category": food_cat,
        "food_model": food_mid,
        "source_category": source_cat,
        "source_model": source_mid,
        "dest_category": dest_cat,
        "dest_model": dest_mid,
        "goal_predicate": goal_predicate,
    }
    return cfgs, roles, selection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj_cfg(name, category, model, position=(0, 0, 0)):
    return {
        "type": "DatasetObject",
        "name": name,
        "category": category,
        "model": model,
        "position": list(position),
        "scale": None,
    }


def _upright_half_height(obj):
    """Half the object's authored upright Z-extent — orientation-independent.

    Uses ``native_bbox * scale`` (the asset's local-frame bounding box scaled
    by the live object scale), which is the Z size the object would have if
    placed at identity orientation. Reading ``obj.aabb`` instead would give
    the world-frame AABB Z, which shrinks when the object is currently
    tilted — leading to placement *below* the surface once the object is
    re-oriented upright.
    """
    nbb = obj.native_bbox
    scale = obj.scale
    return 0.5 * max(0.01, float(nbb[2] * scale[2]))


# ---------------------------------------------------------------------------
# Scene-based pipeline class
# ---------------------------------------------------------------------------

def place_food_on_source(env, food_obj, source_obj, settle_steps=60):
    """Teleport the food just above the source's actual cavity opening,
    then settle into the cavity.

    Drop XY comes from the offline-derived opening centroid in
    ``container_openings.json`` (offset relative to AABB center, applied
    to the live AABB center at runtime). For wide-mouth containers this
    is ~AABB center; for cap-style spouts and asymmetric jars it
    correctly aims at the opening, not the closed body.

    ``settle_steps`` runs enough physics iterations for the food to
    actually descend into the source's interior cavity (1 step @ 60Hz
    = 1.4mm of fall), where Inside / OnTop predicates can succeed.
    """
    import omnigibson as og
    from sentinel.task_generation.utils.food_transfer_pipeline.lookup import (
        container_drop_xy,
    )

    src_top_z = float(source_obj.aabb[1][2])
    food_half_h = _upright_half_height(food_obj)
    z = src_top_z + food_half_h + 0.005

    cx, cy = container_drop_xy(source_obj)
    food_obj.set_position_orientation(
        position=(cx, cy, z),
        orientation=(0, 0, 0, 1),
    )
    food_obj.keep_still()
    for _ in range(settle_steps):
        og.sim.step()
    final_z = float(food_obj.get_position_orientation()[0][2])
    print(f"[Pipeline] Food drop xy=({cx:.3f}, {cy:.3f}) from z={z:.3f}, "
          f"settled to z={final_z:.3f} after {settle_steps} steps")


class TransferPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--food-model", default=None,
                            help="Override food model id (e.g. qkjrwt). "
                                 "Category is inferred from the compat matrix.")
        parser.add_argument("--source-model", default=None,
                            help="Override source container model id.")
        parser.add_argument("--dest-model", default=None,
                            help="Override destination container model id.")
        parser.add_argument("--goal-predicate", default=None,
                            choices=["inside", "ontop"],
                            help="Override goal predicate (inside or ontop)")

    def activity_prefix(self):
        return "auto_transfer_on"

    def scene_family(self, ctx):
        return "transfer"

    def select_objects(self, args, rng):
        _, _, selection = build_transfer_objects(
            rng,
            food_model=args.food_model,
            source_model=args.source_model,
            dest_model=args.dest_model,
            goal_predicate=args.goal_predicate,
        )
        from sentinel.utils.task_spec import estimate_object_set_footprint
        # Models are already pinned by build_transfer_objects — forward them
        # so the picker uses exact per-model footprints, not category medians.
        counts = [
            (selection["food_category"], 1, selection["food_model"]),
            (selection["source_category"], 1, selection["source_model"]),
            (selection["dest_category"], 1, selection["dest_model"]),
        ]
        selection["required_area_m2"] = estimate_object_set_footprint(counts)
        return selection

    def generate_activity(self, activity_name, support_category, support_room,
                          args, rng):
        pre = getattr(args, "_pre_selection", {}) or {}
        return generate_transfer_activity(
            activity_name, support_category, support_room,
            food_category=pre.get("food_category"),
            food_model=pre.get("food_model"),
            source_category=pre.get("source_category"),
            source_model=pre.get("source_model"),
            dest_category=pre.get("dest_category"),
            dest_model=pre.get("dest_model"),
            goal_predicate=args.goal_predicate or pre.get("goal_predicate"),
            rng=rng,
        )

    def identify_objects(self, ctx):
        food_ids = list(ctx.obj_sets.get("food", ()))
        source_ids = list(ctx.obj_sets.get("source", ()))
        dest_ids = list(ctx.obj_sets.get("dest", ()))

        if not food_ids:
            raise RuntimeError("No food objects found in scope.")
        print(f"[Pipeline] Objects: food={food_ids}, source={source_ids}, "
              f"dest={dest_ids}")

        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, food_ids[0])
        ctx._source_obj = get_spawned_obj(ctx.spawned_objects, source_ids[0]) if source_ids else None
        ctx._food_ids = food_ids
        ctx._source_ids = source_ids
        ctx._dest_ids = dest_ids
        ctx.active_objects = {}
        for inst in food_ids + source_ids + dest_ids:
            obj = get_spawned_obj(ctx.spawned_objects, inst)
            if obj is not None:
                ctx.active_objects[inst] = obj

    def place_objects(self, ctx):
        import omnigibson as og

        cx = 0.5 * (ctx.surface_bounds_xy[0][0] + ctx.surface_bounds_xy[1][0])
        cy = 0.5 * (ctx.surface_bounds_xy[0][1] + ctx.surface_bounds_xy[1][1])

        # Compute AABB-aware spread. Each object lives at
        #   center = cx ± (own_half_x + 0.5 * gap_m)
        # so the resulting world AABBs have ``gap_m`` clearance between
        # them — preventing source/dest interpenetration when they have
        # different sizes (e.g. small wineglass + large bowl).
        gap_m = 0.05
        src = ctx._source_obj
        dest_obj = get_spawned_obj(ctx.spawned_objects, ctx._dest_ids[0]) if ctx._dest_ids else None

        # Single set_position_orientation call: position + identity orientation
        # in one shot. Height comes from native_bbox*scale (asset's authored
        # upright Z), not the current world AABB which would shrink if the
        # object is currently tilted from physics drift.
        if src is not None:
            half_x = 0.5 * float(src.native_bbox[0] * src.scale[0])
            half_z = _upright_half_height(src)
            src.set_position_orientation(
                position=(cx - half_x - 0.5 * gap_m, cy,
                          ctx.table_top_z + half_z + 0.002),
                orientation=(0, 0, 0, 1),
            )
            src.keep_still()

        if dest_obj is not None:
            half_x = 0.5 * float(dest_obj.native_bbox[0] * dest_obj.scale[0])
            half_z = _upright_half_height(dest_obj)
            dest_obj.set_position_orientation(
                position=(cx + half_x + 0.5 * gap_m, cy,
                          ctx.table_top_z + half_z + 0.002),
                orientation=(0, 0, 0, 1),
            )
            dest_obj.keep_still()

        og.sim.step()

        if ctx.target_obj is not None and src is not None:
            place_food_on_source(ctx.env, ctx.target_obj, src)

    def goal_conditions(self, ctx):
        goal_pred = ctx.selection.get("goal_predicate", "inside")
        dest_obj = get_spawned_obj(ctx.spawned_objects, ctx._dest_ids[0]) if ctx._dest_ids else None
        if ctx.target_obj and dest_obj:
            return [{"predicate": goal_pred, "subject": ctx.target_obj.name, "reference": dest_obj.name}]
        return []

    def extra_gate_checks(self, ctx):
        from omnigibson.object_states.inside import Inside
        from omnigibson.object_states.on_top import OnTop

        on_source = (
            ctx.target_obj.states[OnTop].get_value(ctx._source_obj)
            or ctx.target_obj.states[Inside].get_value(ctx._source_obj)
        )
        if not on_source:
            print("[Pipeline] Gate: food is NOT on/in source")
            return False
        print("[Pipeline] Gate: food is on/in source — OK")
        return True

    def make_edge_objects(self, ctx):
        from sentinel.utils.franka_edge_align import EdgeAlignObject

        result = []
        for inst, obj in ctx.active_objects.items():
            pos = obj.get_position_orientation()[0]
            role = ("food" if inst in ctx._food_ids else
                    "source" if inst in ctx._source_ids else "dest")
            result.append(EdgeAlignObject(
                name=inst, role=role,
                position_xy=(float(pos[0]), float(pos[1])),
            ))
        return tuple(result)

    def diagnostics_extra(self, ctx):
        return {"pipeline": "transfer"}


def main():
    TransferPipeline().run()


if __name__ == "__main__":
    main()
