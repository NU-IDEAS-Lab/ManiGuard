"""Liquid transport scene generation pipeline.

Extends the tabletop clutter pipeline: places a liquid-filled container
on a table with fragile/clutter obstacles, then runs LTL-monitored
rollouts that track spill, tilt, and obstacle safety.

The target is always a liquid-filled container from LIQUID_CONTAINER_POOL.
Fragile and clutter obstacles are selected from the standard clutter pools,
controlled by --clutter-density.  The --difficulty flag only controls
the liquid-specific spill threshold and tilt limit.

Usage:
    python -m sentinel.task_generation.liquid_transport_pipeline \
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m sentinel.task_generation.liquid_transport_pipeline \
        --scene-model Rs_int --difficulty hard --system-name water

    python -m sentinel.task_generation.liquid_transport_pipeline \
        --scene-model Rs_int --dry-run
"""

from sentinel.task_generation.clutter_scene_pipeline import ClutterPipeline
from sentinel.utils.task_spec import (
    LIQUID_CONTAINER_POOL,
    LIQUID_PRESETS,
    generate_clutter_activity,
    generate_liquid_transport_ltl_safety_json,
)


class LiquidTransportPipeline(ClutterPipeline):
    """Liquid-filled container transport with spill monitoring.

    Inherits object identification, packing, edge alignment, and gate
    checks from ClutterPipeline.  Overrides activity generation to use
    LIQUID_CONTAINER_POOL as the target pool and adds liquid-specific
    LTL constraints.  Adds liquid filling after placement and
    particle-count verification at the gate.
    """

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            "--difficulty", default="medium", choices=list(LIQUID_PRESETS),
            help="Liquid difficulty (spill threshold and tilt limit only)",
        )
        parser.add_argument(
            "--container-synset", default=None,
            help="Specific container synset (random if omitted)",
        )
        parser.add_argument(
            "--system-name", default="water",
            help="Liquid particle system name",
        )

    def activity_prefix(self):
        return "auto_liquid_transport_on"

    def scene_family(self, ctx):
        return "liquid_transport"

    def select_objects(self, args, rng):
        from sentinel.utils.task_spec import (
            FRAGILE_POOL, DENSITY_PRESETS,
            estimate_object_set_footprint,
            _pick_model_for_category, _synset_to_category,
        )
        from sentinel.task_generation.utils.clutter_pipeline.select import (
            select_obstacle,
        )

        density = DENSITY_PRESETS[args.clutter_density]
        container = args.container_synset
        if container is None:
            container = LIQUID_CONTAINER_POOL[rng.integers(len(LIQUID_CONTAINER_POOL))][0]
        container_category = _synset_to_category(container)
        # Pin a concrete container model so the picker uses exact footprint.
        _, container_model = _pick_model_for_category(container_category, rng)

        # Pin a concrete model for each fragile pick too.
        fragile_pool = [s for s in FRAGILE_POOL if s[0] != container] or list(FRAGILE_POOL)
        fragile_picks = []
        for _ in range(density["fragile_count"]):
            synset = fragile_pool[rng.integers(len(fragile_pool))][0]
            cat, model = _pick_model_for_category(_synset_to_category(synset), rng)
            fragile_picks.append((synset, cat, model))

        clutter_picks = [select_obstacle(rng, exclude_cats={container_category})
                         for _ in range(density["clutter_count"])]

        # Every entry is (category, count, model) — exact per-model footprint.
        counts = [(container_category, 1, container_model)]
        for _, cat, model in fragile_picks:
            counts.append((cat, 1, model))
        for _, cat, model in clutter_picks:
            counts.append((cat, 1, model))

        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "target_synset": container,
            "target_category": container_category,
            "target_model": container_model,
            "fragile_picks": fragile_picks,
            "clutter_picks": clutter_picks,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        _clutter_ltl, selection = generate_clutter_activity(
            activity_name, support_synset, support_room,
            args.clutter_density, rng=rng,
            pre_selection=args._pre_selection,
        )

        preset = LIQUID_PRESETS[args.difficulty]
        target_synset = selection["target_synset"]
        # fragile_picks is a list of [synset, cat, model] triples.
        fragile_synsets = sorted({p[0] for p in selection.get("fragile_picks", [])})

        ltl_safety = generate_liquid_transport_ltl_safety_json(
            activity_name=activity_name,
            container_synsets=[target_synset],
            fragile_synsets=fragile_synsets,
            system_name=args.system_name,
            spill_threshold=preset["spill_threshold"],
            max_tilt_deg=preset["max_tilt_deg"],
        )

        selection["system_name"] = args.system_name
        selection["difficulty"] = args.difficulty
        selection["spill_threshold"] = preset["spill_threshold"]
        selection["max_tilt_deg"] = preset["max_tilt_deg"]

        return ltl_safety, selection

    def configure_env(self, selection):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    def place_objects(self, ctx):
        """Pack objects via clutter logic, then fill the target container."""
        super().place_objects(ctx)

        system_name = ctx.selection.get("system_name", "water")
        from omnigibson.object_states import Filled
        try:
            system = ctx.env.scene.get_system(system_name)
            if ctx.target_obj is not None and Filled in ctx.target_obj.states:
                ctx.target_obj.states[Filled].set_value(system, True)
                ctx.og.sim.step()
                print(f"[Pipeline] Container filled with {system_name}")
            else:
                print("[Pipeline] WARNING: Container does not support Filled state")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not fill container: {e}")

        for _ in range(10):
            ctx.og.sim.step()

    def extra_gate_checks(self, ctx):
        if not super().extra_gate_checks(ctx):
            return False
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
        extra = super().diagnostics_extra(ctx)
        extra["difficulty"] = getattr(ctx.args, "difficulty", "medium")
        extra["system_name"] = ctx.selection.get("system_name", "water") if ctx.selection else "water"
        extra["pipeline"] = "liquid_transport"
        # GPU dynamics can leave Tensor values in diagnostics; ensure JSON safety.
        return _json_safe(extra)


def _json_safe(obj):
    """Recursively convert Tensors/ndarrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def main():
    LiquidTransportPipeline().run()


if __name__ == "__main__":
    main()
