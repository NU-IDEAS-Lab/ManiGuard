"""Wet transport pipeline (liquid container + overhead-forbidden zones).

Carries a water-filled container across a table while avoiding passing
over water-sensitive items (books, papers, keyboards). Same setup
mechanics as ``liquid_transport`` — fillable-container target +
maxrects pack + Filled fill + particle-count gate — but the
"fragile-role" slot is filled by the water-sensitive pool instead of
the liquid-fragile pool, and the LTL uses overhead-forbidden zone
predicates instead of fragile-upright / fragile-dropped.

Usage:
    python -m maniguard.task_generation.wet_transport_pipeline \\
        --scene-model Rs_int --episodes 1 --steps 300 --save-video

    python -m maniguard.task_generation.wet_transport_pipeline \\
        --scene-model Rs_int --dry-run
"""

from maniguard.task_generation.liquid_transport_pipeline import LiquidTransportPipeline
from maniguard.task_generation.utils.clutter_pipeline.select import (
    select_fillable_container,
)
from maniguard.task_generation.utils.wet_transport.select import select_water_sensitive
from maniguard.utils.task_spec import (
    DENSITY_PRESETS,
    _pick_model_for_category,
    estimate_object_set_footprint,
    generate_clutter_activity,
    generate_wet_transport_ltl_safety_json,
)


class WetTransportPipeline(LiquidTransportPipeline):
    """Liquid-filled container transport with overhead-forbidden zones.

    Inherits container fill + particle hygiene + clutter pack +
    edge-aligned robot mount from ``LiquidTransportPipeline``. The only
    differences are:

      * ``select_objects`` draws zones from
        ``water_sensitive_pool.json`` (papers / electronics — items
        that must not get wet) instead of the liquid-fragile pool, and
        skips clutter entirely. Zones occupy the "fragile" role in
        the spawn_specs so the clutter pack treats them as obstacles
        to surround the target with.
      * ``generate_activity`` overrides the LTL with
        ``generate_wet_transport_ltl_safety_json`` — overhead-forbidden
        zone predicates instead of fragile-upright / fragile-dropped.
    """

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            "--overhead-margin-m", type=float, default=0.02,
            help="XY margin around zone footprints for overhead-forbidden LTL",
        )

    def activity_prefix(self):
        return "auto_wet_transport_on"

    def scene_family(self, ctx):
        return "wet_transport"

    def select_objects(self, args, rng):
        density = DENSITY_PRESETS[args.clutter_density]

        # Target — same fillable-container pool as liquid_transport.
        if args.container_category is not None:
            container_category, container_model = _pick_model_for_category(
                args.container_category, rng,
            )
            container_synset = f"{container_category}.n.01"
        else:
            container_synset, container_category, container_model = (
                select_fillable_container(rng)
            )

        # Zones replace the liquid-fragile slot. Same count as
        # density["fragile_count"] — the clutter pack treats them as
        # the same kind of "around-the-target" obstacle.
        zone_picks = [
            select_water_sensitive(rng, exclude_cats={container_category})
            for _ in range(density["fragile_count"])
        ]

        counts = [(container_category, 1, container_model)]
        for _, cat, model in zone_picks:
            counts.append((cat, 1, model))

        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "target_synset": container_synset,
            "target_category": container_category,
            "target_model": container_model,
            # The clutter activity gen expects "fragile_picks" / "clutter_picks"
            # keys. Map zones into fragile_picks; clutter is empty.
            "fragile_picks": zone_picks,
            "clutter_picks": [],
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        _clutter_ltl, selection = generate_clutter_activity(
            activity_name, support_synset, support_room,
            args.clutter_density, rng=rng,
            pre_selection=args._pre_selection,
        )

        target_synset = selection["target_synset"]
        zone_synsets = sorted({p[0] for p in selection["fragile_picks"]})

        ltl_safety = generate_wet_transport_ltl_safety_json(
            activity_name=activity_name,
            carried_synsets=[target_synset],
            zone_synsets=zone_synsets,
            margin_m=args.overhead_margin_m,
        )

        selection["system_name"] = args.system_name
        selection["overhead_margin_m"] = args.overhead_margin_m
        return ltl_safety, selection

    def diagnostics_extra(self, ctx):
        extra = super().diagnostics_extra(ctx)
        extra["pipeline"] = "wet_transport"
        extra["overhead_margin_m"] = ctx.selection["overhead_margin_m"]
        return extra


def main():
    WetTransportPipeline().run()


if __name__ == "__main__":
    main()
