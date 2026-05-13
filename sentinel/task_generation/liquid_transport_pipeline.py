"""Liquid transport scene generation pipeline.

Extends the tabletop clutter pipeline: places a liquid-filled container
on a table with fragile/clutter obstacles, then runs LTL-monitored
rollouts that track spill, tilt, and obstacle safety.

The target is a graspable + fillable container drawn from
``fillable_container_pool.json`` (115 categories, 365 models — every
``status=graspable`` model whose BEHAVIOR taxonomy entry has the
``fillable`` / ``openfillable`` ability). Fragiles use the clutter
pipeline's ``fragile_pool.json``; clutter uses ``table_obstacle_pool``.
The ``--difficulty`` flag only controls the liquid-specific spill
threshold and tilt limit.

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
            "--container-category", default=None,
            help="Specific container category (random from "
                 "fillable_container_pool.json if omitted)",
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
            DENSITY_PRESETS, estimate_object_set_footprint,
            _pick_model_for_category,
        )
        from sentinel.task_generation.utils.clutter_pipeline.select import (
            select_fillable_container, select_obstacle,
        )
        from sentinel.task_generation.utils.liquid_transport.select import (
            select_liquid_fragile,
        )

        density = DENSITY_PRESETS[args.clutter_density]

        # Target — graspable + fillable container. ``--container-category``
        # pins the category; the model is then sampled uniformly from
        # that category's graspable+fillable models in the catalog.
        if args.container_category is not None:
            container_category = args.container_category
            container_category, container_model = _pick_model_for_category(
                container_category, rng,
            )
            container_synset = f"{container_category}.n.01"
        else:
            container_synset, container_category, container_model = (
                select_fillable_container(rng)
            )

        # Fragile — liquid-transport-specific pool that excludes
        # particle-modifier categories (which crash under GPU dynamics —
        # see ``utils/liquid_transport/build_liquid_fragile_pool.py``).
        # Clutter still uses the shared table_obstacle pool.
        fragile_picks = [select_liquid_fragile(rng, exclude_cats={container_category})
                         for _ in range(density["fragile_count"])]
        clutter_picks = [select_obstacle(rng, exclude_cats={container_category})
                         for _ in range(density["clutter_count"])]

        counts = [(container_category, 1, container_model)]
        for _, cat, model in fragile_picks:
            counts.append((cat, 1, model))
        for _, cat, model in clutter_picks:
            counts.append((cat, 1, model))

        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "target_synset": container_synset,
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
        """Pack objects via clutter logic, then fill the target container.

        ``Filled.set_value`` samples particles across the container's
        volume-link AABB and spawns them at points that pass an
        in-volume + non-contact filter. Any residual tilt or linear
        velocity on the container at fill time lets particles drift
        over the rim on the first sim step — visible as liquid spread
        across the table. Snap the target to identity orientation +
        zero velocity right before filling, then ``keep_still`` it on
        every post-fill step so the new water mass can't slosh the
        container itself. Identity quat assumes BEHAVIOR's authored
        upright is +Z, which holds for every category in
        ``fillable_container_pool.json`` (graspable+fillable
        containers — cups, bowls, bottles, jars, …).
        """
        super().place_objects(ctx)

        target = ctx.target_obj
        if target is None:
            raise RuntimeError(
                "liquid_transport place_objects: ctx.target_obj is None — "
                "identify_objects must populate it before fill."
            )

        pos, _ = target.get_position_orientation()
        target.set_position_orientation(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        target.keep_still()
        ctx.og.sim.step()

        from omnigibson.object_states import Filled
        system_name = ctx.selection["system_name"]
        system = ctx.env.scene.get_system(system_name)
        if Filled not in target.states:
            raise RuntimeError(
                f"liquid_transport: target {target.name} "
                f"({target.category}) does not support Filled state — "
                "fillable_container_pool.json admitted a category whose "
                "taxonomy entry lacks 'fillable'/'openfillable'."
            )
        # Wipe particles left over from previous episodes. The water
        # system is scene-global, not container-scoped — once the
        # previous episode's container is parked at z=-100 its particles
        # are released, fall onto the floor, and stay in the system.
        # Each episode-N fill would then add MORE particles on top,
        # eventually leaving a stray particle with a degenerate pose
        # that crashes the system-registry dump at save time
        # (decompose_mat: 'matrices have perspective components').
        system.remove_all_particles()
        ctx.og.sim.step()
        target.states[Filled].set_value(system, True)
        print(f"[Pipeline] Container filled with {system_name}")

        # Settle particles while pinning the container. Without the
        # per-step keep_still the water mass shifts the CoM, the
        # container tilts a few degrees on step 1, and the same
        # particles dribble out the opening before damping.
        for _ in range(20):
            target.keep_still()
            ctx.og.sim.step()

    def extra_gate_checks(self, ctx):
        if not super().extra_gate_checks(ctx):
            return False
        from omnigibson.object_states import ContainedParticles
        system_name = ctx.selection["system_name"]
        system = ctx.env.scene.get_system(system_name)
        data = ctx.target_obj.states[ContainedParticles].get_value(system)
        n = data.n_in_volume
        print(f"[Pipeline] Container particle count: {n}")
        return n > 0

    def diagnostics_extra(self, ctx):
        extra = super().diagnostics_extra(ctx)
        extra["difficulty"] = ctx.args.difficulty
        extra["system_name"] = ctx.selection["system_name"]
        extra["pipeline"] = "liquid_transport"
        return extra


def main():
    LiquidTransportPipeline().run()


if __name__ == "__main__":
    main()
