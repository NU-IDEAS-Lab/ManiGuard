"""Table clutter scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, generates BDDL + ltl_safety.json,
packs clutter objects, places robot, and runs LTL-monitored rollouts.

Usage:
    python -m maniguard.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --dry-run

    python -m maniguard.task_generation.clutter_scene_pipeline \
        --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video
"""

import os

from maniguard.task_generation.pipeline_common import (
    BasePipeline,
    get_spawned_obj,
    make_settle_fn,
    predict_inst_ids,
)
from maniguard.utils.maxrects_pack import PackInputDescriptor, solve_pack
from maniguard.utils.task_spec import _load_footprint_catalog, generate_clutter_activity
import logging

# Edge buffer (m): shrink the picked region by this much on every side so
# packed objects keep a generous gap to the actual surface boundary.
_EDGE_BUFFER_M = 0.05
# Min clearance (m): gap between adjacent padded AABBs in the offline pack
# solve. Big enough for physics settle to converge without interpenetration.
_MIN_CLEARANCE_M = 0.05

log = logging.getLogger(__name__)


class ClutterPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--randomize", action="store_true",
                            help="Randomize target, fragile, and clutter object "
                                 "types each episode")

    def activity_prefix(self):
        return "auto_clutter_on"

    def scene_family(self, ctx):
        return "table"

    def select_objects(self, args, rng):
        from maniguard.utils.task_spec import (
            DENSITY_PRESETS, estimate_object_set_footprint,
        )
        from maniguard.task_generation.utils.clutter_pipeline.select import (
            select_target, select_obstacle, select_fragile,
        )

        density = DENSITY_PRESETS[args.clutter_density]
        target_synset, target_category, target_model = select_target(rng)

        # Pin concrete models for every fragile / clutter atom so the
        # picker uses exact per-model footprints. Excluding the target's
        # category keeps fragile/clutter from re-using the target asset.
        fragile_picks = [select_fragile(rng, exclude_cats={target_category})
                         for _ in range(density["fragile_count"])]
        clutter_picks = [select_obstacle(rng, exclude_cats={target_category})
                         for _ in range(density["clutter_count"])]

        # Every entry is (category, count, model) — exact per-model footprint.
        counts = [(target_category, 1, target_model)]
        for _, cat, model in fragile_picks:
            counts.append((cat, 1, model))
        for _, cat, model in clutter_picks:
            counts.append((cat, 1, model))

        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "target_synset": target_synset,
            "target_category": target_category,
            "target_model": target_model,
            "fragile_picks": fragile_picks,
            "clutter_picks": clutter_picks,
        }

    def generate_activity(self, activity_name, support_synset, support_room,
                          args, rng):
        return generate_clutter_activity(
            activity_name, support_synset, support_room,
            args.clutter_density, rng=rng,
            pre_selection=args._pre_selection,
        )

    def identify_objects(self, ctx):
        obj_sets = ctx.obj_sets
        if not obj_sets.get("target"):
            raise RuntimeError("No target objects found.")
        print(f"[Pipeline] Objects: target={obj_sets['target']}, "
              f"fragile={obj_sets.get('fragile', ())}, "
              f"clutter={obj_sets.get('clutter', ())}")
        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, obj_sets["target"][0])
        # inst_id → DatasetObject lookup used by place_objects to apply
        # the offline solver's placements. spawned_objects is already
        # keyed by inst_id, so this is a straight reference.
        ctx._objects_by_inst = dict(ctx.spawned_objects)

    def offline_pack(self, episode_activities, picked, args, support_obj=None):
        """Pre-compute pack placements for every episode (offline).

        Runs in pure Python over ``object_footprints.json`` + the picked
        region from ``placeable_surfaces_v1.json``. The picked region
        carries the per-instance ``scale_xyz`` from
        ``build_placeable_surface_scales.py`` so the world-frame region
        bounds can be sized without a live env. Caches a
        ``PackSolution`` on each episode's ``selection`` dict; the
        runtime ``place_objects`` later teleports objects to the cached
        positions with no further solving.

        ``support_obj`` is unused — kept in the signature only to match
        the ``BasePipeline.offline_pack`` hook signature, which other
        pipelines may need.
        """
        del support_obj  # noqa: F841 — see docstring
        catalog = _load_footprint_catalog()
        # Support scale (applied to the placeable region's local extents).
        # Object extents come from object_footprints.json and are already in
        # world units (objects are spawned at scale=1), so we only scale the
        # surface bounds — not the per-object AABBs.
        scale_xyz = picked["scale_xyz"]
        sx_scale = float(scale_xyz[0])
        sy_scale = float(scale_xyz[1])

        # World-sized region bounds (still in support-local axes; the world
        # rotation+translation is applied at place time). Shrink by the edge
        # buffer so packed objects keep a gap to the actual surface
        # boundary.
        (lx0, ly0), (lx1, ly1) = picked["xy_min"], picked["xy_max"]
        sx0, sy0 = lx0 * sx_scale, ly0 * sy_scale
        sx1, sy1 = lx1 * sx_scale, ly1 * sy_scale
        bx0, by0 = sx0 + _EDGE_BUFFER_M, sy0 + _EDGE_BUFFER_M
        bx1, by1 = sx1 - _EDGE_BUFFER_M, sy1 - _EDGE_BUFFER_M
        region_w, region_h = max(0.0, bx1 - bx0), max(0.0, by1 - by0)
        print(f"[Pipeline] offline_pack: support scale=({sx_scale:.3f}, "
              f"{sy_scale:.3f}); world region "
              f"{sx1 - sx0:.3f}×{sy1 - sy0:.3f} m "
              f"(local {lx1 - lx0:.3f}×{ly1 - ly0:.3f})")

        run_dir = args.run_dir
        scene_label = args.scene_model
        surface_label = f"{picked['category']}/{picked['model']}"

        for ep, (_, selection) in enumerate(episode_activities):
            spawn_specs = selection["spawn_specs"]
            ep_label = f"ep{ep + 1}"

            # Mirror build_task_object_cfgs' inst_id assignment so place_objects can
            # later look up the spawned DatasetObject by inst_id.
            inst_plan = predict_inst_ids(spawn_specs, episode_label=ep_label)

            pack_inputs = []
            target_inst_id = None
            for inst_id, role, cat, model in inst_plan:
                if model is None:
                    raise RuntimeError(
                        f"offline_pack: spec for {role}/{cat} has model=None — "
                        "every spec must pin a concrete model at select_objects() time."
                    )
                if cat not in catalog or model not in catalog[cat]:
                    raise RuntimeError(
                        f"offline_pack: {cat}/{model} missing from "
                        "object_footprints.json — regenerate the catalog "
                        "before running new categories."
                    )
                ext = catalog[cat][model]["extent_xyz"]
                pack_inputs.append(PackInputDescriptor(
                    inst_id=inst_id, role=role,
                    extent_xy=(ext[0], ext[1]),
                    bottom_offset_z=0.5 * max(ext[2], 0.01),
                ))
                if role == "target" and target_inst_id is None:
                    target_inst_id = inst_id

            sol = solve_pack(
                descriptors=pack_inputs,
                region_bounds=((0.0, 0.0), (region_w, region_h)),
                min_clearance=_MIN_CLEARANCE_M,
                target_inst_id=target_inst_id,
            )
            print(f"[Pipeline] Offline pack {ep_label}: "
                  f"placed={len(sol.placements)}/{len(pack_inputs)}, "
                  f"unplaced={sol.unplaced}")

            # Cache the solution; place_objects applies it after spawn.
            selection["_pack_solution"] = sol

            if run_dir:
                from maniguard.utils.pack_viz import save_pack_viz
                viz_path = os.path.join(run_dir, f"pack_planned_{ep_label}.png")
                save_pack_viz(
                    out_path=viz_path,
                    surface_bounds_xy=((sx0, sy0), (sx1, sy1)),
                    pack_region_bounds=((bx0, by0), (bx1, by1)),
                    placements=sol.placements,
                    descriptors=pack_inputs,
                    unplaced=sol.unplaced,
                    min_clearance=_MIN_CLEARANCE_M,
                    episode_label=ep_label,
                    scene_label=scene_label,
                    surface_label=surface_label,
                )

    def place_objects(self, ctx):
        """Apply the offline-computed pack: teleport then settle.

        All solving happened in ``offline_pack``; ``_setup_session``
        already filtered ``spawn_specs`` down to solver-placed objects
        and renumbered inst_ids, so every placement here corresponds to
        a spawned ``DatasetObject``. We translate region-centred,
        support-local coords into world via the live support pose (yaw
        + position; the scale was baked into the solver region).
        """
        import math
        import torch as th
        from maniguard.utils.tabletop_workspace import TabletopZoneSpec
        from maniguard.task_generation.pipeline_common import (
            _try_upright_objects, _yaw_from_quat,
        )

        sol = ctx.selection.get("_pack_solution")
        if sol is None:
            raise RuntimeError(
                "place_objects: no cached _pack_solution. offline_pack must "
                "run inside _setup_session before place_objects (see "
                "BasePipeline._setup_session)."
            )

        # Live support pose. Scale was already baked into the solver region
        # in offline_pack, so placements are in *world-scaled* support-local
        # coords — only apply yaw + translation here.
        pos, quat = ctx.support_obj.get_position_orientation()
        pos = [float(v) for v in pos]
        scale_vec = ctx.support_obj.scale
        sx_scale = float(scale_vec[0])
        sy_scale = float(scale_vec[1])
        yaw = _yaw_from_quat([float(v) for v in quat])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # Region centroid in world-scaled support-local frame. The solver
        # returned placements region-centred, so we add this centroid to
        # recover support-local coords.
        (lx_min, ly_min) = ctx.args._picked_surface["xy_min"]
        (lx_max, ly_max) = ctx.args._picked_surface["xy_max"]
        centroid_x = 0.5 * (lx_min + lx_max) * sx_scale
        centroid_y = 0.5 * (ly_min + ly_max) * sy_scale

        world_positions: dict = {}
        roles: dict = {}
        for p in sol.placements:
            obj = ctx._objects_by_inst.get(p.inst_id)
            if obj is None:
                continue
            ex = p.cx + centroid_x
            ey = p.cy + centroid_y
            wx = pos[0] + cos_y * ex - sin_y * ey
            wy = pos[1] + sin_y * ex + cos_y * ey
            wz = ctx.table_top_z + p.cz
            half_yaw = 0.5 * (yaw + p.yaw)
            obj.set_position_orientation(
                position=(wx, wy, wz),
                orientation=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            )
            # Zero residual velocity — objects parked at z=10 during prior
            # episodes accumulate gravity, and set_position_orientation
            # alone doesn't reset velocity. Without this they punch through
            # the desk on the first settle step.
            obj.keep_still()
            world_positions[p.inst_id] = (wx, wy, wz)
            roles[p.inst_id] = p.role

        active_objects = {inst: ctx._objects_by_inst[inst] for inst in world_positions}

        settle = make_settle_fn(ctx.og, th)
        settle(active_objects)
        _try_upright_objects(ctx.og, active_objects)
        settle(active_objects)

        # TabletopZoneSpec describes the world-frame placeable rectangle
        # for the Franka edge picker.
        (rx0, _ry0), (rx1, _ry1) = ctx.surface_bounds_xy
        long_axis = "x" if (rx1 - rx0) >= (ctx.surface_bounds_xy[1][1]
                                           - ctx.surface_bounds_xy[0][1]) else "y"
        ctx._zone = TabletopZoneSpec(
            surface_bounds=ctx.surface_bounds_xy,
            obstacle_keepout_bounds=None,
            obstacle_keepout_bounds_seq=(),
            red_zone_bounds=ctx.surface_bounds_xy,
            long_axis=long_axis,
        )
        ctx.active_objects = active_objects
        ctx._world_positions = world_positions
        ctx._active_object_summary = [
            {
                "inst_id": inst_id,
                "scene_object_name": getattr(obj, "name", None),
                "category": str(getattr(obj, "category", "")),
                "role": roles.get(inst_id),
            }
            for inst_id, obj in sorted(active_objects.items())
        ]

    def make_edge_objects(self, ctx):
        from maniguard.utils.franka_edge_align import EdgeAlignObject
        sol = ctx.selection["_pack_solution"]
        objects = tuple(
            EdgeAlignObject(
                name=p.inst_id, role=p.role,
                position_xy=(ctx._world_positions[p.inst_id][0],
                             ctx._world_positions[p.inst_id][1]),
            )
            for p in sol.placements
            if p.inst_id in ctx._world_positions
        )
        if not objects:
            raise RuntimeError("No pack objects for edge alignment.")
        return objects

    def goal_conditions(self, ctx):
        if ctx.target_obj:
            return [{"predicate": "grasping", "subject": "robot", "reference": ctx.target_obj.name}]
        return []

    def diagnostics_extra(self, ctx):
        extra = {
            "pipeline": "table",
            "density": ctx.args.clutter_density,
            "selection": ctx.selection,
            "active_object_summary": ctx._active_object_summary,
            "removed_area_objects": list(ctx.removed_area_objects),
            "removed_robot_base_objects": list(ctx.removed_robot_base_objects),
            "resolved_video_views": list(ctx.resolved_video_views),
        }
        return extra


def main():
    ClutterPipeline().run()


if __name__ == "__main__":
    main()
