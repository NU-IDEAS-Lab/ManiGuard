"""Generic scene generation pipeline.

Auto-discovers a suitable tabletop in any scene, computes placement zones,
determines an object budget, generates BDDL + ltl_safety.json files, and
validates the result. Single-surface focus.

Usage:
    # Dry-run (generate BDDL only, no sim):
    python generic_scene_pipeline.py --scene-model house_double_floor_lower --dry-run

    # Full sim run:
    python generic_scene_pipeline.py --scene-model house_double_floor_lower \\
        --episodes 1 --steps 300 --strict-gate
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Root of the SENTINEL-Lite repository.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_RUNS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pipeline_runs")


def parse_args():
    parser = argparse.ArgumentParser(description="Generic scene generation pipeline")
    parser.add_argument("--scene-model", required=True, help="Scene model identifier.")
    parser.add_argument("--activity-name", default=None, help="Activity name. Auto-generates if absent.")
    parser.add_argument("--config", default=None, help="Path to YAML config template.")
    parser.add_argument("--dry-run", action="store_true", help="Generate BDDL only, no sim validation.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("--steps", type=int, default=5000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--mount-gap-m", type=float, default=0.03, help="Desired Franka base gap from edge.")
    parser.add_argument("--jitter-scale", type=float, default=0.01, help="Action jitter sigma.")
    parser.add_argument("--showcase-gui", action="store_true", help="Enable manual GUI camera.")
    parser.add_argument("--strict-gate", dest="strict_gate", action="store_true")
    parser.add_argument("--no-strict-gate", dest="strict_gate", action="store_false")
    parser.set_defaults(strict_gate=True)
    parser.add_argument("--debug-jsonl", default=None, help="Optional JSONL path for diagnostics.")
    # Clutter controls
    parser.add_argument("--clutter-density", default="medium",
                        choices=["low", "medium", "high", "ultra"])
    parser.add_argument("--pack-jitter-xy", type=float, default=None)
    parser.add_argument("--pack-min-clearance", type=float, default=None)
    parser.add_argument("--zone-utilization-cap", type=float, default=None)
    parser.add_argument("--pack-min-scale", type=float, default=None)
    # Output
    parser.add_argument("--run-dir", default=None,
                        help="Directory for all run outputs (video, JSONL, etc). "
                             "Defaults to outputs/pipeline_runs/<scene>_<timestamp>/")
    parser.add_argument("--save-video", action="store_true",
                        help="Record viewer camera to <run-dir>/rollout.mp4 each step.")
    parser.add_argument("--video-fps", type=int, default=30, help="Video frame rate.")
    return parser.parse_args()


_DENSITY_PRESETS = {
    "low": {"fragile_count": 2, "clutter_count": 1},
    "medium": {"fragile_count": 4, "clutter_count": 2},
    "high": {"fragile_count": 6, "clutter_count": 4},
    "ultra": {"fragile_count": 8, "clutter_count": 6},
}


def _append_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _auto_activity_name(scene_model: str) -> str:
    return f"auto_clutter_on_{scene_model}"


def run_dry_run(args):
    """Generate BDDL + ltl_safety.json without starting the simulator."""
    from omnigibson.utils.bddl_generator import (
        BDDLGenConfig,
        ObjectSpec,
        generate_bddl_problem,
        generate_ltl_safety_json,
        write_activity_files,
    )

    activity_name = args.activity_name or _auto_activity_name(args.scene_model)
    density = _DENSITY_PRESETS[args.clutter_density]

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset="countertop.n.01",
        support_room="kitchen",
        goal_synset="cabinet.n.01",
        goal_room="kitchen",
        goal_predicate="inside",
        objects=[
            ObjectSpec(synset="coffee_cup.n.01", count=1, role="target"),
            ObjectSpec(synset="wineglass.n.01", count=density["fragile_count"], role="fragile"),
            ObjectSpec(synset="plate.n.04", count=density["clutter_count"], role="clutter"),
        ],
    )

    bddl_text = generate_bddl_problem(config)
    ltl_safety = generate_ltl_safety_json(
        activity_name=activity_name,
        fragile_synsets=["wineglass.n.01", "plate.n.04"],
        target_synsets=["coffee_cup.n.01"],
    )

    try:
        import bddl
        activity_dir = os.path.join(
            os.path.dirname(bddl.__file__),
            "activity_definitions", activity_name,
        )
    except ImportError:
        activity_dir = os.path.join("generated_activities", activity_name)

    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)
    print(f"[Pipeline] Dry-run complete:")
    print(f"  BDDL:         {bddl_path}")
    print(f"  ltl_safety:   {json_path}")
    print(f"  activity:     {activity_name}")
    print(f"  density:      {args.clutter_density}")
    print(f"\nGenerated BDDL:\n{bddl_text}")
    print(f"\nGenerated LTL formula: {ltl_safety['combined_ltl']}")

    _append_jsonl(args.debug_jsonl, {
        "event": "dry_run",
        "activity_name": activity_name,
        "scene_model": args.scene_model,
        "density": args.clutter_density,
        "bddl_path": bddl_path,
        "json_path": json_path,
    })

    return activity_name, bddl_path, json_path


def run_sim(args, activity_name=None):
    """Full sim-validation path: surface discovery, pack, robot, gate, LTL."""
    import torch as th
    import yaml
    from bddl.object_taxonomy import ObjectTaxonomy

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import get_scene_path
    from omnigibson.utils.bddl_generator import (
        BDDLGenConfig,
        ObjectSpec,
        generate_bddl_problem,
        generate_ltl_safety_json,
        write_activity_files,
    )
    from omnigibson.utils.clutter_pack_layout import (
        ClutterObjectDescriptor,
        validate_pack_integrity,
    )
    from omnigibson.utils.franka_edge_align import (
        DEFAULT_ROLE_WEIGHTS,
        EdgeAlignObject,
        EdgeAlignRequest,
        place_franka_edge_aligned,
    )
    from omnigibson.utils.kitchen_bar_workspace import (
        TabletopZoneSpec,
        bounds_overlap,
        compute_tabletop_zone,
        compute_zone_capacity,
        contains_point,
    )
    from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec
    from omnigibson.utils.pack_retry_loop import PackRetryConfig, run_pack_retry_loop
    from omnigibson.utils.safety_monitor import TaskLTLMonitor
    from omnigibson.utils.surface_discovery import (
        analyze_surface,
        is_obstacle_like,
        is_table_like,
    )
    import omnigibson.utils.transform_utils as T

    gm.ENABLE_OBJECT_STATES = True

    _OBJECT_TAXONOMY = ObjectTaxonomy()

    if activity_name is None:
        activity_name = args.activity_name or _auto_activity_name(args.scene_model)

    # Detect if the task uses substance systems (e.g. water) and configure
    # GPU dynamics accordingly — same logic as the kitchen bar runner.
    needs_gpu_dynamics = False
    try:
        from bddl.activity import Conditions as _Cond
        _cond = _Cond(
            behavior_activity=activity_name,
            activity_definition=0,
            simulator_name="omnigibson",
            predefined_problem=None,
        )
        for synset in _cond.parsed_objects.keys():
            try:
                if "substance" in _OBJECT_TAXONOMY.get_abilities(synset):
                    needs_gpu_dynamics = True
                    print(f"[Pipeline] GPU dynamics enabled (substance detected: {synset})")
                    break
            except Exception:
                continue
    except Exception:
        pass

    if needs_gpu_dynamics:
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
    else:
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = True

    # -- Load / build config ------------------------------------------------
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = _build_default_config(args.scene_model, activity_name)

    cfg["task"]["activity_name"] = activity_name
    cfg["scene"]["scene_model"] = args.scene_model
    cfg["scene"]["scene_instance"] = None
    scene_json = os.path.join(
        get_scene_path(args.scene_model), "json", f"{args.scene_model}_best.json",
    )
    if os.path.isfile(scene_json):
        cfg["scene"]["scene_file"] = scene_json
    cfg["task"]["online_object_sampling"] = True
    cfg["task"]["use_presampled_robot_pose"] = False

    print(f"[Pipeline] scene={args.scene_model}, activity={activity_name}, "
          f"strict_gate={args.strict_gate}")

    env = og.Environment(configs=cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")
            env.reset()
            og.sim.step()

            # -- Surface discovery ------------------------------------------
            surface_info, support_obj = _discover_best_surface(env)
            print(f"[Pipeline] Best surface: {surface_info.surface.name} "
                  f"(score={surface_info.surface.score:.3f}, "
                  f"area={surface_info.surface.area:.3f}, "
                  f"obstacles={len(surface_info.obstacles)})")

            support_aabb_min, support_aabb_max = support_obj.aabb
            surface_bounds_xy = (
                (float(support_aabb_min[0]), float(support_aabb_min[1])),
                (float(support_aabb_max[0]), float(support_aabb_max[1])),
            )
            table_top_z = float(support_aabb_max[2])

            # Obstacle bounds (if any).
            obstacle_bounds_xy = None
            if surface_info.obstacles:
                obs = surface_info.obstacles[0]
                obstacle_bounds_xy = obs.aabb_xy

            # -- Zone computation -------------------------------------------
            zone = compute_tabletop_zone(
                surface_bounds_xy=surface_bounds_xy,
                obstacle_bounds_xy=obstacle_bounds_xy,
                edge_margin_m=0.04,
                obstacle_keepout_margin_m=0.08,
                obstacle_side_clearance_m=0.015,
            )
            print(f"[Pipeline] Zone: red_zone={zone.red_zone_bounds}, "
                  f"long_axis={zone.long_axis}")

            # -- Build object sets ------------------------------------------
            floor_z = _compute_floor_z(env)
            task_spec = build_manipulation_task_spec(activity_name)
            obj_sets = _build_task_object_sets(env, task_spec)

            if len(obj_sets["target_ids"]) == 0:
                raise RuntimeError("No target objects found in object_scope.")

            target_inst = obj_sets["target_ids"][0]
            target_obj = _get_scope_obj(env, target_inst)

            descriptors, objects_by_inst = _build_descriptors(env, obj_sets)
            if not descriptors:
                raise RuntimeError("No clutter-pack descriptors created.")

            # -- Pack retry loop --------------------------------------------
            pack_config = PackRetryConfig(
                pack_jitter_xy=args.pack_jitter_xy or 0.022,
                pack_min_clearance=args.pack_min_clearance or 0.008,
            )

            def _settle(objs):
                for _ in range(3):
                    og.sim.step()
                for _ in range(7):
                    og.sim.step()
                    for obj in objs.values():
                        try:
                            vel = obj.get_linear_velocity()
                            vz = float(vel[2]) if hasattr(vel, '__getitem__') else 0.0
                            obj.set_linear_velocity(th.tensor([0.0, 0.0, min(0.0, vz)]))
                            obj.set_angular_velocity(th.zeros(3))
                        except Exception:
                            pass
                for obj in objs.values():
                    try:
                        if hasattr(obj, "keep_still"):
                            obj.keep_still()
                    except Exception:
                        pass
                og.sim.step()

            def _park(passive_objs):
                (bx0, by0), (bx1, by1) = zone.surface_bounds
                base_x = bx1 + 1.5
                base_y = by0 - 1.2
                for idx, inst in enumerate(sorted(passive_objs.keys())):
                    obj = passive_objs[inst]
                    x = base_x + 0.18 * (idx % 8)
                    y = base_y - 0.18 * (idx // 8)
                    try:
                        obj.set_position_orientation(
                            position=(x, y, floor_z + 0.06),
                            orientation=(0, 0, 0, 1),
                        )
                        if hasattr(obj, "keep_still"):
                            obj.keep_still()
                    except Exception:
                        pass
                og.sim.step()

            def _validate_poses(objs):
                invalid = []
                for inst, obj in objs.items():
                    try:
                        pos, orn = obj.get_position_orientation()
                        p = [float(pos[i]) for i in range(3)]
                        if not all(math.isfinite(v) for v in p):
                            invalid.append(inst)
                    except Exception:
                        invalid.append(inst)
                return invalid

            def _check_interpen(objs, tol):
                inst_ids = sorted(objs.keys())
                hits = []
                for i, a in enumerate(inst_ids):
                    try:
                        aabb_a = objs[a].aabb
                    except Exception:
                        continue
                    for b in inst_ids[i + 1:]:
                        try:
                            aabb_b = objs[b].aabb
                        except Exception:
                            continue
                        if _aabb_overlap_3d(aabb_a, aabb_b, tol):
                            hits.append((a, b))
                return hits

            pack_result = run_pack_retry_loop(
                support_name=getattr(support_obj, "name", "support"),
                descriptors=descriptors,
                objects_by_inst=objects_by_inst,
                red_zone_bounds=zone.red_zone_bounds,
                table_top_z=table_top_z,
                floor_z=floor_z,
                config=pack_config,
                base_seed=args.seed,
                episode=ep,
                settle_fn=_settle,
                park_fn=_park,
                validate_poses_fn=_validate_poses,
                check_interpenetration_fn=_check_interpen,
                obstacle_keepout_bounds=zone.obstacle_keepout_bounds,
            )
            print(f"[Pipeline] Pack solved: attempt={pack_result.attempt_used}, "
                  f"active={len(pack_result.active_descriptors)}, "
                  f"culls={len(pack_result.cull_history)}")

            # Park remaining inactive.
            passive_after = {
                inst: obj for inst, obj in objects_by_inst.items()
                if inst not in pack_result.active_objects_by_inst
            }
            _park(passive_after)

            # -- Integrity check --------------------------------------------
            integrity = validate_pack_integrity(
                pack_spec=pack_result.pack_spec,
                world_positions=pack_result.world_positions,
                pack_origin_world=pack_result.pack_origin,
                pack_yaw=0.0,
                tol_xy=pack_config.integrity_tol_xy,
            )

            # -- Robot placement (auto edge) --------------------------------
            robot = env.robots[0]
            half_ext = _robot_half_extent_xy(robot)
            pack_objects_world = []
            for entry in pack_result.pack_spec.object_entries:
                if entry.inst_id not in pack_result.world_positions:
                    continue
                wx, wy, _ = pack_result.world_positions[entry.inst_id]
                pack_objects_world.append(EdgeAlignObject(
                    name=entry.inst_id,
                    role=entry.role,
                    position_xy=(wx, wy),
                ))

            if not pack_objects_world:
                raise RuntimeError("No pack objects for edge alignment.")

            preferred_edge = None
            if surface_info.approach_edges:
                preferred_edge = surface_info.approach_edges[0]

            edge_result = place_franka_edge_aligned(EdgeAlignRequest(
                table_aabb_xy=zone.surface_bounds,
                pack_objects_world=tuple(pack_objects_world),
                role_weights=DEFAULT_ROLE_WEIGHTS,
                robot_half_extent_xy=half_ext,
                edge_gap_m=args.mount_gap_m,
                edge_margin_m=0.05,
                scan_offsets_m=(0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20),
                preferred_edge=preferred_edge,
            ))

            final_pos = (
                edge_result.base_pose["position"][0],
                edge_result.base_pose["position"][1],
                floor_z,
            )
            robot.set_position_orientation(
                position=final_pos,
                orientation=edge_result.base_pose["orientation"],
            )
            og.sim.step()

            print(f"[Pipeline] Robot: edge={edge_result.edge_label}, "
                  f"gap={edge_result.gap_actual:.3f}")

            # -- Gate -------------------------------------------------------
            robot_pos = [float(v) for v in robot.get_position_orientation()[0][:3]]
            target_pos = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
            target_dist = math.hypot(robot_pos[0] - target_pos[0], robot_pos[1] - target_pos[1])

            gate_pass = (
                all(math.isfinite(v) for v in robot_pos + target_pos)
                and abs(robot_pos[2] - floor_z) <= 0.03
                and len(edge_result.collision_hits) == 0
                and 0.20 <= target_dist <= 1.10
                and integrity.ok
            )
            print(f"[Pipeline] Gate: pass={gate_pass}")

            if args.strict_gate and not gate_pass:
                raise RuntimeError("Strict gate failed.")

            # -- LTL monitor rollout ----------------------------------------
            ltl_monitor = TaskLTLMonitor(
                env=env,
                activity_name=activity_name,
                scene_model=args.scene_model,
                active_objects_by_inst=pack_result.active_objects_by_inst,
            )
            ltl_monitor.reset()
            ltl_monitor.step(0)

            # -- Video recording setup --------------------------------------
            video_writer = None
            if args.save_video:
                _set_showcase_camera(env, target_obj, robot)
                # Let the renderer settle for a few frames before recording.
                for _ in range(3):
                    og.sim.step()
                video_writer = _init_video_writer(args.save_video, ep, args.video_fps)
                if video_writer is not None:
                    print(f"[Pipeline] Recording video to {_video_path(args.save_video, ep)}")

            executed = 0
            terminated = False
            for _ in range(args.steps):
                action = np.zeros_like(
                    np.asarray(robot.action_space.sample(), dtype=np.float32)
                )
                action += rng.normal(0.0, args.jitter_scale, size=action.shape).astype(np.float32)
                if hasattr(robot.action_space, "low"):
                    action = np.clip(action, robot.action_space.low, robot.action_space.high)
                env._pre_step(action)
                og.sim.step()
                executed += 1

                if video_writer is not None:
                    _record_frame(video_writer)

                ltl_info = ltl_monitor.step(executed)
                if ltl_monitor.violated:
                    terminated = True

                if executed % 50 == 0:
                    print(f"[Pipeline] Step {executed}/{args.steps}")
                if terminated:
                    break

            if video_writer is not None:
                _close_video_writer(video_writer)
                print(f"[Pipeline] Video saved: {_video_path(args.save_video, ep)}")

            summary = ltl_monitor.summary()
            print(f"[Pipeline] Episode done: steps={executed}, "
                  f"violated={summary['violated']}")

            _append_jsonl(args.debug_jsonl, {
                "episode": ep + 1,
                "scene_model": args.scene_model,
                "activity_name": activity_name,
                "surface": surface_info.surface.name,
                "pack_attempt_used": pack_result.attempt_used,
                "gate_pass": gate_pass,
                "ltl_violated": summary["violated"],
                "steps_executed": executed,
            })

            if args.showcase_gui:
                _set_showcase_camera(env, target_obj, robot)

    finally:
        print("[Pipeline] Shutdown simulator.")
        try:
            og.clear()
        except Exception as e:
            print(f"[Pipeline] og.clear warning: {e}")


# ---------------------------------------------------------------------------
# Helpers (sim-dependent)
# ---------------------------------------------------------------------------

def _build_default_config(scene_model: str, activity_name: str) -> dict:
    return {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": activity_name,
            "activity_definition_id": 0,
            "activity_conditions_met": False,
            "online_object_sampling": True,
        },
        "robots": [{
            "type": "FrankaMounted",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": True,
            "controller_config": {
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
    }


def _video_path(base_path: str, episode: int) -> str:
    if base_path.endswith(".mp4"):
        stem, ext = base_path[:-4], ".mp4"
    else:
        stem, ext = base_path, ".mp4"
    return f"{stem}_ep{episode + 1}{ext}"


def _init_video_writer(base_path: str, episode: int, fps: int):
    try:
        from omnigibson.learning.utils.obs_utils import create_video_writer
    except ImportError:
        print("[Pipeline] WARNING: PyAV not available — video recording disabled.")
        return None

    import omnigibson as og
    # Probe the viewer camera resolution from the first frame.
    try:
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        rgb = viewer_obs["rgb"]
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
    except Exception:
        h, w = 720, 1280

    fpath = _video_path(base_path, episode)
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
    container, stream = create_video_writer(
        fpath=fpath,
        resolution=(h, w),
        rate=fps,
    )
    return container, stream


def _record_frame(video_writer):
    import omnigibson as og
    try:
        import av
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        rgb = viewer_obs["rgb"]
        # (H, W, 3 or 4) torch tensor → numpy uint8
        frame_np = rgb[..., :3].cpu().numpy().astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        container, stream = video_writer
        for packet in stream.encode(frame):
            container.mux(packet)
    except Exception as e:
        pass  # Skip frame silently to avoid breaking the rollout.


def _close_video_writer(video_writer):
    try:
        container, stream = video_writer
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception:
        pass


def _discover_best_surface(env):
    """Discover the best table-like surface in the scene."""
    from omnigibson.utils.surface_discovery import (
        analyze_surface,
        is_table_like,
    )

    scene_objects_data = []
    obj_map = {}
    for obj in env.scene.objects:
        name = getattr(obj, "name", "")
        cat = str(getattr(obj, "category", ""))
        try:
            aabb_min, aabb_max = obj.aabb
            aabb_xy = (
                (float(aabb_min[0]), float(aabb_min[1])),
                (float(aabb_max[0]), float(aabb_max[1])),
            )
            top_z = float(aabb_max[2])
        except Exception:
            continue
        scene_objects_data.append({
            "name": name, "category": cat, "aabb_xy": aabb_xy, "top_z": top_z,
        })
        obj_map[name] = obj

    best_analysis = None
    best_obj = None
    for data in scene_objects_data:
        if not is_table_like(data["category"]):
            continue
        analysis = analyze_surface(
            data["name"], data["category"], data["aabb_xy"], data["top_z"],
            scene_objects_data,
        )
        if analysis.surface.score <= 0:
            continue
        if best_analysis is None or analysis.surface.score > best_analysis.surface.score:
            best_analysis = analysis
            best_obj = obj_map[data["name"]]

    if best_analysis is None or best_obj is None:
        raise RuntimeError(f"No suitable table-like surface found in scene '{env.scene.scene_model}'.")

    return best_analysis, best_obj


def _compute_floor_z(env):
    floor_z = 0.0
    scope = getattr(env.task, "object_scope", {}) or {}
    for inst, ent in scope.items():
        if not inst.startswith("floor."):
            continue
        if ent is None or not getattr(ent, "exists", False):
            continue
        obj = getattr(ent, "wrapped_obj", None)
        if obj is None:
            continue
        try:
            _, aabb_max = obj.aabb
            floor_z = max(floor_z, float(aabb_max[2]))
        except Exception:
            continue
    return floor_z


def _iter_scope_objects(env):
    scope = getattr(env.task, "object_scope", {}) or {}
    for inst, ent in scope.items():
        if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
            continue
        obj = getattr(ent, "wrapped_obj", None)
        if obj is None:
            continue
        yield inst, obj


def _get_scope_obj(env, inst):
    scope = getattr(env.task, "object_scope", {}) or {}
    ent = scope.get(inst, None)
    if ent is None or not getattr(ent, "exists", False) or getattr(ent, "is_system", False):
        return None
    return getattr(ent, "wrapped_obj", None)


def _build_task_object_sets(env, task_spec):
    available = {inst for inst, _ in _iter_scope_objects(env)}
    target_ids = [i for i in task_spec.target_ids if i in available]
    fragile_ids = [i for i in task_spec.fragile_ids if i in available and i not in target_ids]
    support_ids = [i for i in task_spec.support_ids if i in available]
    excluded_prefixes = ("agent.", "floor.")
    clutter_ids = sorted([
        i for i in available
        if i not in target_ids and i not in fragile_ids and i not in support_ids
        and not i.startswith(excluded_prefixes)
    ])
    if not target_ids:
        for inst, _ in _iter_scope_objects(env):
            if inst.startswith(("coffee_cup.", "cup.", "mug.")):
                target_ids = [inst]
                break
    return {
        "target_ids": tuple(target_ids),
        "fragile_ids": tuple(sorted(fragile_ids)),
        "support_ids": tuple(sorted(support_ids)),
        "clutter_ids": tuple(clutter_ids),
    }


def _build_descriptors(env, obj_sets):
    from omnigibson.utils.clutter_pack_layout import ClutterObjectDescriptor

    descriptors = []
    objects_by_inst = {}
    for role, id_key in [("target", "target_ids"), ("fragile", "fragile_ids"), ("clutter", "clutter_ids")]:
        for inst in obj_sets[id_key]:
            obj = _get_scope_obj(env, inst)
            if obj is None:
                continue
            try:
                aabb_min, aabb_max = obj.aabb
                mn = [float(aabb_min[i]) for i in range(3)]
                mx = [float(aabb_max[i]) for i in range(3)]
                dx = max(0.01, mx[0] - mn[0])
                dy = max(0.01, mx[1] - mn[1])
                dz = max(0.01, mx[2] - mn[2])
                descriptors.append(ClutterObjectDescriptor(
                    instance_id=inst, role=role,
                    half_extent_xy=(0.5 * dx, 0.5 * dy), height=dz,
                ))
                objects_by_inst[inst] = obj
            except Exception:
                continue
    return descriptors, objects_by_inst


def _robot_half_extent_xy(robot):
    links = getattr(robot, "links", {}) or {}
    for key in ("base_link", "base", "base_footprint", "chassis"):
        if key in links:
            aabb = links[key].aabb
            mn = [float(aabb[0][i]) for i in range(3)]
            mx = [float(aabb[1][i]) for i in range(3)]
            return ((mx[0] - mn[0]) * 0.5, (mx[1] - mn[1]) * 0.5)
    for link in links.values():
        try:
            aabb = link.aabb
            mn = [float(aabb[0][i]) for i in range(3)]
            mx = [float(aabb[1][i]) for i in range(3)]
            return ((mx[0] - mn[0]) * 0.5, (mx[1] - mn[1]) * 0.5)
        except Exception:
            continue
    return (0.15, 0.15)


def _aabb_overlap_3d(aabb_a, aabb_b, tol=0.0):
    a_min = [float(aabb_a[0][i]) for i in range(3)]
    a_max = [float(aabb_a[1][i]) for i in range(3)]
    b_min = [float(aabb_b[0][i]) for i in range(3)]
    b_max = [float(aabb_b[1][i]) for i in range(3)]
    return all(
        min(a_max[i], b_max[i]) - max(a_min[i], b_min[i]) > tol
        for i in range(3)
    )


def _set_showcase_camera(env, target_obj, robot):
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    import torch as th

    robot_pos = [float(v) for v in robot.get_position_orientation()[0][:3]]
    target_pos = [float(v) for v in target_obj.get_position_orientation()[0][:3]]
    center = [
        0.5 * (robot_pos[0] + target_pos[0]),
        0.5 * (robot_pos[1] + target_pos[1]),
        max(robot_pos[2] + 0.7, target_pos[2] + 0.25),
    ]
    cam_pos = [center[0] - 1.0, center[1] - 1.1, center[2] + 0.5]
    direction = np.asarray(
        [center[i] - cam_pos[i] for i in range(3)], dtype=np.float32,
    )
    direction /= max(1e-6, np.linalg.norm(direction))
    pan = float(np.arctan2(-direction[0], direction[1]))
    tilt = float(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    cam_quat = T.euler2quat(th.tensor(
        [math.pi / 2 + tilt, 0.0, pan], dtype=th.float32,
    ))
    og.sim.viewer_camera.set_position_orientation(
        position=cam_pos, orientation=cam_quat.tolist(),
    )
    og.sim.enable_viewer_camera_teleoperation()
    print("[Pipeline] Manual GUI mode: camera teleoperation enabled.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_run_dir(args):
    """Create the run directory and set default output paths."""
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(
            _DEFAULT_RUNS_DIR,
            f"{args.scene_model}_{ts}",
        )
    os.makedirs(args.run_dir, exist_ok=True)

    # Default debug-jsonl into run dir if not explicitly set.
    if args.debug_jsonl is None:
        args.debug_jsonl = os.path.join(args.run_dir, "diagnostics.jsonl")

    # Resolve save-video flag to a concrete path inside run dir.
    if args.save_video is True:
        args.save_video = os.path.join(args.run_dir, "rollout.mp4")
    elif args.save_video is False:
        args.save_video = None
    # else: keep user-provided string path as-is

    print(f"[Pipeline] Run directory: {args.run_dir}")


def main():
    args = parse_args()
    _setup_run_dir(args)

    if args.dry_run:
        run_dry_run(args)
    else:
        activity_name = args.activity_name or _auto_activity_name(args.scene_model)
        # Generate BDDL if not already present.
        try:
            import bddl
            activity_dir = os.path.join(
                os.path.dirname(bddl.__file__),
                "activity_definitions", activity_name,
            )
        except ImportError:
            activity_dir = os.path.join("generated_activities", activity_name)

        bddl_path = os.path.join(activity_dir, "problem0.bddl")
        if not os.path.isfile(bddl_path):
            print(f"[Pipeline] BDDL not found, generating via dry-run...")
            run_dry_run(args)

        run_sim(args, activity_name)


if __name__ == "__main__":
    main()
