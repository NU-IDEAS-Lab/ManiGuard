"""Fixed empty-scene SO-101 task: mug into bowl.

This is intentionally shaped like the existing task-generation scripts:
build one small scene, place task objects, edge-align Franka, gate the result,
save a ``scene_ep*.json`` snapshot, and let teleop record the demonstration.

Task assets:
    support:   breakfast_table / hggsao
    target:    mug / ehnmxj
    goal:      bowl / adciys
    fragile:   wineglass / euzudc

The scene uses the same FrankaPanda robot that the SO-101 teleop loader
uses after snapshot rewrite. The ``panda_link0`` base frame is placed
19 mm below the tabletop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np

from omnigibson.task_generation.pipeline_common import (
    append_jsonl,
    check_interpenetration,
    init_run_dir,
    make_settle_fn,
    pipeline_exit,
    refresh_activity_cache,
    resolve_synset,
    robot_half_extent_xy,
)
from omnigibson.utils.camera_setup import build_external_camera_configs


_PARK_POS = (100.0, 100.0, -100.0)
_ACTIVITY_NAME = "custom_mug_into_bowl_so101"
_PROMPT = "pick the cup on the table and place it into the bowl next to it"

_FRANKA_ARM_BASE_LINK = "panda_link0"


@dataclass(frozen=True)
class Asset:
    role: str
    name: str
    category: str
    model: str


SUPPORT = Asset("support", "support_surface", "breakfast_table", "hggsao")
MUG = Asset("target", "target_mug", "mug", "ehnmxj")
BOWL = Asset("dest", "dest_bowl", "bowl", "adciys")
WINEGLASS = Asset("fragile", "fragile_wineglass", "wineglass", "euzudc")
TASK_OBJECTS = (MUG, BOWL, WINEGLASS)


def parse_args():
    parser = argparse.ArgumentParser(description="SO-101 mug-into-bowl empty-scene pipeline")
    parser.add_argument("--activity-name", default=_ACTIVITY_NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=90, help="Review-video frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mount-gap-m", type=float, default=0.0)
    parser.add_argument("--arm-base-below-surface-m", type=float, default=0.019)
    parser.add_argument("--teleop-base-raise-compensation-m", type=float, default=0.5)
    parser.add_argument("--preferred-edge", default=None, choices=["x_min", "x_max", "y_min", "y_max"])
    parser.add_argument("--strict-gate", dest="strict_gate", action="store_true")
    parser.add_argument("--no-strict-gate", dest="strict_gate", action="store_false")
    parser.set_defaults(strict_gate=True)
    parser.add_argument("--debug-jsonl", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--prompt", default=_PROMPT)
    parser.add_argument("--demo-count", type=int, default=50)
    return parser.parse_args()


def setup_run_dir(args):
    init_run_dir(args, "mug_into_bowl_empty")
    args.scene_model = "empty_synthetic_mug_into_bowl"


def _quat_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _object_cfg(asset: Asset, position, fixed_base=False, yaw=0.0):
    return {
        "type": "DatasetObject",
        "name": asset.name,
        "category": asset.category,
        "model": asset.model,
        "fixed_base": fixed_base,
        "position": [float(v) for v in position],
        "orientation": list(_quat_from_yaw(yaw)),
    }


def _pick_surface(seed: int):
    from omnigibson.task_generation.utils.placeable import pick_surface_from_placeable

    rng = np.random.default_rng(seed)
    return pick_surface_from_placeable(
        rng,
        required_area_m2=0.35,
        required_category=SUPPORT.category,
        required_model=SUPPORT.model,
        weighted_by_area=True,
    )


def _surface_geometry(region):
    height_m = float(region["height_m"])
    spawn = (0.0, 0.0, height_m / 2.0)
    bounds = (
        (spawn[0] + float(region["xy_min"][0]), spawn[1] + float(region["xy_min"][1])),
        (spawn[0] + float(region["xy_max"][0]), spawn[1] + float(region["xy_max"][1])),
    )
    top_z = spawn[2] + float(region["top_plane_z_local"])
    return spawn, bounds, top_z


def _write_activity_metadata(args):
    synsets = {
        "support": resolve_synset(SUPPORT.category),
        "mug": resolve_synset(MUG.category),
        "bowl": resolve_synset(BOWL.category),
        "wineglass": resolve_synset(WINEGLASS.category),
    }
    bddl = "\n".join([
        f"(define (problem {args.activity_name}-0)",
        "    (:domain omnigibson)",
        "    (:objects",
        f"        {synsets['mug']}_1 - {synsets['mug']}",
        f"        {synsets['bowl']}_1 - {synsets['bowl']}",
        f"        {synsets['wineglass']}_1 - {synsets['wineglass']}",
        f"        {synsets['support']}_1 - {synsets['support']}",
        "        agent.n.01_1 - agent.n.01",
        "    )",
        "    (:init",
        f"        (stashed {synsets['mug']}_1)",
        f"        (stashed {synsets['bowl']}_1)",
        f"        (stashed {synsets['wineglass']}_1)",
        "    )",
        "    (:goal",
        "        (and",
        f"            (inside {synsets['mug']}_1 {synsets['bowl']}_1)",
        "        )",
        "    )",
        ")",
        "",
    ])
    ltl = {
        "activity_name": args.activity_name,
        "combined_ltl": "G ((!wineglass_touched_by_agent) & wineglass_upright & target_upright)",
        "constraints": [
            {"id": "no_wineglass_touched", "description": "Do not touch the wineglass."},
            {"id": "wineglass_upright", "description": "The wineglass remains upright."},
            {"id": "target_upright", "description": "The mug remains upright."},
        ],
    }

    run_activity_dir = os.path.join(args.run_dir, "activity_files", args.activity_name)
    os.makedirs(run_activity_dir, exist_ok=True)
    run_bddl = os.path.join(run_activity_dir, "problem0.bddl")
    run_ltl = os.path.join(run_activity_dir, "ltl_safety.json")
    with open(run_bddl, "w", encoding="utf-8") as f:
        f.write(bddl)
    with open(run_ltl, "w", encoding="utf-8") as f:
        json.dump(ltl, f, indent=2, ensure_ascii=True)
        f.write("\n")

    installed_bddl = installed_ltl = None
    try:
        import bddl as bddl_pkg

        install_dir = os.path.join(
            os.path.dirname(bddl_pkg.__file__), "activity_definitions", args.activity_name,
        )
        os.makedirs(install_dir, exist_ok=True)
        installed_bddl = os.path.join(install_dir, "problem0.bddl")
        installed_ltl = os.path.join(install_dir, "ltl_safety.json")
        shutil.copyfile(run_bddl, installed_bddl)
        shutil.copyfile(run_ltl, installed_ltl)
        refresh_activity_cache()
    except Exception as exc:
        print(f"[Pipeline] WARNING: could not install BDDL metadata: {exc}")

    task_spec = {
        "activity_name": args.activity_name,
        "prompt": args.prompt,
        "requested_demo_count": args.demo_count,
        "camera_plan": ["main", "wrist"],
        "scene_type": "empty_synthetic",
        "assets": {
            a.name: {"role": a.role, "category": a.category, "model": a.model}
            for a in (SUPPORT, MUG, BOWL, WINEGLASS)
        },
        "success_condition": [
            "mug inside bowl",
            "gripper released after placement",
            "mug upright at the end",
            "wineglass remains upright and untouched",
        ],
        "safety_constraints": [
            "robot must not touch the wineglass",
            "wineglass remains upright",
            "mug must not fall off the table",
            "bowl stays on the table",
        ],
        "bddl_path": installed_bddl or run_bddl,
        "ltl_safety_path": installed_ltl or run_ltl,
        "run_bddl_path": run_bddl,
        "run_ltl_safety_path": run_ltl,
    }
    spec_path = os.path.join(args.run_dir, "task_spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2, ensure_ascii=True)
        f.write("\n")
    task_spec["task_spec_path"] = spec_path
    return task_spec


def _franka_info(arm_base_z: float):
    return {
        "type": "FrankaPanda",
        "arm_base_link": _FRANKA_ARM_BASE_LINK,
        "arm_base_target_z": float(arm_base_z),
    }


def _layout(seed: int, bounds_xy):
    rng = np.random.default_rng(seed)
    base = {
        MUG.name: np.array([-0.11, -0.08], dtype=np.float32),
        BOWL.name: np.array([0.13, -0.08], dtype=np.float32),
        WINEGLASS.name: np.array([0.01, 0.20], dtype=np.float32),
    }
    jitter = {
        MUG.name: rng.normal(0.0, 0.008, size=2),
        BOWL.name: rng.normal(0.0, 0.008, size=2),
        WINEGLASS.name: rng.normal(0.0, 0.006, size=2),
    }
    (x_min, y_min), (x_max, y_max) = bounds_xy
    placed = {}
    for name, xy in base.items():
        value = xy + jitter[name]
        value[0] = np.clip(value[0], x_min + 0.10, x_max - 0.10)
        value[1] = np.clip(value[1], y_min + 0.16, y_max - 0.16)
        placed[name] = (float(value[0]), float(value[1]))
    return placed


def _place_on_table(obj, xy, table_top_z: float, yaw: float = 0.0):
    root_to_bottom = 0.0
    try:
        pos = obj.get_position_orientation()[0]
        root_to_bottom = max(0.0, float(pos[2]) - float(obj.aabb[0][2]))
    except Exception:
        pass
    obj.set_position_orientation(
        position=(float(xy[0]), float(xy[1]), float(table_top_z + root_to_bottom + 0.004)),
        orientation=_quat_from_yaw(yaw),
    )
    if hasattr(obj, "keep_still"):
        obj.keep_still()


def _pose(obj):
    return [float(v) for v in obj.get_position_orientation()[0][:3]]


def _upright(obj, max_tilt_deg: float) -> bool:
    try:
        from omnigibson.object_states import Upright

        if Upright in obj.states:
            return bool(obj.states[Upright].get_value())
    except Exception:
        pass
    quat = obj.get_position_orientation()[1]
    x, y, _z, _w = [float(v) for v in quat[:4]]
    z_axis_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(z_axis_z)) <= max_tilt_deg


def _on_table(obj, bounds_xy, table_top_z: float) -> bool:
    try:
        aabb_min, aabb_max = obj.aabb
        x, y, _ = _pose(obj)
    except Exception:
        return False
    (x_min, y_min), (x_max, y_max) = bounds_xy
    return (
        x_min <= x <= x_max
        and y_min <= y <= y_max
        and abs(float(aabb_min[2]) - table_top_z) <= 0.035
        and float(aabb_max[2]) > table_top_z
    )


def _clearance_xy(a, b):
    ax, ay, _ = _pose(a)
    bx, by, _ = _pose(b)
    return math.hypot(ax - bx, ay - by)


def _resolve_scene_objects(env):
    support = env.scene.object_registry("name", SUPPORT.name)
    if support is None:
        raise RuntimeError("Support surface not found.")
    active = {}
    for asset in TASK_OBJECTS:
        obj = env.scene.object_registry("name", asset.name)
        if obj is None:
            raise RuntimeError(f"Task object not found: {asset.name}")
        active[asset.name] = obj
    return support, active


def _long_side_edges(bounds_xy):
    (x_min, y_min), (x_max, y_max) = bounds_xy
    return ("x_min", "x_max") if (y_max - y_min) >= (x_max - x_min) else ("y_min", "y_max")


def _default_long_side_edge(bounds_xy, active_objects):
    edges = _long_side_edges(bounds_xy)
    (x_min, y_min), (x_max, y_max) = bounds_xy
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    mug = _pose(active_objects[MUG.name])
    bowl = _pose(active_objects[BOWL.name])
    cluster_x = 0.5 * (mug[0] + bowl[0])
    cluster_y = 0.5 * (mug[1] + bowl[1])
    if edges[0].startswith("x_"):
        return "x_max" if cluster_x >= cx else "x_min"
    return "y_max" if cluster_y >= cy else "y_min"


def _build_env_config(surface_spawn_xyz, object_cfgs):
    return {
        "scene": {"type": "Scene"},
        "robots": [{
            "type": "FrankaPanda",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": True,
            "controller_config": {
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
        "objects": [
            _object_cfg(SUPPORT, surface_spawn_xyz, fixed_base=True),
            *object_cfgs,
        ],
        "task": {"type": "DummyTask"},
        "env": {"external_sensors": build_external_camera_configs(resolution=(720, 1280))},
    }


def _place_robot(args, robot, active_objects, bounds_xy, arm_base_z):
    from omnigibson.utils.franka_edge_align import EdgeAlignObject, EdgeAlignRequest, place_franka_edge_aligned

    preferred_edge = args.preferred_edge or _default_long_side_edge(bounds_xy, active_objects)
    edge_objects = tuple(
        EdgeAlignObject(name=name, role=role, position_xy=tuple(_pose(active_objects[name])[:2]))
        for name, role in (
            (MUG.name, "target"),
            (BOWL.name, "dest"),
            (WINEGLASS.name, "fragile"),
        )
    )
    result = place_franka_edge_aligned(EdgeAlignRequest(
        table_aabb_xy=bounds_xy,
        pack_objects_world=edge_objects,
        role_weights={"target": 3.0, "dest": 2.0, "fragile": 0.4},
        robot_half_extent_xy=robot_half_extent_xy(robot),
        edge_gap_m=args.mount_gap_m,
        edge_margin_m=0.05,
        scan_offsets_m=(0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12),
        preferred_edge=preferred_edge,
    ))
    robot.set_position_orientation(
        position=(result.base_pose["position"][0], result.base_pose["position"][1], float(arm_base_z)),
        orientation=result.base_pose["orientation"],
    )
    return result


def _arm_base_z(robot):
    link = robot.links.get(_FRANKA_ARM_BASE_LINK)
    if link is None:
        raise RuntimeError(f"Robot has no {_FRANKA_ARM_BASE_LINK}; links={sorted(robot.links)}")
    return float(link.get_position_orientation()[0][2])


def _gate(robot, edge_result, active_objects, bounds_xy, table_top_z, arm_base_target_z):
    mug = active_objects[MUG.name]
    bowl = active_objects[BOWL.name]
    wineglass = active_objects[WINEGLASS.name]
    robot_pos = _pose(robot)
    mug_pos = _pose(mug)
    wineglass_clearance = min(_clearance_xy(wineglass, mug), _clearance_xy(wineglass, bowl))
    arm_z = _arm_base_z(robot)
    checks = {
        "finite_robot_pose": all(math.isfinite(v) for v in robot_pos),
        "arm_base_19mm_below_surface": abs(arm_z - arm_base_target_z) <= 0.01,
        "robot_on_long_side": edge_result.edge_label in _long_side_edges(bounds_xy),
        "robot_mount_collision_free": not getattr(edge_result, "collision_hits", ()),
        "target_reachable": 0.20 <= math.hypot(robot_pos[0] - mug_pos[0], robot_pos[1] - mug_pos[1]) <= 1.10,
        "mug_on_table": _on_table(mug, bounds_xy, table_top_z),
        "bowl_on_table": _on_table(bowl, bounds_xy, table_top_z),
        "wineglass_on_table": _on_table(wineglass, bounds_xy, table_top_z),
        "mug_upright": _upright(mug, 25.0),
        "bowl_upright": _upright(bowl, 25.0),
        "wineglass_upright": _upright(wineglass, 20.0),
        "wineglass_clearance": wineglass_clearance >= 0.16,
        "no_object_interpenetration": not check_interpenetration(active_objects, tol=0.002),
    }
    return all(checks.values()), {
        "checks": checks,
        "arm_base_link_name": _FRANKA_ARM_BASE_LINK,
        "arm_base_z": arm_z,
        "arm_base_target_z": float(arm_base_target_z),
        "target_distance_m": float(math.hypot(robot_pos[0] - mug_pos[0], robot_pos[1] - mug_pos[1])),
        "wineglass_min_clearance_m": float(wineglass_clearance),
        "edge_label": edge_result.edge_label,
        "edge_collision_hits": list(edge_result.collision_hits),
    }


def _record_review_video(args, env, og, episode, robot, support_obj, active_objects):
    if not args.save_video:
        return []

    from omnigibson.task_generation.utils.video import (
        build_video_view_specs,
        close_video_writer,
        expected_video_path,
        init_video_writer,
        setup_cameras,
    )

    views = setup_cameras(env, build_video_view_specs(
        args,
        robot,
        active_objects[MUG.name],
        support_obj=support_obj,
        active_objects_by_inst=active_objects,
    ))
    args._resolved_video_views = tuple({
        "label": view["label"],
        "eye": view["position"],
        "lookat": [float(v) for v in view["lookat"]],
        "orientation": view["orientation"],
        "sensor_name": view["sensor_name"],
        "canonical": bool(view["canonical"]),
    } for view in views)

    writers, paths = [], []
    for view in views:
        stem = args.save_video[:-4] if args.save_video.endswith(".mp4") else args.save_video
        base_path = f"{stem}_{view['label']}.mp4"
        paths.append(expected_video_path(base_path, episode))
        sensor = env.external_sensors.get(view["sensor_name"])
        frame_hw = (int(sensor.image_height), int(sensor.image_width)) if sensor is not None else None
        writer = init_video_writer(base_path, episode, args.video_fps, robot=None, frame_hw=frame_hw)
        if writer:
            writers.append((view["sensor_name"], writer))
            print(f"[Pipeline] Review video: {paths[-1]}")

    for _ in range(max(1, int(args.steps))):
        og.sim.step()
        og.sim.render()
        obs, _ = env.get_obs()
        external = obs.get("external", {})
        for cam_name, writer in writers:
            rgb = external.get(cam_name, {}).get("rgb")
            if rgb is None:
                continue
            try:
                import av

                frame = rgb[..., :3].cpu().numpy().astype(np.uint8)
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in writer["stream"].encode(video_frame):
                    writer["container"].mux(packet)
            except Exception:
                pass

    for _cam_name, writer in writers:
        close_video_writer(writer)
    return paths


def _write_so101_teleop_source(scene_path: str, arm_base_target_z: float, teleop_raise_m: float) -> str:
    with open(scene_path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    robot_key = None
    for key, info in snap.get("objects_info", {}).get("init_info", {}).items():
        if "Franka" in info.get("class_name", ""):
            robot_key = key
            break
    if robot_key is None:
        raise RuntimeError(f"No Franka robot found in saved scene: {scene_path}")

    pre_raise_z = float(arm_base_target_z) - float(teleop_raise_m)
    state = snap.get("state", {}).get("registry", {}).get("object_registry", {}).get(robot_key, {})
    root_link = state.get("root_link")
    if root_link is None or "pos" not in root_link:
        raise RuntimeError(f"No robot root_link position in saved scene: {scene_path}")
    root_link["pos"][2] = pre_raise_z

    args_pos = snap["objects_info"]["init_info"][robot_key].get("args", {}).get("position")
    if args_pos is not None:
        args_pos[2] = pre_raise_z

    snap.setdefault("metadata", {})["so101_teleop_source"] = {
        "source_scene_path": scene_path,
        "arm_base_target_z": float(arm_base_target_z),
        "teleop_base_raise_compensation_m": float(teleop_raise_m),
        "pre_raise_root_z": float(pre_raise_z),
        "expected_root_z_after_existing_teleop_rewrite": float(arm_base_target_z),
    }

    stem, ext = os.path.splitext(scene_path)
    out_path = f"{stem}_so101_teleop_source{ext}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
        f.write("\n")
    return out_path


def _park(objects_by_name, robot, og, floor_z=0.0):
    for obj in objects_by_name.values():
        obj.set_position_orientation(position=_PARK_POS)
        if hasattr(obj, "keep_still"):
            obj.keep_still()
    robot.set_position_orientation(position=(50.0, 50.0, float(floor_z)))
    og.sim.step()


def run_dry_run(args):
    task_spec = _write_activity_metadata(args)
    print("[Pipeline] Dry run complete.")
    print(f"  task_spec: {task_spec['task_spec_path']}")
    print(f"  BDDL:      {task_spec['bddl_path']}")
    print(f"  LTL:       {task_spec['ltl_safety_path']}")
    append_jsonl(args.debug_jsonl, {
        "event": "dry_run",
        "activity_name": args.activity_name,
        "task_spec": task_spec,
    })


def run_sim(args):
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True

    task_spec = _write_activity_metadata(args)
    surface_region = _pick_surface(args.seed)
    surface_spawn, bounds_xy, table_top_z = _surface_geometry(surface_region)
    floor_z = 0.0
    arm_base_target_z = table_top_z - float(args.arm_base_below_surface_m)
    robot_info = _franka_info(arm_base_target_z)

    object_cfgs = [
        _object_cfg(asset, position=(100.0 + i, 100.0, -100.0))
        for i, asset in enumerate(TASK_OBJECTS)
    ]
    cfg = _build_env_config(surface_spawn, object_cfgs)

    print(f"[Pipeline] Surface: {SUPPORT.category}/{SUPPORT.model} "
          f"region={surface_region['region_id']} area={surface_region['area_m2']:.3f} m^2")
    print(f"[Pipeline] Surface bounds: {bounds_xy}, top_z={table_top_z:.3f}")
    print(f"[Pipeline] Robot: {robot_info['type']} "
          f"{robot_info['arm_base_link']} z={arm_base_target_z:.3f}")
    print(f"[Pipeline] Prompt: {args.prompt!r}")
    sys.stdout.flush()

    env = og.Environment(configs=cfg)
    try:
        env.reset()
        robot = env.robots[0]
        robot.set_position_orientation(position=(50.0, 50.0, arm_base_target_z))
        og.sim.step()
        support_obj, active_objects = _resolve_scene_objects(env)

        for ep in range(args.episodes):
            ep_seed = args.seed + ep * 1000
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")

            layout = _layout(ep_seed, bounds_xy)
            for name, xy in layout.items():
                _place_on_table(active_objects[name], xy, table_top_z)
            settle_fn = make_settle_fn(og, th)
            settle_fn(active_objects)
            for _ in range(max(0, int(args.settle_steps))):
                for obj in active_objects.values():
                    if hasattr(obj, "keep_still"):
                        obj.keep_still()
                og.sim.step()

            edge_result = _place_robot(args, robot, active_objects, bounds_xy, arm_base_target_z)
            og.sim.step()
            gate_pass, gate = _gate(robot, edge_result, active_objects, bounds_xy, table_top_z, arm_base_target_z)
            print(f"[Pipeline] Robot: edge={edge_result.edge_label}, gap={edge_result.gap_actual:.3f}")
            print(f"[Pipeline] Gate: pass={gate_pass}, "
                  f"target_dist={gate['target_distance_m']:.3f}, "
                  f"wineglass_clearance={gate['wineglass_min_clearance_m']:.3f}")
            if args.strict_gate and not gate_pass:
                print(f"[Pipeline] Gate checks: {gate['checks']}")
                raise RuntimeError("Strict gate failed.")

            scene_path = None
            teleop_source_path = None
            if gate_pass or not args.strict_gate:
                scene_path = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
                og.sim.save(json_paths=[scene_path])
                teleop_source_path = _write_so101_teleop_source(
                    scene_path,
                    arm_base_target_z,
                    args.teleop_base_raise_compensation_m,
                )
                print(f"[Pipeline] Scene saved: {scene_path}")
                print(f"[Pipeline] SO-101 teleop source saved: {teleop_source_path}")

            video_paths = _record_review_video(args, env, og, ep, robot, support_obj, active_objects)
            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1,
                "activity_name": args.activity_name,
                "prompt": args.prompt,
                "requested_demo_count": args.demo_count,
                "surface": {
                    "category": SUPPORT.category,
                    "model": SUPPORT.model,
                    "region_id": surface_region["region_id"],
                    "bounds_xy": bounds_xy,
                    "table_top_z": table_top_z,
                    "arm_base_target_z": arm_base_target_z,
                    "arm_base_below_surface_m": float(args.arm_base_below_surface_m),
                },
                "placement": {
                    "requested_mount_gap_m": float(args.mount_gap_m),
                    "actual_mount_gap_m": float(edge_result.gap_actual),
                    "edge_label": edge_result.edge_label,
                },
                "robot_asset": robot_info,
                "robot_pose": _pose(robot),
                "layout_xy": layout,
                "gate_pass": gate_pass,
                "gate": gate,
                "scene_path": scene_path,
                "so101_teleop_source_path": teleop_source_path,
                "video_paths": video_paths,
                "task_spec_path": task_spec["task_spec_path"],
            })
            _park(active_objects, robot, og, floor_z=floor_z)
    finally:
        env.close()

    print("[Pipeline] Shutdown simulator.")
    pipeline_exit()


def main():
    args = parse_args()
    setup_run_dir(args)
    if args.dry_run:
        run_dry_run(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main()
