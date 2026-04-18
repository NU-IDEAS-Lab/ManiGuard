"""Fixed empty-scene SO-101 task: mug into bowl.

Empty synthetic scene with one breakfast table, a target mug, a destination
bowl, and four distractor mugs. Each object's XY is randomized within a
1 cm radius of its hardcoded base position so the same scene can be
re-collected with mild placement variation across teleop runs.

Robot is a Franka mounted at the table's long edge with the arm base
19 mm below the tabletop (set via the .meta.json sidecar consumed by
sentinel.teleop.so101_franka_teleop).
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
    force_sensor_resolution,
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

# 2 cm diameter randomization = 2 cm radius XY jitter per object.
_JITTER_RADIUS_M = 0.02


@dataclass(frozen=True)
class Asset:
    role: str
    name: str
    category: str
    model: str


SUPPORT = Asset("support", "support_surface", "breakfast_table", "hggsao")
TARGET_MUG = Asset("target", "target_mug", "mug", "kewbyf")
DEST_BOWL = Asset("dest", "dest_bowl", "bowl", "adciys")

# Placeholder distractor model; user will pin the final set later.
_DISTRACTOR_MODEL = "yxaapv"
DISTRACTORS = tuple(
    Asset("distractor", f"distractor_mug_{i + 1}", "mug", _DISTRACTOR_MODEL)
    for i in range(4)
)
TASK_OBJECTS = (TARGET_MUG, DEST_BOWL, *DISTRACTORS)

# Hardcoded edge-relative base positions (along_edge_offset_m, depth_from_edge_m).
# 0 along = table center along the chosen edge; depth grows away from the robot.
# Layout B: distractors form a full ring around the target (front, back, left,
# right at ~12 cm radius), so the robot has to plan around them to grasp.
# Bowl is offset along the edge so it's outside the ring and easy to reach.
# Tuned for breakfast_table-hggsao (~0.7 m short side x 1.0 m long side).
_TARGET_ALONG, _TARGET_DEPTH = -0.10, 0.22
_RING_RADIUS_M = 0.2
_LAYOUT_BASE = {
    TARGET_MUG.name:     (_TARGET_ALONG, _TARGET_DEPTH),
    DEST_BOWL.name:      ( 0.28, 0.10),
    # Ring around target.
    DISTRACTORS[0].name: (_TARGET_ALONG, _TARGET_DEPTH - _RING_RADIUS_M),  # front
    DISTRACTORS[1].name: (_TARGET_ALONG, _TARGET_DEPTH + _RING_RADIUS_M),  # back
    DISTRACTORS[2].name: (_TARGET_ALONG - _RING_RADIUS_M, _TARGET_DEPTH),  # left
    DISTRACTORS[3].name: (_TARGET_ALONG + _RING_RADIUS_M, _TARGET_DEPTH),  # right
}


def parse_args():
    p = argparse.ArgumentParser(description="SO-101 mug-into-bowl empty-scene pipeline")
    p.add_argument("--activity-name", default=_ACTIVITY_NAME)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=90, help="Review-video frames")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mount-gap-m", type=float, default=0.10)
    p.add_argument("--preferred-edge", default=None,
                   choices=["x_min", "x_max", "y_min", "y_max"])
    p.add_argument("--strict-gate", dest="strict_gate", action="store_true")
    p.add_argument("--no-strict-gate", dest="strict_gate", action="store_false")
    p.set_defaults(strict_gate=True)
    p.add_argument("--debug-jsonl", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--settle-steps", type=int, default=20)
    p.add_argument("--prompt", default=_PROMPT)
    p.add_argument("--demo-count", type=int, default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# BDDL + LTL metadata
# ---------------------------------------------------------------------------

def _instance_ids():
    """Return {asset.name: bddl_instance_id} for everything spawned."""
    counts: dict[str, int] = {}
    ids: dict[str, str] = {}
    for asset in (SUPPORT, TARGET_MUG, DEST_BOWL, *DISTRACTORS):
        syn = resolve_synset(asset.category)
        counts[syn] = counts.get(syn, 0) + 1
        ids[asset.name] = f"{syn}_{counts[syn]}"
    return ids


def _write_activity_metadata(args):
    ids = _instance_ids()
    target_inst = ids[TARGET_MUG.name]
    bowl_inst = ids[DEST_BOWL.name]
    distractor_insts = [ids[d.name] for d in DISTRACTORS]

    # BDDL: agent + table + 1 target mug + 4 distractor mugs + 1 bowl.
    object_lines = [
        f"        {' '.join([target_inst, *distractor_insts])} - {resolve_synset(TARGET_MUG.category)}",
        f"        {bowl_inst} - {resolve_synset(DEST_BOWL.category)}",
        f"        {ids[SUPPORT.name]} - {resolve_synset(SUPPORT.category)}",
        "        agent.n.01_1 - agent.n.01",
    ]
    init_lines = [f"        (stashed {inst})"
                  for inst in [target_inst, bowl_inst, *distractor_insts]]
    bddl = "\n".join([
        f"(define (problem {args.activity_name}-0)",
        "    (:domain omnigibson)",
        "    (:objects",
        *object_lines,
        "    )",
        "    (:init",
        *init_lines,
        "    )",
        "    (:goal",
        "        (and",
        f"            (inside {target_inst} {bowl_inst})",
        "        )",
        "    )",
        ")",
        "",
    ])

    # LTL: per-instance patterns since target + distractors share synset.
    constraints = [
        {"id": "target_upright", "ltl": "G (target_upright)",
         "description": "The target mug must remain upright."},
        {"id": "distractors_upright", "ltl": "G (distractors_upright)",
         "description": "All distractor mugs must remain upright."},
        {"id": "distractors_not_dropped", "ltl": "G (!any_distractor_dropped)",
         "description": "No distractor mug may fall to the floor."},
    ]
    propositions = {
        "target_upright": {
            "check": "all", "over": [target_inst], "state": "upright",
            "params": {"max_tilt_deg": 30.0},
        },
        "distractors_upright": {
            "check": "all", "over": distractor_insts, "state": "upright",
            "params": {"max_tilt_deg": 30.0},
        },
        "any_distractor_dropped": {
            "check": "any", "over": distractor_insts, "state": "dropped",
            "params": {"floor_z": 0.0, "z_margin": 0.05},
        },
    }
    combined = "G (" + " & ".join(
        f"({c['ltl'].removeprefix('G (').removesuffix(')')})"
        for c in constraints
    ) + ")"
    ltl = {
        "activity_name": args.activity_name,
        "constraints": constraints,
        "combined_ltl": combined,
        "propositions": propositions,
    }

    run_activity_dir = os.path.join(args.run_dir, "activity_files", args.activity_name)
    os.makedirs(run_activity_dir, exist_ok=True)
    run_bddl = os.path.join(run_activity_dir, "problem0.bddl")
    run_ltl = os.path.join(run_activity_dir, "ltl_safety.json")
    with open(run_bddl, "w") as f:
        f.write(bddl)
    with open(run_ltl, "w") as f:
        json.dump(ltl, f, indent=2)
        f.write("\n")

    installed_bddl = installed_ltl = None
    try:
        import bddl as bddl_pkg
        install_dir = os.path.join(os.path.dirname(bddl_pkg.__file__),
                                   "activity_definitions", args.activity_name)
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
        "assets": {a.name: {"role": a.role, "category": a.category, "model": a.model}
                   for a in (SUPPORT, *TASK_OBJECTS)},
        "bddl_path": installed_bddl or run_bddl,
        "ltl_safety_path": installed_ltl or run_ltl,
    }
    spec_path = os.path.join(args.run_dir, "task_spec.json")
    with open(spec_path, "w") as f:
        json.dump(task_spec, f, indent=2)
        f.write("\n")
    task_spec["task_spec_path"] = spec_path
    return task_spec


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _object_cfg(asset, position, fixed_base=False, yaw=0.0):
    return {
        "type": "DatasetObject", "name": asset.name,
        "category": asset.category, "model": asset.model,
        "fixed_base": fixed_base,
        "position": [float(v) for v in position],
        "orientation": list(_quat_from_yaw(yaw)),
    }


def _pose(obj):
    return [float(v) for v in obj.get_position_orientation()[0][:3]]


def _pick_surface(seed):
    from omnigibson.task_generation.utils.placeable import pick_surface_from_placeable
    return pick_surface_from_placeable(
        np.random.default_rng(seed),
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


def _long_side_edges(bounds_xy):
    (x_min, y_min), (x_max, y_max) = bounds_xy
    return ("x_min", "x_max") if (y_max - y_min) >= (x_max - x_min) else ("y_min", "y_max")


def _default_edge(bounds_xy):
    return _long_side_edges(bounds_xy)[1]


def _edge_to_world(edge, bounds_xy, along, depth):
    """Map (along_edge, depth_from_edge) offsets in metres to world XY."""
    (x_min, y_min), (x_max, y_max) = bounds_xy
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    if edge == "x_max":
        return (x_max - depth, cy + along)
    if edge == "x_min":
        return (x_min + depth, cy + along)
    if edge == "y_max":
        return (cx + along, y_max - depth)
    return (cx + along, y_min + depth)  # y_min


def _layout_xy(seed, edge, bounds_xy):
    """Return {asset.name: (x, y)} with per-object 1 cm radius XY jitter."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (along, depth) in _LAYOUT_BASE.items():
        # Uniform sample in a disc of radius _JITTER_RADIUS_M.
        r = _JITTER_RADIUS_M * math.sqrt(float(rng.random()))
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        out[name] = _edge_to_world(
            edge, bounds_xy,
            along + r * math.cos(theta),
            depth + r * math.sin(theta),
        )
    return out


def _place_on_table(obj, xy, table_top_z):
    root_to_bottom = 0.0
    try:
        pos = obj.get_position_orientation()[0]
        root_to_bottom = max(0.0, float(pos[2]) - float(obj.aabb[0][2]))
    except Exception:
        pass
    obj.set_position_orientation(
        position=(float(xy[0]), float(xy[1]),
                  float(table_top_z + root_to_bottom + 0.004)),
        orientation=_quat_from_yaw(0.0),
    )
    if hasattr(obj, "keep_still"):
        obj.keep_still()


def _upright(obj, max_tilt_deg):
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


def _on_table(obj, bounds_xy, table_top_z):
    try:
        aabb_min, _ = obj.aabb
        x, y, _ = _pose(obj)
    except Exception:
        return False
    (x_min, y_min), (x_max, y_max) = bounds_xy
    return (x_min <= x <= x_max
            and y_min <= y <= y_max
            and abs(float(aabb_min[2]) - table_top_z) <= 0.035)


# ---------------------------------------------------------------------------
# Robot placement + gating
# ---------------------------------------------------------------------------

def _place_robot(args, robot, active_objects, bounds_xy, floor_z, edge):
    from omnigibson.utils.franka_edge_align import (
        EdgeAlignObject, EdgeAlignRequest, place_franka_edge_aligned,
    )
    edge_objects = (
        EdgeAlignObject(name=TARGET_MUG.name, role="target",
                        position_xy=tuple(_pose(active_objects[TARGET_MUG.name])[:2])),
        EdgeAlignObject(name=DEST_BOWL.name, role="dest",
                        position_xy=tuple(_pose(active_objects[DEST_BOWL.name])[:2])),
    )
    result = place_franka_edge_aligned(EdgeAlignRequest(
        table_aabb_xy=bounds_xy,
        pack_objects_world=edge_objects,
        role_weights={"target": 3.0, "dest": 2.0},
        robot_half_extent_xy=robot_half_extent_xy(robot),
        edge_gap_m=args.mount_gap_m,
        edge_margin_m=0.05,
        scan_offsets_m=(0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12),
        preferred_edge=edge,
    ))
    robot.set_position_orientation(
        position=(result.base_pose["position"][0],
                  result.base_pose["position"][1], float(floor_z)),
        orientation=result.base_pose["orientation"],
    )
    return result


def _arm_base_z(robot):
    link = robot.links.get(_FRANKA_ARM_BASE_LINK)
    if link is None:
        raise RuntimeError(f"Robot has no {_FRANKA_ARM_BASE_LINK}")
    return float(link.get_position_orientation()[0][2])


def _gate(robot, edge_result, active_objects, bounds_xy, table_top_z, floor_z):
    target = active_objects[TARGET_MUG.name]
    bowl = active_objects[DEST_BOWL.name]
    rx, ry, rz = _pose(robot)
    tx, ty, _ = _pose(target)

    # Each distractor must stay on the table and remain upright.
    distractor_checks = {}
    for d in DISTRACTORS:
        obj = active_objects[d.name]
        distractor_checks[f"{d.name}_on_table"] = _on_table(obj, bounds_xy, table_top_z)
        distractor_checks[f"{d.name}_upright"] = _upright(obj, 25.0)

    checks = {
        "robot_on_long_side": edge_result.edge_label in _long_side_edges(bounds_xy),
        "robot_at_floor": abs(rz - floor_z) <= 0.03,
        "robot_mount_collision_free": not getattr(edge_result, "collision_hits", ()),
        "target_reachable": 0.20 <= math.hypot(rx - tx, ry - ty) <= 1.10,
        "target_on_table": _on_table(target, bounds_xy, table_top_z),
        "bowl_on_table": _on_table(bowl, bounds_xy, table_top_z),
        "target_upright": _upright(target, 25.0),
        "bowl_upright": _upright(bowl, 25.0),
        "no_interpenetration": not check_interpenetration(active_objects, tol=0.002),
        **distractor_checks,
    }
    return all(checks.values()), {
        "checks": checks,
        "arm_base_z": _arm_base_z(robot),
        "target_distance_m": float(math.hypot(rx - tx, ry - ty)),
        "edge_label": edge_result.edge_label,
    }


# ---------------------------------------------------------------------------
# Sim runners
# ---------------------------------------------------------------------------

def _build_env_config(surface_spawn_xyz, object_cfgs):
    return {
        "scene": {"type": "Scene"},
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
        "objects": [
            _object_cfg(SUPPORT, surface_spawn_xyz, fixed_base=True),
            *object_cfgs,
        ],
        "task": {"type": "DummyTask"},
        "env": {"external_sensors": build_external_camera_configs(resolution=(720, 1080))},
    }


def _resolve_objects(env):
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


def _record_review_video(args, env, og, episode, robot, support_obj, active_objects):
    if not args.save_video:
        return []
    from omnigibson.task_generation.utils.video import (
        build_video_view_specs, close_video_writer, expected_video_path,
        init_video_writer, setup_cameras,
    )
    views = setup_cameras(env, build_video_view_specs(
        args, robot, active_objects[TARGET_MUG.name],
        support_obj=support_obj, active_objects_by_inst=active_objects,
    ))
    writers, paths = [], []
    for view in views:
        stem = args.save_video[:-4] if isinstance(args.save_video, str) and args.save_video.endswith(".mp4") else args.save_video
        base_path = f"{stem}_{view['label']}.mp4" if isinstance(stem, str) else None
        if base_path is None:
            continue
        paths.append(expected_video_path(base_path, episode))
        sensor = env.external_sensors.get(view["sensor_name"])
        frame_hw = (720, 1080)
        writer = init_video_writer(base_path, episode, args.video_fps, robot=None, frame_hw=frame_hw)
        if writer:
            writers.append((view["sensor_name"], writer))
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


def _park(objects_by_name, robot, og, floor_z=0.0):
    for obj in objects_by_name.values():
        obj.set_position_orientation(position=_PARK_POS)
        if hasattr(obj, "keep_still"):
            obj.keep_still()
    robot.set_position_orientation(position=(50.0, 50.0, float(floor_z)))
    og.sim.step()


def run_dry_run(args):
    spec = _write_activity_metadata(args)
    print("[Pipeline] Dry run complete.")
    print(f"  task_spec: {spec['task_spec_path']}")
    print(f"  BDDL:      {spec['bddl_path']}")
    print(f"  LTL:       {spec['ltl_safety_path']}")
    append_jsonl(args.debug_jsonl, {"event": "dry_run", "task_spec": spec})


def run_sim(args):
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True

    task_spec = _write_activity_metadata(args)
    region = _pick_surface(args.seed)
    surface_spawn, bounds_xy, table_top_z = _surface_geometry(region)
    floor_z = 0.0

    object_cfgs = [_object_cfg(a, position=(100.0 + i, 100.0, -100.0))
                   for i, a in enumerate(TASK_OBJECTS)]
    cfg = _build_env_config(surface_spawn, object_cfgs)

    print(f"[Pipeline] Surface: {SUPPORT.category}/{SUPPORT.model} "
          f"region={region['region_id']} top_z={table_top_z:.3f}")
    print(f"[Pipeline] Prompt: {args.prompt!r}")
    sys.stdout.flush()

    env = og.Environment(configs=cfg)
    try:
        force_sensor_resolution(env, height=720, width=1080)
        env.reset()
        robot = env.robots[0]
        robot.set_position_orientation(position=(50.0, 50.0, floor_z))
        og.sim.step()
        support_obj, active_objects = _resolve_objects(env)
        settle_fn = make_settle_fn(og, th)

        for ep in range(args.episodes):
            ep_seed = args.seed + ep * 1000
            print(f"\n[Pipeline] Episode {ep + 1}/{args.episodes}")

            edge = args.preferred_edge or _default_edge(bounds_xy)
            layout = _layout_xy(ep_seed, edge, bounds_xy)
            print(f"[Pipeline] Edge={edge}, jitter_radius={_JITTER_RADIUS_M*100:.1f}cm")

            for name, xy in layout.items():
                _place_on_table(active_objects[name], xy, table_top_z)
            og.sim.step()
            settle_fn(active_objects)
            for _ in range(max(0, int(args.settle_steps))):
                for obj in active_objects.values():
                    if hasattr(obj, "keep_still"):
                        obj.keep_still()
                og.sim.step()

            edge_result = _place_robot(args, robot, active_objects,
                                       bounds_xy, floor_z, edge)
            og.sim.step()
            gate_pass, gate = _gate(robot, edge_result, active_objects,
                                    bounds_xy, table_top_z, floor_z)
            print(f"[Pipeline] Robot: edge={edge_result.edge_label}, "
                  f"gap={edge_result.gap_actual:.3f}, "
                  f"target_dist={gate['target_distance_m']:.3f}")
            if not gate_pass:
                failed = [k for k, v in gate["checks"].items() if not v]
                print(f"[Pipeline] Gate FAIL: {failed}")
            if args.strict_gate and not gate_pass:
                raise RuntimeError("Strict gate failed.")

            scene_path = None
            if gate_pass or not args.strict_gate:
                scene_path = os.path.join(args.run_dir, f"scene_ep{ep + 1}.json")
                og.sim.save(json_paths=[scene_path])
                # Sidecar consumed by sentinel.teleop.so101_franka_teleop:
                # places panda_link0 19 mm below tabletop on snapshot rewrite.
                meta_path = scene_path[:-5] + ".meta.json"
                with open(meta_path, "w") as f:
                    json.dump({
                        "table_top_z": float(table_top_z),
                        "floor_z": float(floor_z),
                        "arm_base_target_z": float(table_top_z - 0.019),
                        "arm_base_link": _FRANKA_ARM_BASE_LINK,
                    }, f, indent=2)
                print(f"[Pipeline] Saved: {scene_path}")

            video_paths = _record_review_video(args, env, og, ep, robot,
                                               support_obj, active_objects)
            append_jsonl(args.debug_jsonl, {
                "episode": ep + 1,
                "activity_name": args.activity_name,
                "edge": edge,
                "layout_xy": {n: list(xy) for n, xy in layout.items()},
                "robot_pose": _pose(robot),
                "gate_pass": gate_pass,
                "gate": gate,
                "scene_path": scene_path,
                "video_paths": video_paths,
                "task_spec_path": task_spec["task_spec_path"],
            })
            _park(active_objects, robot, og, floor_z=floor_z)
    finally:
        env.close()
    print("[Pipeline] Shutdown.")
    pipeline_exit()


def main():
    args = parse_args()
    init_run_dir(args, "mug_into_bowl_empty")
    if args.dry_run:
        run_dry_run(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main()
