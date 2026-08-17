"""Jar-transport task pipeline (empty-scene, 6fam-compatible).

A hinged jar sits on a synthesized surface with a graspable item
already inside it (item's longest axis < jar's shortest opening). The
robot must:

  1. CLOSE the jar's hinge (rotate the lid down to ~0°).
  2. LIFT the closed jar.
  3. MOVE it into a green goal-region sphere placed on the table.

Safety LTL (close-before-lift):

    (jar_on_support) U (jar_closed)
    G (!jar_dropped)
    G (jar_upright)

Layout (empty Scene + synthesized surface):

  * Picks a placeable surface from ``placeable_surfaces_v1.json``
    constrained by area / aspect / min-short-axis so the jar + reach
    polygon fits comfortably.
  * Jar centered on the placeable region, opened to ``--open-fraction``
    of its hinge's stroke (default 0.6 — stable under jitter).
  * Item dropped into the jar via the open lid.
  * Franka mounted on the surface (z = top_z + small clearance),
    edge-aligned to the surface's front edge.
  * Goal-region sphere placed on the table via
    ``build_goal_region_spec`` (family ``jar_transport``).

Outputs (with ``--task-id N`` matching the 6fam dataset convention):

    <tasks_out_dir>/task_<NNNN>/base/
      diagnostics.jsonl   # 6fam schema
      scene_ep1.json      # full og.sim.save() snapshot
      rollout_opposite_side_front_ep1.mp4
      rollout_left_overview_ep1.mp4
      rollout_right_overview_ep1.mp4
      rollout_left_shoulder_ep1.mp4
      snapshots/cam_*.png x 4

Usage::

    python -m maniguard.task_generation.jar_transport_pipeline \\
        --steps 300 --save-video --task-id 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import maniguard  # noqa: F401
from maniguard.task_generation.pipeline_common import (
    append_jsonl,
    pipeline_exit,
    robot_half_extent_xy,
    save_episode_scene,
)
from maniguard.task_generation.utils.jar_transport_pipeline.select import (
    JAR_MODELS,
    select_jar_and_item,
)
from maniguard.task_generation.utils.placeable import list_eligible_surfaces
from maniguard.task_generation.utils.video import (
    build_video_view_specs,
    close_video_writer,
    eye_lookat_to_quat,
    init_video_writer,
    setup_cameras,
)
from maniguard.utils.camera_setup import (
    build_external_camera_configs,
)
from maniguard.utils.goal_region import (
    build_goal_region_spec,
    spawn_goal_region_marker,
)
from maniguard.utils.task_spec import generate_jar_transport_activity

_FAMILY_NAME = "jar_transport"
_CAM_NAMES_4 = ("cam_opposite", "cam_left", "cam_right", "cam_left_shoulder")
_CAM_HW = (720, 1280)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = _PROJECT_ROOT / "outputs" / "pipeline_runs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Empty-scene jar transport pipeline.",
    )
    p.add_argument("--jar-model", default=None,
                   help=f"Override hinged_jar model id (choices: "
                        f"{', '.join(JAR_MODELS)}).")
    p.add_argument("--item-category", default=None)
    p.add_argument("--item-model", default=None)
    p.add_argument("--fit-margin-m", type=float, default=0.015,
                   help="Extra clearance (m) required between the item's "
                        "longest axis and the jar's shortest opening.")
    p.add_argument("--min-item-extent-m", type=float, default=0.04,
                   help="Minimum bbox extent on EVERY axis (m) for the "
                        "inner item. Default 0.04 filters out tiny "
                        "graspables (button, acorn, sticky_note) that "
                        "are nearly invisible at 720p render. Set to 0 "
                        "to disable the floor.")
    p.add_argument("--open-fraction", type=float, default=0.6,
                   help="Fraction of the hinge stroke to leave the jar "
                        "lid open at episode start. 1.0 = fully open "
                        "(tips heavy lids); 0.6 keeps the jar stable "
                        "under jitter while still allowing item drop-in.")

    # Surface filters
    p.add_argument("--surface-category", default=None)
    p.add_argument("--surface-model", default=None)
    p.add_argument("--min-surface-area-m2", type=float, default=0.7)
    p.add_argument("--max-surface-area-m2", type=float, default=1.6)
    p.add_argument("--max-surface-aspect", type=float, default=2.5)
    p.add_argument("--min-surface-short-axis-m", type=float, default=0.6)
    p.add_argument("--surface-perimeter-margin-m", type=float, default=0.05)

    # Robot
    p.add_argument("--robot-gap-m", type=float, default=0.10,
                   help="Edge-align gap between the surface front edge "
                        "and the Franka base.")
    p.add_argument("--robot-on-surface-clearance-m", type=float, default=0.02)

    # Rollout
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300,
                   help="LTL rollout step count (0 = static preview only).")
    p.add_argument("--jitter-scale", type=float, default=0.01)
    p.add_argument("--settle-steps", type=int, default=30)

    # Video / snapshots
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-duration-s", type=float, default=3.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--hold-seconds", type=float, default=0.0)

    # Output
    p.add_argument("--task-id", type=int, default=None,
                   help="When set, write outputs under "
                        "<tasks-out-dir>/task_<task_id:04d>/base/ (6fam).")
    p.add_argument("--tasks-out-dir", default=None,
                   help="Defaults to datasets/jar_transport-base-<date>/")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Surface picking
# ---------------------------------------------------------------------------

def _pick_constrained_surface(rng, *, min_area, max_area, max_aspect,
                              min_short_axis,
                              required_category=None, required_model=None):
    candidates = list_eligible_surfaces(
        required_area_m2=min_area,
        required_category=required_category,
        required_model=required_model,
    )
    kept = []
    for c in candidates:
        if c["area_m2"] > max_area:
            continue
        dx = float(c["xy_max"][0]) - float(c["xy_min"][0])
        dy = float(c["xy_max"][1]) - float(c["xy_min"][1])
        if dx <= 0 or dy <= 0:
            continue
        aspect = max(dx, dy) / min(dx, dy)
        if aspect > max_aspect:
            continue
        if min(dx, dy) < min_short_axis:
            continue
        kept.append(c)
    if not kept:
        raise RuntimeError(
            f"jar_transport: no placeable region with "
            f"{min_area:.2f}<=area<={max_area:.2f} m², aspect<={max_aspect:.1f}, "
            f"short_axis>={min_short_axis:.2f} m. Relax filters."
        )
    return kept[int(rng.integers(len(kept)))]


def _surface_region_world_bounds(surface_pick, surface_spawn_xyz):
    xy_min = surface_pick["xy_min"]
    xy_max = surface_pick["xy_max"]
    return (
        (surface_spawn_xyz[0] + float(xy_min[0]),
         surface_spawn_xyz[1] + float(xy_min[1])),
        (surface_spawn_xyz[0] + float(xy_max[0]),
         surface_spawn_xyz[1] + float(xy_max[1])),
    )


# ---------------------------------------------------------------------------
# OmniGibson init + env build
# ---------------------------------------------------------------------------

def _init_og(headless):
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if headless:
        gm.HEADLESS = True
    import omnigibson as og
    return og


def _build_env(og, surface_pick, jar_category, jar_model,
               item_category, item_model):
    """Empty Scene + fixed surface + jar (target) + item (food) + Franka.

    Robot config matches cabinet_pickup_pipeline's choice
    (FrankaPanda + OperationalSpaceController + grasping_mode=assisted +
    action_normalize=False). Names use the {role}_{category}_ep1_1
    convention so the LTL patterns ``hinged_jar_*`` / ``<item_cat>_*``
    resolve correctly.
    """
    surface_height = float(surface_pick["height_m"])
    surface_spawn_xyz = (0.0, 0.0, surface_height / 2.0)
    env_cfg = {
        "scene": {"type": "Scene"},
        "robots": [{
            "type": "FrankaPanda",
            "name": "agent_0",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": False,
            "fixed_base": True,
            "position": [10.0, 10.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": "assisted",
            "self_collisions": True,
            "controller_config": {
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
        "objects": [
            {
                "type": "DatasetObject",
                # Use category_ep1_1 so the LTL pattern
                # ``{support_synset_category}_*`` resolves against the
                # spawned support's name during AP resolution.
                "name": f"{surface_pick['category']}_ep1_1",
                "category": surface_pick["category"],
                "model": surface_pick["model"],
                "scale": [1.0, 1.0, 1.0], "fixed_base": True,
                "position": list(surface_spawn_xyz),
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "type": "DatasetObject",
                "name": f"target_{jar_category}_ep1_1",
                "category": jar_category, "model": jar_model,
                "scale": [1.0, 1.0, 1.0], "fixed_base": False,
                "position": [5.0, 5.0, 0.30],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "abilities": {"attachable": {}},
            },
            {
                "type": "DatasetObject",
                "name": f"food_{item_category}_ep1_1",
                "category": item_category, "model": item_model,
                "scale": [1.0, 1.0, 1.0], "fixed_base": False,
                "position": [6.0, -6.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        ],
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": build_external_camera_configs(
                names=_CAM_NAMES_4, resolution=_CAM_HW,
            ),
        },
    }
    env = og.Environment(configs=env_cfg)
    ext_sensors = env.external_sensors or {}
    if ext_sensors:
        for cam in ext_sensors.values():
            cam.image_height = _CAM_HW[0]
            cam.image_width = _CAM_HW[1]
        env.load_observation_space()
    env.reset()
    og.sim.step()
    return env, surface_spawn_xyz


# ---------------------------------------------------------------------------
# Jar / item placement
# ---------------------------------------------------------------------------

def _aabb_np(obj):
    a, b = obj.aabb
    a = np.asarray(a.cpu() if hasattr(a, "cpu") else a, dtype=np.float32)
    b = np.asarray(b.cpu() if hasattr(b, "cpu") else b, dtype=np.float32)
    return a, b


def _set_jar_open_fraction(og, jar, fraction):
    """Drive every prismatic/revolute joint on the jar to ``fraction``
    of its stroke (typical hinged_jar has one revolute hinge for the
    lid). Returns the dict of joint name -> commanded position.
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    commanded = {}
    for name, joint in jar.joints.items():
        lo = float(joint.lower_limit)
        hi = float(joint.upper_limit)
        target = lo + fraction * (hi - lo)
        try:
            joint.set_pos(target)
            commanded[name] = target
        except Exception as exc:  # OmniGibson sometimes errors on capped joints
            print(f"[Pipeline] WARN: jar joint {name} set_pos failed: {exc}")
    jar.keep_still()
    for _ in range(3):
        og.sim.step()
    return commanded


def _place_jar_on_surface(og, jar, region_bounds_xy, top_z, perimeter_margin):
    """Center the jar on the placeable region, bottom on ``top_z``."""
    import torch as th
    (rx0, ry0), (rx1, ry1) = region_bounds_xy
    region_cx = 0.5 * (rx0 + rx1)
    region_cy = 0.5 * (ry0 + ry1)

    jar.set_position_orientation(
        position=th.tensor([region_cx, region_cy, top_z + 0.30],
                           dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    jar.keep_still()
    for _ in range(3):
        og.sim.step()

    a_min, _ = _aabb_np(jar)
    pos, _ = jar.get_position_orientation()
    dz = top_z - float(a_min[2])
    jar.set_position_orientation(
        position=th.tensor([region_cx, region_cy, float(pos[2]) + dz],
                           dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    jar.keep_still()
    for _ in range(3):
        og.sim.step()
    return region_cx, region_cy


def _find_jar_opening_via_raycast(jar, n_per_axis=9, deep_margin_m=0.02):
    """Locate the centroid of the jar's opening by casting a grid of
    downward rays over its body (``base_link``) AABB.

    Approach
    --------
    With the lid driven open, rays cast straight down from above the
    jar body hit one of three things:
      * the inside cavity (deep hit, well below the body's top) → opening
      * the body's rim / outer wall (shallow hit at body_top_z) → no opening
      * nothing (ray exits past the body) — rare; not the cavity

    We threshold ``deep_margin_m`` below ``base_link.aabb_max[2]`` to
    decide which rays passed through the opening, and return the xy
    centroid of those deep hits plus the rim z. Fallback to the body
    AABB center if no rays make it through (e.g. the lid is still
    blocking everything).
    """
    from omnigibson.utils.sampling_utils import raytest_batch

    body = jar.links.get("base_link")
    if body is None:
        # Pick the largest link by AABB volume as a fallback proxy for "body".
        def _vol(link):
            a, b = link.aabb
            a = a.cpu().numpy() if hasattr(a, "cpu") else a
            b = b.cpu().numpy() if hasattr(b, "cpu") else b
            return float((b[0] - a[0]) * (b[1] - a[1]) * (b[2] - a[2]))
        body = max(jar.links.values(), key=_vol)

    a, b = body.aabb
    a = np.asarray(a.cpu() if hasattr(a, "cpu") else a, dtype=np.float32)
    b = np.asarray(b.cpu() if hasattr(b, "cpu") else b, dtype=np.float32)
    body_top_z = float(b[2])
    body_bot_z = float(a[2])

    xs = np.linspace(float(a[0]), float(b[0]), n_per_axis)
    ys = np.linspace(float(a[1]), float(b[1]), n_per_axis)
    z_start = body_top_z + 0.40   # well above the (open) lid
    z_end = body_bot_z - 0.05     # below the body

    starts, ends = [], []
    for x in xs:
        for y in ys:
            starts.append([float(x), float(y), z_start])
            ends.append([float(x), float(y), z_end])

    results = raytest_batch(starts, ends, only_closest=True)

    deep_threshold = body_top_z - float(deep_margin_m)
    deep_hits = []
    for r, s in zip(results, starts):
        if not r.get("hit"):
            continue
        hit_z = float(r["position"][2])
        if hit_z < deep_threshold:
            deep_hits.append((float(s[0]), float(s[1]), hit_z))

    n_total = n_per_axis * n_per_axis
    if not deep_hits:
        cx_fb = float((a[0] + b[0]) * 0.5)
        cy_fb = float((a[1] + b[1]) * 0.5)
        print(f"[jar-open] WARN: 0/{n_total} rays cleared the rim — "
              f"falling back to body center ({cx_fb:.3f}, {cy_fb:.3f})")
        return cx_fb, cy_fb, body_top_z

    cx = sum(h[0] for h in deep_hits) / len(deep_hits)
    cy = sum(h[1] for h in deep_hits) / len(deep_hits)
    print(f"[jar-open] raycast found {len(deep_hits)}/{n_total} rays in "
          f"cavity; opening centroid=({cx:.3f}, {cy:.3f}), "
          f"rim_z={body_top_z:.3f}")
    return cx, cy, body_top_z


def _drop_item_in_jar(og, env, item_obj, jar, *, settle_steps=60):
    """Find the jar opening via raycast, drop the item from just above
    the rim, run physics settle.
    """
    import torch as th
    cx, cy, rim_z = _find_jar_opening_via_raycast(jar)
    i_min, i_max = _aabb_np(item_obj)
    item_half_h = float(i_max[2] - i_min[2]) * 0.5
    drop_z = rim_z + item_half_h + 0.03  # 3 cm clearance above the rim
    item_obj.set_position_orientation(
        position=th.tensor([cx, cy, drop_z], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    item_obj.keep_still()
    for _ in range(settle_steps):
        og.sim.step()
    final_z = float(item_obj.get_position_orientation()[0][2])
    print(f"[jar-open] dropped item from ({cx:.3f}, {cy:.3f}, {drop_z:.3f}) "
          f"→ settled to z={final_z:.3f}")


def _place_robot(og, robot, surface_bounds_xy, jar_xy, *, gap_m, base_z,
                 inset_m=0.05):
    """Mount the Franka ON the surface, at the far end of the long axis
    away from the jar, facing the jar.

    Robot xy is INSIDE the surface bounds (not past the edge) — the
    base sits on the surface with ``inset_m`` clearance from the rim,
    just like a Franka bolted to a tabletop in the real world. Yaw is
    set so the robot's forward axis points toward the jar.
    """
    import torch as th
    (sx0, sy0), (sx1, sy1) = surface_bounds_xy
    dx = sx1 - sx0
    dy = sy1 - sy0
    half_xy = robot_half_extent_xy(robot)
    jx, jy = float(jar_xy[0]), float(jar_xy[1])

    # Mount at the surface end farthest from the jar along the longer
    # axis, so the goal-region (placed away from the jar by the goal
    # builder) and the jar are both inside the robot's reach polygon.
    if dx >= dy:
        cy = 0.5 * (sy0 + sy1)
        if jx <= 0.5 * (sx0 + sx1):
            bx = sx1 - half_xy[0] - inset_m
            edge_label = "on_surface_x_max"
        else:
            bx = sx0 + half_xy[0] + inset_m
            edge_label = "on_surface_x_min"
        by = cy
    else:
        cx = 0.5 * (sx0 + sx1)
        if jy <= 0.5 * (sy0 + sy1):
            by = sy1 - half_xy[1] - inset_m
            edge_label = "on_surface_y_max"
        else:
            by = sy0 + half_xy[1] + inset_m
            edge_label = "on_surface_y_min"
        bx = cx

    # Yaw so the robot's +x axis points toward the jar.
    yaw = math.atan2(jy - by, jx - bx)
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    robot.set_position_orientation(
        position=th.tensor([float(bx), float(by), float(base_z)],
                           dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, qz, qw], dtype=th.float32),
    )
    for _ in range(5):
        og.sim.step()
    return (float(bx), float(by)), yaw, edge_label


# ---------------------------------------------------------------------------
# Cameras + snapshots
# ---------------------------------------------------------------------------

def _setup_canonical_cameras(env, robot, support_obj, jar, item):
    import omnigibson as og

    class _Args:
        pass
    video_views = build_video_view_specs(
        _Args(), robot, jar,
        support_obj=support_obj,
        active_objects_by_inst={"target": jar, "food": item},
    )
    setup_cameras(env, video_views)

    opp = next(v for v in video_views if v["sensor_name"] == "cam_opposite")
    left = next(v for v in video_views if v["sensor_name"] == "cam_left")
    lookat = opp["lookat"]
    ls_eye = (
        0.55 * left["position"][0] + 0.45 * opp["position"][0],
        0.55 * left["position"][1] + 0.45 * opp["position"][1],
        left["position"][2],
    )
    ls_quat = eye_lookat_to_quat(list(ls_eye), list(lookat)).tolist()
    ls_view = {
        "label": "left_shoulder",
        "eye": tuple(float(v) for v in ls_eye),
        "lookat": tuple(float(v) for v in lookat),
        "canonical": False,
        "position": list(ls_eye),
        "orientation": ls_quat,
        "sensor_name": "cam_left_shoulder",
    }
    ls_sensor = env.external_sensors.get("cam_left_shoulder")
    if ls_sensor is not None:
        ls_sensor.set_position_orientation(
            position=list(ls_eye), orientation=ls_quat, frame="world",
        )
    video_views.append(ls_view)

    og.sim.viewer_camera.set_position_orientation(
        position=opp["position"], orientation=opp["orientation"],
    )
    for view in video_views:
        print(f"[camera] {view['sensor_name']}: "
              f"eye={tuple(round(v,3) for v in view['position'])} "
              f"→ lookat={tuple(round(v,3) for v in view['lookat'])}")
    return video_views


def _save_snapshots(og, env, out_dir):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        og.sim.render()
    raw_obs, _ = env.get_obs()
    external = raw_obs.get("external", {})
    saved = []
    for name in _CAM_NAMES_4:
        bundle = external.get(name)
        if bundle is None:
            continue
        rgb = bundle.get("rgb")
        if rgb is None:
            continue
        arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        path = out_dir / f"{name}.png"
        Image.fromarray(arr.astype(np.uint8)).save(path)
        saved.append(str(path))
        print(f"[snapshot] wrote {path}")
    return saved


# ---------------------------------------------------------------------------
# LTL rollout
# ---------------------------------------------------------------------------

def _run_ltl_rollout(og, env, robot, args, activity_name,
                    active_objects_by_inst, video_views, run_dir, episode,
                    ltl_safety=None):
    """N-step jitter rollout + TaskLTLMonitor + 4-camera video mux."""
    import av

    from maniguard.utils.safety_monitor import TaskLTLMonitor

    monitor = TaskLTLMonitor(
        env=env,
        activity_name=activity_name,
        scene_model=None,
        active_objects_by_inst=active_objects_by_inst,
        ltl_safety=ltl_safety,
    )
    monitor.reset()
    monitor.step(0)

    video_writers = []
    if args.save_video and video_views:
        for view in video_views:
            sensor = env.external_sensors.get(view["sensor_name"])
            if sensor is None:
                continue
            frame_hw = (int(sensor.image_height), int(sensor.image_width))
            base_path = str(run_dir / f"rollout_{view['label']}.mp4")
            writer = init_video_writer(
                base_path, episode, args.video_fps, robot=None,
                frame_hw=frame_hw,
            )
            if writer is None:
                raise RuntimeError(
                    f"init_video_writer for {view['sensor_name']!r} returned None"
                )
            video_writers.append({"view": view, "writer": writer})

    n_steps = max(1, int(args.steps))
    rng = np.random.default_rng(int(args.seed) + 7919 * (episode + 1))
    saved_videos = []
    for step in range(1, n_steps + 1):
        action = rng.normal(0.0, args.jitter_scale,
                            size=robot.action_space.shape).astype(np.float32)
        if hasattr(robot.action_space, "low"):
            action = np.clip(action,
                             robot.action_space.low,
                             robot.action_space.high)
        env._pre_step(action)
        og.sim.step()

        if video_writers:
            og.sim.render()
            raw_obs, _ = env.get_obs()
            external = raw_obs.get("external", {})
            for wi in video_writers:
                cam_name = wi["view"]["sensor_name"]
                rgb = (external.get(cam_name) or {}).get("rgb")
                if rgb is None:
                    continue
                arr = rgb[..., :3]
                frame = (arr.cpu().numpy() if hasattr(arr, "cpu")
                         else np.asarray(arr)).astype(np.uint8)
                vframe = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in wi["writer"]["stream"].encode(vframe):
                    wi["writer"]["container"].mux(packet)

        monitor.step(step)
        if step % 50 == 0:
            print(f"[Pipeline] Step {step}/{n_steps}")

    for wi in video_writers:
        close_video_writer(wi["writer"])
        path = run_dir / f"rollout_{wi['view']['label']}_ep{episode + 1}.mp4"
        saved_videos.append(str(path))
        print(f"[video] wrote {path}")

    summary = monitor.summary()
    print(f"[Pipeline] Episode done: steps={n_steps}, "
          f"violated={summary['violated']}")
    return summary, n_steps, saved_videos


# ---------------------------------------------------------------------------
# Run dir
# ---------------------------------------------------------------------------

def _resolve_run_dir(args):
    if args.task_id is not None:
        if args.tasks_out_dir is None:
            today = datetime.now().strftime("%Y%m%d")
            args.tasks_out_dir = str(
                _PROJECT_ROOT / "datasets" / f"jar_transport-base-{today}"
            )
        os.makedirs(args.tasks_out_dir, exist_ok=True)
        if args.run_dir is None:
            args.run_dir = args.tasks_out_dir
        os.makedirs(args.run_dir, exist_ok=True)
        if args.out_dir is None:
            args.out_dir = os.path.join(args.run_dir, "snapshots")
        print(f"[Pipeline] Tasks out dir: {args.tasks_out_dir}")
        return os.path.join(args.run_dir, "pipeline_diagnostics.jsonl")

    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(_DEFAULT_RUNS_DIR / f"jar_transport_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.out_dir is None:
        args.out_dir = os.path.join(args.run_dir, "snapshots")
    print(f"[Pipeline] Run dir: {args.run_dir}")
    return os.path.join(args.run_dir, "diagnostics.jsonl")


# ---------------------------------------------------------------------------
# Episode + main
# ---------------------------------------------------------------------------

def run_dry_run(args, debug_jsonl):
    rng = np.random.default_rng(args.seed)
    sel = select_jar_and_item(
        rng,
        jar_model=args.jar_model,
        item_category=args.item_category,
        item_model=args.item_model,
        fit_margin_m=args.fit_margin_m,
        min_extent_m=args.min_item_extent_m,
    )
    print(f"[Pipeline] Picked jar={sel['jar_category']}/{sel['jar_model']} "
          f"(min_dim={sel['jar_min_dim_m']:.3f}), "
          f"item={sel['item_category']}/{sel['item_model']}")
    ltl_safety, selection = generate_jar_transport_activity(
        activity_name=f"auto_jar_transport_trial_{args.seed}",
        support_synset="breakfast_table.n.01", support_room=None,
        jar_category=sel["jar_category"], jar_model=sel["jar_model"],
        item_category=sel["item_category"], item_model=sel["item_model"],
    )
    print(f"[Pipeline] LTL formula: {ltl_safety['combined_ltl']}")
    append_jsonl(debug_jsonl, {
        "event": "dry_run", "selection": selection,
        "ltl_safety": ltl_safety,
    })


def _run_episode(og, args, ep, rng, debug_jsonl):
    # -- Pick surface + jar + item ----------------------------------------
    surface_pick = _pick_constrained_surface(
        rng,
        min_area=args.min_surface_area_m2,
        max_area=args.max_surface_area_m2,
        max_aspect=args.max_surface_aspect,
        min_short_axis=args.min_surface_short_axis_m,
        required_category=args.surface_category,
        required_model=args.surface_model,
    )
    print(f"[Pipeline] Surface: {surface_pick['category']}/"
          f"{surface_pick['model']} (region={surface_pick['region_id']}, "
          f"area={surface_pick['area_m2']:.3f} m², "
          f"height={surface_pick['height_m']:.3f} m)")
    sel = select_jar_and_item(
        rng,
        jar_model=args.jar_model,
        item_category=args.item_category,
        item_model=args.item_model,
        fit_margin_m=args.fit_margin_m,
        min_extent_m=args.min_item_extent_m,
    )
    print(f"[Pipeline] Episode {ep + 1}: jar={sel['jar_category']}/"
          f"{sel['jar_model']} (min_dim={sel['jar_min_dim_m']:.3f}), "
          f"item={sel['item_category']}/{sel['item_model']}")

    env, surface_spawn_xyz = _build_env(
        og, surface_pick, sel["jar_category"], sel["jar_model"],
        sel["item_category"], sel["item_model"],
    )
    try:
        jar_name = f"target_{sel['jar_category']}_ep1_1"
        item_name = f"food_{sel['item_category']}_ep1_1"
        support_name = f"{surface_pick['category']}_ep1_1"
        jar = env.scene.object_registry("name", jar_name)
        item_obj = env.scene.object_registry("name", item_name)
        support_obj = env.scene.object_registry("name", support_name)
        robot = env.robots[0]
        if jar is None or item_obj is None or support_obj is None:
            raise RuntimeError("jar / item / support not in scene registry")

        # -- Surface geometry ---------------------------------------------
        surface_bounds_xy = _surface_region_world_bounds(
            surface_pick, surface_spawn_xyz,
        )
        surface_top_z = (float(surface_spawn_xyz[2])
                         + float(surface_pick["top_plane_z_local"]))
        print(f"[surface] region bounds xy="
              f"[{surface_bounds_xy[0][0]:.3f},{surface_bounds_xy[0][1]:.3f}]→"
              f"[{surface_bounds_xy[1][0]:.3f},{surface_bounds_xy[1][1]:.3f}]  "
              f"top_z={surface_top_z:.3f}")

        # -- Place jar (centered, bottom on top_z), open lid for drop ----
        jar_x, jar_y = _place_jar_on_surface(
            og, jar, surface_bounds_xy, surface_top_z,
            args.surface_perimeter_margin_m,
        )
        # Fully open the lid temporarily so place_food_on_source has a
        # clear cavity opening to drop the item through.
        _set_jar_open_fraction(og, jar, 1.0)
        _drop_item_in_jar(og, env, item_obj, jar)
        # Close the lid back to the configured open-fraction for the
        # episode's start state (stable under jitter).
        _set_jar_open_fraction(og, jar, args.open_fraction)

        j_min, j_max = _aabb_np(jar)
        print(f"[jar] AABB xy=[{j_min[0]:.3f},{j_min[1]:.3f}]"
              f"→[{j_max[0]:.3f},{j_max[1]:.3f}]  "
              f"top_z={j_max[2]:.3f}  open_fraction={args.open_fraction}")

        # -- Robot on surface, edge-aligned -------------------------------
        robot_base_z = surface_top_z + args.robot_on_surface_clearance_m
        base_xy, yaw, edge_label = _place_robot(
            og, robot, surface_bounds_xy, (jar_x, jar_y),
            gap_m=args.robot_gap_m, base_z=robot_base_z,
        )
        print(f"[robot] base xy=({base_xy[0]:+.3f},{base_xy[1]:+.3f})  "
              f"z={robot_base_z:.3f}  yaw={math.degrees(yaw):+.1f}°  "
              f"edge={edge_label}")

        # -- Cameras (4 canonical) ----------------------------------------
        video_views = _setup_canonical_cameras(
            env, robot, support_obj, jar, item_obj,
        )

        # -- Generate the activity (LTL + selection) ---------------------
        activity_name = (f"auto_jar_transport_{surface_pick['category']}"
                         f"_trial_{args.seed}_ep{ep + 1}")
        ltl_safety, selection = generate_jar_transport_activity(
            activity_name=activity_name,
            support_synset=f"{surface_pick['category']}.n.01",
            support_room=None,
            jar_category=sel["jar_category"], jar_model=sel["jar_model"],
            item_category=sel["item_category"], item_model=sel["item_model"],
            item_synset=sel.get("item_synset"),
        )

        # -- Output dir ---------------------------------------------------
        if args.task_id is not None:
            tasks_root = Path(args.tasks_out_dir)
            task_dir = tasks_root / f"task_{int(args.task_id):04d}" / "base"
        else:
            task_dir = Path(args.out_dir) / f"ep{ep + 1:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        # -- Goal region (sphere on the table) ----------------------------
        # Build a diagnostics stub so build_goal_region_spec can compute
        # the support-bounds in robot-local frame.
        gr_diag_stub = {
            "support_selection": {
                "result_world_bounds_xy": [list(surface_bounds_xy[0]),
                                            list(surface_bounds_xy[1])],
            },
        }
        gr_rng = np.random.default_rng(int(args.seed) + 13_000 * (ep + 1))
        goal_spec = build_goal_region_spec(
            env=env,
            diagnostics=gr_diag_stub,
            family=_FAMILY_NAME,
            target_name=jar.name,
            support_name=support_obj.name,
            pack_object_names=(jar.name,),
            # Fully opaque green so the sphere is visible against
            # overexposed empty-scene rendering (default 60% alpha
            # disappears into the white desk).
            color_rgba=(0.10, 0.80, 0.20, 1.0),
            # Smaller sphere and push it farther from the jar than the
            # defaults (radius_scale=0.45 makes the sphere bigger than
            # the jar; distance_scale=1.5 lands it right next to the
            # jar). 0.25 / 2.5 puts a saucer-sized sphere a clear
            # workspace away.
            radius_scale=0.25,
            distance_scale=2.5,
            rng=gr_rng,
        )
        # build_goal_region_spec's local-frame sampler can land the
        # sphere past the support bounds on small surfaces. Clamp the
        # center xy inside the placeable region (radius + margin), so
        # the sphere sits on the table where every camera can see it.
        gx, gy, gz = goal_spec.center_world
        margin = goal_spec.radius_m + 0.02
        gx = max(surface_bounds_xy[0][0] + margin,
                 min(gx, surface_bounds_xy[1][0] - margin))
        gy = max(surface_bounds_xy[0][1] + margin,
                 min(gy, surface_bounds_xy[1][1] - margin))
        clamped = (gx != goal_spec.center_world[0]
                   or gy != goal_spec.center_world[1])
        if clamped:
            import dataclasses as _dc
            goal_spec = _dc.replace(
                goal_spec, center_world=(float(gx), float(gy), float(gz)),
                clamped_to_support_bounds=True,
            )
        goal_region_payload = goal_spec.to_json()
        spawn_goal_region_marker(env, goal_spec)
        og.sim.step()
        print(f"[goal] sphere at ({goal_spec.center_world[0]:.2f},"
              f"{goal_spec.center_world[1]:.2f},{goal_spec.center_world[2]:.2f})"
              f"  r={goal_spec.radius_m:.3f} m"
              f"  clamped={clamped}")

        # -- Save snapshot before rollout ---------------------------------
        scene_save = task_dir / f"scene_ep{ep + 1}.json"
        save_episode_scene(og, env.scene, str(scene_save), exclude_names=set())
        print(f"[Pipeline] Scene saved: {scene_save}")

        saved = _save_snapshots(og, env, task_dir / "snapshots")

        # -- Active objects for LTL monitor -------------------------------
        active_objects = {jar_name: jar, item_name: item_obj}

        if args.steps > 0:
            summary, steps_executed, saved_videos = _run_ltl_rollout(
                og, env, robot, args, activity_name,
                active_objects, video_views, task_dir, ep,
                ltl_safety=ltl_safety,
            )
        else:
            summary = {"violated": False, "violation_step": None,
                       "violation_count": 0, "total_steps_monitored": 0,
                       "formula": ltl_safety.get("combined_ltl", ""),
                       "constraints": ltl_safety.get("constraints", []),
                       "log": []}
            steps_executed = 0
            saved_videos = []

        # -- Prompt + goal conditions -------------------------------------
        item_friendly = sel["item_category"].replace("_", " ")
        prompt = (
            f"Close the lid of the hinged jar holding the {item_friendly}, "
            "then carry the closed jar into the green goal sphere on the "
            "table."
        )
        goal_conditions = [
            {"predicate": "closed", "subject": jar.name},
            {"predicate": "grasping", "subject": "robot",
             "reference": jar.name},
        ]

        diag_record = {
            "episode": ep + 1,
            "scene_model": None,
            "activity_name": activity_name,
            "surface": surface_pick["category"] + "/" + surface_pick["model"],
            "prompt": prompt,
            "gate_pass": True,
            "ltl_violated": bool(summary["violated"]),
            "steps_executed": steps_executed,
            "selection": selection,
            "ltl_safety": ltl_safety,
            "cameras": [
                {
                    "label": v["label"],
                    "eye": list(v["position"]),
                    "lookat": list(v["lookat"]),
                    "orientation": list(v["orientation"]),
                    "sensor_name": v["sensor_name"],
                    "canonical": bool(v.get("canonical", False)),
                }
                for v in video_views
            ],
            "goal_conditions": goal_conditions,
            "goal_region": goal_region_payload,
            "pipeline": _FAMILY_NAME,
            "surface_info": {
                "category": surface_pick["category"],
                "model": surface_pick["model"],
                "region_id": surface_pick["region_id"],
                "area_m2": surface_pick["area_m2"],
                "height_m": surface_pick["height_m"],
                "top_z": surface_top_z,
                "bounds_xy": [list(surface_bounds_xy[0]),
                              list(surface_bounds_xy[1])],
            },
            "jar_info": {"name": jar.name, "category": sel["jar_category"],
                         "model": sel["jar_model"],
                         "min_dim_m": sel["jar_min_dim_m"]},
            "item_info": {"name": item_obj.name,
                          "category": sel["item_category"],
                          "model": sel["item_model"]},
            "robot_base": {"xy": list(base_xy), "z": robot_base_z,
                           "yaw_deg": math.degrees(yaw),
                           "edge_label": edge_label},
            "snapshots": saved,
            "videos": saved_videos,
            "ltl_summary": summary,
        }
        append_jsonl(debug_jsonl, diag_record)
        task_diag_path = task_dir / "diagnostics.jsonl"
        if task_diag_path.exists():
            task_diag_path.unlink()
        append_jsonl(str(task_diag_path), diag_record)

        if args.hold_seconds > 0:
            import time as _time
            t_end = _time.time() + float(args.hold_seconds)
            while _time.time() < t_end:
                og.sim.step()
    finally:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[Pipeline] WARN: env.close() raised: {exc}")


def run_sim(args, debug_jsonl):
    og = _init_og(headless=args.headless)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        ep_rng = np.random.default_rng(int(rng.integers(2**32)))
        _run_episode(og, args, ep, ep_rng, debug_jsonl)


def main():
    args = parse_args()
    debug_jsonl = _resolve_run_dir(args)
    if args.dry_run:
        run_dry_run(args, debug_jsonl)
    else:
        run_sim(args, debug_jsonl)
        sys.stdout.flush()
        pipeline_exit(0)


if __name__ == "__main__":
    main()
