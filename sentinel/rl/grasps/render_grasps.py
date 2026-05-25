#!/usr/bin/env python3
"""Per-object Franka grasp evaluation + video/visualisation pipeline.

For each row of ``sentinel/task_generation/utils/franka_graspability.csv`` (or any
compatible CSV), spawns the target floating in front of a FrankaPanda,
queries the GraspGen ZMQ server for 6-DoF grasp candidates, and runs
each by descending confidence: cuRobo motion plan → close gripper
(hard-pinned target) → release pin → gravity hold. The first candidate
that survives the gravity hold writes the success MP4.

Per-object artifacts (in ``--output-dir``):

  - ``{cat}_{model}_grasps_{top,iso,front,side}.png`` (always): top-K
    GraspGen poses overlaid on the sampled point cloud.
  - ``{cat}_{model}.mp4`` (success only): viewer-camera replay of the
    holding attempt, including the cuRobo approach trajectory.
  - ``{cat}_{model}_pcd_{top,iso,front,side}.png`` (failure only):
    point-cloud scatter of the object that failed, for diagnosis.

GraspGen server must be reachable at ``localhost:5556`` (override via
``GRASPGEN_HOST`` / ``GRASPGEN_PORT`` env vars or ``--graspgen-host`` /
``--graspgen-port``). Resume-friendly: rows whose MP4 or pcd PNG
already exist on disk are skipped.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
    SENTINEL_SKIP_LONGFINGER=1 \\
        python -m sentinel.rl.grasps.render_grasps \\
            --limit 20 --output-dir outputs/grasp_datasets/survey/videos
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Render successful Franka grasps from the survey CSV into MP4s."
    )
    p.add_argument("--csv", type=Path,
                   default=Path("sentinel/task_generation/utils/franka_graspability.csv"),
                   help="(category, model, status, ...) CSV that drives "
                        "the per-object loop. Status filtering is "
                        "controlled by --exclude-statuses.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_datasets/survey/videos"))
    p.add_argument("--limit", type=int, default=20,
                   help="Cap on number of graspable rows to render in this run "
                        "(0 = no cap). The CSV order is preserved; resume by "
                        "skipping objects whose .mp4 already exists.")
    p.add_argument("--per-object-timeout", type=float, default=120.0,
                   help="Hard wall-clock budget per object (s). Phase A "
                        "stops iterating candidates when this trips and "
                        "returns whatever holds were collected so far.")
    # Sampling — choose among GraspGen ZMQ (learned) and the geometric
    # samplers in this package.
    p.add_argument("--sampler", type=str, default="obb",
                   choices=["obb", "graspgen"],
                   help="Which grasp-pose sampler to use. Default: obb "
                        "(geometric, OBB-based assisted-grasp sampler).")
    p.add_argument("--max-candidates", type=int, default=400,
                   help="Cap on sampler candidates considered per object.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graspgen-confidence-threshold", type=float, default=0.0,
                   help="GraspGen-only: drop grasps below this discriminator "
                        "score client-side. Server's top-K already enforces a "
                        "soft threshold, so 0.0 keeps the server's full top-K.")
    p.add_argument("--graspgen-num-grasps", type=int, default=200,
                   help="GraspGen-only: diffusion samples drawn per object.")
    p.add_argument("--graspgen-topk", type=int, default=100,
                   help="GraspGen-only: server returns top-K by discriminator score.")
    p.add_argument("--graspgen-host", type=str, default=None,
                   help="GraspGen-only: override GRASPGEN_HOST (default localhost).")
    p.add_argument("--graspgen-port", type=int, default=None,
                   help="GraspGen-only: override GRASPGEN_PORT (default 5556).")
    # Scene geometry. Without --with-surface, target floats in mid-air at
    # --object-xyz; with --with-surface, a support object is spawned once
    # and the target is placed `--object-lift-above-surface` m above its
    # top face at `--surface-xy`.
    p.add_argument("--object-xyz", type=float, nargs=3, default=[0.55, 0.0, 0.55])
    p.add_argument("--with-surface", action="store_true",
                   help="Spawn a support surface (e.g. breakfast_table) once "
                        "and place each target above its top face — gives the "
                        "grasp eval a realistic table-top context.")
    p.add_argument("--surface-category", type=str, default="breakfast_table",
                   help="DatasetObject category for the support surface.")
    p.add_argument("--surface-model", type=str, default=None,
                   help="Specific model id; None picks a random model from the "
                        "category.")
    p.add_argument("--surface-xy", type=float, nargs=2, default=[0.55, 0.0],
                   help="World xy for the support surface center.")
    p.add_argument("--object-lift-above-surface", type=float, default=0.05,
                   help="Clearance (m) between target's spawn pose and the "
                        "support's top face. Small positive value gives the "
                        "fingers room to slip underneath.")
    p.add_argument("--franka-xy", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--franka-z", type=float, default=0.72)
    p.add_argument("--max-reach", type=float, default=0.95)
    # Step counts.
    p.add_argument("--settle-open-steps", type=int, default=8)
    p.add_argument("--close-steps", type=int, default=20)
    p.add_argument("--gravity-hold-steps", type=int, default=30)
    p.add_argument("--release-steps", type=int, default=6)
    p.add_argument("--min-z-after-hold", type=float, default=None)
    # Video
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Write every Nth sim step into the video (1 = every step).")
    p.add_argument("--camera-eye", type=float, nargs=3, default=[1.0, -0.7, 1.1],
                   help="Viewer camera position in world (m).")
    p.add_argument("--camera-lookat", type=float, nargs=3, default=[0.55, 0.0, 0.55],
                   help="Viewer camera lookat point in world (m).")
    p.add_argument("--video-width", type=int, default=960)
    p.add_argument("--video-height", type=int, default=540)
    # Per-object diagnostic PNGs (always-on grasp viz; pcd viz on failure).
    p.add_argument("--viz-topk", type=int, default=100,
                   help="Top-K grasp candidates to draw in the per-object "
                        "PNGs (0 disables grasp PNGs).")
    p.add_argument("--viz-image-size", type=int, default=720)
    p.add_argument("--viz-approach-len", type=float, default=0.04,
                   help="Length (m) of approach-axis line per grasp in "
                        "the diagnostic PNGs.")
    p.add_argument("--no-pcd-on-fail", action="store_true",
                   help="Skip the point-cloud PNG dump when an object "
                        "fails (default: dump it for diagnostic purposes).")
    # Deprecated: .pt files now go into the per-object subfolder.
    p.add_argument("--save-grasp-dataset", type=Path, default=None,
                   help="(Deprecated, ignored) .pt files are now saved "
                        "into each object's subfolder automatically.")
    p.add_argument("--num-target-grasps", type=int, default=1,
                   help="Phase A stops once this many valid grasps have "
                        "been collected per object. 1 = fastest per object; "
                        "raise to 10+ to build a more varied RL reset "
                        "dataset (slower).")
    p.add_argument("--save-video", action="store_true",
                   help="Phase B: replay the first valid grasp's saved "
                        "approach trajectory with frame capture and write "
                        "the success MP4. Disabled by default (Phase A "
                        "is the producer; rendering the video is opt-in).")
    p.add_argument("--max-obj-to-eef-after-hold", type=float, default=0.15,
                   help="After the gravity hold settles, target distance "
                        "to eef must be at most this much (m) for a grasp "
                        "to count. Catches phantom AG fires.")
    # Optional overrides for ad-hoc tests (e.g. re-rendering a stood-up
    # alarm clock instead of a flat one).
    p.add_argument("--exclude-statuses", type=str,
                   default="too_large,no_grasp,no_candidates,timeout",
                   help="Comma-separated CSV statuses to skip. Default keeps "
                        "only ``status=graspable`` rows. Pass 'too_large' "
                        "to attempt every row that isn't oversize.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Optional 'category:model' list overriding the CSV. "
                        "Useful for ad-hoc tests on a few specific objects.")
    p.add_argument("--target-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="Spawn-orientation override (degrees, intrinsic ZYX) "
                        "applied to every target. '90 0 0' stands a clock on "
                        "its bottom edge.")
    return p.parse_args()


def _rpy_deg_to_quat_xyzw(rpy_deg):
    """Intrinsic ZYX (roll-pitch-yaw, in degrees) to xyzw quat."""
    roll, pitch, yaw = [np.deg2rad(a) for a in rpy_deg]
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


def _read_graspable(csv_path: Path, exclude_statuses=("too_large",)):
    """Yield ``(category, model)`` for each row whose status is not in
    ``exclude_statuses``.

    Default skips only ``too_large`` rows (objects that don't fit the
    workspace). Pass a wider tuple to restrict further (e.g.
    ``("too_large", "no_grasp", "no_candidates", "timeout")`` for the
    survey's ``status=graspable`` rows only).
    """
    excl = set(exclude_statuses)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") in excl:
                continue
            yield row["category"], row["model"]


def _run_sampler(args, mesh, rng):
    """Dispatch to the configured grasp-pose sampler.

    Returns ``(poses (N, 4, 4) float32, scores (N,) float32)`` in mesh-
    local frame. All samplers agree on the Franka pose convention and
    eef_link origin so the downstream cuRobo / physics-validation path
    is identical regardless of which sampler ran.
    """
    if args.sampler == "graspgen":
        from sentinel.rl.grasps.graspgen_sampler import sample_graspgen_grasps
        return sample_graspgen_grasps(
            mesh,
            num_grasps=args.graspgen_num_grasps,
            topk_num_grasps=args.graspgen_topk,
            confidence_threshold=args.graspgen_confidence_threshold,
            host=args.graspgen_host,
            port=args.graspgen_port,
            rng=rng,
        )
    if args.sampler == "obb":
        from sentinel.rl.grasps.obb_sampler import (
            OBBConfig, sample_obb_assisted_grasps,
        )
        return sample_obb_assisted_grasps(
            mesh,
            config=OBBConfig(max_candidates=args.max_candidates),
            rng=rng,
        )
    raise ValueError(f"Unknown sampler: {args.sampler!r}")


def _build_env_config(args, franka_base_z: float) -> dict:
    """Empty Scene + FrankaPanda only. Target is spawned floating per-object."""
    return dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[dict(
            type="FrankaPanda",
            name="agent_0",
            obs_modalities=["rgb"],
            action_type="continuous",
            action_normalize=True,
            grasping_mode="assisted",
            self_collisions=True,
            position=[args.franka_xy[0], args.franka_xy[1], franka_base_z],
            orientation=[0.0, 0.0, 0.0, 1.0],
            controller_config={
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        )],
        objects=[],
        task=dict(type="DummyTask"),
    )


def _eye_lookat_to_quat(eye, lookat):
    """Camera quat (xyzw) from eye + lookat in world frame.

    Identical to ``sentinel.task_generation.utils.video.eye_lookat_to_quat``
    but inlined to avoid pulling in the task-generation deps.
    """
    import math
    import torch as th
    import omnigibson.utils.transform_utils as T
    d = np.asarray(lookat, dtype=np.float32) - np.asarray(eye, dtype=np.float32)
    d = d / max(1e-6, np.linalg.norm(d))
    return T.euler2quat(th.tensor([
        math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))),
        0.0,
        float(np.arctan2(-d[0], d[1])),
    ], dtype=th.float32)).tolist()


def _capture_frame(viewer_cam, target_hw=None):
    """Pull RGB from viewer camera; downscale to ``target_hw`` if given."""
    obs = viewer_cam.get_obs()[0]
    rgb = obs.get("rgb")
    if rgb is None:
        return None
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    arr = arr.astype(np.uint8)
    if target_hw is not None and arr.shape[:2] != tuple(target_hw):
        # Light-touch resize via PIL — keeps the dependency surface small.
        from PIL import Image
        arr = np.asarray(Image.fromarray(arr).resize(
            (target_hw[1], target_hw[0]), Image.BILINEAR
        ))
    return arr


def _render_one(env, primitives, category, model, args, obj_dir,
                viewer_cam, deadline):
    """Two-phase per-object eval: search for valid grasps, then optionally
    record a video of the first one.

    Phase A (always): GraspGen → ``collect_valid_grasps`` (no frames).
    Phase B (only if ``args.save_video`` and Phase A produced ≥1 grasp):
        replay the saved cuRobo trajectory of the first held grasp with
        frame capture, then close + gravity-hold (also captured) and write
        the MP4 if it still holds.

    All per-object artifacts (PNGs, MP4, .pt) are written into ``obj_dir``.
    The caller is responsible for renaming the directory with a
    ``_success`` / ``_fail`` suffix after this function returns.

    Returns the number of held grasps (0 = failure path).
    """
    import imageio
    import omnigibson as og
    import torch as th
    from omnigibson.objects import DatasetObject

    from sentinel.rl.grasps.collector import (
        GraspCollectorConfig,
        _build_action,
        _phase1_step,
        _reset_controller_goals,
        collect_valid_grasps,
        save_grasp_dataset,
    )
    from sentinel.rl.grasps.mesh import mesh_from_og_object
    from sentinel.rl.grasps._viz_helpers import (
        render_grasp_views,
        render_point_cloud_views,
    )

    robot = env.robots[0]
    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    initial_joint_pos = robot.get_joint_positions().clone()
    zero_arm_cmd = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)
    open_gripper_q = robot.joint_upper_limits[gripper_control_idx].clone()
    target_hw = (args.video_height, args.video_width)
    stem = f"{category}_{model}"
    out_dir = obj_dir
    video_path = obj_dir / f"{stem}.mp4"

    # Spawn target.
    name = f"render_target_{category}_{model}"
    try:
        obj = DatasetObject(name=name, category=category, model=model)
        env.scene.add_object(obj)
    except Exception as e:  # noqa: BLE001
        try:
            existing = env.scene.object_registry("name", name)
            if existing is not None:
                env.scene.remove_object(existing)
        except Exception:  # noqa: BLE001
            pass
        print(f"  spawn failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return 0

    init_pos = th.tensor(args.object_xyz, dtype=th.float32)
    init_quat = th.tensor(_rpy_deg_to_quat_xyzw(args.target_rpy), dtype=th.float32)
    obj.set_position_orientation(position=init_pos, orientation=init_quat)
    obj.root_link.set_linear_velocity(th.zeros(3))
    obj.root_link.set_angular_velocity(th.zeros(3))
    obj.root_link.disable_gravity()

    # If a support surface exists, recompute init_pos so the target's BOTTOM
    # rests `object_lift_above_surface` above the table top — regardless of
    # the object's height. Without this, tall objects (e.g. alphabet_abacus
    # at 39 cm) spawn with their center 5 cm above the table → bottom 15 cm
    # below the surface → PhysX penetration → "swirling" as the pin fights
    # the contact resolver.
    if getattr(args, "_support_top_z", None) is not None:
        # Settle one substep so AABB reflects the chosen orientation.
        for _ in range(2):
            og.sim.step()
        aabb_min, aabb_max = obj.aabb
        center_to_bottom = float(init_pos[2]) - float(aabb_min[2])
        sx, sy = args._surface_xy
        new_z = (args._support_top_z
                 + float(args.object_lift_above_surface)
                 + center_to_bottom)
        init_pos = th.tensor([sx, sy, new_z], dtype=th.float32)
        obj.set_position_orientation(position=init_pos, orientation=init_quat)
        obj.root_link.set_linear_velocity(th.zeros(3))
        obj.root_link.set_angular_velocity(th.zeros(3))
        print(f"  target init: bottom on table  "
              f"(center_to_bottom={center_to_bottom:.3f}m  z={new_z:.3f})",
              flush=True)

    try:
        # Brief pinned settle so PhysX/USD initialise the new prim.
        for _ in range(8):
            if time.time() > deadline:
                return 0
            _phase1_step(env, robot, obj, init_pos, init_quat, hard_pin=True)

        try:
            mesh = mesh_from_og_object(obj, use_visual=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  mesh extraction failed: {exc}", flush=True)
            return 0

        # Grasp candidates (in target-local frame; all samplers agree on
        # the Franka pose convention and eef_link origin).
        rng = np.random.default_rng(args.seed)
        candidates, scores = _run_sampler(args, mesh, rng)
        if len(candidates) == 0:
            print(f"  sampler={args.sampler}: 0 candidates", flush=True)
            return 0
        if len(candidates) > args.max_candidates:
            candidates = candidates[:args.max_candidates]
            scores = scores[:args.max_candidates]
        print(f"  sampler={args.sampler}: {len(candidates)} grasps, "
              f"score range={scores[-1]:.3f}-{scores[0]:.3f}", flush=True)

        # Diagnostic: top-K grasp PNGs (always written).
        if args.viz_topk > 0:
            try:
                import trimesh as _tm
                viz_pts, _ = _tm.sample.sample_surface(mesh, 4000)
                viz_pts = np.asarray(viz_pts, dtype=np.float32)
                vn = min(args.viz_topk, len(candidates))
                render_grasp_views(
                    viz_pts,
                    eef_pos=candidates[:vn, :3, 3],
                    approach=candidates[:vn, :3, 2],
                    scores=scores[:vn],
                    out_dir=out_dir, stem=stem,
                    image_size=args.viz_image_size,
                    approach_len=args.viz_approach_len,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  grasp PNG dump failed: {exc}", flush=True)

        # Phase A: validate.
        # Include the target in cuRobo's obstacle world — assisted-grasp
        # pre-grasp pose is OPEN-gripper at the chord midpoint and must
        # not clip the target along the approach.
        primitives._motion_generator.update_obstacles(ignore_objects=[])
        cfg = GraspCollectorConfig(
            num_target_grasps=args.num_target_grasps,
            settle_open_steps=args.settle_open_steps,
            close_steps=args.close_steps,
            gravity_hold_steps=args.gravity_hold_steps,
            max_reach=args.max_reach,
            max_obj_to_eef_after_hold=args.max_obj_to_eef_after_hold,
        )

        n_held_so_far = [0]

        def _on_progress(ci, result):
            if result is not None:
                n_held_so_far[0] += 1
                print(f"    cand {ci}: HELD "
                      f"(obj_to_eef={result['obj_to_eef']:.3f}m, "
                      f"traj_len={len(result['approach_traj'])}) "
                      f"[{n_held_so_far[0]}/{cfg.num_target_grasps}]",
                      flush=True)

        held = collect_valid_grasps(
            env, robot, primitives, obj,
            init_pos, init_quat,
            candidates_local=candidates,
            cfg=cfg, deadline=deadline,
            on_progress=_on_progress,
            verbose=True,
        )

        if not held:
            print(f"    Phase A: 0 valid grasps after "
                  f"{len(candidates)} candidates", flush=True)
            # Pcd dump for diagnostics.
            if not args.no_pcd_on_fail:
                try:
                    import trimesh as _tm
                    pcd_pts, _ = _tm.sample.sample_surface(mesh, 8000)
                    pcd_pts = np.asarray(pcd_pts, dtype=np.float32)
                    render_point_cloud_views(
                        pcd_pts, out_dir=out_dir, stem=stem,
                        image_size=args.viz_image_size,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  pcd PNG dump failed: {exc}", flush=True)
            return 0

        print(f"    Phase A: {len(held)} valid grasps collected", flush=True)
        # Save .pt dataset into the per-object folder.
        try:
            pt_path = obj_dir / f"grasps_{stem}.pt"
            save_grasp_dataset(held, pt_path, target_name=stem)
            print(f"  wrote {pt_path.name} (N={len(held)})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  grasp dataset save failed: {exc}", flush=True)

        if not args.save_video:
            return len(held)

        # Phase B: replay the first held grasp's saved trajectory through
        # the same physics kernel Phase A used, capturing frames per step.
        from sentinel.rl.grasps.collector import run_grasp_attempt

        frames: list = []
        frame_state = {"i": 0}

        def _capture(_phase=None):
            frame_state["i"] += 1
            if frame_state["i"] % args.frame_stride == 0:
                fr = _capture_frame(viewer_cam, target_hw)
                if fr is not None:
                    frames.append(fr)

        # Reset to a clean state — the per-candidate iteration in Phase A
        # left the robot mid-grasp on whichever candidate ran last; if we
        # don't release/reset before the pre-attempt frames, a stale AG
        # bond can drag the target around when we re-pin it.
        robot.release_grasp_immediately(arm)
        robot.set_joint_positions(initial_joint_pos)
        obj.set_position_orientation(position=init_pos, orientation=init_quat)
        obj.root_link.disable_gravity()
        _reset_controller_goals(robot)
        for _ in range(4):
            if time.time() > deadline:
                return len(held)
            _phase1_step(env, robot, obj, init_pos, init_quat, hard_pin=True)

        # Pre-attempt: a few frames showing the target floating before the
        # arm starts to move.
        for _ in range(8):
            if time.time() > deadline:
                return len(held)
            _phase1_step(env, robot, obj, init_pos, init_quat, hard_pin=True)
            _capture()

        result = run_grasp_attempt(
            env, robot, obj, init_pos, init_quat,
            joint_traj=th.from_numpy(held[0]["approach_traj"]),
            cfg=cfg,
            open_gripper_q=open_gripper_q,
            zero_arm_cmd=zero_arm_cmd,
            arm_control_idx=arm_control_idx,
            gripper_control_idx=gripper_control_idx,
            initial_joint_pos=initial_joint_pos,
            deadline=deadline,
            frame_callback=_capture,
        )
        if result is None:
            # PhysX non-determinism: replay didn't hold this run. .pt
            # remains valid, but no MP4 this time.
            print("    Phase B: replay didn't hold; MP4 skipped.", flush=True)
            return len(held)

        writer = imageio.get_writer(
            str(video_path), fps=args.fps, codec="libx264",
            macro_block_size=1, quality=7,
        )
        try:
            for fr in frames:
                writer.append_data(fr)
        finally:
            writer.close()
        return len(held)
    finally:
        try:
            robot.release_grasp_immediately(arm)
        except Exception:  # noqa: BLE001
            pass
        try:
            release_action = _build_action(robot, zero_arm_cmd, gripper_cmd=+1.0)
            for _ in range(args.release_steps):
                robot.apply_action(release_action)
                og.sim.step()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.scene.remove_object(obj)
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.set_joint_positions(initial_joint_pos)
            _reset_controller_goals(robot)
            og.sim.step()
        except Exception:  # noqa: BLE001
            pass


def main():
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.targets:
        rows = []
        for s in args.targets:
            if ":" not in s:
                raise SystemExit(f"bad --targets {s!r}; expected category:model")
            c, m = s.split(":", 1)
            rows.append((c.strip(), m.strip()))
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} explicit targets "
              f"(--targets), ignoring CSV.", flush=True)
    else:
        excl = tuple(s.strip() for s in args.exclude_statuses.split(",") if s.strip())
        rows = list(_read_graspable(csv_path, exclude_statuses=excl))
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} rows in CSV "
              f"(excluded statuses: {excl}).",
              flush=True)
    if not rows:
        print("Nothing to render.", flush=True)
        return

    # Resume: skip rows whose per-object subfolder already exists
    # (suffixed with _success or _fail from a prior run).
    # Clean up un-suffixed dirs left by mid-object crashes so they retry.
    def _already_done(c, m):
        stem = f"{c}_{m}"
        incomplete = out_dir / stem
        if incomplete.is_dir():
            import shutil
            shutil.rmtree(incomplete, ignore_errors=True)
        return ((out_dir / f"{stem}_success").is_dir()
                or (out_dir / f"{stem}_fail").is_dir())

    pending = [(c, m) for (c, m) in rows if not _already_done(c, m)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(pending)} pending "
          f"({len(rows) - len(pending)} already rendered).", flush=True)
    if args.limit > 0:
        pending = pending[:args.limit]
        print(f"  capped at --limit {args.limit}", flush=True)
    if not pending:
        print("Nothing to render.", flush=True)
        return

    franka_base_z = float(args.franka_z)
    print(f"  franka_base_z={franka_base_z:.3f}  target xyz={args.object_xyz}",
          flush=True)

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    # Shorten OG's assisted-grasp commit window so AG fires after ~2 action
    # steps of stable contact (default is 10, ~333 ms) instead of waiting
    # for a perfectly stable squeeze. Same patch as the GELLO teleop batch
    # uses; helps the gravity-hold step retain phantom-but-real-enough
    # grasps that the strict default window flickers off. Must run before
    # og.Environment() constructs the robot (read at robot _post_load).
    from omnigibson.robots.manipulation_robot import m as _ag_macros
    _ag_macros.GRASP_WINDOW = 1.0 / 300.0
    _ag_macros.RELEASE_WINDOW = 1.0 / 300.0

    import omnigibson as og
    import torch as th

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG ...", flush=True)
    env = og.Environment(configs=_build_env_config(args, franka_base_z=franka_base_z))
    env.reset()

    # Optional support surface — spawned once and shared across all targets.
    # The surface and the per-attempt target are both in cuRobo's obstacle
    # world (see update_obstacles(ignore_objects=[]) in collector.py and in
    # the Phase A call below). Open-gripper approach must clear both.
    args._support_top_z = None
    args._surface_xy = None
    if args.with_surface:
        from omnigibson.objects import DatasetObject
        support_name = "render_support_surface"
        try:
            # fixed_base anchors the support to world via a fixed joint, so
            # the franka arm and target prevent it from being knocked around
            # during planning + close+hold attempts.
            support = DatasetObject(
                name=support_name,
                category=args.surface_category,
                model=args.surface_model,
                fixed_base=True,
            )
            env.scene.add_object(support)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"--with-surface: failed to spawn "
                f"{args.surface_category}/{args.surface_model}: {exc}"
            )
        sx, sy = args.surface_xy
        support.set_position_orientation(
            position=th.tensor([sx, sy, 0.0], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        # Settle so AABB reflects the resting pose.
        for _ in range(20):
            og.sim.step()
        aabb_min, aabb_max = support.aabb
        support_top_z = float(aabb_max[2])
        args._support_top_z = support_top_z
        args._surface_xy = (float(sx), float(sy))
        # Provisional spawn xyz (correct xy + sentinel z = support_top_z +
        # lift). Per-object code overrides z after AABB lookup so tall
        # objects don't penetrate the table.
        args.object_xyz = [
            float(sx), float(sy),
            support_top_z + float(args.object_lift_above_surface),
        ]
        # Re-mount the Franka on the support surface (table-mounted convention).
        # The default --franka-z=0.72 sits 64 cm above the table top, which
        # makes every grasp a long reach-down and routinely fails kinematics.
        # Set the base 2 mm above the table top (Franka is bolted to the
        # table). Must happen BEFORE cuRobo init below — cuRobo reads the
        # robot's current pose to build its kinematic + collision world.
        robot = env.robots[0]
        mounted_z = support_top_z + 0.002
        robot.set_position_orientation(
            position=th.tensor([args.franka_xy[0], args.franka_xy[1], mounted_z],
                               dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        for _ in range(5):
            og.sim.step()
        print(f"  support: {args.surface_category}/{getattr(support, 'model', '?')}  "
              f"top_z={support_top_z:.3f}  fixed_base=True  "
              f"target_xyz_seed={[round(v, 3) for v in args.object_xyz]}",
              flush=True)
        print(f"  franka re-mounted at z={mounted_z:.3f} (was {franka_base_z:.3f})",
              flush=True)

    # Set viewer camera to a fixed side-3/4 angle so all videos share framing.
    quat = _eye_lookat_to_quat(args.camera_eye, args.camera_lookat)
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(args.camera_eye, dtype=th.float32),
        orientation=th.tensor(quat, dtype=th.float32),
    )
    # Step once so the viewer settles on the new pose before we try to read
    # frames; on a fresh boot the first get_obs() can return a black frame.
    og.sim.step()
    viewer_cam = og.sim.viewer_camera

    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )
    print(f"[{time.strftime('%H:%M:%S')}] initializing cuRobo ...", flush=True)
    primitives = StarterSemanticActionPrimitives(env, env.robots[0],
                                                 enable_head_tracking=False)
    print(f"[{time.strftime('%H:%M:%S')}] cuRobo ready.", flush=True)

    n_done = n_ok = n_fail = 0
    try:
        for cat, mdl in pending:
            n_done += 1
            t0 = time.time()
            deadline = t0 + args.per_object_timeout
            stem = f"{cat}_{mdl}"
            obj_dir = out_dir / stem
            obj_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{time.strftime('%H:%M:%S')}] "
                  f"({n_done}/{len(pending)}) {cat}/{mdl}", flush=True)
            n_held = 0
            try:
                n_held = _render_one(
                    env, primitives, cat, mdl, args,
                    obj_dir, viewer_cam, deadline,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  ! {cat}/{mdl} failed: {exc}", flush=True)
                msg = str(exc)
                if ("NoneType" in msg and "view" in msg) or \
                   "articulation_view" in msg.lower():
                    print(f"[{time.strftime('%H:%M:%S')}] FATAL: OG "
                          f"articulation state corrupted, exiting so "
                          f"watchdog can restart.", flush=True)
                    sys.stdout.flush()
                    sys.exit(2)

            elapsed = time.time() - t0
            # Rename folder with success/fail suffix.
            suffix = "_success" if n_held > 0 else "_fail"
            final_dir = out_dir / f"{stem}{suffix}"
            try:
                if final_dir.exists():
                    import shutil
                    shutil.rmtree(final_dir)
                obj_dir.rename(final_dir)
            except Exception:  # noqa: BLE001
                pass
            if n_held > 0:
                n_ok += 1
                print(f"  -> {n_held} valid grasps  ({elapsed:.1f}s)",
                      flush=True)
            else:
                n_fail += 1
                print(f"  -> FAILED  ({elapsed:.1f}s)", flush=True)
    finally:
        print(f"\n[{time.strftime('%H:%M:%S')}] DONE. "
              f"ok={n_ok} fail={n_fail} dir={out_dir}", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
