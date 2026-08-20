"""Test self-stacking stability for graspable objects.

For each candidate (category, model) drawn from
``docs/graspability_classified.csv`` (status=graspable):

  1. Spawn N copies at the same XY in an empty scene, identity orientation.
  2. Settle physics for K steps (gravity-only).
  3. Shake test: apply a random horizontal velocity to every copy and
     settle again, to discount stacks that look stable but actually fall
     under any disturbance.
  4. Stability check: every copy's XY centre must lie within
     ``bbox_tol * native_xy_bbox`` of the column anchor — per-axis (x and
     y both checked independently, NOT by area).

Candidates are processed in batches that re-create the OG env each time
to bound memory + per-stage Kit time. Results write incrementally so a
crash mid-sweep doesn't lose the prior batches.

Usage:
    # Smoke test, 10 candidates with video
    python tools/test_stack_self_stability.py --n-objects 10 --copies 2 \\
        --save-video outputs/stack_self_smoke

    # Full sweep, 5 copies, headless, batched
    python tools/test_stack_self_stability.py --n-objects 0 --copies 5 \\
        --batch-size 100 --headless \\
        --output outputs/stack_self_full.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ManiGuard repo root is 4 levels up from this file:
#   <repo>/maniguard/task_generation/utils/stack_pipeline/test_stack_self_stability.py
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
CSV_PATH = ROOT / "docs" / "graspability_classified.csv"

FLOOR_Z = 0.0


def load_graspable_pool(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["status"] == "graspable":
                rows.append((r["category"], r["model"]))
    return rows


def _build_external_sensors(args):
    if not args.save_video:
        return []
    return [
        {
            "sensor_type": "VisionSensor",
            "name": name,
            "relative_prim_path": f"/{name}",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": args.video_resolution,
                "image_width": args.video_resolution,
            },
        }
        for name in ("cam_front", "cam_top", "cam_perspective")
    ]


def _open_video_writers(args, batch_idx):
    if not args.save_video:
        return []
    import av
    os.makedirs(args.save_video, exist_ok=True)
    writers = []
    for name in ("cam_front", "cam_top", "cam_perspective"):
        path = os.path.join(args.save_video, f"batch{batch_idx:03d}_{name}.mp4")
        container = av.open(path, mode="w")
        stream = container.add_stream("h264", rate=args.video_fps)
        stream.width = args.video_resolution
        stream.height = args.video_resolution
        stream.pix_fmt = "yuv420p"
        writers.append({"name": name, "container": container,
                        "stream": stream, "path": path})
    return writers


def _run_one_batch(batch_candidates, args, batch_idx):
    """Spawn → place → settle → shake → resettle → check for one batch.

    Creates a fresh OG env, returns a list of result dicts (one per
    candidate). The env is torn down before returning so memory doesn't
    accumulate across batches.
    """
    import omnigibson as og
    from omnigibson.objects import DatasetObject

    from maniguard.task_generation.pipeline_common import hide_walls_and_ceiling
    from maniguard.task_generation.utils.video import eye_lookat_to_quat

    # On batches >0 the prior scene + objects must be fully torn down.
    # og.clear() rebuilds a fresh simulator (and stage) so the next
    # og.Environment() lands in a clean slate. Without this, scene-prim
    # transforms accumulate perspective components and re-loading fails
    # with "Some matrices have perspective components".
    if batch_idx > 0:
        og.clear()

    cfg = {
        "scene": {"type": "Scene"},
        "robots": [],
        "task": {"type": "DummyTask"},
    }
    sensors_cfg = _build_external_sensors(args)
    if sensors_cfg:
        cfg["env"] = {"external_sensors": sensors_cfg}

    env = og.Environment(configs=cfg)

    if sensors_cfg:
        for cam in (env.external_sensors or {}).values():
            cam.image_height = args.video_resolution
            cam.image_width = args.video_resolution
        env.load_observation_space()
    og.sim.step()

    # Spawn upfront — no add/remove during play.
    columns = []
    for col_idx, (cat, model) in enumerate(batch_candidates):
        base_x = col_idx * args.col_spacing
        objs = []
        spawn_err = None
        for i in range(args.copies):
            name = f"col{col_idx}_{cat}_{model}_{i}"
            try:
                obj = DatasetObject(name=name, category=cat, model=model)
                env.scene.add_object(obj)
                objs.append(obj)
            except Exception as exc:
                spawn_err = str(exc)
                break
        columns.append({"cat": cat, "model": model, "base_x": base_x,
                        "objs": objs, "spawn_err": spawn_err})
    og.sim.step()

    for col in columns:
        objs = col["objs"]
        if not objs:
            continue
        nbb = objs[0].native_bbox
        scale = objs[0].scale
        h = max(0.01, float(nbb[2] * scale[2]))
        clearance = 0.005
        for i, obj in enumerate(objs):
            z = FLOOR_Z + 0.005 + (i + 0.5) * h + i * clearance
            obj.set_position_orientation(
                position=(col["base_x"], 0.0, z),
                orientation=(0, 0, 0, 1),
            )
            obj.keep_still()

    hide_walls_and_ceiling(env)

    # Camera positioning + writers (only for batches that record).
    video_writers = _open_video_writers(args, batch_idx)
    if video_writers:
        view_x = 0.5 * (len(columns) - 1) * args.col_spacing
        row_w = (len(columns) - 1) * args.col_spacing
        cam_views = [
            ("cam_front", (view_x, -1.2, 0.25), (view_x, 0.0, 0.15)),
            ("cam_top",   (view_x, 0.0, max(1.5, row_w * 0.6)),
                          (view_x, 0.0, 0.0)),
            ("cam_perspective", (view_x - 1.0, -1.0, 0.5),
                                (view_x, 0.0, 0.15)),
        ]
        for name, eye, lookat in cam_views:
            quat = eye_lookat_to_quat(eye, lookat).tolist()
            sensor = (env.external_sensors or {}).get(name)
            if sensor is not None:
                sensor.set_position_orientation(
                    position=eye, orientation=quat, frame="world",
                )
        front_eye, front_lookat = cam_views[0][1], cam_views[0][2]
        og.sim.viewer_camera.set_position_orientation(
            position=front_eye,
            orientation=eye_lookat_to_quat(front_eye, front_lookat).tolist(),
        )
        for _ in range(3):
            og.sim.step()
        og.sim.render()

    def capture_frame():
        if not video_writers:
            return
        og.sim.render()
        raw_obs, _ = env.get_obs()
        external = raw_obs.get("external", {})
        import av
        for w in video_writers:
            cam_obs = external.get(w["name"], {})
            rgb = cam_obs.get("rgb")
            if rgb is None:
                continue
            frame = rgb[..., :3].cpu().numpy().astype(np.uint8)
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in w["stream"].encode(video_frame):
                w["container"].mux(packet)

    # Phase 1: gravity settle.
    for _ in range(args.settle_steps):
        og.sim.step()
        capture_frame()

    # Phase 2: shake test (skip if shake_velocity == 0).
    if args.shake_velocity > 0.0:
        import torch as th
        rng_shake = np.random.default_rng(args.seed + 7919)
        for col in columns:
            for obj in col["objs"]:
                vx = float(rng_shake.uniform(-args.shake_velocity,
                                             args.shake_velocity))
                vy = float(rng_shake.uniform(-args.shake_velocity,
                                             args.shake_velocity))
                try:
                    obj.set_linear_velocity(th.tensor([vx, vy, 0.0]))
                except Exception:
                    # Fall back to tiny xy teleport jitter.
                    pos = obj.get_position_orientation()[0]
                    obj.set_position_orientation(
                        position=(float(pos[0]) + 0.005 * vx / max(0.01, args.shake_velocity),
                                  float(pos[1]) + 0.005 * vy / max(0.01, args.shake_velocity),
                                  float(pos[2])),
                        orientation=(0, 0, 0, 1),
                    )
        for _ in range(args.shake_settle_steps):
            og.sim.step()
            capture_frame()

    # Stability check on post-shake positions.
    results = []
    for col in columns:
        cat, model = col["cat"], col["model"]
        objs = col["objs"]
        if not objs:
            results.append({"category": cat, "model": model, "stable": False,
                            "reason": col.get("spawn_err") or "no_objects"})
            continue

        nbb = objs[0].native_bbox
        scale = objs[0].scale
        orig_dx = max(0.01, float(nbb[0] * scale[0]))
        orig_dy = max(0.01, float(nbb[1] * scale[1]))
        cx = col["base_x"]
        cy = 0.0
        tol = args.bbox_tol
        ax = (cx - 0.5 * orig_dx * tol, cx + 0.5 * orig_dx * tol)
        ay = (cy - 0.5 * orig_dy * tol, cy + 0.5 * orig_dy * tol)

        positions = []
        all_in = True
        for obj in objs:
            pos = obj.get_position_orientation()[0]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            positions.append([px, py, pz])
            if not (ax[0] <= px <= ax[1] and ay[0] <= py <= ay[1]):
                all_in = False

        results.append({
            "category": cat, "model": model, "stable": all_in,
            "native_xy_bbox": [round(orig_dx, 4), round(orig_dy, 4)],
            "anchor_xy": [round(cx, 4), round(cy, 4)],
            "positions": [[round(v, 4) for v in p] for p in positions],
        })

    # Close video writers + env.
    for w in video_writers:
        try:
            for packet in w["stream"].encode():
                w["container"].mux(packet)
            w["container"].close()
        except Exception as exc:
            print(f"  video close failed for {w['name']}: {exc}")
    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-objects", type=int, default=10,
                   help="Sample N candidates. 0 = all graspable.")
    p.add_argument("--copies", type=int, default=2,
                   help="Number of identical copies stacked per column (>=2).")
    p.add_argument("--settle-steps", type=int, default=30,
                   help="Physics steps for gravity settle (≥10 per spec).")
    p.add_argument("--shake-velocity", type=float, default=0.3,
                   help="Random horizontal velocity (m/s) applied to every "
                        "copy after the gravity settle. 0.0 disables shake.")
    p.add_argument("--shake-settle-steps", type=int, default=30,
                   help="Physics steps to settle after the shake impulse.")
    p.add_argument("--bbox-tol", type=float, default=1.05,
                   help="Stack stable iff each copy's x AND y both lie within "
                        "tol * native_xy_bbox of the column anchor.")
    p.add_argument("--col-spacing", type=float, default=2.0,
                   help="Distance (m) between adjacent test columns. Must "
                        "exceed the largest object's bbox + max shake "
                        "displacement so a fallen copy from one column "
                        "can't disturb its neighbour. Default 2.0 m is "
                        "safe for the BEHAVIOR-1K graspable pool (max "
                        "bbox ≈ 0.5 m) at the default shake velocity.")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Candidates per env load. The env is recreated "
                        "between batches to bound memory. 0 = no batching.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=str(CSV_PATH))
    p.add_argument("--output", default=str(HERE / "stack_self_full.json"),
                   help="Path for the per-candidate JSON results. Default "
                        "is alongside the script so the build pipeline picks "
                        "it up automatically.")
    p.add_argument("--save-video", default=None,
                   help="Directory to write per-batch MP4s. Each batch "
                        "produces 3 cameras: front, top-down, perspective.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-resolution", type=int, default=512)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="If --output exists, skip candidates already in it "
                        "and append remaining results.")
    args = p.parse_args()

    if args.headless:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"

    # Macros must be set BEFORE the first env is created — they get locked
    # once any setting is read by env init. Setting per-batch crashes on
    # the second batch with "Cannot set attribute, already used".
    from omnigibson.macros import gm
    gm.ENABLE_TRANSITION_RULES = False

    pool = load_graspable_pool(args.csv)
    print(f"Loaded {len(pool)} graspable candidates from {args.csv}")
    rng = np.random.default_rng(args.seed)
    if args.n_objects > 0 and len(pool) > args.n_objects:
        idx = rng.choice(len(pool), args.n_objects, replace=False)
        pool = [pool[i] for i in idx]
    print(f"Testing {len(pool)} candidates "
          f"({args.copies} copies/stack, settle={args.settle_steps}, "
          f"shake={args.shake_velocity}m/s + {args.shake_settle_steps} steps)")

    all_results = []
    if args.resume and os.path.isfile(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        all_results = existing.get("results", [])
        done_keys = {(r["category"], r["model"]) for r in all_results}
        before = len(pool)
        pool = [(c, m) for (c, m) in pool if (c, m) not in done_keys]
        print(f"Resume: loaded {len(all_results)} existing results, "
              f"skipping {before - len(pool)} done candidates "
              f"({len(pool)} remaining)")

    batch_size = args.batch_size or max(1, len(pool))
    n_batches = max(1, (len(pool) + batch_size - 1) // batch_size)
    print(f"Batching: {n_batches} batches of up to {batch_size} candidates")

    t_start = time.time()
    for b in range(n_batches):
        batch = pool[b * batch_size:(b + 1) * batch_size]
        t_batch = time.time()
        print(f"\n=== Batch {b + 1}/{n_batches}: {len(batch)} candidates ===")
        results = _run_one_batch(batch, args, b)
        all_results.extend(results)
        n_stable_so_far = sum(1 for r in all_results if r["stable"])
        elapsed = time.time() - t_start
        batch_t = time.time() - t_batch
        print(f"  batch done in {batch_t:.1f}s | "
              f"running stable: {n_stable_so_far}/{len(all_results)} | "
              f"elapsed: {elapsed:.0f}s")

        # Per-candidate result lines.
        for r in results:
            flag = "STABLE  " if r["stable"] else "UNSTABLE"
            extra = f" reason={r.get('reason')}" if not r["stable"] and r.get("reason") else ""
            print(f"  [{flag}] {r['category']}/{r['model']}{extra}")

        # Write incrementally.
        with open(args.output, "w") as f:
            json.dump({
                "args": vars(args),
                "n_candidates": len(pool),
                "n_processed": len(all_results),
                "n_stable": n_stable_so_far,
                "results": all_results,
            }, f, indent=2)

    n_stable_final = sum(1 for r in all_results if r["stable"])
    print(f"\nFinal: stable {n_stable_final}/{len(all_results)} "
          f"in {time.time() - t_start:.0f}s → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
