"""Top-surface raycast scan for graspable objects.

Spawns N candidate objects in an empty scene at identity orientation,
each in its own XY column, then for each one casts a GRID×GRID grid of
rays from above downward over the object's world-frame XY AABB. The
first hit per ray is the top-surface z. Cells are kept iff the surface
normal is mostly upward (``normal_z > 0.85``) — the actual up-facing
geometry where a stack item would land.

For ``stack_flat`` targets this is the resting surface; for
``stack_receptacle`` targets it reveals the cavity outline + depth.

Usage:
    # 2-object smoke test
    python maniguard/task_generation/utils/stack_pipeline/scan_top_surface.py \\
        --n-objects 0 --grid 24

    # 50-object batch with PNG grid visualization
    python maniguard/task_generation/utils/stack_pipeline/scan_top_surface.py \\
        --n-objects 50 --grid 24 \\
        --output scan_top_50.json --plot scan_top_50.png \\
        --plot-rows 10 --plot-cols 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
CSV_PATH = ROOT / "docs" / "graspability_classified.csv"

DEFAULT_TARGETS = [
    ("wine_bottle", "inkqch"),
    ("wineglass", "aakcyj"),
]
COL_SPACING_M = 1.5
RAY_PADDING_M = 0.05
NORMAL_Z_MIN = 0.85


def load_graspable_pool():
    out = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["status"] == "graspable":
                out.append((r["category"], r["model"]))
    return out


def scan_one(psqi, obj, grid):
    aabb_min, aabb_max = obj.aabb
    x0, y0 = float(aabb_min[0]), float(aabb_min[1])
    x1, y1 = float(aabb_max[0]), float(aabb_max[1])
    z_top = float(aabb_max[2]) + RAY_PADDING_M
    ray_len = float(aabb_max[2] - aabb_min[2]) + 2 * RAY_PADDING_M
    aabb_min_z = float(aabb_min[2])

    xs = np.linspace(x0, x1, grid)
    ys = np.linspace(y0, y1, grid)
    heights_raw = np.full((grid, grid), np.nan, dtype=np.float32)
    heights_facing = np.full((grid, grid), np.nan, dtype=np.float32)
    normals_z = np.full((grid, grid), np.nan, dtype=np.float32)
    n_hit = 0
    n_facing = 0

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            hit = psqi.raycast_closest(
                origin=[float(x), float(y), z_top],
                dir=[0.0, 0.0, -1.0],
                distance=float(ray_len),
            )
            if not hit.get("hit"):
                continue
            n_hit += 1
            z_rel = float(hit["position"][2]) - aabb_min_z
            nz = float(hit["normal"][2])
            heights_raw[ix, iy] = z_rel
            normals_z[ix, iy] = nz
            if nz > NORMAL_Z_MIN:
                heights_facing[ix, iy] = z_rel
                n_facing += 1

    return {
        "aabb_min": [float(v) for v in aabb_min],
        "aabb_max": [float(v) for v in aabb_max],
        "n_hits_total": n_hit,
        "n_hits_facing": n_facing,
        "n_total": grid * grid,
        "top_z_relative_min":
            float(np.nanmin(heights_facing)) if n_facing else float("nan"),
        "top_z_relative_max":
            float(np.nanmax(heights_facing)) if n_facing else float("nan"),
        "heights_raw": heights_raw.tolist(),
        "heights_facing": heights_facing.tolist(),
        "normals_z": normals_z.tolist(),
    }


def render_grid_plot(results, path, n_rows, n_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(results.items())
    if len(items) > n_rows * n_cols:
        items = items[: n_rows * n_cols]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_axis_off()
    for idx, (key, info) in enumerate(items):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_axis_on()
        h = np.asarray(info["heights_facing"], dtype=np.float32) * 1000
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("lightgray")
        sx = (info["aabb_max"][0] - info["aabb_min"][0]) * 100
        sy = (info["aabb_max"][1] - info["aabb_min"][1]) * 100
        if info["n_hits_facing"] > 0:
            ax.imshow(np.ma.masked_invalid(h).T, origin="lower", cmap=cmap,
                      extent=[0, sx, 0, sy], aspect="equal")
        else:
            ax.imshow(np.full_like(h, np.nan).T, cmap=cmap, origin="lower")
        ax.set_title(f"{key}\n{info['n_hits_facing']}/{info['n_total']}",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Wrote {path}")


def _run_one_batch(targets, grid, batch_idx):
    """Spawn one batch's worth of objects, scan each, tear down. Returns
    {key: per-object dict}. ``og.clear()`` between batches is required —
    re-creating an env without clearing the prior stage trips PhysX's
    perspective-component check (same fix as the stack-self test)."""
    import omnigibson as og
    from omnigibson.objects import DatasetObject

    if batch_idx > 0:
        og.clear()

    cfg = {
        "scene": {"type": "Scene"},
        "robots": [],
        "task": {"type": "DummyTask"},
    }
    env = og.Environment(configs=cfg)
    og.sim.step()

    objs = []
    spawn_errs = {}
    for i, (cat, model) in enumerate(targets):
        try:
            obj = DatasetObject(name=f"target_{i}_{cat}_{model}",
                                category=cat, model=model)
            env.scene.add_object(obj)
            objs.append((i, cat, model, obj))
        except Exception as exc:
            spawn_errs[(cat, model)] = str(exc)
    og.sim.step()

    for i, _, _, obj in objs:
        nbb = obj.native_bbox
        scale = obj.scale
        h = max(0.01, float(nbb[2] * scale[2]))
        obj.set_position_orientation(
            position=(i * COL_SPACING_M, 0.0, 0.5 + h),
            orientation=(0, 0, 0, 1),
        )
        obj.keep_still()
    og.sim.step()

    psqi = og.sim.psqi
    results = {}
    for _, cat, model, obj in objs:
        results[f"{cat}/{model}"] = scan_one(psqi, obj, grid)
    for (cat, model), err in spawn_errs.items():
        results[f"{cat}/{model}"] = {"spawn_error": err}

    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-objects", type=int, default=0,
                   help="Sample N graspable candidates (0 = ALL graspable; "
                        "use a small N for smoke tests).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Objects per env load. The env is rebuilt with "
                        "og.clear() between batches to bound stage state.")
    p.add_argument("--output", default="scan_top_surface_smoke.json",
                   help="Filename (relative to this script's dir if no slash).")
    p.add_argument("--plot", default=None,
                   help="Optional PNG of per-object heightmap grid.")
    p.add_argument("--plot-rows", type=int, default=10)
    p.add_argument("--plot-cols", type=int, default=5)
    p.add_argument("--resume", action="store_true",
                   help="If --output exists, skip already-scanned (cat,model) "
                        "pairs and append remaining results.")
    args = p.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

    pool = load_graspable_pool()
    if args.n_objects == 0 and not args.n_objects:
        # 0 = all
        targets = pool
    elif args.n_objects > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pool), min(args.n_objects, len(pool)), replace=False)
        targets = [pool[i] for i in idx]
    else:
        targets = DEFAULT_TARGETS

    out_path = args.output
    if "/" not in out_path:
        out_path = HERE / out_path
    out_path = Path(out_path)

    all_results = {}
    if args.resume and out_path.is_file():
        with open(out_path) as f:
            existing = json.load(f)
        all_results = existing.get("results", {})
        before = len(targets)
        targets = [(c, m) for (c, m) in targets if f"{c}/{m}" not in all_results]
        print(f"Resume: loaded {len(all_results)} existing, "
              f"skipping {before - len(targets)} done "
              f"({len(targets)} remaining)")

    bs = max(1, args.batch_size)
    n_batches = max(1, (len(targets) + bs - 1) // bs)
    print(f"Scanning {len(targets)} objects at {args.grid}×{args.grid} "
          f"in {n_batches} batches of up to {bs}")

    # Set macro before any env is created.
    from omnigibson.macros import gm
    gm.ENABLE_TRANSITION_RULES = False

    import time
    t_start = time.time()
    for b in range(n_batches):
        batch = targets[b * bs:(b + 1) * bs]
        t_batch = time.time()
        print(f"\n=== Batch {b + 1}/{n_batches}: {len(batch)} objects ===")
        batch_results = _run_one_batch(batch, args.grid, b)
        all_results.update(batch_results)
        n_ok = sum(1 for r in batch_results.values() if "spawn_error" not in r)
        n_err = sum(1 for r in batch_results.values() if "spawn_error" in r)
        print(f"  batch done in {time.time() - t_batch:.1f}s "
              f"(scanned {n_ok}, spawn-fail {n_err}) | "
              f"total processed: {len(all_results)} | "
              f"elapsed: {time.time() - t_start:.0f}s")

        with open(out_path, "w") as f:
            json.dump({
                "args": vars(args),
                "normal_z_min": NORMAL_Z_MIN,
                "n_processed": len(all_results),
                "results": all_results,
            }, f, indent=2)

    print(f"\nFinal: {len(all_results)} entries in "
          f"{time.time() - t_start:.0f}s → {out_path}")

    if args.plot:
        plot_path = args.plot if "/" in args.plot else HERE / args.plot
        # Only plot entries with actual scan data (skip spawn errors).
        plottable = {k: v for k, v in all_results.items()
                     if "spawn_error" not in v}
        render_grid_plot(plottable, plot_path, args.plot_rows, args.plot_cols)


if __name__ == "__main__":
    main()
