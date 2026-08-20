"""Derive per-object features from the top-surface scan.

Reads ``scan_top_full.json`` (raycast heightmaps from above) and computes,
for each object, the largest axis-aligned square that fits within the
up-facing surface at:

  * **z_max** — the highest reachable up-facing surface. For flat-top
    objects this is the actual top; for receptacles this is usually the
    rim ring.

  * **z_min** — the lowest reachable up-facing surface. For flat-top
    objects this equals z_max within tolerance. For receptacles, it's
    the cavity floor — but only where the rays from above could
    actually reach it. Narrow-mouthed objects (vases, bottles) may
    block the cavity floor from being visible at all, in which case
    z_min just describes whatever was reachable through the opening.

For each level, we compute the binary mask of cells whose height is
within ``tol_m`` of that level, then run the standard largest-square-
of-1s DP. The result is reported as the square side length in metres
(cells × cell pitch).

Output: a sidecar JSON ``derived_top_features.json`` keyed by
``"<category>/<model>"``. Does NOT mutate the original scan file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "scan_top_full.json"
DEFAULT_OUTPUT = HERE / "derived_top_features.json"


def largest_square_in_mask(mask: np.ndarray) -> int:
    """Largest axis-aligned square of True cells. Returns side in cells."""
    if mask.size == 0:
        return 0
    h, w = mask.shape
    dp = np.zeros((h, w), dtype=np.int32)
    best = 0
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue
            if i == 0 or j == 0:
                dp[i, j] = 1
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            if dp[i, j] > best:
                best = int(dp[i, j])
    return best


def derive_one(info: dict, tol_m: float) -> dict:
    if "spawn_error" in info:
        return {"spawn_error": info["spawn_error"]}

    h = np.asarray(info["heights_facing"], dtype=np.float32)
    aabb_min = info["aabb_min"]
    aabb_max = info["aabb_max"]
    grid = h.shape[0]

    # Cell pitch in metres (assume linspace endpoint-inclusive sampling
    # over the AABB span).
    span_x = aabb_max[0] - aabb_min[0]
    span_y = aabb_max[1] - aabb_min[1]
    pitch_x = span_x / max(1, grid - 1)
    pitch_y = span_y / max(1, grid - 1)
    pitch = float(min(pitch_x, pitch_y))

    valid = ~np.isnan(h)
    n_facing = int(valid.sum())
    if n_facing == 0:
        return {
            "n_facing": 0,
            "z_max": float("nan"),
            "z_min": float("nan"),
            "square_at_z_max_side_m": 0.0,
            "square_at_z_max_side_cells": 0,
            "square_at_z_min_side_m": 0.0,
            "square_at_z_min_side_cells": 0,
            "cell_pitch_m": pitch,
        }

    z_max = float(np.nanmax(h))
    z_min = float(np.nanmin(h))

    mask_max = valid & (np.abs(h - z_max) <= tol_m)
    mask_min = valid & (np.abs(h - z_min) <= tol_m)

    s_max = largest_square_in_mask(mask_max)
    s_min = largest_square_in_mask(mask_min)

    return {
        "n_facing": n_facing,
        "cell_pitch_m": pitch,
        "z_max": z_max,
        "z_min": z_min,
        "z_range_m": float(z_max - z_min),
        "tol_m": float(tol_m),
        "square_at_z_max_side_cells": int(s_max),
        "square_at_z_max_side_m": float(s_max * pitch),
        "square_at_z_min_side_cells": int(s_min),
        "square_at_z_min_side_m": float(s_min * pitch),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--tol-m", type=float, default=0.005,
                   help="Cells count toward the plateau if their height "
                        "is within ±tol_m of z_max / z_min. Default 5 mm.")
    args = p.parse_args()

    with open(args.input) as f:
        doc = json.load(f)
    src_results = doc.get("results", {})
    print(f"Reading {len(src_results)} entries from {args.input}")

    derived = {}
    n_err = 0
    for key, info in src_results.items():
        d = derive_one(info, args.tol_m)
        if "spawn_error" in d:
            n_err += 1
        derived[key] = d

    with open(args.output, "w") as f:
        json.dump({
            "tol_m": args.tol_m,
            "n_entries": len(derived),
            "n_spawn_errors": n_err,
            "results": derived,
        }, f, indent=2)
    print(f"Wrote {args.output} ({len(derived)} entries, "
          f"{n_err} spawn errors carried over)")

    # Quick stats / examples.
    valid = [(k, v) for k, v in derived.items()
             if "spawn_error" not in v and v["n_facing"] > 0]
    valid.sort(key=lambda kv: -kv[1]["square_at_z_max_side_m"])
    print("\nTop 10 by largest plateau square at z_max:")
    for k, v in valid[:10]:
        print(f"  {k:40s} side={v['square_at_z_max_side_m']*100:5.1f} cm "
              f"(z_max={v['z_max']*100:.1f}, range={v['z_range_m']*100:.1f} cm)")

    cavity_like = [(k, v) for k, v in valid
                   if v["z_range_m"] > 0.02 and v["square_at_z_min_side_m"] > 0]
    cavity_like.sort(key=lambda kv: -kv[1]["square_at_z_min_side_m"])
    print("\nTop 10 'cavity-like' (z_range > 2 cm) by largest square at z_min:")
    for k, v in cavity_like[:10]:
        print(f"  {k:40s} z_min_side={v['square_at_z_min_side_m']*100:5.1f} cm "
              f"z_max_side={v['square_at_z_max_side_m']*100:5.1f} cm "
              f"depth={v['z_range_m']*100:.1f} cm")


if __name__ == "__main__":
    main()
