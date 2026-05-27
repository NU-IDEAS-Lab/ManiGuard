"""Derive per-container opening location + size from the existing
top-down raycast scan (`stack_pipeline/scan_top_full.json`).

Pure-Python (no OmniGibson). Mirrors `derive_top_features.py`'s cavity
detection (lowest upward-facing surface + tolerance band, identical to
what stack_recep_compat already filters against), and additionally
records the cavity centroid XY and floor Z so the food placer can drop
the food directly above the actual opening — not the AABB center.

Per container model in the wide-opening universe, output:

  scan_status:
    "ok" if cavity floor was found,
    "no_cavity" if no upward-facing hit was reachable (closed object),
    "spawn_error" if the underlying scan errored.

  opening_centroid_xy_relative_to_aabb_center_m: [dx, dy]
      Offset of the cavity-floor centroid from the AABB XY center, in
      container-local frame. Frame-agnostic: at runtime, recompute the
      AABB center and add this offset.

  opening_floor_z_above_aabb_min_m: dz
      Cavity-floor world Z minus AABB min Z. Same as `z_min` in
      derive_top_features.

  opening_floor_depth_below_aabb_top_m: cavity depth (rim → floor).

  opening_square_side_m:
      Side of the largest axis-aligned square that fits in the cavity
      mask — the food-fit dimension. Identical to
      `square_at_z_min_side_m` in derive_top_features.

  opening_area_m2:
      n_cavity_cells × cell area.

  n_cavity_cells / n_components:
      Diagnostics. n_components > 1 hints at multi-region cavities
      (e.g. handles or a divider in the middle of the opening).

Reads:
    maniguard/task_generation/utils/stack_pipeline/scan_top_full.json
    docs/graspability_classified.csv
        (used as the candidate filter — only entries that are graspable
         AND wide_opening_container ∈ {perfect, possible} are output.)

Writes:
    maniguard/task_generation/utils/food_transfer_pipeline/container_openings.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]   # repo root: …/ManiGuard
SCAN_PATH = HERE.parent / "stack_pipeline" / "scan_top_full.json"
GRASP_CSV = ROOT / "docs" / "graspability_classified.csv"
DEFAULT_OUT = HERE / "container_openings.json"

_USABLE_SUITABILITY = {"perfect", "possible"}

CAVITY_TOL_M = 0.005   # cells within 5 mm of the lowest upward-facing hit
                        # count as "cavity floor". Same tolerance the
                        # stack_recep pipeline already uses.


def _largest_square_side_cells(mask: np.ndarray) -> int:
    """Largest axis-aligned square of True cells, side in cells.
    Same DP as derive_top_features.largest_square_in_mask."""
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
                dp[i, j] = 1 + min(int(dp[i-1, j]), int(dp[i, j-1]), int(dp[i-1, j-1]))
            if dp[i, j] > best:
                best = int(dp[i, j])
    return best


def _connected_components(mask: np.ndarray) -> int:
    visited = np.zeros_like(mask)
    n = 0
    rows, cols = mask.shape
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] and not visited[i, j]:
                n += 1
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    if not (0 <= a < rows and 0 <= b < cols):
                        continue
                    if visited[a, b] or not mask[a, b]:
                        continue
                    visited[a, b] = True
                    stack.extend([(a+1, b), (a-1, b), (a, b+1), (a, b-1)])
    return n


def _derive_one(entry: dict) -> dict:
    if "spawn_error" in entry:
        return {"scan_status": f"spawn_error: {entry['spawn_error']}"}

    h = np.asarray(entry["heights_facing"], dtype=np.float32)  # height above aabb_min, NaN if not facing
    aabb_min = entry["aabb_min"]
    aabb_max = entry["aabb_max"]
    grid = h.shape[0]

    span_x = float(aabb_max[0] - aabb_min[0])
    span_y = float(aabb_max[1] - aabb_min[1])
    span_z = float(aabb_max[2] - aabb_min[2])
    if grid < 2 or span_x <= 0 or span_y <= 0 or span_z <= 0:
        return {"scan_status": "degenerate_aabb"}

    pitch_x = span_x / (grid - 1)
    pitch_y = span_y / (grid - 1)
    cell_area = pitch_x * pitch_y
    cell_pitch = float(min(pitch_x, pitch_y))

    valid = ~np.isnan(h)
    if not valid.any():
        return {"scan_status": "no_cavity"}

    z_min = float(np.nanmin(h))
    cavity_mask = valid & (np.abs(h - z_min) <= CAVITY_TOL_M)
    n_cavity = int(cavity_mask.sum())
    if n_cavity == 0:
        return {"scan_status": "no_cavity"}

    # Cell world XY (rays were cast from a linspace over the AABB span).
    xs = np.linspace(aabb_min[0], aabb_max[0], grid)
    ys = np.linspace(aabb_min[1], aabb_max[1], grid)
    ix, iy = np.where(cavity_mask)
    open_xs = xs[ix]
    open_ys = ys[iy]
    centroid_world_x = float(np.mean(open_xs))
    centroid_world_y = float(np.mean(open_ys))

    aabb_cx = 0.5 * (aabb_min[0] + aabb_max[0])
    aabb_cy = 0.5 * (aabb_min[1] + aabb_max[1])
    dx = centroid_world_x - aabb_cx
    dy = centroid_world_y - aabb_cy

    side_cells = _largest_square_side_cells(cavity_mask)
    side_m = float(side_cells * cell_pitch)
    area_m2 = float(n_cavity * cell_area)

    return {
        "scan_status": "ok",
        "opening_centroid_xy_relative_to_aabb_center_m": [dx, dy],
        "opening_floor_z_above_aabb_min_m": float(z_min),
        "opening_floor_depth_below_aabb_top_m": float(span_z - z_min),
        "opening_square_side_m": side_m,
        "opening_square_side_cells": int(side_cells),
        "opening_area_m2": area_m2,
        "n_cavity_cells": n_cavity,
        "n_components": int(_connected_components(cavity_mask)),
        "aabb_size_m": [span_x, span_y, span_z],
        "cell_pitch_m": cell_pitch,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N candidates (smoke test).")
    args = p.parse_args()

    scan = json.loads(SCAN_PATH.read_text())["results"]
    candidate_keys = []
    with open(GRASP_CSV) as f:
        for r in csv.DictReader(f):
            if r["status"] != "graspable":
                continue
            if r.get("wide_opening_container", "") not in _USABLE_SUITABILITY:
                continue
            candidate_keys.append(f"{r['category']}/{r['model']}")
    if args.limit:
        candidate_keys = candidate_keys[: args.limit]

    out_containers = {}
    n_ok = n_no = n_err = n_skip = 0
    for key in candidate_keys:
        entry = scan.get(key)
        if entry is None:
            n_skip += 1
            continue
        result = _derive_one(entry)
        cat, model = key.split("/", 1)
        out_containers.setdefault(cat, {"models": []})["models"].append(
            {"model": model, **result})
        st = result.get("scan_status", "?")
        if st == "ok":
            n_ok += 1
        elif st == "no_cavity":
            n_no += 1
        else:
            n_err += 1

    out = {
        "metadata": {
            "source_scan": str(SCAN_PATH.relative_to(ROOT)),
            "source_universe": str(GRASP_CSV.relative_to(ROOT)),
            "cavity_tol_m": CAVITY_TOL_M,
            "n_candidates": len(candidate_keys),
            "n_skipped_not_in_scan": n_skip,
            "n_ok": n_ok,
            "n_no_cavity": n_no,
            "n_error": n_err,
        },
        "containers": out_containers,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")
    print(f"  candidates={len(candidate_keys)}  ok={n_ok}  "
          f"no_cavity={n_no}  error={n_err}  skipped={n_skip}")


if __name__ == "__main__":
    main()
