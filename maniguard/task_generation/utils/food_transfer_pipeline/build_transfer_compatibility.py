"""Regenerate transfer_compatibility.json from food + container geometry.

Geometric filter: a food fits in a container iff
    max(food.bbox_dims_m) <= container.opening_square_side_m

That is, the food's longest 3D dimension (any of x/y/z) must fit inside
the largest axis-aligned square that fits in the cavity opening (from
the top-down raycast in derive_container_openings). 3D-rotation
independent — the food drops in regardless of how it's tumbled.

Asset-readiness filter (from docs/graspability_classified.csv):
  * food: status=graspable AND food_transfer_target in {perfect, possible}
  * container: status=graspable AND wide_opening_container in {perfect, possible}

Models flagged ``too_large``, ``not_ready``, ``no_grasp``, or
``degenerate_bbox`` are excluded — these have unresolved asset complaints.

Reads:
    maniguard/task_generation/utils/food_transfer_pipeline/food_cross_sections.json
    maniguard/task_generation/utils/food_transfer_pipeline/container_openings.json
    docs/graspability_classified.csv
Writes:
    maniguard/task_generation/utils/food_transfer_pipeline/transfer_compatibility.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]   # repo root: …/ManiGuard
FOOD_PATH = HERE / "food_cross_sections.json"
OPEN_PATH = HERE / "container_openings.json"
OUT_PATH = HERE / "transfer_compatibility.json"
GRASP_CSV = ROOT / "docs" / "graspability_classified.csv"

_USABLE_SUITABILITY = {"perfect", "possible"}


def load_foods() -> list[tuple[str, str, float]]:
    """Return [(category, model, max_bbox_dim_m), ...].

    ``max_bbox_dim_m`` is ``max(bbox_dims_m)`` — the food's longest 3D dim
    across any axis. The filter requires this to fit through the
    container's largest inscribed-square opening.
    """
    with open(FOOD_PATH) as f:
        raw = json.load(f)
    rows = []
    for cat, cat_info in raw.items():
        for m in cat_info.get("models", []):
            bbox = m.get("bbox_dims_m") or []
            if not bbox:
                continue
            rows.append((cat, m["model"], float(max(bbox))))
    return rows


def load_containers() -> list[dict]:
    """Return container entries with opening geometry, only ``scan_status='ok'``.

    Each entry: ``{category, model, opening_square_side_m, centroid_xy,
    floor_z_above_aabb_min_m, floor_depth_below_aabb_top_m, area_m2}``.
    """
    with open(OPEN_PATH) as f:
        raw = json.load(f)
    rows = []
    for cat, cat_info in raw.get("containers", {}).items():
        for m in cat_info.get("models", []):
            if m.get("scan_status") != "ok":
                continue
            side = m.get("opening_square_side_m")
            if side is None or side <= 0:
                continue
            rows.append({
                "category": cat,
                "model": m["model"],
                "opening_square_side_m": float(side),
                "opening_centroid_xy_relative_to_aabb_center_m":
                    m.get("opening_centroid_xy_relative_to_aabb_center_m", [0.0, 0.0]),
                "opening_floor_z_above_aabb_min_m":
                    float(m.get("opening_floor_z_above_aabb_min_m", 0.0)),
                "opening_floor_depth_below_aabb_top_m":
                    float(m.get("opening_floor_depth_below_aabb_top_m", 0.0)),
                "opening_area_m2": float(m.get("opening_area_m2", 0.0)),
            })
    return rows


def load_readiness() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return (food_ok, container_ok) sets of (category, model) keys.

    A model qualifies iff status == "graspable" AND its role-specific
    suitability column is "perfect" or "possible".
    """
    food_ok: set[tuple[str, str]] = set()
    container_ok: set[tuple[str, str]] = set()
    with open(GRASP_CSV) as f:
        for r in csv.DictReader(f):
            if r["status"] != "graspable":
                continue
            key = (r["category"], r["model"])
            if r.get("food_transfer_target", "") in _USABLE_SUITABILITY:
                food_ok.add(key)
            if r.get("wide_opening_container", "") in _USABLE_SUITABILITY:
                container_ok.add(key)
    return food_ok, container_ok


def main() -> int:
    foods = load_foods()
    containers = load_containers()
    food_ok, container_ok = load_readiness()
    print(f"Loaded {len(foods)} foods, {len(containers)} containers; "
          f"readiness sets: {len(food_ok)} food / {len(container_ok)} container.")

    # Filter by asset readiness up front.
    foods = [(c, m, mj) for (c, m, mj) in foods if (c, m) in food_ok]
    containers = [c for c in containers if (c["category"], c["model"]) in container_ok]
    print(f"After readiness filter: {len(foods)} foods, {len(containers)} containers.")

    # Sort foods by (category, model) for deterministic output.
    foods.sort()

    out: dict[str, dict] = {}
    for c in containers:
        side = c["opening_square_side_m"]
        fit = [
            {"category": f_cat, "model": f_model}
            for f_cat, f_model, f_max in foods
            if f_max <= side
        ]
        cx, cy = c["opening_centroid_xy_relative_to_aabb_center_m"]
        out[f"{c['category']}/{c['model']}"] = {
            "category": c["category"],
            "model": c["model"],
            "opening_square_side_m": round(side, 4),
            "opening_centroid_xy_relative_to_aabb_center_m":
                [round(float(cx), 4), round(float(cy), 4)],
            "opening_floor_z_above_aabb_min_m":
                round(c["opening_floor_z_above_aabb_min_m"], 4),
            "opening_floor_depth_below_aabb_top_m":
                round(c["opening_floor_depth_below_aabb_top_m"], 4),
            "opening_area_m2": round(c["opening_area_m2"], 6),
            "n_food_models_fit": len(fit),
            "food_models": fit,
        }

    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            old = json.load(f)
        old_total = sum(e.get("n_food_models_fit", 0) for e in old.values())
        new_total = sum(e["n_food_models_fit"] for e in out.values())
        print(f"Old: {len(old)} containers, {old_total} fit-pairs total.")
        print(f"New: {len(out)} containers, {new_total} fit-pairs total.")
    else:
        print(f"New: {len(out)} containers, "
              f"{sum(e['n_food_models_fit'] for e in out.values())} fit-pairs total.")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
