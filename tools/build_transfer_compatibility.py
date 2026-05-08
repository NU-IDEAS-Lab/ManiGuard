"""Regenerate transfer_compatibility.json from food + container geometry.

Geometric filter: a food fits in a container iff
    max(food.bbox_dims_m) <= container.opening_minor_m

That is, the food's longest 3D dimension (any of x/y/z) must fit through
the container's narrowest opening dim. Fully 3D-rotation-independent: the
food drops in regardless of how it's tumbled. Strictest possible
orientation-free fit test.

Asset-readiness filter (from docs/graspability_classified.csv):
  * food: status=graspable AND food_transfer_target in {perfect, possible}
  * container: status=graspable AND wide_opening_container in {perfect, possible}

Models flagged ``too_large``, ``not_ready``, ``no_grasp``, or
``degenerate_bbox`` are excluded — these have unresolved asset complaints.

Reads:
    sentinel/task_generation/utils/food_cross_sections.json
    sentinel/task_generation/utils/wide_opening_sizes.json
    docs/graspability_classified.csv
Writes:
    sentinel/task_generation/utils/transfer_compatibility.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = ROOT / "sentinel" / "task_generation" / "utils"
FOOD_PATH = UTILS_DIR / "food_cross_sections.json"
OPEN_PATH = UTILS_DIR / "wide_opening_sizes.json"
OUT_PATH = UTILS_DIR / "transfer_compatibility.json"
GRASP_CSV = ROOT / "docs" / "graspability_classified.csv"

_USABLE_SUITABILITY = {"perfect", "possible"}


def load_foods() -> list[tuple[str, str, float]]:
    """Return [(category, model, max_bbox_dim_m), ...].

    ``max_bbox_dim_m`` is ``max(bbox_dims_m)`` — the food's longest 3D dim
    across any axis. The filter requires this to fit through the
    container's narrowest opening dim.
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


def load_containers() -> list[tuple[str, str, float, float]]:
    """Return [(category, model, opening_major_m, opening_minor_m), ...]."""
    with open(OPEN_PATH) as f:
        raw = json.load(f)
    rows = []
    for cat, cat_info in raw.items():
        for m in cat_info.get("models", []):
            op_maj = m.get("opening_major_m")
            op_min = m.get("opening_minor_m")
            if op_maj is None or op_min is None:
                continue
            rows.append((cat, m["model"], float(op_maj), float(op_min)))
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
    containers = [(c, m, mj, mn) for (c, m, mj, mn) in containers if (c, m) in container_ok]
    print(f"After readiness filter: {len(foods)} foods, {len(containers)} containers.")

    # Sort foods by (category, model) for deterministic output.
    foods.sort()

    out: dict[str, dict] = {}
    for c_cat, c_model, op_maj, op_min in containers:
        fit = [
            {"category": f_cat, "model": f_model}
            for f_cat, f_model, f_max in foods
            if f_max <= op_min
        ]
        out[f"{c_cat}/{c_model}"] = {
            "category": c_cat,
            "model": c_model,
            "opening_major_m": round(op_maj, 4),
            "opening_minor_m": round(op_min, 4),
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
