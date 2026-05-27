"""Build ``fragile_pool.json`` from the graspability CSV + footprint catalog.

Filter:
  * ``status == "graspable"`` (from ``docs/graspability_classified.csv``).
  * Height ``extent_z >= Z_MIN_M`` — must stand tall enough to have a
    well-defined upright orientation.
  * Aspect ratio ``extent_z / min(extent_x, extent_y) >= ASPECT_MIN`` —
    must be narrow enough to tip.
  * Minimum thickness ``min(extent_x, extent_y) >= MIN_THICKNESS_M``.
  * XY symmetry ``max(extent_x, extent_y) / min(extent_x, extent_y) <=
    MAX_XY_RATIO``. Together with the min-thickness floor this excludes
    flat wall-mounted items (``painting``, ``signpost``, ``wall_clock``,
    ``outlet``) — they pass the aspect rule because they're thin in one
    XY dimension while being tall in z, but they're not column-like.

The geometric definition aligns with the LTL safety predicate
``all_fragiles_upright``: a flat bowl can be ceramic-fragile but cannot
"fall over", so the gate doesn't catch it either way.

Output schema (category-keyed → synset + model list) mirrors
``clutter_target_pool.json`` / ``table_obstacle_pool.json`` so the
runtime selector can sample uniformly by category then by model.

Run:
    conda activate behavior
    python -m maniguard.task_generation.utils.clutter_pipeline.build_fragile_pool
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

from bddl.object_taxonomy import ObjectTaxonomy

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
CSV_PATH = os.path.join(_REPO, "docs", "graspability_classified.csv")
FOOTPRINTS_PATH = os.path.join(_HERE, "..", "object_footprints.json")
OUT_PATH = os.path.join(_HERE, "fragile_pool.json")

# Geometric thresholds. ASPECT_MIN=3.0 selects tall-and-narrow objects
# (bottles, atomizers, candles, dispensers, vases, wineglasses).
# MIN_THICKNESS_M=0.03 + MAX_XY_RATIO=3.0 enforce column-like footprints,
# so flat wall-mounted items (painting, signpost, wall_clock) are
# excluded even though they're geometrically tall.
Z_MIN_M = 0.10
ASPECT_MIN = 3.0
MIN_THICKNESS_M = 0.03
MAX_XY_RATIO = 3.0


def main():
    tax = ObjectTaxonomy()
    with open(FOOTPRINTS_PATH) as f:
        footprints = json.load(f)

    graspable: set[tuple[str, str]] = set()
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "graspable":
                graspable.add((row["category"], row["model"]))

    pool: dict[str, list[str]] = defaultdict(list)
    for cat, model in graspable:
        info = footprints.get(cat, {}).get(model)
        if info is None:
            continue
        ex, ey, ez = info["extent_xyz"]
        min_xy, max_xy = min(ex, ey), max(ex, ey)
        if ez < Z_MIN_M or min_xy < MIN_THICKNESS_M:
            continue
        if max_xy / min_xy > MAX_XY_RATIO:
            continue
        if ez / min_xy < ASPECT_MIN:
            continue
        pool[cat].append(model)

    skipped_no_synset: list[str] = []
    out: dict[str, dict] = {}
    for cat in sorted(pool):
        synset = tax.get_synset_from_category(cat)
        if synset is None:
            skipped_no_synset.append(cat)
            continue
        out[cat] = {"synset": synset, "models": sorted(pool[cat])}

    payload = {
        "metadata": {
            "source_csv": os.path.relpath(CSV_PATH, _REPO),
            "footprint_catalog": os.path.relpath(FOOTPRINTS_PATH, _REPO),
            "filter": (
                f"status=graspable AND extent_z >= {Z_MIN_M} AND "
                f"min(extent_x, extent_y) >= {MIN_THICKNESS_M} AND "
                f"max(extent_x, extent_y) / min(extent_x, extent_y) <= "
                f"{MAX_XY_RATIO} AND "
                f"extent_z / min(extent_x, extent_y) >= {ASPECT_MIN}"
            ),
            "z_min_m": Z_MIN_M,
            "aspect_min": ASPECT_MIN,
            "min_thickness_m": MIN_THICKNESS_M,
            "max_xy_ratio": MAX_XY_RATIO,
            "categories": len(out),
            "models": sum(len(v["models"]) for v in out.values()),
            "skipped_no_synset": sorted(skipped_no_synset),
        },
    }
    payload.update(out)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    rel = os.path.relpath(OUT_PATH, _REPO)
    print(f"wrote {rel}: {payload['metadata']['categories']} categories, "
          f"{payload['metadata']['models']} models "
          f"(z>={Z_MIN_M}, min_xy>={MIN_THICKNESS_M}, "
          f"xy_ratio<={MAX_XY_RATIO}, aspect>={ASPECT_MIN})")


if __name__ == "__main__":
    main()
