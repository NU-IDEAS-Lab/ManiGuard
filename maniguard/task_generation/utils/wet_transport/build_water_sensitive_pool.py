"""Build ``water_sensitive_pool.json`` from the graspability CSV.

The pool is the curated set of categories that must NOT get wet during a
``wet_transport`` task — books, papers, electronics. For each curated
category, this script collects every ``status=graspable`` model from
``docs/graspability_classified.csv`` and writes them out at
category-keyed ``{synset, models}`` granularity, matching the shape
``fragile_pool.json`` / ``fillable_container_pool.json`` already use so
the runtime selector can sample uniformly by category then by model.

No geometric filter: the LTL safety predicate for wet_transport is
"don't pass over zone" — purely spatial, independent of object shape.
Any graspable book/laptop/etc. is a valid zone.

Categories with NO graspable models in the CSV are skipped silently
(monitor, tablet have only ``no_grasp`` / ``not_ready`` entries today;
laptop has only ``not_ready``).

Run:
    conda activate behavior
    python -m maniguard.task_generation.utils.wet_transport.build_water_sensitive_pool
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
OUT_PATH = os.path.join(_HERE, "water_sensitive_pool.json")

# The curated water-sensitive set. Same as the old WATER_SENSITIVE_POOL
# constant in task_spec.py — papers and electronics that shouldn't get
# wet. Build admits every graspable model under these categories; the
# selector samples uniformly by category-then-by-model.
WATER_SENSITIVE_CATEGORIES = (
    "hardback",
    "notebook",
    "letter",
    "newspaper",
    "magazine",
    "folder",
    "laptop",
    "keyboard",
    "tablet",
    "monitor",
)


def main():
    tax = ObjectTaxonomy()

    pool: dict[str, list[str]] = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "graspable":
                continue
            cat = row["category"]
            if cat not in WATER_SENSITIVE_CATEGORIES:
                continue
            pool[cat].append(row["model"])

    skipped_no_synset: list[str] = []
    skipped_no_graspable = sorted(
        c for c in WATER_SENSITIVE_CATEGORIES if c not in pool
    )

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
            "filter": "status=graspable AND category ∈ WATER_SENSITIVE_CATEGORIES",
            "curated_categories": list(WATER_SENSITIVE_CATEGORIES),
            "categories": len(out),
            "models": sum(len(v["models"]) for v in out.values()),
            "skipped_no_synset": sorted(skipped_no_synset),
            "skipped_no_graspable": skipped_no_graspable,
        },
    }
    payload.update(out)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    rel = os.path.relpath(OUT_PATH, _REPO)
    print(
        f"wrote {rel}: {payload['metadata']['categories']} categories, "
        f"{payload['metadata']['models']} models "
        f"(no-graspable skipped: {skipped_no_graspable or 'none'})"
    )


if __name__ == "__main__":
    main()
