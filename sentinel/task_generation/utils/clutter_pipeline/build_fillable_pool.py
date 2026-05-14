"""Build ``fillable_container_pool.json`` from graspability + BEHAVIOR taxonomy.

Filter:
  * ``status == "graspable"`` (from ``docs/graspability_classified.csv``).
  * BEHAVIOR taxonomy abilities include ``"fillable"`` or
    ``"openfillable"`` (so the object can host a substance via
    ``Filled.set_value`` at runtime).
  * Model is present in ``object_footprints.json`` (i.e. the asset
    metadata yielded a usable bbox).

Output schema matches the sibling pools (``clutter_target_pool.json`` /
``table_obstacle_pool.json``): category-keyed → synset + model list.

Consumed by ``liquid_transport_pipeline`` as the target pool, and by
``wet_transport_pipeline`` as the carried-container pool.

Run:
    conda activate behavior
    python -m sentinel.task_generation.utils.clutter_pipeline.build_fillable_pool
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
OUT_PATH = os.path.join(_HERE, "fillable_container_pool.json")

_FILLABLE_ABILITIES = ("fillable", "openfillable")


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
        if cat not in footprints or model not in footprints[cat]:
            continue
        synset = tax.get_synset_from_category(cat)
        if synset is None:
            continue
        abilities = tax.get_abilities(synset)
        if not any(a in abilities for a in _FILLABLE_ABILITIES):
            continue
        pool[cat].append(model)

    out: dict[str, dict] = {}
    for cat in sorted(pool):
        synset = tax.get_synset_from_category(cat)
        out[cat] = {"synset": synset, "models": sorted(pool[cat])}

    payload = {
        "metadata": {
            "source_csv": os.path.relpath(CSV_PATH, _REPO),
            "footprint_catalog": os.path.relpath(FOOTPRINTS_PATH, _REPO),
            "filter": (
                f"status=graspable AND BEHAVIOR ability in "
                f"{list(_FILLABLE_ABILITIES)} AND model in "
                f"object_footprints.json"
            ),
            "categories": len(out),
            "models": sum(len(v["models"]) for v in out.values()),
        },
    }
    payload.update(out)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    rel = os.path.relpath(OUT_PATH, _REPO)
    print(f"wrote {rel}: {payload['metadata']['categories']} categories, "
          f"{payload['metadata']['models']} models")


if __name__ == "__main__":
    main()
