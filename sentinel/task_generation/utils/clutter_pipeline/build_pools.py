"""Build clutter_target_pool.json and table_obstacle_pool.json from the
graspability CSV.

Filter (both pools):
  * status == graspable
  * <column> in {perfect, possible} where column is "clutter_target"
    (target_pool) or "table_obstacle" (obstacle_pool)

Output schema (category-keyed → synset + model list):
  {
    "metadata": {...source CSV path, counts, column...},
    "<category>": {
      "synset": "<synset.n.0X>",
      "models": ["<model_id>", ...]
    },
    ...
  }

The synset is resolved via BEHAVIOR ``OBJECT_TAXONOMY.get_synset_from_category``
so downstream code can keep using synset-level spawn specs while pinning a
specific graspable model id.

Run:
    conda activate behavior
    python -m sentinel.task_generation.utils.clutter_pipeline.build_pools
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
TARGET_OUT = os.path.join(_HERE, "clutter_target_pool.json")
OBSTACLE_OUT = os.path.join(_HERE, "table_obstacle_pool.json")

ACCEPT = {"perfect", "possible"}


def _build(column: str, taxonomy: ObjectTaxonomy):
    pool: dict[str, list[str]] = defaultdict(list)
    skipped_no_synset: list[str] = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "graspable":
                continue
            if row[column] not in ACCEPT:
                continue
            pool[row["category"]].append(row["model"])

    out: dict[str, dict] = {}
    for cat in sorted(pool):
        try:
            synset = taxonomy.get_synset_from_category(cat)
        except Exception:
            skipped_no_synset.append(cat)
            continue
        if synset is None:
            skipped_no_synset.append(cat)
            continue
        out[cat] = {"synset": synset, "models": sorted(pool[cat])}
    return out, skipped_no_synset


def main():
    tax = ObjectTaxonomy()
    target_pool, target_skipped = _build("clutter_target", tax)
    obstacle_pool, obstacle_skipped = _build("table_obstacle", tax)

    def _write(path, pool, column, skipped):
        payload = {
            "metadata": {
                "source_csv": os.path.relpath(CSV_PATH, _REPO),
                "filter": f"status=graspable AND {column} in {sorted(ACCEPT)}",
                "categories": len(pool),
                "models": sum(len(v["models"]) for v in pool.values()),
                "skipped_no_synset": sorted(skipped),
            },
        }
        payload.update(pool)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        rel = os.path.relpath(path, _REPO)
        print(f"  wrote {rel}: "
              f"{payload['metadata']['categories']} categories, "
              f"{payload['metadata']['models']} models"
              + (f"  (skipped {len(skipped)} no-synset: {skipped[:5]}{'…' if len(skipped) > 5 else ''})"
                 if skipped else ""))

    _write(TARGET_OUT, target_pool, "clutter_target", target_skipped)
    _write(OBSTACLE_OUT, obstacle_pool, "table_obstacle", obstacle_skipped)


if __name__ == "__main__":
    main()
