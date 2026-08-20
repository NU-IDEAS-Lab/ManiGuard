"""Regenerate ``object_footprints.json`` from BEHAVIOR asset metadata.

Walks every ``behavior-1k/datasets/behavior-1k-assets/objects/<cat>/<model>/misc/metadata.json``,
extracts ``bbox_size = [x, y, z]``, and emits a category-first catalog with
``footprint_m2 = x * y`` per model. ``task_spec.estimate_object_set_footprint``
consumes this catalog so the scene/surface picker sizes correctly for the
diverse data-driven pools.

The unscaled XY footprint is the right quantity for sizing decisions because
clutter objects spawn at scale 1.0 by default. Models whose metadata lacks
``bbox_size`` are recorded under ``metadata.skipped``.

Run:
    conda activate behavior
    python -m maniguard.task_generation.utils.build_object_footprints

Pure-Python — no OmniGibson import. ~8.6k metadata files, <5 s total.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
ASSETS_ROOT = os.path.join(_REPO, "behavior-1k", "datasets", "behavior-1k-assets", "objects")
OUT_PATH = os.path.join(_HERE, "object_footprints.json")


def main():
    t0 = time.time()
    catalog: dict[str, dict[str, dict]] = defaultdict(dict)
    skipped: list[tuple[str, str, str]] = []  # (category, model, reason)
    scanned = 0

    for category in sorted(os.listdir(ASSETS_ROOT)):
        cat_dir = os.path.join(ASSETS_ROOT, category)
        if not os.path.isdir(cat_dir):
            continue
        for model in sorted(os.listdir(cat_dir)):
            meta_path = os.path.join(cat_dir, model, "misc", "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            scanned += 1
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as exc:
                skipped.append((category, model, f"parse_error: {exc!s}"))
                continue
            bbox = meta.get("bbox_size")
            if not (isinstance(bbox, list) and len(bbox) == 3):
                skipped.append((category, model, "missing_bbox_size"))
                continue
            try:
                x, y, z = float(bbox[0]), float(bbox[1]), float(bbox[2])
            except (TypeError, ValueError):
                skipped.append((category, model, "non_numeric_bbox"))
                continue
            if not (x > 0 and y > 0 and z > 0):
                skipped.append((category, model, "degenerate_bbox"))
                continue
            catalog[category][model] = {
                "extent_xyz": [round(x, 6), round(y, 6), round(z, 6)],
                "footprint_m2": round(x * y, 6),
            }

    # Sort models within each category for stable diffs.
    out = {cat: dict(sorted(models.items())) for cat, models in sorted(catalog.items())}
    payload = {
        "metadata": {
            "source": "behavior-1k/datasets/behavior-1k-assets/objects/<cat>/<model>/misc/metadata.json",
            "field": "bbox_size (unscaled, axis-aligned)",
            "categories": len(out),
            "models": sum(len(m) for m in out.values()),
            "skipped": len(skipped),
            "skipped_reasons": dict((r, sum(1 for _, _, rr in skipped if rr == r))
                                    for r in sorted({rr for _, _, rr in skipped})),
        },
    }
    payload.update(out)

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    rel = os.path.relpath(OUT_PATH, _REPO)
    print(f"Wrote {rel}: {payload['metadata']['categories']} categories, "
          f"{payload['metadata']['models']} models, "
          f"skipped {payload['metadata']['skipped']} "
          f"({payload['metadata']['skipped_reasons']}) "
          f"in {time.time() - t0:.1f}s "
          f"(scanned {scanned} metadata.json files)")


if __name__ == "__main__":
    main()
