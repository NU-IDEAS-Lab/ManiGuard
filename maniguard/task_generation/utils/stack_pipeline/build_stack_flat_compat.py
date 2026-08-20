"""Build stack_flat_compatibility.json — geometric (target, stack_item) pairs.

A stack_flat task: a flat target sits on the support surface; N copies of a
stack_item rest on its top. Compatibility filter:

    max(item.native_bbox_xy) <= target.square_at_z_max_side_m

i.e. the stack item's longer XY dimension must fit within the side length
of the largest axis-aligned square that fits on the target's flat top
plateau (computed from raycast, normal-filtered up-facing surface).

Sources:
  * Targets   — maniguard/task_generation/utils/stack_pipeline/derived_top_features.json
                (per-object flat-top square at z_max, from the 24×24
                top-surface raycast). Filtered by status=graspable +
                no unresolved complaint.
  * Stack items — maniguard/task_generation/utils/stack_pipeline/stack_same_pool.json
                (already passes self-stack stability + graspable + no
                unresolved complaint). Carries native_xy_bbox.

Output schema (category-first so random selection is unbiased toward
categories with many models):
  {
    "<target_category>": [
      {
        "model": ...,
        "square_at_z_max_side_m": float,
        "z_max_m": float,
        "n_items_fit": int,
        "items": [
          {"category": ..., "model": ..., "native_xy_bbox": [dx, dy]},
          ...
        ]
      },
      ...
    ],
    ...
  }

Excluded from targets:
  * z_max < 1 cm (basically no top to rest on — these are rugs, paper)
    or square side < 5 cm (plateau too small to be useful for stacking)
  * Whole categories listed in TARGET_EXCLUDE_CATS below — semantic
    exclusions for objects that pass the geometry test but make no
    sense as a stack-flat *target* (e.g. a pizza isn't a target you'd
    retrieve from under a stack).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
DERIVED_PATH = HERE / "derived_top_features.json"
STACK_SAME_PATH = HERE / "stack_same_pool.json"
GRASP_CSV = ROOT / "docs" / "graspability_classified.csv"
COMPLAINTS_PATH = (
    ROOT / "behavior-1k" / "bddl3" / "bddl" / "generated_data" / "complaints.json"
)
OUT_PATH = HERE / "stack_flat_compatibility.json"

# Geometry gates on the target's flat top.
MIN_Z_MAX_M = 0.01            # plateau must sit ≥ 1cm above the floor
MIN_PLATEAU_SIDE_M = 0.05     # plateau must be ≥ 5cm × 5cm

# Categories where a model can pass the geometry filter but isn't a
# semantically valid retrieve-from-under-a-stack target — food items,
# closed containers, narrow-necked vessels, etc. Add categories here
# only after confirming the failure mode visually.
TARGET_EXCLUDE_CATS = frozenset({
    # No-op for now; populate from real-world inspection later.
})


def load_unresolved_complaints():
    out = set()
    with open(COMPLAINTS_PATH) as f:
        complaints = json.load(f)
    for c in complaints:
        if c.get("processed", False):
            continue
        key = c["object"]
        if "-" not in key:
            continue
        cat, model = key.rsplit("-", 1)
        out.add((cat, model))
    return out


def load_graspable_set():
    out = set()
    with open(GRASP_CSV) as f:
        for r in csv.DictReader(f):
            if r["status"] == "graspable":
                out.add((r["category"], r["model"]))
    return out


def load_targets(graspable, unresolved):
    """Return list of target dicts that pass geometry + readiness gates."""
    with open(DERIVED_PATH) as f:
        derived = json.load(f)["results"]
    targets = []
    for key, d in derived.items():
        if "spawn_error" in d:
            continue
        cat, model = key.split("/", 1)
        if (cat, model) not in graspable or (cat, model) in unresolved:
            continue
        if cat in TARGET_EXCLUDE_CATS:
            continue
        if d.get("z_max", float("nan")) < MIN_Z_MAX_M:
            continue
        side = d.get("square_at_z_max_side_m", 0.0)
        if side < MIN_PLATEAU_SIDE_M:
            continue
        targets.append({
            "category": cat, "model": model,
            "square_at_z_max_side_m": float(side),
            "z_max_m": float(d["z_max"]),
        })
    return targets


def load_stack_items():
    """Return list of stack-item dicts (model + xy bbox) from stack_same_pool."""
    with open(STACK_SAME_PATH) as f:
        pool = json.load(f)
    items = []
    for cat, entries in pool.items():
        for e in entries:
            bbox = e.get("native_xy_bbox") or [None, None]
            if not bbox or bbox[0] is None or bbox[1] is None:
                continue
            items.append({
                "category": cat, "model": e["model"],
                "native_xy_bbox": [float(bbox[0]), float(bbox[1])],
                "max_dim": max(float(bbox[0]), float(bbox[1])),
            })
    return items


def main():
    graspable = load_graspable_set()
    unresolved = load_unresolved_complaints()
    targets = load_targets(graspable, unresolved)
    items = load_stack_items()

    print(f"Targets after geometry + readiness gates: {len(targets)}")
    print(f"Stack items (from verified self-stack pool): {len(items)}")

    by_cat = defaultdict(list)
    for t in targets:
        side = t["square_at_z_max_side_m"]
        fit = []
        for it in items:
            if it["max_dim"] <= side:
                fit.append({
                    "category": it["category"],
                    "model": it["model"],
                    "native_xy_bbox": it["native_xy_bbox"],
                })
        if not fit:
            continue
        by_cat[t["category"]].append({
            "model": t["model"],
            "square_at_z_max_side_m": round(side, 4),
            "z_max_m": round(t["z_max_m"], 4),
            "n_items_fit": len(fit),
            "items": fit,
        })

    # Sort models within each category for deterministic output.
    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: e["model"])
    out = {cat: by_cat[cat] for cat in sorted(by_cat.keys())}

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    n_targets = sum(len(v) for v in out.values())
    n_pairs = sum(e["n_items_fit"] for v in out.values() for e in v)
    print(f"\nWrote {OUT_PATH}: {len(out)} categories / "
          f"{n_targets} targets / {n_pairs} fit-pairs total")
    flat = [(cat, e) for cat, lst in out.items() for e in lst]
    print("\nTop 10 (cat, model) by # of compatible stack items:")
    for cat, e in sorted(flat, key=lambda x: -x[1]["n_items_fit"])[:10]:
        print(f"  {cat}/{e['model']:30s} side={e['square_at_z_max_side_m']*100:5.1f} cm  "
              f"items_fit={e['n_items_fit']}")
    print("\nTop 10 categories by model count:")
    for cat, lst in sorted(out.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {cat:30s} {len(lst)} models")


if __name__ == "__main__":
    main()
