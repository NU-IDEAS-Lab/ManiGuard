#!/usr/bin/env python3
"""Build a fresh CSV listing every (category, model) in the BEHAVIOR-1K
object dataset, with a prefilter status from each object's
``misc/metadata.json`` bbox.

Output schema matches the columns ``render_grasps.py --csv`` expects::

    category,model,status,...

``status=too_large`` for objects whose largest bbox dim exceeds
``--max-dim`` (default 0.5 m, matching the original antipodal survey),
``status=pending`` otherwise. Pass ``--exclude-statuses too_large``
to ``render_grasps.py`` and it will iterate the rest.

Usage::

    python scripts/build_grasp_target_csv.py \\
        --output sentinel/utils/franka_graspability_full.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path,
                   default=Path("behavior-1k/datasets/behavior-1k-assets/objects"))
    p.add_argument("--output", type=Path,
                   default=Path("sentinel/utils/franka_graspability_full.csv"))
    p.add_argument("--max-dim", type=float, default=0.5,
                   help="Largest-bbox-dim threshold (m). Objects above this "
                        "are flagged 'too_large'.")
    p.add_argument("--complaints-json", type=Path,
                   default=Path("behavior-1k/bddl3/bddl/generated_data/complaints.json"),
                   help="BDDL complaints file. Objects with at least one "
                        "unprocessed (``processed=False``) complaint are "
                        "flagged 'not_ready'. Set to '' to skip.")
    return p.parse_args()


def _load_not_ready_set(complaints_json: Path) -> set[tuple[str, str]]:
    """Parse BDDL complaints.json and return the set of (category, model)
    pairs with at least one outstanding (unprocessed) complaint.

    Resolved complaints (``processed=True``) don't count — those went
    through review and are not blockers anymore. The ``processed`` field
    is occasionally a string ``'FALSE'`` instead of a bool; treat any
    non-truthy value as outstanding.
    """
    if not complaints_json or not complaints_json.exists():
        return set()
    not_ready: set[tuple[str, str]] = set()
    for entry in json.loads(complaints_json.read_text()):
        obj = entry.get("object", "")
        if "-" not in obj:
            continue
        cat, mdl = obj.rsplit("-", 1)
        processed = entry.get("processed", False)
        if processed is True:
            continue
        not_ready.add((cat, mdl))
    return not_ready


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    not_ready = _load_not_ready_set(args.complaints_json)
    print(f"Loaded {len(not_ready)} not-ready (cat, model) pairs from "
          f"{args.complaints_json}")

    n_total = 0
    n_too_large = 0
    n_degenerate = 0
    n_no_meta = 0
    n_not_ready = 0
    n_pending = 0

    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "model", "status", "n_cand", "n_tried",
                    "elapsed_s", "note"])
        for cat_dir in sorted(args.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            for model_dir in sorted(cat_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                n_total += 1
                cat, mdl = cat_dir.name, model_dir.name
                meta = model_dir / "misc" / "metadata.json"
                if not meta.exists():
                    n_no_meta += 1
                    w.writerow([cat, mdl, "no_metadata", 0, 0, 0.0, ""])
                    continue
                try:
                    bbox = json.loads(meta.read_text()).get("bbox_size")
                except Exception:  # noqa: BLE001
                    bbox = None
                if bbox is None:
                    n_no_meta += 1
                    w.writerow([cat, mdl, "no_metadata", 0, 0, 0.0, ""])
                    continue
                if any((x is None or x <= 0) for x in bbox):
                    n_degenerate += 1
                    w.writerow([cat, mdl, "degenerate_bbox", 0, 0, 0.0,
                                str(bbox)])
                    continue
                max_dim = float(max(bbox))
                if max_dim > args.max_dim:
                    n_too_large += 1
                    w.writerow([cat, mdl, "too_large", 0, 0, 0.0,
                                f"max_dim={max_dim:.3f}"])
                    continue
                if (cat, mdl) in not_ready:
                    n_not_ready += 1
                    w.writerow([cat, mdl, "not_ready", 0, 0, 0.0,
                                f"max_dim={max_dim:.3f}"])
                    continue
                n_pending += 1
                w.writerow([cat, mdl, "pending", 0, 0, 0.0,
                            f"max_dim={max_dim:.3f}"])

    print(f"Wrote {args.output}")
    print(f"  total            : {n_total}")
    print(f"  too_large (>{args.max_dim:.2f}m): {n_too_large}")
    print(f"  degenerate_bbox  : {n_degenerate}")
    print(f"  no_metadata      : {n_no_meta}")
    print(f"  not_ready (BDDL outstanding complaint): {n_not_ready}")
    print(f"  pending (attempt): {n_pending}")


if __name__ == "__main__":
    main()
