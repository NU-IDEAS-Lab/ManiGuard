#!/usr/bin/env python3
"""Update franka_graspability_full.csv in place after a `no_grasp` re-evaluation.

Scans the per-object output directories produced by render_grasps and rewrites
the ``status`` column: rows previously marked ``no_grasp`` whose re-run
produced a ``<category>_<model>_success`` directory are flipped to
``graspable`` (preserving original n_cand/n_tried/elapsed_s untouched but
tagging the note column). Rows whose re-run wrote ``<category>_<model>_fail``
or had no output directory at all stay ``no_grasp``.

Usage:
    python tools/update_graspability_from_recheck.py \
        --csv sentinel/task_generation/utils/franka_graspability_full.csv \
        --output-dir outputs/grasp_datasets/no_grasp_recheck
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=Path("sentinel/task_generation/utils/franka_graspability_full.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_datasets/no_grasp_recheck"))
    p.add_argument("--target-status", default="no_grasp",
                   help="Only rows whose current status matches this are eligible "
                        "for update.")
    p.add_argument("--new-status", default="graspable",
                   help="Status to write when the re-run succeeded.")
    p.add_argument("--note-tag", default="rechecked",
                   help="String appended to the note column for updated rows.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report changes without writing back.")
    args = p.parse_args()

    out_dir: Path = args.output_dir.resolve()
    if not out_dir.is_dir():
        sys.exit(f"output dir not found: {out_dir}")

    # Build {(category, model): "success" | "fail"} from output dirs.
    results: dict[tuple[str, str], str] = {}
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        for suffix in ("_success", "_fail"):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                # category and model are separated by the LAST underscore
                # in our naming, but model strings have no underscore by
                # convention. Split from the right once.
                if "_" not in stem:
                    continue
                category, model = stem.rsplit("_", 1)
                results[(category, model)] = suffix[1:]   # "success" / "fail"
                break

    print(f"[scan] {len(results)} per-object output dirs found in {out_dir}")

    # Read the CSV, rewrite eligible rows.
    csv_path: Path = args.csv.resolve()
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    n_total = sum(1 for r in rows if r.get("status") == args.target_status)
    n_promoted = 0
    n_still_failed = 0
    n_no_artifact = 0

    for row in rows:
        if row.get("status") != args.target_status:
            continue
        key = (row["category"], row["model"])
        outcome = results.get(key)
        if outcome == "success":
            row["status"] = args.new_status
            note = row.get("note", "")
            row["note"] = (
                f"{note} | {args.note_tag}" if note else args.note_tag
            )
            n_promoted += 1
        elif outcome == "fail":
            n_still_failed += 1
        else:
            n_no_artifact += 1

    print(f"[update] eligible (status={args.target_status!r}): {n_total}")
    print(f"  promoted to {args.new_status!r}: {n_promoted}")
    print(f"  still failed (re-run produced _fail): {n_still_failed}")
    print(f"  no re-run artifact: {n_no_artifact}")

    if args.dry_run:
        print("[dry-run] not writing back")
        return

    if n_promoted == 0:
        print("[update] no changes — leaving CSV untouched")
        return

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[update] wrote {csv_path}")


if __name__ == "__main__":
    main()
