"""Prune user-decided bad tasks from a finalized ManiGuard-Bench family + reindex contiguously.

After a full ``run_finalize_base`` run, fails land in ``base_manifest.jsonl`` (+ a ``drop_list.json``
CANDIDATE list). The user reviews them: tool bugs get fixed (re-validate, not dropped); genuinely
unreasonable tasks (e.g. jar task_0004 — light container + heavy lid tips over) get pruned here.

``prune_and_reindex(family, drop)`` deletes the dropped task folders and renames the survivors to a
gap-free ``task_0000..`` sequence — the bench keeps its OWN contiguous index, decoupled from the
read-only source. Only the folder name + the per-task ``_finalize_row.json``/manifest ``task`` field
carry the index (activity_name's ``trial_N`` is a decoupled generation label that moves with the
task; the snapshot/videos carry no index), so this is a pure rename + manifest rebuild. Writes
``_index_map.json`` (``bench task -> original task``, = source task pre-prune) for traceability and
deletes ``drop_list.json``.

Usage:
  python -m maniguard.data.bench_builder.prune_reindex --family jar_transport --drop 4
  python -m maniguard.data.bench_builder.prune_reindex --family clutter_pickup --drop 3,17 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

OUT_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
ROW_FILE = "_finalize_row.json"


def _norm_task(tok: str) -> str:
    tok = tok.strip()
    return tok if tok.startswith("task_") else f"task_{int(tok):04d}"


def _parse_drop(spec: str) -> set[str]:
    return {_norm_task(t) for t in spec.split(",") if t.strip()}


def prune_and_reindex(family: str, drop: set[str], out_root: str = OUT_ROOT_DEFAULT,
                      *, dry_run: bool = False) -> dict:
    fam_dir = Path(out_root) / family
    if not fam_dir.is_dir():
        raise FileNotFoundError(f"family dir not found: {fam_dir}")
    tasks = sorted(d.name for d in fam_dir.glob("task_*") if d.is_dir())
    missing = sorted(drop - set(tasks))
    if missing:
        raise ValueError(f"--drop names not present in {fam_dir}: {missing}")

    keep = [t for t in tasks if t not in drop]                       # sorted ascending
    index_map = {f"task_{i:04d}": old for i, old in enumerate(keep)}  # new -> original(=source)
    renames = [(old, new) for new, old in index_map.items() if new != old]  # new<=old -> asc-safe

    plan = {
        "family": family, "n_before": len(tasks), "n_dropped": len(drop), "n_after": len(keep),
        "dropped": sorted(drop), "renames": renames,
    }
    if dry_run:
        plan["dry_run"] = True
        return plan

    # 1. delete dropped task folders
    for t in sorted(drop):
        shutil.rmtree(fam_dir / t)
    # 2. reindex survivors contiguously (ascending old-index order is collision-safe: each new
    #    slot was already vacated by a drop or a prior rename). Fix each row file's task field.
    #    renames already in ascending old order because index_map iterates keep ascending.
    for old, new in renames:
        (fam_dir / old).rename(fam_dir / new)
        row_path = fam_dir / new / "base" / ROW_FILE
        if row_path.exists():
            r = json.loads(row_path.read_text(encoding="utf-8"))
            r["task"] = new
            row_path.write_text(json.dumps(r, default=float), encoding="utf-8")
    # 3. rebuild base_manifest.jsonl (drop removed rows, remap task names; data unchanged)
    old2new = {v: k for k, v in index_map.items()}
    manifest = fam_dir / "base_manifest.jsonl"
    new_rows = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ot = row.get("task")
            if ot in drop:
                continue
            nt = old2new.get(ot, ot)
            row["task"] = nt
            for sub in ("finalize", "validate"):
                if isinstance(row.get(sub), dict) and "task" in row[sub]:
                    row[sub]["task"] = nt
            new_rows.append(row)
        new_rows.sort(key=lambda r: r["task"])
        manifest.write_text("\n".join(json.dumps(r, default=float) for r in new_rows) + "\n",
                            encoding="utf-8")
    # 4. write traceability map; discard the (now-resolved) candidate drop_list
    (fam_dir / "_index_map.json").write_text(json.dumps(index_map, indent=2) + "\n", encoding="utf-8")
    dl = fam_dir / "drop_list.json"
    if dl.exists():
        dl.unlink()
    plan["manifest_rows"] = len(new_rows)
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune dropped tasks + reindex a bench family contiguously.")
    ap.add_argument("--family", required=True)
    ap.add_argument("--drop", required=True, help="task(s) to drop: '4' or '4,7' or 'task_0004'")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="print the plan without touching files")
    args = ap.parse_args()
    plan = prune_and_reindex(args.family, _parse_drop(args.drop), args.out_root, dry_run=args.dry_run)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
