#!/usr/bin/env python3
"""Consolidate bigrun_v1 eval outputs from multiple boxes into ONE clean final dataset.

The raw per-box trees are working state: results.jsonl files may contain crashed
retry rows before the final completed row, cells re-run on another box after a
mid-run migration (duplicate completed rows across sources), and benign archive
copies of the same file merged across trees. The OFFICIAL dataset must contain
exactly ONE completed row per grid cell and that row's artifacts — nothing else.

Rules (deterministic):
  * a cell = (model, family, seed, scene_name); the full grid is enumerated from
    the local bench (task dirs x 5 levels) x 3 models x 3 seeds;
  * only rows with status == "completed" are candidates (crashed / load_failed
    rows are retry archaeology and are DROPPED from the final tree);
  * when several sources hold a completed row for the same cell, the FIRST
    source in the given priority order wins (same episode_seed => near-clone
    trajectories, so the choice is immaterial but must be deterministic);
  * the winning cell dir is copied whole (videos + ltl json + client log follow
    the last-writer-wins attempt, which is the completed one), with
    results.jsonl REWRITTEN to hold only the single winning completed row;
  * a ledger (JSON) records: per-cell source, duplicates resolved, crashed rows
    dropped, nan_terminated counts, and the MISSING cell list (must be empty
    before the dataset is called final).

Usage:
  python tools/consolidate_bigrun.py --sources ARCH_FL:path1 ARCH_TW:path2 NO:path3 JP:path4 \\
      --bench outputs/lerobot_datasets/maniguard-bench \\
      --emit outputs/eval_logs/bigrun_v1_final [--dry-run]

Each source path must contain a ``bigrun_v1/`` tree. Priority = CLI order.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

MODELS = ("pi05", "gr00t", "smolvla")
SEEDS = ("seed0", "seed1", "seed2")
LEVELS = ("base", "target", "language", "location", "env")
FAM2BENCH = {
    "clutter": "clutter_pickup", "cabinet": "cabinet_pickup", "stack": "stack_retrieve",
    "jar": "jar_transport", "lid": "lid_transport", "dusty": "dusty_transfer",
}


def expected_grid(bench_root: Path) -> set[tuple[str, str, str, str]]:
    cells = set()
    for fam, bfam in FAM2BENCH.items():
        tasks = sorted(p.name for p in (bench_root / bfam).glob("task_*") if p.is_dir())
        for model in MODELS:
            for seed in SEEDS:
                for t in tasks:
                    for lv in LEVELS:
                        cells.add((model, fam, seed, f"{t}/{lv}"))
    return cells


def scan_source(name: str, root: Path):
    """Yield (cell_key, run_dir, completed_row) for every completed row in the tree."""
    base = root / "bigrun_v1"
    for rp in base.glob("*/*/*/*/results.jsonl"):
        model, fam, seed, run_name = rp.parts[-5], rp.parts[-4], rp.parts[-3], rp.parts[-2]
        rows = []
        for line in rp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        completed = [r for r in rows if r.get("status") == "completed"]
        dropped = len(rows) - len(completed)
        for r in completed:
            key = (model, fam, seed, r.get("scene_name", ""))
            yield key, rp.parent, r, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="NAME:PATH entries; priority = order given")
    ap.add_argument("--bench", required=True, help="local maniguard-bench root")
    ap.add_argument("--emit", default=None, help="write the clean final tree here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = expected_grid(Path(args.bench))
    print(f"expected grid cells: {len(grid)}")

    chosen: dict[tuple, tuple[str, Path, dict]] = {}
    dupes = Counter()
    dropped_rows = 0
    for entry in args.sources:
        name, _, path = entry.partition(":")
        root = Path(path)
        n_new = n_dup = 0
        for key, run_dir, row, dropped in scan_source(name, root):
            dropped_rows += dropped
            if key in chosen:
                dupes[key] += 1
                n_dup += 1
                continue
            chosen[key] = (name, run_dir, row)
            n_new += 1
        print(f"source {name:8s}: +{n_new} cells, {n_dup} duplicates skipped")

    extra = [k for k in chosen if k not in grid]
    missing = sorted(grid - set(chosen))
    nan_n = sum(1 for _, _, r in chosen.values() if r.get("nan_terminated"))
    print(f"\nunique completed cells: {len(chosen)} / {len(grid)}")
    print(f"missing: {len(missing)} | off-grid extras: {len(extra)} | "
          f"cross-source duplicates resolved: {sum(dupes.values())} | "
          f"non-completed rows dropped: {dropped_rows} | nan_terminated kept (valid failures): {nan_n}")
    for k in missing[:20]:
        print("  MISSING", "/".join(k))
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")

    if args.emit and not args.dry_run:
        emit = Path(args.emit)
        emit.mkdir(parents=True, exist_ok=True)
        for key, (src, run_dir, row) in chosen.items():
            model, fam, seed, scene = key
            rn = scene.replace("/", "_")
            dst = emit / model / fam / seed / rn
            if dst.exists():
                continue
            shutil.copytree(run_dir, dst)
            # official tree: exactly one completed row, no retry archaeology
            (dst / "results.jsonl").write_text(
                json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")
        ledger = {
            "expected": len(grid), "unique_completed": len(chosen),
            "missing": ["/".join(k) for k in missing],
            "duplicates_resolved": {"/".join(k): v for k, v in dupes.items()},
            "dropped_non_completed_rows": dropped_rows,
            "nan_terminated_kept": nan_n,
            "source_priority": args.sources,
            "per_cell_source": {"/".join(k): v[0] for k, v in chosen.items()},
        }
        (emit / "LEDGER.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        print(f"\nfinal tree + LEDGER.json written to {emit}")
        print("DATASET IS FINAL ONLY IF missing == 0.")


if __name__ == "__main__":
    main()
