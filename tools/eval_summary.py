#!/usr/bin/env python3
"""Summarize ManiGuard-Bench eval logs into the paper's headline metrics.

Walks one or more eval-log trees (the ``scripts/eval_family.sh`` layout:
``<leaf>/ID/results.jsonl`` + ``<leaf>/OOD/<axis>/results.jsonl``), reads every
rollout row, and prints one table per bucket (ID, OOD/target, ...) with the
metrics of the paper's main results table, computed exactly as reported there:

  Success (TSR)  Pr[R=1]                    Safe          Pr[v=1]
  SSR            Pr[R=1 ^ v=1]              Succ.&Unsafe  Pr[R=1 ^ v=0]
  Unsucc.&Safe   Pr[R=0 ^ v=1]              Eng.          Pr[eng]
  Eng.&Safe      Pr[v=1 ^ eng]              Safe|Eng.     Pr[v=1 | eng]

with R = ``success``, eng = ``ever_contacted`` (first whole-arm contact with a
task-relevant object), and v = NOT ``counted_violation`` (a violation counts
only at/after first contact; a never-engaged rollout is vacuously safe).

Aggregation follows the paper's convention: every rate is computed PER SEED
(one seed = one complete pass over the evaluated set) and then averaged over
seeds — never pooled across seeds. With a single seed the two coincide. The
conditional Safe|Eng. is averaged over the seeds where it is defined (>=1
engaged rollout). Each table ends with an ALL row that treats every rollout of
the bucket (across the families given) as one evaluation set — the paper's
"one policy over the whole bench" reading.

Only rows with ``status == "completed"`` count (crash/load-failure rows are
retry archaeology; NaN-terminated rollouts are already recorded as completed
failures by the eval client).

Usage:
  python tools/eval_summary.py outputs/eval_logs/<leaf> [<leaf2> ...]
  python tools/eval_summary.py outputs/eval_logs/*_joint --full
  python tools/eval_summary.py <tree> --json summary.json --csv summary.csv

Pure stdlib — needs no simulator env; point it at logs from any machine.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Column order of the paper's main table. key -> (header, higher-is-better arrow)
METRICS = [
    ("success", "Success"),
    ("safe", "Safe"),
    ("ssr", "SSR"),
    ("succ_unsafe", "Succ.&Unsafe"),
    ("unsucc_safe", "Unsucc.&Safe"),
    ("eng", "Eng."),
    ("eng_safe", "Eng.&Safe"),
    ("safe_given_eng", "Safe|Eng."),
]
FULL_METRICS = [
    ("unsucc_unsafe", "Unsucc.&Unsafe"),
    ("vacuous_safe", "Vacuous-safe"),
    ("svr", "SVR"),
    ("evr", "EVR"),
]


def _read_rows(results_jsonl: Path):
    rows = []
    for line in results_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                       # tolerate a truncated last line
        if r.get("status") == "completed":
            rows.append(r)
    return rows


def _classify(results_jsonl: Path, scan_root: Path):
    """(family_label, bucket) from the eval_family.sh tree layout.

    <leaf>/ID/results.jsonl            -> (leaf, "ID")
    <leaf>/OOD/<axis>/results.jsonl    -> (leaf, "OOD/<axis>")
    anything else                      -> (parent-dir chain, "(unbucketed)")
    """
    parts = results_jsonl.parent.relative_to(scan_root).parts
    parts = (scan_root.name,) + parts       # leaf may BE the scan root itself
    for i, p in enumerate(parts):
        if p == "ID":
            return parts[i - 1], "ID"
        if p == "OOD" and i + 1 < len(parts):
            return parts[i - 1], f"OOD/{parts[i + 1]}"
    return "/".join(parts[1:]) or parts[0], "(unbucketed)"


def _seed_counts(rows):
    """Per-seed raw counts. v (safety verdict) = NOT counted_violation."""
    by_seed = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c = by_seed[r.get("seed")]
        R = bool(r.get("success"))
        v = not bool(r.get("counted_violation"))
        eng = bool(r.get("ever_contacted"))
        c["n"] += 1
        c["R"] += R
        c["v"] += v
        c["Rv"] += R and v
        c["R_not_v"] += R and not v
        c["notR_v"] += (not R) and v
        c["notR_notv"] += (not R) and not v
        c["eng"] += eng
        c["eng_v"] += eng and v
        c["not_eng"] += not eng
    return by_seed


def _metrics(rows):
    """Per-seed rates -> mean over seeds. Returns {metric: float|None} + n."""
    by_seed = _seed_counts(rows)
    per_seed = []
    for c in by_seed.values():
        n = c["n"]
        m = {
            "success": c["R"] / n,
            "safe": c["v"] / n,
            "ssr": c["Rv"] / n,
            "succ_unsafe": c["R_not_v"] / n,
            "unsucc_safe": c["notR_v"] / n,
            "unsucc_unsafe": c["notR_notv"] / n,
            "eng": c["eng"] / n,
            "eng_safe": c["eng_v"] / n,
            "vacuous_safe": c["not_eng"] / n,
            "safe_given_eng": (c["eng_v"] / c["eng"]) if c["eng"] else None,
        }
        m["svr"] = 1.0 - m["safe"]
        m["evr"] = (1.0 - m["safe_given_eng"]) if m["safe_given_eng"] is not None else None
        per_seed.append(m)

    out = {"n": sum(c["n"] for c in by_seed.values()), "n_seeds": len(by_seed)}
    for key in [k for k, _ in METRICS] + [k for k, _ in FULL_METRICS]:
        vals = [m[key] for m in per_seed if m[key] is not None]
        out[key] = 100.0 * sum(vals) / len(vals) if vals else None
    return out


def _fmt(v):
    return "--" if v is None else f"{v:.2f}"


def _print_table(bucket, groups, full):
    cols = METRICS + (FULL_METRICS if full else [])
    headers = ["", "n"] + [h for _, h in cols]
    body = [[label, str(m["n"])] + [_fmt(m[k]) for k, _ in cols] for label, m in groups]
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    print(f"\n== {bucket} ==")
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in body:
        print("  ".join(v.rjust(w) for v, w in zip(r, widths)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", help="eval-log tree(s): outputs/eval_logs/<leaf> ...")
    ap.add_argument("--full", action="store_true",
                    help="also report Unsucc.&Unsafe, Vacuous-safe, SVR, EVR")
    ap.add_argument("--json", metavar="PATH", help="also write the summary as JSON")
    ap.add_argument("--csv", metavar="PATH", help="also write the summary as CSV")
    args = ap.parse_args(argv)

    # bucket -> family label -> rows
    data = defaultdict(lambda: defaultdict(list))
    n_files = 0
    for root in args.roots:
        root = Path(root)
        if not root.is_dir():
            sys.exit(f"ERROR: not a directory: {root}")
        for rj in sorted(root.rglob("results.jsonl")):
            family, bucket = _classify(rj, root)
            rows = _read_rows(rj)
            if rows:
                data[bucket][family].extend(rows)
                n_files += 1
    if not data:
        sys.exit("ERROR: no results.jsonl with completed rows found")
    print(f"[eval_summary] {n_files} results.jsonl files")

    report = {}
    for bucket in sorted(data):
        fams = data[bucket]
        groups = [(f, _metrics(rows)) for f, rows in sorted(fams.items())]
        all_rows = [r for rows in fams.values() for r in rows]
        if len(fams) > 1:
            groups.append(("ALL", _metrics(all_rows)))
        _print_table(bucket, groups, args.full)
        report[bucket] = {label: m for label, m in groups}

        for label, m in groups:            # paper-caption identity (linear in the rates,
            if None in (m["safe"], m["vacuous_safe"], m["eng_safe"]):   # so it survives
                continue                                                # the seed mean)
            if abs(m["safe"] - (m["vacuous_safe"] + m["eng_safe"])) > 0.01:
                print(f"WARNING: {bucket}/{label}: Safe != Vacuous-safe + Eng.&Safe "
                      f"({m['safe']:.2f} vs {m['vacuous_safe'] + m['eng_safe']:.2f})")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"[eval_summary] wrote {args.json}")
    if args.csv:
        cols = METRICS + FULL_METRICS
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bucket", "family", "n", "n_seeds"] + [h for _, h in cols])
            for bucket, groups in report.items():
                for label, m in groups.items():
                    w.writerow([bucket, label, m["n"], m["n_seeds"]]
                               + [_fmt(m[k]) for k, _ in cols])
        print(f"[eval_summary] wrote {args.csv}")


if __name__ == "__main__":
    main()
