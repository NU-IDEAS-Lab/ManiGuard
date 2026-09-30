#!/usr/bin/env python3
"""Summarize completed evaluation rows by family, bucket, and seed.

Read results.jsonl files beneath the supplied roots. Compute success, safety,
joint success/safety, engagement, and conditional safety from success,
counted_violation, and ever_contacted. Average per-seed rates equally; conditional
rates omit seeds with no engaged rollouts. An ALL row pools families within each
seed before averaging.

Failed, malformed, and unplanned records are rejected before reporting rates.
Expected coverage comes from expected_rollouts.json files written by the family
runner, or --expected-manifest. Historical logs without a plan require explicit
--allow-unverified; this does not suppress invalid-row or duplicate checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
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


def _validate_row(row, source):
    if not isinstance(row, dict):
        raise ValueError(f"{source}: expected a JSON object")
    if row.get("status") != "completed":
        raise ValueError(f"{source}: unresolved result status {row.get('status')!r}")
    if not isinstance(row.get("scene_name"), str) or not row["scene_name"].strip():
        raise ValueError(f"{source}: missing scene_name")
    if "seed" not in row or (row["seed"] is not None and type(row["seed"]) is not int):
        raise ValueError(f"{source}: seed must be an integer or null")
    for key in ("success", "counted_violation", "ever_contacted"):
        if type(row.get(key)) is not bool:
            raise ValueError(f"{source}: {key} must be a Boolean, not missing/null")
    if row.get("ltl_monitored") is not True:
        raise ValueError(f"{source}: ltl_monitored must be true to summarize safety")
    if row["counted_violation"] and not row["ever_contacted"]:
        raise ValueError(f"{source}: counted_violation requires ever_contacted")


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON value {value}")


def _read_rows(results_jsonl: Path):
    rows = []
    for line_number, line in enumerate(results_jsonl.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line, parse_constant=_reject_nonfinite)
        except ValueError as exc:
            raise ValueError(f"{results_jsonl}:{line_number}: invalid JSON: {exc}") from exc
        _validate_row(r, f"{results_jsonl}:{line_number}")
        rows.append(r)
    return rows


def _expected_rows(manifests):
    expected = Counter()
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path}: cannot read expected manifest: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError(f"{path}: expected schema_version 1")
        records = manifest.get("rollouts")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}: expected a nonempty rollouts list")
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: rollout {index} must be an object")
            file = row.get("results_file")
            scene = row.get("scene_name")
            if not isinstance(file, str) or not file or Path(file).is_absolute():
                raise ValueError(f"{path}: rollout {index} requires a relative results_file")
            target = (path.parent / file).resolve()
            if path.parent.resolve() not in target.parents or target.name != "results.jsonl":
                raise ValueError(f"{path}: invalid results_file {file!r}")
            if not isinstance(scene, str) or not scene.strip():
                raise ValueError(f"{path}: rollout {index} requires scene_name")
            if "seed" not in row or (row["seed"] is not None and type(row["seed"]) is not int):
                raise ValueError(f"{path}: rollout {index} requires integer/null seed")
            expected[(target, scene, row["seed"])] += 1
    return expected


def _check_coverage(expected, actual):
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        messages = []
        for label, counts in [("missing", missing), ("unexpected", extra)]:
            if counts:
                examples = [f"{file}: {scene}, seed={seed}, count={count}"
                            for (file, scene, seed), count in list(counts.items())[:5]]
                messages.append(f"{label} {sum(counts.values())} rollout(s): " + "; ".join(examples))
        raise ValueError("Expected rollout coverage mismatch: " + " | ".join(messages))


def _classify(results_jsonl: Path, scan_root: Path):
    """(family_label, bucket) from the eval_family.sh tree layout.

    <leaf>/ID/results.jsonl            -> (leaf, "ID")
    <leaf>/OOD/<axis>/results.jsonl    -> (leaf, "OOD/<axis>")
    anything else                      -> (parent-dir chain, "(unbucketed)")
    """
    parts = results_jsonl.parent.relative_to(scan_root).parts
    parts = (scan_root.name,) + parts       # leaf may BE the scan root itself
    def family_before(index):
        label = parts[index - 1]
        if re.fullmatch(r"seed[_-]?\d+", label):
            return parts[index - 2] if index >= 2 else scan_root.parent.name
        return label

    for i, p in enumerate(parts):
        if p == "ID":
            return family_before(i), "ID"
        if p == "OOD" and i + 1 < len(parts):
            return family_before(i), f"OOD/{parts[i + 1]}"
    return "/".join(parts[1:]) or parts[0], "(unbucketed)"


def _seed_counts(rows):
    """Per-seed raw counts. v (safety verdict) = NOT counted_violation."""
    by_seed = defaultdict(lambda: defaultdict(int))
    for r in rows:
        _validate_row(r, "metric input")
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
    ap.add_argument("--expected-manifest", action="append", default=[], metavar="PATH",
                    help="Expected rollout manifest; otherwise discover expected_rollouts.json under roots")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="Inspect historical logs without a manifest; coverage is not verified")
    ap.add_argument("--expected-seeds", nargs="+", type=int,
                    help="Require exactly these seeds in every family/condition, including entirely absent runs")
    args = ap.parse_args(argv)

    # bucket -> family label -> rows
    data = defaultdict(lambda: defaultdict(list))
    n_files = 0
    roots = [Path(root).resolve() for root in args.roots]
    for i, root in enumerate(roots):
        if not root.is_dir():
            sys.exit(f"ERROR: not a directory: {root}")
        if any(root == other or root in other.parents or other in root.parents for other in roots[:i]):
            sys.exit(f"ERROR: overlapping input roots: {root}")
    manifests = ([Path(p).resolve() for p in args.expected_manifest] if args.expected_manifest else
                 [p.resolve() for root in roots for p in root.rglob("expected_rollouts.json")])
    if len(manifests) != len(set(manifests)):
        sys.exit("ERROR: duplicate expected manifests")
    if not manifests and not args.allow_unverified:
        sys.exit("ERROR: no expected_rollouts.json found; supply --expected-manifest or explicitly "
                 "use --allow-unverified for historical log inspection")
    actual = Counter()
    identities = set()
    try:
        expected = _expected_rows(manifests)
        for root in roots:
            for rj in sorted(root.rglob("results.jsonl")):
                family, bucket = _classify(rj, root)
                rows = _read_rows(rj)
                for row in rows:
                    actual[(rj.resolve(), row["scene_name"], row["seed"])] += 1
                    identity = (family, bucket, row["scene_name"], row["seed"])
                    if not manifests and identity in identities:
                        raise ValueError(f"{rj}: duplicate rollout {identity}")
                    identities.add(identity)
                if rows:
                    data[bucket][family].extend(rows)
                    n_files += 1
        if manifests:
            _check_coverage(expected, actual)
        else:
            print("WARNING: expected scenario/seed coverage is not verified (no manifest).", file=sys.stderr)
        if args.expected_seeds is not None:
            required = set(args.expected_seeds)
            for bucket, families in data.items():
                for family, rows in families.items():
                    observed = {r["seed"] for r in rows}
                    if observed != required:
                        raise ValueError(
                            f"{family}/{bucket}: seed coverage mismatch; "
                            f"missing seeds {sorted(required - observed)}, "
                            f"unexpected seeds {sorted(observed - required, key=str)}"
                        )
    except (OSError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
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
