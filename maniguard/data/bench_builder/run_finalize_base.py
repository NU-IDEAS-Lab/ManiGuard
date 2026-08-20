"""Batch driver to finalize 6fam-base tasks into ManiGuard-Bench (one fresh process per task).

OmniGibson cannot cleanly reload Kit within one process (and commonly segfaults during shutdown
*after* a clean save), so each task is finalized in a FRESH subprocess and success is judged by
output-file presence, not the worker's exit code — same proven pattern as
``replay_empty_from_dataset._run_subprocess_per_task``.

Two modes (one module):
  * WORKER  (``--worker``): finalize ONE task, write its manifest row to ``<out-base>/_finalize_row.json``
    BEFORE the teardown segfault, exit. (No validation here — the driver validates offline.)
  * DRIVER  (default): for the selected tasks, spawn workers (``--jobs`` in parallel), then run the
    OFFLINE ``validate_base_task`` in this (clean, no-OmniGibson) process and write per-task rows to
    ``<out-root>/<family>/base_manifest.jsonl``. Read-only on 6fam-base; writes only maniguard-bench.

Usage:
  # finalize jar tasks 0 and 1 with 2 parallel workers (the P1.0 self-check)
  python -m maniguard.data.bench_builder.run_finalize_base --family jar_transport --tasks 0-1 --jobs 2
  # full family, resumable
  python -m maniguard.data.bench_builder.run_finalize_base --family clutter_pickup --jobs 2 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC_ROOT_DEFAULT = "outputs/lerobot_datasets/6fam-base"
OUT_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
ROW_FILE = "_finalize_row.json"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")


# ---------------------------------------------------------------------------- helpers

def _is_complete(out_base: Path, episode: int) -> bool:
    """Output is a complete finalized task: snapshot + diagnostics + the 4 videos."""
    if not (out_base / f"scene_ep{episode}.json").is_file():
        return False
    if not (out_base / "diagnostics.jsonl").is_file():
        return False
    return all((out_base / f"rollout_{lbl}_ep{episode}.mp4").is_file() for lbl in VIDEO_LABELS)


def _scene_file_in(src_base: Path, episode: int) -> Path | None:
    for name in (f"scene_ep{episode}.json", f"scene_ep{episode}_replay.json"):
        if (src_base / name).is_file():
            return src_base / name
    return None


def _select_tasks(src_fam: Path, spec: str | None, src_subdir: str, episode: int) -> list[str]:
    """Task dirs in src_fam that hold a loadable source base, filtered by --tasks spec."""
    available = sorted(
        d.name for d in src_fam.glob("task_*")
        if d.is_dir() and _scene_file_in(d / src_subdir, episode) is not None
    )
    if not spec:
        return available
    avail = set(available)

    def norm(tok: str) -> str:
        tok = tok.strip()
        return tok if tok.startswith("task_") else f"task_{int(tok):04d}"

    chosen: list[str] = []
    if "," in spec or spec.replace("task_", "").isdigit():
        for tok in spec.split(","):
            t = norm(tok)
            if t in avail:
                chosen.append(t)
    elif "-" in spec:
        lo, hi = spec.split("-", 1)
        for n in range(int(lo), int(hi) + 1):
            t = f"task_{n:04d}"
            if t in avail:
                chosen.append(t)
    else:
        t = norm(spec)
        if t in avail:
            chosen.append(t)
    # preserve sorted order, drop dups
    return [t for t in available if t in set(chosen)]


def _worker_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OMNIGIBSON_HEADLESS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _combine_status(*statuses: str | None) -> str:
    s = set(statuses)
    if "fail" in s:
        return "fail"
    if "warn" in s:
        return "warn"
    return "ok"


# ---------------------------------------------------------------------------- worker

def _run_worker(src_base: Path, out_base: Path, family: str, episode: int) -> None:
    """Finalize ONE task; persist the manifest row before the (possibly segfaulting) teardown."""
    from maniguard.data.bench_builder.finalize_base import finalize_base_task

    out_base.mkdir(parents=True, exist_ok=True)
    try:
        row = finalize_base_task(src_base, out_base, family=family, episode=episode)
    except Exception as e:  # noqa: BLE001 — record the reason so the driver can report it
        import traceback
        row = {
            "task": src_base.parent.name, "family": family, "status": "fail",
            "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
        }
    (out_base / ROW_FILE).write_text(json.dumps(row, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------- driver

def _spawn_worker(src_base: Path, out_base: Path, family: str, episode: int, env: dict, timeout: int):
    cmd = [
        sys.executable, "-m", "maniguard.data.bench_builder.run_finalize_base",
        "--worker", "--src-base", str(src_base), "--out-base", str(out_base),
        "--family", family, "--episode", str(episode),
    ]
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return "timeout"


def _row_for_task(task: str, out_fam: Path, family: str, episode: int) -> dict:
    """Build the manifest row for a finalized task: read the worker's finalize row + offline QC."""
    from maniguard.data.bench_builder.validate_base import validate_base_task

    out_base = out_fam / task / "base"
    row_path = out_base / ROW_FILE
    if not row_path.exists():
        return {"task": task, "family": family, "status": "fail",
                "error": "worker produced no row (crashed before writing)"}
    fin_row = json.loads(row_path.read_text(encoding="utf-8"))
    if not _is_complete(out_base, episode):
        return {"task": task, "family": family, "status": "fail",
                "error": "incomplete output (missing snapshot/videos)", "finalize": fin_row}
    qc = validate_base_task(out_base, family=family, episode=episode)
    return {
        "task": task, "family": family,
        "status": _combine_status(fin_row.get("status"), qc.get("status")),
        "finalize": fin_row, "validate": qc,
    }


def _driver(args: argparse.Namespace) -> int:
    src_fam = Path(args.src_root) / args.family
    out_fam = Path(args.out_root) / args.family
    if not src_fam.is_dir():
        print(f"[finalize] ERROR: source family dir not found: {src_fam}", flush=True)
        return 2
    tasks = _select_tasks(src_fam, args.tasks, args.src_subdir, args.episode)
    if not tasks:
        print(f"[finalize] no matching tasks in {src_fam} (--tasks {args.tasks!r})", flush=True)
        return 1
    out_fam.mkdir(parents=True, exist_ok=True)
    env = _worker_env()

    to_run = [t for t in tasks
              if not (args.skip_existing and _is_complete(out_fam / t / "base", args.episode)
                      and (out_fam / t / "base" / ROW_FILE).exists())]
    skipped = [t for t in tasks if t not in set(to_run)]
    print(f"[finalize] {args.family}: {len(tasks)} tasks ({len(to_run)} to run, {len(skipped)} skip), "
          f"jobs={args.jobs} -> {out_fam}", flush=True)

    rows: list[dict] = []
    manifest = out_fam / "base_manifest.jsonl"
    mf = manifest.open("w", encoding="utf-8")

    def _record(task: str, tag: str, done: int, total: int) -> None:
        row = _row_for_task(task, out_fam, args.family, args.episode)
        rows.append(row)
        mf.write(json.dumps(row, default=float) + "\n")
        mf.flush()
        warn = ""
        if row.get("validate", {}).get("warnings"):
            warn = f"  warn={row['validate']['warnings']}"
        if row.get("status") == "fail":
            warn = f"  {row.get('error') or row.get('validate', {}).get('fails')}"
        print(f"[{tag} {done}/{total}] {task}: {row['status']}{warn}", flush=True)

    for i, t in enumerate(skipped, 1):
        _record(t, "skip", i, len(skipped))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(_spawn_worker, src_fam / t / args.src_subdir, out_fam / t / "base",
                      args.family, args.episode, env, args.timeout): t
            for t in to_run
        }
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            fut.result()  # worker exit code ignored (teardown segfault) — success judged by output
            done += 1
            _record(t, "run", done, len(to_run))

    mf.close()
    counts = Counter(r["status"] for r in rows)
    print(f"=== {args.family}: {dict(counts)} ({len(rows)} tasks) -> {manifest}", flush=True)

    # drop_list.json: written ONLY when there are fails — a CANDIDATE list for human review
    # (NOT an auto-drop). The user reviews each against the manifest + videos: tool bugs get
    # FIXED (then re-validate, not dropped); genuinely-unreasonable tasks get pruned via
    # prune_reindex.py. Deleted clean once pruned (the deterministic checks re-flag bad tasks
    # on any re-run, so the list never needs to persist). Not written when everything passes.
    drop_path = out_fam / "drop_list.json"
    fail_rows = [r for r in rows if r["status"] == "fail"]
    if fail_rows:
        drop_path.write_text(json.dumps({
            "family": args.family,
            "note": "CANDIDATES for review — not an auto-drop. Fix tool bugs (re-validate); "
                    "prune only genuinely-unreasonable tasks via prune_reindex.py.",
            "candidates": [{
                "task": r["task"],
                "fails": r.get("validate", {}).get("fails") or [],
                "warnings": r.get("validate", {}).get("warnings") or [],
                "error": r.get("error"),
            } for r in fail_rows],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"    FAILED: {[r['task'] for r in fail_rows]}  -> {drop_path}", flush=True)
    elif drop_path.exists():
        drop_path.unlink()  # a prior run's stale candidate list; this run is clean
    return 1 if fail_rows else 0


# ---------------------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description="Finalize 6fam-base tasks into ManiGuard-Bench.")
    ap.add_argument("--worker", action="store_true", help="internal: finalize ONE task in this process")
    ap.add_argument("--src-base", help="worker: source base dir")
    ap.add_argument("--out-base", help="worker: output base dir")
    ap.add_argument("--family", required=True)
    ap.add_argument("--src-root", default=SRC_ROOT_DEFAULT)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--src-subdir", default="base", help="subdir under each task holding the source snapshot")
    ap.add_argument("--tasks", default=None, help="'0-22' range, or 'task_0000,task_0005' / '0,5' list; default all")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker subprocesses (try 2; drop to 1 on CUDA OOM)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="per-task worker timeout (s)")
    args = ap.parse_args()

    if args.worker:
        if not args.src_base or not args.out_base:
            ap.error("--worker requires --src-base and --out-base")
        _run_worker(Path(args.src_base), Path(args.out_base), args.family, args.episode)
        return 0
    return _driver(args)


if __name__ == "__main__":
    sys.exit(main())
