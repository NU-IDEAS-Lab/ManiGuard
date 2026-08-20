"""Re-finalize the `base/` of cabinet tasks whose spawn was edited in T1.

T1 (both-front / near-edge / role-swap / target-mode / drawer-fix) rewrote the
`base/scene_ep1.json` + `base/diagnostics.jsonl` of 17 cabinet tasks, but their
4 `base/rollout_*.mp4` still show the OLD layout AND their diagnostics runtime
stats (`gate_pass` / `ltl_violated` / `steps_executed` / `surface_info` / the
`bench` stability block) are STALE — the spawn editor carried them verbatim
from the pre-edit diagnostics instead of recomputing them on the new layout.

This tool re-runs the EXACT bench-build pipeline on each edited task so the
renewed instance is byte-for-byte how the dataset was originally produced:
``finalize_base.finalize_base_task`` builds the env from the (edited) snapshot,
bakes the canonical mounted robot + init pose, idle-steps under gravity with the
arm held by the stiff Isaac drive, and in that SAME idle-step re-renders the 4
review videos, re-stamps `cameras`, steps a fresh LTL monitor, and recomputes
`gate_pass` / `surface_info` / the `bench` stats — while carrying every edited
task-identity field (target_info / obstacle_info / blocker_mode / goal_conditions
/ ltl_safety / prompt / selection) through its allowlist. So unlike a render-only
pass it refreshes the videos AND the diagnostics together, consistently.

Why not ``run_finalize_base``: that DRIVER points ``src`` at the READ-ONLY
6fam-base source (which lacks the T1 edits) and would wipe them. We call the
underlying ``finalize_base_task`` directly with ``src`` = the EDITED base dir, so
it finalizes FROM the edits. ``finalize_base_task`` never writes to ``src`` — it
writes only to ``out`` — so we point ``out`` at a per-task temp dir and the
parent process (no OmniGibson, immune to the teardown segfault) copies the 6
outputs back over ``base/`` only after verifying they are complete.

Same fresh-subprocess-per-task pattern as the perturb drivers (OmniGibson can't
cleanly reload Kit in-process and may segfault on teardown after a clean write;
success is judged by output presence, not exit code).

Usage:
  # one task (validation)
  python -m tools.bench_surgery.cabinet.rerender_base --tasks task_0034
  # the 17 modified tasks (default), single GPU process
  python -m tools.bench_surgery.cabinet.rerender_base
  # explicit subset, modest parallel fan-out
  python -m tools.bench_surgery.cabinet.rerender_base --tasks task_0001,task_0003 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
FAMILY = "cabinet_pickup"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")
ROW_FILE = "_rerender_row.json"

# The 17 tasks T1 edited (each has a *.bak_bothfront / *.bak_roleswap backup).
MODIFIED_TASKS = [
    "task_0001", "task_0002", "task_0003", "task_0004", "task_0007", "task_0012",
    "task_0013", "task_0015", "task_0018", "task_0020", "task_0021", "task_0022",
    "task_0023", "task_0026", "task_0028", "task_0031", "task_0034",
]


# ---------------------------------------------------------------------------- helpers

def _tmp_dir(base_dir: Path) -> Path:
    return base_dir.parent / "_rerender_tmp"  # per-task (parent = task_XXXX) -> parallel-safe


def _complete(d: Path, episode: int) -> bool:
    return (
        (d / f"scene_ep{episode}.json").is_file()
        and (d / "diagnostics.jsonl").is_file()
        and all((d / f"rollout_{lbl}_ep{episode}.mp4").is_file() for lbl in VIDEO_LABELS)
    )


def _worker_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OMNIGIBSON_HEADLESS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


# ---------------------------------------------------------------------------- worker

def _run_worker(base_dir: Path, episode: int, src_base: Path | None = None) -> None:
    """Re-finalize ONE base into a temp dir (the parent does the copy-back). ``src_base``
    (a regenerated 6fam-style base/ in a scratch dir) overrides the finalize SOURCE — used
    to swap a task's content (surface/target) by regenerating it and finalizing it INTO the
    bench `base_dir`. Default (None) re-finalizes the bench base in place."""
    from maniguard.data.bench_builder.finalize_base import finalize_base_task

    src = src_base or base_dir
    tmp = _tmp_dir(base_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        row = finalize_base_task(src, tmp, family=FAMILY, episode=episode)
    except Exception as e:  # noqa: BLE001
        import traceback
        row = {"task": base_dir.parent.name, "status": "fail",
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    (tmp / ROW_FILE).write_text(json.dumps(row, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------- driver

def _spawn_worker(base_dir: Path, episode: int, env: dict, timeout: int, src_base: Path | None = None):
    cmd = [
        sys.executable, "-m", "tools.bench_surgery.cabinet.rerender_base",
        "--worker", "--base-dir", str(base_dir), "--episode", str(episode),
    ]
    if src_base is not None:
        cmd += ["--src-base", str(src_base)]
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return "timeout"


def _commit_back(base_dir: Path, episode: int) -> tuple[bool, str]:
    """After a successful worker: copy the 6 finalized outputs over base/ (preserving the
    *.bak_* backups + any siblings), then drop the temp dir. Returns (ok, status_str)."""
    tmp = _tmp_dir(base_dir)
    if not tmp.is_dir():
        return False, "no-tmp"
    row = {}
    rp = tmp / ROW_FILE
    if rp.is_file():
        row = json.loads(rp.read_text(encoding="utf-8"))
    if row.get("status") == "fail":
        shutil.rmtree(tmp, ignore_errors=True)
        return False, f"finalize-fail({row.get('error')})"
    if not _complete(tmp, episode):
        shutil.rmtree(tmp, ignore_errors=True)
        return False, "incomplete-output"
    for name in ([f"scene_ep{episode}.json", "diagnostics.jsonl"]
                 + [f"rollout_{lbl}_ep{episode}.mp4" for lbl in VIDEO_LABELS]):
        shutil.copy2(tmp / name, base_dir / name)
    shutil.rmtree(tmp, ignore_errors=True)
    return True, f"ok(status={row.get('status')})"


def _driver(args: argparse.Namespace) -> int:
    out_fam = Path(args.bench_root) / FAMILY
    tasks = (
        [t.strip() if t.strip().startswith("task_") else f"task_{int(t):04d}"
         for t in args.tasks.split(",")]
        if args.tasks else list(MODIFIED_TASKS)
    )
    env = _worker_env()
    results: dict[str, str] = {}

    src_root = Path(args.src_root) if args.src_root else None

    def work(task: str) -> tuple[str, str]:
        base_dir = out_fam / task / "base"
        src_base = (src_root / task / "base") if src_root else None
        check = src_base or base_dir
        if not (check / f"scene_ep{args.episode}.json").is_file():
            return task, ("missing-src" if src_base else "missing-base")
        rc = _spawn_worker(base_dir, args.episode, env, args.timeout, src_base)
        ok, status = _commit_back(base_dir, args.episode)
        return task, (status if ok else f"{status} [rc={rc}]")

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(work, t): t for t in tasks}
            for fut in as_completed(futs):
                task, status = fut.result()
                results[task] = status
                print(f"[rerender] {task}: {status}", flush=True)
    else:
        for t in tasks:
            task, status = work(t)
            results[task] = status
            print(f"[rerender] {task}: {status}", flush=True)

    n_ok = sum(1 for s in results.values() if s.startswith("ok"))
    print(f"\n[rerender] {n_ok}/{len(tasks)} base re-finalized OK")
    for t in tasks:
        if not results.get(t, "").startswith("ok"):
            print(f"  FAILED: {t} -> {results.get(t)}")
    return 0 if n_ok == len(tasks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="internal: re-finalize one base dir")
    ap.add_argument("--base-dir", type=str, help="worker: the task's bench base/ dir (finalize DEST)")
    ap.add_argument("--src-base", type=str, default=None,
                    help="worker: override finalize SOURCE (a regenerated scratch base/ dir)")
    ap.add_argument("--src-root", type=str, default=None,
                    help="driver: regen-source root; finalize <src-root>/task_NNNN/base -> bench base")
    ap.add_argument("--bench-root", type=str, default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--tasks", type=str, default=None,
                    help="comma list (e.g. task_0034 or 1,3,34); default = the 17 modified tasks")
    ap.add_argument("--jobs", type=int, default=1, help="parallel subprocess fan-out (single GPU; keep modest)")
    ap.add_argument("--timeout", type=int, default=900, help="per-task subprocess timeout (s)")
    ap.add_argument("--episode", type=int, default=1)
    args = ap.parse_args()

    if args.worker:
        _run_worker(Path(args.base_dir), args.episode,
                    Path(args.src_base) if args.src_base else None)
        return 0
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
