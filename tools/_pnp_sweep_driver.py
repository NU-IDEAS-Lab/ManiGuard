#!/usr/bin/env python3
"""Multi-task sweep driver for pick_and_place_from_dataset.

Manifest:
  - task_0000  -> quota 50
  - all other clutter_pickup tasks with a valid schema -> quota 5
  - skipped: task_0032 (no diagnostics.jsonl), task_0058 (target not in init),
    task_0059 (no diagnostics.jsonl)

For each task: sweep seeds 0..MAX_SEEDS, run the PnP script, record outcome.
Stop when quota met or seed budget exhausted.

State is persisted to ``<COLLECT_ROOT>/progress.json`` after every run, so
the driver is resumable: re-run and it skips tasks whose quota is met and
seeds already attempted.

Per-run output: ``<COLLECT_ROOT>/<task_id>/seed_XX/`` with ``result.json``,
``trajectory.pt`` (when PLAN succeeded), ``run.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path("/data/Projects/SENTINEL-Lite/datasets/6fam-base-20260513/clutter_pickup")
SCRIPT_ROOT = Path("/data/Projects/SENTINEL-Lite")

SKIP_TASKS = {"task_0032", "task_0058", "task_0059"}

DEFAULT_QUOTA_T0 = 50
DEFAULT_QUOTA_REST = 5
DEFAULT_MAX_SEEDS = 30          # seed budget per task
DEFAULT_TIMEOUT_S = 1200        # per-seed wall-clock cap (subprocess timeout)
DEFAULT_EARLY_BAIL_STREAK = 8   # consecutive failures w/ 0 successes → abandon task


def task_quota(task_id: str, q_t0: int, q_rest: int) -> int:
    if task_id == "task_0000":
        return q_t0
    return q_rest


def list_tasks() -> list[str]:
    out = []
    for d in sorted(DATA_ROOT.glob("task_*")):
        if d.name in SKIP_TASKS:
            continue
        if not (d / "base" / "diagnostics.jsonl").is_file():
            continue
        # symlink scene_ep1.json -> scene_ep1_replay.json if needed
        base = d / "base"
        sf = base / "scene_ep1.json"
        if not sf.exists() and (base / "scene_ep1_replay.json").exists():
            try:
                os.symlink("scene_ep1_replay.json", sf)
            except FileExistsError:
                pass
        out.append(d.name)
    return out


def seed_dir(collect_root: Path, task_id: str, seed: int) -> Path:
    return collect_root / task_id / f"seed_{seed:02d}"


def already_attempted(collect_root: Path, task_id: str, seed: int) -> tuple[bool, bool]:
    """Return (was_attempted, was_success). Treat a missing result.json as
    not-attempted (we'll re-run)."""
    sd = seed_dir(collect_root, task_id, seed)
    rj = sd / "result.json"
    if not rj.is_file():
        return False, False
    try:
        r = json.loads(rj.read_text())
        return True, bool(r.get("phase_b", {}).get("success"))
    except Exception:  # noqa: BLE001
        return False, False


def count_successes(collect_root: Path, task_id: str) -> int:
    td = collect_root / task_id
    if not td.is_dir():
        return 0
    n = 0
    for sd in td.glob("seed_*"):
        rj = sd / "result.json"
        if not rj.is_file():
            continue
        try:
            r = json.loads(rj.read_text())
            if r.get("phase_b", {}).get("success"):
                n += 1
        except Exception:  # noqa: BLE001
            pass
    return n


def run_one(task_dir: Path, out_dir: Path, seed: int, timeout_s: int,
            max_candidates: int, pick_timeout: float, transport_timeout: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    env = os.environ.copy()
    env.update({
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMNIGIBSON_HEADLESS": "1",
    })
    cmd = [
        "conda", "run", "-n", "behavior", "--no-capture-output",
        "python", "-m", "tools.pick_and_place_from_dataset",
        "--task-dir", str(task_dir),
        "--episode", "1",
        "--max-candidates", str(max_candidates),
        "--pick-timeout", str(pick_timeout),
        "--transport-timeout", str(transport_timeout),
        "--ik-precheck",
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    t0 = time.time()
    with log_path.open("w") as logf:
        try:
            rc = subprocess.run(
                cmd, cwd=str(SCRIPT_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout_s,
            ).returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = -1
            timed_out = True
    wall = time.time() - t0
    return rc, wall, timed_out


def write_progress(progress_path: Path, state: dict):
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(progress_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect-root", type=Path,
                   default=Path("/data/Projects/SENTINEL-Lite/outputs/pnp_multitask_collect"))
    p.add_argument("--quota-t0", type=int, default=DEFAULT_QUOTA_T0)
    p.add_argument("--quota-rest", type=int, default=DEFAULT_QUOTA_REST)
    p.add_argument("--max-seeds", type=int, default=DEFAULT_MAX_SEEDS)
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--pick-timeout", type=float, default=600.0)
    p.add_argument("--transport-timeout", type=float, default=60.0)
    p.add_argument("--per-run-timeout", type=int, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--early-bail-streak", type=int,
                   default=DEFAULT_EARLY_BAIL_STREAK,
                   help="Abandon a task after this many consecutive failures "
                        "when zero successes have been recorded for it.")
    p.add_argument("--seed-collect-t0", type=Path,
                   default=Path("/data/Projects/SENTINEL-Lite/outputs/pnp_collect_10"),
                   help="Existing task_0000 successes to seed the collection.")
    args = p.parse_args()

    collect_root = args.collect_root
    collect_root.mkdir(parents=True, exist_ok=True)
    progress_path = collect_root / "progress.json"

    # Seed task_0000 from the prior collect_10 sweep if not already there.
    t0_dir = collect_root / "task_0000"
    if not t0_dir.exists() and args.seed_collect_t0.is_dir():
        t0_dir.mkdir(parents=True, exist_ok=True)
        for sd in args.seed_collect_t0.glob("seed_*"):
            dst = t0_dir / sd.name
            if not dst.exists():
                # Just hardlink the result.json + trajectory.pt + run.log
                dst.mkdir(parents=True, exist_ok=True)
                for f in sd.iterdir():
                    if f.is_file():
                        try:
                            os.link(f, dst / f.name)
                        except (FileExistsError, OSError):
                            pass
        print(f"[driver] seeded task_0000 from {args.seed_collect_t0}", flush=True)

    tasks = list_tasks()
    print(f"[driver] tasks to process: {len(tasks)}  "
          f"(skipping {sorted(SKIP_TASKS)})", flush=True)
    print(f"[driver] quotas: task_0000={args.quota_t0}  others={args.quota_rest}",
          flush=True)
    print(f"[driver] max_seeds/task={args.max_seeds}  "
          f"per_run_timeout={args.per_run_timeout}s", flush=True)

    state: dict = {
        "start_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {},
        "config": {
            "quota_t0": args.quota_t0, "quota_rest": args.quota_rest,
            "max_seeds": args.max_seeds, "per_run_timeout": args.per_run_timeout,
        },
    }
    if progress_path.exists():
        try:
            state = json.loads(progress_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    t_start = time.time()
    for ti, task_id in enumerate(tasks):
        quota = task_quota(task_id, args.quota_t0, args.quota_rest)
        task_dir = DATA_ROOT / task_id / "base"

        n_ok = count_successes(collect_root, task_id)
        state.setdefault("tasks", {})[task_id] = {
            "quota": quota, "successes": n_ok, "attempts": 0, "done": n_ok >= quota,
        }
        write_progress(progress_path, state)

        if n_ok >= quota:
            print(f"[driver] [{ti+1}/{len(tasks)}] {task_id}: quota met "
                  f"({n_ok}/{quota}) — skipping", flush=True)
            continue

        print(f"[driver] [{ti+1}/{len(tasks)}] {task_id}: starting "
              f"({n_ok}/{quota} done, budget {args.max_seeds} seeds, "
              f"early_bail_streak={args.early_bail_streak})",
              flush=True)
        fail_streak = 0  # consecutive failures since last success
        for seed in range(args.max_seeds):
            if n_ok >= quota:
                break
            # Early-bail: if no successes yet for this task and we have a
            # long failure streak, declare the task unsolvable and skip.
            if n_ok == 0 and fail_streak >= args.early_bail_streak:
                print(f"[driver]   {task_id}: EARLY BAIL after "
                      f"{fail_streak} consecutive failures (0 successes) "
                      f"— moving on", flush=True)
                state["tasks"][task_id]["early_bailed"] = True
                state["tasks"][task_id]["bail_streak"] = fail_streak
                write_progress(progress_path, state)
                break
            already_done, already_ok = already_attempted(collect_root, task_id, seed)
            if already_done:
                if already_ok:
                    n_ok = count_successes(collect_root, task_id)
                    fail_streak = 0
                else:
                    fail_streak += 1
                continue
            out_dir = seed_dir(collect_root, task_id, seed)
            rc, wall, timed_out = run_one(
                task_dir, out_dir, seed,
                timeout_s=args.per_run_timeout,
                max_candidates=args.max_candidates,
                pick_timeout=args.pick_timeout,
                transport_timeout=args.transport_timeout,
            )
            # Re-tally from result.json (more authoritative than rc).
            rj = out_dir / "result.json"
            ok = False; fail_step = "-"
            if rj.is_file():
                try:
                    r = json.loads(rj.read_text())
                    ok = bool(r.get("phase_b", {}).get("success"))
                    fail_step = str(r.get("fail_step", "-"))
                except Exception as e:  # noqa: BLE001
                    fail_step = f"parse_err:{e}"
            else:
                fail_step = "no_result_json"
                if timed_out:
                    fail_step = "timeout"
            if ok:
                n_ok += 1
                fail_streak = 0
            else:
                fail_streak += 1
            print(f"[driver]   {task_id} seed={seed}  ok={ok}  "
                  f"fail_step={fail_step}  wall={wall:.0f}s  "
                  f"running={n_ok}/{quota}  fail_streak={fail_streak}  "
                  f"(rc={rc} timed_out={timed_out})",
                  flush=True)

            # Per-step progress update.
            state["tasks"][task_id]["successes"] = n_ok
            state["tasks"][task_id]["attempts"] = seed + 1
            state["tasks"][task_id]["done"] = n_ok >= quota
            state["wall_so_far_s"] = time.time() - t_start
            write_progress(progress_path, state)

        if n_ok < quota:
            print(f"[driver] [{ti+1}/{len(tasks)}] {task_id}: BUDGET EXHAUSTED "
                  f"({n_ok}/{quota} after {args.max_seeds} seeds)", flush=True)

    state["finish_wall"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["wall_total_s"] = time.time() - t_start
    write_progress(progress_path, state)
    total_ok = sum(t.get("successes", 0) for t in state["tasks"].values())
    total_q = sum(t.get("quota", 0) for t in state["tasks"].values())
    print(f"[driver] DONE  total successes: {total_ok}/{total_q}  "
          f"wall={state['wall_total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
