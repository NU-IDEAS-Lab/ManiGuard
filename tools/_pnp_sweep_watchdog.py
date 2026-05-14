#!/usr/bin/env python3
"""Watchdog: peek at the multi-task sweep's progress.json and print a
status summary. Safe to run any time; does not touch the running driver.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def fmt_dur(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect-root", type=Path,
                   default=Path("/data/Projects/SENTINEL-Lite/outputs/pnp_multitask_collect"))
    p.add_argument("--driver-log",
                   default="/tmp/pnp_multitask_driver.log",
                   help="Path to the driver's tee'd stdout for tail summary.")
    p.add_argument("--tail", type=int, default=12,
                   help="Lines of driver log tail to show.")
    args = p.parse_args()

    pf = args.collect_root / "progress.json"
    if not pf.is_file():
        print(f"[watchdog] no progress.json at {pf} — driver not started?")
        return 0

    state = json.loads(pf.read_text())
    tasks = state.get("tasks", {})

    n_tasks = len(tasks)
    n_done = sum(1 for t in tasks.values() if t.get("done"))
    n_in_progress = sum(1 for t in tasks.values()
                        if not t.get("done") and t.get("attempts", 0) > 0)
    total_ok = sum(t.get("successes", 0) for t in tasks.values())
    total_q = sum(t.get("quota", 0) for t in tasks.values())
    total_attempts = sum(t.get("attempts", 0) for t in tasks.values())
    wall = state.get("wall_so_far_s") or state.get("wall_total_s") or 0

    finished = "finish_wall" in state
    header = "FINISHED" if finished else "RUNNING"
    print(f"[watchdog] === {header} ===")
    print(f"[watchdog] start={state.get('start_wall', '?')}  "
          f"wall={fmt_dur(wall)}", flush=True)
    print(f"[watchdog] tasks: done={n_done}/{n_tasks}  "
          f"in_progress={n_in_progress}", flush=True)
    rate_pct = (100.0 * total_ok / total_attempts) if total_attempts else float("nan")
    rate_str = f"{rate_pct:.1f}%" if total_attempts else "—"
    print(f"[watchdog] successes: {total_ok}/{total_q}  "
          f"attempts={total_attempts}  global_rate={rate_str}", flush=True)
    if total_ok and total_attempts:
        avg_per_attempt = wall / max(total_attempts, 1)
        remaining_attempts = max(0, (total_q - total_ok) / max(total_ok / total_attempts, 0.1))
        eta = avg_per_attempt * remaining_attempts
        print(f"[watchdog] avg_per_attempt={avg_per_attempt:.0f}s  "
              f"ETA≈{fmt_dur(eta)} ({int(remaining_attempts)} more attempts)",
              flush=True)

    # Per-task table (only the not-yet-done ones plus a tail of finished).
    print(f"[watchdog] per-task progress:")
    for tid in sorted(tasks):
        t = tasks[tid]
        s = t.get("successes", 0); q = t.get("quota", 0)
        a = t.get("attempts", 0); d = t.get("done", False)
        mark = "✓" if d else (" " if a == 0 else "·")
        # Show task_0000 always, then unfinished, then last 3 finished.
        is_t0 = tid == "task_0000"
        if not (is_t0 or not d or a > 0):
            continue
        print(f"[watchdog]   {mark} {tid:11s}  {s:>2d}/{q:<2d}  "
              f"attempts={a}", flush=True)

    # Driver log tail.
    if Path(args.driver_log).is_file():
        print(f"[watchdog] driver log tail ({args.driver_log}):")
        with open(args.driver_log) as f:
            lines = f.readlines()
        for line in lines[-args.tail:]:
            print(f"[watchdog]   {line.rstrip()}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
