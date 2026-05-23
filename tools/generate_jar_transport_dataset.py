"""Batch driver for jar_transport_pipeline.

One subprocess per task (env init is heavy; we don't want a single
crashed task to take the whole dataset down). Each task gets a distinct
seed, which drives the (jar model, item category, item model) and
surface picks. Output layout matches the 6fam convention:

    datasets/jar_transport-base-<YYYYMMDD>/
        task_0000/
            stdout.log
            base/
                diagnostics.jsonl
                scene_ep1.json
                rollout_opposite_side_front_ep1.mp4
                rollout_left_overview_ep1.mp4
                rollout_right_overview_ep1.mp4
                rollout_left_shoulder_ep1.mp4
                snapshots/cam_*.png
        ...
        batch_summary.tsv

Example::

    python -m tools.generate_jar_transport_dataset --num-tasks 50
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_MODULE = "sentinel.task_generation.jar_transport_pipeline"


def _build_env(args):
    env = os.environ.copy()
    env.setdefault("VK_ICD_FILENAMES",
                   "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("OMNIGIBSON_HEADLESS", "1")
    return env


def _build_cmd(args, task_id, seed):
    return [
        sys.executable, "-m", _PIPELINE_MODULE,
        "--headless",
        "--seed", str(seed),
        "--steps", str(args.steps),
        "--save-video",
        "--task-id", str(task_id),
        "--tasks-out-dir", str(args.tasks_out_dir),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-tasks", type=int, default=50)
    p.add_argument("--start-task-id", type=int, default=0)
    p.add_argument("--steps", type=int, default=300,
                   help="Per-task LTL rollout step count.")
    p.add_argument("--tasks-out-dir", default=None,
                   help="Defaults to datasets/jar_transport-base-<YYYYMMDD>/")
    p.add_argument("--task-timeout", type=int, default=600,
                   help="Per-task subprocess timeout (seconds).")
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Skip failing tasks and continue (default).")
    args = p.parse_args()

    if args.tasks_out_dir is None:
        today = datetime.now().strftime("%Y%m%d")
        args.tasks_out_dir = str(
            _PROJECT_ROOT / "datasets" / f"jar_transport-base-{today}"
        )
    Path(args.tasks_out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[Batch] Output: {args.tasks_out_dir}")
    print(f"[Batch] Generating {args.num_tasks} tasks starting at "
          f"task_id={args.start_task_id}")

    env = _build_env(args)
    summary_path = Path(args.tasks_out_dir) / "batch_summary.tsv"
    with summary_path.open("w") as fh:
        fh.write("task_id\tseed\tstatus\twall_s\n")
        fh.flush()

        ok = 0
        fail = 0
        t_start = time.time()
        for i in range(args.num_tasks):
            task_id = args.start_task_id + i
            seed = task_id
            cmd = _build_cmd(args, task_id, seed)
            t0 = time.time()
            print(f"\n[Batch] task_{task_id:04d}  seed={seed}", flush=True)
            log_path = (Path(args.tasks_out_dir)
                        / f"task_{task_id:04d}" / "stdout.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("w") as logfh:
                    rc = subprocess.run(
                        cmd, env=env, cwd=str(_PROJECT_ROOT),
                        stdout=logfh, stderr=subprocess.STDOUT,
                        timeout=args.task_timeout,
                    ).returncode
                wall = time.time() - t0
                status = "ok" if rc == 0 else f"rc={rc}"
                if rc == 0:
                    ok += 1
                else:
                    fail += 1
                    if not args.continue_on_error:
                        fh.write(f"{task_id}\t{seed}\t{status}\t{wall:.1f}\n")
                        fh.flush()
                        return 1
            except subprocess.TimeoutExpired:
                wall = time.time() - t0
                status = "timeout"
                fail += 1
            print(f"[Batch] task_{task_id:04d} {status} ({wall:.1f}s)  "
                  f"ok={ok} fail={fail} "
                  f"elapsed={time.time() - t_start:.0f}s",
                  flush=True)
            fh.write(f"{task_id}\t{seed}\t{status}\t{wall:.1f}\n")
            fh.flush()

    print(f"\n[Batch] Done: ok={ok}, fail={fail}, "
          f"total_wall={time.time() - t_start:.0f}s")
    print(f"[Batch] Summary: {summary_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
