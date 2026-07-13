"""Parallel ``to_lerobot``: shard the family BY TASK, run the UNMODIFIED serial converter
(``to_lerobot.convert``) on each task in a separate process, then merge the per-task LeRobot
datasets into one with ``lerobot_merge``.

The serial converter is single-threaded and video-decode bound (~25 s/episode); on a many-core box
this shards it to ~N× (dusty: 243 min -> 13 min, 18.5×, 2026-07-11). Because each shard runs the
converter UNCHANGED on one task, per-episode output is byte-identical by construction; the merge only
re-offsets the 3 global index columns + rebuilds meta, and is proven byte-identical to a full serial
run (see ``lerobot_diff``). ``--verify`` (default on) self-checks the merged dataset against the
converter's OWN logic (prompt-table + per-episode indices + LeRobotDataset load + file counts).

Usage:
    python -m maniguard.data.datagen.to_lerobot_parallel \\
        --dataset v1 --family dusty_transfer \\
        --repo-id IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam [--procs N] [--no-verify]

Drop-in replacement for a serial ``to_lerobot`` run; same ``--dataset/--family/--repo-id/--out-root``.
Run from the repo root (so ``python -m`` puts ``maniguard`` on the path in the shard subprocesses).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from maniguard.data.datagen import lerobot_merge, reader


def _shard_one(dataset: str, family: str, task: str, work: str, repo_id: str) -> Path:
    """Convert ONE task via a symlink shard view (keeps the serial converter UNCHANGED, so the
    per-episode output is byte-identical). Returns the shard's LeRobot dataset root."""
    from maniguard.data.datagen import to_lerobot

    work = Path(work)
    sh_ds = work / "sh" / task
    (sh_ds / family).mkdir(parents=True, exist_ok=True)
    link = sh_ds / family / task
    target = (reader.ROOT / dataset / family / task).resolve()
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)

    out = work / "shards" / task
    if out.exists():
        shutil.rmtree(out)
    ds_name = str(sh_ds.relative_to(reader.ROOT))  # a dataset path under reader.ROOT
    to_lerobot.convert(dataset=ds_name, family=family, out_root=str(out), repo_id=repo_id)
    return out / family


def _run_shards_parallel(dataset, family, tasks, work, repo_id, procs) -> list[str]:
    """Spawn one shard subprocess per task (capped at ``procs`` concurrent). Returns failed tasks."""
    pending, running, failed = list(tasks), [], []
    done = 0
    while pending or running:
        while pending and len(running) < procs:
            task = pending.pop(0)
            logf = open(Path(work) / "logs" / f"{task}.log", "w")
            p = subprocess.Popen(
                [sys.executable, "-m", "maniguard.data.datagen.to_lerobot_parallel",
                 "--_shard-one", dataset, family, task, str(work), repo_id],
                stdout=logf, stderr=subprocess.STDOUT,
            )
            running.append((task, p, logf))
        time.sleep(1)
        for item in list(running):
            task, p, logf = item
            if p.poll() is not None:
                logf.close()
                running.remove(item)
                done += 1
                if p.returncode != 0:
                    failed.append(task)
                    print(f"[parallel]   shard FAILED: {task} (see {work}/logs/{task}.log)")
                else:
                    print(f"[parallel]   shard done: {task} ({done}/{len(tasks)})", flush=True)
    return failed


def _verify(merged_root, dataset, family, repo_id) -> list[str]:
    """Self-check the merged dataset against the serial converter's OWN logic. Returns issues."""
    from maniguard.data.datagen import to_lerobot

    issues: list[str] = []
    merged_root = Path(merged_root)
    info = json.load(open(merged_root / "meta" / "info.json"))
    ne = info["total_episodes"]
    ch = info.get("chunks_size", 1000)
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    tdirs = list(reader.iter_traj_dirs(dataset, family))
    metas = [reader.load_meta(d) for d in tdirs]
    exp_prompts, exp_ti = to_lerobot.build_prompt_table(metas)
    merged_tasks = [json.loads(x)["task"] for x in open(merged_root / "meta" / "tasks.jsonl")]
    if exp_prompts != merged_tasks:
        issues.append(f"tasks.jsonl != build_prompt_table ({len(merged_tasks)} vs {len(exp_prompts)})")
    if len(exp_ti) != ne:
        issues.append(f"episode count {ne} != #trajs {len(exp_ti)}")

    gframe = bad = 0
    for e in range(ne):
        c = e // ch
        t = pq.read_table(merged_root / "data" / f"chunk-{c:03d}" / f"episode_{e:06d}.parquet")
        epi = t.column("episode_index").to_numpy()
        ti = t.column("task_index").to_numpy()
        idx = t.column("index").to_numpy()
        length = len(idx)
        exp_t = exp_ti[e] if e < len(exp_ti) else None
        if not (epi.min() == e and epi.max() == e and ti.min() == exp_t and ti.max() == exp_t
                and np.array_equal(idx, np.arange(gframe, gframe + length))):
            bad += 1
        gframe += length
    if bad:
        issues.append(f"{bad}/{ne} episodes wrong episode_index/task_index/index")
    if gframe != info["total_frames"]:
        issues.append(f"frame total {gframe} != info {info['total_frames']}")

    npq = len(list(merged_root.glob("data/**/*.parquet")))
    nmp4 = len(list(merged_root.glob("videos/**/*.mp4")))
    if npq != ne:
        issues.append(f"parquet count {npq} != {ne}")
    if nmp4 != ne * len(vkeys):
        issues.append(f"mp4 count {nmp4} != {ne * len(vkeys)}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id, root=merged_root)
        if ds.meta.total_episodes != ne:
            issues.append(f"LeRobotDataset episodes {ds.meta.total_episodes} != {ne}")
    except Exception as ex:  # noqa: BLE001
        issues.append(f"LeRobotDataset load failed: {ex}")
    return issues


def convert_parallel(dataset, family, repo_id, *, out_root=None, procs=None, verify=True) -> dict:
    """Parallel drop-in for ``to_lerobot.convert``: shard by task -> parallel convert -> merge ->
    (optional) self-verify. Output at ``<out_root>/<family>`` (out_root defaults to
    ``outputs/datagen/<dataset>_lerobot_format``). Byte-identical to a serial run."""
    root = reader.ROOT
    fam_dir = root / dataset / family
    tasks = sorted(p.name for p in fam_dir.glob("task_*") if p.is_dir())
    if not tasks:
        raise SystemExit(f"[parallel] no task_* under {fam_dir}")
    procs = min(procs or max(1, (os.cpu_count() or 4) - 2), len(tasks))
    work = root / f"_parallel_{dataset}_{family}"
    if work.exists():
        shutil.rmtree(work)
    (work / "logs").mkdir(parents=True)

    print(f"[parallel] {family}: {len(tasks)} tasks x ~40 eps, {procs} procs", flush=True)
    t0 = time.time()
    failed = _run_shards_parallel(dataset, family, tasks, work, repo_id, procs)
    if failed:
        raise SystemExit(f"[parallel] shard(s) FAILED: {failed} — see {work}/logs/, work kept for debug")
    print(f"[parallel] all {len(tasks)} shards done in {time.time() - t0:.0f}s; merging", flush=True)

    out_root = out_root or str(root / f"{dataset}_lerobot_format")
    dest = Path(out_root) / family
    shard_dirs = [work / "shards" / t / family for t in tasks]
    summary = lerobot_merge.merge_shards(dest, shard_dirs)
    print(f"[parallel] merged: {summary}", flush=True)

    if verify:
        issues = _verify(dest, dataset, family, repo_id)
        if issues:
            print("[parallel] VERIFY FAILED:")
            for i in issues:
                print("  " + i)
            raise SystemExit("[parallel] verification FAILED — NOT identical to serial; work kept")
        print("[parallel] VERIFY OK (prompt-table + per-episode indices + load + counts)", flush=True)

    shutil.rmtree(work)
    print(f"[parallel] DONE -> {dest}  ({time.time() - t0:.0f}s total)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--_shard-one", nargs=5, metavar=("DATASET", "FAMILY", "TASK", "WORK", "REPO"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--dataset")
    ap.add_argument("--family", help="datagen OUTPUT dir name, e.g. dusty_transfer")
    ap.add_argument("--repo-id")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--procs", type=int, default=None, help="max concurrent shard procs (default: cpu-2)")
    ap.add_argument("--no-verify", action="store_true", help="skip the merged-dataset self-check")
    a = ap.parse_args()

    if a._shard_one:
        _shard_one(*a._shard_one)
        return
    if not (a.dataset and a.family and a.repo_id):
        ap.error("--dataset, --family and --repo-id are required")
    convert_parallel(a.dataset, a.family, a.repo_id, out_root=a.out_root, procs=a.procs, verify=not a.no_verify)


if __name__ == "__main__":
    main()
