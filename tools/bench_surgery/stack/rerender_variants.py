"""Re-render the 4 review MP4s of stack_retrieve task variants whose goal marker moved (goal-shrink).

RENDER-ONLY (bench_builder.render.render_task, NOT the finalizer): the scene is already finalized and
nothing but the green goal-region marker moved, so we just rebuild the env from the patched
``scene_ep{ep}.json`` and re-shoot the 4 idle-step rollout videos. A fresh subprocess per variant
(OmniGibson segfaults on teardown). Driver runs a small staggered pool (concurrent OG boots can OOM a
16 GB GPU) and copies the refreshed ``diagnostics`` (cameras) back.

Usage:
  python -m tools.bench_surgery.stack.rerender_variants --tasks task_0000,task_0002 \
      --variants base,env,language,location,target --jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
FAMILY = "stack_retrieve"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")


def _worker(vdir: Path, episode: int) -> None:
    from maniguard.data.bench_builder.render import render_task
    diag = json.loads(open(vdir / "diagnostics.jsonl").readline())
    new_diag = render_task(vdir / f"scene_ep{episode}.json", diag, vdir, episode=episode)
    with open(vdir / "diagnostics.jsonl", "w") as f:
        f.write(json.dumps(new_diag) + "\n")
    print(f"[rerender] {vdir} done", flush=True)
    sys.stdout.flush()
    os._exit(0)              # skip OmniGibson's segfaulting teardown (render + write already committed)


def _worker_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("OMNIGIBSON_DATA_PATH", "/home/yiyanpeng/project/SENTINEL-Lite-data/datasets")
    env["OMNIGIBSON_HEADLESS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _complete(vdir: Path, episode: int, since: float) -> bool:
    return all((vdir / f"rollout_{lbl}_ep{episode}.mp4").is_file()
               and (vdir / f"rollout_{lbl}_ep{episode}.mp4").stat().st_mtime >= since
               for lbl in VIDEO_LABELS)


def _driver(a: argparse.Namespace) -> int:
    root = Path(a.bench_root) / FAMILY
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    jobs = [root / t / v for t in tasks for v in variants
            if (root / t / v / f"scene_ep{a.episode}.json").is_file()]
    print(f"[rerender] {len(jobs)} variants, jobs={a.jobs}, stagger={a.stagger}s", flush=True)
    started = time.monotonic()
    procs: dict = {}          # Popen -> (vdir, launch_ts)
    i = 0
    last_launch = -1e9
    results = {}
    while i < len(jobs) or procs:
        for p in list(procs):
            if p.poll() is not None:
                vdir, _ = procs.pop(p)
                ok = _complete(vdir, a.episode, started)
                results[str(vdir)] = "ok" if ok else f"FAILED(rc={p.returncode})"
                print(f"[rerender] {vdir.parent.name}/{vdir.name}: {results[str(vdir)]}", flush=True)
        now = time.monotonic()
        if i < len(jobs) and len(procs) < a.jobs and (now - last_launch) >= a.stagger:
            vdir = jobs[i]
            cmd = [sys.executable, "-m", "tools.bench_surgery.stack.rerender_variants", "--worker",
                   "--dir", str(vdir), "--episode", str(a.episode)]
            log = open(vdir / "_rerender.log", "w")
            p = subprocess.Popen(cmd, env=_worker_env(), stdout=log, stderr=subprocess.STDOUT)
            procs[p] = (vdir, now)
            last_launch = now
            i += 1
            print(f"[rerender] launched {vdir.parent.name}/{vdir.name} ({i}/{len(jobs)})", flush=True)
        time.sleep(4)
    n_ok = sum(1 for r in results.values() if r == "ok")
    print(f"\n[rerender] {n_ok}/{len(jobs)} re-rendered OK", flush=True)
    for k, v in sorted(results.items()):
        if v != "ok":
            print(f"  {v}: {k}", flush=True)
    return 0 if n_ok == len(jobs) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--dir", type=str)
    ap.add_argument("--tasks", type=str)
    ap.add_argument("--variants", default="base,env,language,location,target")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--stagger", type=int, default=70, help="seconds between launches (avoid boot OOM)")
    ap.add_argument("--bench-root", default=BENCH_ROOT_DEFAULT)
    a = ap.parse_args()
    if a.worker:
        _worker(Path(a.dir), a.episode)
        return 0
    return _driver(a)


if __name__ == "__main__":
    raise SystemExit(main())
