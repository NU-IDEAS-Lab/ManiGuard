"""Sweep driver — collect demos across many base tasks of a family.

Parallelism unit = the task (og.clear() can't switch tasks in one OG process, so each task
runs in a FRESH ``driver.run_task`` subprocess). This sweep runs its assigned tasks
SEQUENTIALLY, one subprocess each, on one GPU, logging each task to its own file. For multi-GPU
parallelism, launch one sweep per shard (``--shard i --num-shards N --gpu G``) in separate tmux
sessions — shards own disjoint tasks (round-robin), so output dirs / logs never collide.

Save isolation (no two processes ever write the same file):
  - demos    outputs/datagen/<dataset>/<bench_family>/<task>/traj_NNN/  (task owned by 1 shard)
  - per task outputs/datagen/<dataset>/<bench_family>/<task>/_summary.json
  - log      outputs/datagen/<dataset>/_logs/<task>.log

  conda activate behavior
  PYTHONPATH=$HOME/project/ManiGuard python -m maniguard.data.datagen.sweep \
      --family clutter --dataset v1 --limit-tasks 5 --target 50 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# skeleton key -> bench dataset family dir (output uses the bench dir name, matching the bench layout)
BENCH_FAMILY = {
    "clutter": "clutter_pickup", "jar": "jar_transport", "lid": "lid_transport",
    "dusty": "dusty_transfer", "stack": "stack_retrieve", "cabinet": "cabinet_pickup",
}
BENCH_ROOT = Path("outputs/lerobot_datasets/maniguard-bench")
OUT_ROOT = Path("outputs/datagen")


def _tasks(bench_family: str):
    base = BENCH_ROOT / bench_family
    return sorted(p.name for p in base.glob("task_*") if (p / "base").is_dir())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="clutter", choices=list(BENCH_FAMILY))
    ap.add_argument("--dataset", required=True, help="collection name, e.g. v1 (kept apart from smoke/test)")
    ap.add_argument("--target", type=int, default=50, help="success+safe demos PER TASK (each task is "
                    "collected until it reaches this many, or hits the attempt cap)")
    ap.add_argument("--max-attempts", type=int, default=None, help="per-task attempt cap passed to the "
                    "driver (default in driver: target*4). Raise it (e.g. target*8) so a low-success "
                    "task still reaches target; it also bounds a truly un-collectable task so the sweep "
                    "can't hang.")
    ap.add_argument("--limit-tasks", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--tasks", nargs="*", default=None, help="explicit task names (overrides --limit-tasks)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--score", dest="score", action="store_true", default=True)
    ap.add_argument("--no-score", dest="score", action="store_false")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip tasks whose _summary.json already reaches the target")
    a = ap.parse_args()

    bench_family = BENCH_FAMILY[a.family]
    all_tasks = a.tasks or _tasks(bench_family)
    if not a.tasks and a.limit_tasks:
        all_tasks = all_tasks[: a.limit_tasks]
    my_tasks = [t for i, t in enumerate(all_tasks) if i % a.num_shards == a.shard]

    # per-FAMILY logs (under the same family subdir as the demo data) so the 6 families never collide
    # — task names repeat across families (every family has a task_0000), so a shared _logs/ would
    # overwrite (clutter's task_0001.log clobbered by cabinet's, etc.).
    logdir = OUT_ROOT / a.dataset / bench_family / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] family={a.family}({bench_family}) dataset={a.dataset} target={a.target} "
          f"gpu={a.gpu} shard={a.shard}/{a.num_shards} -> {len(my_tasks)} tasks: {my_tasks}",
          flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    env["OMNIGIBSON_HEADLESS"] = "1"
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env["PYTHONPATH"] = str(Path.cwd())

    t_sweep = time.time()
    done = []
    for ti, task in enumerate(my_tasks):
        out_task = OUT_ROOT / a.dataset / bench_family / task
        if a.skip_existing and (out_task / "_summary.json").exists():
            s = json.loads((out_task / "_summary.json").read_text())
            if s.get("n_success", 0) >= a.target:
                print(f"[sweep] ({ti + 1}/{len(my_tasks)}) {task} already {s['n_success']} — skip",
                      flush=True)
                done.append(s)
                continue
        task_dir = BENCH_ROOT / bench_family / task / "base"
        log = logdir / f"{task}.log"
        cmd = [sys.executable, "-u", "-m", "maniguard.data.datagen.driver",
               "--task-dir", str(task_dir), "--family", a.family, "--dataset", a.dataset,
               "--target", str(a.target)]
        if a.max_attempts:
            cmd += ["--max-attempts", str(a.max_attempts)]
        cmd += (["--score"] if a.score else [])
        print(f"[sweep] ({ti + 1}/{len(my_tasks)}) {task} -> {log}", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
        dt = time.time() - t0
        sm = out_task / "_summary.json"
        s = json.loads(sm.read_text()) if sm.exists() else {"task": task, "n_success": 0, "n_attempts": 0}
        s["wall_s"] = round(dt, 1)
        done.append(s)
        flag = "OK" if s.get("n_success", 0) >= a.target else "⚠️ UNDER-TARGET"
        print(f"[sweep] ({ti + 1}/{len(my_tasks)}) {task} {flag} {s.get('n_success')}/{a.target} "
              f"({s.get('n_attempts')} att, {dt / 60:.1f} min)", flush=True)

    tot = time.time() - t_sweep
    nok = sum(s.get("n_success", 0) for s in done)
    under = [s for s in done if s.get("n_success", 0) < a.target]
    print(f"\n[sweep] ALL DONE: {len(done)} tasks, {nok} demos, {tot / 60:.1f} min total "
          f"({tot / 3600:.2f} h)", flush=True)
    for s in done:
        ok = s.get("n_success", 0) >= a.target
        print(f"    {s.get('task')}: {s.get('n_success')}/{a.target}{'' if ok else '  ⚠️ UNDER'} "
              f"({s.get('n_attempts')} att, {s.get('wall_s', 0) / 60:.1f} min)", flush=True)
    if under:
        print(f"[sweep] ⚠️ {len(under)} task(s) UNDER target {a.target}: "
              f"{[s.get('task') for s in under]} — re-run (it RESUMES / tops up to target) or "
              f"check those objects' grasps", flush=True)
    (logdir / f"_sweep_shard{a.shard}.json").write_text(                 # per-family (under its _logs/)
        json.dumps({"family": a.family, "target": a.target, "total_s": round(tot, 1),
                    "n_demos": nok, "tasks": done}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
