"""Build the `target` (appearance) perturbation level for ManiGuard-Bench.

For each locked base task, the `target/` variant is the SAME scene as `base/`
with the family's target object recolored by a strong ``diffuse_tint`` — a
large, deterministic, out-of-distribution appearance shift.

``diffuse_tint`` is an asset/USD material property that OmniGibson does NOT
serialize into ``scene_ep1.json`` (verified), so the recolor cannot be baked
into the snapshot. Instead each `target/` instance is fully self-describing:

  * ``scene_ep1.json``  — a byte copy of ``base/`` (objects/poses identical).
  * ``diagnostics.jsonl`` — base diagnostics + a ``perturbation`` block
        ``{"kind":"target","recolor":{object,category,diffuse_tint,orig_color}}``.
  * 4 review videos — re-rendered with the tint applied (shows the OOD look).

Any consumer (this renderer, the eval client) loads a `target/` instance the
same uniform way as any other instance: build env → reset → pose →
``perturbation.apply_perturbation(env, diag)`` (which re-applies the recolor
from the declared spec). The sim never branches on base-vs-perturbation.

Same fresh-subprocess-per-task pattern as ``run_finalize_base`` (OmniGibson
can't cleanly reload Kit in-process and may segfault on teardown after a clean
write; success is judged by output presence, not exit code).

Usage:
  # a few sample tasks (smoke)
  python -m maniguard.data.bench_builder.perturb_target --family clutter_pickup --tasks 0,5 --jobs 2
  # full family, resumable
  python -m maniguard.data.bench_builder.perturb_target --family clutter_pickup --jobs 2 --skip-existing
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
LEVEL = "target"
ROW_FILE = "_target_row.json"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")


# ---------------------------------------------------------------------------- helpers

def _load_diag(base_dir: Path, episode: int) -> dict:
    txt = (base_dir / "diagnostics.jsonl").read_text(encoding="utf-8")
    try:
        d = json.loads(txt)
        return d if isinstance(d, dict) else d[0]
    except json.JSONDecodeError:
        return json.loads([ln for ln in txt.splitlines() if ln.strip()][0])


def _is_complete(out_dir: Path, episode: int) -> bool:
    if not (out_dir / f"scene_ep{episode}.json").is_file():
        return False
    if not (out_dir / "diagnostics.jsonl").is_file():
        return False
    return all((out_dir / f"rollout_{lbl}_ep{episode}.mp4").is_file() for lbl in VIDEO_LABELS)


def _select_tasks(out_fam: Path, spec: str | None, episode: int) -> list[str]:
    """Locked base tasks of this family (those with a complete base/ snapshot)."""
    available = sorted(
        d.name for d in out_fam.glob("task_*")
        if (d / "base" / f"scene_ep{episode}.json").is_file()
    )
    if not spec:
        return available
    avail = set(available)

    def norm(tok: str) -> str:
        tok = tok.strip()
        return tok if tok.startswith("task_") else f"task_{int(tok):04d}"

    chosen: list[str] = []
    if "," in spec or spec.replace("task_", "").isdigit():
        chosen = [norm(t) for t in spec.split(",")]
    elif "-" in spec:
        lo, hi = spec.split("-", 1)
        chosen = [f"task_{n:04d}" for n in range(int(lo), int(hi) + 1)]
    else:
        chosen = [norm(spec)]
    return [t for t in available if t in (set(chosen) & avail)]


def _worker_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OMNIGIBSON_HEADLESS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


# ---------------------------------------------------------------------------- worker

def _run_worker(base_dir: Path, out_dir: Path, family: str, episode: int) -> None:
    """Build ONE task's target/ variant; persist the row before teardown."""
    task = base_dir.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        row = _make_target_variant(base_dir, out_dir, family, episode)
    except Exception as e:  # noqa: BLE001
        import traceback
        row = {"task": task, "family": family, "status": "fail",
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    (out_dir / ROW_FILE).write_text(json.dumps(row, default=float), encoding="utf-8")


def _make_target_variant(base_dir: Path, out_dir: Path, family: str, episode: int) -> dict:
    import shutil

    import omnigibson as og
    import torch as th

    from maniguard.data.bench_builder.perturbation import (
        albedo_add_for, apply_recolor, average_object_color, find_object_by_category,
        pick_tint, resolve_target_category,
    )
    from maniguard.data.bench_builder.render import _build_og_config, render_views
    from maniguard.utils.robot_pose import BENCH_INIT_QPOS

    task = base_dir.parent.name
    diag = _load_diag(base_dir, episode)
    scene_file = base_dir / f"scene_ep{episode}.json"

    cat = resolve_target_category(diag, family)
    if not cat:
        raise RuntimeError(f"no target category for family={family}")

    # Fluid tasks (clutter/lid liquid variants carry selection.system_name) only
    # simulate under the PhysX GPU pipeline; the CPU default NaN-segfaults on the
    # first physics step. Must be set BEFORE the env is built (mirrors finalize).
    if (diag.get("selection") or {}).get("system_name"):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    og_cfg = _build_og_config(scene_file, diag, 256)
    if og.sim is not None:
        og.sim.stop()
        og.clear()
    env = og.Environment(configs=og_cfg)
    env.reset()
    robot = env.robots[0]
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()
    og.sim.step()

    target = find_object_by_category(env, cat)
    if target is None:
        raise RuntimeError(f"target object (category={cat}) not in scene")
    orig = average_object_color(target) or [1.0, 1.0, 1.0]
    task_index = int(task.split("_")[1])
    tint = pick_tint(orig, task_index)
    add = albedo_add_for(orig)
    n_mat = apply_recolor(target, tint, add)
    og.sim.step()
    if n_mat == 0:
        raise RuntimeError(f"recolor applied to 0 materials of {target.name}")

    out_diag = copy.deepcopy(diag)
    out_diag["perturbation"] = {
        "kind": "target",
        "recolor": {
            "object": target.name, "category": cat,
            "diffuse_tint": tint, "albedo_add": add, "orig_color": orig,
        },
    }

    # Render the recolored env (render_views re-stamps diag['cameras']).
    out_diag, stats = render_views(env, out_diag, out_dir, episode=episode)

    # scene snapshot = byte copy of base (the tint is NOT baked; it lives in diag).
    shutil.copy2(scene_file, out_dir / f"scene_ep{episode}.json")
    (out_dir / "diagnostics.jsonl").write_text(
        json.dumps(out_diag, default=float), encoding="utf-8")

    return {
        "task": task, "family": family, "status": "ok",
        "target": target.name, "diffuse_tint": tint, "orig_color": orig,
        "n_materials": n_mat,
        "arm_drift": stats.get("arm_drift"), "obj_disp": stats.get("obj_disp"),
    }


# ---------------------------------------------------------------------------- driver

def _spawn_worker(base_dir: Path, out_dir: Path, family: str, episode: int, env: dict, timeout: int):
    cmd = [
        sys.executable, "-m", "maniguard.data.bench_builder.perturb_target",
        "--worker", "--base-dir", str(base_dir), "--out-dir", str(out_dir),
        "--family", family, "--episode", str(episode),
    ]
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return "timeout"


def _row_for_task(task: str, out_fam: Path, family: str, episode: int) -> dict:
    """Worker row + offline structural QC (reuse validate_base_task — target's
    scene/robot/pose/cameras/videos are identical to base) + recolor sanity."""
    from maniguard.data.bench_builder.validate_base import validate_base_task

    out_dir = out_fam / task / LEVEL
    row_path = out_dir / ROW_FILE
    if not row_path.exists():
        return {"task": task, "family": family, "status": "fail",
                "error": "worker produced no row (crashed before writing)"}
    work_row = json.loads(row_path.read_text(encoding="utf-8"))
    if work_row.get("status") == "fail":
        return {"task": task, "family": family, "status": "fail",
                "error": work_row.get("error"), "worker": work_row}
    if not _is_complete(out_dir, episode):
        return {"task": task, "family": family, "status": "fail",
                "error": "incomplete output (missing snapshot/videos)", "worker": work_row}
    qc = validate_base_task(out_dir, family=family, episode=episode)
    # recolor sanity: the perturbation block must be well-formed
    diag = _load_diag(out_dir, episode)
    rc = (diag.get("perturbation") or {}).get("recolor") or {}
    recolor_ok = bool(rc.get("object") and rc.get("diffuse_tint"))
    status = qc.get("status")
    if not recolor_ok:
        status = "fail"
    return {
        "task": task, "family": family, "status": status,
        "recolor_ok": recolor_ok, "worker": work_row, "validate": qc,
    }


def _driver(args: argparse.Namespace) -> int:
    out_fam = Path(args.bench_root) / args.family
    if not out_fam.is_dir():
        print(f"[target] ERROR: family dir not found: {out_fam}", flush=True)
        return 2
    tasks = _select_tasks(out_fam, args.tasks, args.episode)
    if not tasks:
        print(f"[target] no matching base tasks in {out_fam} (--tasks {args.tasks!r})", flush=True)
        return 1
    env = _worker_env()

    to_run = [t for t in tasks
              if not (args.skip_existing and _is_complete(out_fam / t / LEVEL, args.episode))]
    skipped = [t for t in tasks if t not in set(to_run)]
    print(f"[target] {args.family}: {len(tasks)} base tasks ({len(to_run)} to run, "
          f"{len(skipped)} skip), jobs={args.jobs} -> {out_fam}/*/target", flush=True)

    rows: list[dict] = []
    manifest = out_fam / "target_manifest.jsonl"
    mf = manifest.open("w", encoding="utf-8")

    def _record(task: str, tag: str, done: int, total: int) -> None:
        row = _row_for_task(task, out_fam, args.family, args.episode)
        rows.append(row)
        mf.write(json.dumps(row, default=float) + "\n")
        mf.flush()
        extra = ""
        tint = (row.get("worker") or {}).get("diffuse_tint")
        if tint:
            extra = f"  tint={tint}"
        if row["status"] == "fail":
            extra = f"  {row.get('error') or row.get('validate', {}).get('fails')}"
        print(f"[{tag} {done}/{total}] {task}: {row['status']}{extra}", flush=True)

    for i, t in enumerate(skipped, 1):
        _record(t, "skip", i, len(skipped))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(_spawn_worker, out_fam / t / "base", out_fam / t / LEVEL,
                      args.family, args.episode, env, args.timeout): t
            for t in to_run
        }
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            fut.result()
            done += 1
            _record(t, "run", done, len(to_run))

    mf.close()
    counts = Counter(r["status"] for r in rows)
    print(f"=== {args.family} target: {dict(counts)} ({len(rows)} tasks) -> {manifest}", flush=True)
    fails = [r["task"] for r in rows if r["status"] == "fail"]
    if fails:
        print(f"    FAILED: {fails}", flush=True)
    return 1 if fails else 0


# ---------------------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description="Build the target (appearance) perturbation level.")
    ap.add_argument("--worker", action="store_true", help="internal: build ONE task's target variant")
    ap.add_argument("--base-dir", help="worker: the task's base/ dir")
    ap.add_argument("--out-dir", help="worker: the task's target/ dir")
    ap.add_argument("--family", required=True)
    ap.add_argument("--bench-root", default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--tasks", default=None, help="'0-22' / 'task_0000,task_0005' / '0,5'; default all")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker subprocesses")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="per-task worker timeout (s)")
    args = ap.parse_args()

    if args.worker:
        if not args.base_dir or not args.out_dir:
            ap.error("--worker requires --base-dir and --out-dir")
        _run_worker(Path(args.base_dir), Path(args.out_dir), args.family, args.episode)
        return 0
    return _driver(args)


if __name__ == "__main__":
    sys.exit(main())
