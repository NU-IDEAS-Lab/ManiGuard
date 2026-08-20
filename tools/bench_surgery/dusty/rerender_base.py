"""Re-finalize the ``base/`` of dusty_transfer tasks whose dest was swapped (tools.bench_surgery.dusty.swap_dest).

Mirrors ``tools.bench_surgery.cabinet.rerender_base`` for the jar family: the swap tool rewrote each task's
``base/scene_ep1.json`` + ``base/diagnostics.jsonl`` (new stacked objects), but the 4
``base/rollout_*.mp4`` still show the OLD objects and the diagnostics runtime stats
(``gate_pass`` / ``ltl_violated`` / ``steps_executed`` / ``surface_info`` / ``bench``) are STALE.

This re-runs the EXACT bench-build finalize on each edited task: ``finalize_base_task`` builds the env
from the EDITED snapshot, bakes the canonical mounted robot + init pose, idle-steps under gravity (so
the swapped content SETTLES to its true rest), re-renders the 4 review videos, re-stamps ``cameras``, steps
a fresh LTL monitor, and recomputes ``gate_pass`` / ``surface_info`` / ``bench`` — while carrying every
edited task-identity field (selection / goal_region / ltl_safety / prompt / stack_mode / stack_height)
through its allowlist. Fresh subprocess per task (OmniGibson can segfault on teardown after a clean
write); success is judged by output presence, then the parent copies the 6 outputs back over ``base/``.

Usage:
  python -m tools.bench_surgery.dusty.rerender_base --tasks task_0022,task_0026
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
FAMILY = "dusty_transfer"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")
ROW_FILE = "_rerender_row.json"


def _tmp_dir(base_dir: Path) -> Path:
    return base_dir.parent / "_rerender_tmp"


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


def _run_worker(base_dir: Path, episode: int) -> None:
    from maniguard.data.bench_builder.finalize_base import finalize_base_task
    tmp = _tmp_dir(base_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        row = finalize_base_task(base_dir, tmp, family=FAMILY, episode=episode)
    except Exception as e:  # noqa: BLE001
        import traceback
        row = {"task": base_dir.parent.name, "status": "fail",
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    (tmp / ROW_FILE).write_text(json.dumps(row, default=float), encoding="utf-8")


def _commit_back(base_dir: Path, episode: int) -> tuple[bool, str]:
    tmp = _tmp_dir(base_dir)
    if not tmp.is_dir():
        return False, "no-tmp"
    row = {}
    if (tmp / ROW_FILE).is_file():
        row = json.loads((tmp / ROW_FILE).read_text(encoding="utf-8"))
    if row.get("status") == "fail":
        shutil.rmtree(tmp, ignore_errors=True)
        return False, f"finalize-fail({row.get('error')})"
    if not _complete(tmp, episode):
        shutil.rmtree(tmp, ignore_errors=True)
        return False, "incomplete-output"
    # DUST GUARD: the finalize reload can drop the particle group (attachment-uuid
    # mismatch). If the fresh snapshot lost it, re-inject the pre-finalize group
    # (positions are dest-local, so they stay valid after the settle).
    new_scene_p = tmp / f"scene_ep{episode}.json"
    old_scene_p = base_dir / f"scene_ep{episode}.json"
    new_scene = json.loads(new_scene_p.read_text())
    old_scene = json.loads(old_scene_p.read_text())
    def _dust(tree):
        return (tree.get("state", {}).get("registry", {})
                .get("system_registry", {}) or {}).get("dust")
    nd = _dust(new_scene)
    od = _dust(old_scene)
    dust_note = "kept"
    if od and (not nd or not nd.get("n_particles")):
        new_scene.setdefault("state", {}).setdefault("registry", {})\
            .setdefault("system_registry", {})["dust"] = od
        nested = (new_scene.get("init_info", {}).get("args", {}) or {}).get("scene_file")
        if isinstance(nested, dict):
            nested.setdefault("state", {}).setdefault("registry", {})\
                .setdefault("system_registry", {})["dust"] = od
        new_scene_p.write_text(json.dumps(new_scene))
        dust_note = "REINJECTED"
    for name in ([f"scene_ep{episode}.json", "diagnostics.jsonl"]
                 + [f"rollout_{lbl}_ep{episode}.mp4" for lbl in VIDEO_LABELS]):
        shutil.copy2(tmp / name, base_dir / name)
    shutil.rmtree(tmp, ignore_errors=True)
    return True, f"ok(status={row.get('status')}, dust={dust_note})"


def _driver(args: argparse.Namespace) -> int:
    out_fam = Path(args.bench_root) / FAMILY
    tasks = [t.strip() if t.strip().startswith("task_") else f"task_{int(t):04d}"
             for t in args.tasks.split(",")]
    env = _worker_env()
    results: dict[str, str] = {}
    for t in tasks:
        base_dir = out_fam / t / "base"
        if not (base_dir / f"scene_ep{args.episode}.json").is_file():
            results[t] = "missing-base"
            print(f"[rerender] {t}: missing-base", flush=True)
            continue
        cmd = [sys.executable, "-m", "tools.bench_surgery.dusty.rerender_base", "--worker",
               "--base-dir", str(base_dir), "--episode", str(args.episode)]
        try:
            rc = subprocess.run(cmd, env=env, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        ok, status = _commit_back(base_dir, args.episode)
        results[t] = status if ok else f"{status} [rc={rc}]"
        print(f"[rerender] {t}: {results[t]}", flush=True)
    n_ok = sum(1 for s in results.values() if s.startswith("ok"))
    print(f"\n[rerender] {n_ok}/{len(tasks)} base re-finalized OK")
    return 0 if n_ok == len(tasks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--base-dir", type=str)
    ap.add_argument("--bench-root", type=str, default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--tasks", type=str, help="comma list, e.g. task_0022,task_0026")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--episode", type=int, default=1)
    args = ap.parse_args()
    if args.worker:
        _run_worker(Path(args.base_dir), args.episode)
        return 0
    if not args.tasks:
        ap.error("--tasks required")
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
