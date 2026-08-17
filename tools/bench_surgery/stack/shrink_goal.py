"""Shrink a stack_retrieve task's retrieval GOAL toward the target's initial position.

WHY: every chopping-board stack task places the goal_region ~0.55-0.72 m sideways (+Y robot-local)
from where the target sits, pushing the goal to 0.74-0.87 m radial reach — BEYOND the Franka
comfortable planar reach (REACH_COMFORT 0.72). The re-stack DEST is comfort-clamped (stack.py Fix 4)
but the retrieval GOAL never is, so ``t_transport`` hits IK_FAIL and every reach-fallback pullback
fails on the far goal. Moving the goal ``frac`` of the way toward the target's initial position pulls
the reach back inside comfort while keeping a genuine "retrieve it aside" motion.

The goal_region is SHARED by datagen (executor ``ctx.goal_center``) AND eval (goal_checker success
sphere), so the shrink MUST edit the bench task itself (not a datagen-only knob) for train/eval
consistency. This patches, per task's ``base/``:
  * ``diagnostics.jsonl``  goal_region.center_world  (executor + eval read this)
                           goal_region.anchor_local_xy (kept consistent for provenance/reach prints)
  * ``scene_ep1.json``     the goal marker's ``root_link.pos`` at its 2 stateful locations (so the
                           rendered green sphere sits at the new goal; the 2 spawn-arg copies carry
                           no position).

Idempotent: writes ``*.bak_shrinkgoal`` once and always shrinks from that ORIGINAL, so re-running
never double-shrinks. After this, run ``tools.bench_surgery.stack.rerender_base`` to re-render the base videos +
re-finalize (its allowlist carries the edited goal_region through), then datagen-test.

Usage:
  python -m tools.bench_surgery.stack.shrink_goal --tasks task_0002,task_0004 --frac 0.30 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
FAMILY = "stack_retrieve"
REACH_COMFORT = 0.72   # Franka comfortable planar reach (matches stack.py)


def _read_diag(base_dir: Path) -> dict:
    with open(base_dir / "diagnostics.jsonl") as f:
        return json.loads(f.readline())


def _marker_pos_nodes(scene: dict, marker: str) -> list:
    """The scene_ep locations that carry the marker's world position (``root_link.pos``). The runtime
    ``state`` copy is always inline; the ``init_info.args.scene_file`` copy is inline only for base tasks
    (some perturbation variants store scene_file as an external path STRING -> skip it there)."""
    reg = scene["state"]["registry"]["object_registry"]
    nodes = [reg[marker]["root_link"]]
    sf = (scene.get("init_info") or {}).get("args", {}).get("scene_file")
    if isinstance(sf, dict):
        try:
            nodes.append(sf["state"]["registry"]["object_registry"][marker]["root_link"])
        except (KeyError, TypeError):
            pass
    return nodes


def _target_world_xy(scene: dict, target_name: str) -> np.ndarray:
    reg = scene["state"]["registry"]["object_registry"]
    pos = reg[target_name]["root_link"]["pos"]
    return np.array(pos[:2], float)


def shrink_task(base_dir: Path, frac: float, dry: bool) -> dict:
    diag_p = base_dir / "diagnostics.jsonl"
    scene_p = base_dir / "scene_ep1.json"
    bak_diag = base_dir / "diagnostics.jsonl.bak_shrinkgoal"
    bak_scene = base_dir / "scene_ep1.json.bak_shrinkgoal"

    # always shrink from the pristine ORIGINAL (idempotent)
    if bak_diag.exists() and not dry:
        shutil.copy2(bak_diag, diag_p)
        shutil.copy2(bak_scene, scene_p)
    elif not bak_diag.exists() and not dry:
        shutil.copy2(diag_p, bak_diag)
        shutil.copy2(scene_p, bak_scene)

    diag = _read_diag(base_dir)
    scene = json.load(open(scene_p))
    gr = diag["goal_region"]
    marker = gr["marker_name"]
    target = gr["target_name"]

    center = np.array(gr["center_world"], float)          # [x,y,z]
    tgt_xy = _target_world_xy(scene, target)
    new_xy = center[:2] - frac * (center[:2] - tgt_xy)    # move frac toward target initial
    new_center = np.array([new_xy[0], new_xy[1], center[2]], float)

    # reach (robot-local anchor magnitude) before/after — anchor shrinks toward the pack center
    anc = np.array(gr["anchor_local_xy"], float)
    pack = np.array(gr["pack_bbox_robot_local_xy"], float).mean(axis=0)
    new_anc = anc - frac * (anc - pack)
    R_old, R_new = float(np.linalg.norm(anc)), float(np.linalg.norm(new_anc))

    info = {
        "task": base_dir.parent.name, "target": target,
        "center_old": center[:2].round(4).tolist(), "center_new": new_center[:2].round(4).tolist(),
        "shift_m": round(float(np.linalg.norm(new_center[:2] - center[:2])), 4),
        "R_old": round(R_old, 3), "R_new": round(R_new, 3),
        "under_comfort": bool(R_new <= REACH_COMFORT),
    }
    if dry:
        return info

    # --- write goal_region ---
    gr["center_world"] = new_center.tolist()
    gr["anchor_local_xy"] = new_anc.tolist()
    with open(diag_p, "w") as f:
        f.write(json.dumps(diag) + "\n")

    # --- move the marker in scene_ep (2 stateful copies) ---
    for node in _marker_pos_nodes(scene, marker):
        node["pos"] = new_center.tolist()
    with open(scene_p, "w") as f:
        json.dump(scene, f)

    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, help="comma list, e.g. task_0002,task_0004")
    ap.add_argument("--frac", type=float, default=0.30, help="fraction of goal->target distance to move")
    ap.add_argument("--variants", default="base",
                    help="comma list of variant subdirs to shrink (base,env,language,location,target). "
                         "Each variant shrinks toward ITS OWN target position (location moves the layout).")
    ap.add_argument("--bench-root", default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    root = Path(a.bench_root) / FAMILY
    print(f"{'task/variant':22s} {'target':20s} {'R_old':6s} {'R_new':6s} {'shift':6s} {'centre_old -> centre_new'}")
    n = 0
    for t in tasks:
        for v in variants:
            vd = root / t / v
            if not (vd / "diagnostics.jsonl").is_file():
                print(f"{t + '/' + v:22s} MISSING"); continue
            info = shrink_task(vd, a.frac, a.dry_run)
            n += 1
            print(f"{t + '/' + v:22s} {info['target']:20s} {info['R_old']:<6.3f} {info['R_new']:<6.3f} "
                  f"{info['shift_m']:<6.3f} {info['center_old']} -> {info['center_new']}")
    print(f"\n[shrink_goal] {'DRY-RUN, ' if a.dry_run else ''}{n} instance(s) frac={a.frac}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
