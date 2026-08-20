"""In-place CONTAINER pose surgery for lid_transport bench tasks.

Light layout surgery (dusty task_0006 layout-swap precedent): rotate the container
about its own z (bring a comfortable annotated grasp azimuth toward the robot),
translate it toward the robot (reach-envelope rescue), or flip it 180 deg about x
(asset with the F attachment meta-link authored at the BOTTOM: can/damllm).

Edits BOTH json trees (top-level + nested init_info.args.scene_file), zeroes
velocities, backs up to ``*.bak_pose``. Re-finalize with ``tools.bench_surgery.lid.rerender_base``
afterwards (gravity settle + camera/gate/LTL re-bake).

Usage:
  python -m tools.bench_surgery.lid.adjust_pose --task task_0003 --yaw-deg 90 --apply
  python -m tools.bench_surgery.lid.adjust_pose --task task_0024 --toward-robot-m 0.20 --apply
  python -m tools.bench_surgery.lid.adjust_pose --task task_0028 --flip --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/lid_transport")


def _container_name(diag: dict, tree: dict, role: str = "container") -> str:
    if role == "goal":
        return str(diag["goal_region"]["marker_name"])
    sel = {x["role"]: x for x in diag["selection"]["spawn_specs"]}
    spec = sel.get("lid") if role == "lid" else (sel.get("container") or sel.get("target"))
    for nm, info in tree["objects_info"]["init_info"].items():
        a = info.get("args", {})
        if (a.get("category"), a.get("model")) == (spec["category"], spec["model"]):
            return nm
    raise ValueError("container instance not found")


def _trees(scene):
    yield scene
    nested = (scene.get("init_info", {}).get("args", {}) or {}).get("scene_file")
    if isinstance(nested, dict):
        yield nested


def adjust(task: str, yaw_deg: float, toward_m: float, flip: bool, apply: bool,
           role: str = "container", dxy=(0.0, 0.0)) -> str:
    base = BENCH / task / "base"
    scene_p, diag_p = base / "scene_ep1.json", base / "diagnostics.jsonl"
    diag = json.loads(diag_p.read_text())
    scene = json.loads(scene_p.read_text())
    ops = []
    n_edit = 0
    top_reg = scene["state"]["registry"]["object_registry"]
    rob_key = next(k for k in top_reg if k.startswith(("agent", "robot")))
    rob_pos = np.asarray(top_reg[rob_key]["root_link"]["pos"], float)
    for tree in _trees(scene):
        reg = tree["state"]["registry"]["object_registry"]
        cont = _container_name(diag, tree, role=role)
        root = reg[cont]["root_link"]
        pos = np.asarray(root["pos"], float)
        quat = np.asarray(root["ori"], float)
        if yaw_deg:
            quat = (Rot.from_euler("z", yaw_deg, degrees=True) * Rot.from_quat(quat)).as_quat()
            ops.append(f"yaw+{yaw_deg}deg")
        if flip:
            # 180 about x THROUGH the object's own frame; raise so the (previous) top
            # face rests near the support -- the rerender's gravity settle finishes it
            quat = (Rot.from_quat(quat) * Rot.from_euler("x", 180, degrees=True)).as_quat()
            pos[2] += 0.02
            ops.append("flip180x")
        if dxy != (0.0, 0.0):
            pos[0] += float(dxy[0])
            pos[1] += float(dxy[1])
            ops.append(f"dxy+{dxy}")
        if toward_m:
            d = rob_pos[:2] - pos[:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            pos[:2] = pos[:2] + d * toward_m
            ops.append(f"toward_robot+{toward_m}m")
        root["pos"] = [float(v) for v in pos]
        root["ori"] = [float(v) for v in quat]
        for k in ("lin_vel", "ang_vel"):
            if k in root:
                root[k] = [0.0, 0.0, 0.0]
        n_edit += 1
    if role == "goal":
        # keep the diag's goal spec in lockstep with the marker object
        gr = diag["goal_region"]
        cw = np.asarray(gr["center_world"], float)
        cw[:2] = pos[:2]
        gr["center_world"] = [float(v) for v in cw]
        ops.append("diag.center_world synced")
    if not apply:
        return f"{task}: {ops} trees={n_edit} DRY-RUN"
    bak = scene_p.with_suffix(".json.bak_pose")
    if not bak.exists():
        shutil.copy2(scene_p, bak)
        shutil.copy2(diag_p, diag_p.with_suffix(".jsonl.bak_pose"))
    scene_p.write_text(json.dumps(scene))
    diag_p.write_text(json.dumps(diag) + "\n")
    return f"{task}: {ops} APPLIED trees={n_edit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--toward-robot-m", type=float, default=0.0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--role", default="container", choices=["container", "lid", "goal"])
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(adjust(a.task, a.yaw_deg, a.toward_robot_m, a.flip, a.apply,
                 role=a.role, dxy=(a.dx, a.dy)))


if __name__ == "__main__":
    main()
