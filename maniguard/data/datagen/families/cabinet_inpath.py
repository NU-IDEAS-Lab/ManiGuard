"""Phase 0b (cabinet) — independent OFFLINE recompute of which objects block the drawer's
opening path, cross-checked against the bench's ``placement.in_path`` flag.

No sim: uses the cached cabinet drawer geometry (``cabinet_geom.json``, root-local drawer-link
AABB), each task's ``scene_ep1`` (cabinet root pose + object world poses) and ``diagnostics``
(per-task world ``slide_dir`` + the flags to verify), and object footprints from ``mesh_db``.

Method (all in the world xy plane):
  * transform the drawer-link AABB to world by the cabinet root pose -> its leading face along
    ``slide_dir`` (``d_front``) and its perpendicular span (``p_lo..p_hi``);
  * the opening **corridor** = ``d ∈ [d_front, d_front + (stroke - j_extract)]`` (the drawer's
    full further travel) × the perpendicular span;
  * an object (square footprint = its position ± half its max horizontal bbox) is ``in_path`` if
    its d-range AND p-range overlap the corridor.

Any disagreement with the diagnostics flag is reported for the user to review.

  PYTHONPATH=$HOME/project/ManiGuard python -m maniguard.data.datagen.families.cabinet_inpath
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.families.cabinet_geom import drawer_world_projection, slide_axes

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/cabinet_pickup")
ANN = Path("outputs/grasp_annotation")


def _obj_world(scene, name):
    reg = scene["state"]["registry"]["object_registry"]
    for k, v in reg.items():
        if name in k:
            return np.asarray(v["root_link"]["pos"], float), np.asarray(v["root_link"]["ori"], float)
    return None, None


def _model_of(scene, name):
    a = scene.get("objects_info", {}).get("init_info", {}).get(name, {}).get("args", {})
    return a.get("category"), a.get("model")


def recompute_in_path(diag, scene, geom, mesh_db, *, margin=0.0):
    """Return {object_name: bool} recomputed in_path for target + obstacle."""
    ci = diag["cabinet_info"]
    d, p = slide_axes(ci["slide_dir"])             # world opening dir + perpendicular (xy)
    travel = float(ci["stroke_m"]) - float(geom["j_extract"])

    rp, rq = _obj_world(scene, ci["name"])
    g = geom["drawer_aabb_root_local"]
    d_front, _d_back, p_lo, p_hi, _zl, _zh = drawer_world_projection(rp, rq, g["lo"], g["hi"], d, p)
    corridor_d = (d_front, d_front + travel)

    out = {}
    for info_key in ("target_info", "obstacle_info"):
        info = diag.get(info_key) or {}
        nm = info.get("name")
        if not nm:
            continue
        op, oq = _obj_world(scene, nm)
        cat, model = _model_of(scene, nm)
        bb = mesh_db["objects"].get(f"{cat}/{model}", {}).get("bbox_size", [0.05, 0.05, 0.05])
        # ORIENTED footprint: the object's bbox in its own frame, rotated into world, projected
        # onto (d, p). A circular max-dim radius over-flags flat/wide objects lying off to the
        # side (verified false positives on griddle_pan / stockpot), so use the real OBB.
        Ro = Rot.from_quat(oq).as_matrix()[:2, :2]
        hx, hy = bb[0] / 2 + margin, bb[1] / 2 + margin
        corners = np.array([[sx * hx, sy * hy] for sx in (-1, 1) for sy in (-1, 1)])
        wc = (Ro @ corners.T).T + op[:2]
        od, opp = wc @ d, wc @ p
        d_ovl = (od.max() >= corridor_d[0]) and (od.min() <= corridor_d[1])
        p_ovl = (opp.max() >= p_lo) and (opp.min() <= p_hi)
        out[nm] = bool(d_ovl and p_ovl)
    return out


def main() -> int:
    geom = json.load(open(ANN / "cabinet_geom.json"))
    mesh_db = json.load(open(ANN / "mesh_db.json"))
    tasks = sorted(glob.glob(str(BENCH / "task_*/base")))

    mism = []
    rows = []
    for tdir in tasks:
        tdir = Path(tdir)
        diag = json.loads(open(tdir / "diagnostics.jsonl").readline())
        scene = json.load(open(tdir / "scene_ep1.json"))
        recomputed = recompute_in_path(diag, scene, geom, mesh_db)
        for info_key in ("target_info", "obstacle_info"):
            info = diag.get(info_key) or {}
            nm = info.get("name")
            if not nm:
                continue
            bench_flag = bool(info.get("placement", {}).get("in_path"))
            mine = recomputed.get(nm)
            role = "target" if info_key == "target_info" else "obstacle"
            rows.append((tdir.parent.name, role, nm, bench_flag, mine))
            if mine != bench_flag:
                mism.append((tdir.parent.name, role, nm, bench_flag, mine))

    print(f"[in_path] {len(tasks)} tasks, {len(rows)} object-flags checked, "
          f"{len(mism)} MISMATCHES vs diagnostics\n")
    if mism:
        print("  task        role      object                          bench  recomputed")
        for t, role, nm, b, m in mism:
            print(f"  {t:<11} {role:<9} {nm:<32} {str(b):<6} {m}")
    else:
        print("  (recompute agrees with every diagnostics in_path flag)")

    # blocker_mode agreement summary
    from collections import Counter
    agree = Counter()
    for tdir in tasks:
        tdir = Path(tdir)
        diag = json.loads(open(tdir / "diagnostics.jsonl").readline())
        scene = json.load(open(tdir / "scene_ep1.json"))
        rc = recompute_in_path(diag, scene, geom, mesh_db)
        t_in = rc.get(diag["target_info"]["name"], False)
        o_in = rc.get(diag["obstacle_info"]["name"], False)
        mine_mode = "both" if (t_in and o_in) else "target" if t_in else "obstacle" if o_in else "none"
        agree[(diag.get("blocker_mode"), mine_mode)] += 1
    print("\n  blocker_mode (diagnostics -> recomputed) counts:")
    for (bd, mn), c in sorted(agree.items()):
        flag = "  <-- DIFF" if bd != mn else ""
        print(f"    {bd:>9} -> {mn:<9} : {c}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
