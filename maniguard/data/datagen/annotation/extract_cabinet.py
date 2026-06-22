"""Phase 0a-2 (cabinet) — extract the articulated cabinet for handle annotation + measure its
drawer geometry for the offline in-path check.

All 35 cabinet_pickup tasks use ONE cabinet model, so this runs once. It:
  1. spawns the cabinet, sets the drawer joint to the task's initial open fraction,
  2. extracts the cabinet's object-local visual mesh -> mesh_db (so the viser tool can show it
     and the user can annotate a side grasp on the handle),
  3. measures the drawer LINK's AABB in the cabinet-ROOT-local frame + records the joint at
     extraction (`j_extract`) and the slide/stroke metadata -> ``cabinet_geom.json``.

At runtime the handle world pose tracks the drawer: read the live drawer-link pose, or add
``slide_dir * (joint - j_extract)`` to the root-local annotated grasp.

  VK_ICD_FILENAMES=... CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 PYTHONPATH=$HOME/project/ManiGuard \
  python -u -m maniguard.data.datagen.annotation.extract_cabinet
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/cabinet_pickup")
OUT = Path("outputs/grasp_annotation")


def _cabinet_meta() -> dict:
    """Representative cabinet model + drawer metadata from the first task's diagnostics."""
    f = sorted(glob.glob(str(BENCH / "task_*/base/diagnostics.jsonl")))[0]
    diag = json.loads(open(f).readline())
    ci = diag["cabinet_info"]
    scene = json.load(open(f.replace("diagnostics.jsonl", "scene_ep1.json")))
    args = scene["objects_info"]["init_info"][ci["name"]]["args"]
    return {"category": args["category"], "model": args["model"], "info": ci,
            "source_task": f"cabinet_pickup/{Path(f).parent.parent.name}"}


def main() -> int:
    from maniguard.data.datagen.primitives import scene as scenemod
    from maniguard.data.datagen.primitives.grasp_obb import _to_np, _pose_to_mat, mesh_from_og_object

    meta = _cabinet_meta()
    cat, model, ci = meta["category"], meta["model"], meta["info"]
    j_extract = float(ci["open_fraction"]) * float(ci["stroke_m"])     # 0.2 * 0.36 = 0.072
    print(f"[cabinet] {cat}/{model} drawer joint={ci['joint']} link={ci['link']} "
          f"j_extract={j_extract:.4f} stroke={ci['stroke_m']:.3f}", flush=True)

    og = scenemod.init_omnigibson(headless=True)
    import omnigibson as og_mod

    env_cfg = {"env": {"action_frequency": 30, "rendering_frequency": 30},
               "scene": {"type": "Scene"},
               "objects": [{"type": "DatasetObject", "name": "cab", "category": cat,
                            "model": model, "position": [0, 0, 0.5], "fixed_base": True}],
               "robots": []}
    env = og_mod.Environment(configs=env_cfg)
    og.sim.step()
    cab = env.scene.object_registry("name", "cab")

    # set the drawer joint to the task's initial open fraction
    try:
        ji = list(cab.joints.keys()).index(ci["joint"])
        q = cab.get_joint_positions()
        q[ji] = j_extract
        cab.set_joint_positions(q)
        for _ in range(3):
            og.sim.step()
        print(f"[cabinet] set joint[{ji}]={j_extract:.4f}; joints={list(cab.joints.keys())}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[cabinet] (joint-set skip: {e})", flush=True)

    # drawer LINK AABB in cabinet-ROOT-local frame
    rp, rq = cab.get_position_orientation()
    T_world_root = _pose_to_mat(_to_np(rp), _to_np(rq))
    T_root_world = np.linalg.inv(T_world_root)
    link = cab.links[ci["link"]]
    lo_w, hi_w = link.aabb
    lo_w, hi_w = _to_np(lo_w), _to_np(hi_w)
    corners = np.array([[x, y, z] for x in (lo_w[0], hi_w[0])
                        for y in (lo_w[1], hi_w[1]) for z in (lo_w[2], hi_w[2])])
    corners_local = (T_root_world[:3, :3] @ corners.T).T + T_root_world[:3, 3]
    drawer_lo = corners_local.min(0)
    drawer_hi = corners_local.max(0)
    print(f"[cabinet] drawer link AABB (root-local): lo={np.round(drawer_lo,3)} "
          f"hi={np.round(drawer_hi,3)}", flush=True)

    # extract the cabinet mesh for the annotation tool + add it to mesh_db
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    mesh = mesh_from_og_object(cab, use_visual=True)
    key = f"{cat}/{model}"
    fname = key.replace("/", "__") + ".glb"
    mesh.export(OUT / "meshes" / fname)

    db = json.load(open(OUT / "mesh_db.json"))
    db["objects"][key] = {
        "category": cat, "model": model,
        "upright_orientation_xyzw": [float(v) for v in _to_np(rq)],
        "mesh": f"meshes/{fname}", "bbox_size": [float(v) for v in mesh.extents],
        "source_task": meta["source_task"], "grasps": [],
        "articulated": True, "handle_object": True, "j_extract": j_extract,
    }
    json.dump(db, open(OUT / "mesh_db.json", "w"), indent=2)

    geom = {"category": cat, "model": model, "joint": ci["joint"], "link": ci["link"],
            "slide_axis": ci["slide_axis"], "slide_sign": ci["slide_sign"],
            "stroke_m": float(ci["stroke_m"]), "j_extract": j_extract,
            "drawer_aabb_root_local": {"lo": [float(v) for v in drawer_lo],
                                       "hi": [float(v) for v in drawer_hi]},
            "interior_bbox": ci["interior_bbox"]}
    json.dump(geom, open(OUT / "cabinet_geom.json", "w"), indent=2)
    print(f"[cabinet] DONE — mesh -> {fname}, geom -> {OUT}/cabinet_geom.json", flush=True)
    try:
        og.sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
