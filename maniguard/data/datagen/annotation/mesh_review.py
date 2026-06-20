"""Fast mesh-only grasp review — NO sim load.

For each annotated object, renders the object (shown UPRIGHT, its scene orientation) + the
longfinger gripper at each annotated grasp, from a few fixed viewpoints, as ONE per-object
PNG. Pure trimesh + matplotlib — the SAME geometry the viser tool shows — so it skips the
~1-2 min OmniGibson load entirely (seconds per object).

Workflow: run this for the quick per-family check right after annotating; run
``validate_grasps.py`` LATER (when you have a big time block) for the heavier real
bench-camera sim summary.

The grasp's true approach direction is DERIVED from the stored eef pose (gripper +Z =
fingertips/approach) and shown in each column title — independent of the hand-typed
``approach_hint`` field (which is often left at the default).

  conda activate behavior          # needs trimesh/scipy/matplotlib, NOT sim
  PYTHONPATH=$HOME/project/ManiGuard \
  python -m maniguard.data.datagen.annotation.mesh_review [--family clutter] [--object cat/model]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANN_DIR = Path("outputs/grasp_annotation")
ANN = ANN_DIR / "grasp_annotations.json"
MESH_DB = ANN_DIR / "mesh_db.json"
GRIPPER = ANN_DIR / "gripper_longfinger.glb"
OUT = ANN_DIR / "mesh_review"

# family stem -> source_task prefix (source_task lives in mesh_db, not the annotations).
FAMILY_STEMS = {
    "clutter": "clutter_pickup/", "jar": "jar_transport/", "lid": "lid_transport/",
    "dusty": "dusty_transfer/", "stack": "stack_retrieve/", "cabinet": "cabinet_pickup/",
}

# (label, elev, azim) viewpoints — oblique 3/4, side, and near-top for grasp coverage.
VIEWS = [("oblique", 18.0, -60.0), ("side", 10.0, 30.0), ("top", 78.0, -90.0)]


def _verts(path, n=None):
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    v = np.asarray(m.vertices, float)
    if n and len(v) > n:
        v = v[np.linspace(0, len(v) - 1, n).astype(int)]
    return v


def classify_approach(a):
    """Derived approach class from the actual eef pose (gripper +Z = fingertips lead),
    given the approach vector in the UPRIGHT/world frame. Returns (label, theta_deg,
    confident) where theta is the angle from straight-down (0=top_down, 90=side, 180=up).

    Hard thresholds (user-set top_down cutoff = 60deg, mirrored): top_down 0-60, side
    60-120, bottom_up 120-180 — a tilted top grasp up to 60deg still counts as top_down.
    Always confident (no ambiguous band). Shared by mesh_review + fix_approach_tags."""
    a = np.asarray(a, float)
    a = a / (np.linalg.norm(a) + 1e-9)
    theta = float(np.degrees(np.arccos(np.clip(-a[2], -1.0, 1.0))))
    if theta <= 60.0:
        return "top_down", theta, True
    if theta >= 120.0:
        return "bottom_up", theta, True
    return "side", theta, True


def _add_triad_inset(ax, el, az):
    """Small world-frame XYZ triad (R=X, G=Y, B=Z) as a FIXED-position inset at the
    lower-left of the leftmost subplot — screen-anchored (not data-anchored), so it sits in
    the same spot across all rows (origin aligned in the column) and never overlaps the
    point cloud. Shares the row's view angle so it reads as the world orientation."""
    ti = ax.inset_axes([0.0, 0.0, 0.30, 0.30], projection="3d")
    ti.patch.set_alpha(0.0)
    ti.set_axis_off()
    for vec, col, name in (((1, 0, 0), "r", "X"), ((0, 1, 0), "g", "Y"),
                           ((0, 0, 1), "b", "Z")):
        v = np.asarray(vec, float)
        ti.quiver(0, 0, 0, v[0], v[1], v[2], color=col, linewidth=1.4,
                  arrow_length_ratio=0.25)
        t = v * 1.25
        ti.text(t[0], t[1], t[2], name, color=col, fontsize=7, ha="center", va="center")
    for setlim in (ti.set_xlim, ti.set_ylim, ti.set_zlim):
        setlim(-1.45, 1.45)
    ti.set_box_aspect((1, 1, 1))
    ti.view_init(elev=el, azim=az)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=None, choices=list(FAMILY_STEMS),
                    help="only objects whose source task is in these families")
    ap.add_argument("--object", default=None, help="only this object key 'cat/model'")
    args = ap.parse_args()

    ann = json.load(open(ANN))
    src = json.load(open(MESH_DB))["objects"] if MESH_DB.exists() else {}
    items = [(k, v) for k, v in ann["objects"].items() if v.get("grasps")]
    if args.family:
        stems = tuple(FAMILY_STEMS[f] for f in args.family)
        items = [(k, v) for k, v in items
                 if str(src.get(k, {}).get("source_task", "")).startswith(stems)]
    if args.object:
        items = [(k, v) for k, v in items if k == args.object]
    if not items:
        print("[mesh_review] nothing to render (check --family/--object and annotations).")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    GRIP = _verts(GRIPPER, 1400)
    print(f"[mesh_review] {len(items)} objects, "
          f"{sum(len(v['grasps']) for _, v in items)} grasps -> {OUT}", flush=True)

    for key, rec in items:
        R_up = Rot.from_quat(rec["upright_orientation_xyzw"]).as_matrix()
        objw = (R_up @ _verts(ANN_DIR / rec["mesh"], 2200).T).T   # upright display frame
        grasps = rec["grasps"]
        ncol, nrow = len(grasps), len(VIEWS)
        fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow))
        for ci, gr in enumerate(grasps):
            T = np.eye(4)
            T[:3, :3] = R_up @ Rot.from_quat(gr["orientation_xyzw"]).as_matrix()
            T[:3, 3] = R_up @ np.asarray(gr["position"], float)   # = (T_up @ T_local)
            gw = (T[:3, :3] @ GRIP.T).T + T[:3, 3]
            lab, th, conf = classify_approach(T[:3, 2])           # gripper +Z = approach
            tag = lab if conf else f"?{lab}/side"
            allp = np.vstack([objw, gw])
            ctr = (allp.max(0) + allp.min(0)) / 2
            half = (allp.max(0) - allp.min(0)).max() / 2 * 1.05
            for ri, (vn, el, az) in enumerate(VIEWS):
                ax = fig.add_subplot(nrow, ncol, ri * ncol + ci + 1, projection="3d")
                ax.scatter(objw[:, 0], objw[:, 1], objw[:, 2], s=2, c="0.6", alpha=0.45)
                ax.scatter(gw[:, 0], gw[:, 1], gw[:, 2], s=2, c="tab:blue", alpha=0.5)
                for setlim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
                    setlim(ctr[i] - half, ctr[i] + half)
                ax.view_init(elev=el, azim=az)
                ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
                if ci == 0:               # row label + small world triad on leftmost column
                    ax.text2D(-0.04, 0.5, vn, transform=ax.transAxes, rotation="vertical",
                              va="center", ha="center", fontsize=11, fontweight="bold")
                    _add_triad_inset(ax, el, az)
                if ri == 0:               # grasp id + derived approach on the top row
                    ax.set_title(f"#{gr['id']} [{tag} {th:.0f}°]"
                                 + (f" {gr['label']}" if gr.get("label") else ""),
                                 fontsize=8)
        fig.suptitle(f"{key}   ({ncol} grasps)   mesh-only   "
                     f"rows: {' / '.join(v[0] for v in VIEWS)}   "
                     f"(gray=object · blue=gripper)", fontsize=11)
        fig.tight_layout()
        out = OUT / f"{key.replace('/', '__')}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[mesh_review] {key}: {ncol} grasps -> {out}", flush=True)

    print("[mesh_review] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
