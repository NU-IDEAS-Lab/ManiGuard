"""Programmatic straight-down (top-down) grasp generator for the STICKY-grasp families.

Why this exists
---------------
The cabinet family grasps objects with ``grasping_mode="sticky"`` (the targets are slabs wider
than the gripper in both horizontal axes, so force-closure is impossible — sticky magnetises on
first finger contact). Hand-annotated EDGE grasps (tilted, palm-flipped) caused three coupled
failures during the carry-into-drawer: a tilted approach makes the place ``fit_yaw`` reorient
rotate the held object about a TILTED axis → it tips past the LTL ``upright`` gate; the slab
hangs off-centre → the compliant wrist sags ~7 cm (place_lift / place_across reach-undershoot);
and the palm-flip winds the wrist toward a limit (singularity-adjacent contortion).

A straight-down grasp CENTRED over the object's centre-of-mass removes all three at once: the
reorient becomes a pure yaw about the vertical (object stays level → upright preserved), the load
hangs straight below the wrist (no lateral torque → no sag), and the wrist sits in its natural
down-pointing pose. Sticky makes this valid even for too-wide slabs: the fingers need only TOUCH
the top, not close around it.

What it generates
-----------------
For each object key, a fan of straight-down grasps at the top-surface point above the CoM:
``approach = world -Z`` (eef +Z down, the convention from ``annotate_tool``), finger-separation
(eef -Y) swept over ``n_yaw`` angles in ``[0, pi)`` (the +pi roll is covered at scoring time by
``grasp_select.roll_disambig``, which IK's both rolls and keeps the wrist farthest from a limit).
Grasps are stored in the object's LOCAL frame (the DB convention) tagged ``source="topdown_gen"``;
the cabinet family prefers these over the legacy edge grasps.

Usage
-----
  conda run -n behavior python -u -m maniguard.data.datagen.annotation.generate_topdown_grasps \
      --objects fruitcake/nmxadm graduated_cylinder/egpkea          # specific objects
  conda run -n behavior python -u -m maniguard.data.datagen.annotation.generate_topdown_grasps \
      --cabinet-all                                                 # every cabinet_pickup object
  ... add --apply to write into the DB (default is a dry-run preview).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

# ---- robot / gripper geometry (measured on the longfinger Franka; see scratchpad/measure_eef_tip) ----
FINGER_OFFSET = 0.114      # eef_link -> fingertip distance along the approach axis (m)
GRIPPER_WIDTH = 0.08       # max finger separation (m); the fingers land ON the top face only if the
#                            object's extent along the finger-separation axis exceeds this.
STRADDLE_FRAC = 0.40       # narrow object: drop the fingertips this fraction of the object height below
#                            the top so the fingers straddle the upper body and grip the sides on close.
STRADDLE_MAX = 0.06        # ...but never deeper than this (keep clear of the table for a short object).
GRASP_INSET = 0.015        # WIDE object: press the fingertips this far BELOW the top surface. The relocate
#                            descend is collision-off for the grasp target, so the fingers penetrate the top
#                            slightly and a contact registers -> sticky attaches. Fingertips resting EXACTLY at
#                            the surface only graze (no contact force) and sticky never fires (the v6 miss).

ANN_PATH = Path("outputs/grasp_annotation/grasp_annotations.json")
MESH_DB_PATH = Path("outputs/grasp_annotation/mesh_db.json")
MESH_DIR = Path("outputs/grasp_annotation")


def _eef_R(closing_angle: float) -> np.ndarray:
    """eef rotation (upright/world frame) for a straight-DOWN grasp whose finger-separation axis
    lies horizontally at ``closing_angle``. Convention (annotate_tool): eef +Z = approach (down),
    eef -Y = closing (between fingers). Returns a proper rotation matrix (det = +1)."""
    z_col = np.array([0.0, 0.0, -1.0])                       # approach = straight down
    closing = np.array([np.cos(closing_angle), np.sin(closing_angle), 0.0])  # eef -Y
    y_col = -closing
    x_col = np.cross(y_col, z_col)                           # right-handed: x = y × z
    x_col /= np.linalg.norm(x_col)
    R = np.column_stack([x_col, y_col, z_col])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, f"non-rotation det={np.linalg.det(R)}"
    return R


def _top_surface_z(mesh: trimesh.Trimesh, cx: float, cy: float, fallback_top: float) -> float:
    """Raycast straight down at (cx, cy) and return the highest surface z there (so a domed/uneven
    top is grasped where the fingers actually land). Falls back to the bbox top if the ray misses."""
    origin = np.array([[cx, cy, fallback_top + 1.0]])
    direction = np.array([[0.0, 0.0, -1.0]])
    try:
        locs, _, _ = mesh.ray.intersects_location(origin, direction)
        if len(locs):
            return float(locs[:, 2].max())
    except Exception:  # noqa: BLE001  (embree/pyembree may be absent; fall back)
        pass
    return float(fallback_top)


def generate_for_object(key: str, mesh_db: dict, n_yaw: int = 6) -> list[dict]:
    """Build the straight-down grasp fan for one object key. Returns a list of grasp dicts in the
    object's LOCAL frame (DB convention), ids starting at 0; the caller renumbers + merges."""
    md = mesh_db["objects"][key]
    R_up = Rot.from_quat(np.asarray(md["upright_orientation_xyzw"], float)).as_matrix()
    mesh = trimesh.load(str(MESH_DIR / md["mesh"]), force="mesh")
    mesh.apply_transform(np.block([[R_up, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))  # -> upright frame

    lo, hi = mesh.bounds
    com = np.asarray(mesh.center_mass, float)                # CoM projected to xy = the centred grasp xy
    cx, cy = float(com[0]), float(com[1])
    top_z = _top_surface_z(mesh, cx, cy, fallback_top=float(hi[2]))
    obj_h = float(hi[2] - lo[2])
    ext_x, ext_y = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    # finger-separation aligned to the LONGER horizontal axis (so the fingers most reliably land on
    # the top for objects where one dim is narrower than the gripper); yaw 0 = that alignment.
    base_angle = 0.0 if ext_x >= ext_y else np.pi / 2
    extent_along_sep = max(ext_x, ext_y)
    if extent_along_sep >= GRIPPER_WIDTH:
        tip_z = top_z - min(GRASP_INSET, 0.4 * obj_h)        # WIDE: press fingertips INTO the top (clear
        mode = "wide(top)"                                   # contact for sticky); never below 40% of height
    else:
        tip_z = top_z - min(STRADDLE_FRAC * obj_h, STRADDLE_MAX)   # NARROW: straddle the upper body
        mode = "narrow(straddle)"
    eef_z = tip_z + FINGER_OFFSET

    print(f"[topdown_gen] {key}: bbox=({ext_x:.3f},{ext_y:.3f},{obj_h:.3f}) com_xy=({cx:.3f},{cy:.3f}) "
          f"top_z={top_z:.3f} {mode} eef_z={eef_z:.3f} n_yaw={n_yaw}", flush=True)

    R_up_T = R_up.T
    pos_upright = np.array([cx, cy, eef_z])
    pos_local = R_up_T @ pos_upright                         # store in the object's LOCAL frame
    grasps = []
    for i in range(n_yaw):
        ang = base_angle + (np.pi * i / n_yaw)               # sweep [0, pi); +pi roll handled by roll_disambig
        R_eef_upright = _eef_R(ang)
        R_eef_local = R_up_T @ R_eef_upright
        q_local = Rot.from_matrix(R_eef_local).as_quat()     # xyzw
        grasps.append({
            "id": i,
            "position": [float(v) for v in pos_local],
            "orientation_xyzw": [float(v) for v in q_local],
            "approach_hint": "top_down",
            "label": f"topdown_yaw{int(np.degrees(np.pi * i / n_yaw))}",
            "source": "topdown_gen",
            "validated": None,
        })
    return grasps


def cabinet_object_keys() -> list[str]:
    """Every distinct target/obstacle object model used across the cabinet_pickup tasks."""
    keys = set()
    for d in sorted(glob.glob("outputs/lerobot_datasets/maniguard-bench/cabinet_pickup/task_*/base")):
        try:
            diag = json.loads(Path(d, "diagnostics.jsonl").read_text().splitlines()[0])
        except Exception:  # noqa: BLE001
            continue
        for role in ("target_info", "obstacle_info"):
            ci = diag.get(role) or {}
            if ci.get("category") and ci.get("model"):
                keys.add(f"{ci['category']}/{ci['model']}")
    return sorted(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=None, help="object keys e.g. fruitcake/nmxadm")
    ap.add_argument("--cabinet-all", action="store_true", help="every cabinet_pickup target/obstacle model")
    ap.add_argument("--n-yaw", type=int, default=6, help="straight-down grasps per object (yaws in [0,pi))")
    ap.add_argument("--apply", action="store_true", help="write into the DB (default: dry-run preview)")
    a = ap.parse_args()

    mesh_db = json.loads(MESH_DB_PATH.read_text())
    keys = list(a.objects or [])
    if a.cabinet_all:
        keys += cabinet_object_keys()
    keys = sorted(set(keys))
    if not keys:
        ap.error("give --objects and/or --cabinet-all")

    ann = json.loads(ANN_PATH.read_text())
    ann.setdefault("objects", {})
    n_ok = 0
    for key in keys:
        if key not in mesh_db.get("objects", {}):
            print(f"[topdown_gen] SKIP {key}: not in mesh_db (run extract_meshes first)", flush=True)
            continue
        try:
            gen = generate_for_object(key, mesh_db, n_yaw=a.n_yaw)
        except Exception as e:  # noqa: BLE001
            print(f"[topdown_gen] FAIL {key}: {type(e).__name__}: {e}", flush=True)
            continue
        if a.apply:
            entry = ann["objects"].setdefault(key, {"grasps": []})
            # keep legacy (non-generated) grasps so OTHER families are unaffected; replace only our own
            kept = [g for g in entry.get("grasps", []) if g.get("source") != "topdown_gen"]
            for j, g in enumerate(gen):
                g["id"] = len(kept) + j                       # gap-free ids within the object
            entry["grasps"] = kept + gen
        n_ok += 1

    if a.apply:
        ANN_PATH.write_text(json.dumps(ann, indent=1))
        print(f"[topdown_gen] APPLIED {n_ok}/{len(keys)} objects -> {ANN_PATH}", flush=True)
    else:
        print(f"[topdown_gen] dry-run: {n_ok}/{len(keys)} objects OK (re-run with --apply to write)", flush=True)


if __name__ == "__main__":
    main()
