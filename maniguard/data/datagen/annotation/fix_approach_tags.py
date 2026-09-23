"""Derive approach_hint from each stored grasp orientation.

Transform the end-effector +Z axis into the upright object frame and classify its
angle from downward: top_down through 60 degrees, side between 60 and 120 degrees,
and bottom_up from 120 degrees. The current classifier has no ambiguous band.
Report changes by default; pass --apply to update grasp_annotations.json.
"""
from __future__ import annotations

import argparse
import json

from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.annotation.mesh_review import (
    ANN,
    FAMILY_STEMS,
    MESH_DB,
    classify_approach,
    obj_in_family,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=None, choices=list(FAMILY_STEMS))
    ap.add_argument("--object", default=None, help="only this object key 'cat/model'")
    ap.add_argument("--apply", action="store_true", help="write corrections back to json")
    args = ap.parse_args()

    ann = json.load(open(ANN))
    src = json.load(open(MESH_DB))["objects"] if MESH_DB.exists() else {}
    keys = [k for k, v in ann["objects"].items() if v.get("grasps")]
    if args.family:
        keys = [k for k in keys if obj_in_family(src.get(k, {}), args.family)]
    if args.object:
        keys = [k for k in keys if k == args.object]

    changes, uncertain, unchanged = [], [], 0
    for key in keys:
        rec = ann["objects"][key]
        R_up = Rot.from_quat(rec["upright_orientation_xyzw"]).as_matrix()
        for gr in rec["grasps"]:
            R_local = Rot.from_quat(gr["orientation_xyzw"]).as_matrix()
            a_world = R_up @ R_local[:, 2]                 # gripper +Z in upright/world
            lab, theta, conf = classify_approach(a_world)
            old = gr.get("approach_hint")
            if not conf:
                uncertain.append((key, gr["id"], old, lab, theta))
            elif old != lab:
                changes.append((key, gr["id"], old, lab, theta))
                gr["approach_hint"] = lab
            else:
                unchanged += 1

    print(f"[fix_approach] {len(keys)} objects | "
          f"{len(changes)} to correct, {len(uncertain)} uncertain, {unchanged} already ok")
    if changes:
        print("\n  CORRECTIONS (confident):")
        for k, gid, old, lab, th in changes:
            print(f"    {k} #{gid}: {old!r} -> {lab!r}  ({th:.0f}deg)")
    if uncertain:
        print("\n  UNCERTAIN — please review & decide (left unchanged):")
        for k, gid, old, lab, th in uncertain:
            print(f"    {k} #{gid}: hint={old!r}, between {lab!r}/side  ({th:.0f}deg)")

    if args.apply and changes:
        json.dump(ann, open(ANN, "w"), indent=2)
        print(f"\n[fix_approach] APPLIED {len(changes)} corrections -> {ANN}")
    elif changes:
        print("\n[fix_approach] dry-run (no --apply): nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
