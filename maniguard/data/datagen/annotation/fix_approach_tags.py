"""Correct the hand-typed ``approach_hint`` of each annotated grasp from its ACTUAL stored
eef pose — NO sim.

The viser tool's ``approach_hint`` is a manual dropdown (defaults to top_down) and is often
left wrong (e.g. a side grasp still tagged top_down). This derives the true approach class
from the grasp orientation (gripper +Z = approach), expressed in the object's UPRIGHT/world
frame, and:
  * CONFIDENT class that differs from the stored hint  -> corrected in grasp_annotations.json
  * AMBIGUOUS (in-between top_down/side or side/bottom_up band) -> left as-is and REPORTED
    for the user to decide.

Default is a dry-run (report only). Pass --apply to write the corrections back.

  conda activate behavior
  PYTHONPATH=$HOME/project/ManiGuard \
  python -m maniguard.data.datagen.annotation.fix_approach_tags [--family clutter] [--apply]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.annotation.mesh_review import (
    ANN, ANN_DIR, FAMILY_STEMS, MESH_DB, classify_approach,
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
        stems = tuple(FAMILY_STEMS[f] for f in args.family)
        keys = [k for k in keys
                if str(src.get(k, {}).get("source_task", "")).startswith(stems)]
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
