"""Bench surgery: move / re-yaw the hinged jar of a jar_transport base task (JSON-only edit).

Extreme-placement tail tasks (far / bad lid-flop direction) get their jar TRANSLATED toward the
robot and/or YAW-rotated so the flopped lid faces a workable direction. The content item moves
rigidly with the jar (same translation; positions rotate about the jar's vertical axis for yaw).
The goal region, support, and robot are untouched. Re-finalize afterwards with
``tools/bench_surgery/jar/rerender_base.py`` (behavior env python) to settle, re-monitor LTL and re-render.

Usage:
  python -m tools.bench_surgery.jar.move_task --task task_0011 --dxy 0.0 0.08        # translate (m, world XY)
  python -m tools.bench_surgery.jar.move_task --task task_0015 --yaw-deg 90          # rotate about jar centre
  (both flags may be combined; translation applies after the rotation)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/jar_transport")
JAR = "target_hinged_jar_ep1_1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dxy", nargs=2, type=float, default=(0.0, 0.0))
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--robot-z", type=float, default=None,
                    help="set the robot (agent_0) base z. Family norm: sitting surface + 0~2cm; "
                         "task_0014's bench build hung the base on the desk's middle-divider top "
                         "edge, 33cm above the actual surface.")
    ap.add_argument("--lid-deg", type=float, default=None,
                    help="set the lid hinge joint angle (deg). A 179deg dead-flat lid on a high "
                         "pedestal leaves NO under-lid room for the ride bar; 150 matches the "
                         "natural rest angle of the other jar models (still fully open).")
    args = ap.parse_args()

    base = BENCH / args.task / "base"
    sp = base / "scene_ep1.json"
    bak = base / "scene_ep1.json.bak_move"
    if not bak.exists():
        shutil.copy2(sp, bak)
    sc = json.loads(sp.read_text())
    reg = sc["state"]["registry"]["object_registry"]

    jar = reg[JAR]["root_link"]
    jp = np.asarray(jar["pos"], float)
    jq = np.asarray(jar["ori"], float)
    d = np.array([args.dxy[0], args.dxy[1], 0.0], float)
    rot = R.from_euler("z", args.yaw_deg, degrees=True)

    items = [k for k in reg if k.startswith("food_")]
    for name in [JAR] + items:
        e = reg[name]["root_link"]
        p = np.asarray(e["pos"], float)
        q = np.asarray(e["ori"], float)
        p2 = jp + rot.apply(p - jp) + d                    # rigid about the jar centre, then slide
        q2 = (rot * R.from_quat(q)).as_quat()
        e["pos"] = [float(x) for x in p2]
        e["ori"] = [float(x) for x in q2]
        e["lin_vel"] = [0.0, 0.0, 0.0]
        e["ang_vel"] = [0.0, 0.0, 0.0]
        print(f"[move] {name}: pos {np.round(p,3).tolist()} -> {np.round(p2,3).tolist()}"
              + (f"  yaw{args.yaw_deg:+.0f}deg" if args.yaw_deg else ""))

    if args.robot_z is not None:
        rl = reg["agent_0"]["root_link"]
        old_z = float(rl["pos"][2])
        rl["pos"] = [float(rl["pos"][0]), float(rl["pos"][1]), float(args.robot_z)]
        rl["lin_vel"] = [0.0, 0.0, 0.0]
        rl["ang_vel"] = [0.0, 0.0, 0.0]
        print(f"[move] agent_0: base z {old_z:.3f} -> {args.robot_z:.3f}")

    if args.lid_deg is not None:
        jj = reg[JAR]["joint_pos"]
        old = float(np.ravel(np.asarray(jj, float))[0])
        reg[JAR]["joint_pos"] = [float(np.radians(args.lid_deg))]
        reg[JAR]["joint_vel"] = [0.0]
        print(f"[move] {JAR}: lid angle {np.degrees(old):.0f}deg -> {args.lid_deg:.0f}deg")

    sp.write_text(json.dumps(sc))
    print(f"[move] {args.task} written (backup: {bak.name})")


if __name__ == "__main__":
    main()
