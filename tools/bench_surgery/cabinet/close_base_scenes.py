"""Method-2 surgical edit: close the drawer in every cabinet_pickup BASE scene.

The bench spawns the target drawer 0.2-open (joint_pos[target] = 0.2*stroke), which made the
both-mode datagen do a redundant Phase-1 "close first" reach that a tall in-path obstacle blocks
(see docs/superpowers/plans/2026-06-24-cabinet-closed-drawer-method2.md). We instead spawn ALL
drawers fully CLOSED, so the demo flow is the natural open -> place -> close.

This is an OFFLINE, point-by-point edit of the saved scene_ep1.json (NOT the task-generation
pipeline): set the cabinet articulation's joint_pos / joint_vel to zero (all drawers shut). A
gravity idle-step at re-finalize (cabinet_rerender_base.py) settles the rest.

Idempotent: backs up each scene to scene_ep1.json.bak_method2 ONCE (the original open state), then
zeroes the joints. Re-runs skip the backup so the open-state backup is never clobbered.

Usage:
    python tools/bench_surgery/cabinet/close_base_scenes.py            # apply to all 35 base scenes
    python tools/bench_surgery/cabinet/close_base_scenes.py --dry-run  # report only, no writes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

BENCH = "outputs/lerobot_datasets/maniguard-bench/cabinet_pickup"


def _cabinet_key(registry: dict) -> str:
    keys = [k for k in registry if "bamfsz" in k or "cabinet" in k.lower()]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly 1 cabinet object, found {keys}")
    return keys[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scenes = sorted(glob.glob(os.path.join(args.bench, "task_*/base/scene_ep1.json")))
    if not scenes:
        raise SystemExit(f"no base scenes under {args.bench}")
    print(f"{len(scenes)} base scenes")

    for f in scenes:
        d = json.load(open(f))
        reg = d["state"]["registry"]["object_registry"]
        cab = reg[_cabinet_key(reg)]
        before = list(cab["joint_pos"])
        nonzero = [(i, round(v, 4)) for i, v in enumerate(before) if abs(v) > 1e-4]
        task = os.path.basename(os.path.dirname(os.path.dirname(f)))

        if not nonzero:
            print(f"  {task}: already closed {before} -> skip")
            continue

        if args.dry_run:
            print(f"  {task}: open {nonzero} -> would zero")
            continue

        bak = f + ".bak_method2"
        if not os.path.exists(bak):
            shutil.copy2(f, bak)                       # preserve the ORIGINAL open state once

        cab["joint_pos"] = [0.0] * len(cab["joint_pos"])
        cab["joint_vel"] = [0.0] * len(cab["joint_vel"])
        json.dump(d, open(f, "w"))
        print(f"  {task}: closed (was {nonzero}); backup -> {os.path.basename(bak)}")


if __name__ == "__main__":
    main()
