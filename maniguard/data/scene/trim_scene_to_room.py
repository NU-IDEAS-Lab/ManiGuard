#!/usr/bin/env python3
"""Trim benchmark scene snapshots to the target room only.

Reads diagnostics.jsonl for the room, removes objects from other rooms,
assigns in_rooms to pipeline-spawned objects (task objects + robot) that
have empty in_rooms, and overwrites scene_ep1.json in place.

Usage:
    # Single scene
    python -m maniguard.data.scene.trim_scene_to_room \
        datasets/safety-benchmark/transfer_trial20_benchmark_safe/trial_0

    # Batch: all scenes under a benchmark root
    python -m maniguard.data.scene.trim_scene_to_room \
        datasets/safety-benchmark --batch
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

STRUCTURAL_CATEGORIES = {"walls", "floors", "ceilings", "door", "window"}


def trim_scene_info_to_room(scene_info: dict, room: str, *, keep_robot: bool = True) -> tuple[dict, dict]:
    """Return a trimmed copy of ``scene_info`` that keeps only one room.

    This is the in-memory variant of ``trim_scene`` and is intended for
    lightweight validation / runtime prep where we do not want to write an
    intermediate scene file to disk.
    """
    trimmed = copy.deepcopy(scene_info)
    init = trimmed.get("objects_info", {}).get("init_info", {})
    state_reg = trimmed.get("state", {}).get("registry", {}).setdefault("object_registry", {})

    keep, drop = [], []
    for name, info in list(init.items()):
        cat = info.get("args", {}).get("category", "")
        rooms = info.get("args", {}).get("in_rooms", [])
        cls_mod = info.get("class_module", "")
        is_robot = "robot" in cls_mod.lower()

        if is_robot:
            if keep_robot:
                info["args"].setdefault("in_rooms", [])
                if room not in info["args"]["in_rooms"]:
                    info["args"]["in_rooms"].append(room)
                keep.append(name)
            else:
                drop.append(name)
        elif rooms and room in rooms:
            keep.append(name)
        elif rooms and room not in rooms:
            drop.append(name)
        elif cat and cat not in STRUCTURAL_CATEGORIES:
            info["args"]["in_rooms"] = [room]
            keep.append(name)
        else:
            drop.append(name)

    for name in drop:
        init.pop(name, None)
        state_reg.pop(name, None)

    return trimmed, {
        "room": room,
        "kept": len(keep),
        "dropped": len(drop),
    }


def trim_scene(scene_dir: Path, dry_run: bool = False) -> dict:
    scene_file = scene_dir / "scene_ep1.json"
    diag_file = scene_dir / "diagnostics.jsonl"

    if not scene_file.is_file() or not diag_file.is_file():
        return {"status": "skip", "reason": "missing files"}

    diag = json.loads(diag_file.read_text(encoding="utf-8").strip().split("\n")[0])
    room = (diag.get("support_selection") or {}).get("room_instance")
    if not room:
        return {"status": "skip", "reason": "no room_instance in diagnostics"}

    scene = json.loads(scene_file.read_text(encoding="utf-8"))
    scene_class = scene.get("init_info", {}).get("class_name", "")
    if scene_class != "InteractiveTraversableScene":
        return {"status": "skip", "reason": f"class={scene_class}, not InteractiveTraversableScene"}
    scene, trim_stats = trim_scene_info_to_room(scene, room, keep_robot=True)

    if not dry_run:
        scene_file.write_text(json.dumps(scene, indent=2), encoding="utf-8")

    return {
        "status": "trimmed",
        "room": room,
        "kept": trim_stats["kept"],
        "dropped": trim_stats["dropped"],
    }


def iter_scene_dirs(root: Path):
    """Yield every dir under root that has scene_ep1.json + diagnostics.jsonl."""
    for p in sorted(root.rglob("scene_ep1.json")):
        d = p.parent
        if (d / "diagnostics.jsonl").is_file():
            yield d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Scene directory or benchmark root (with --batch)")
    p.add_argument("--batch", action="store_true", help="Process all scenes under path recursively")
    p.add_argument("--dry-run", action="store_true", help="Don't write, just report")
    args = p.parse_args()

    if args.batch:
        dirs = list(iter_scene_dirs(args.path))
    else:
        dirs = [args.path]

    total = trimmed = skipped = 0
    for d in dirs:
        total += 1
        result = trim_scene(d, dry_run=args.dry_run)
        if result["status"] == "trimmed":
            trimmed += 1
            print(f"  {d.relative_to(args.path) if args.batch else d.name}: "
                  f"keep={result['kept']} drop={result['dropped']} room={result['room']}")
        else:
            skipped += 1
            if result["reason"] != "missing files":
                print(f"  {d.relative_to(args.path) if args.batch else d.name}: "
                      f"skip ({result['reason']})")

    print(f"\nTotal: {total}, trimmed: {trimmed}, skipped: {skipped}")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
