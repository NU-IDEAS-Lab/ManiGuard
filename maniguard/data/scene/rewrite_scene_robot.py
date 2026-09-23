#!/usr/bin/env python3
"""Copy a scene snapshot with a selected Franka robot class and IK controller.

Rewrites the first Franka entry, clears its saved controller goals and
adds ``--z-offset`` to the base position (default: 0.5 m). The offset is
applied regardless of the source robot class. Companion diagnostics and
an optional safety JSON file are copied to the new directory.

Usage:
    python -m maniguard.data.scene.rewrite_scene_robot \
        /path/to/source_scene /path/to/output_scene --z-offset 0.5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def rewrite_snapshot_robot(
    src_snapshot: Path,
    dst_snapshot: Path,
    *,
    z_offset: float = 0.5,
    target_robot: str = "FrankaPanda",
) -> None:
    """Set the first Franka entry to target_robot and raise its base by z_offset.

    Replace the controller configuration with IK and binary gripper control,
    and clear saved controller goals to avoid loading incompatible state.
    """
    snap = json.loads(src_snapshot.read_text(encoding="utf-8"))

    robot_key = None
    for k, info in snap.get("objects_info", {}).get("init_info", {}).items():
        if "Franka" in info.get("class_name", ""):
            robot_key = k
            break
    if robot_key is None:
        raise RuntimeError(f"No Franka robot found in {src_snapshot}")

    entry = snap["objects_info"]["init_info"][robot_key]
    entry["class_module"] = "omnigibson.robots.franka"
    entry["class_name"] = target_robot
    entry["args"].pop("expected_file_hash", None)
    # Use IK arm commands and binary gripper commands for this output snapshot.
    entry["args"]["controller_config"] = {
        "arm_0": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": None,
            "mode": "binary",
        },
    }

    # Stub the controller state: the snapshot was taken with an OSC
    # controller (no control_filter field); FrankaMounted's default load
    # is OSC too. Our IK controller would try to rehydrate filter state
    # that doesn't exist, so blank the goals.
    robot_state = (
        snap.get("state", {}).get("registry", {}).get("object_registry", {}).get(robot_key)
    )
    if robot_state is not None:
        robot_state["controllers"] = {
            "arm_0": {"goal_is_valid": False, "goal": None},
            "gripper_0": {"goal_is_valid": False, "goal": None},
        }
        root_link = robot_state.get("root_link")
        if root_link is not None and "pos" in root_link and z_offset:
            root_link["pos"][2] = float(root_link["pos"][2]) + float(z_offset)
    if z_offset:
        args_pos = entry.get("args", {}).get("position")
        if args_pos is not None:
            args_pos[2] = float(args_pos[2]) + float(z_offset)

    dst_snapshot.parent.mkdir(parents=True, exist_ok=True)
    dst_snapshot.write_text(json.dumps(snap, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("src_dir", type=Path,
                   help="Source scene dir (must contain scene_ep1.json + diagnostics.jsonl)")
    p.add_argument("dst_dir", type=Path,
                   help="Destination scene dir; will be created")
    p.add_argument("--z-offset", type=float, default=0.5,
                   help="Meters to raise the robot base along z. Set 0 to swap robot only.")
    p.add_argument("--target-robot", default="FrankaPanda",
                   help="Target robot class name (default FrankaPanda)")
    args = p.parse_args()

    src = args.src_dir.resolve()
    dst = args.dst_dir.resolve()
    if not (src / "scene_ep1.json").is_file():
        raise SystemExit(f"Missing scene_ep1.json in {src}")
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}")

    rewrite_snapshot_robot(
        src / "scene_ep1.json",
        dst / "scene_ep1.json",
        z_offset=args.z_offset,
        target_robot=args.target_robot,
    )
    # Copy the companion files verbatim (diagnostics.jsonl, ltl_safety.json
    # if present, ...). The registry keys don't depend on which robot is
    # in the scene, so they are reusable as-is.
    for side_file in ["diagnostics.jsonl", "ltl_safety.json"]:
        src_side = src / side_file
        if src_side.is_file():
            shutil.copy2(src_side, dst / side_file)

    print(f"[rewrite] {src.name} -> {dst.name}")
    print(f"[rewrite]   robot: FrankaMounted -> {args.target_robot}")
    print(f"[rewrite]   z offset: +{args.z_offset} m")
    print(f"[rewrite] wrote: {dst}")


if __name__ == "__main__":
    main()
