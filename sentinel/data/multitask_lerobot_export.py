#!/usr/bin/env python3
"""Multi-task Stage 2: per-task rendered HDF5 -> single LeRobot v2.1 dataset.

Walks a directory tree of the form

    <input_root>/<category>/<task_id>/<base>/scene_ep*.hdf5

(e.g. ``outputs/gello_teleop_rendered/table/task_0000/base/scene_ep1.hdf5``)
and looks up the per-task language prompt from a parallel diagnostics tree

    <diag_root>/<category>/<task_id>/<base>/diagnostics.jsonl

(each line has ``"prompt": "..."`` among other fields).

All episodes from all tasks are merged into ONE LeRobot dataset whose
``meta/tasks.jsonl`` enumerates the unique prompt strings, and each
frame's ``task_index`` resolves to the right prompt at training time.

Schema written matches ``sentinel/data/lerobot_export.py`` (LIBERO-compatible
columns), so openpi's ``LeRobotLiberoDataConfig`` can repack it directly
IF action is 7D. For 8D action (e.g. gello joint-target + gripper), see
the action_dim flag below.

Usage:
    .venv-lerobot/bin/python -m sentinel.data.multitask_lerobot_export \\
        --input-root outputs/gello_teleop_rendered/table \\
        --diag-root datasets/final_unique_accepted-goal_region_sphere-full-perturbed_with_base-20260426/table \\
        --repo-id sentinel/clutter_pickup_libero \\
        --root outputs/lerobot_datasets/sentinel/clutter_pickup_libero \\
        --action-dim 8 \\
        --push-to-hub IDEAS-Lab-Northwestern/sim-clutter-pickup \\
        --hub-private
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch


def _load_prompt(diag_path: Path) -> str:
    """Read the first line of diagnostics.jsonl, extract `prompt`."""
    with diag_path.open() as f:
        first = f.readline()
    if not first.strip():
        raise RuntimeError(f"{diag_path} is empty")
    rec = json.loads(first)
    if "prompt" not in rec:
        raise KeyError(f"{diag_path} first line lacks 'prompt'; keys={list(rec.keys())}")
    return rec["prompt"]


def _load_episode_from_hdf5(path: Path) -> tuple[list[dict], int]:
    """Read one rendered HDF5 -> list of per-frame dicts (frame['task'] is set later).

    Returns (frames, n_action) where N+1 frames of state are paired with
    N actions (last state dropped -- no action follows it).
    """
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if len(demo_keys) != 1:
            raise RuntimeError(f"{path}: expected 1 demo, got {demo_keys}")
        demo = f["data"][demo_keys[0]]
        image = np.asarray(demo["obs/image"])
        wrist_image = np.asarray(demo["obs/wrist_image"])
        state = np.asarray(demo["obs/state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)

    n_action = len(action)
    image = image[:n_action]
    wrist_image = wrist_image[:n_action]
    state = state[:n_action]

    frames = []
    for t in range(n_action):
        frames.append({
            "image": torch.from_numpy(image[t]),
            "wrist_image": torch.from_numpy(wrist_image[t]),
            "state": torch.from_numpy(state[t]),
            "actions": torch.from_numpy(action[t]),
        })
    return frames, n_action


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-root", type=Path, required=True,
                   help="e.g. outputs/gello_teleop_rendered/table")
    p.add_argument("--diag-root", type=Path, required=True,
                   help="parallel tree to --input-root with diagnostics.jsonl per task")
    p.add_argument("--subdir", default="base",
                   help="Subdirectory within each task_id holding scene_ep*.hdf5 + diagnostics.jsonl")
    p.add_argument("--repo-id", required=True, help="Logical LeRobot repo id (metadata only)")
    p.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--state-dim", type=int, default=8)
    p.add_argument("--action-dim", type=int, default=7,
                   help="7 for EEF-delta (matches LIBERO out-of-the-box); 8 for joint-target style.")
    p.add_argument("--push-to-hub", default=None,
                   help="HF repo id to push to (auto-creates v2.1 codebase tag).")
    p.add_argument("--hub-private", action="store_true")
    p.add_argument("--limit-per-task", type=int, default=None,
                   help="(Debug) cap episodes ingested per task_id.")
    args = p.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as e:
        raise SystemExit(f"lerobot not importable: {e}\nInstall lerobot<0.4 in .venv-lerobot.")

    # ----- discover tasks -----
    task_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir() and p.name.startswith("task_")])
    if not task_dirs:
        raise SystemExit(f"No task_* subdirs under {args.input_root}")
    print(f"[Multitask] {len(task_dirs)} task directories under {args.input_root}")

    # ----- prepare LeRobot dataset -----
    state_names = [f"state_{i}" for i in range(args.state_dim)]
    if args.state_dim == 8:
        state_names = ["eef_x", "eef_y", "eef_z",
                       "axisangle_x", "axisangle_y", "axisangle_z",
                       "gripper_l", "gripper_r"]
    action_names = [f"action_{i}" for i in range(args.action_dim)]

    features = {
        "image": {
            "dtype": "video",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (args.state_dim,),
            "names": state_names,
        },
        "actions": {
            "dtype": "float32",
            "shape": (args.action_dim,),
            "names": action_names,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
        robot_type="FrankaPanda",
    )

    # ----- iterate tasks -----
    total_eps = 0
    total_frames = 0
    prompts_seen: set[str] = set()

    for task_dir in task_dirs:
        ep_dir = task_dir / args.subdir
        diag_path = args.diag_root / task_dir.name / args.subdir / "diagnostics.jsonl"

        if not ep_dir.exists():
            print(f"[Multitask] SKIP {task_dir.name}: no {args.subdir}/ subdir")
            continue
        if not diag_path.exists():
            print(f"[Multitask] SKIP {task_dir.name}: no diagnostics at {diag_path}")
            continue

        try:
            prompt = _load_prompt(diag_path)
        except Exception as e:
            print(f"[Multitask] SKIP {task_dir.name}: prompt parse failed -- {e}")
            continue

        ep_files = sorted(ep_dir.glob("scene_ep*.hdf5"))
        if args.limit_per_task is not None:
            ep_files = ep_files[: args.limit_per_task]
        if not ep_files:
            print(f"[Multitask] SKIP {task_dir.name}: no scene_ep*.hdf5 in {ep_dir}")
            continue

        prompts_seen.add(prompt)
        print(f"[{task_dir.name}] {len(ep_files)} eps  prompt='{prompt[:60]}...'")

        for ep_path in ep_files:
            frames, n = _load_episode_from_hdf5(ep_path)
            for frame in frames:
                dataset.add_frame(frame, task=prompt)
            dataset.save_episode()
            total_eps += 1
            total_frames += n

    print(f"\n[Multitask] Done.")
    print(f"  unique prompts: {len(prompts_seen)}")
    print(f"  episodes      : {total_eps}")
    print(f"  total frames  : {total_frames}")
    print(f"  dataset root  : {dataset.root}")

    if args.push_to_hub:
        print(f"\n[Multitask] Pushing to HF: {args.push_to_hub}")
        dataset.repo_id = args.push_to_hub
        dataset.push_to_hub(
            tags=["panda", "omnigibson", "sim", "sentinel", "multitask"],
            license="apache-2.0",
            private=args.hub_private,
            push_videos=True,
            tag_version=True,
        )
        print(f"[Multitask] Pushed + v2.1 tag auto-created.")


if __name__ == "__main__":
    main()
