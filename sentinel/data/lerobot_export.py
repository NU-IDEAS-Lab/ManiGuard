#!/usr/bin/env python3
"""Stage 2: rendered HDF5 -> LeRobot v2.1 dataset.

Consumes the output of `tools/playback_teleop_to_hdf5.py` and writes a
LeRobot dataset at ${HF_LEROBOT_HOME:-~/.cache/huggingface/lerobot}/<repo_id>.

Feature layout matches RLinf's OmniGibsonDataConfig repack transform
(RLinf/rlinf/models/embodiment/openpi/dataconfig/omnigibson_dataconfig.py):

    image         (video HxWx3 uint8) -> observation/image
    wrist_image   (video HxWx3 uint8) -> observation/wrist_image
    state         (float32, 7)        -> observation/state
    actions       (float32, 7)        -> actions
    task          (string)            -> prompt  (via LeRobot's task index)

HDF5-side action has N frames, state has N+1. We drop state[-1] (final
state, no action follows) and pair (state[t], action[t]) per frame.

Environment requirements:
    lerobot >= 0.5
    torch, torchvision, numpy, h5py, pyarrow

behavior conda env lacks torchvision. Create a dedicated venv for this
stage and run Stage 2 in it:

    cd ~/Desktop/projects/SENTINEL-Lite
    uv venv --python 3.11 .venv-lerobot
    source .venv-lerobot/bin/activate
    uv pip install lerobot h5py

    python tools/hdf5_to_lerobot.py \
        --input-dir outputs/teleop_rendered \
        --repo-id sentinel/goblet_pick_place \
        --prompt "pick up the goblet and place it on the plate" \
        --fps 30

Resulting dataset is then read by RLinf SFT when
`data.data_path` points at the parent of the repo_id directory.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def _load_episode_from_hdf5(path: str) -> list[dict]:
    """Read one rendered HDF5 -> list of per-frame dicts for LeRobotDataset.

    Expects the structure produced by playback_teleop_to_hdf5.py:
        data/demo_0/obs/image (N+1, H, W, 3) uint8
        data/demo_0/obs/wrist_image (N+1, H, W, 3) uint8
        data/demo_0/obs/state (N+1, 7) float32
        data/demo_0/action (N, 7) float32
    """
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if len(demo_keys) != 1:
            raise RuntimeError(
                f"{path}: expected exactly one demo, got {demo_keys}"
            )
        demo = f["data"][demo_keys[0]]
        image = np.asarray(demo["obs/image"])
        wrist_image = np.asarray(demo["obs/wrist_image"])
        state = np.asarray(demo["obs/state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)

    n_action = len(action)
    # Drop the trailing state (has no paired action). This leaves N frames
    # with (state[t], image[t], action[t]) triples.
    image = image[:n_action]
    wrist_image = wrist_image[:n_action]
    state = state[:n_action]

    # LeRobotDataset.add_frame expects CHW or HWC uint8 -- HWC is fine.
    # We pass torch tensors; lerobot handles the conversion to video.
    frames = []
    for t in range(n_action):
        frames.append({
            "image": torch.from_numpy(image[t]),
            "wrist_image": torch.from_numpy(wrist_image[t]),
            "state": torch.from_numpy(state[t]),
            "actions": torch.from_numpy(action[t]),
        })
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="outputs/teleop_rendered",
                   help="Directory of HDF5s produced by playback_teleop_to_hdf5.py")
    p.add_argument("--repo-id", required=True,
                   help="LeRobot dataset identifier, e.g. sentinel/goblet_pick_place")
    p.add_argument("--prompt", required=True,
                   help="Task description applied to every episode")
    p.add_argument("--root", default=None,
                   help="Root dir where the dataset folder will be written. "
                        "Defaults to $HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot.")
    p.add_argument("--fps", type=int, default=30,
                   help="Playback FPS recorded in dataset metadata. Pick what you want RLinf to think it is.")
    p.add_argument("--resolution", type=int, default=256,
                   help="Image side length (H = W). Must match playback stage.")
    p.add_argument("--single", type=str, default=None,
                   help="If set, only ingest this one HDF5 file (smoke test).")
    args = p.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"lerobot not importable: {e}\n"
            f"Install in an isolated venv (see this script's docstring)."
        )

    # Build feature spec. Flat names match OmniGibsonDataConfig's RepackTransform.
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
            "shape": (8,),
            "names": [
                "eef_x", "eef_y", "eef_z",
                "axisangle_x", "axisangle_y", "axisangle_z",
                "gripper_l", "gripper_r",
            ],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dpos_x", "dpos_y", "dpos_z", "drot_x", "drot_y", "drot_z", "gripper"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,   # passes through to metadata create; None -> default cache
        use_videos=True,
        robot_type="FrankaPanda",
    )

    if args.single is not None:
        input_files = [args.single]
    else:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.hdf5")))
    if not input_files:
        raise SystemExit(f"No HDF5 files found in {args.input_dir}")

    print(f"[Stage2] Writing {len(input_files)} episode(s) to {dataset.root}")
    for i, path in enumerate(input_files):
        print(f"  [{i+1}/{len(input_files)}] {path}")
        frames = _load_episode_from_hdf5(path)
        for frame in frames:
            dataset.add_frame(frame, task=args.prompt)
        dataset.save_episode()

    print(f"\n[Stage2] Done. Dataset root: {dataset.root}")
    print(f"[Stage2] repo_id: {args.repo_id}")
    print(f"[Stage2] Episodes: {len(input_files)}")


if __name__ == "__main__":
    main()
