#!/usr/bin/env python3
"""Compute openpi-compatible norm_stats.json for a LeRobot dataset.

OpenPI expects a `norm_stats.json` at
  {model_path}/assets/{asset_id}/norm_stats.json

containing per-feature mean/std/q01/q99 for state and actions. This
script reads a LeRobot v2.1 dataset, streams every frame's `state` and
`actions` columns, and writes the stats.

Run once after the dataset is built, before SFT:

    python tools/compute_norm_stats.py \\
        --dataset-root outputs/lerobot_datasets/sentinel/goblet_pick_place \\
        --output-dir $SENTINEL_PI05_BASE/assets/sentinel_goblet_pick_place
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _compute_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True,
                   help="Path to the LeRobot dataset root, e.g. "
                        "outputs/lerobot_datasets/sentinel/goblet_pick_place")
    p.add_argument("--output-dir", required=True,
                   help="Directory where norm_stats.json will be written. "
                        "Typically {SENTINEL_PI05_BASE}/assets/{asset_id}/")
    args = p.parse_args()

    import pyarrow.parquet as pq

    dataset_root = Path(args.dataset_root)
    parquet_dir = dataset_root / "data"
    if not parquet_dir.exists():
        raise SystemExit(f"No data/ subdir in {dataset_root}; is this a LeRobot dataset?")

    # Concatenate all parquet files. LeRobot 0.3.x used one file per
    # episode (`episode_*.parquet`); 0.4.x switched to one file per chunk
    # (`file-*.parquet`). Catch both.
    parquet_paths = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No *.parquet files under {parquet_dir}")
    print(f"[NormStats] Scanning {len(parquet_paths)} parquet file(s)...")

    states = []
    actions = []
    for path in parquet_paths:
        tab = pq.read_table(path, columns=["state", "actions"])
        # Each row stores a list; convert to numpy.
        states.append(np.array(tab["state"].to_pylist(), dtype=np.float32))
        actions.append(np.array(tab["actions"].to_pylist(), dtype=np.float32))
    state_arr = np.concatenate(states, axis=0)
    action_arr = np.concatenate(actions, axis=0)
    print(f"[NormStats] state: shape={state_arr.shape}, action: shape={action_arr.shape}")

    stats = {
        "norm_stats": {
            "state": _compute_stats(state_arr),
            "actions": _compute_stats(action_arr),
        }
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "norm_stats.json"
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"[NormStats] Wrote {out_path}")


if __name__ == "__main__":
    main()
