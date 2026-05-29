#!/usr/bin/env python3
"""Extract object-centric waypoint trajectories from annotated source demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from _common import (
    configure_playback_config,
    create_env,
    load_env_config_from_hdf5,
    load_json_or_yaml,
    prepare_output_path,
    shutdown_og,
)
from _playback import MinimalPlaybackWrapper
from _waypoints import WaypointExtractor, find_subsequence, save_waypoints_hdf5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf5", required=True, help="Source teleop HDF5 from stage 1")
    parser.add_argument("--annotations", required=True, help="annotations.json from stage 2")
    parser.add_argument("--output-hdf5", required=True, help="Path to write waypoints.hdf5")
    parser.add_argument("--env-config", default=None, help="Optional env_config.json from stage 2")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--expected-sequence", default=None, help="Optional comma-separated sequence, e.g. pick,place")
    parser.add_argument("--playback-frequency", type=float, default=1000.0)
    parser.add_argument("--eef-z-offset", type=float, default=0.0)
    parser.add_argument("--pick-min-start-distance", type=float, default=0.15)
    parser.add_argument("--pick-max-distance", type=float, default=0.30)
    parser.add_argument("--place-min-start-distance", type=float, default=0.15)
    parser.add_argument("--place-max-distance", type=float, default=0.30)
    parser.add_argument("--open-min-start-distance", type=float, default=0.05)
    parser.add_argument("--open-max-distance", type=float, default=0.50)
    parser.add_argument("--close-min-start-distance", type=float, default=0.05)
    parser.add_argument("--close-max-distance", type=float, default=0.50)
    parser.add_argument("--end-distance-threshold", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_annotations(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _build_parser().parse_args()
    output_path = prepare_output_path(args.output_hdf5, overwrite=args.overwrite)
    config = load_json_or_yaml(args.env_config) if args.env_config else load_env_config_from_hdf5(args.input_hdf5)
    config = configure_playback_config(config, frequency=args.playback_frequency)
    expected_sequence = (
        [item.strip() for item in args.expected_sequence.split(",") if item.strip()]
        if args.expected_sequence
        else []
    )
    annotations = _load_annotations(args.annotations)

    env = create_env(config)
    all_waypoints = {}
    with h5py.File(args.input_hdf5, "r") as input_hdf5:
        n_episodes = int(input_hdf5["data"].attrs.get("n_episodes", len(input_hdf5["data"].keys())))
        for episode_id in range(n_episodes):
            episode_key = f"demo_{episode_id}"
            signals = annotations.get(episode_key, [])
            selected_signals = find_subsequence(signals, expected_sequence)
            if selected_signals is None:
                print(f"[Waypoints] skipping {episode_key}: did not find sequence {expected_sequence}")
                continue
            episode_grp = input_hdf5["data"][episode_key]
            action_dataset = episode_grp.get("action", episode_grp.get("actions"))
            max_frames = int(episode_grp.attrs.get("num_samples", len(action_dataset)))
            extractor = WaypointExtractor(
                env,
                selected_signals,
                pick_min_start_distance=args.pick_min_start_distance,
                pick_max_distance=args.pick_max_distance,
                place_min_start_distance=args.place_min_start_distance,
                place_max_distance=args.place_max_distance,
                open_min_start_distance=args.open_min_start_distance,
                open_max_distance=args.open_max_distance,
                close_min_start_distance=args.close_min_start_distance,
                close_max_distance=args.close_max_distance,
                end_distance_threshold=args.end_distance_threshold,
                max_frames=max_frames,
                robot_id=args.robot_id,
                eef_z_offset=args.eef_z_offset,
            )
            wrapper = MinimalPlaybackWrapper(
                env,
                input_hdf5,
                step_callback=extractor.step,
                episode_start_callback=extractor.episode_start_callback,
            )
            print(f"[Waypoints] Replaying {episode_key}")
            wrapper.playback_episode(episode_id)
            extractor.finalize()
            all_waypoints[episode_key] = extractor.get_waypoints()

    save_waypoints_hdf5(all_waypoints, str(output_path))
    print(f"[Waypoints] wrote {output_path}")
    shutdown_og()


if __name__ == "__main__":
    main()
