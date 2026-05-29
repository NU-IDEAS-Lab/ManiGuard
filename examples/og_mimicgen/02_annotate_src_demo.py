#!/usr/bin/env python3
"""Annotate source HDF5 demonstrations with pick/place/open/close signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from _annotation import AnnotationManager
from _common import (
    configure_playback_config,
    create_env,
    dump_json,
    load_env_config_from_hdf5,
    prepare_output_path,
    shutdown_og,
)
from _playback import MinimalPlaybackWrapper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf5", required=True, help="Source teleop HDF5 from stage 1")
    parser.add_argument("--output-dir", required=True, help="Directory for annotations.json and env_config.json")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--playback-frequency", type=float, default=1000.0)
    parser.add_argument("--invert-gripper-action", action="store_true")
    parser.add_argument("--no-place-gripper-check", action="store_true")
    parser.add_argument("--min-finger-contacts-for-pick", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir)
    annotations_path = prepare_output_path(out_dir / "annotations.json", overwrite=args.overwrite)
    env_config_path = prepare_output_path(out_dir / "env_config.json", overwrite=args.overwrite)

    config = configure_playback_config(load_env_config_from_hdf5(args.input_hdf5), frequency=args.playback_frequency)
    env = create_env(config)
    manager = AnnotationManager(
        env,
        robot_id=args.robot_id,
        invert_gripper_action=args.invert_gripper_action,
        check_gripper_open_during_place=not args.no_place_gripper_check,
        min_finger_contacts_for_pick=args.min_finger_contacts_for_pick,
    )

    all_annotations = {}
    with h5py.File(args.input_hdf5, "r") as input_hdf5:
        n_episodes = int(input_hdf5["data"].attrs.get("n_episodes", len(input_hdf5["data"].keys())))
        wrapper = MinimalPlaybackWrapper(
            env,
            input_hdf5,
            step_callback=manager.step,
            episode_start_callback=manager.episode_start_callback,
        )
        for episode_id in range(n_episodes):
            print(f"[Annotate] Replaying demo_{episode_id}")
            wrapper.playback_episode(episode_id)
            all_annotations[f"demo_{episode_id}"] = manager.get_annotations()

    dump_json(all_annotations, annotations_path)
    dump_json(config, env_config_path)
    print(f"[Annotate] wrote {annotations_path}")
    print(f"[Annotate] wrote {env_config_path}")
    shutdown_og()


if __name__ == "__main__":
    main()
