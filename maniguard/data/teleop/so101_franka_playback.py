"""Replay a trajectory recorded by so101_franka_teleop.py.

The input HDF5 stores state+action per step (plus the scene config as an
attr), so replay reconstructs the same env, restores state each step, and
applies the recorded actions. Observations are (optionally) re-rendered
into an output HDF5.

Examples:
    # Watch in the viewer, no obs dump
    python -m maniguard.data.teleop.so101_franka_playback \\
        --input outputs/teleop/demo.hdf5

    # Dump RGB observations to a new HDF5
    python -m maniguard.data.teleop.so101_franka_playback \\
        --input outputs/teleop/demo.hdf5 \\
        --output outputs/teleop/demo_obs.hdf5 --record
"""

import argparse
import os

import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to collected hdf5")
    p.add_argument("--output", default=None,
                   help="Path to output hdf5 (required when --record). "
                        "Defaults to <input>_playback.hdf5.")
    p.add_argument("--record", action="store_true",
                   help="Record RGB observations into the output HDF5.")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode id to replay (default 0).")
    p.add_argument("--all", action="store_true",
                   help="Replay every episode in the HDF5 instead of one.")
    p.add_argument("--n-render-iterations", type=int, default=1,
                   help="Higher = cleaner frames, lower = faster (default 1).")
    p.add_argument("--with-physics", action="store_true",
                   help="Keep physics + robot control enabled during playback. "
                        "Slower but closer to a live re-simulation. Off by default: "
                        "objects become visual-only and states are just 'scrubbed' "
                        "frame by frame, which is ~30x faster.")
    args = p.parse_args()

    # Required by DataPlaybackWrapper
    gm.ENABLE_TRANSITION_RULES = False

    if args.output is None:
        stem, ext = os.path.splitext(args.input)
        args.output = f"{stem}_playback{ext or '.hdf5'}"

    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=args.input,
        output_path=args.output,
        robot_obs_modalities=["rgb"] if args.record else [],
        n_render_iterations=args.n_render_iterations,
        only_successes=False,
        overwrite=True,
        include_robot_control=args.with_physics,
        include_contacts=args.with_physics,
    )

    if args.all:
        env.playback_dataset(record_data=args.record)
    else:
        env.playback_episode(episode_id=args.episode, record_data=args.record)

    if args.record:
        env.save_data()
        print(f"[Playback] Observations saved to: {args.output}")
    else:
        print("[Playback] Done. (no observations recorded — pass --record to dump obs)")

    og.clear()


if __name__ == "__main__":
    main()
