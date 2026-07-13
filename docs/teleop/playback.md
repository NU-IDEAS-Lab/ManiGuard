# Teleop playback

## What it does

Replays a trajectory recorded by `so101_franka_teleop.py` or
`gello_franka_teleop.py` (any HDF5 produced via OmniGibson's
`DataCollectionWrapper`). The input HDF5 stores state + action per step
plus the scene config as an attribute, so playback reconstructs the same
env, restores state each step, and applies the recorded actions.
Observations are not in the recording by default — pass `--record` to
re-render RGB into a new HDF5.

!!! note "Two playback entry points"
    - **Teleop → SFT rendering (general):** `maniguard/data/playback.py`
      (`ManiGuardPlaybackWriter`) is the canonical step that turns any teleop HDF5
      into SFT-ready observations. It selects the state convention
      (`--controller joint|eef`, joint is the ManiGuard default) and camera count
      (`--cams 2|3`), and writes the 8-D `obs/state` used by the SFT datasets.
    - **SO-101 replay (this page):** `so101_franka_playback.py` is the SO-101-specific
      replay/render entry documented below.

## Prerequisites

| Item | Notes |
|---|---|
| Python env | `behavior` conda env (Python 3.10) |
| Input | HDF5 from `DataCollectionWrapper` (any of the teleop entry points produce these via `--output-hdf5`) |

## Run

Watch in the viewer, no obs dump:

```bash
conda activate behavior

python -m maniguard.data.teleop.so101_franka_playback \
    --input outputs/teleop/demo.hdf5
```

Re-render RGB into a new HDF5:

```bash
python -m maniguard.data.teleop.so101_franka_playback \
    --input outputs/teleop/demo.hdf5 \
    --output outputs/teleop/demo_obs.hdf5 \
    --record
```

## How it works

`DataPlaybackWrapper.create_from_hdf5` rebuilds the env from the
recording's saved config. `gm.ENABLE_TRANSITION_RULES=False` is required
by `DataPlaybackWrapper`. With `--with-physics`,
`include_robot_control=True` and `include_contacts=True` are set so
playback rolls forward through the physics engine; otherwise object
states are scrubbed directly each frame (visual-only).

`--all` calls `playback_dataset(record_data=…)`; otherwise
`playback_episode(episode_id=…)`. With `--record`, observations are
materialised into the output HDF5; otherwise the playback runs through
the viewer only.

## Source

`maniguard/data/teleop/so101_franka_playback.py`
