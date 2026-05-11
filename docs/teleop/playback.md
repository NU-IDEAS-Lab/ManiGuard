# Teleop playback

## What it does

Replays a trajectory recorded by `so101_franka_teleop.py` or
`gello_franka_teleop.py` (any HDF5 produced via OmniGibson's
`DataCollectionWrapper`). The input HDF5 stores state + action per step
plus the scene config as an attribute, so playback reconstructs the same
env, restores state each step, and applies the recorded actions.
Observations are not in the recording by default — pass `--record` to
re-render RGB into a new HDF5.

## Prerequisites

| Item | Notes |
|---|---|
| Python env | `behavior` conda env (Python 3.10) |
| Input | HDF5 from `DataCollectionWrapper` (any of the teleop entry points produce these via `--output-hdf5`) |

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--input` | (required) | Path to the recorded HDF5. |
| `--output` | `<input>_playback.hdf5` | Output HDF5 (only used when `--record`). |
| `--record` | off | Re-render and store RGB observations in the output HDF5. |
| `--episode` | `0` | Episode id to replay. Ignored when `--all`. |
| `--all` | off | Replay every episode in the HDF5. |
| `--n-render-iterations` | `1` | Higher = cleaner frames, lower = faster. |
| `--with-physics` | off | Keep physics + robot control enabled (live re-simulation). Off by default — objects become visual-only and states are scrubbed frame-by-frame, ~30× faster. |

## Run

Watch in the viewer, no obs dump:

```bash
conda activate behavior

python -m sentinel.teleop.so101_franka_playback \
    --input outputs/teleop/demo.hdf5
```

Re-render RGB into a new HDF5:

```bash
python -m sentinel.teleop.so101_franka_playback \
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

## Outputs

| Artifact | Notes |
|---|---|
| `<output>` | New HDF5 with re-rendered RGB observations. Only when `--record`. |

## Source

`sentinel/teleop/so101_franka_playback.py`
