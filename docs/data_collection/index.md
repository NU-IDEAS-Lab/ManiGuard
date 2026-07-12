# Data collection

The data-collection stage turns a frozen task (from [task generation](../pipelines/index.md))
into manipulation demonstrations. There are two routes:

1. **Human teleop** — drive the simulated Franka with an SO-101 or GELLO leader
   (raw HDF5 → playback render → SFT HDF5 → LeRobot v2.1).
2. **Scripted datagen** — the mature 6-family pipeline replays a frozen bench task
   and executes it autonomously (cuRobo motion planning + a per-instance human
   grasp-annotation DB), recording success+safe demos and converting RAW → LeRobot
   v2.1. See [Sim datagen pipeline](../datagen/pipeline.md).

```
task-gen / bench frozen scene
        │
        ├─►  human teleop  ─►  raw HDF5  ─►  playback render  ─►  SFT HDF5
        │     (data/teleop)                  (data/playback)         │
        │                                                            ├─►  LeRobot v2.1 → SFT
        └─►  scripted datagen  ─►  RAW (hdf5 + videos)  ─────────────┘
              (data/datagen)         → to_lerobot
```

## The `data/` package

All dataset-producing code lives under `maniguard/data/`, grouped by function:

| Subpackage | Role | Documented in |
|---|---|---|
| `data/teleop/` | SO-101 / GELLO human teleop → raw HDF5 | this section |
| `data/datagen/` | scripted 6-family sim demo collection → RAW → LeRobot | [Sim datagen](../datagen/pipeline.md) |
| `data/playback.py` | replay a teleop HDF5 with physics, render SFT observations | [playback](../teleop/playback.md) |
| `data/lerobot/` | teleop HDF5 → LeRobot v2.1 export + norm stats | [SFT](../sft/dataset_and_config.md) |
| `data/real_teleop/` | real-robot npz → LeRobot DROID (joint) schema | [SFT (real)](../sft/dataset_and_config.md) |
| `data/scene/` | benchmark / scene snapshot utilities (repair, trim, robot rewrite, HF resolve) | [Evaluation](../evaluation/index.md) |
| `data/perturbation_scaling.py` | generate single-level perturbation task sets from base tasks | [Evaluation](../evaluation/index.md) |

The teleop + playback rows are covered here; scripted datagen has its own
[section](../datagen/pipeline.md); the rest produce or prepare data for the SFT and
Evaluation stages and are documented there.

## Route 1 — human teleop

Two physical leaders drive the sim Franka, plus a tool that replays recorded
sessions:

| Leader | Follower control | Entry point |
|---|---|---|
| SO-101 (LeRobot, 5-DoF) | EE delta → IK | [`...teleop.so101_franka_teleop`](../teleop/so101_franka.md) |
| GELLO (7-DoF Dynamixel) | Joint mirroring (no IK) | [`...teleop.gello_franka_teleop`](../teleop/gello_franka.md) |
| Recorded HDF5 | DataPlaybackWrapper | [`...teleop.so101_franka_playback`](../teleop/playback.md) |

The SO-101 leader needs LeRobot (Python 3.12) while OmniGibson / Isaac need
Python 3.10, so it runs split across two processes bridged by a ZMQ PUB/SUB
socket; GELLO's Dynamixel SDK is import-compatible with the `behavior` env and
runs in a single process with 1:1 joint mirroring.

```
Terminal 1 (Python 3.12)              Terminal 2 (Python 3.10)
┌──────────────────────┐             ┌──────────────────────────┐
│  lerobot venv         │   ZMQ PUB  │  behavior conda env       │
│  SO-101 leader arm    │ ─────────► │  OmniGibson + Franka      │
│  joint reading + FK   │   60 Hz    │  IK controller + physics  │
│  so101_server.py      │            │  so101_franka_teleop.py   │
└──────────────────────┘             └──────────────────────────┘
```

Raw teleop HDF5 is then rendered into SFT observations by
[`data/playback.py`](../teleop/playback.md).

## Route 2 — scripted datagen

Instead of a human, the mature datagen pipeline replays a frozen bench task and
executes it autonomously: cuRobo motion planning drives boxy per-family motion
segments, grasps come from a per-instance human annotation DB, and an LTL gate plus
a success gate keep only success+safe demos. This scales demo collection to thousands
of trajectories without hardware, then converts RAW → LeRobot v2.1. See
[Sim datagen pipeline](../datagen/pipeline.md) and
[RAW → LeRobot conversion](../datagen/lerobot_conversion.md).
