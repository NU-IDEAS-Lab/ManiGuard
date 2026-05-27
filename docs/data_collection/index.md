# Data collection

The data-collection stage turns a frozen task (from [task generation](../pipelines/index.md))
into manipulation demonstrations. There are two routes, both producing
SFT-ready HDF5 that the [SFT stage](../openpi_sim_teleop_sft.md) exports to
LeRobot v2.1:

1. **Human teleop** — drive the simulated Franka with an SO-101 or GELLO leader.
2. **Scripted cuRobo** — replay a frozen scene and let cuRobo motion planning
   execute the task autonomously, recording the rollout.

```
task-gen frozen scene
        │
        ├─►  human teleop  ─►  raw HDF5  ─►  playback render  ─┐
        │     (data/teleop)                  (data/playback)    │
        │                                                       ├─►  SFT HDF5
        └─►  cuRobo replay + execute  ─────────────────────────┘     │
              (data/curobo, records via _sft_recorder)               ▼
                                                          data/lerobot export → SFT
```

## The `data/` package

All dataset-producing code lives under `maniguard/data/`, grouped by function:

| Subpackage | Role | Documented in |
|---|---|---|
| `data/teleop/` | SO-101 / GELLO human teleop → raw HDF5 | this section |
| `data/curobo/` | scripted cuRobo demo collection → SFT HDF5 | [cuRobo demo collection](curobo.md) |
| `data/playback.py` | replay a teleop HDF5 with physics, render SFT observations | [playback](../teleop/playback.md) |
| `data/lerobot/` | HDF5 → LeRobot v2.1 export, norm stats, joint-action recovery | [SFT](../openpi_sim_teleop_sft.md) |
| `data/real_teleop/` | real-robot npz → sim HDF5 / LeRobot DROID schema | [SFT (real)](../openpi_real_teleop_sft.md) |
| `data/scene/` | benchmark / scene snapshot utilities (repair, trim, robot rewrite, HF resolve) | [Evaluation](../one_machine_pro6000_eval.md) |
| `data/perturbation_scaling.py` | generate single-level perturbation task sets from base tasks | [Evaluation](../one_machine_pro6000_eval.md) |

The first three rows are *collection* and covered here; the rest produce or
prepare data for the SFT and Evaluation stages and are documented there.

## Route 1 — human teleop

Two physical leaders drive the sim Franka, plus a tool that replays recorded
sessions:

| Leader | Follower control | Entry point |
|---|---|---|
| SO-101 (LeRobot, 5-DoF) | EE delta → IK | [`...teleop.so101_franka_teleop`](../teleop/so101_franka.md) |
| GELLO (7-DoF Dynamixel) | Joint mirroring (no IK) | [`...teleop.gello_franka_teleop`](../teleop/gello_franka.md) |
| GELLO (batched grasp capture) | Joint mirroring, per-object `.pt` | [`...teleop.gello_grasp_batch`](../teleop/gello_grasp_batch.md) |
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

## Route 2 — scripted cuRobo

Instead of a human, cuRobo motion planning executes the task on a replayed
frozen scene and the rollout is recorded directly as SFT data. This scales demo
collection without hardware. See [cuRobo demo collection](curobo.md).

!!! tip "cuRobo interactive data collection"
    We also have an interactive cuRobo mode where you directly drag a **ghost
    gripper** to the desired end-effector pose and cuRobo plans the arm to reach
    it — ideal for specifying grasp points on hard-to-grasp objects. See
    [ghost-gripper teleop](curobo.md#interactive-ghost-gripper-teleop).

## Grasp data

[`data/curobo/collect_grasps_in_scene.py`](curobo.md#grasp-collection) gathers
reachable Franka grasps for a target in a benchmark scene (GraspGen proposals +
reachability filtering) — feeding both scripted collection and the
[GraspGen / cuRobo grasp pipeline](../graspgen_pipeline.md) used for RL resets.
