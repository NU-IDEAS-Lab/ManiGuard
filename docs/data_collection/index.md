# Data collection

The data-collection stage turns a frozen task (from [task generation](../pipelines/index.md))
into manipulation demonstrations. There are two collection methods:

1. **[Teleop](#teleop)** (human-driven) — drive the **simulated** Franka with an
   SO-101 or GELLO leader arm; real Franka teleop captures convert the same way.
2. **[Scripted datagen](#scripted-datagen)** (autonomous, sim only) — the primary
   data engine: the 6-family pipeline replays a frozen bench task and executes it
   autonomously, keeping only success + LTL-safe demos.

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
| `data/teleop/` | SO-101 / GELLO human teleop → raw HDF5 | [Teleop](#teleop) |
| `data/datagen/` | scripted 6-family sim demo collection → RAW → LeRobot | [Scripted datagen](#scripted-datagen) |
| `data/playback.py` | replay a teleop HDF5 with physics, render SFT observations | [playback](../teleop/playback.md) |
| `data/lerobot/` | sim teleop HDF5 → LeRobot v2.1 multitask export | [Sim teleop → LeRobot](../teleop/teleop_to_lerobot.md) |
| `data/real_teleop/` | real-robot npz → LeRobot (DROID joint / sim-compatible HDF5) | [Real teleop](../teleop/real_teleop.md) |
| `data/scene/` | benchmark / scene snapshot utilities (repair, trim, robot rewrite, HF resolve) | [Evaluation](../evaluation/index.md) |
| `data/perturbation_scaling.py` | generate single-level perturbation task sets from base tasks | [Evaluation](../evaluation/index.md) |

## Teleop

Two leader arms drive the simulated Franka — both record the same raw-HDF5 stream,
and the downstream render/export flow is identical:

```
leader arm (SO-101 / GELLO)  ─►  sim Franka  ─►  raw HDF5  ─►  playback render  ─►  LeRobot v2.1
```

| Leader | Mapping | Setup & usage |
|---|---|---|
| **SO-101** (LeRobot, 5-DoF) | EE delta → IK | [SO-101 → Franka](../teleop/so101_franka.md) · [ZMQ server](../teleop/so101_server.md) |
| **GELLO** (7-DoF Dynamixel) | 1:1 joint mirroring | [GELLO → Franka](../teleop/gello_franka.md) · [calibration](../teleop/gello_calibration.md) |

After a session: [Playback / render](../teleop/playback.md) replays the raw HDF5
with physics and renders SFT observations;
[Sim teleop → LeRobot](../teleop/teleop_to_lerobot.md) exports the result to a
LeRobot v2.1 dataset. **Real** Franka teleop captures (npz) convert through the
same export — see [Real-robot teleop](../teleop/real_teleop.md).

Teleop is one way to produce demos (and the reference for task feasibility), but
the datasets that actually feed SFT at scale come from the scripted pipeline below.

## Scripted datagen

Instead of a human, the executor replays a frozen bench task and performs it
**autonomously** — no hardware, no per-trajectory review — scaling collection to
thousands of trajectories. This is the primary data engine behind the shipped
`datagen-<fam>-v1-joint-5cam` datasets.

```
grasp annotation DB ─►┐
                      ├─►  cuRobo planning  ─►  per-family motion segments  ─►  demo
frozen bench task  ─► ┘         (execute + record every step)                   │
                                                                success gate ───┤ keep only
                                                                LTL safety gate ┘ success+safe
```

| What | How |
|---|---|
| Grasps | per-instance **human annotation DB** — authored once per object in a GUI |
| Motion | **cuRobo** plans boxy per-family motion segments (one skeleton per family) |
| Filtering | a demo is kept only if it ends in **success** and was **never LTL-violated** |
| Output | RAW (MP4 + HDF5) → **LeRobot v2.1** per family, ready for SFT |

The full recipe is documented step by step in the **Scripted datagen** section of the
sidebar: [Overview & datasets](../datagen/index.md) →
[Grasp annotation](../datagen/annotation.md) → [Collection](../datagen/collection.md) →
[Review & conversion](../datagen/conversion.md) →
[Families & gotchas](../datagen/families.md).
