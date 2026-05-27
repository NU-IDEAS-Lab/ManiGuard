# SO-101 → Franka teleop

## What it does

Subscribes to the [SO-101 ZMQ server](so101_server.md) and drives a Franka
Panda inside OmniGibson. SO-101 EE pose deltas (between successive ZMQ
messages) are scaled and forwarded as `TeleopAction.right` to the Franka's
`InverseKinematicsController`; the gripper is mapped from the SO-101's
0-1 normalized value to the Franka gripper's binary `+1` / `-1`. Loads a
pipeline-generated scene snapshot (`scene_ep*.json`) and optionally records
the trajectory to HDF5 via `DataCollectionWrapper`. Runs in the `behavior`
conda env (Python 3.10).

## Prerequisites

| Item | Notes |
|---|---|
| Python env | `behavior` conda env, with `pip install pyzmq` |
| Server running | `python teleop_bridge/so101_server.py` (or `--mock`) in a separate terminal |
| Snapshot | `scene_ep*.json` produced by any `maniguard.task_generation.*_pipeline` run |
| Optional | `diagnostics.jsonl` next to the snapshot — if present, an auto goal-checker fires success when the recorded goal region is satisfied |

## Run

Mock SO-101 (no hardware) + sample snapshot:

```bash
# Terminal 1: lerobot venv
python teleop_bridge/so101_server.py --mock

# Terminal 2: behavior env
conda activate behavior
python -m maniguard.data.teleop.so101_franka_teleop \
    --snapshot outputs/pipeline_runs/<run>/scene_ep1.json
```

Real hardware + record to HDF5, only saving successes:

```bash
python -m maniguard.data.teleop.so101_franka_teleop \
    --snapshot outputs/pipeline_runs/<run>/scene_ep1.json \
    --output-hdf5 outputs/teleop/<name>.hdf5 \
    --only-successes
```

## How it works

`_build_from_snapshot` loads the saved scene, finds the `Franka*` entry in
`objects_info.init_info`, swaps its controller stack to
`InverseKinematicsController` + `MultiFingerGripperController` (binary
mode), and writes the rewritten snapshot next to the original as
`*_teleop.json`. Saved controller goals are dropped (the snapshot was
recorded with a different controller stack whose goal-state shape doesn't
match IK). When swapping `FrankaMounted → FrankaPanda`, the base is lifted
by 0.5 m so the arm doesn't sit on the floor.

The main loop calls `SO101TeleopAgent.get_action(robot)` from
`maniguard/data/teleop/so101_teleop.py`:

1. Pull the latest ZMQ message (CONFLATE=1, RCVTIMEO=100 ms).
2. Compute `delta_pos = (ee_pos - prev_ee_pos) * position_scale` (gated by
   a `1 mm` deadzone).
3. Compute `delta_euler` from `R_current @ R_prev.T`, scaled by
   `rotation_scale`.
4. Map `gripper > threshold` (with optional invert) to `+1` / `-1`.
5. Pack into a 7-vector (`pos[3], euler[3], gripper`), wrap as a
   `TeleopAction(right=…)`, and convert via
   `robot.teleop_data_to_action(action)`.

If `diagnostics.jsonl` is present next to the snapshot, an auto goal
checker (`maniguard.eval.goal_checker.build_goal_checker`) breaks the loop
the moment the goal region fires.

A separate `LidSnapper` (from `maniguard.utils.lid_attach`) runs after each
`env.step()` and eagerly attaches a lid/cap to its container when placed
within range and the gripper has released — no-op when no eligible pair is
in the scene.

## Hotkeys

| Key | Action |
|---|---|
| Q | Clean exit (flushes HDF5; preferred over Ctrl+C, which Isaac Sim's carb layer intercepts and can leave the HDF5 truncated to a 96-byte header) |
| C | Save checkpoint (only when recording) |
| R | Roll back to last checkpoint (only when recording) |
| S | Toggle manual success override for the current episode (only when recording) |

## Source

`maniguard/data/teleop/so101_franka_teleop.py` — entry point.
`maniguard/data/teleop/so101_teleop.py` — `SO101TeleopAgent` / `SO101TeleopConfig`.
