# GELLO grasp batch

## What it does

Per-object grasp data collection via GELLO. Iterates over rows in
`franka_graspability.csv` (or an explicit `--targets` list); for each
object, spawns it on a flat tabletop, lets gravity settle it, resets the
Franka to the calibration pose, and lets the operator drive GELLO to
attempt a grasp. Per object: SPACE toggles gripper, S captures the
current frame as a saved grasp, N advances to the next object (writing
`grasps_{cat}_{model}.pt`), R retries (resets the target + arm), K
skips, Q quits and saves.

Resume is automatic: rows whose `.pt` already exists in `--output-dir`
are skipped (override with `--overwrite`). Output `.pt` format is
bit-compatible with the survey-pipeline writer (`save_grasp_dataset`)
and consumed downstream by
`maniguard.rl.grasps.reset.GraspDatasetResetter`.

## Prerequisites

| Item | Notes |
|---|---|
| Python env | `behavior` conda env (Python 3.10) |
| Hardware | GELLO leader (same setup as the [GELLO → Franka teleop](gello_franka.md) page) |
| CSV | `maniguard/task_generation/utils/franka_graspability.csv` (or pass `--targets cat:model …`) |
| Optional | `gm.DEBUG=True` (`--debug-ag`) renders AG raycast endpoints as small green spheres so you can see whether the rays land inside the object when the gripper closes |

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--csv` | `maniguard/task_generation/utils/franka_graspability.csv` | Source of (category, model) rows. |
| `--output-dir` | `outputs/grasp_datasets/teleop/tensors` | Where `grasps_{cat}_{model}.pt` files land. |
| `--limit` | `50` | Cap on pending objects (0 = no cap). |
| `--exclude-statuses` | `too_large` | Comma-separated CSV statuses to skip. |
| `--targets` | None | Optional `category:model` list overriding the CSV. |
| `--overwrite` | off | Re-do objects whose `.pt` already exists. |
| `--object-xyz` | `0.55 0.0 0.55` | Spawn position; Z above tabletop so the object settles. |
| `--target-rpy` | `0 0 0` | Spawn orientation in degrees (intrinsic ZYX). |
| `--franka-xy` | `0.0 0.0` | Franka base XY. |
| `--franka-z` | `0.72` | Franka base Z. |
| `--table-top-z` | `0.50` | World Z of the tabletop surface (a fixed-base cube object). |
| `--table-size` | `0.8 0.8` | Tabletop XY plan size. |
| `--gello-port` | `GELLO_PORT` (see [GELLO → Franka teleop](gello_franka.md)) | USB serial path for the GELLO FTDI. |
| `--invert-gripper` | off | Swap which SPACE state means open vs close. |
| `--start-gripper-open` | off | Each object starts with gripper OPEN (default CLOSED). |
| `--grasping-mode` | `assisted` | Default differs from the other teleop entries — AG-fired holds are what we want to count for grasp capture. |
| `--gpu-dynamics` | off | Enable `gm.USE_GPU_DYNAMICS` (only needed for fluids). |
| `--debug-ag` | off | `gm.DEBUG=True`; visualises AG raycast endpoints as green spheres. Side effect: verbose OG logs. |

## Run

```bash
conda activate behavior

python -m maniguard.teleop.gello_grasp_batch \
    --csv maniguard/task_generation/utils/franka_graspability.csv \
    --limit 50 \
    --output-dir outputs/grasp_datasets/teleop/tensors
```

Run on an explicit list:

```bash
python -m maniguard.teleop.gello_grasp_batch \
    --targets bowl_abc1234 mug_xyz5678 \
    --output-dir outputs/grasp_datasets/teleop/tensors
```

## Hotkeys

GUI focus on the OmniGibson viewport is required.

| Key | Action |
|---|---|
| SPACE | Toggle gripper open/close |
| S | Capture current frame as a grasp (append to per-object buffer) |
| N | Next object (writes `.pt` if buffer non-empty) |
| R | Retry current object (resets robot + target, clears in-flight `approach_traj`) |
| K | Skip current object (no save) |
| Q | Quit (writes `.pt` for current object if buffer non-empty) |

## How it works

Startup:

1. Read CSV / `--targets`, filter by `--exclude-statuses`, drop already-done
   `.pt` files (unless `--overwrite`), apply `--limit`.
2. Apply globals **before** `og.Environment(...)`: `gm.ENABLE_OBJECT_STATES=True`,
   `gm.ENABLE_TRANSITION_RULES=False`, optional `gm.USE_GPU_DYNAMICS`,
   optional `gm.DEBUG`. Drop AG `GRASP_WINDOW` and `RELEASE_WINDOW` from the
   default ~333 ms (10 action steps) to one physics step (~3.3 ms), so AG
   commits even with operator hand-wobble.
3. Connect GELLO before booting OG (fail fast on USB / power issues).
4. Build a single floor + tabletop env with `FrankaPanda` (`JointController`,
   position, absolute, `action_normalize=False`); seed at
   `GELLO_CALIBRATION_FRANKA_POSE`.

Per object loop:

1. `DatasetObject(category=cat, model=mdl)` is added to the scene at
   `--object-xyz` with `--target-rpy`.
2. `_reset_for_object` releases any prior grasp, restores the robot to the
   anchor pose, drops the object on the table, and steps the sim ~20×
   to settle gravity.
3. Capture `ramp_source = robot.get_joint_positions()[arm_idx]`.
4. Per step: read GELLO (`leader.get_joint_state()[:7]`), ramp from
   `ramp_source` over `GELLO_RAMP_STEPS`, then track live; pack arm +
   gripper command and `env.step()`.
5. On S, `_capture_grasp` records `(rel_position, rel_orientation_xyzw,
   gripper_qpos, arm_joint_pos, approach_traj)` in the object frame.
6. While the gripper is closing without AG firing, every `DBG_AG_EVERY=30`
   steps the loop introspects `_find_gripper_contacts` and
   `_find_gripper_raycast_collisions` to print which gate is rejecting the
   grasp (target in contacts? in raycast set? in their intersection? per-finger
   touch count out of 2?).
7. On N / Q with non-empty buffer, write `grasps_{cat}_{model}.pt` via
   `save_grasp_dataset`. Cleanup mirrors `render_grasps`: release grasp,
   apply 6 open-gripper steps to drain close-torque, remove the object,
   reset the robot.

If OG's articulation view goes null mid-run (a known OG bug —
`get_joint_positions()` returns `None` from then on), the script logs
"FATAL" and `sys.exit(2)`. Re-running resumes from saved `.pt` files.

## Outputs

| Artifact | Notes |
|---|---|
| `grasps_{category}_{model}.pt` | Per-object dict list with `rel_position`, `rel_orientation_xyzw`, `gripper_qpos`, `arm_joint_pos`, `approach_traj`. Format identical to `maniguard.rl.grasps.collector.save_grasp_dataset`'s output, so they drop directly into `GraspDatasetResetter`. |

## Source

`maniguard/teleop/gello_grasp_batch.py`
