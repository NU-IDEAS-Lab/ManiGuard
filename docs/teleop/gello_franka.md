# GELLO → Franka teleop

## What it does

Reads 7 calibrated joint angles from a GELLO leader arm (Dynamixel
servos over USB-FTDI) and drives the Franka Panda in OmniGibson via a
`JointController` (mode = position, absolute) — no IK on the follower
side, since GELLO is kinematically 1:1 with Franka. Loads a
pipeline-generated scene snapshot (`scene_ep*.json`) and optionally
records the trajectory to HDF5. The gripper has no physical leader
counterpart yet and is toggled with the SPACE key.

Differences from the SO-101 path:

- Single process — the Dynamixel SDK runs in the `behavior` conda env
  alongside OmniGibson (no ZMQ bridge required).
- Follower controller is `JointController` (raw radians), not IK.
- On startup, Franka is seeded at a deterministic
  `GELLO_CALIBRATION_FRANKA_POSE` and ramps to the leader's live reading
  over `GELLO_RAMP_STEPS` (60 ≈ 2 s at 30 Hz) to avoid jolts.

## Prerequisites

| Item | Notes |
|---|---|
| Python env | `behavior` conda env (Python 3.10) |
| Hardware | GELLO leader (7 Dynamixel servos, IDs 1-7) + USB-FTDI cable |
| Calibration | per-joint trim constants in `gello_franka_teleop.py` (`GELLO_JOINT_OFFSETS`, `GELLO_JOINT_SIGNS`); regenerate with `behavior-1k/joylo/scripts/gello_get_offset.py` after re-flashing IDs / replacing servos / changing finger geometry. **See [GELLO calibration](gello_calibration.md) for the full procedure.** |
| `joylo` | The `gello.robots.dynamixel` import resolves to `behavior-1k/joylo/` (added to `sys.path` automatically). `joylo` is intentionally not pip-installed — its `setup.py` pulls in unrelated deps (telemoma / pyglm / joycon / pybullet) |
| Snapshot | `scene_ep*.json` produced by any `maniguard.task_generation.*_pipeline` run |

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--snapshot` | (required) | Path to a pipeline `scene_ep*.json`. |
| `--output-hdf5` | None | If set, wraps env in `DataCollectionWrapper` and writes trajectory here. |
| `--only-successes` | off | Only persist successful episodes (toggled with the S key). |
| `--steps` | `10000` | Max sim steps before forced exit. |
| `--gello-port` | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNJP-if00-port0` | USB serial path for the GELLO FTDI. |
| `--invert-gripper` | off | Swap which SPACE state means open vs close. |
| `--start-gripper-open` | off | Begin with the gripper OPEN (default starts CLOSED). |
| `--grasping-mode` | `physical` | `physical` / `assisted` / `sticky` — same semantics as the SO-101 entry. |
| `--no-lid-snap` | off | Disable the eager lid/cap → container snap-attach. |
| `--lid-snap-range-m` | `0.05` | Max lid↔container distance for the eager snap to fire. |
| `--gpu-dynamics` | off | Set `gm.USE_GPU_DYNAMICS=True`. Required for fluid / particle / cloth scenes. Costs VRAM; only enable when needed. |

## Run

```bash
conda activate behavior

python -m maniguard.teleop.gello_franka_teleop \
    --snapshot outputs/teleop_scenes/table/scene_ep0000.json \
    --output-hdf5 outputs/jixing_teleop2_hdf5/table/scene_ep0000.hdf5
```

`VK_ICD_FILENAMES` is set with `os.environ.setdefault` to
`/usr/share/vulkan/icd.d/nvidia_icd.json` before any OmniGibson import. An
explicit shell override still wins.

## Hotkeys

GUI focus on the OmniGibson viewport is required for keyboard events to
register.

| Key | Action |
|---|---|
| SPACE | Toggle gripper open/close |
| S | Toggle success flag (forces save under `--only-successes`) |
| C | Save checkpoint (only when recording) |
| R | Roll back to last checkpoint (only when recording) |
| Q | Clean exit (writes HDF5) |

## How it works

Startup:

1. Connect to the GELLO leader (`DynamixelRobot(joint_ids=1..7,
   joint_offsets=GELLO_JOINT_OFFSETS, joint_signs=GELLO_JOINT_SIGNS)`)
   **before** booting OmniGibson — port-busy / no-power failures surface
   in seconds rather than after a 30-90 s OG startup.
2. `_build_from_snapshot` rewrites the snapshot's robot entry to use
   `JointController` (position, absolute, `action_normalize=False`,
   `use_delta_commands=False`) and overwrites `joint_pos[0:7]` with
   `GELLO_CALIBRATION_FRANKA_POSE` so `env.reset()` doesn't snap to the
   snapshot's saved arm pose.
3. After `env.reset()`, capture `ramp_source = robot.get_joint_positions()[:7]`
   for the ramp.

Per step:

1. `target = leader.get_joint_state()[:7]`.
2. For the first `GELLO_RAMP_STEPS` (=60) steps, blend
   `(1-α) * ramp_source + α * target`; afterwards command `target`
   directly.
3. Pack the gripper command (`±1`, see `--invert-gripper`) and call
   `env.step(action)`.
4. Run `LidSnapper.try_snap(robot=robot)` (unless `--no-lid-snap`).
5. If a sibling `diagnostics.jsonl` exists, evaluate the auto goal checker
   and break on success. The S-key manual override always wins.

When swapping `FrankaMounted → FrankaPanda` from a snapshot, the base is
lifted by 0.5 m. Saved controller goals are nulled (the snapshot's
controller stack — typically OperationalSpace — has an incompatible
goal-state shape).

## Outputs

| Artifact | Notes |
|---|---|
| `<snapshot>_gello_teleop.json` | Rewritten snapshot with the JointController stack — co-located with `--snapshot` |
| HDF5 at `--output-hdf5` | States + actions + transitions, no obs (use `DataPlaybackWrapper` later to materialise obs) |

## Source

`maniguard/teleop/gello_franka_teleop.py`
