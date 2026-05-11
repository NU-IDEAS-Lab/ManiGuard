# SO-101 ZMQ server

## What it does

Reads joint positions from a physical SO-101 leader arm (5 arm joints +
gripper) over USB serial, computes forward kinematics to get the
end-effector pose, and publishes the result as a pickled dict on a ZMQ PUB
socket at a fixed rate (default 60 Hz). Runs in the `lerobot` Python 3.12
venv. A `--mock` mode emits a sinusoidal trajectory with no hardware.

## Prerequisites

| Item | Notes |
|---|---|
| Python env | dedicated `lerobot` venv (Python 3.12) — `pip install lerobot pyzmq placo` |
| Hardware | SO-101 leader arm + Feetech STS3215 servos + 6-8 V external power |
| USB | `/dev/ttyACM*` accessible (`sudo usermod -aG dialout $USER`, then re-login) |
| Calibration | run `lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0` once per arm |
| URDF | clone `https://github.com/TheRobotStudio/SO-ARM100.git` to get `Simulation/SO101/so101_new_calib.urdf` and the `assets/*.stl` meshes (URDF references meshes via relative paths, so run from the URDF's parent directory or pass an absolute path) |

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--port` | `/dev/ttyACM0` | Serial port for the SO-101 control board. |
| `--zmq-port` | `5557` | TCP port for `zmq.PUB`. Must match the consumer. |
| `--hz` | `60.0` | Publish rate. |
| `--mock` | off | Use `SO101MockReader` + `SO101MockFK` (sinusoidal motion, no hardware, no URDF). |
| `--urdf` | None | Path to `so101_new_calib.urdf`. Without it, real-hardware runs fall back to the planar mock FK and print a warning. |

## Run

Mock mode (no hardware):

```bash
conda activate lerobot
python teleop_bridge/so101_server.py --mock
```

Real hardware with FK:

```bash
conda activate lerobot
cd teleop_bridge/SO-ARM100/Simulation/SO101
python <repo>/teleop_bridge/so101_server.py \
    --port /dev/ttyACM0 \
    --urdf <repo>/teleop_bridge/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

Expected startup log:
```
ZMQ PUB bound on tcp://*:5557
SO-101 leader connected on /dev/ttyACM0
FK solver loaded from ...
Publishing at 60.0 Hz. Press Ctrl+C to stop.
```

## How it works

Per tick (~16.7 ms at 60 Hz):

1. `SO101LeRobotReader.read()` calls `lerobot`'s `SO101Leader.get_action()`
   to obtain `{"shoulder_pan.pos": deg, ..., "gripper.pos": 0-100}`. Arm
   joints are returned in degrees; gripper is normalised to `[0, 1]`.
2. `SO101FKComputer.compute()` (placo-based `RobotKinematics`) targets
   `gripper_frame_link` and returns a 4x4 transform; the server splits it
   into `(pos: (3,), rot: (3,3))`.
3. The dict is pickled and sent via `zmq.PUB` on `tcp://*:<zmq-port>`.

Message schema (pickled Python dict):

| Field | Type | Notes |
|---|---|---|
| `ee_pos` | `np.ndarray (3,)` | EE position in metres |
| `ee_rot` | `np.ndarray (3,3)` | EE rotation matrix |
| `gripper` | `float` | normalised 0-1 |
| `joints_deg` | `np.ndarray (5,)` | raw arm joints (debugging) |
| `timestamp` | `float` | `time.time()` at publish |

## Outputs

A single ZMQ PUB socket on `tcp://*:<zmq-port>` (default `5557`). One
message per tick, no on-disk artifacts.

## Source

`teleop_bridge/so101_server.py`
