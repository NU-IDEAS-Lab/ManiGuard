# SO-101 Leader Arm → Franka Teleoperation Guide

This guide walks through setting up teleoperation of a simulated Franka Panda robot in OmniGibson using a physical SO-101 leader arm from HuggingFace's LeRobot ecosystem.

## Architecture

```
Terminal 1 (Python 3.12)              Terminal 2 (Python 3.10)
┌─────────────────────┐              ┌──────────────────────────┐
│  lerobot venv        │              │  behavior conda env      │
│                      │              │                          │
│  SO-101 leader arm   │   ZMQ PUB   │  OmniGibson + Franka     │
│  joint reading       │──────────>  │  IK controller           │
│  + Forward Kinematics│   60 Hz     │  + Physics simulation    │
│                      │              │                          │
│  so101_server.py     │              │  so101_franka_teleop.py  │
└─────────────────────┘              └──────────────────────────┘
```

Two separate Python environments are required because LeRobot needs Python 3.12+ while OmniGibson/Isaac Sim requires Python 3.10. ZMQ bridges the gap with sub-millisecond latency.

**Data flow:**
```
SO-101 joints (deg) → FK → EE pose (4x4) → ZMQ → delta computation → Franka IK → env.step()
```

## Prerequisites

- NVIDIA GPU with Isaac Sim support
- SO-101 leader arm with Feetech STS3215 servos
- USB serial connection to the arm's control board
- External power supply for the servos (6-8V; USB only provides communication)

## Step 1: Install LeRobot (Python 3.12 venv)

```bash
# Create a dedicated venv
conda create -n lerobot python=3.12 -y
conda activate lerobot

# Install lerobot and dependencies
pip install lerobot pyzmq placo
```

Verify:
```bash
python -c "from lerobot.teleoperators.so_leader.so_leader import SO101Leader; print('OK')"
```

## Step 2: Install ZMQ in behavior env

```bash
conda activate behavior
pip install pyzmq
```

## Step 3: Hardware Setup

### Connect the SO-101

1. Plug the SO-101 control board into USB
2. Connect the external power supply to the control board (servos need 6-8V)
3. Verify the device appears:
   ```bash
   ls /dev/ttyACM*
   ```
   You should see `/dev/ttyACM0` (or similar).

### Serial port permissions

```bash
sudo usermod -aG dialout $USER
```

**Log out and back in** for the group change to take effect. For a quick temporary fix:
```bash
sudo chmod 666 /dev/ttyACM0
```

### Find the correct port

```bash
conda activate lerobot
lerobot-find-port
```

Follow the prompts: unplug USB → press Enter → replug USB → press Enter. It will report the port.

## Step 4: Calibrate the SO-101

First-time calibration (only needed once per arm):

```bash
conda activate lerobot
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0
```

Follow the on-screen instructions:
1. Move the arm to the middle of its range of motion → press Enter
2. Move each joint (except wrist_roll) through its full range → press Enter

Calibration data is saved automatically and reused on subsequent runs.

> **Note:** The calibration command uses `--teleop.type`, not `--robot.type`. The `--robot.type` flag is for follower arms.

## Step 5: Download the SO-101 URDF

The URDF is needed for forward kinematics computation:

```bash
cd ~/Desktop/projects/SENTINEL-Lite/teleop_bridge
git clone https://github.com/TheRobotStudio/SO-ARM100.git --depth 1
```

The URDF and mesh files are at:
```
SO-ARM100/Simulation/SO101/so101_new_calib.urdf
SO-ARM100/Simulation/SO101/assets/*.stl
```

> **Important:** The URDF references mesh files via relative paths (`assets/...`). You must run the server from the URDF's parent directory, or pass the full absolute path.

## Step 6: Run the Teleop Server

### Mock mode (no hardware, for testing)

```bash
conda activate lerobot
cd ~/Desktop/projects/SENTINEL-Lite/teleop_bridge
python so101_server.py --mock
```

The mock server generates sinusoidal arm motion at 60Hz.

### Real hardware

```bash
conda activate lerobot
cd ~/Desktop/projects/SENTINEL-Lite/teleop_bridge/SO-ARM100/Simulation/SO101
python ~/Desktop/projects/SENTINEL-Lite/teleop_bridge/so101_server.py \
    --port /dev/ttyACM0 \
    --urdf ~/Desktop/projects/SENTINEL-Lite/teleop_bridge/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

You should see:
```
ZMQ PUB bound on tcp://*:5557
SO-101 leader connected on /dev/ttyACM0
FK solver loaded from ...
Publishing at 60.0 Hz. Press Ctrl+C to stop.
```

## Step 7: Run the Franka Teleop Demo

In a **separate terminal**:

```bash
conda activate behavior
cd ~/Desktop/projects/SENTINEL-Lite
python -m omnigibson.examples.teleoperation.so101_franka_teleop \
    --snapshot outputs/pipeline_runs/<run>/scene_ep1.json
```

The snapshot must be a pipeline-generated `scene_ep*.json` (produced by any of
the `task_generation/*_pipeline.py` scripts). To record a trajectory, add
`--output-hdf5 outputs/teleop/<name>.hdf5`.

## Tuning Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pos-scale` | 5.0 | Position sensitivity. SO-101 workspace is ~13cm; increase to amplify motion |
| `--rot-scale` | 1.0 | Rotation sensitivity |
| `--zmq-port` | 5557 | Must match between server and demo |
| `--steps` | 10000 | Number of simulation steps before auto-exit |

**Server-side flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hz` | 60 | Publish rate |
| `--mock` | off | Use mock data (no hardware) |
| `--port` | /dev/ttyACM0 | Serial port for SO-101 |
| `--urdf` | none | Path to SO-101 URDF for FK |

## Troubleshooting

### `ModuleNotFoundError: No module named 'zmq'`
Install pyzmq in the active environment:
```bash
pip install pyzmq
```

### `PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'`
```bash
sudo usermod -aG dialout $USER
# Then log out and back in, or temporarily:
sudo chmod 666 /dev/ttyACM0
```

### `Missing motor IDs` during calibration
- Check that the external power supply is connected and on
- Verify the correct serial port with `lerobot-find-port`
- If servos are new, their IDs may all be 1 (conflict). Connect one at a time and set IDs:
  ```bash
  # In a Python shell with lerobot
  from lerobot.teleoperators.so_leader.so_leader import SO101Leader, SO101LeaderConfig
  config = SO101LeaderConfig(port='/dev/ttyACM0')
  robot = SO101Leader(config)
  robot.setup_motors()  # Follow prompts to set IDs 1-6
  ```

### `Mesh ... could not be found` when loading URDF
The URDF uses relative paths to STL files. Either:
- `cd` into `SO-ARM100/Simulation/SO101/` before running the server
- Pass the absolute path to the URDF file

### `ImportError: cannot import name 'solutions' from 'mediapipe'`
Only relevant if using the original OmniGibson keyboard teleop demo. Fix:
```bash
pip install mediapipe==0.10.14
```

### Isaac Sim segfault / `TypeError: Unable to write from unknown dtype`
numpy 2.x is incompatible with Isaac Sim. Downgrade:
```bash
pip install numpy==1.26.4
```

## File Reference

| File | Environment | Purpose |
|------|-------------|---------|
| `teleop_bridge/so101_server.py` | lerobot (3.12) | Reads SO-101, computes FK, publishes EE pose via ZMQ |
| `OmniGibson/omnigibson/teleop/so101_teleop.py` | behavior (3.10) | ZMQ subscriber, delta computation, TeleopAction generation |
| `OmniGibson/omnigibson/examples/teleoperation/so101_franka_teleop.py` | behavior (3.10) | Teleop entry point: loads arbitrary scene snapshots, records trajectories |
| `OmniGibson/omnigibson/examples/teleoperation/so101_franka_playback.py` | behavior (3.10) | Replay a recorded trajectory HDF5, optionally dump observations |
