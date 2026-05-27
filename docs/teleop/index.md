# Teleoperation

ManiGuard supports two physical leader devices for teleoperating a
simulated Franka Panda inside OmniGibson, plus a replay tool for recorded
sessions and a batched grasp-collection mode.

| Leader | Follower control | Entry point |
|---|---|---|
| SO-101 (LeRobot, 5-DoF) | EE delta → IK | `maniguard.teleop.so101_franka_teleop` |
| GELLO (7-DoF Dynamixel) | Joint mirroring (no IK) | `maniguard.teleop.gello_franka_teleop` |
| GELLO (7-DoF Dynamixel) | Joint mirroring, batched per-object grasp capture | `maniguard.teleop.gello_grasp_batch` |
| Recorded HDF5 | DataPlaybackWrapper | `maniguard.teleop.so101_franka_playback` |

## Two-environment architecture (SO-101)

The SO-101 leader requires LeRobot, which needs Python 3.12. OmniGibson /
Isaac Sim require Python 3.10. The two are bridged with a ZMQ PUB/SUB socket
at sub-millisecond latency. GELLO does not have this constraint — its
Dynamixel SDK is import-compatible with the `behavior` conda env, so the
GELLO entries run in a single process.

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

Data flow:
```
SO-101 joints (deg) → FK → EE pose (4x4) → ZMQ → delta computation
  → Franka IK → env.step()
```

GELLO data flow (single process):
```
GELLO joints (rad) → DynamixelRobot.get_joint_state()
  → Franka JointController (mode=position, absolute) → env.step()
```

## When to use which

| Use case | Leader |
|---|---|
| Quick demos, no GELLO hardware, want to validate pipeline snapshots | SO-101 (or `--mock` server, no hardware) |
| 7-DoF demonstrations for behavior cloning / SFT | GELLO (1:1 joint mapping; no IK redundancy) |
| Batched per-object grasp datasets that drop into `GraspDatasetResetter` | `gello_grasp_batch` |
| Replay a previously-recorded HDF5, optionally re-render observations | `so101_franka_playback` (works for either leader's recordings) |

## Pages

| Page | Script | Notes |
|---|---|---|
| [SO-101 ZMQ server](so101_server.md) | `teleop_bridge/so101_server.py` | Python 3.12 lerobot venv |
| [SO-101 → Franka teleop](so101_franka.md) | `maniguard/teleop/so101_franka_teleop.py` | Python 3.10 behavior env |
| [GELLO → Franka teleop](gello_franka.md) | `maniguard/teleop/gello_franka_teleop.py` | Python 3.10 behavior env |
| [GELLO grasp batch](gello_grasp_batch.md) | `maniguard/teleop/gello_grasp_batch.py` | Per-object .pt grasp dataset |
| [Playback](playback.md) | `maniguard/teleop/so101_franka_playback.py` | DataPlaybackWrapper wrapper |

The SO-101 client logic lives in `maniguard/teleop/so101_teleop.py`
(`SO101TeleopAgent` / `SO101TeleopConfig`); it is imported by
`so101_franka_teleop.py` and is not a standalone entry point.
