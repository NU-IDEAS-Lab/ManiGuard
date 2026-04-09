"""
SO-101 Leader Arm ZMQ Server

Runs in the lerobot Python 3.12 venv.
Reads SO-101 leader arm joint positions, computes FK to get end-effector pose,
and publishes via ZMQ for the OmniGibson Franka teleop client.

Usage:
    # In lerobot venv
    python so101_server.py --port /dev/ttyACM0 --zmq-port 5557

    # Mock mode (no hardware)
    python so101_server.py --mock --zmq-port 5557
"""

import argparse
import pickle
import time

import numpy as np
import zmq


class SO101MockReader:
    """Mock SO-101 reader for testing without hardware."""

    def __init__(self):
        self._joint_pos_deg = np.array([0.0, 90.0, -90.0, 0.0, 0.0])  # 5 arm joints
        self._gripper = 0.5  # normalized 0-1
        self._t = 0.0

    def read(self):
        """Return mock joint positions with gentle sinusoidal motion."""
        self._t += 0.016  # ~60Hz
        offset = 5.0 * np.sin(self._t * 0.5)
        joints = self._joint_pos_deg.copy()
        joints[1] += offset  # shoulder_lift oscillates
        gripper = 0.5 + 0.5 * np.sin(self._t * 0.3)
        return joints, gripper


class SO101LeRobotReader:
    """Reads SO-101 leader arm via lerobot SDK."""

    # SO-101 joint names (5 arm joints, excluding gripper)
    ARM_JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    def __init__(self, port: str = "/dev/ttyACM0"):
        from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO101Leader

        config = SO101LeaderConfig(port=port)
        self._robot = SO101Leader(config)
        self._robot.connect()
        print(f"SO-101 leader connected on {port}")

    def read(self):
        """Read current joint positions (degrees) and gripper state (0-100)."""
        # get_action() returns {"motor_name.pos": value, ...}
        action = self._robot.get_action()
        joints = np.array([action[f"{name}.pos"] for name in self.ARM_JOINT_NAMES])
        gripper_raw = action["gripper.pos"]
        # Normalize gripper: lerobot returns 0-100 range, map to 0-1
        gripper = np.clip(gripper_raw / 100.0, 0.0, 1.0)
        return joints, gripper

    def disconnect(self):
        self._robot.disconnect()


class SO101FKComputer:
    """Computes forward kinematics for SO-101 using lerobot's placo-based solver."""

    def __init__(self, urdf_path: str):
        from lerobot.model.kinematics import RobotKinematics

        self._kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=SO101LeRobotReader.ARM_JOINT_NAMES,
        )
        print(f"FK solver loaded from {urdf_path}")

    def compute(self, joint_pos_deg: np.ndarray):
        """Compute EE pose from joint positions in degrees.

        Returns:
            pos: (3,) end-effector position in meters
            rot: (3, 3) rotation matrix
        """
        T = self._kin.forward_kinematics(joint_pos_deg)
        return T[:3, 3].copy(), T[:3, :3].copy()


class SO101MockFK:
    """Mock FK for testing without placo/URDF."""

    def __init__(self):
        # Approximate SO-101 arm lengths (meters)
        self._L1 = 0.065  # shoulder to elbow
        self._L2 = 0.065  # elbow to wrist
        self._L3 = 0.035  # wrist to gripper

    def compute(self, joint_pos_deg: np.ndarray):
        """Simple planar FK approximation for mock testing."""
        q = np.deg2rad(joint_pos_deg)
        # q[0]=pan, q[1]=shoulder_lift, q[2]=elbow, q[3]=wrist_flex, q[4]=wrist_roll
        c_pan, s_pan = np.cos(q[0]), np.sin(q[0])

        # Compute in the vertical plane, then rotate by pan
        elbow_angle = q[1] + q[2]
        wrist_angle = elbow_angle + q[3]

        r = (
            self._L1 * np.cos(q[1])
            + self._L2 * np.cos(elbow_angle)
            + self._L3 * np.cos(wrist_angle)
        )
        z = (
            self._L1 * np.sin(q[1])
            + self._L2 * np.sin(elbow_angle)
            + self._L3 * np.sin(wrist_angle)
        )

        x = r * c_pan
        y = r * s_pan
        pos = np.array([x, y, z])

        # Simple rotation: pan + wrist_roll around z
        total_yaw = q[0] + q[4]
        total_pitch = wrist_angle
        cy, sy = np.cos(total_yaw), np.sin(total_yaw)
        cp, sp = np.cos(total_pitch), np.sin(total_pitch)
        rot = np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp,  cy, sy * sp],
            [-sp,      0,  cp],
        ])

        return pos, rot


def main():
    parser = argparse.ArgumentParser(description="SO-101 ZMQ Teleop Server")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--zmq-port", type=int, default=5557, help="ZMQ publish port")
    parser.add_argument("--hz", type=float, default=60.0, help="Publish rate")
    parser.add_argument("--mock", action="store_true", help="Use mock reader + FK (no hardware)")
    parser.add_argument("--urdf", type=str, default=None, help="Path to SO-101 URDF")
    args = parser.parse_args()

    # Setup ZMQ publisher
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.zmq_port}")
    print(f"ZMQ PUB bound on tcp://*:{args.zmq_port}")

    # Setup reader and FK
    if args.mock:
        reader = SO101MockReader()
        fk = SO101MockFK()
        print("Running in MOCK mode")
    else:
        reader = SO101LeRobotReader(port=args.port)
        if args.urdf is None:
            print("WARNING: No URDF provided, using mock FK. Pass --urdf for real FK.")
            fk = SO101MockFK()
        else:
            fk = SO101FKComputer(urdf_path=args.urdf)

    dt = 1.0 / args.hz
    print(f"Publishing at {args.hz} Hz. Press Ctrl+C to stop.")

    try:
        while True:
            t0 = time.monotonic()

            joints_deg, gripper = reader.read()
            ee_pos, ee_rot = fk.compute(joints_deg)

            msg = {
                "ee_pos": ee_pos,       # (3,) meters
                "ee_rot": ee_rot,       # (3,3) rotation matrix
                "gripper": gripper,     # float 0-1
                "joints_deg": joints_deg,  # (5,) for debugging
                "timestamp": time.time(),
            }
            sock.send(pickle.dumps(msg))

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sock.close()
        ctx.term()
        if hasattr(reader, "disconnect"):
            reader.disconnect()


if __name__ == "__main__":
    main()
