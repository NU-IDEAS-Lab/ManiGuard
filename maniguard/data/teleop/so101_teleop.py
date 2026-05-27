"""
SO-101 Teleop Agent for OmniGibson

Receives end-effector pose from the SO-101 ZMQ server and converts it
into TeleopAction for controlling a Franka robot in OmniGibson.

The SO-101 workspace is mapped to Franka workspace using relative deltas:
  delta_pos = (current_so101_ee - prev_so101_ee) * position_scale
  delta_rot = relative_rotation(prev_so101_rot, current_so101_rot)
"""

import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch as th
import zmq

import omnigibson.utils.transform_utils as T


@dataclass
class SO101TeleopConfig:
    """Configuration for SO-101 teleop."""
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5557
    position_scale: float = 5.0     # SO-101 workspace is small (~13cm reach), scale up for Franka
    rotation_scale: float = 1.0     # rotation scaling
    deadzone: float = 0.001         # meters, ignore tiny movements
    # Gripper mapping. The leader reports a 0-1 value (lerobot normalizes raw
    # encoder counts to its calibrated range). Semantics of "which end of the
    # range is physically closed" depend on how you calibrated the SO-101,
    # so both a threshold and an invert flag are exposed.
    gripper_threshold: float = 0.5  # above -> one state, below -> the other
    gripper_invert: bool = False    # flip if your leader reports closed=high
    gripper_debug: bool = False     # print raw value each get_action() call


class SO101TeleopAgent:
    """
    Receives SO-101 EE pose via ZMQ and produces TeleopAction-compatible
    data for OmniGibson's ManipulationRobot.teleop_data_to_action().

    The agent tracks delta motion: how the SO-101 EE moves between frames
    is scaled and applied as the Franka arm command.
    """

    def __init__(self, config: SO101TeleopConfig = None):
        self.config = config or SO101TeleopConfig()

        # ZMQ subscriber
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)  # only keep latest message
        self._sock.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        addr = f"tcp://{self.config.zmq_host}:{self.config.zmq_port}"
        self._sock.connect(addr)
        print(f"SO101TeleopAgent connected to {addr}")

        # State tracking
        self._prev_pos = None
        self._prev_rot = None
        self._prev_gripper = 0.5
        self._connected = False

    def get_action(self, robot):
        """
        Get teleop action for the robot.

        Args:
            robot: OmniGibson robot instance (FrankaPanda)

        Returns:
            th.Tensor: action tensor ready for env.step(), or None if no data
        """
        from omnigibson.utils.teleop_utils import TeleopAction

        msg = self._receive()
        if msg is None:
            # No new data, return zero action
            action = TeleopAction()
            return robot.teleop_data_to_action(action)

        ee_pos = msg["ee_pos"]   # (3,)
        ee_rot = msg["ee_rot"]   # (3,3)
        gripper = msg["gripper"]  # float 0-1

        if self._prev_pos is None:
            # First frame: initialize reference, no movement
            self._prev_pos = ee_pos.copy()
            self._prev_rot = ee_rot.copy()
            self._prev_gripper = gripper
            action = TeleopAction()
            return robot.teleop_data_to_action(action)

        # Compute position delta
        delta_pos = (ee_pos - self._prev_pos) * self.config.position_scale
        if np.linalg.norm(delta_pos) < self.config.deadzone:
            delta_pos = np.zeros(3)

        # Compute rotation delta as euler angles
        # R_delta = R_current @ R_prev^T
        R_delta = ee_rot @ self._prev_rot.T
        delta_euler = self._rotmat_to_euler(R_delta) * self.config.rotation_scale

        # Map gripper to binary-mode native range: +1 = open, -1 = closed.
        is_open = gripper > self.config.gripper_threshold
        if self.config.gripper_invert:
            is_open = not is_open
        gripper_val = 1.0 if is_open else -1.0
        if self.config.gripper_debug:
            print(f"[SO101] gripper raw={gripper:.3f} -> {'OPEN' if is_open else 'CLOSE'}")

        # Build TeleopAction
        # ManipulationRobot expects right[0:3]=pos, right[3:6]=euler, right[6]=gripper
        right_action = th.tensor(
            [delta_pos[0], delta_pos[1], delta_pos[2],
             delta_euler[0], delta_euler[1], delta_euler[2],
             gripper_val],
            dtype=th.float32,
        )

        action = TeleopAction()
        action.right = right_action

        # Update state
        self._prev_pos = ee_pos.copy()
        self._prev_rot = ee_rot.copy()
        self._prev_gripper = gripper

        self._connected = True
        return robot.teleop_data_to_action(action)

    def _receive(self):
        """Try to receive latest message from ZMQ."""
        try:
            data = self._sock.recv(zmq.NOBLOCK)
            return pickle.loads(data)
        except zmq.Again:
            return None

    @staticmethod
    def _rotmat_to_euler(R):
        """Convert 3x3 rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0
        return np.array([roll, pitch, yaw])

    @property
    def is_connected(self):
        return self._connected

    def reset(self):
        """Reset reference pose (call when starting a new episode)."""
        self._prev_pos = None
        self._prev_rot = None

    def stop(self):
        self._sock.close()
        self._ctx.term()
