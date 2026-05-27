"""Recover absolute joint-position trajectories from SFT rollout HDF5s.

The SFT recorder stores per-step ``obs/state`` (eef_8d = [eef_pos(3),
eef_aa(3), gripper_q(2)]) and the full serialized ``og.sim.dump_state``
(``states``, length varies with the number of scene objects). The arm joint
positions live inside that packed dump but are unlabelled. We locate them by
matching the two gripper finger joints (which appear verbatim in the dump)
against ``obs/state[:, 6:8]`` -- the 7 arm joints are the contiguous block
immediately preceding that finger pair (Franka ``get_joint_positions`` order
is [arm0..arm6, finger0, finger1]).

This lets us re-export the existing rollouts with *absolute joint* state and
actions (DROID-style) without re-collecting, so a policy can drive a
JointController directly at eval (no eef->joint IK shim, no task-space drift).
"""

from __future__ import annotations

import h5py
import numpy as np


def locate_joint_block(states: np.ndarray, gripper2: np.ndarray, atol: float = 1e-4) -> int | None:
    """Return the start index of the 9-joint block [arm(7), finger(2)] within
    each serialized ``states`` row, or None if not found.

    ``gripper2`` is ``obs/state[:, 6:8]`` (the two finger joint positions).
    """
    n = min(8, len(states))
    for off in range(7, states.shape[1] - 1):
        if np.allclose(states[:n, off:off + 2], gripper2[:n], atol=atol):
            return off - 7
    return None


def extract_joint_trajectory(hdf5_path: str) -> dict:
    """Extract absolute joint state/action arrays from one rollout HDF5.

    Returns a dict with:
        joint_state  (N, 8) float32 : [arm_q(7), gripper_pos(1, mean finger)]
        joint_action (N, 8) float32 : [arm_q[t+1](7), gripper_cmd(1, binary)]
                                       (last step repeats arm_q to hold)
        eef          (N, 8) float32 : recorded eef_8d (for FK validation)
    The action's arm part is the *next-step* absolute joint target (what a
    position controller should be commanded); the gripper part is the recorded
    binary open/close command (action[:, 6]).
    """
    with h5py.File(hdf5_path, "r") as f:
        d = f["data/demo_0"]
        eef = np.asarray(d["obs/state"], dtype=np.float32)
        states = np.asarray(d["states"], dtype=np.float32)
        act = np.asarray(d["action"], dtype=np.float32)

    grip2 = eef[:, 6:8]
    start = locate_joint_block(states, grip2)
    if start is None:
        raise RuntimeError(f"{hdf5_path}: could not locate joint block in serialized states")

    j9 = states[:, start:start + 9]            # [arm(7), finger(2)]
    arm = j9[:, :7]
    grip_pos = j9[:, 7:9].mean(axis=1, keepdims=True)
    joint_state = np.concatenate([arm, grip_pos], axis=1).astype(np.float32)

    arm_next = np.vstack([arm[1:], arm[-1:]])  # joints[t+1]; last holds
    grip_cmd = act[:, 6:7]                      # recorded binary gripper command
    joint_action = np.concatenate([arm_next, grip_cmd], axis=1).astype(np.float32)

    return {"joint_state": joint_state, "joint_action": joint_action, "eef": eef}
