"""Execute a cuRobo joint trajectory via the JointController — Layer-1 primitive.

The datagen robot runs the ``joint_position_impedance`` preset (a JointController arm
in absolute-position mode + a MultiFingerGripperController). To run a cuRobo segment
we feed each ``(T, 7)`` arm waypoint straight into the arm action slot (raw radians,
no clipping — the preset's ``command_input_limits=None``) and hold the gripper at a
binary command; the impedance drive tracks the waypoints. Every stepped frame is
handed to the recorder (``record_step(arm_q_cmd, gripper_cmd)``), which is the ONLY
place the arm is commanded — exactly the contract the recorder assumes.

Replicated clean from ``collector._build_action`` / ``_build_hold_action`` (the action
assembly) — datagen does not import that tree.
"""
from __future__ import annotations

from typing import Any

# Gripper command convention (MultiFingerGripperController smooth mode + JointController
# both read +1/-1 as the joint upper/lower limit = fully open / fully closed).
OPEN = +1.0
CLOSE = -1.0


def build_joint_action(robot, arm_q, gripper_cmd: float):
    """Flat action vector: arm slot ← absolute joint targets ``arm_q`` (7,), gripper
    slot ← ``gripper_cmd`` (+1 open / -1 close). Assumes the datagen preset
    (JointController arm + MultiFingerGripper)."""
    import torch as th

    arm = robot.default_arm
    action = th.zeros(robot.action_dim, dtype=th.float32)
    action[robot.arm_action_idx[arm]] = th.as_tensor(arm_q, dtype=th.float32).reshape(-1)
    action[robot.gripper_action_idx[arm]] = float(gripper_cmd)
    return action


def execute_trajectory(env, robot, arm_traj, *, gripper_cmd: float = OPEN,
                       recorder: Any = None, steps_per_waypoint: int = 1) -> int:
    """Drive the arm through ``arm_traj`` (T,7) absolute-joint waypoints via the
    JointController, holding the gripper at ``gripper_cmd``. Records every stepped
    frame if ``recorder`` is set. Returns the number of env steps taken.

    The recorder reads the ACHIEVED joints after each step; ``arm_q_cmd`` is the
    commanded waypoint. The arm is moved ONLY here (never a stray zero action)."""
    import torch as th

    n = 0
    for q in arm_traj:
        q = th.as_tensor(q, dtype=th.float32).reshape(-1)
        action = build_joint_action(robot, q, gripper_cmd)
        q_np = q.cpu().numpy()
        for _ in range(int(steps_per_waypoint)):
            env.step(action)
            if recorder is not None:
                recorder.record_step(arm_q_cmd=q_np, gripper_cmd=gripper_cmd)
            n += 1
    return n


def actuate_gripper(env, robot, *, close: bool, n_steps: int,
                    recorder: Any = None) -> int:
    """Hold the arm at its current joints and drive the gripper open/closed for
    ``n_steps`` (settle before / close after a grasp). Records every step if
    ``recorder`` is set. Returns the number of env steps taken."""
    arm = robot.default_arm
    gripper_cmd = CLOSE if close else OPEN
    n = 0
    for _ in range(int(n_steps)):
        cur_q = robot.get_joint_positions()[robot.arm_control_idx[arm]]
        action = build_joint_action(robot, cur_q, gripper_cmd)
        env.step(action)
        if recorder is not None:
            recorder.record_step(arm_q_cmd=cur_q.cpu().numpy(), gripper_cmd=gripper_cmd)
        n += 1
    return n
