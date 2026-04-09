import torch

from rlinf.envs.sentinel.sentinel_env import (
    gripper_joint_positions_from_policy_scalar,
    policy_gripper_scalar_from_joint_positions,
)


def test_policy_gripper_scalar_uses_open_zero_closed_one_convention():
    lower = torch.tensor([0.0, 0.0], dtype=torch.float32)
    upper = torch.tensor([0.04, 0.04], dtype=torch.float32)

    open_joints = torch.tensor([0.04, 0.04], dtype=torch.float32)
    closed_joints = torch.tensor([0.0, 0.0], dtype=torch.float32)

    open_scalar = policy_gripper_scalar_from_joint_positions(open_joints, lower, upper)
    closed_scalar = policy_gripper_scalar_from_joint_positions(closed_joints, lower, upper)

    assert torch.allclose(open_scalar, torch.tensor([0.0], dtype=torch.float32))
    assert torch.allclose(closed_scalar, torch.tensor([1.0], dtype=torch.float32))


def test_gripper_joint_positions_from_policy_scalar_inverts_back_to_joint_space():
    lower = torch.tensor([0.0, 0.0], dtype=torch.float32)
    upper = torch.tensor([0.04, 0.04], dtype=torch.float32)

    open_joints = gripper_joint_positions_from_policy_scalar(0.0, lower, upper)
    half_closed_joints = gripper_joint_positions_from_policy_scalar(0.5, lower, upper)
    closed_joints = gripper_joint_positions_from_policy_scalar(1.0, lower, upper)

    assert torch.allclose(open_joints, torch.tensor([0.04, 0.04], dtype=torch.float32))
    assert torch.allclose(half_closed_joints, torch.tensor([0.02, 0.02], dtype=torch.float32))
    assert torch.allclose(closed_joints, torch.tensor([0.0, 0.0], dtype=torch.float32))
