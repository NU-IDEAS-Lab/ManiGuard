import torch

from rlinf.envs.sentinel.sentinel_env import (
    DEFAULT_POLICY_WRIST_LOCAL_POSITION_OFFSET,
    compute_policy_wrist_local_orientation,
    compute_policy_wrist_local_pose,
    resolve_wrist_sensor_name,
)


def test_resolve_wrist_sensor_name_prefers_camera_link_over_eef_link():
    robot_obs = {
        "agent_0:eef_link:Camera:0": {"rgb": object()},
        "agent_0:camera_link:Camera:0": {"rgb": object()},
    }

    sensor_name, reason, candidates = resolve_wrist_sensor_name(robot_obs)

    assert sensor_name == "agent_0:camera_link:Camera:0"
    assert reason == "exact_suffix::camera_link:camera:0"
    assert candidates == [
        "agent_0:eef_link:Camera:0",
        "agent_0:camera_link:Camera:0",
    ]


def test_resolve_wrist_sensor_name_uses_eef_link_for_local_franka_asset():
    robot_obs = {
        "agent_0:eef_link:Camera:0": {"rgb": object()},
    }

    sensor_name, reason, candidates = resolve_wrist_sensor_name(robot_obs)

    assert sensor_name == "agent_0:eef_link:Camera:0"
    assert reason == "exact_suffix::eef_link:camera:0"
    assert candidates == ["agent_0:eef_link:Camera:0"]


def test_resolve_wrist_sensor_name_falls_back_to_single_rgb_sensor():
    robot_obs = {
        "agent_0:unexpected_mount:Camera:0": {"rgb": object()},
    }

    sensor_name, reason, candidates = resolve_wrist_sensor_name(robot_obs)

    assert sensor_name == "agent_0:unexpected_mount:Camera:0"
    assert reason == "single_rgb_fallback"
    assert candidates == ["agent_0:unexpected_mount:Camera:0"]


def test_compute_policy_wrist_local_pose_uses_default_offset():
    base = torch.tensor([0.05, 0.0, -0.13], dtype=torch.float32)

    pose = compute_policy_wrist_local_pose(base)

    assert torch.allclose(
        pose,
        base + torch.tensor(DEFAULT_POLICY_WRIST_LOCAL_POSITION_OFFSET, dtype=torch.float32),
    )


def test_compute_policy_wrist_local_pose_prefers_explicit_override():
    base = torch.tensor([0.05, 0.0, -0.13], dtype=torch.float32)
    override = [0.07, 0.0, -0.09]

    pose = compute_policy_wrist_local_pose(
        base,
        local_position_offset=[9.0, 9.0, 9.0],
        local_position_override=override,
    )

    assert torch.allclose(pose, torch.tensor(override, dtype=torch.float32))


def test_compute_policy_wrist_local_orientation_uses_base_orientation_by_default():
    base = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    orientation = compute_policy_wrist_local_orientation(base)

    assert torch.allclose(orientation, base)


def test_compute_policy_wrist_local_orientation_prefers_explicit_override():
    base = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    override = [0.5, 0.5, 0.5, 0.5]

    orientation = compute_policy_wrist_local_orientation(
        base,
        local_orientation_override=override,
    )

    assert torch.allclose(orientation, torch.tensor(override, dtype=torch.float32))
