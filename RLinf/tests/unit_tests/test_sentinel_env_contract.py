import torch
import omnigibson.utils.transform_utils as T

from rlinf.envs.sentinel.sentinel_env import (
    build_camera_orientation_quat,
    build_tabletop_oblique_support_view,
    build_droid_left_shoulder_view,
    build_workspace_camera_lookat_point,
    gripper_joint_positions_from_policy_scalar,
    policy_gripper_scalar_from_joint_positions,
    pull_back_camera_eye,
    synthesize_scene_relative_ready_eef_orientation,
    synthesize_scene_relative_ready_eef_position,
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


def test_synthesize_scene_relative_ready_eef_position_stays_above_table_and_inside_surface():
    desired = synthesize_scene_relative_ready_eef_position(
        robot_base_position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        workspace_center=torch.tensor([0.45, 0.25, 0.8], dtype=torch.float32),
        surface_bounds_xy=((0.20, 0.10), (0.60, 0.40)),
        table_top_z=0.75,
        object_top_z=0.82,
        workspace_standoff_m=0.18,
        height_above_table_m=0.30,
        max_height_above_table_m=0.42,
        min_object_clearance_m=0.18,
        surface_margin_m=0.06,
    )

    assert desired.shape == (3,)
    assert 0.26 <= float(desired[0]) <= 0.54
    assert 0.16 <= float(desired[1]) <= 0.34
    assert float(desired[2]) >= 1.00


def test_synthesize_scene_relative_ready_eef_position_caps_tall_object_clearance():
    desired = synthesize_scene_relative_ready_eef_position(
        robot_base_position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        workspace_center=torch.tensor([0.45, 0.25, 0.8], dtype=torch.float32),
        surface_bounds_xy=((0.20, 0.10), (0.60, 0.40)),
        table_top_z=0.75,
        object_top_z=1.55,
        workspace_standoff_m=0.18,
        height_above_table_m=0.30,
        max_height_above_table_m=0.42,
        min_object_clearance_m=0.18,
        surface_margin_m=0.06,
    )

    assert float(desired[2]) <= 1.17
    assert float(desired[2]) >= 1.05


def test_build_tabletop_oblique_support_view_is_workspace_anchored():
    view = build_tabletop_oblique_support_view(
        robot_base_position=[0.0, 0.0, 0.0],
        workspace_center=[0.50, 0.20, 0.90],
        table_top_z=0.78,
        surface_bounds_xy=((0.20, 0.00), (0.80, 0.40)),
        backoff_m=0.60,
        lateral_offset_m=0.42,
        height_above_table_m=0.72,
        lookat_height_above_table_m=0.08,
        pullback_m=0.18,
    )

    assert view["label"] == "tabletop_oblique_support"
    assert len(view["eye"]) == 3
    assert len(view["lookat"]) == 3
    assert abs(view["lookat"][0] - 0.50) < 1e-6
    assert abs(view["lookat"][1] - 0.20) < 1e-6
    assert abs(view["lookat"][2] - 0.86) < 1e-6
    assert view["eye"][2] > view["lookat"][2]


def test_build_droid_left_shoulder_view_stays_robot_side_and_looks_at_workspace():
    view = build_droid_left_shoulder_view(
        robot_base_position=[0.0, 0.0, 0.0],
        target_position=[0.45, 0.20, 0.80],
        workspace_center=[0.50, 0.20, 0.90],
        table_top_z=0.78,
        forward_backoff_m=0.08,
        lateral_offset_m=0.58,
        height_above_table_m=0.72,
        pullback_m=0.18,
    )

    assert view["label"] == "droid_left_shoulder"
    assert len(view["eye"]) == 3
    assert len(view["lookat"]) == 3
    assert abs(view["lookat"][0] - 0.50) < 1e-6
    assert abs(view["lookat"][1] - 0.20) < 1e-6
    assert abs(view["lookat"][2] - 0.90) < 1e-6
    assert view["eye"][1] > 0.0


def test_pull_back_camera_eye_preserves_lookat_and_increases_distance():
    lookat = [0.5, 0.2, 0.9]
    eye = [0.0, 0.6, 1.3]

    pulled = pull_back_camera_eye(eye, lookat, pullback_m=0.18)

    eye_t = torch.tensor(eye, dtype=torch.float32)
    lookat_t = torch.tensor(lookat, dtype=torch.float32)
    pulled_t = torch.tensor(pulled, dtype=torch.float32)
    base_ray = eye_t - lookat_t
    pulled_ray = pulled_t - lookat_t
    base_dist = torch.linalg.norm(base_ray)
    pulled_dist = torch.linalg.norm(pulled_ray)

    assert pulled.shape == (3,)
    assert torch.allclose(base_ray / base_dist, pulled_ray / pulled_dist, atol=1e-6)
    assert float(pulled_dist) > float(base_dist)
    assert abs(float(pulled_dist - base_dist) - 0.18) < 1e-5


def test_build_workspace_camera_lookat_point_stays_on_workspace_semantics():
    lookat = build_workspace_camera_lookat_point(
        workspace_center=[0.45, 0.25, 0.88],
        table_top_z=0.74,
        object_top_z=0.93,
        height_above_table_m=0.10,
    )

    assert lookat.shape == (3,)
    assert torch.allclose(torch.as_tensor(lookat[:2]), torch.tensor([0.45, 0.25], dtype=torch.float32))
    assert float(lookat[2]) >= 0.84
    assert float(lookat[2]) <= 0.93


def test_build_camera_orientation_quat_aligns_camera_forward_with_lookat():
    eye = [0.30, 0.10, 1.05]
    lookat = [0.45, 0.25, 0.86]

    quat = build_camera_orientation_quat(eye, lookat)
    rotation = T.quat2mat(quat)
    forward_world = rotation @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    desired_forward = torch.tensor(lookat, dtype=torch.float32) - torch.tensor(eye, dtype=torch.float32)
    desired_forward = desired_forward / torch.linalg.norm(desired_forward)

    assert quat.shape == (4,)
    assert torch.allclose(forward_world, desired_forward, atol=1e-5)


def test_synthesize_scene_relative_ready_eef_orientation_aims_mounted_wrist_at_workspace():
    pose = synthesize_scene_relative_ready_eef_orientation(
        ready_eef_world_position=[0.33, 0.16, 1.02],
        workspace_center=[0.45, 0.25, 0.88],
        table_top_z=0.74,
        object_top_z=0.93,
        wrist_camera_local_position=[0.05, 0.0, -0.13],
        wrist_camera_local_orientation=[0.7010571, 0.7010573, 0.09229438, 0.09229999],
        wrist_camera_local_position_offset=[0.02, 0.0, 0.04],
        ready_lookat_height_above_table_m=0.10,
    )

    desired_eef_world_orientation = pose["desired_eef_world_orientation"]
    desired_camera_world_orientation = pose["desired_camera_world_orientation"]
    desired_camera_world_position = pose["desired_camera_world_position"]
    desired_camera_lookat = pose["desired_camera_lookat"]
    forward_world = T.quat2mat(desired_camera_world_orientation) @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    desired_forward = desired_camera_lookat - desired_camera_world_position
    desired_forward = desired_forward / torch.linalg.norm(desired_forward)
    recomposed_camera_quat = T.quat_multiply(
        desired_eef_world_orientation,
        pose["effective_wrist_local_orientation"],
    )

    assert desired_eef_world_orientation.shape == (4,)
    assert desired_camera_world_orientation.shape == (4,)
    assert desired_camera_world_position.shape == (3,)
    assert desired_camera_lookat.shape == (3,)
    assert torch.allclose(recomposed_camera_quat, desired_camera_world_orientation, atol=1e-5)
    assert float(torch.dot(forward_world, desired_forward)) > 0.999
