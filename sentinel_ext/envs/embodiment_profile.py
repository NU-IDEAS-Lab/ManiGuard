from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentinelEmbodimentProfile:
    name: str
    robot_type: str
    robot_name: str
    reset_pose_mode: str
    ready_eef_workspace_blend: float
    ready_eef_standoff_m: float
    ready_eef_height_above_table_m: float
    ready_eef_max_height_above_table_m: float
    ready_eef_min_object_clearance_m: float
    ready_eef_surface_margin_m: float
    ready_eef_lookat_height_above_table_m: float
    ready_gripper_scalar: float | None
    external_camera_resolution: tuple[int, int]
    wrist_camera_resolution: tuple[int, int]
    main_camera_mode: str
    main_camera_backoff_m: float
    main_camera_lateral_offset_m: float
    main_camera_height_above_table_m: float
    main_camera_lookat_height_above_table_m: float
    main_camera_pullback_m: float
    wrist_camera_mode: str
    wrist_camera_lookat_height_above_table_m: float
    wrist_sensor_suffix_priority: tuple[str, ...]
    wrist_sensor_token_priority: tuple[tuple[str, ...], ...]
    wrist_local_position_offset: tuple[float, float, float]
    wrist_local_orientation_override: tuple[float, float, float, float] | None
    reset_settle_steps: int
    arm_controller_name: str
    arm_motor_type: str
    arm_use_delta_commands: bool
    arm_use_impedances: bool
    gripper_controller_name: str
    gripper_mode: str
    gripper_inverted: bool
    gripper_input_limits: tuple[tuple[float], tuple[float]]
    arm_mode: str | None = None
    state_mode: str = "joint"  # "joint" (7 joints + 1 gripper) or "eef" (3 pos + 3 axisangle + 2 gripper_qpos)
    action_dim: int = 8


FRANKA_TABLETOP_SINGLE_ARM_V1 = SentinelEmbodimentProfile(
    name="franka_tabletop_single_arm_v1",
    robot_type="FrankaMounted",
    robot_name="agent_0",
    reset_pose_mode="scene_relative_ready_eef_pose_v2",
    ready_eef_workspace_blend=0.65,
    ready_eef_standoff_m=0.18,
    ready_eef_height_above_table_m=0.30,
    ready_eef_max_height_above_table_m=0.42,
    ready_eef_min_object_clearance_m=0.18,
    ready_eef_surface_margin_m=0.06,
    ready_eef_lookat_height_above_table_m=0.10,
    ready_gripper_scalar=0.0,
    external_camera_resolution=(240, 416),
    wrist_camera_resolution=(240, 416),
    main_camera_mode="droid_left_shoulder_v1",
    main_camera_backoff_m=0.08,
    main_camera_lateral_offset_m=0.58,
    main_camera_height_above_table_m=0.72,
    main_camera_lookat_height_above_table_m=0.08,
    main_camera_pullback_m=0.42,
    wrist_camera_mode="mounted_local_v1",
    wrist_camera_lookat_height_above_table_m=0.10,
    wrist_sensor_suffix_priority=(
        ":camera_link:camera:0",
        ":eef_link:camera:0",
    ),
    wrist_sensor_token_priority=(
        ("camera_link", "camera"),
        ("eef_link", "camera"),
        ("wrist",),
        ("realsense",),
        ("d405",),
        ("hand",),
    ),
    wrist_local_position_offset=(0.02, 0.0, 0.04),
    wrist_local_orientation_override=None,
    reset_settle_steps=10,
    arm_controller_name="JointController",
    arm_motor_type="position",
    arm_use_delta_commands=False,
    arm_use_impedances=False,
    gripper_controller_name="MultiFingerGripperController",
    gripper_mode="smooth",
    gripper_inverted=True,
    gripper_input_limits=((0.0,), (1.0,)),
    arm_mode=None,
    state_mode="joint",
    action_dim=8,
)

FRANKA_TABLETOP_LIBERO_V1 = SentinelEmbodimentProfile(
    name="franka_tabletop_libero_v1",
    robot_type="FrankaMounted",
    robot_name="agent_0",
    reset_pose_mode="scene_relative_ready_eef_pose_v2",
    ready_eef_workspace_blend=0.65,
    ready_eef_standoff_m=0.18,
    ready_eef_height_above_table_m=0.30,
    ready_eef_max_height_above_table_m=0.42,
    ready_eef_min_object_clearance_m=0.18,
    ready_eef_surface_margin_m=0.06,
    ready_eef_lookat_height_above_table_m=0.10,
    ready_gripper_scalar=0.0,
    external_camera_resolution=(256, 256),
    wrist_camera_resolution=(256, 256),
    main_camera_mode="libero_agentview_v1",
    main_camera_backoff_m=0.65,
    main_camera_lateral_offset_m=0.0,
    main_camera_height_above_table_m=0.60,
    main_camera_lookat_height_above_table_m=0.10,
    main_camera_pullback_m=0.0,
    wrist_camera_mode="mounted_local_v1",
    wrist_camera_lookat_height_above_table_m=0.10,
    wrist_sensor_suffix_priority=(
        ":camera_link:camera:0",
        ":eef_link:camera:0",
    ),
    wrist_sensor_token_priority=(
        ("camera_link", "camera"),
        ("eef_link", "camera"),
        ("wrist",),
        ("realsense",),
        ("d405",),
        ("hand",),
    ),
    wrist_local_position_offset=(0.02, 0.0, 0.04),
    wrist_local_orientation_override=None,
    reset_settle_steps=10,
    arm_controller_name="InverseKinematicsController",
    arm_motor_type="velocity",
    arm_use_delta_commands=False,
    arm_use_impedances=False,
    gripper_controller_name="MultiFingerGripperController",
    gripper_mode="smooth",
    gripper_inverted=True,
    gripper_input_limits=((-1.0,), (1.0,)),
    arm_mode="pose_delta_ori",
    state_mode="eef",
    action_dim=7,
)


_PROFILE_REGISTRY = {
    FRANKA_TABLETOP_SINGLE_ARM_V1.name: FRANKA_TABLETOP_SINGLE_ARM_V1,
    FRANKA_TABLETOP_LIBERO_V1.name: FRANKA_TABLETOP_LIBERO_V1,
}


def get_sentinel_embodiment_profile(name: str | None) -> SentinelEmbodimentProfile:
    profile_name = name or FRANKA_TABLETOP_SINGLE_ARM_V1.name
    try:
        return _PROFILE_REGISTRY[profile_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Sentinel embodiment profile: {profile_name}. "
            f"Available profiles: {sorted(_PROFILE_REGISTRY)}"
        ) from exc
