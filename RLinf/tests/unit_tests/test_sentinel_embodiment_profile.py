from rlinf.envs.sentinel.embodiment_profile import (
    FRANKA_TABLETOP_SINGLE_ARM_V1,
    get_sentinel_embodiment_profile,
)
from rlinf.envs.sentinel.sentinel_env import SentinelEnv


def test_get_sentinel_embodiment_profile_returns_franka_tabletop_v1_by_default():
    profile = get_sentinel_embodiment_profile(None)

    assert profile == FRANKA_TABLETOP_SINGLE_ARM_V1
    assert profile.name == "franka_tabletop_single_arm_v1"
    assert profile.robot_type == "FrankaMounted"
    assert profile.main_camera_mode == "droid_left_shoulder_v1"
    assert profile.main_camera_pullback_m > 0.0
    assert profile.wrist_camera_mode == "mounted_local_v1"


def test_apply_embodiment_profile_to_env_cfg_propagates_sensor_resolutions():
    env = SentinelEnv.__new__(SentinelEnv)
    env.embodiment_profile = FRANKA_TABLETOP_SINGLE_ARM_V1
    env_cfg = {
        "env": {
            "external_sensors": [
                {
                    "sensor_kwargs": {
                        "image_height": 1,
                        "image_width": 1,
                    }
                }
            ]
        },
        "robots": [
            {
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": 1,
                            "image_width": 1,
                        }
                    }
                }
            }
        ],
    }

    env._apply_embodiment_profile_to_env_cfg(env_cfg)

    assert env_cfg["env"]["external_sensors"][0]["sensor_kwargs"] == {
        "image_height": 240,
        "image_width": 416,
    }
    assert env_cfg["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"] == {
        "image_height": 240,
        "image_width": 416,
    }


def test_profile_exposes_scene_relative_ready_pose_contract():
    env = SentinelEnv.__new__(SentinelEnv)
    env.embodiment_profile = FRANKA_TABLETOP_SINGLE_ARM_V1

    assert FRANKA_TABLETOP_SINGLE_ARM_V1.reset_pose_mode == "scene_relative_ready_eef_pose_v2"
    assert FRANKA_TABLETOP_SINGLE_ARM_V1.ready_eef_standoff_m > 0.0
    assert FRANKA_TABLETOP_SINGLE_ARM_V1.ready_eef_height_above_table_m > 0.0
    assert FRANKA_TABLETOP_SINGLE_ARM_V1.ready_eef_max_height_above_table_m >= (
        FRANKA_TABLETOP_SINGLE_ARM_V1.ready_eef_height_above_table_m
    )
    assert FRANKA_TABLETOP_SINGLE_ARM_V1.ready_eef_lookat_height_above_table_m > 0.0
    assert env.embodiment_profile.ready_gripper_scalar == 0.0
