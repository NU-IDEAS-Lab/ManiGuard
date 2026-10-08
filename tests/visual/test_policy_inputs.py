from types import SimpleNamespace

import numpy as np
import torch


def test_policy_receives_selected_native_source_pixels_without_client_resize():
    from maniguard.eval.benchmark import extract_obs, query_policy
    from maniguard.eval.eval_config import EvalConfig

    frames = {
        key: np.full((640, 640, 3), i, np.uint8)
        for i, key in enumerate(("image_left", "image_right", "image_left_shoulder", "image_opposite", "wrist_image"))
    }
    env = SimpleNamespace(_maniguard_source_capture=SimpleNamespace(last_observation={"camera_frames": frames}))
    robot = SimpleNamespace(
        default_arm="0",
        gripper_control_idx={"0": torch.tensor([7, 8])},
        arm_control_idx={"0": torch.arange(7)},
        get_joint_positions=lambda: torch.zeros(9),
        get_relative_eef_position=lambda: torch.zeros(3),
        get_relative_eef_orientation=lambda: torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    cfg = EvalConfig(state_mode="joint", external_cam="right", action_dim=8)
    obs = extract_obs(env, robot, "example", cfg)
    assert obs["overview_image"] is frames["image_right"]
    assert obs["wrist_images"] is frames["wrist_image"]
    sent = []

    def infer(payload):
        sent.append(payload)
        return {"actions": np.zeros((8, 8))}

    query_policy(SimpleNamespace(infer=infer), obs, "openpi", cfg)
    np.testing.assert_array_equal(sent[0]["observation/image_left"], frames["image_right"])
    np.testing.assert_array_equal(sent[0]["observation/wrist_image"], frames["wrist_image"])
