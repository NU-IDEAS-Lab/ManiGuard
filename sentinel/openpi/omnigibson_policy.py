"""
Transforms for OmniGibson / IsaacLab-style environments with 7D EEF control.
State: eef_pos(3) + axisangle(3) + gripper(1) = 7D
Action: delta_pos(3) + delta_rot(3) + gripper(1) = 7D, gripper binarized via sign()
"""
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OmniGibsonInputs(transforms.DataTransformFn):
    """Convert OmniGibson observations to openpi model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OmniGibsonOutputs(transforms.DataTransformFn):
    """Convert openpi model outputs to OmniGibson action format.
    Extracts 7D actions and binarizes the gripper command via sign().
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7])
        # Binarize gripper (last dim) to {-1, +1}
        actions[:, -1] = np.sign(actions[:, -1])
        return {"actions": actions}
