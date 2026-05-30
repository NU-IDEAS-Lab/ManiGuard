"""Policy I/O transforms for sim 2-cam LIBERO-style datasets.

Task-agnostic: any sim dataset rendered with the ManiGuard 3-cam layout
(image_left / image_right / wrist_image) can be fed to a pi0.5 policy under the
standard LIBERO 2-cam convention by using ONLY the left external overview + the
wrist, and blacking out the third slot:

    base_0_rgb        <- image_left   (external, third-person overview)
    left_wrist_0_rgb  <- wrist_image  (wrist)
    right_wrist_0_rgb <- zeros, masked off   (standard pi0/pi0.5 2-cam layout)

The dataset's ``image_right`` stream is simply not mapped (dropped). This keeps
the policy input identical to openpi's LIBERO convention regardless of which sim
task produced the data — only the dataset (repo_id) and the action semantics
differ between tasks.

State : 8-D — ``[eef_pos(3), axisangle(3), gripper(2)]`` for eef datasets, or
        ``[joint_q(7), gripper_pos(1)]`` for joint datasets.
Action: dataset-native dim, set by the data config via
        ``Sim2CamOutputs(action_dim=...)``: 7 for EEF-delta, 8 for absolute
        joint. Any abs->delta conversion is handled at the data-config level
        (DeltaActions/AbsoluteActions), not here.

Only depends on openpi's public ``transforms`` + ``models.model`` interfaces, so
it works against a pristine openpi clone (no openpi edits required).
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_sim_2cam_example() -> dict:
    """Random observation for smoke-testing a policy server."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do the task",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Sim2CamInputs(transforms.DataTransformFn):
    """Repack a sim 2-cam observation into pi0.5's image dict.

    base_0_rgb <- image_left, left_wrist_0_rgb <- wrist_image. right_wrist_0_rgb
    is zero-filled and masked off (pi0/pi0.5); for pi0-FAST it is unmasked per
    that model's convention.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        image_left = _parse_image(data["observation/image_left"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": image_left,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(image_left),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Sim2CamOutputs(transforms.DataTransformFn):
    """Strip action padding to the dataset's native dim. Inference only.

    action_dim = 7 for EEF-delta datasets, 8 for absolute-joint datasets.
    """

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
