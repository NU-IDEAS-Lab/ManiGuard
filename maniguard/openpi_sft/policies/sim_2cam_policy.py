"""Map ManiGuard observations to openpi policy inputs and trim output action padding.

Use observation/image_left as base_0_rgb and observation/wrist_image as
left_wrist_0_rgb. Fill right_wrist_0_rgb with zeros; mask it for pi0/pi0.5 and
leave it enabled for PI0_FAST. The data configuration selects which dataset
overview supplies observation/image_left.

Pass state, prompt, and optional actions through. Sim2CamOutputs returns the
configured native action dimensions; joint delta/absolute conversion belongs
to the data configuration.
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
