# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Panda policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaEEOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class FrankaEEInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Padding is handled later by openpi's
    # PadStatesAndActions transform after normalization/tokenization.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False

    def __call__(self, data: dict) -> dict:
        assert data["observation/state"].shape == (8,), (
            f"Expected state shape (8,), got {data['observation/state'].shape}"
        )
        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(
                np.asarray(data["observation/state"]).copy()
            ).float()

        state = data["observation/state"]

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        padded_image = np.zeros_like(base_image)

        # Pi0.5 shares the same three-image contract as pi0 for our Franka tabletop adapter.
        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (
                base_image,
                padded_image,
                wrist_image,
            )
            image_masks = (np.True_, np.True_, np.True_)
        else:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (
                base_image,
                wrist_image,
                padded_image,
            )
            image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Keep actions in the environment/native 8D space here. Padding to the
        # model action dimension happens later in PadStatesAndActions.
        # Actions are only available during training.
        if "actions" in data:
            assert len(data["actions"].shape) == 2 and data["actions"].shape[-1] == 8, (
                f"Expected actions shape (N, 8), got {data['actions'].shape}"
            )
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
