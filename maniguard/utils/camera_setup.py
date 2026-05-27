"""Canonical camera setup shared across task-generation, teleop, training, and eval.

Three external cameras (opposite / left / right of the robot) are the
canonical layout. The pipeline computes their positions per episode; every
downstream consumer (teleop, finetuning, evaluation) just loads the same
sensor configs and optionally picks which view to feed the policy.

Policy input composition: if `policy_external_cameras` selects one camera,
its RGB is returned unchanged. If it selects multiple, they are horizontally
concatenated along the width axis -- simplest scheme that keeps a single
(H, W, 3) tensor for downstream models.
"""

from __future__ import annotations

from typing import Iterable, Sequence


CAMERA_RESOLUTION = 256
EXTERNAL_CAMERA_NAMES = ("cam_opposite", "cam_left", "cam_right")
POLICY_EXTERNAL_CAMERAS_DEFAULT = ("cam_opposite",)


def build_external_camera_configs(
    names: Sequence[str] = EXTERNAL_CAMERA_NAMES,
    resolution: int | tuple[int, int] | None = None,
    modalities: Sequence[str] = ("rgb",),
) -> list[dict]:
    """Return a list of VisionSensor config dicts ready for env_config['external_sensors'].

    Args:
        names: sensor names; also used for the /{name} relative_prim_path.
        resolution: int for square, or (height, width) tuple, or None to
            leave image_height/image_width unset -- OmniGibson's VisionSensor
            will fall back to Kit's viewport default (typically 1280x720).
            Only set this when you need a specific size for policy input
            (training/eval); pair it with the setter+load_observation_space
            workaround at the call site, since sensor_kwargs alone is
            unreliable (see StanfordVL/OmniGibson#266, #1875).
        modalities: sensor modalities; defaults to ["rgb"].
    """
    sensor_kwargs: dict = {}
    if resolution is not None:
        if isinstance(resolution, int):
            image_height = image_width = int(resolution)
        else:
            image_height, image_width = int(resolution[0]), int(resolution[1])
        sensor_kwargs = {"image_height": image_height, "image_width": image_width}
    return [
        {
            "sensor_type": "VisionSensor",
            "name": name,
            "relative_prim_path": f"/{name}",
            "modalities": list(modalities),
            "sensor_kwargs": sensor_kwargs,
        }
        for name in names
    ]


def compose_main_image(rgb_by_camera: dict, camera_names: Iterable[str]):
    """Concatenate selected camera RGB frames along width to form a single 'main' image.

    A single camera is returned unchanged. Multiple cameras are concatenated in
    the given order along the width axis (axis=-2 for (H, W, 3) tensors).
    Works for numpy arrays and torch tensors.
    """
    names = list(camera_names)
    if not names:
        raise ValueError("compose_main_image requires at least one camera name")
    if len(names) == 1:
        return rgb_by_camera[names[0]]
    imgs = [rgb_by_camera[n] for n in names]
    # Prefer the tensor's own concat to stay on the same device / backend.
    if hasattr(imgs[0], "__module__") and "torch" in type(imgs[0]).__module__:
        import torch
        return torch.cat(imgs, dim=-2)
    import numpy as np
    return np.concatenate(imgs, axis=-2)


def normalize_policy_cameras(value) -> list[str]:
    """Coerce config values (None / str / list) into a list of camera names."""
    if value is None:
        return list(POLICY_EXTERNAL_CAMERAS_DEFAULT)
    if isinstance(value, str):
        return [value]
    return list(value)
