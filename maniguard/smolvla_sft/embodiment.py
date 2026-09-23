"""Map the five-camera datagen export into a two-camera SmolVLA dataset.

Keep a selected external view and wrist_image, rename state to observation.state
and actions to action, and drop unused views and actions_commanded. The numerical
state/action representation remains eight-dimensional joint data. The model
preparation path uses the stored absolute joint actions without a delta transform.

This module contains only constants and mapping helpers and imports no model
runtime. Training and evaluation must use the same external view.
"""

from __future__ import annotations

# Upstream base checkpoint fine-tuned by ``lerobot-train --policy.path=...``.
BASE_MODEL = "lerobot/smolvla_base"

# Native state and action dimensions: seven arm joints plus one gripper value.
# Model-side padding is handled by the SmolVLA preprocessing stack.
STATE_DIM = 8
ACTION_DIM = 8

# --- datagen export (source) key names -------------------------------------
# The 5 image streams + state/action columns as written by
# ``maniguard.data.datagen.data_format.lerobot_features`` (flat, non-standard).
_SRC_WRIST = "wrist_image"
# Third-person overviews, short name -> flat dataset stream. Exactly one is kept
# as the policy overview (the rest dropped), chosen per family at prepare time.
_SRC_OVERVIEWS = {
    "opposite": "image_opposite",
    "left": "image_left",
    "right": "image_right",
    "left_shoulder": "image_left_shoulder",
}
EXTERNAL_CAM_CHOICES = tuple(_SRC_OVERVIEWS)  # opposite / left / right / left_shoulder
DEFAULT_EXTERNAL_CAM = "left"

# --- SmolVLA (target) standard key names -----------------------------------
# Names are free (SmolVLA is agnostic to camera naming); only requirement is that
# train and eval agree. We keep two descriptive, prefix-correct keys.
OVERVIEW_KEY = "observation.images.top"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def rename_map(external_cam: str = DEFAULT_EXTERNAL_CAM) -> dict[str, str]:
    """datagen flat key -> SmolVLA standard key for the kept streams.

    ``external_cam`` selects which of the four third-person overviews becomes the
    single ``observation.images.top`` view (train/eval MUST use the same choice).
    Streams not in this map are dropped from the SmolVLA copy (see
    ``dropped_streams``).
    """
    if external_cam not in EXTERNAL_CAM_CHOICES:
        raise ValueError(
            f"external_cam must be one of {EXTERNAL_CAM_CHOICES}, got {external_cam!r}"
        )
    return {
        _SRC_OVERVIEWS[external_cam]: OVERVIEW_KEY,
        _SRC_WRIST: WRIST_KEY,
        "state": STATE_KEY,
        "actions": ACTION_KEY,
    }


def dropped_streams(external_cam: str = DEFAULT_EXTERNAL_CAM) -> list[str]:
    """datagen streams excluded from the 2-cam SmolVLA copy: the 3 unused
    overviews + the redundant ``actions_commanded`` column."""
    kept = set(rename_map(external_cam))
    all_src = {*_SRC_OVERVIEWS.values(), _SRC_WRIST, "state", "actions", "actions_commanded"}
    return sorted(all_src - kept)
