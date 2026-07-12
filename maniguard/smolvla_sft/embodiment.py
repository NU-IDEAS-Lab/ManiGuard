"""ManiGuard SmolVLA embodiment contract: how the 5-cam datagen LeRobot export
maps into the 2-cam, standard-keyed dataset SmolVLA / ``lerobot-train`` expects.

Self-contained (pure Python, no ``lerobot`` / ``torch`` imports) so the
``tools/smolvla_sft`` scripts can import it directly. This is the single source of
truth for the SmolVLA data mapping — mirrors ``gr00t_sft.maniguard_embodiment``
for the GR00T path and ``openpi_sft.data_configs`` for the pi0.5 path.

Why a rename step (unlike GR00T / openpi): both of those consume our dataset's
flat keys through an indirection layer (GR00T's ``modality.json`` ``original_key``,
openpi's ``RepackTransform``). ``lerobot-train`` has no such indirection — it
classifies features purely by key prefix (``observation.images.*`` -> visual,
``observation.state`` -> state, ``action`` -> action). Our export uses flat keys
(``image_left`` / ``state`` / ``actions``), so ``prepare_dataset.py`` must
physically rename them (and drop the unused streams) into a standard-keyed copy.

Embodiment: Franka Panda, 8-D joint state/action (7 arm joints + 1 gripper).
SmolVLA pads state/action to its ``max_state_dim`` / ``max_action_dim`` (32), and
trains on the dataset's ABSOLUTE joint actions directly (no delta transform — the
model outputs absolute joint targets that feed a JointController at eval, same
end-to-end joint contract as the other two paths).

2 camera views (one third-person overview + wrist), matching what pi0.5
(``external_cam="left"``) and GR00T (2-cam) consume, so all three models train on
identical inputs (benchmark parity). SmolVLA natively supports more views —
adding one back is a one-line change to ``rename_map`` + ``DROPPED_STREAMS``.
"""

from __future__ import annotations

# Upstream base checkpoint fine-tuned by ``lerobot-train --policy.path=...``.
BASE_MODEL = "lerobot/smolvla_base"

# 8-D joint state/action (7 arm joints + 1 gripper). SmolVLA left-pads both to its
# max_state_dim / max_action_dim (32); nothing to configure here.
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
