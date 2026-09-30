"""Register the ManiGuard simulation modality configuration as NEW_EMBODIMENT.

Map state/actions columns into seven arm joints and one gripper value, and read
image_left plus wrist_image. Predict 16-step action chunks with state-relative
arm actions and absolute gripper commands. MODALITY_JSON maps these modalities
to the existing dataset keys; no source columns or video files are renamed.

This module depends on gr00t and registers at import time. Import one embodiment
configuration per process.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Future-action chunk length the diffusion head predicts (matches the pi0.5 +
# LIBERO convention; open_loop_eval should use the same --action-horizon).
ACTION_HORIZON = 16

# Active camera views fed to the VLM, in order: 2-cam (one overview + wrist),
# matching the pi0.5 SFT (external_cam="left") for benchmark parity. GR00T natively
# supports more views — to add one (e.g. "image_right"), append its key here; its
# original_key mapping is already registered in ``_VIDEO_ORIGINAL_KEY`` below, then
# re-run stats. Maps to dataset video dirs via ``MODALITY_JSON["video"]``.
VIDEO_KEYS = ["image_left", "wrist"]

# Known GR00T modality_key -> dataset video feature/dir. A superset of the active
# ``VIDEO_KEYS`` so extra overviews can be enabled by name without editing this map.
_VIDEO_ORIGINAL_KEY = {
    "image_left": "image_left",
    "image_right": "image_right",
    "wrist": "wrist_image",
}

MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],  # current frame only
        modality_keys=list(VIDEO_KEYS),
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
        # NOTE: oxe_droid (also Franka joints) uses plain min-max here; sin/cos
        # encoding of the radian arm joints is available via
        # ``sin_cos_embedding_keys=["single_arm"]`` if angle wraparound hurts.
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # 7 arm joints: state-relative chunks (N1.6-native, smoother).
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: absolute open/close target.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Body of ``<dataset>/meta/modality.json``. ``original_key`` makes GR00T read our
# existing on-disk names instead of its defaults (observation.state / action /
# observation.images.*), so the export needs no renaming.
#   state/action: the 8-D ``state`` / ``actions`` columns sliced 0:7 (arm) + 7:8 (gripper)
#   video: GR00T key -> stored LeRobot video feature key
#   annotation: ``task_index`` column -> task string via meta/tasks.jsonl
MODALITY_JSON = {
    "state": {
        "single_arm": {"start": 0, "end": 7, "original_key": "state"},
        "gripper": {"start": 7, "end": 8, "original_key": "state"},
    },
    "action": {
        "single_arm": {"start": 0, "end": 7, "original_key": "actions"},
        "gripper": {"start": 7, "end": 8, "original_key": "actions"},
    },
    "video": {key: {"original_key": _VIDEO_ORIGINAL_KEY[key]} for key in VIDEO_KEYS},
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}

# Registers MODALITY_CONFIG -> MODALITY_CONFIGS["new_embodiment"]. Asserts it is
# not already registered, so import this module exactly once per process.
register_modality_config(MODALITY_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
