"""Register a DROID-schema real-robot modality configuration as NEW_EMBODIMENT.

Read joint_position and gripper_position as separate state columns, and use
exterior_image_1_left plus wrist_image_left. Actions contain seven joint velocities
in rad/s and one gripper target. Both action groups use ABSOLUTE representation,
meaning the processor uses their stored values without subtracting joint position.

The 16-frame action chunk spans approximately 1.07 seconds at 15 Hz. The unused
second exterior-camera placeholder is not mapped. Import this configuration or
the simulation configuration once per process.
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

# Same chunk length as sim so the two sides are comparable in steps. NOTE the duration is
# NOT the same: 16 steps is 1.07 s at real's 15 fps vs 0.53 s at sim's 30 fps.
ACTION_HORIZON = 16

# 2-cam (one exterior + wrist), matching the pi0.5/pi0 real configs.
VIDEO_KEYS = ["image_left", "wrist"]

# GR00T modality_key -> DROID-schema video feature stored on disk.
_VIDEO_ORIGINAL_KEY = {
    "image_left": "exterior_image_1_left",
    "wrist": "wrist_image_left",
}

MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],  # current frame only
        modality_keys=list(VIDEO_KEYS),
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # 7 arm joints: JOINT VELOCITY -- absolute, i.e. used as stored. NEVER RELATIVE
            # here; see the module docstring. This is the single line that separates a valid
            # real checkpoint from one trained on `velocity - position`.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: next-frame open/close target, normalized 0=open 1=closed.
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

# Body of ``<dataset>/meta/modality.json``. Unlike sim, state comes from TWO separate
# columns, so each slice indexes into its own ``original_key`` (joint_position is (7,) ->
# 0:7; gripper_position is (1,) -> 0:1). The action column is a single 8-D vector, sliced
# 0:7 / 7:8 exactly as in sim.
MODALITY_JSON = {
    "state": {
        "single_arm": {"start": 0, "end": 7, "original_key": "joint_position"},
        "gripper": {"start": 0, "end": 1, "original_key": "gripper_position"},
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

# Registers MODALITY_CONFIG -> MODALITY_CONFIGS["new_embodiment"]. Asserts it is not
# already registered, so import this module (or the sim one, never both) once per process.
register_modality_config(MODALITY_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
