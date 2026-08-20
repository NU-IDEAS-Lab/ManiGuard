"""GR00T ``NEW_EMBODIMENT`` modality config for the ManiGuard REAL Franka (DROID schema).

The real-robot counterpart of ``maniguard_embodiment.py``. Same robot, same two camera
slots, same 16-step chunk -- but the **recorded quantities differ**, so this is a separate
config rather than a flag on the sim one. Self-contained (depends only on ``gr00t``) so it
can be passed to ``launch_finetune.py --modality-config-path`` inside the Isaac-GR00T venv.

Consumed by the real SFT runs over the ``real-{clutter,jar}-60-droid-refined`` and
``real-cab-higher-firsthalf-60-droid-refined`` datasets, built by
``maniguard/data/real_teleop/real_teleop_to_droid.py``.

★ THE ONE THING THAT MUST NOT BE COPIED FROM SIM: the arm action representation.
    sim  stores ABSOLUTE JOINT TARGETS  -> rep=RELATIVE lets GR00T predict state-relative
         chunks and re-add the state at inference (its equivalent of openpi's
         ``use_delta_joint_actions=True``).
    real stores JOINT VELOCITY (rad/s)  -> rep must be ABSOLUTE. With RELATIVE, GR00T would
         compute ``velocity - joint_position``, which is dimensionally meaningless. This is
         the error openpi's own DROID config warns about: "We assume joint velocity actions,
         so we should not apply an additional delta transform ... it would differentiate a
         velocity twice."
The gripper is ABSOLUTE on both sides (a normalized open/close target, 0=open 1=closed).

Consequence at inference: with no RELATIVE group, ``StateActionProcessor.unapply_action``
performs **no** state re-addition, so the served chunk is joint velocity as-is. The real
client must apply it as ``delta = action / 15`` (15 Hz), with NO clip -- see
``maniguard/serve/gr00t_native.py --real``.

Schema differences vs sim (all handled by ``original_key``; nothing on disk is renamed):
    state    sim: one 8-D ``state`` column
             real: TWO columns -- ``joint_position`` (7,) + ``gripper_position`` (1,)
    actions  sim: absolute joint targets      real: [joint_velocity(7), gripper_target(1)]
    video    sim: image_left / wrist_image    real: exterior_image_1_left / wrist_image_left
    fps      sim: 30                          real: 15  (so a 16-step chunk is 1.07 s, not 0.53 s)

The modality KEYS are deliberately kept identical to sim (``image_left``/``wrist``,
``single_arm``/``gripper``) even though the underlying columns differ: only ``original_key``
changes. That keeps ``gr00t_native.py``'s observation packing branch-free -- the sim/real
difference lives in the checkpoint, where it belongs.

``exterior_image_2_left`` exists in the real datasets but is an ALL-ZERO placeholder (there
is no second exterior camera on the rig) -- deliberately NOT mapped.
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
