"""Single source of truth for the ManiGuard datagen dataset schema.

The cuRobo-collected SFT data is **joint-native** (the env's cuRobo emits a joint
trajectory → JointController execution → record joints directly; no eef↔joint
conversion, no sim-state reverse-engineering).

Per timestep we record:
  - ``state``             (8,) = ``[arm_q(7), gripper(1, mean finger)]``
  - ``actions``           (8,) = ``[arm_q[t+1](7), gripper_cmd(1, binary)]``
        DEFAULT action = (b) next-achieved absolute joint (DROID-style; robust to
        the rough cuRobo solver, self-consistent with the recorded images).
  - ``actions_commanded`` (8,) = ``[curobo_target_q(7), gripper_cmd(1)]``
        EXTRA = (a) the cuRobo COMMANDED joint target (free to record live; kept for
        comparison / future switch). Not consumed by the default SFT config.
  - five 256² image streams: 4 third-person (from the SHARED bench ``camera_setup``)
    + the injected wrist.

Plus a per-episode MimicGen sidecar (NOT in the LeRobot parquet): serialized sim
states + object-centric ``datagen_info`` — kept now so the future MimicGen
amplification layer needs no re-collection. See ``MIMICGEN_SIDECAR`` below.
"""
from __future__ import annotations

RESOLUTION = 256
FPS = 30
# Keyframe interval for the recorded trajectory MP4s. Datasets are consumed by
# random-frame access in the SFT dataloader, so a keyframe every VIDEO_GOP frames
# keeps that decode cheap (a large GOP forces decoding ~GOP/2 frames per sample,
# which starves the GPU). 10 kills the decode tail at ~2.6x the file size.
VIDEO_GOP = 10
ROBOT_TYPE = "FrankaPanda"

ARM_DOF = 7
STATE_DIM = 8           # arm_q(7) + gripper(1)
ACTION_DIM = 8          # arm_q(7) + gripper_cmd(1)

# --- image streams: lerobot key -> the OG external-sensor name it captures from.
# The four third-person views come from the shared bench camera_setup
# (maniguard/utils/camera_setup.py, EXTERNAL_CAMERA_NAMES); the wrist is injected
# under panda_hand by the recorder (no OG external sensor).
THIRD_PERSON_CAMS = {
    "image_opposite":      "cam_opposite",
    "image_left":          "cam_left",
    "image_right":         "cam_right",
    "image_left_shoulder": "cam_left_shoulder",
}
WRIST_KEY = "wrist_image"
IMAGE_KEYS = (*THIRD_PERSON_CAMS.keys(), WRIST_KEY)   # 5 streams

# Downstream SFT/eval pick ONE third-person view via the data config's
# ``external_cam`` (routed to observation/image_left → pi0.5 base_0_rgb). The
# dataset always ships all five; the choice is downstream + per-family. The config
# value is the SHORT name (``opposite``/``left``/``right``/``left_shoulder``); the
# dataset stream it selects is ``image_<name>``.
EXTERNAL_CAM_CHOICES = tuple(k[len("image_"):] for k in THIRD_PERSON_CAMS)   # 4 short names

STATE_NAMES = [f"arm_q{i}" for i in range(ARM_DOF)] + ["gripper"]
ACTION_NAMES = [f"arm_q{i}_next" for i in range(ARM_DOF)] + ["gripper_cmd"]
ACTION_COMMANDED_NAMES = [f"arm_q{i}_cmd" for i in range(ARM_DOF)] + ["gripper_cmd"]

# Per-episode MimicGen sidecar (HDF5 group layout; written alongside the LeRobot
# dataset, consumed by the future MimicGen engine — see doc §8).
MIMICGEN_SIDECAR = {
    "states": "serialized og.sim.dump_state(serialized=True) per step (replay)",
    "datagen_info/eef_pose": "(N,4,4) world eef pose per step",
    "datagen_info/object_poses/<obj>": "(N,4,4) world pose per tracked object",
    "datagen_info/gripper_action": "(N,) binary gripper command",
    "datagen_info/subtask_term_signals/<sig>": "(N,) bool subtask-termination flags",
}


def lerobot_features(resolution: int = RESOLUTION) -> dict:
    """LeRobot v2.1 feature schema for the datagen SFT dataset (5 video streams +
    joint state + joint actions [achieved] + joint actions_commanded)."""
    def _img():
        return {"dtype": "video", "shape": (resolution, resolution, 3),
                "names": ["height", "width", "channel"]}

    feats = {key: _img() for key in IMAGE_KEYS}
    feats["state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES}
    feats["actions"] = {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES}
    feats["actions_commanded"] = {"dtype": "float32", "shape": (ACTION_DIM,),
                                  "names": ACTION_COMMANDED_NAMES}
    return feats
