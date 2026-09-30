"""Shared schema constants for RAW and LeRobot demonstration data.

Each state contains seven achieved arm joints and the mean finger position.
actions contains the next recorded arm joints and the current recorded binary
gripper command; the final arm action repeats the last achieved joints.
actions_commanded retains the joint targets submitted for the recorded step.

Five video streams contain four external views and one wrist view at 256 by 256
pixels and 30 Hz. RAW HDF5 can additionally contain serialized simulation states
and datagen_info/gripper_action; these are not copied into LeRobot Parquet.
"""
from __future__ import annotations

RESOLUTION = 256
FPS = 30
# Keyframe interval for trajectory MP4s. Frequent keyframes reduce the number
# of frames decoded for random-frame access during SFT, at the cost of larger files.
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

# Auxiliary HDF5 datasets written by the recorder.
MIMICGEN_SIDECAR = {
    "states": "optional serialized simulation states, padded when their lengths differ",
    "states_len": "present only for padded states; original length of each recorded state",
    "datagen_info/gripper_action": "binary gripper command per recorded step",
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
