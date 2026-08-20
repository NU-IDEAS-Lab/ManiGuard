"""Load the human grasp-annotation DB and turn a target's object-local grasp frames into
WORLD eef-target poses at runtime — the bridge from ``annotation/`` (the DB) to the
executor.

Each annotated grasp is an eef_link TARGET pose in the object-LOCAL frame
(``position`` + ``orientation_xyzw``). At runtime, given the target object's live world
pose, the world eef target is::

    T_eef_world = T_object_world @ T_grasp_local

Pure numpy/scipy — no OmniGibson / cuRobo, so it imports cheaply and unit-tests without a
sim. (Same transform the annotation ``validate_grasps`` loader verified to 0.0 mm.)

The released database (1,547 grasps / 221 object instances) is the HF dataset
``IDEAS-Lab-Northwestern/maniguard-grasp-annotations`` — download it into
``outputs/grasp_annotation/`` (the default ``ANN_PATH`` below).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ANN_PATH = Path("outputs/grasp_annotation/grasp_annotations.json")


def load_db(path: str | Path = ANN_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _pose_to_mat(pos, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
    T[:3, 3] = np.asarray(pos, float)
    return T


def _mat_to_pose(T: np.ndarray):
    """(position (3,), quat_xyzw (4,))."""
    return T[:3, 3].copy(), Rot.from_matrix(T[:3, :3]).as_quat()


def has_target(db: dict, key: str) -> bool:
    """True if object ``key`` ('category/model') has at least one annotated grasp."""
    return bool(db.get("objects", {}).get(key, {}).get("grasps"))


def target_grasps_world(db: dict, key: str, obj_pos, obj_quat_xyzw) -> list[dict]:
    """World eef-target pose for each annotated grasp of object ``key``, given the
    object's live world pose. Returns ``[{id, eef_pos (3,), eef_quat (4, xyzw), approach}]``
    ordered by the stored grasp id. Raises KeyError if the object isn't in the DB."""
    rec = db["objects"][key]
    T_obj = _pose_to_mat(obj_pos, obj_quat_xyzw)
    out = []
    for g in rec["grasps"]:
        T_local = _pose_to_mat(g["position"], g["orientation_xyzw"])
        pos, quat = _mat_to_pose(T_obj @ T_local)
        out.append({
            "id": int(g["id"]),
            "eef_pos": pos,
            "eef_quat": quat,
            "approach": g.get("approach_hint", "top_down"),
        })
    return out
