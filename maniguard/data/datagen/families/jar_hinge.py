"""Hinged-jar geometry: read the lid's revolute joint from a live OmniGibson object and turn it
into the world hinge pivot + axis + lid pose, then the arc-close waypoints. The pure math (rotation
about a world axis, extension direction, insert pose, close angle) is numpy/scipy so it unit-tests
without a sim; ``read_hinge`` is the only sim-dependent function (validated by the Phase-A smoke)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

WORLD_UP = np.array([0.0, 0.0, 1.0])

# --- Phase-A tunables (see the design spec §5; per-model close margin overridable via datagen_hints) ---
INSERT_DEPTH_M = 0.02        # depth below the lid top the fingers straddle (~20% of the long Franka finger)
CLOSE_MARGIN_DEG = 15.0      # drive the arc this far PAST the vertical tipping point (the "slightly past 2α")
ARC_STEP_DEG = 10.0          # max angular step per SERVO waypoint (sets the arc segment count)
RETREAT_M = 0.04             # retreat distance after the arc (3-5 cm)
LID_STANDOFF_M = 0.05        # pre-insert standoff back along the approach axis
CLOSE_DIR = -1.0             # orbit sign about +axis that CLOSES (decreases the joint; PhysX convention)


def _u(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------------------
#  pure rotation primitives
# ---------------------------------------------------------------------------

def rotate_vec_about_axis(vec, axis, theta) -> np.ndarray:
    """Rotate a free vector about a unit ``axis`` (through origin) by ``theta`` rad."""
    return Rot.from_rotvec(_u(axis) * float(theta)).apply(np.asarray(vec, float))


def rotate_pose_about_axis(pos, quat_xyzw, anchor, axis, theta):
    """Rotate a pose about the world line ``(anchor, axis)`` by ``theta`` rad. Position orbits the
    anchor; orientation is left-multiplied by the same rotation (co-rotates)."""
    Rw = Rot.from_rotvec(_u(axis) * float(theta))
    anchor = np.asarray(anchor, float)
    pos2 = anchor + Rw.apply(np.asarray(pos, float) - anchor)
    quat2 = (Rw * Rot.from_quat(np.asarray(quat_xyzw, float))).as_quat()
    return pos2, quat2


def unit_perp(vec, axis) -> np.ndarray:
    """Unit vector of ``vec``'s component perpendicular to ``axis`` (zero if parallel)."""
    vec = np.asarray(vec, float)
    ax = _u(axis)
    perp = vec - np.dot(vec, ax) * ax
    n = np.linalg.norm(perp)
    return perp / n if n > 1e-12 else np.zeros(3)


def angle_between(a, b) -> float:
    a, b = _u(a), _u(b)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


# ---------------------------------------------------------------------------
#  close direction / closed check / extension / insert pose / arc angle
# ---------------------------------------------------------------------------

def is_closed(angle, lower, upper, frac: float = 0.05) -> bool:
    """Mirror ``safety_monitor._is_open_via_joints``: closed = within ``frac`` of the joint range from
    ``lower_limit`` (hinged_jar has abilities={} so this is the bench's ``jar_closed`` convention)."""
    lo, hi = float(lower), float(upper)
    if hi <= lo:
        return True
    return float(angle) <= (1.0 - frac) * lo + frac * hi


def lid_extension_dir(anchor, axis, lid_tip) -> np.ndarray:
    """Unit lid direction #1 = the tip-from-hinge vector projected perpendicular to the hinge axis."""
    return unit_perp(np.asarray(lid_tip, float) - np.asarray(anchor, float), axis)


def insert_pose(anchor, axis, lid_tip, depth: float = INSERT_DEPTH_M):
    """The straddle pose: position ``depth`` below the lid tip toward the hinge; approach (eef +Z)
    points down the lid toward the hinge; finger-closing axis (eef +Y) is perpendicular to the flap
    face (= axis x e), so the lid sits between the pads. Returns ``(pos(3,), quat_xyzw(4,))``."""
    anchor = np.asarray(anchor, float)
    lid_tip = np.asarray(lid_tip, float)
    down = _u(anchor - lid_tip)                       # from the tip toward the hinge (down the lid)
    pos = lid_tip + float(depth) * down               # straddle `depth` below the tip
    e = lid_extension_dir(anchor, axis, lid_tip)      # #1 lid extension (up the lid)
    z = down                                          # approach (eef +Z) points down the lid
    y = _u(np.cross(_u(axis), e))                     # finger-closing axis perpendicular to the flap face
    x = _u(np.cross(y, z))
    y = _u(np.cross(z, x))                            # re-orthogonalize
    quat = Rot.from_matrix(np.column_stack([x, y, z])).as_quat()
    return pos, quat


def arc_close_angle(e, axis, margin_rad, close_dir: float = CLOSE_DIR, eps: float = 1e-3) -> float:
    """Signed rotation about ``+axis`` to drive the lid past the vertical tipping point by
    ``margin_rad`` in the closing direction (0-drive if the lid already sits past vertical)."""
    axis = _u(axis)
    phi = angle_between(unit_perp(e, axis), unit_perp(WORLD_UP, axis))     # e -> vertical, unsigned
    e_test = rotate_vec_about_axis(e, axis, close_dir * eps)               # nudge in the closing dir
    on_open_side = angle_between(unit_perp(e_test, axis), unit_perp(WORLD_UP, axis)) < phi
    drive = phi if on_open_side else 0.0                                   # already past vertical -> 0
    return close_dir * (drive + float(margin_rad))


def arc_waypoints(insert_p, insert_q, anchor, axis, total_angle, step_deg: float = ARC_STEP_DEG):
    """Cumulative-rotation waypoints of the insert pose about ``(anchor, axis)``. Returns
    ``[(theta_k, pos_k(3,), quat_k(4,))]`` with ``theta_k`` the signed cumulative angle."""
    n = max(1, int(np.ceil(abs(np.degrees(total_angle)) / float(step_deg))))
    out = []
    for k in range(1, n + 1):
        theta = total_angle * (k / n)
        pos_k, quat_k = rotate_pose_about_axis(insert_p, insert_q, anchor, axis, theta)
        out.append((float(theta), pos_k, quat_k))
    return out


# ---------------------------------------------------------------------------
#  sim read (OmniGibson) — NOT unit-tested; validated by the Phase-A smoke
# ---------------------------------------------------------------------------

@dataclass
class HingeFrame:
    anchor: np.ndarray
    axis: np.ndarray
    angle: float
    lower: float
    upper: float
    lid_pos: np.ndarray
    lid_quat: np.ndarray
    lid_tip: np.ndarray


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, float)


def _link_by_body(obj, body_path_attr):
    """The RigidPrim whose prim path is the joint's body0/body1 target (last path component)."""
    return obj.links[body_path_attr.split("/")[-1]]


def read_hinge(obj) -> HingeFrame:
    """World hinge pivot + axis + live angle + lid link pose + lid tip for a hinged_jar.

    Composes the joint's static local frame on the PARENT link with the parent link's LIVE world
    pose (Fabric), so a per-task yaw / open angle are handled automatically. Falls back to the
    local frame on the CHILD link if the parent frame is trivial (identity localPos0)."""
    joint = next(j for j in obj.joints.values() if j.is_revolute)
    angle = float(joint.get_state()[0])
    lower, upper = float(joint.lower_limit), float(joint.upper_limit)

    parent = _link_by_body(obj, joint.body0)
    p_pos, p_quat = (_to_np(v) for v in parent.get_position_orientation())
    lpos0, lquat0 = _to_np(joint.local_position_0), _to_np(joint.local_orientation_0)
    # world joint frame = parent_world (pos, R) composed with the joint's local frame on the parent
    frame_R = Rot.from_quat(p_quat) * Rot.from_quat(lquat0)
    anchor = np.asarray(p_pos, float) + Rot.from_quat(p_quat).apply(lpos0)

    lid = _link_by_body(obj, joint.body1)
    lid_pos, lid_quat = (_to_np(v) for v in lid.get_position_orientation())

    # fallback: if localPos0 is trivially at the parent origin, use the child-frame authoring instead
    if float(np.linalg.norm(lpos0)) < 1e-6 and Rot.from_quat(lquat0).magnitude() < 1e-6:
        lpos1, lquat1 = _to_np(joint.local_position_1), _to_np(joint.local_orientation_1)
        frame_R = Rot.from_quat(lid_quat) * Rot.from_quat(lquat1)
        anchor = np.asarray(lid_pos, float) + Rot.from_quat(lid_quat).apply(lpos1)

    axis_unit = {"X": [1.0, 0, 0], "Y": [0, 1.0, 0], "Z": [0, 0, 1.0]}[joint.axis]
    axis = _u(frame_R.apply(axis_unit))

    lo, hi = (_to_np(v) for v in lid.aabb)                      # world AABB of the lid link
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    lid_tip = corners[int(np.argmax(np.linalg.norm(corners - anchor, axis=1)))]  # farthest from the hinge

    return HingeFrame(anchor=anchor, axis=axis, angle=angle, lower=lower, upper=upper,
                      lid_pos=np.asarray(lid_pos, float), lid_quat=np.asarray(lid_quat, float),
                      lid_tip=lid_tip)
