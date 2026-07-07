"""Hinged-jar geometry: read the lid's revolute joint from a live OmniGibson object and turn it
into the world hinge pivot + axis + lid pose, then the arc-close waypoints. The pure math (rotation
about a world axis, extension direction, insert pose, close angle) is numpy/scipy so it unit-tests
without a sim; ``read_hinge`` is the only sim-dependent function (validated by the Phase-A smoke)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

WORLD_UP = np.array([0.0, 0.0, 1.0])

# --- Phase-A tunables (per-model extra drive overridable via datagen_hints) ---
CLOSE_MARGIN_DEG = 25.0      # drive past the tipping point by this much (covers the pre-contact free-wheel)
ARC_STEP_DEG = 2.0           # max angular step per SERVO arc segment (orientation snaps once per segment;
#                              small steps keep the rigid AG-held lid from fighting the hinge and hauling
#                              the light jar off the table)
RETREAT_M = 0.10             # retreat along the hinge axis so the fingers fully clear the disc rim
LID_STANDOFF_M = 0.10        # pre-engage standoff back along the hinge axis (robot side)
CLOSE_DIR = -1.0             # orbit sign about +axis that CLOSES (decreases the joint; PhysX convention, smoke-verified)
FINGERTIP_M = 0.104          # eef_link -> closed fingertip along eef +Z (from gripper_longfinger.glb)
RIM_INSET_M = 0.04           # contact radial station = measured hull rim MINUS this (deep enough that
#                              the pad-vs-lip contact can migrate rimward during the arc without walking
#                              off the disc — 2cm walked off and wedged on the rim edge;
#                              max torque arm; the disc plane geometry is MEASURED from the lid link's
#                              collision hull, never assumed — the kijnrj lid slab floats +25..41mm off
#                              the plane through the joint anchor (hinge-knuckle offset), so any
#                              anchor-plane assumption straddles empty air)
PAD_HALF_GAP_M = 0.040       # OPEN half-gap of the pad INNER FACES from the centreline (per-z-slice
#                              measured from the longfinger glb: faces at +-4.0cm at every finger section)
STRADDLE_CLEAR_M = 0.012     # pushing pad face starts this far off the slab's +f face (clears the lip
#                              during the slide-in; consumed as ~6deg of free-wheel at the arc start)
# --- lid-ride (the user's teleop maneuver): one finger bar under the lid, single straight ride ---
RIDE_OPEN_DEG = 12.0         # finger bar lies this far BELOW the lid's underside line (into the wedge)
RIDE_D0_FRAC = 0.55          # initial contact station along the lid (fraction of measured reach)
RIDE_TIP_EXTRA_M = 0.025     # fingertip goes this much deeper than the contact station (contact mid-bar)
RIDE_START_CLEAR_M = 0.015   # bar starts this far below the underside (engages after the first ride cm)
RIDE_HZ_FRAC = 0.60          # ride END: contact point this high above the hinge (fraction of reach)
RIDE_XM_FRAC = 0.18          # ...and this far toward the mouth side (lid ends ~17deg past vertical)
END_CLEAR_DEG = 20.0         # arc end must stay this far from the CLOSED (mouth) plane — finger-pinch clearance
MIN_PAST_VERT_DEG = 12.0     # arc end must be at least this far past the vertical tipping point


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


def face_normal(axis, e) -> np.ndarray:
    """Disc-face normal ``f = axis x e`` (perpendicular to the lid plane, which contains the hinge
    axis and the lid extension). Pushing the lid along ``-f`` torques it about the hinge in the
    CLOSING (-axis) direction: ``tau = e x (-f) = -axis`` (matches the smoke-verified CLOSE_DIR)."""
    return _u(np.cross(_u(axis), _u(e)))


def straddle_pose_from_hull(anchor, axis, ext_dir, hull_pts, side_sign: float,
                            rim_inset: float = RIM_INSET_M, clear: float = STRADDLE_CLEAR_M,
                            pad_half_gap: float = PAD_HALF_GAP_M, fingertip_m: float = FINGERTIP_M):
    """SIDE-ENTRY straddle pose placed from the lid link's MEASURED collision hull (no geometry
    assumptions): the hand comes in horizontally along the hinge axis from the robot's side, the
    open finger gap straddling the ACTUAL lid slab (which may float several cm off the plane through
    the joint anchor — hinge-knuckle offset), fingertip at the slab's chord centre, radially just
    inside the measured rim. The pushing (+f-side) pad face starts ``clear`` above the slab's top
    face, so the close arc engages after a few degrees of free-wheel and the lid then rides its own
    hinge circle (unilateral contact — nothing can drag the jar).

    ``hull_pts``: (N,3) world points of the lid link's collision hull.
    ``side_sign``: +1/-1 so ``side_sign*axis`` points from the lid TOWARD the robot.
    Returns ``(eef_pos(3,), eef_quat(4, xyzw), reach(float))``: approach (eef +Z) =
    ``-side_sign*axis``; finger-closing axis (eef +Y) = disc normal ``f``."""
    anchor = np.asarray(anchor, float)
    e = _u(ext_dir)
    ax = _u(axis)
    f = face_normal(axis, e)
    rel = np.asarray(hull_pts, float) - anchor
    rad, yy, hh = rel @ e, rel @ ax, rel @ f
    reach = float(rad.max())                              # true rim (hull is exact at the rim)
    band = rad > 0.5 * reach                              # outer half: the slab proper (skip the knuckle)
    h_top = float(hh[band].max())                         # slab top face (+f side, incl. any lip)
    y_mid = float((yy[band].min() + yy[band].max()) / 2)  # chord centre along the hinge axis
    contact_rad = reach - float(rim_inset)
    gap_c = h_top + float(clear) - float(pad_half_gap)    # gap centre so the +f pad face = h_top+clear
    tip_pt = anchor + contact_rad * e + gap_c * f + y_mid * ax
    w = float(side_sign) * ax                             # unit, lid -> robot side
    z = -w                                                # approach: slide inward along the hinge axis
    y = f                                                 # closing axis = disc normal (⊥ axis ⊥ z)
    x = _u(np.cross(y, z))
    y = _u(np.cross(z, x))                                # re-orthogonalize (defensive)
    quat = Rot.from_matrix(np.column_stack([x, y, z])).as_quat()
    eef_pos = tip_pt - float(fingertip_m) * z             # eef one fingertip-length back (robot side)
    return eef_pos, quat, reach


def drive_angle(e, axis, extra_rad: float = 0.0, close_dir: float = CLOSE_DIR,
                clear_rad: float = np.radians(END_CLEAR_DEG),
                min_margin_rad: float = np.radians(MIN_PAST_VERT_DEG), eps: float = 1e-3) -> float:
    """Signed close-arc drive (the user's 2α rule, capped): drive ``2α (+extra)`` about ``+axis`` in
    the closing direction — ending the mirror ``α`` past vertical on the CLOSED side — but never
    closer than ``clear_rad`` to the mouth plane (finger-pinch clearance; gravity finishes from
    there) and always at least ``min_margin_rad`` past the vertical tipping point."""
    axis = _u(axis)
    alpha = angle_between(unit_perp(e, axis), unit_perp(WORLD_UP, axis))   # lid angle from vertical
    e_test = rotate_vec_about_axis(e, axis, close_dir * eps)
    on_open_side = angle_between(unit_perp(e_test, axis), unit_perp(WORLD_UP, axis)) < alpha
    if not on_open_side:                                    # already past vertical toward closed
        return close_dir * float(min_margin_rad)
    lo = alpha + float(min_margin_rad)                      # must pass the tipping point + margin
    hi = alpha + np.pi / 2.0 - float(clear_rad)             # must NOT pinch the fingers at the mouth
    drive = float(np.clip(2.0 * alpha + float(extra_rad), lo, max(lo, hi)))
    return close_dir * drive


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
    ext_dir: np.ndarray        # TRUE lid radial/extension unit (from lid_quat local axes, not the AABB corner)
    reach: float               # lid length along ext_dir (anchor -> far edge)
    half_width: float          # half extent of the lid AABB along the hinge axis (disc half-width)


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, float)


def _link_by_body(obj, body_path_attr):
    """The RigidPrim whose prim path is the joint's body0/body1 target (last path component)."""
    return obj.links[body_path_attr.split("/")[-1]]


def ride_plan(anchor, axis, ext_dir, hull_pts, side_sign: float, close_dir: float = CLOSE_DIR,
              open_deg: float = RIDE_OPEN_DEG, roll_flip: bool = False, bar_flip: bool = False,
              skew_deg: float = 0.0):
    """The user's teleop close maneuver, parameterized from the lid link's MEASURED hull: a finger
    BAR (finger plane ⊥ lid plane: x̂ = hinge axis) laid in the free wedge UNDER the flopped lid,
    then ONE straight-line translation (fixed orientation) that lifts the lid — the lid rests on the
    bar's upper edge under gravity and pivots about its own hinge, the contact sliding freely
    (unilateral: nothing can drag the jar). The ride ends with the contact above the hinge and past
    it toward the mouth, so the lid is past vertical and gravity closes it.

    Returns ``(eef_start(3,), eef_quat(4, xyzw), eef_end(3,), f(3,))``.
    ``side_sign``: +1/-1 so ``side_sign*axis`` points toward the robot (chooses the wrist roll)."""
    anchor = np.asarray(anchor, float)
    e = _u(ext_dir)
    ax = _u(axis)
    f = face_normal(axis, e)
    rel = np.asarray(hull_pts, float) - anchor
    rad, yy, hh = rel @ e, rel @ ax, rel @ f
    reach = float(rad.max())
    band = rad > 0.5 * reach
    h_top = float(hh[band].max())                          # underside (+f) face incl. lip
    y_mid = float((yy[band].min() + yy[band].max()) / 2)   # chord centre along the hinge axis

    # bar direction: 12deg below the lid line, pointing INTO the wedge (toward the hinge, wrist at
    # the wedge mouth). Opening rotation = -close_dir about +axis.
    e_bar = rotate_vec_about_axis(e, ax, -close_dir * np.radians(float(open_deg)))
    # bar_flip: fingers point OUTWARD (hinge->tip) and the WRIST sits under the lid — shortens the
    # total reach by ~2x the finger length; the pads' last cm supports the lid near its RIM (max
    # torque arm). Needed for FAR jar placements where the default (wrist beyond the lid tip) pose
    # family is wholly outside the arm's envelope.
    z = _u(e_bar) if bar_flip else -_u(e_bar)              # wrist -> tips
    w_rob = float(side_sign) * ax                          # unit toward the robot side
    if skew_deg:
        # skew the bar toward the robot (a diagonal bar under the lid supports it just the same);
        # pulls the WRIST several cm closer for far jar placements
        g = np.radians(float(skew_deg))
        z = _u(np.cos(g) * z - np.sin(g) * w_rob)
    x = _u(np.cross(f, z))                                 # frame: y ~ disc normal, x ~ hinge axis
    if float(np.dot(x, float(side_sign) * ax)) < 0.0:      # anchor the default branch to the ORIGINAL
        x = -x                                             # orientation (x along side_sign*axis)
    y = _u(np.cross(z, x))
    if roll_flip:                                          # 180deg wrist-roll branch (same support
        x, y = -x, -y                                      # mechanics, the other IK branch)
    quat = Rot.from_matrix(np.column_stack([x, y, z])).as_quat()

    d0 = RIDE_D0_FRAC * reach                              # initial contact station on the lid
    if bar_flip:
        # wrist directly under the lid at d0; fingertip pokes past the rim into free air
        eef_start = (anchor + d0 * _u(e_bar) + (h_top + RIDE_START_CLEAR_M) * f + y_mid * ax)
    else:
        tip = (anchor + (d0 + RIDE_TIP_EXTRA_M) * _u(e_bar) + (h_top + RIDE_START_CLEAR_M) * f
               + y_mid * ax)                               # fingertip deeper than the contact, below the face
        eef_start = tip - FINGERTIP_M * z

    # ride translation: carry the CONTACT point to (above the hinge + toward the mouth)
    m = _u(np.array([-e[0], -e[1], 0.0]))                  # horizontal unit toward the mouth/closing side
    contact_start = anchor + d0 * e + h_top * f
    contact_end = anchor + RIDE_HZ_FRAC * reach * WORLD_UP + RIDE_XM_FRAC * reach * m
    eef_end = eef_start + (contact_end - contact_start)
    return eef_start, quat, eef_end, f


def lid_link(obj):
    """The lid link (the revolute joint's child body) of a hinged_jar."""
    joint = next(j for j in obj.joints.values() if j.is_revolute)
    return _link_by_body(obj, joint.body1)


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

    # TRUE extension direction: the lid link's local axes give the disc's real radial axis (robust for a
    # thin tilted disc, where the AABB-farthest corner is skewed off the radial line). Take the two lid-
    # local axes NOT aligned with the hinge, and pick the (signed) one pointing most toward the lid bulk.
    Rl = Rot.from_quat(lid_quat).as_matrix()
    ref = lid_tip - anchor
    cands = [s * Rl[:, i] for i in range(3) if abs(np.dot(Rl[:, i], axis)) <= 0.9 for s in (1.0, -1.0)]
    ext_dir = _u(max(cands, key=lambda a: float(np.dot(a, ref))))
    ext_dir = _u(ext_dir - np.dot(ext_dir, axis) * axis)        # force perpendicular to the hinge axis
    reach = float(np.dot(lid_tip - anchor, ext_dir))            # lid length along the radial (>= 0)
    proj = corners @ axis                                       # disc width along the hinge axis
    half_width = float(proj.max() - proj.min()) / 2.0

    return HingeFrame(anchor=anchor, axis=axis, angle=angle, lower=lower, upper=upper,
                      lid_pos=np.asarray(lid_pos, float), lid_quat=np.asarray(lid_quat, float),
                      lid_tip=lid_tip, ext_dir=ext_dir, reach=reach, half_width=half_width)
