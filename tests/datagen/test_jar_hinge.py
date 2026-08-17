"""Unit tests for the pure hinge geometry helpers (numpy/scipy only, no sim). The sim-dependent
``read_hinge`` is validated by the Phase-A smoke, not here."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from maniguard.data.datagen.families.jar_hinge import (
    CLOSE_MARGIN_DEG,
    END_CLEAR_DEG,
    MIN_PAST_VERT_DEG,
    PAD_HALF_GAP_M,
    STRADDLE_CLEAR_M,
    angle_between,
    arc_close_angle,
    arc_waypoints,
    drive_angle,
    face_normal,
    is_closed,
    lid_extension_dir,
    rotate_pose_about_axis,
    rotate_vec_about_axis,
    straddle_pose_from_hull,
    unit_perp,
)

# --- Task 1: rotation primitives ---

def test_rotate_vec_about_z_90deg():
    v = rotate_vec_about_axis([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-9)


def test_rotate_pose_orbits_anchor_and_corotates():
    pos = np.array([1.0, 0.0, 0.0])
    quat = R.identity().as_quat()
    p2, q2 = rotate_pose_about_axis(pos, quat, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(p2, [0.0, 1.0, 0.0], atol=1e-9)          # position orbits
    assert np.allclose(R.from_quat(q2).as_matrix()[:, 0], [0.0, 1.0, 0.0], atol=1e-9)  # +X -> +Y


def test_rotate_pose_about_offset_anchor():
    pos = np.array([2.0, 0.0, 0.0])
    p2, _ = rotate_pose_about_axis(pos, R.identity().as_quat(),
                                   [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi)
    assert np.allclose(p2, [0.0, 0.0, 0.0], atol=1e-9)          # 180deg about the (1,0,0) anchor


def test_unit_perp_and_angle():
    v = unit_perp([1.0, 0.0, 1.0], [0.0, 0.0, 1.0])            # drop the z-component
    assert np.allclose(v, [1.0, 0.0, 0.0], atol=1e-9)
    assert abs(angle_between([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) - np.pi / 2) < 1e-9


# --- Task 2: close direction / closed / extension / insert / arc ---

def test_is_closed_threshold():
    assert is_closed(0.05, 0.0, 2.0)          # range [0,2], frac .05 -> open once pos > 0.1
    assert not is_closed(0.5, 0.0, 2.0)
    assert is_closed(0.0, 0.0, 2.0)


def test_lid_extension_perp_to_axis():
    e = lid_extension_dir(anchor=[0, 0, 0], axis=[0, 1, 0], lid_tip=[0.1, 0.3, 0.1])
    assert abs(np.dot(e, [0, 1, 0])) < 1e-9                     # perpendicular to the axis
    assert np.allclose(e, np.array([0.1, 0.0, 0.1]) / np.linalg.norm([0.1, 0.0, 0.1]), atol=1e-9)


def _synth_hull(anchor, axis, e, reach=0.128, h_lo=0.025, h_hi=0.041, halfw=0.062):
    """Synthetic lid-slab hull: a slab floating h_lo..h_hi off the anchor plane (knuckle offset),
    spanning radial 0..reach and y -halfw..+halfw — mimics the measured kijnrj lid."""
    f = face_normal(axis, e)
    pts = []
    for rr in np.linspace(0.0, reach, 12):
        for yv in np.linspace(-halfw, halfw, 7):
            for hv in (h_lo, h_hi):
                pts.append(np.asarray(anchor, float) + rr * e + yv * np.asarray(axis, float) + hv * f)
    return np.array(pts)


def test_straddle_pose_from_hull_centres_gap_on_the_slab():
    # hinge axis +Y, lid tilted; robot on the +Y side (side_sign=+1) -> approach = -Y (inward)
    anchor, axis = np.array([0, 0, 0.5]), np.array([0, 1.0, 0])
    e = unit_perp([0.3, 0.0, 0.2], axis)
    f = face_normal(axis, e)
    hull = _synth_hull(anchor, axis, e)
    eef_pos, q, reach = straddle_pose_from_hull(anchor, axis, e, hull, side_sign=+1.0)
    M = R.from_quat(q).as_matrix()
    assert np.allclose(M[:, 2], [0, -1, 0], atol=1e-9)          # approach = -side_sign*axis
    assert np.allclose(M[:, 1], f, atol=1e-9)                   # closing axis = disc normal
    assert abs(reach - 0.128) < 1e-6                            # measured rim
    from maniguard.data.datagen.families.jar_hinge import RIM_INSET_M
    fingertip = eef_pos + M[:, 2] * 0.104
    rel = fingertip - anchor
    assert abs(float(rel @ e) - (0.128 - RIM_INSET_M)) < 1e-6   # radially: rim - inset
    # the gap centre sits so the +f pad FACE is `clear` above the slab top (h_top=0.041)
    gap_h = float(rel @ f)
    assert abs(gap_h - (0.041 + STRADDLE_CLEAR_M - PAD_HALF_GAP_M)) < 1e-6
    assert abs(float(rel @ axis) - 0.0) < 1e-6                  # chord centre (y_mid = 0)


def test_drive_angle_2alpha_and_caps():
    axis = [0, 1, 0]
    # alpha=30deg past-vertical open side: 2a=60 within caps -> drive = -60deg (closing sign)
    e30 = R.from_rotvec(np.radians(30) * np.array([0, 1, 0])).apply([0, 0, 1.0])
    d = drive_angle(e30, axis)
    assert abs(np.degrees(abs(d)) - 60) < 1.0 and d < 0
    # alpha=89deg (lid nearly flat back): 2a=178 must cap at a + 90 - END_CLEAR
    e89 = R.from_rotvec(np.radians(89) * np.array([0, 1, 0])).apply([0, 0, 1.0])
    d89 = drive_angle(e89, axis)
    assert abs(np.degrees(abs(d89)) - (89 + 90 - END_CLEAR_DEG)) < 1.0
    # alpha=5deg: 2a=10 < a+MIN_PAST_VERT -> floor at a+margin
    e5 = R.from_rotvec(np.radians(5) * np.array([0, 1, 0])).apply([0, 0, 1.0])
    d5 = drive_angle(e5, axis)
    assert abs(np.degrees(abs(d5)) - (5 + MIN_PAST_VERT_DEG)) < 1.0


def test_arc_close_angle_open_side_drives_through_vertical():
    e = R.from_rotvec(np.radians(30) * np.array([0, 1, 0])).apply([0, 0, 1])   # +Z tilted 30deg about +Y
    ang = arc_close_angle(e, axis=[0, 1, 0], margin_rad=np.radians(CLOSE_MARGIN_DEG))
    assert ang < 0                                             # closing sign
    assert abs(abs(np.degrees(ang)) - (30 + CLOSE_MARGIN_DEG)) < 1.0


def test_arc_waypoints_count_and_endpoint():
    wps = arc_waypoints([1, 0, 0], R.identity().as_quat(), [0, 0, 0], [0, 0, 1],
                        total_angle=np.radians(45), step_deg=10.0)
    assert len(wps) == 5                                      # ceil(45/10)
    theta_last, pos_last, _ = wps[-1]
    assert abs(np.degrees(theta_last) - 45) < 1e-6
    assert np.allclose(pos_last, R.from_rotvec(np.radians(45) * np.array([0, 0, 1])).apply([1, 0, 0]))
