"""Unit tests for the dusty family — pure geometry/contract pieces (no sim)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def test_servo_orient_waypoints_endpoints_and_monotonic():
    from maniguard.data.datagen.executor.geometry import servo_orient_waypoints
    q0 = Rot.identity().as_quat()
    q1 = Rot.from_euler("x", 90, degrees=True).as_quat()
    qs = servo_orient_waypoints(q0, q1, 6)
    assert len(qs) == 6
    np.testing.assert_allclose(np.abs(np.dot(qs[-1], q1)), 1.0, atol=1e-6)  # ends at target
    angs = [np.degrees((Rot.from_quat(q0).inv() * Rot.from_quat(q)).magnitude()) for q in qs]
    assert all(b > a for a, b in zip(angs, angs[1:]))                        # monotone ramp
    np.testing.assert_allclose(angs[-1], 90.0, atol=1e-4)


def test_orient_slerp_defaults_off():
    from maniguard.data.datagen.executor.contracts import MotionSegment
    seg = MotionSegment("x", np.zeros(3), np.array([0, 0, 0, 1.0]))
    assert seg.orient_slerp is False


def test_family_abort_carries_stage_detail():
    from maniguard.data.datagen.executor.contracts import FamilyAbort
    e = FamilyAbort("wipe_incomplete", remaining=3)
    assert e.stage == "wipe_incomplete" and e.detail == {"remaining": 3}


def test_source_cat_model_from_diagnostics():
    from maniguard.data.datagen.families.dusty import source_cat_model
    diag = {"selection": {"spawn_specs": [
        {"role": "food", "category": "potato", "model": "lgupkq"},
        {"role": "source", "category": "chopping_board", "model": "ktxcvz"},
        {"role": "dest", "category": "stockpot", "model": "azoiaq"}]}}
    assert source_cat_model(diag) == ("chopping_board", "ktxcvz")


def test_food_margin_filter_drops_near_grasps():
    from maniguard.data.datagen.families.dusty import filter_grasps_by_food_margin
    g = np.array([[0.0, 0.0], [0.10, 0.0], [0.0, 0.20]])
    keep = filter_grasps_by_food_margin(g, food_xy=np.array([0.0, 0.0]), margin_m=0.08)
    assert keep == [1, 2]


def test_dynamic_food_margin_scales_with_food_size():
    from maniguard.data.datagen.families.dusty import dynamic_food_margin
    # small food (6cm): floored at 0.08
    assert dynamic_food_margin([0, 0, 0], [0.06, 0.05, 0.05]) == 0.08
    # long food (16cm): half-extent 0.08 + 0.05 clearance
    np.testing.assert_allclose(dynamic_food_margin([0, 0, 0], [0.16, 0.06, 0.05]), 0.13)


def test_wipe_next_xy_nearest_and_done():
    from maniguard.data.datagen.families.dusty import wipe_next_xy
    rem = np.array([[1.0, 0.0], [0.2, 0.1]])
    np.testing.assert_allclose(wipe_next_xy(rem, np.array([0.0, 0.0])), [0.2, 0.1])
    assert wipe_next_xy(np.zeros((0, 2)), np.array([0.0, 0.0])) is None


def test_pour_stance_far_edge_over_dest():
    from maniguard.data.datagen.families.dusty import pour_stance_and_axis
    # gripper grabs the rim at (-r, 0) of a source centred at origin; dest at (0.5, 0)
    stance, axis = pour_stance_and_axis(
        grasp_xy=np.array([-0.15, 0.0]), src_center_xy=np.array([0.0, 0.0]),
        dest_center_xy=np.array([0.5, 0.0]), r_src=0.15)
    far_edge = stance + np.array([0.15, 0.0])            # far edge lands on dest centre
    np.testing.assert_allclose(far_edge, [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-6)  # horizontal, ⟂ far_dir
    assert abs(axis[2]) < 1e-9


def test_pour_axis_dips_far_edge():
    from maniguard.data.datagen.families.dusty import pour_stance_and_axis
    from scipy.spatial.transform import Rotation as R
    _, axis = pour_stance_and_axis(
        grasp_xy=np.array([-0.15, 0.0]), src_center_xy=np.array([0.0, 0.0]),
        dest_center_xy=np.array([0.5, 0.0]), r_src=0.15)
    # rotating the far-edge offset (+x) by +30° about the axis must move it DOWN
    tilted = R.from_rotvec(axis * np.deg2rad(30)).apply(np.array([0.15, 0.0, 0.0]))
    assert tilted[2] < -0.01
