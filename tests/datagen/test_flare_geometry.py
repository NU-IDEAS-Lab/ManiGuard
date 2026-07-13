"""Pure-numpy tests for the place_across flare/seed helpers. No OG/sim."""
import numpy as np

from maniguard.data.datagen.executor.geometry import (
    arm_flare, noflare_seed, quat_yaw, elbow_lateral_offset,
)


def test_arm_flare_is_abs_j2():
    assert arm_flare([0.1, -1.3, 0.0, -2.5, 0.0, 2.0, 0.75]) == 0.0
    assert abs(arm_flare([0, 0, 1.2, 0, 0, 0, 0]) - 1.2) < 1e-9
    assert abs(arm_flare([0, 0, -0.8, 0, 0, 0, 0]) - 0.8) < 1e-9


def test_noflare_seed_zeros_j2_and_keeps_reach_joints():
    q = np.array([0.3, -1.0, 0.9, -2.4, 0.1, 2.1, 0.7])
    seed = noflare_seed(q, base_pos=[0.0, 0.0, 0.0], base_yaw=0.0, target_xy=[1.0, 0.0])
    assert seed[2] == 0.0                                  # j2 zeroed (no flare)
    assert abs(seed[0]) < 1e-9                             # target +x at base_yaw 0 -> j0 = 0
    assert np.allclose(seed[[1, 3, 4, 5, 6]], q[[1, 3, 4, 5, 6]])   # reach + wrist kept


def test_noflare_seed_j0_aims_at_target_azimuth():
    q = np.zeros(7)
    s1 = noflare_seed(q, [0.0, 0.0, 0.0], 0.0, [0.0, 1.0])
    assert abs(s1[0] - np.pi / 2) < 1e-9                   # target +y -> j0 = +90deg
    s2 = noflare_seed(q, [0.0, 0.0, 0.0], np.pi / 2, [0.0, 1.0])
    assert abs(s2[0]) < 1e-9                               # base already yawed +90 -> j0 = 0


def test_quat_yaw_identity_and_90():
    assert abs(quat_yaw([0, 0, 0, 1])) < 1e-9              # identity -> yaw 0
    assert abs(quat_yaw([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]) - np.pi / 2) < 1e-9


def test_elbow_lateral_offset_in_plane_is_zero():
    off = elbow_lateral_offset([0, 0], [0.2, 0.0], [0.6, 0.0])
    assert off is not None and off < 1e-9


def test_elbow_lateral_offset_perpendicular():
    off = elbow_lateral_offset([0, 0], [0.2, 0.1], [0.6, 0.0])
    assert off is not None and abs(off - 0.1) < 1e-9


def test_elbow_lateral_offset_degenerate_returns_none():
    assert elbow_lateral_offset([0, 0], [0.1, 0.0], [0.0, 0.0]) is None
