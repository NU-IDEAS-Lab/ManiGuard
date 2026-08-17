"""Unit tests for the pure helpers behind singularity-aware grasp selection.

Only the cuRobo-free helpers are unit-tested here; the IK/margin scoring and the cabinet
place-carry checks are validated by sim collection runs (see the plan/spec).
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from maniguard.data.datagen.executor.grasp_select import joint_margin, roll_variants


def test_roll_variants_preserve_approach_flip_close():
    q0 = R.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
    a, b = roll_variants(q0)
    Ma, Mb = R.from_quat(a).as_matrix(), R.from_quat(b).as_matrix()
    assert np.allclose(Ma[:, 2], Mb[:, 2], atol=1e-6)   # approach (eef +Z) unchanged
    assert np.allclose(Ma[:, 1], -Mb[:, 1], atol=1e-6)  # closing (eef +Y) flipped


def test_roll_variants_involution():
    q0 = R.from_euler("xyz", [-15, 40, 5], degrees=True).as_quat()
    _, b = roll_variants(q0)
    _, bb = roll_variants(b)
    rel = (R.from_quat(bb) * R.from_quat(q0).inv()).magnitude()
    assert rel < 1e-6 or abs(rel - 2 * np.pi) < 1e-6   # rolling twice == identity


def test_joint_margin_basic():
    m = joint_margin([0.0, 0.5], [-1.0, 0.0], [1.0, 1.0])
    assert abs(m - 0.5) < 1e-9                          # j1 margin 0.5 is the min


def test_joint_margin_at_and_outside_limit():
    assert abs(joint_margin([1.0], [-1.0], [1.0]) - 0.0) < 1e-9    # at upper limit -> 0
    assert joint_margin([1.2], [-1.0], [1.0]) < 0                  # outside -> negative
