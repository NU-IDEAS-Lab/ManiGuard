import numpy as np

from maniguard.data.datagen.executor.geometry import aabb_sphere_hit, object_forward_extent


def test_forward_extent_points_along_direction():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.1, 0.1])
    eef = np.array([0.0, 0.05, 0.05])
    # object extends +0.2 along +x from the eef
    assert np.isclose(object_forward_extent(lo, hi, eef, np.array([1.0, 0, 0])), 0.2, atol=1e-6)
    assert np.isclose(object_forward_extent(lo, hi, eef, np.array([-1.0, 0, 0])), 0.0, atol=1e-6)


def test_aabb_sphere_hit_with_offset():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])
    c = np.array([0.3, 0.05, 0.05])
    assert not aabb_sphere_hit(lo, hi, c, 0.05)                             # closest AABB pt 0.2 away -> miss
    assert aabb_sphere_hit(lo, hi, c, 0.05, offset=np.array([0.2, 0, 0]))   # shift AABB +0.2 -> touches
