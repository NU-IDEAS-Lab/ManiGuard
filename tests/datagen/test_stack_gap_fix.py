"""Per-task adaptive-gap opt-out for the two stack tasks the widened dest makes unreachable.

task_0025 (chopping_board/drjnag) and task_0026 (folder/lktggf) fail 100% servo_ik_fail because the
adaptive re-stack gap pushes their dest ~0.1m past the arm's IK envelope. ``_dest_gap_max`` disables the
gap (``None`` -> the minimal ``GAP`` dest) for exactly those two keys; ``dest_center`` then returns a dest
pulled back toward the pack (reachable). Every other task keeps ``GAP_MAX`` untouched.
"""
import numpy as np

from maniguard.data.datagen.families.stack import _MINIMAL_GAP_KEYS, _dest_gap_max
from maniguard.data.datagen.families.stack_geom import dest_center


def test_dest_gap_max_disables_only_the_two_tasks():
    # the two blocked tasks -> adaptive gap OFF (None)
    assert _dest_gap_max("chopping_board/drjnag", 0.20) is None   # task_0025
    assert _dest_gap_max("folder/lktggf", 0.20) is None           # task_0026
    assert _MINIMAL_GAP_KEYS == {"chopping_board/drjnag", "folder/lktggf"}
    # every other task (incl. OTHER chopping_board models) keeps the adaptive ceiling byte-identically
    assert _dest_gap_max("chopping_board/mwmzzv", 0.20) == 0.20   # task_0000/0005 — must NOT be touched
    assert _dest_gap_max("chopping_board/iocgzv", 0.20) == 0.20   # task_0003/0004
    assert _dest_gap_max("bowl/tvtive", 0.20) == 0.20             # task_0013
    assert _dest_gap_max("half_waffle/vufawe", 0.20) == 0.20      # task_0023 — the case the gap was built for


def test_disabling_gap_pulls_the_dest_back_toward_the_pack():
    """On a roomy table the adaptive gap widens the dest away from the pack; disabling it (gap_max=None)
    returns the minimal-GAP dest, which sits CLOSER to the pack -> a shorter, reachable re-stack carry."""
    pack_lo, pack_hi = np.array([-0.1, -0.1]), np.array([0.1, 0.1])   # pack centre = origin
    right = np.array([1.0, 0.0])                                       # dest is built to the +x side
    surf_lo, surf_hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])   # large surface (no on-surface bind)
    robot_xy = np.array([0.0, -0.5])                                   # close enough that reach_comfort won't bind
    kw = dict(gap=0.065, surf_lo_xy=surf_lo, surf_hi_xy=surf_hi, robot_xy=robot_xy,
              reach_max=0.85, rail_half=0.0, eef_off=0.0)
    pack_center = 0.5 * (pack_lo + pack_hi)

    adaptive = dest_center(pack_lo, pack_hi, right, 0.05, gap_max=0.20, reach_comfort=0.72, **kw)
    minimal = dest_center(pack_lo, pack_hi, right, 0.05, gap_max=None, reach_comfort=0.72, **kw)

    assert adaptive is not None and minimal is not None
    d_adaptive = float(np.linalg.norm(adaptive - pack_center))
    d_minimal = float(np.linalg.norm(minimal - pack_center))
    assert d_minimal < d_adaptive                       # disabling the gap pulls the dest IN
    assert np.isclose(d_minimal, 0.1 + 0.05 + 0.065)    # == pack_half_r + stack_half + minimal gap
