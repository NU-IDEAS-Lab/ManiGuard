"""Offline tests for blocker_placement's obstacle branch: the relocated obstacle parks BEHIND the
closed drawer face (the dead zone the drawer never sweeps), hugging the +p near-robot table edge,
not in front of the robot. Pure numpy — no OG/sim."""
import numpy as np

from maniguard.data.datagen.families.cabinet_geom import (
    EDGE_MARGIN,
    OBSTACLE_BACK_OFFSET,
    CabinetLayout,
    blocker_placement,
)


def _layout(*, d_front=-0.15, j_current=0.07, robot_xy=(-0.6, -0.1),
            table_lo=(-0.5, -0.5), table_hi=(0.5, 0.5)):
    """task_0002-like layout: d=+Y (drawer opens), p=-X (+p=near-robot, robot to the -X side)."""
    return CabinetLayout(
        d=np.array([0.0, 1.0]), p=np.array([-1.0, 0.0]), cab_xy=np.array([0.0, -0.4]),
        d_front=d_front, d_back=d_front - 0.4, p_lo=-0.2, p_hi=0.2, drawer_floor_z=0.4,
        stroke=0.36, j_extract=0.072, j_current=j_current,
        table_lo=np.array(table_lo, float), table_hi=np.array(table_hi, float),
        robot_xy=np.array(robot_xy, float))


def test_obstacle_parks_behind_closed_face():
    L = _layout()
    oh = 0.04
    place = blocker_placement(L, [-0.1, 0.05], oh, "obstacle", open_dist=0.30)
    face = L.d_front - L.j_current                       # closed cabinet face d-coord = -0.22
    dc, pc = float(place @ L.d), float(place @ L.p)
    assert dc < face                                     # BEHIND the face (the drawer never sweeps here)
    assert abs(dc - (face - oh - OBSTACLE_BACK_OFFSET)) < 1e-9
    assert abs(pc - (0.5 - (oh + EDGE_MARGIN))) < 1e-9   # +p near-robot edge, pulled in by half+margin


def test_obstacle_not_in_front_of_robot():
    # the OLD bug: obstacle at the robot's straight-ahead foot (robot@d = -0.1), in front of the face.
    L = _layout()
    place = blocker_placement(L, [-0.1, 0.05], 0.04, "obstacle", open_dist=0.30)
    face = L.d_front - L.j_current
    assert float(place @ L.d) < face                     # never in front (where it blocked the next pick)
