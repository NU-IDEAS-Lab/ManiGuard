import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.executor import grasp_select as gs
from maniguard.data.datagen.executor.contracts import GraspCand


def _quat_approach_world_z_down():
    # eef +Z -> world -Z (straight down): rotate 180 deg about world X
    return Rot.from_euler("x", np.pi).as_quat()


def _quat_approach_horizontal():
    # eef +Z -> world +X (horizontal): rotate +90 deg about world Y
    return Rot.from_euler("y", np.pi / 2).as_quat()


def _quat_approach_tilt(deg):
    # eef +Z tilted `deg` from straight-down, in the x-z plane
    down = Rot.from_euler("x", np.pi)            # +Z -> -Z
    return (Rot.from_euler("y", np.radians(deg)) * down).as_quat()


def test_is_top_down_straight_down_true():
    assert gs.is_top_down(_quat_approach_world_z_down()) is True


def test_is_top_down_horizontal_false():
    assert gs.is_top_down(_quat_approach_horizontal()) is False


def test_is_top_down_brackets_45deg_cut():
    assert gs.is_top_down(_quat_approach_tilt(30)) is True
    assert gs.is_top_down(_quat_approach_tilt(60)) is False


def test_is_top_down_roll_invariant():
    q = _quat_approach_world_z_down()
    q_rolled = (Rot.from_quat(q) * Rot.from_rotvec([0, 0, np.pi])).as_quat()  # 180 about approach
    assert gs.is_top_down(q) == gs.is_top_down(q_rolled) is True


def _cand(cid, *, reachable, margin, is_td, score=0.5):
    c = GraspCand(id=cid, eef_pos=np.zeros(3), eef_quat=np.array([0, 0, 0, 1.0]))
    c.reachable, c.margin, c.is_top_down, c.score = reachable, margin, is_td, score
    return c


def test_rank_key_prefers_top_down_over_higher_margin_side():
    side = _cand(1, reachable=True, margin=0.9, is_td=False)   # higher margin
    top = _cand(2, reachable=True, margin=0.3, is_td=True)     # lower margin, top-down
    ranked = sorted([side, top], key=lambda c: gs.rank_key(c, True), reverse=True)
    assert [c.id for c in ranked] == [2, 1]


def test_rank_key_ignores_top_down_when_flag_off():
    side = _cand(1, reachable=True, margin=0.9, is_td=False)
    top = _cand(2, reachable=True, margin=0.3, is_td=True)
    ranked = sorted([side, top], key=lambda c: gs.rank_key(c, False), reverse=True)
    assert [c.id for c in ranked] == [1, 2]   # margin only


def test_rank_key_unreachable_sorts_last():
    good = _cand(1, reachable=True, margin=0.1, is_td=False)
    bad = _cand(2, reachable=False, margin=0.9, is_td=True)
    ranked = sorted([bad, good], key=lambda c: gs.rank_key(c, True), reverse=True)
    assert [c.id for c in ranked] == [1, 2]


def test_score_grasps_prefer_top_down_orders_topdown_first(monkeypatch):
    import torch as th

    from maniguard.data.datagen.executor import grasp_select as gsmod

    class _Res:
        salvaged = False
        pos_err = 0.0
        rot_err = 0.0

        def __init__(self, q):
            self.arm_traj = th.tensor([q], dtype=th.float32)

    # every solve returns a comfortable mid-range config (equal margins for both grasps)
    monkeypatch.setattr(gsmod, "solve_segment", lambda *a, **k: _Res([0.0] * 7))

    class _World:
        motion_gen = None

        def update_obstacles(self, ignore_objects=None):
            pass

    class _Robot:
        default_arm = "0"
        arm_control_idx = {"0": np.arange(7)}
        joint_lower_limits = np.full(7, -3.0)
        joint_upper_limits = np.full(7, 3.0)

        def get_joint_positions(self):
            return np.zeros(7)

    side = GraspCand(id=1, eef_pos=np.zeros(3), eef_quat=_quat_approach_horizontal(), approach="side")
    top = GraspCand(id=2, eef_pos=np.zeros(3), eef_quat=_quat_approach_world_z_down(), approach="top_down")
    out = gsmod.score_grasps(_World(), _Robot(), object(), [side, top], prefer_top_down=True)
    assert [c.id for c in out] == [2, 1]            # top-down first despite equal margins
    assert out[0].is_top_down is True and out[1].is_top_down is False

    # flag off -> equal margins, stable input order preserved
    side2 = GraspCand(id=1, eef_pos=np.zeros(3), eef_quat=_quat_approach_horizontal())
    top2 = GraspCand(id=2, eef_pos=np.zeros(3), eef_quat=_quat_approach_world_z_down())
    out2 = gsmod.score_grasps(_World(), _Robot(), object(), [side2, top2], prefer_top_down=False)
    assert out2[0].is_top_down is False and out2[1].is_top_down is True


def test_relocate_prefer_top_down_default_false():
    from maniguard.data.datagen.executor.contracts import FamilySkeleton

    class _Fam(FamilySkeleton):
        def grasp_candidates(self, ctx):
            return []

        def derive_segments(self, ctx, grasp, params):
            return []

    assert _Fam().relocate_prefer_top_down() is False


def test_cabinet_relocate_prefer_top_down_true():
    from maniguard.data.datagen.families.cabinet import CabinetSkeleton

    sk = object.__new__(CabinetSkeleton)        # bypass __init__ (DB / geom file deps)
    assert sk.relocate_prefer_top_down() is True


def _cabinet_stub_with_two_grasps():
    from maniguard.data.datagen.families.cabinet import CabinetSkeleton

    sk = object.__new__(CabinetSkeleton)
    sk._db = {"objects": {"cat/mod": {
        "bbox_size": [0.05, 0.05, 0.10],
        "upright_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grasps": [
            {"id": 5, "position": [0.0, 0.0, 0.02], "orientation_xyzw": [0, 0, 0, 1], "approach_hint": "side"},
            {"id": 7, "position": [0.0, 0.0, 0.02], "orientation_xyzw": [0, 0, 0, 1], "approach_hint": "top_down"},
        ]}}}
    return sk


def test_best_grasp_id_prefers_top_down_subset():
    sk = _cabinet_stub_with_two_grasps()
    sk._p = {"obstacle_gids": [5, 7],            # side scored first, but...
             "topdown_by_key_gid": {("cat/mod", 5): False, ("cat/mod", 7): True}}
    assert sk._best_grasp_id("cat/mod") == 7     # top-down preferred over the side


def test_best_grasp_id_falls_back_when_no_top_down():
    sk = _cabinet_stub_with_two_grasps()
    sk._p = {"obstacle_gids": [5, 7],
             "topdown_by_key_gid": {("cat/mod", 5): False, ("cat/mod", 7): False}}
    assert sk._best_grasp_id("cat/mod") == 5     # no top-down -> full pool, first central
