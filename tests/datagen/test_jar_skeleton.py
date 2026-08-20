"""Unit tests for the sim-free parts of the jar family skeleton: the side-grasp filter and the
derive_segments Phase-A/Phase-B structure (the hinge cache is pre-seeded to avoid a sim)."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from maniguard.data.datagen.executor.contracts import GraspCand, Grip, Mode, SampleParams, TaskContext
from maniguard.data.datagen.families import jar_hinge as JH
from maniguard.data.datagen.families.jar import JarSkeleton


def _cand(cid, approach_deg_from_down):
    # build a quat whose eef +Z (approach) is `approach_deg_from_down` off straight-down
    q = (R.from_rotvec(np.radians(approach_deg_from_down) * np.array([1, 0, 0]))
         * R.from_matrix(np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))).as_quat()
    return GraspCand(id=cid, eef_pos=np.zeros(3), eef_quat=q)


# --- Task 4: side-grasp filter ---

def test_side_filter_keeps_side_drops_topdown():
    cands = [_cand(0, 5.0), _cand(1, 80.0)]      # g0 near straight-down, g1 near horizontal (side)
    kept = JarSkeleton._side_filter(cands)
    assert [c.id for c in kept] == [1]


def test_side_filter_fallback_when_all_topdown():
    cands = [_cand(0, 2.0), _cand(1, 8.0)]
    kept = JarSkeleton._side_filter(cands)
    assert [c.id for c in kept] == [0, 1]        # nothing side -> keep all (never yield zero)


# --- Task 5: derive_segments structure ---

class _FakeObj:
    def get_position_orientation(self):
        return np.zeros(3), R.identity().as_quat()


def _ctx():
    return TaskContext(env=None, robot=None, target=_FakeObj(), target_key="hinged_jar/kijnrj",
                       target_name="jar1", goal_center=np.array([0.3, 0.0, 0.67]), goal_radius=0.06,
                       diagnostics={})


def _seed_hinge(skel, ctx):
    # pre-populate the cache (test seam) so derive_segments runs without a sim
    axis = np.array([0.0, 1.0, 0.0])
    e = JH.unit_perp(np.array([0.15, 0.0, 0.22]), axis)
    hf = JH.HingeFrame(anchor=np.array([0.0, 0.0, 0.5]), axis=axis,
                       angle=2.6, lower=0.0, upper=2.7, lid_pos=np.array([0.0, 0.0, 0.5]),
                       lid_quat=R.identity().as_quat(), lid_tip=np.array([0.15, 0.0, 0.72]),
                       ext_dir=e, reach=0.13, half_width=0.065)
    hull = np.array([hf.anchor + rr * e + yv * axis + hv * JH.face_normal(axis, e)
                     for rr in np.linspace(0, 0.128, 8) for yv in (-0.06, 0.06) for hv in (0.025, 0.041)])
    rs, rq, re_, fN = JH.ride_plan(hf.anchor, hf.axis, e, hull, side_sign=+1.0)
    skel._h[ctx.target_name] = {"hf": hf, "e": e, "ride_start": rs, "ride_q": rq,
                                "ride_end": re_, "fN": fN, "hull": hull, "s": 1.0,
                                "desk_top": -10.0}


def test_derive_segments_phase_order_and_modes():
    skel = JarSkeleton()
    ctx = _ctx()
    _seed_hinge(skel, ctx)
    grasp = GraspCand(id=3, eef_pos=np.array([0.2, 0.0, 0.6]),
                      eef_quat=R.from_matrix(np.column_stack([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])).as_quat())
    segs = skel.derive_segments(ctx, grasp, SampleParams())
    names = [s.name for s in segs]
    # Phase A = the lid-ride: under (FREE) -> slip -> ONE straight ride -> retrace back out
    assert names[:4] == ["lid_under", "lid_slip", "lid_ride", "lid_back"]
    assert segs[0].mode == Mode.FREE and segs[0].grip == Grip.OPEN
    # the gripper NEVER closes on the lid (unilateral rest contact; no coupling to drag the jar)
    assert all(s.grip != Grip.CLOSE for s in segs[:4])
    assert segs[1].mode == Mode.SERVO and segs[2].mode == Mode.SERVO
    assert segs[2].servo_step_m is not None                      # fine, slow ride
    assert segs[3].replay_reverse                                # retreat retraces the ride path
    # the ride is ONE fixed-orientation straight segment (identical quat across Phase A)
    for s in segs[1:4]:
        assert np.allclose(s.eef_quat, segs[0].eef_quat)
    assert names[-5:] == ["side_pre_grasp", "descend", "lift", "transport", "to_goal"]
    assert any(s.name == "descend" and s.grip == Grip.CLOSE for s in segs)   # Phase B still grasps the jar


# --- Task 6: registration ---

def test_jar_registered_in_family():
    from maniguard.data.datagen.families import FAMILY
    assert FAMILY.get("jar") is JarSkeleton
