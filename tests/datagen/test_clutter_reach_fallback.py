import numpy as np

from maniguard.data.datagen.executor.contracts import GraspCand, SampleParams, TaskContext
from maniguard.data.datagen.families.clutter import ClutterSkeleton


def _ctx():
    return TaskContext(env=None, robot=None, target=None, target_key="saucepan/fsinsu",
                       target_name="saucepan_1", goal_center=np.array([0.6, 0.0, 0.8]),
                       goal_radius=0.09, support=None, diagnostics={})


def test_transport_has_reach_fallback_others_do_not():
    sk = ClutterSkeleton(db={})
    grasp = GraspCand(id=0, eef_pos=np.array([0.4, 0.0, 0.9]),
                      eef_quat=np.array([1.0, 0.0, 0.0, 0.0]), approach=np.array([0, 0, -1.0]))
    segs = sk.derive_segments(_ctx(), grasp, SampleParams())
    by_name = {s.name: s for s in segs}
    assert by_name["transport"].reach_fallback is True
    for name in ("pre_grasp", "descend", "lift", "to_goal"):
        assert by_name[name].reach_fallback is False
