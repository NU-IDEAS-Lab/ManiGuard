"""Failed predicate reads must not become Boolean safety observations."""
from types import SimpleNamespace

import pytest

from maniguard.utils.safety_monitor import TaskLTLMonitor
from omnigibson.object_states import ContainedParticles, Dropped, Open, Touching
from omnigibson.object_states.contact_particles import ContactParticles


class State:
    def __init__(self, value):
        self.value = value
        self.error = None

    def get_value(self, *args):
        if self.error:
            raise self.error
        return self.value


class Object:
    def __init__(self, name):
        self.name = name
        self.states = {}
        self.joints = {}
        self.pose_error = None
        self.aabb_error = None

    def get_position_orientation(self):
        if self.pose_error:
            raise self.pose_error
        return [0, 0, 1], [0, 0, 0, 1]

    @property
    def aabb(self):
        if self.aabb_error:
            raise self.aabb_error
        return ([-0.2, -0.2, 0], [0.2, 0.2, 0.5])


def make_monitor(prop, objects, *, formula="G !hazard"):
    system = SimpleNamespace(n_particles=10)
    env = SimpleNamespace(
        task=SimpleNamespace(object_scope={}),
        scene=SimpleNamespace(get_system=lambda name: system),
    )
    return TaskLTLMonitor(
        env, ltl_safety={"combined_ltl": formula, "propositions": {"hazard": prop}},
        active_objects_by_inst={o.name: o for o in objects},
    )


@pytest.mark.parametrize("kind", ["unary", "binary", "spill", "inverted", "carried_pose", "zone_aabb", "contact_particles"])
def test_failed_read_raises_without_recording_a_safe_step(kind):
    obj, support = Object("cup_0"), Object("table_0")
    error = RuntimeError("sensor read unavailable")
    broken = State(False)
    broken.error = error
    if kind == "unary":
        obj.states[Dropped] = broken
        prop = {"state": "dropped", "over": ["cup_0"]}
    elif kind == "binary":
        obj.states[Touching] = broken
        prop = {"state": "touching", "over": ["cup_0"], "relative_to": ["table_0"]}
    elif kind == "spill":
        obj.states[ContainedParticles] = broken
        prop = {"check": "spill", "over": ["cup_0"], "system_name": "water"}
    elif kind == "inverted":
        obj.pose_error = error
        prop = {"check": "inverted", "over": ["cup_0"]}
    elif kind in ("carried_pose", "zone_aabb"):
        if kind == "carried_pose":
            obj.pose_error = error
        else:
            support.aabb_error = error
        prop = {"check": "overhead_forbidden", "carried": ["cup_0"], "zones": ["table_0"]}
    else:
        support.states[ContactParticles] = broken
        prop = {"check": "particles_on_surface", "surface": ["table_0"]}
    monitor = make_monitor(prop, [obj, support])
    with pytest.raises(RuntimeError, match="hazard"):
        monitor.step(0)
    assert monitor.summary()["log"] == []
    assert monitor.violation_step is None  # A read error is not a measured violation.


def test_existing_open_state_failure_must_not_fall_back_to_closed():
    obj = Object("jar_0")
    obj.states[Open] = State(False)
    obj.states[Open].error = RuntimeError("articulation unavailable")
    # A nominal closed joint must not hide a failed installed Open evaluator.
    obj.joints = {"hinge": SimpleNamespace(lower_limit=0, upper_limit=1, get_state=lambda: [0])}
    monitor = make_monitor({"state": "open", "over": ["jar_0"]}, [obj])
    with pytest.raises(RuntimeError, match="hazard"):
        monitor.step(0)


def test_missing_open_state_still_uses_joint_fallback():
    obj = Object("jar_0")
    joint = SimpleNamespace(lower_limit=0, upper_limit=1, position=0.0)
    joint.get_state = lambda: [joint.position]
    obj.joints = {"hinge": joint}
    monitor = make_monitor({"state": "open", "over": ["jar_0"]}, [obj])
    assert not monitor.step(0)["doomed"]
    joint.position = 0.2
    assert monitor.step(1)["doomed"]


def test_joint_read_failure_in_fallback_is_not_closed():
    obj = Object("jar_0")

    def broken():
        raise RuntimeError("hinge read unavailable")

    obj.joints = {"hinge": SimpleNamespace(lower_limit=0, upper_limit=1, get_state=broken)}
    monitor = make_monitor({"state": "open", "over": ["jar_0"]}, [obj])
    with pytest.raises(RuntimeError, match="hazard"):
        monitor.step(0)


def test_failed_trace_requires_reset_before_monitoring_can_resume():
    obj = Object("cup_0")
    state = State(False)
    obj.states[Dropped] = state
    monitor = make_monitor({"state": "dropped", "over": ["cup_0"]}, [obj])
    assert not monitor.step(0)["doomed"]
    state.error = RuntimeError("temporary unavailable")
    with pytest.raises(RuntimeError):
        monitor.step(1)
    summary = monitor.summary()
    assert [r["step"] for r in summary["log"]] == [0]
    assert summary["error"]["step"] == 1
    state.error = None
    with pytest.raises(RuntimeError, match="reset"):
        monitor.step(2)
    monitor.reset()
    assert monitor.summary()["error"] is None
    state.value = True
    assert monitor.step(0)["doomed"]


def test_spill_read_failure_after_baseline_is_not_measured_spill():
    obj = Object("cup_0")
    state = State(SimpleNamespace(n_in_volume=100))
    obj.states[ContainedParticles] = state
    monitor = make_monitor({"check": "spill", "over": ["cup_0"]}, [obj])
    assert not monitor.step(0)["doomed"]
    state.error = RuntimeError("particles unavailable")
    with pytest.raises(RuntimeError, match="hazard"):
        monitor.step(1)
    assert not monitor.violated
    assert [r["step"] for r in monitor.summary()["log"]] == [0]
