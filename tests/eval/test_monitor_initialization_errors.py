"""A declared safety specification must have a complete, usable monitor."""
from types import SimpleNamespace

import pytest

from maniguard.utils import safety_monitor
from omnigibson.object_states import Dropped


def make_task():
    state = SimpleNamespace(value=False)
    state.get_value = lambda: state.value
    obj = SimpleNamespace(name="cup_0", states={Dropped: state})
    env = SimpleNamespace(task=SimpleNamespace(object_scope={}), scene=SimpleNamespace())
    spec = {
        "combined_ltl": "G !hazard",
        "propositions": {"hazard": {"state": "dropped", "over": ["cup_0"]}},
    }
    return env, obj, state, spec


def build(env, obj, spec):
    return safety_monitor.TaskLTLMonitor(
        env, ltl_safety=spec, activity_name="initialization_test",
        active_objects_by_inst={"cup_0": obj},
    )


def test_unknown_proposition_state_cannot_be_silently_omitted():
    env, obj, _, spec = make_task()
    spec["propositions"]["hazard"]["state"] = "missing_state_type"
    with pytest.raises(ValueError, match="hazard"):
        build(env, obj, spec)


def test_formula_ap_without_evaluator_cannot_default_to_false():
    env, obj, _, spec = make_task()
    spec["combined_ltl"] = "G (!hazard & !undefined_hazard)"
    with pytest.raises(ValueError, match="undefined_hazard"):
        build(env, obj, spec)


def test_declared_propositions_without_formula_are_rejected():
    env, obj, _, spec = make_task()
    del spec["combined_ltl"]
    with pytest.raises(ValueError, match="formula"):
        build(env, obj, spec)


def test_missing_spot_cannot_disable_a_declared_specification(monkeypatch):
    env, obj, _, spec = make_task()
    monkeypatch.setattr(safety_monitor, "spot_runtime_available", lambda **kw: False)
    monkeypatch.setattr(safety_monitor, "get_spot_runtime_status", lambda **kw: {"error": "Spot unavailable"})
    with pytest.raises(RuntimeError, match="Spot"):
        build(env, obj, spec)


def test_automaton_construction_error_propagates_with_original_cause(monkeypatch):
    env, obj, _, spec = make_task()
    failure = RuntimeError("automaton backend failure")

    def broken_backend(*args, **kwargs):
        raise failure

    monkeypatch.setattr(safety_monitor, "LTLMonitor", broken_backend)
    with pytest.raises(RuntimeError) as error:
        build(env, obj, spec)
    assert error.value.__cause__ is failure


def test_active_specification_cannot_step_with_a_missing_backend():
    env, obj, _, spec = make_task()
    monitor = build(env, obj, spec)
    monitor._monitor = None  # Simulate loss of an initialized backend.
    with pytest.raises(RuntimeError, match="monitor"):
        monitor.step(0)
    assert monitor.summary()["log"] == []


def test_valid_monitor_still_detects_a_later_violation():
    env, obj, state, spec = make_task()
    monitor = build(env, obj, spec)
    assert not monitor.step(0)["doomed"]
    state.value = True
    assert monitor.step(3)["doomed"]
    assert monitor.violation_step == 3


def test_explicitly_disabled_monitor_does_not_require_spot(monkeypatch):
    env, obj, _, _ = make_task()
    monkeypatch.setattr(safety_monitor, "spot_runtime_available", lambda **kw: False)
    monkeypatch.setattr(safety_monitor, "get_spot_runtime_status", lambda **kw: {"error": "Spot unavailable"})
    monitor = build(env, obj, {})
    assert monitor.step(0)["state"] is None


def test_constant_formula_needs_no_proposition_evaluators():
    env, obj, _, _ = make_task()
    monitor = build(env, obj, {"combined_ltl": "true"})
    assert not monitor.step(0)["doomed"]


def test_instance_state_ap_autogeneration_remains_supported():
    env, obj, state, _ = make_task()
    monitor = build(env, obj, {"combined_ltl": "G !cup_0_dropped"})
    assert not monitor.step(0)["doomed"]
    state.value = True
    assert monitor.step(1)["doomed"]
