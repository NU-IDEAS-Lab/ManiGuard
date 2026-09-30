"""Initialization errors are not evidence that a generated scene is safe."""
from types import SimpleNamespace

import pytest

from maniguard.task_generation.pipeline_common import validate_ltl_step0
from maniguard.utils import safety_monitor
from omnigibson.object_states import Dropped


def inputs(value=False):
    state = SimpleNamespace(get_value=lambda: value)
    obj = SimpleNamespace(name="cup_0", states={Dropped: state})
    env = SimpleNamespace(task=SimpleNamespace(object_scope={}), scene=SimpleNamespace())
    spec = {"combined_ltl": "G !dropped", "propositions": {
        "dropped": {"state": "dropped", "over": ["cup_0"]},
    }}
    return env, {"cup_0": obj}, spec


def test_monitor_construction_failure_does_not_pass_gate(monkeypatch):
    env, active, spec = inputs()

    def broken(*args, **kwargs):
        raise RuntimeError("cannot initialize safety monitor")

    monkeypatch.setattr(safety_monitor, "TaskLTLMonitor", broken)
    with pytest.raises(RuntimeError, match="cannot initialize"):
        validate_ltl_step0(env, "task", None, active, spec)


def test_monitor_step_failure_does_not_pass_gate(monkeypatch):
    env, active, spec = inputs()

    def broken(*args, **kwargs):
        raise RuntimeError("cannot evaluate initial predicates")

    monkeypatch.setattr(safety_monitor.TaskLTLMonitor, "step", broken)
    with pytest.raises(RuntimeError, match="cannot evaluate"):
        validate_ltl_step0(env, "task", None, active, spec)


@pytest.mark.parametrize("dropped", [False, True])
def test_valid_evaluation_distinguishes_safe_from_violating_state(dropped):
    env, active, spec = inputs(dropped)
    assert validate_ltl_step0(env, "task", None, active, spec) == (not dropped, {"dropped": dropped})


def test_no_specification_keeps_optional_unmonitored_path():
    env, active, _ = inputs()
    assert validate_ltl_step0(env, "task", None, active, None) == (True, {})
