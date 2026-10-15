"""A spill specification needs a nonempty measured baseline."""

from types import SimpleNamespace

import pytest
from omnigibson.object_states import ContainedParticles

from maniguard.utils.safety_monitor import TaskLTLMonitor


def monitor_for(count):
    state = SimpleNamespace(n=count)
    particle_state = SimpleNamespace(get_value=lambda system: SimpleNamespace(n_in_volume=state.n))
    obj = SimpleNamespace(name="cup_0", states={ContainedParticles: particle_state})
    env = SimpleNamespace(
        task=SimpleNamespace(object_scope={}), scene=SimpleNamespace(get_system=lambda name: object())
    )
    monitor = TaskLTLMonitor(
        env,
        ltl_safety={
            "combined_ltl": "G !spill",
            "propositions": {"spill": {"check": "spill", "over": ["cup_0"], "params": {"spill_threshold": 0.15}}},
        },
        active_objects_by_inst={"cup_0": obj},
    )
    return monitor, state


def test_empty_liquid_baseline_is_initialization_error():
    monitor, _ = monitor_for(0)
    with pytest.raises(RuntimeError, match="baseline"):
        monitor.step(0)
    assert monitor.summary()["error"]["step"] == 0


def test_spill_threshold_is_unchanged_for_positive_baseline():
    monitor, state = monitor_for(100)
    assert not monitor.step(0)["doomed"]
    state.n = 85
    assert not monitor.step(1)["doomed"]
    state.n = 84
    assert monitor.step(2)["doomed"]


def test_losing_every_particle_after_valid_initialization_is_violation():
    monitor, state = monitor_for(100)
    monitor.step(0)
    state.n = 0
    assert monitor.step(1)["doomed"]
