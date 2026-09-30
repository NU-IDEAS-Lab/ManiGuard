"""Invalid initial states cannot enter policy evaluation as safe rollouts."""
from types import SimpleNamespace

import pytest

from maniguard.eval import benchmark
from maniguard.utils.safety_monitor import TaskLTLMonitor
from omnigibson.object_states import Dropped


def monitor_with_drop(dropped):
    state = SimpleNamespace(value=dropped)
    state.get_value = lambda: state.value
    obj = SimpleNamespace(name='jar_0', states={Dropped: state})
    env = SimpleNamespace(task=SimpleNamespace(object_scope={}),
                          scene=SimpleNamespace(object_registry=lambda key, name: obj))
    spec = {'combined_ltl': 'G (!dropped)', 'propositions': {
        'dropped': {'state': 'dropped', 'check': 'any', 'over': ['jar_0']},
    }}
    monitor = TaskLTLMonitor(env, ltl_safety=spec, active_objects_by_inst={'jar_0': obj})
    monitor.reset()
    return monitor, state


def test_initial_violation_is_rejected_with_scene_and_labels():
    monitor, state = monitor_with_drop(True)
    with pytest.raises(ValueError) as exc:
        benchmark._validate_initial_safety(monitor, 'task_0002/env')
    assert 'task_0002/env' in str(exc.value)
    assert 'dropped' in str(exc.value)


def test_valid_initial_state_keeps_monitor_history_for_later_violation():
    monitor, state = monitor_with_drop(False)
    benchmark._validate_initial_safety(monitor, 'task_0000/base')
    state.value = True
    assert monitor.step(7)['doomed']
    assert monitor.violation_step == 7
    assert [x['step'] for x in monitor.summary()['log']] == [0, 7]


def test_missing_automaton_is_not_a_valid_initial_state():
    monitor, state = monitor_with_drop(False)
    monitor._monitor = None
    with pytest.raises(RuntimeError, match='monitor'):
        benchmark._validate_initial_safety(monitor, 'task_0000/base')
