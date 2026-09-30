"""Safety must read the objects named in a task specification."""
from types import SimpleNamespace

import pytest

from maniguard.utils.safety_monitor import (
    ObjectResolver, TaskLTLMonitor, build_active_objects_for_ltl,
)
from omnigibson.object_states import ContainedParticles, Dropped, Upright


class MutableState:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def get_value(self, *args):
        self.reads += 1
        return self.value


class Scene:
    def __init__(self, objects):
        self.objects = objects
        self.water = object()

    def object_registry(self, key, name):
        assert key == 'name'
        return next((obj for obj in self.objects if obj.name == name), None)

    def get_system(self, name):
        assert name == 'water'
        return self.water


def make_env():
    obj = SimpleNamespace(name='teacup_178', category='teacup', states={
        Dropped: MutableState(False), Upright: MutableState(True),
        ContainedParticles: MutableState(SimpleNamespace(n_in_volume=100)),
    })
    env = SimpleNamespace(task=SimpleNamespace(object_scope={}), robots=[], scene=Scene([obj]))
    return env, obj


def spec():
    return {
        'activity_name': 'clutter_task_binding_test',
        'combined_ltl': 'G ((!dropped) & upright & (!spilled))',
        'propositions': {
            'dropped': {'state': 'dropped', 'check': 'any', 'over': ['teacup_178']},
            'upright': {'state': 'upright', 'check': 'all', 'over': ['teacup_178']},
            'spilled': {'check': 'spill', 'over': ['teacup_178'], 'system_name': 'water',
                        'params': {'spill_threshold': 0.15}},
        },
    }


def test_scene_instance_name_resolves_through_generated_mapping():
    env, obj = make_env()
    active = build_active_objects_for_ltl(env, spec(), None)
    resolved = ObjectResolver(env, active).resolve_patterns(['teacup_178'])
    assert list(resolved.values()) == [obj]


def test_name_resolution_does_not_escape_active_object_pool():
    env, obj = make_env()
    other = SimpleNamespace(name='teacup_999', category='teacup', states={})
    env.scene.objects.append(other)
    resolver = ObjectResolver(env, {'teacup.n.01_0': obj})
    assert resolver.resolve_patterns(['teacup_999']) == {}


def test_category_pattern_deduplicates_aliases_of_same_scene_object():
    env, obj = make_env()
    resolver = ObjectResolver(env, {'teacup_0': obj, 'teacup.n.01_0': obj})
    assert list(resolver.resolve_patterns(['teacup_*']).values()) == [obj]


def test_synset_pattern_still_resolves_aliases():
    env, obj = make_env()
    resolver = ObjectResolver(env, {'teacup.n.01_0': obj})
    assert list(resolver.resolve_patterns(['teacup.n.01_*']).values()) == [obj]


@pytest.mark.parametrize('violation', ['drop', 'tilt', 'spill'])
def test_real_monitor_rejects_changed_object_state(violation):
    env, obj = make_env()
    task_spec = spec()
    active = build_active_objects_for_ltl(env, task_spec, None)
    monitor = TaskLTLMonitor(env, ltl_safety=task_spec, active_objects_by_inst=active)
    assert monitor._monitor is not None
    monitor.reset()
    assert not monitor.step(0)['doomed']
    if violation == 'drop':
        obj.states[Dropped].value = True
    elif violation == 'tilt':
        obj.states[Upright].value = False
    else:
        obj.states[ContainedParticles].value = SimpleNamespace(n_in_volume=80)
    verdict = monitor.step(1)
    assert verdict['doomed']
    assert monitor.violated and monitor.violation_step == 1
    assert all(state.reads > 0 for state in obj.states.values())


@pytest.mark.parametrize('field,pattern', [('over', 'missing_container'), ('relative_to', 'missing_support')])
def test_missing_declared_pattern_prevents_monitor_initialization(field, pattern):
    env, obj = make_env()
    p = {'state': 'ontop', 'check': 'all', 'over': ['teacup_178'], 'relative_to': ['teacup_178']}
    p[field] = [pattern]
    task_spec = {'activity_name': 'missing_binding_case', 'combined_ltl': 'G supported', 'propositions': {'supported': p}}
    with pytest.raises(ValueError) as error:
        TaskLTLMonitor(env, ltl_safety=task_spec, active_objects_by_inst={'teacup_178': obj})
    message = str(error.value)
    assert all(x in message for x in ['missing_binding_case', 'supported', field, pattern])


def test_each_pattern_must_resolve_even_when_another_matches():
    env, obj = make_env()
    task_spec = {'activity_name': 'partial_match', 'combined_ltl': 'G upright',
                 'propositions': {'upright': {'state': 'upright', 'check': 'all',
                                             'over': ['teacup_178', 'missing_obstacle']}}}
    with pytest.raises(ValueError, match='missing_obstacle'):
        TaskLTLMonitor(env, ltl_safety=task_spec, active_objects_by_inst={'teacup_178': obj})


def test_exact_scene_name_takes_precedence_over_synthetic_alias():
    env, obj = make_env()
    other = SimpleNamespace(name='teacup_0', category='teacup', states={})
    resolver = ObjectResolver(env, {'teacup_0': obj, 'teacup_0_0': other})
    assert list(resolver.resolve_patterns(['teacup_0']).values()) == [other]


@pytest.mark.parametrize('patterns', [['teacup_*', 'teacup_0'], ['teacup_0', 'teacup_*']])
def test_overlapping_patterns_do_not_overwrite_different_objects(patterns):
    env, obj = make_env()
    other = SimpleNamespace(name='teacup_0', category='teacup', states={})
    resolver = ObjectResolver(env, {'teacup_0': obj, 'teacup_0_0': other})
    resolved = list(resolver.resolve_patterns(patterns).values())
    assert len(resolved) == 2
    assert {id(x) for x in resolved} == {id(obj), id(other)}


def test_exact_and_synset_patterns_evaluate_object_only_once():
    env, obj = make_env()
    resolver = ObjectResolver(env, {'teacup_0': obj, 'teacup.n.01_0': obj})
    assert list(resolver.resolve_patterns(['teacup_178', 'teacup.n.01_*']).values()) == [obj]
