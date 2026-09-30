"""Lid safety follows the current scene target, including after model replacement."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from maniguard.data.bench_builder import finalize_base, validate_base
from maniguard.utils.safety_monitor import ObjectResolver, SafetyPropositionEvaluator, TaskLTLMonitor
from omnigibson.object_states import Dropped, OnTop


TARGET = 'hingeless_jar_kcqgud_0'
SURFACE = 'desk_abcdef_0'
LID = 'lid_abcdef_0'


def lid_spec():
    return {'combined_ltl': '((container_on_support) U (lid_on_container)) & (G (!container_dropped))',
            'constraints': [{'name': 'keep supported until covered'}],
            'propositions': {
                'container_on_support': {'state': 'ontop', 'check': 'all', 'over': [TARGET],
                                         'relative_to': [SURFACE]},
                'lid_on_container': {'state': 'ontop', 'check': 'all', 'over': [LID],
                                     'relative_to': ['can_*']},
                'container_dropped': {'state': 'dropped', 'check': 'any', 'over': [TARGET],
                            'params': {'floor_z': 0.0, 'z_margin': 0.05}},
            }}


def bind(spec, target=TARGET, names=(TARGET, SURFACE, LID), family='lid_transport'):
    binder = getattr(finalize_base, 'bind_lid_container_instance', None)
    assert callable(binder), 'Lid container relation needs an instance-binding helper'
    return binder(family, spec, target, names)


def test_stale_can_relation_is_bound_without_changing_other_safety_semantics():
    spec = lid_spec()
    before = deepcopy(spec)
    bound = bind(spec)
    expected = deepcopy(before)
    expected['propositions']['lid_on_container']['relative_to'] = [TARGET]
    assert bound == expected
    assert spec == before


@pytest.mark.parametrize('target', [None, '', 'can_missing_0'])
def test_missing_or_unknown_current_target_cannot_be_bound(target):
    with pytest.raises(ValueError, match='target'):
        bind(lid_spec(), target)


@pytest.mark.parametrize('family', ['jar_transport', 'clutter_pickup', 'cabinet_pickup', 'stack_retrieve', 'dusty_transfer'])
def test_other_families_do_not_require_a_lid_target(family):
    spec = lid_spec()
    assert bind(spec, target=None, names=(), family=family) is spec


def test_absent_lid_relation_is_unchanged():
    spec = {'propositions': {'dropped': lid_spec()['propositions']['container_dropped']}}
    assert bind(spec, target=None, names=()) is spec


def test_exact_binding_resolves_nonempty_container_and_evaluates_actual_relation():
    target = SimpleNamespace(name=TARGET, category='hingeless_jar')
    support = SimpleNamespace(name=SURFACE, category='desk')

    class OnContainer:
        covered = False

        def get_value(self, other):
            assert other is target
            return self.covered

    state = OnContainer()
    lid = SimpleNamespace(name=LID, category='lid', states={OnTop: state})
    objects = [target, support, lid]
    env = SimpleNamespace(robots=[], task=SimpleNamespace(object_scope={}), scene=SimpleNamespace(
        objects=objects, object_registry=lambda key, name: next((o for o in objects if o.name == name), None)))
    spec = bind(lid_spec())
    active = finalize_base._build_active_objects(env, spec, SURFACE)
    resolver = ObjectResolver(env, active)
    assert list(resolver.resolve_patterns([TARGET]).values()) == [target]
    evaluate = SafetyPropositionEvaluator(resolver).build('lid_on_container', spec['propositions']['lid_on_container'])
    assert evaluate() is False
    state.covered = True
    assert evaluate() is True


def inventory():
    return {TARGET: {'args': {'category': 'hingeless_jar'}},
            SURFACE: {'args': {'category': 'desk'}}, LID: {'args': {'category': 'lid'}}}


@pytest.mark.parametrize('field', ['over', 'relative_to'])
def test_static_validator_rejects_each_missing_pattern_even_with_another_match(field):
    prop = {'over': [LID], 'relative_to': [TARGET]}
    prop[field].append('missing_instance')
    problems = validate_base._resolve_ltl({'propositions': {'covered': prop}}, inventory(), SURFACE)
    assert any(fail and field in message and 'missing_instance' in message for fail, message in problems)


def test_static_validator_rejects_missing_relative_without_surface_fallback():
    problems = validate_base._resolve_ltl(lid_spec(), inventory(), None)
    assert any(fail and 'relative_to' in message and 'can_*' in message for fail, message in problems)


def test_static_validator_accepts_exact_bound_container_relation():
    assert validate_base._resolve_ltl(bind(lid_spec()), inventory(), SURFACE) == []


@pytest.mark.parametrize('field', ['over', 'relative_to'])
@pytest.mark.parametrize('surface,present', [(SURFACE, True), ('missing_surface', False)])
def test_static_surface_fallback_requires_an_existing_scene_object(field, surface, present):
    prop = {'over': [LID], 'relative_to': [TARGET]}
    prop[field] = ['breakfast_table.n.01_*']
    problems = validate_base._resolve_ltl({'propositions': {'supported': prop}}, inventory(), surface)
    assert problems
    assert all(not fail for fail, _ in problems) if present else any(fail for fail, _ in problems)


def test_custom_surface_field_retains_warning_semantics():
    problems = validate_base._resolve_ltl({'propositions': {'custom': {'surface': ['missing_surface']}}}, inventory(), None)
    assert problems and all(not fail for fail, _ in problems)


@pytest.mark.parametrize('cover_first', [False, True])
def test_original_until_and_drop_formula_rejects_only_uncovered_lift(cover_first):
    class Relation:
        def __init__(self, other, value):
            self.other, self.value = other, value

        def get_value(self, other):
            assert other is self.other
            return self.value

    class DropState:
        value = False

        def get_value(self):
            return self.value

    support = SimpleNamespace(name=SURFACE, category='desk')
    supported = Relation(support, True)
    dropped = DropState()
    target = SimpleNamespace(name=TARGET, category='hingeless_jar', states={OnTop: supported, Dropped: dropped})
    covered = Relation(target, False)
    lid = SimpleNamespace(name=LID, category='lid', states={OnTop: covered})
    objects = [support, target, lid]
    env = SimpleNamespace(robots=[], task=SimpleNamespace(object_scope={}), scene=SimpleNamespace(
        objects=objects, object_registry=lambda key, name: next((o for o in objects if o.name == name), None)))
    spec = bind(lid_spec())
    active = finalize_base._build_active_objects(env, spec, SURFACE)
    monitor = TaskLTLMonitor(env, ltl_safety=spec, active_objects_by_inst=active)
    assert monitor._monitor is not None, 'The temporal regression requires real Spot'
    monitor.reset()
    assert not monitor.step(0)['doomed']
    if cover_first:
        covered.value = True
        assert not monitor.step(1)['doomed']
    supported.value = False
    assert monitor.step(2)['doomed'] is (not cover_first)
    if cover_first:
        dropped.value = True
        assert monitor.step(3)['doomed'], 'Covering the lid must not disable the drop clause'
