"""A task support remains the same instance when background furniture is present."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from maniguard.data.bench_builder import finalize_base
from maniguard.utils.safety_monitor import (
    ObjectResolver, SafetyPropositionEvaluator, build_active_objects_for_ltl,
)
from omnigibson.object_states import OnTop


@pytest.mark.parametrize('family,prop', [('jar_transport', 'jar_on_support'),
                                        ('lid_transport', 'container_on_support')])
def test_support_is_one_instance_and_other_relations_are_preserved(family, prop):
    spec = {'combined_ltl': 'G support', 'propositions': {
        prop: {'state': 'ontop', 'check': 'all', 'over': ['jar_*'], 'relative_to': ['desk.n.01_*']},
        'lid_on_container': {'state': 'ontop', 'check': 'all', 'over': ['lid_*'], 'relative_to': ['jar_*']},
    }}
    before = deepcopy(spec)
    bound = finalize_base.bind_support_instance(family, spec, 'nightstand_abc_0')
    assert spec == before
    assert bound['propositions'][prop]['relative_to'] == ['nightstand_abc_0']
    assert bound['propositions'][prop]['check'] == 'all'
    assert bound['propositions']['lid_on_container'] == spec['propositions']['lid_on_container']
    assert bound['combined_ltl'] == spec['combined_ltl']


def test_room_desks_do_not_make_supported_jar_unsupported():
    support = SimpleNamespace(name='desk_abc_0', category='desk')
    background = SimpleNamespace(name='desk_xyz_1', category='desk')

    class OnSupport:
        def get_value(self, other):
            return other is support

    jar = SimpleNamespace(name='jar_0', category='jar', states={OnTop: OnSupport()})
    scene = SimpleNamespace(objects=[jar, support, background],
                            object_registry=lambda key, name: next((o for o in [jar, support, background] if o.name == name), None))
    env = SimpleNamespace(scene=scene, robots=[], task=SimpleNamespace(object_scope={}))
    spec = {'propositions': {'jar_on_support': {
        'state': 'ontop', 'check': 'all', 'over': ['jar_*'], 'relative_to': ['desk.n.01_*'],
    }}}
    before = ObjectResolver(env, build_active_objects_for_ltl(env, spec, support.name))
    assert not SafetyPropositionEvaluator(before).build('support', spec['propositions']['jar_on_support'])()
    bound = finalize_base.bind_support_instance('jar_transport', spec, support.name)
    after = ObjectResolver(env, build_active_objects_for_ltl(env, bound, support.name))
    assert SafetyPropositionEvaluator(after).build('support', bound['propositions']['jar_on_support'])()


def test_missing_support_cannot_be_silently_bound():
    spec = {'propositions': {'jar_on_support': {'relative_to': ['desk_*']}}}
    with pytest.raises(ValueError, match='support'):
        finalize_base.bind_support_instance('jar_transport', spec, None)


def test_other_families_are_unchanged():
    spec = {'propositions': {'supported': {'relative_to': ['desk_*']}}}
    assert finalize_base.bind_support_instance('stack_retrieve', spec, None) == spec
