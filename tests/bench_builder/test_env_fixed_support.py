"""Environment support mobility follows the base task."""
import copy
import numpy as np
import pytest
from maniguard.data.bench_builder.perturb_env import build_merged_scene_info

def _scene(objects):
    return {
        'objects_info': {'init_info': {
            name: {'class_name': 'DatasetObject', 'args': args}
            for name, (args, _) in objects.items()
        }},
        'state': {'registry': {'object_registry': {
            name: {'root_link': {'pos': pos, 'ori': [0, 0, 0, 1]}}
            for name, (_, pos) in objects.items()
        }}},
    }


@pytest.mark.parametrize("fixed_base", [True, False])
def test_merged_anchor_inherits_only_base_support_fixed_state(fixed_base):
    base = _scene({'support': ({'fixed_base': fixed_base}, [0, 0, 0.5]),
                   'jar': ({'fixed_base': False}, [0, 0, 1])})
    room = _scene({'anchor': ({'fixed_base': not fixed_base, 'scale': [2, 3, 4]}, [1, 2, 0.5]),
                   'chair': ({'fixed_base': False}, [2, 3, 0.2])})
    before = copy.deepcopy((base, room))
    merged, injected, _, _ = build_merged_scene_info(base, room, 'support', 'anchor', 'room', np.eye(4))
    init = merged['objects_info']['init_info']
    assert init['anchor']['args']['fixed_base'] is fixed_base
    assert init['anchor']['args']['scale'] == [2, 3, 4]
    assert init['chair']['args']['fixed_base'] is False
    assert init['jar']['args']['fixed_base'] is False
    assert injected == ['jar']
    assert (base, room) == before
