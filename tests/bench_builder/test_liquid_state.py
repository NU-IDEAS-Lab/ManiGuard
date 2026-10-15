import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_env_fixed_support import _scene

from maniguard.data.bench_builder.perturb_env import build_merged_scene_info
from maniguard.data.scene.liquid_state import (
    capture_movable_liquids,
    restore_liquid_systems,
    restore_moved_liquids,
    transform_liquid_state,
)


def water():
    return {
        "n_instancers": 1,
        "instancer_particle_counts": [1],
        "particle_states": {
            "0": {
                "particle_positions": [[1.0, 2.0, 3.0]],
                "particle_velocities": [[1.0, 0.0, 0.0]],
                "particle_orientations": [[0.0, 0.0, 0.0, 1.0]],
                "particle_scales": [[1.0, 1.0, 1.0]],
            }
        },
    }


def test_particle_transform_preserves_count_and_applies_pose_and_velocity_rotation():
    state = water()
    before = copy.deepcopy(state)
    T = np.array([[0.0, -1.0, 0.0, 10.0], [1.0, 0.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
    result = transform_liquid_state(state, T)
    particle = result["particle_states"]["0"]
    np.testing.assert_allclose(particle["particle_positions"], [[8.0, 21.0, 33.0]])
    np.testing.assert_allclose(particle["particle_velocities"], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(particle["particle_orientations"], [[0.0, 0.0, 2**-0.5, 2**-0.5]])
    assert result["instancer_particle_counts"] == [1] and state == before


def test_environment_merge_moves_fluid_with_objects():
    base = _scene({"support": ({"fixed_base": True}, [0, 0, 0.5]), "cup": ({"fixed_base": False}, [1, 2, 3])})
    base["state"]["registry"]["system_registry"] = {"water": water()}
    room = _scene({"anchor": ({"fixed_base": True}, [10, 20, 0.5])})
    T = np.eye(4)
    T[:3, 3] = [10, 20, 0]
    merged, *_ = build_merged_scene_info(base, room, "support", "anchor", "room", T)
    assert merged["state"]["registry"]["object_registry"]["cup"]["root_link"]["pos"] == [11.0, 22.0, 3.0]
    assert merged["state"]["registry"]["system_registry"]["water"]["particle_states"]["0"]["particle_positions"] == [
        [11.0, 22.0, 3.0]
    ]


@pytest.mark.parametrize("state", [{}, {"water": {"instancer_particle_counts": []}}])
def test_reconstruction_requires_nonempty_serialized_liquid(state):
    with pytest.raises(ValueError, match="nonempty"):
        restore_liquid_systems(None, state, {"water"})


def test_layout_retry_restores_original_fluid_at_new_container_pose():
    obj = SimpleNamespace(name="cup", get_position_orientation=lambda: ([0, 0, 0], [0, 0, 0, 1]))
    seen = []
    system = SimpleNamespace(dump_state=lambda **kw: water(), load_state=lambda state, **kw: seen.append(state))
    env = SimpleNamespace(
        task=SimpleNamespace(object_scope={}),
        scene=SimpleNamespace(objects=[obj], get_system=lambda name: system, object_registry=lambda *a: obj),
    )
    spec = {"ltl_safety": {"propositions": {"spill": {"check": "spill", "over": ["cup"]}}}}
    saved = capture_movable_liquids(env, spec)
    for x in [10, 20]:
        obj.get_position_orientation = lambda x=x: ([x, 0, 0], [0, 0, 0, 1])
        restore_moved_liquids(env, saved)
    np.testing.assert_allclose(seen[0]["particle_states"]["0"]["particle_positions"], [[11, 2, 3]])
    np.testing.assert_allclose(seen[1]["particle_states"]["0"]["particle_positions"], [[21, 2, 3]])
