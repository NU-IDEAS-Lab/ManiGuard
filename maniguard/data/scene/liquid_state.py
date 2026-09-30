"""Preserve physical liquid state when rebuilding or moving frozen task objects.

Evaluation restores serialized particles; it never refills an empty task. Explicit
benchmark construction can use the generator's Filled state to create particles.
"""

from __future__ import annotations

import copy

import numpy as np


def spill_systems(diagnostics: dict) -> set[str]:
    return {
        p.get("system_name", "water")
        for p in ((diagnostics.get("ltl_safety") or {}).get("propositions") or {}).values()
        if p.get("check") == "spill"
    }


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def pose_matrix(position, orientation):
    from scipy.spatial.transform import Rotation

    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(_array(orientation)).as_matrix()
    mat[:3, 3] = _array(position)
    return mat


def transform_liquid_state(state: dict, transform) -> dict:
    """Transform serialized physical particle poses and velocities in scene coordinates."""
    from scipy.spatial.transform import Rotation

    result = copy.deepcopy(state)
    transform = _array(transform)
    rot = Rotation.from_matrix(transform[:3, :3])
    for instancer in result.get("particle_states", {}).values():
        positions = _array(instancer["particle_positions"]).reshape(-1, 3)
        instancer["particle_positions"] = (positions @ transform[:3, :3].T + transform[:3, 3]).tolist()
        velocities = _array(instancer["particle_velocities"]).reshape(-1, 3)
        instancer["particle_velocities"] = (velocities @ transform[:3, :3].T).tolist()
        if len(positions):
            instancer["particle_orientations"] = (
                (rot * Rotation.from_quat(_array(instancer["particle_orientations"]))).as_quat().tolist()
            )
    return result


def restore_liquid_systems(env, states: dict, systems) -> None:
    """Restore serialized particles after explicit object reconstruction."""
    if not systems:
        return
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    for name in sorted(systems):
        state = states.get(name)
        if not state or sum(state.get("instancer_particle_counts", [])) <= 0:
            raise ValueError(f"Missing nonempty serialized liquid state for {name}")
        env.scene.get_system(name).load_state(
            recursively_convert_to_torch(copy.deepcopy(state)),
            serialized=False,
        )


def capture_movable_liquids(env, diagnostics: dict) -> dict:
    """Capture liquid state and its container poses before a layout-edit retry loop."""
    from maniguard.utils.safety_monitor import ObjectResolver

    result = {}
    resolver = ObjectResolver(env, {obj.name: obj for obj in env.scene.objects})
    for prop in ((diagnostics.get("ltl_safety") or {}).get("propositions") or {}).values():
        if prop.get("check") != "spill":
            continue
        name = prop.get("system_name", "water")
        subjects = resolver.resolve_patterns(prop.get("over", []))
        if not subjects:
            raise ValueError(f"No container resolves for liquid system {name}")
        entry = result.setdefault(
            name, {"state": copy.deepcopy(env.scene.get_system(name).dump_state(serialized=False)), "containers": {}}
        )
        for obj in subjects.values():
            entry["containers"][obj.name] = pose_matrix(*obj.get_position_orientation())
    return result


def restore_moved_liquids(env, captured: dict) -> None:
    """Move saved fluid with its containers and restore it before each validation attempt."""
    states = {}
    for name, entry in captured.items():
        transforms = []
        for obj_name, original in entry["containers"].items():
            obj = env.scene.object_registry("name", obj_name)
            if obj is None:
                raise ValueError(f"Liquid container disappeared: {obj_name}")
            transforms.append(pose_matrix(*obj.get_position_orientation()) @ np.linalg.inv(original))
        transform = transforms[0]
        if any(not np.allclose(transform, other, atol=1e-6) for other in transforms[1:]):
            raise ValueError(f"Containers sharing {name} must move together during layout construction")
        states[name] = transform_liquid_state(entry["state"], transform)
    restore_liquid_systems(env, states, states.keys())
