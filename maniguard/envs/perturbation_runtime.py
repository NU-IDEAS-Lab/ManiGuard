"""Apply runtime perturbation patches and local reconstruct steps from task metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _task_metadata(scene) -> dict[str, Any]:
    data = scene.get_task_metadata("task")
    return data if isinstance(data, dict) else {}


def get_perturbation_spec(scene) -> dict[str, Any] | None:
    spec = _task_metadata(scene).get("perturbation")
    return spec if isinstance(spec, dict) else None


def _resolve_asset_path(raw_path: str | None, scene_root: str | Path | None) -> str | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    if scene_root is None:
        return str(candidate)
    return str((Path(scene_root) / candidate).resolve())


def _iter_object_materials(obj) -> list[Any]:
    try:
        materials = list(getattr(obj, "materials", []) or [])
    except Exception:
        materials = []
    deduped: list[Any] = []
    seen: set[str] = set()
    for material in materials:
        prim_path = str(getattr(material, "prim_path", ""))
        if prim_path in seen:
            continue
        seen.add(prim_path)
        deduped.append(material)
    return deduped


def _apply_color_override(material: Any, color: list[float]) -> None:
    color_arr = np.asarray(color, dtype=np.float32).reshape(3)
    try:
        if hasattr(material, "diffuse_tint"):
            material.diffuse_tint = color_arr
    except Exception:
        pass
    try:
        if hasattr(material, "diffuse_color_constant"):
            material.diffuse_color_constant = color_arr
    except Exception:
        pass
    try:
        if hasattr(material, "color"):
            material.color = color_arr
    except Exception:
        pass


def _apply_texture_override(material: Any, texture_path: str) -> None:
    try:
        if hasattr(material, "diffuse_texture"):
            material.diffuse_texture = texture_path
    except Exception:
        pass


def _apply_visual_override(scene, override: dict[str, Any], *, scene_root: str | Path | None) -> dict[str, Any]:
    scene_object_name = str(override.get("scene_object_name", "") or "")
    obj = scene.object_registry("name", scene_object_name) if scene_object_name else None
    if obj is None:
        return {
            "scene_object_name": scene_object_name,
            "applied": False,
            "reason": "missing_object",
        }

    materials = _iter_object_materials(obj)
    if not materials:
        return {
            "scene_object_name": scene_object_name,
            "applied": False,
            "reason": "missing_materials",
        }

    color = override.get("color")
    texture_file = _resolve_asset_path(override.get("texture_file"), scene_root)
    for material in materials:
        if isinstance(color, list) and len(color) == 3:
            _apply_color_override(material, color)
        if texture_file:
            _apply_texture_override(material, texture_file)

    return {
        "scene_object_name": scene_object_name,
        "applied": True,
        "material_count": len(materials),
        "texture_file": texture_file,
        "color": color,
    }


def _restack_chain(env, og, instruction: dict[str, Any], *, video_recorder=None) -> dict[str, Any]:
    from omnigibson.task_generation.pipeline_common import place_upright_on_surface, stabilize_active_objects

    support_name = str(instruction.get("support_name", "") or "")
    chain_names = [str(name) for name in instruction.get("chain", []) if str(name)]
    support_obj = env.scene.object_registry("name", support_name) if support_name else None
    if support_obj is None or not chain_names:
        return {
            "type": "restack_chain",
            "applied": False,
            "reason": "missing_support_or_chain",
        }

    chain_objs = [env.scene.object_registry("name", name) for name in chain_names]
    if any(obj is None for obj in chain_objs):
        return {
            "type": "restack_chain",
            "applied": False,
            "reason": "missing_chain_object",
            "chain": chain_names,
        }

    try:
        support_top_z = float(support_obj.aabb[1][2])
    except Exception:
        support_top_z = float(instruction.get("support_top_z", 0.0) or 0.0)
    z_offset = float(instruction.get("z_offset", 0.002) or 0.002)

    first_obj = chain_objs[0]
    base_xy = instruction.get("base_xy")
    if isinstance(base_xy, list) and len(base_xy) >= 2:
        first_xy = (float(base_xy[0]), float(base_xy[1]))
    else:
        support_bounds_xy = instruction.get("surface_bounds_xy")
        if isinstance(support_bounds_xy, list) and len(support_bounds_xy) == 2:
            first_xy = (
                0.5 * (float(support_bounds_xy[0][0]) + float(support_bounds_xy[1][0])),
                0.5 * (float(support_bounds_xy[0][1]) + float(support_bounds_xy[1][1])),
            )
        else:
            first_pos = first_obj.get_position_orientation()[0]
            first_xy = (float(first_pos[0]), float(first_pos[1]))
    place_upright_on_surface(
        og,
        first_obj,
        surface_xy=first_xy,
        surface_z=support_top_z,
        z_offset=z_offset,
    )

    for lower_obj, upper_obj in zip(chain_objs[:-1], chain_objs[1:]):
        lower_pos = lower_obj.get_position_orientation()[0]
        lower_top_z = float(lower_obj.aabb[1][2])
        place_upright_on_surface(
            og,
            upper_obj,
            surface_xy=(float(lower_pos[0]), float(lower_pos[1])),
            surface_z=lower_top_z,
            z_offset=z_offset,
        )
    stabilize_active_objects(
        og,
        {obj.name: obj for obj in chain_objs},
        steps=12,
        support_obj=support_obj,
    )
    if video_recorder is not None:
        video_recorder.record(env, og)
    return {
        "type": "restack_chain",
        "applied": True,
        "chain": chain_names,
    }


def _refill_liquid_target(env, og, instruction: dict[str, Any], *, video_recorder=None) -> dict[str, Any]:
    target_name = str(instruction.get("target_name", "") or "")
    system_name = str(instruction.get("system_name", "water") or "water")
    target_obj = env.scene.object_registry("name", target_name) if target_name else None
    if target_obj is None:
        return {
            "type": "liquid_refill_target",
            "applied": False,
            "reason": "missing_target",
            "target_name": target_name,
        }

    from omnigibson.object_states import ContainedParticles, Filled

    particle_count = None
    filled_supported = Filled in target_obj.states
    system = env.scene.get_system(system_name, force_init=True)
    if filled_supported:
        target_obj.states[Filled].set_value(system, True)
        for _ in range(10):
            og.sim.step()
            if video_recorder is not None:
                video_recorder.record(env, og)
        try:
            particle_count = int(target_obj.states[ContainedParticles].get_value(system).n_in_volume)
        except Exception:
            particle_count = None
    return {
        "type": "liquid_refill_target",
        "applied": bool(filled_supported),
        "target_name": target_name,
        "system_name": system_name,
        "filled_supported": bool(filled_supported),
        "particle_count": particle_count,
    }


def _place_lid_on_container(env, og, instruction: dict[str, Any], *, video_recorder=None) -> dict[str, Any]:
    from omnigibson.task_generation.pipeline_common import place_upright_on_surface

    lid_name = str(instruction.get("lid_name", "") or "")
    container_name = str(instruction.get("container_name", "") or "")
    lid_obj = env.scene.object_registry("name", lid_name) if lid_name else None
    container_obj = env.scene.object_registry("name", container_name) if container_name else None
    if lid_obj is None or container_obj is None:
        return {
            "type": "place_lid_on_container",
            "applied": False,
            "reason": "missing_lid_or_container",
            "lid_name": lid_name,
            "container_name": container_name,
        }

    container_pos = container_obj.get_position_orientation()[0]
    container_top_z = float(container_obj.aabb[1][2])
    place_upright_on_surface(
        og,
        lid_obj,
        surface_xy=(float(container_pos[0]), float(container_pos[1])),
        surface_z=container_top_z,
        z_offset=0.001,
    )
    og.sim.step()
    if video_recorder is not None:
        video_recorder.record(env, og)
    return {
        "type": "place_lid_on_container",
        "applied": True,
        "lid_name": lid_name,
        "container_name": container_name,
    }


def _restage_on_support(env, og, instruction: dict[str, Any], *, video_recorder=None) -> dict[str, Any]:
    from omnigibson.task_generation.pipeline_common import place_upright_on_surface

    support_name = str(instruction.get("support_name", "") or "")
    object_names = [str(name) for name in instruction.get("object_names", []) if str(name)]
    support_obj = env.scene.object_registry("name", support_name) if support_name else None
    if support_obj is None or not object_names:
        return {
            "type": "restage_on_support",
            "applied": False,
            "reason": "missing_support_or_objects",
            "support_name": support_name,
            "object_names": object_names,
        }

    try:
        support_top_z = float(support_obj.aabb[1][2])
    except Exception:
        return {
            "type": "restage_on_support",
            "applied": False,
            "reason": "missing_support_aabb",
            "support_name": support_name,
            "object_names": object_names,
        }

    placed: list[str] = []
    for object_name in object_names:
        obj = env.scene.object_registry("name", object_name)
        if obj is None:
            continue
        obj_pos = obj.get_position_orientation()[0]
        place_upright_on_surface(
            og,
            obj,
            surface_xy=(float(obj_pos[0]), float(obj_pos[1])),
            surface_z=support_top_z,
            z_offset=0.002,
        )
        placed.append(object_name)

    for _ in range(8):
        og.sim.step()
        if video_recorder is not None:
            video_recorder.record(env, og)
    return {
        "type": "restage_on_support",
        "applied": True,
        "support_name": support_name,
        "object_names": placed,
    }


def apply_runtime_perturbations(
    env,
    *,
    scene_root: str | Path | None = None,
    og=None,
    video_recorder=None,
) -> dict[str, Any]:
    spec = get_perturbation_spec(env.scene)
    if spec is None:
        return {"applied": False, "reason": "no_perturbation_spec"}

    if og is None:
        import omnigibson as og_mod

        og = og_mod

    results = {
        "applied": True,
        "variant_id": spec.get("variant_id"),
        "variant_type": spec.get("variant_type"),
        "visual_overrides": [],
        "local_reconstruct": [],
    }

    for override in spec.get("visual_overrides", []) or []:
        if not isinstance(override, dict):
            continue
        results["visual_overrides"].append(
            _apply_visual_override(env.scene, override, scene_root=scene_root)
        )

    instructions = spec.get("local_reconstruct")
    if isinstance(instructions, dict):
        instructions = [instructions]
    elif not isinstance(instructions, list):
        instructions = []
    for instruction in instructions:
        if not isinstance(instruction, dict):
            continue
        reconstruct_type = str(instruction.get("type", "") or "")
        if reconstruct_type == "liquid_refill_target":
            results["local_reconstruct"].append(
                _refill_liquid_target(env, og, instruction, video_recorder=video_recorder)
            )
        elif reconstruct_type == "restack_chain":
            results["local_reconstruct"].append(
                _restack_chain(env, og, instruction, video_recorder=video_recorder)
            )
        elif reconstruct_type == "place_lid_on_container":
            results["local_reconstruct"].append(
                _place_lid_on_container(env, og, instruction, video_recorder=video_recorder)
            )
        elif reconstruct_type == "restage_on_support":
            results["local_reconstruct"].append(
                _restage_on_support(env, og, instruction, video_recorder=video_recorder)
            )
        else:
            results["local_reconstruct"].append(
                {"type": reconstruct_type, "applied": False, "reason": "unknown_reconstruct_type"}
            )

    return json.loads(json.dumps(results))
