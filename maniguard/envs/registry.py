from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from maniguard.utils.goal_region import build_task_prompt


def _normalize_synset_name(synset: str) -> str:
    base = synset.split(".n.")[0]
    return base.replace("_", " ")


def _normalize_label(name: str) -> str:
    return name.replace("_", " ")


def build_prompt(target_synset: str, support_label: str | None = None) -> str:
    target_name = _normalize_synset_name(target_synset)
    if support_label:
        support_name = _normalize_label(support_label)
        return f"Pick up the {target_name} on the {support_name}."
    return f"Pick up the {target_name}."


DEFAULT_SENTINEL_ROBOT_NAME = "agent_0"


@dataclass(frozen=True)
class ManiGuardSceneSpec:
    scene_name: str
    scene_file: str
    diagnostics_file: str
    activity_name: str
    problem_file: str
    target_synset: str
    target_object_name: str
    support_object_name: str | None
    support_object_label: str | None
    prompt: str


def _scene_object_registry(scene_info: dict) -> dict:
    state = scene_info.get("state", {})
    if "registry" in state:
        return state["registry"].setdefault("object_registry", {})
    return state.setdefault("object_registry", {})


def _is_scene_robot(obj_info: dict) -> bool:
    class_module = str(obj_info.get("class_module", ""))
    class_name = str(obj_info.get("class_name", ""))
    return class_module.startswith("omnigibson.robots.") or class_name.endswith(("Robot", "Mounted", "Panda"))


def extract_scene_robot_setup(
    scene_info: dict,
    robot_name: str = DEFAULT_SENTINEL_ROBOT_NAME,
) -> dict | None:
    init_info = scene_info.get("objects_info", {}).get("init_info", {})
    state_registry = _scene_object_registry(scene_info)

    for scene_object_name, obj_info in init_info.items():
        if not _is_scene_robot(obj_info):
            continue
        state_info = state_registry.get(scene_object_name, {})
        root_link = state_info.get("root_link", {})
        return {
            "scene_object_name": scene_object_name,
            "name": robot_name,
            "position": root_link.get("pos"),
            "orientation": root_link.get("ori"),
            "reset_joint_pos": state_info.get("joint_pos"),
        }
    return None


def strip_scene_robots_from_scene_info(scene_info: dict) -> dict:
    runtime_scene_info = copy.deepcopy(scene_info)
    init_info = runtime_scene_info.get("objects_info", {}).get("init_info", {})
    state_registry = _scene_object_registry(runtime_scene_info)

    robot_names = [name for name, obj_info in init_info.items() if _is_scene_robot(obj_info)]
    for robot_name in robot_names:
        init_info.pop(robot_name, None)
        state_registry.pop(robot_name, None)

    return runtime_scene_info


def _read_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


_INSTANCE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]+_[0-9]+")
_ONTOP_PATTERN = re.compile(r"\(ontop\s+([^\s()]+)\s+([^\s()]+)\)")


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _extract_declared_instances(problem_text: str) -> list[str]:
    return _dedupe(
        token
        for token in _INSTANCE_TOKEN_PATTERN.findall(problem_text)
        if "." in token and token != "-"
    )


def _extract_support_instances(problem_text: str) -> list[str]:
    return _dedupe(match.group(2) for match in _ONTOP_PATTERN.finditer(problem_text))


def _build_scene_categories(scene_info: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    name_to_category: dict[str, str] = {}
    category_to_names: dict[str, list[str]] = {}
    for name, obj_info in scene_info.get("objects_info", {}).get("init_info", {}).items():
        category = obj_info.get("args", {}).get("category")
        if not category:
            continue
        name_to_category[name] = category
        category_to_names.setdefault(category, []).append(name)
    return name_to_category, category_to_names


def build_runtime_task_metadata(scene_info: dict, diagnostics: dict, problem_text: str) -> dict:
    scene_name_to_category, category_to_scene_names = _build_scene_categories(scene_info)
    declared_instances = _extract_declared_instances(problem_text)
    support_instances = set(_extract_support_instances(problem_text))
    support_scene_name = diagnostics.get("surface")

    inst_to_name: dict[str, str] = {}
    used_scene_names: set[str] = set()

    for entry in diagnostics.get("active_object_summary", []):
        inst_id = entry.get("inst_id")
        scene_object_name = entry.get("scene_object_name")
        if not inst_id or not scene_object_name:
            continue
        if scene_object_name not in scene_name_to_category:
            continue
        inst_to_name[inst_id] = scene_object_name
        used_scene_names.add(scene_object_name)

    if support_scene_name in scene_name_to_category:
        for inst_id in support_instances:
            inst_to_name.setdefault(inst_id, support_scene_name)
        used_scene_names.add(support_scene_name)

    for inst_id in declared_instances:
        if inst_id in inst_to_name:
            continue
        if inst_id.startswith("agent.n."):
            inst_to_name[inst_id] = "agent_0"
            continue

        category = inst_id.split(".n.")[0]
        candidates = [
            scene_name
            for scene_name in category_to_scene_names.get(category, [])
            if scene_name not in used_scene_names
        ]
        if not candidates and support_scene_name in scene_name_to_category and inst_id in support_instances:
            candidates = [support_scene_name]
        if len(candidates) == 1:
            inst_to_name[inst_id] = candidates[0]
            used_scene_names.add(candidates[0])

    missing_instances = [
        inst_id for inst_id in declared_instances if inst_id not in inst_to_name and not inst_id.startswith("agent.n.")
    ]
    if missing_instances:
        import logging
        logging.getLogger(__name__).warning(
            f"Could not map {len(missing_instances)} BDDL instances to scene objects: {missing_instances}. "
            f"Proceeding with partial mapping."
        )

    return {"inst_to_name": inst_to_name}


def build_runtime_scene_info(scene_info: dict, diagnostics: dict, problem_text: str) -> dict:
    runtime_scene_info = copy.deepcopy(scene_info)
    metadata = dict(runtime_scene_info.get("metadata") or {})
    task_metadata = dict(metadata.get("task") or {})
    task_metadata.update(build_runtime_task_metadata(scene_info, diagnostics, problem_text))
    metadata["task"] = task_metadata
    runtime_scene_info["metadata"] = metadata
    return runtime_scene_info


def _resolve_scene_names(
    benchmark_root: Path,
    scene_names: Iterable[str] | None = None,
    max_scenes: int | None = None,
) -> list[str]:
    if scene_names:
        names = list(scene_names)
    else:
        names = sorted(p.name for p in benchmark_root.iterdir() if p.is_dir())
    if max_scenes is not None:
        names = names[: max_scenes]
    return names


def build_scene_registry(
    benchmark_root: str | Path,
    activity_root: str | Path,
    scene_names: Iterable[str] | None = None,
    max_scenes: int | None = None,
) -> list[ManiGuardSceneSpec]:
    benchmark_root = Path(benchmark_root)
    activity_root = Path(activity_root)
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {benchmark_root}")
    if not activity_root.is_dir():
        raise FileNotFoundError(f"Activity root not found: {activity_root}")

    registry: list[ManiGuardSceneSpec] = []
    for scene_name in _resolve_scene_names(
        benchmark_root=benchmark_root,
        scene_names=scene_names,
        max_scenes=max_scenes,
    ):
        scene_dir = benchmark_root / scene_name
        scene_file = scene_dir / "scene_ep1.json"
        diagnostics_file = scene_dir / "diagnostics.jsonl"
        if not scene_file.is_file():
            raise FileNotFoundError(f"Missing scene snapshot: {scene_file}")
        if not diagnostics_file.is_file():
            raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_file}")

        diagnostics = _read_first_jsonl(diagnostics_file)
        scene_info = json.loads(scene_file.read_text(encoding="utf-8"))
        activity_name = diagnostics["activity_name"]
        problem_file = activity_root / activity_name / "problem0.bddl"
        if not problem_file.is_file():
            raise FileNotFoundError(f"Missing dataset-local BDDL: {problem_file}")

        target_object_name = ""
        for entry in diagnostics.get("active_object_summary", []):
            if entry.get("role") == "target":
                target_object_name = entry.get("scene_object_name", "")
                break
        if not target_object_name:
            import logging
            logging.getLogger(__name__).warning(
                f"Skipping scene {scene_name}: no target object in {diagnostics_file}"
            )
            continue

        target_synset = diagnostics["selection"]["target_synset"]
        support_object_name = diagnostics.get("surface")
        support_object_label = None
        if support_object_name:
            support_object_label = (
                scene_info.get("objects_info", {})
                .get("init_info", {})
                .get(support_object_name, {})
                .get("args", {})
                .get("category")
            ) or support_object_name
        registry.append(
            ManiGuardSceneSpec(
                scene_name=scene_name,
                scene_file=str(scene_file),
                diagnostics_file=str(diagnostics_file),
                activity_name=activity_name,
                problem_file=str(problem_file),
                target_synset=target_synset,
                target_object_name=target_object_name,
                support_object_name=support_object_name,
                support_object_label=support_object_label,
                prompt=str(diagnostics.get("prompt") or build_task_prompt(scene_info, diagnostics, goal_region=diagnostics.get("goal_region"))),
            )
        )

    if not registry:
        raise ValueError(f"No ManiGuard scenes discovered under {benchmark_root}")
    return registry


def slice_scene_registry_for_worker(
    registry: list[ManiGuardSceneSpec],
    num_envs: int,
    seed_offset: int,
) -> list[ManiGuardSceneSpec]:
    if not registry:
        raise ValueError("registry must not be empty")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    specs: list[ManiGuardSceneSpec] = []
    for env_idx in range(num_envs):
        global_idx = seed_offset * num_envs + env_idx
        specs.append(registry[global_idx % len(registry)])
    return specs
