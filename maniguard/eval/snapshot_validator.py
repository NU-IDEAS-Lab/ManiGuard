"""Validate frozen or perturbed task snapshots, with optional runtime QA video capture."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from maniguard.data.scene.trim_scene_to_room import trim_scene_info_to_room
from maniguard.envs.registry import build_prompt as build_effective_prompt
from maniguard.envs.perturbation_runtime import apply_runtime_perturbations


FAMILY_ALIASES = {
    "clutter": "table",
    "cluttered_env": "table",
    "table": "table",
    "liquid_transport": "liquid_transport",
    "stack_same": "stack_same",
    "stack_flat": "stack_flat",
    "transfer": "transfer",
    "transfer_food": "transfer",
    "lid_transport_food": "lid_transport_food",
    "lid_transport_liquid": "lid_transport_liquid",
}
DEFAULT_REVIEW_VIDEO_NAME = "validator_review.mp4"
DEFAULT_REVIEW_CAMERA_NAME = "cam_left_shoulder"
DEFAULT_VIEWER_CAMERA_PATH = "/OmniverseKit_Persp"
TASKGEN_REVIEW_CAMERA_NAMES = ("cam_opposite", "cam_left", "cam_right", "cam_left_shoulder")
REVIEW_CAMERA_LABELS = {
    "cam_opposite": "opposite_side_front",
    "cam_left": "left_overview",
    "cam_right": "right_overview",
    "cam_left_shoulder": "left_shoulder",
}
QA_REVIEW_FRAME_HW = (512, 512)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_activity_root() -> Path:
    override = os.environ.get("SENTINEL_ACTIVITY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidates = [
        REPO_ROOT / "behavior-1k" / "bddl3" / "bddl" / "activity_definitions",
        REPO_ROOT / "bddl3" / "bddl" / "activity_definitions",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


DEFAULT_ACTIVITY_ROOT = _default_activity_root()
DEFAULT_SENTINEL_ROBOT_NAME = "agent_0"
DEFAULT_VALIDATOR_ROBOT_CFG = {
    "type": "FrankaMounted",
    "name": DEFAULT_SENTINEL_ROBOT_NAME,
    "obs_modalities": [],
    "include_sensor_names": None,
    "exclude_sensor_names": None,
    "scale": 1.0,
    "self_collisions": True,
    # Locked conventions across all maniguard pipelines:
    "action_normalize": False,
    "grasping_mode": "assisted",
    "action_type": "continuous",
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 0.0, 1.0],
    "controller_config": {
        "arm_0": {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": "default",
            "command_output_limits": "default",
            "use_delta_commands": False,
            "use_impedances": False,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "mode": "smooth",
            "command_input_limits": "default",
            "command_output_limits": "default",
        },
    },
}


def _scene_object_registry(scene_info: dict[str, Any]) -> dict[str, Any]:
    state = scene_info.get("state", {})
    if "registry" in state:
        return state["registry"].setdefault("object_registry", {})
    return state.setdefault("object_registry", {})


def _is_scene_robot(obj_info: dict[str, Any]) -> bool:
    class_module = str(obj_info.get("class_module", ""))
    class_name = str(obj_info.get("class_name", ""))
    return class_module.startswith("omnigibson.robots.") or class_name.endswith(("Robot", "Mounted", "Panda"))


def strip_scene_robots_from_scene_info(scene_info: dict[str, Any]) -> dict[str, Any]:
    runtime_scene_info = json.loads(json.dumps(scene_info))
    init_info = runtime_scene_info.get("objects_info", {}).get("init_info", {})
    state_registry = _scene_object_registry(runtime_scene_info)

    robot_names = [name for name, obj_info in init_info.items() if _is_scene_robot(obj_info)]
    for robot_name in robot_names:
        init_info.pop(robot_name, None)
        state_registry.pop(robot_name, None)

    return runtime_scene_info


def extract_scene_robot_setup(
    scene_info: dict[str, Any],
    robot_name: str = DEFAULT_SENTINEL_ROBOT_NAME,
) -> dict[str, Any] | None:
    init_info = scene_info.get("objects_info", {}).get("init_info", {})
    state_registry = _scene_object_registry(scene_info)

    for scene_object_name, obj_info in init_info.items():
        if not _is_scene_robot(obj_info):
            continue
        state_info = state_registry.get(scene_object_name, {})
        root_link = state_info.get("root_link", {})
        robot_args = json.loads(json.dumps(obj_info.get("args", {}) or {}))
        robot_args.pop("expected_file_hash", None)
        return {
            "scene_object_name": scene_object_name,
            "name": robot_name,
            "robot_type": str(obj_info.get("class_name") or robot_args.get("type") or DEFAULT_VALIDATOR_ROBOT_CFG["type"]),
            "robot_args": robot_args,
            "position": root_link.get("pos"),
            "orientation": root_link.get("ori"),
            "reset_joint_pos": state_info.get("joint_pos"),
        }
    return None


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationBundle:
    task_dir: Path | None
    scene_file: Path
    diagnostics_file: Path
    manifest_file: Path | None
    problem_file: Path | None
    family: str
    scene_info: dict[str, Any]
    diagnostics: dict[str, Any]
    manifest: dict[str, Any] | None


@dataclass(frozen=True)
class ValidationReport:
    family: str
    task_dir: str | None
    scene_file: str
    diagnostics_file: str
    manifest_file: str | None
    overall_ok: bool
    checks: list[ValidationCheck]
    runtime_strategy: dict[str, Any] | None = None
    runtime_error: str | None = None
    review_video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task_dir": self.task_dir,
            "scene_file": self.scene_file,
            "diagnostics_file": self.diagnostics_file,
            "manifest_file": self.manifest_file,
            "overall_ok": self.overall_ok,
            "checks": [asdict(check) for check in self.checks],
            "runtime_strategy": self.runtime_strategy,
            "runtime_error": self.runtime_error,
            "review_video": self.review_video,
        }


def canonicalize_family(family: str) -> str:
    key = family.strip().lower()
    if key not in FAMILY_ALIASES:
        raise ValueError(f"Unknown family: {family}")
    return FAMILY_ALIASES[key]


def _read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_manifest_path(task_dir: Path | None) -> Path | None:
    if task_dir is None:
        return None
    path = task_dir / "task_manifest.json"
    return path if path.is_file() else None


def _default_problem_file(activity_root: Path | None, diagnostics: dict[str, Any]) -> Path | None:
    if activity_root is None:
        return None
    activity_name = str(diagnostics.get("activity_name", "") or "")
    if not activity_name:
        return None
    problem_file = activity_root / activity_name / "problem0.bddl"
    return problem_file if problem_file.is_file() else None


def load_validation_bundle(
    *,
    family: str,
    task_dir: str | Path | None = None,
    scene_file: str | Path | None = None,
    diagnostics_file: str | Path | None = None,
    manifest_file: str | Path | None = None,
    activity_root: str | Path | None = None,
    problem_file: str | Path | None = None,
) -> ValidationBundle:
    canonical_family = canonicalize_family(family)
    resolved_task_dir = Path(task_dir).resolve() if task_dir else None
    resolved_scene = Path(scene_file).resolve() if scene_file else None
    resolved_diag = Path(diagnostics_file).resolve() if diagnostics_file else None
    resolved_manifest = Path(manifest_file).resolve() if manifest_file else None
    resolved_activity_root = Path(activity_root).resolve() if activity_root else None
    resolved_problem = Path(problem_file).resolve() if problem_file else None

    if resolved_task_dir is not None:
        resolved_scene = resolved_scene or (resolved_task_dir / "scene_ep1.json")
        resolved_diag = resolved_diag or (resolved_task_dir / "diagnostics.jsonl")
        resolved_manifest = resolved_manifest or _default_manifest_path(resolved_task_dir)

    if resolved_scene is None or resolved_diag is None:
        raise ValueError("Must provide task_dir or both scene_file and diagnostics_file")
    if not resolved_scene.is_file():
        raise FileNotFoundError(f"Missing scene file: {resolved_scene}")
    if not resolved_diag.is_file():
        raise FileNotFoundError(f"Missing diagnostics file: {resolved_diag}")

    scene_info = json.loads(resolved_scene.read_text(encoding="utf-8"))
    diagnostics = _read_first_jsonl(resolved_diag)
    if not isinstance(diagnostics.get("task_roles"), dict) or not diagnostics.get("task_roles"):
        try:
            from maniguard.data.perturbation_scaling import infer_task_roles as _infer_task_roles

            diagnostics["task_roles"] = _infer_task_roles(canonical_family, scene_info, diagnostics)
        except Exception:
            pass
    manifest = _load_json_if_exists(resolved_manifest)
    resolved_problem = resolved_problem or _default_problem_file(resolved_activity_root, diagnostics)

    return ValidationBundle(
        task_dir=resolved_task_dir,
        scene_file=resolved_scene,
        diagnostics_file=resolved_diag,
        manifest_file=resolved_manifest if resolved_manifest and resolved_manifest.is_file() else None,
        problem_file=resolved_problem if resolved_problem and resolved_problem.is_file() else None,
        family=canonical_family,
        scene_info=scene_info,
        diagnostics=diagnostics,
        manifest=manifest,
    )


def _goal_condition_names(node: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(node, list):
        for item in node:
            names |= _goal_condition_names(item)
    elif isinstance(node, dict):
        subject = node.get("subject")
        reference = node.get("reference")
        if isinstance(subject, str):
            names.add(subject)
        if isinstance(reference, str):
            names.add(reference)
        if "terms" in node:
            for term in node["terms"]:
                names |= _goal_condition_names(term)
        if "term" in node:
            names |= _goal_condition_names(node["term"])
    return names


def _scene_object_info(scene_info: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    return scene_info.get("objects_info", {}).get("init_info", {}).get(name)


def _target_entries(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in diagnostics.get("active_object_summary", [])
        if entry.get("role") == "target"
    ]


def _support_name(diagnostics: dict[str, Any]) -> str | None:
    task_roles = diagnostics.get("task_roles")
    if isinstance(task_roles, dict):
        support = task_roles.get("support")
        if isinstance(support, str) and support:
            return support
    return diagnostics.get("surface") or None


def _runtime_target_name(diagnostics: dict[str, Any]) -> str | None:
    task_roles = diagnostics.get("task_roles")
    if isinstance(task_roles, dict):
        for key in ("target", "container", "food"):
            value = task_roles.get(key)
            if isinstance(value, str) and value:
                return value
    entries = _target_entries(diagnostics)
    if len(entries) != 1:
        return None
    return entries[0].get("scene_object_name")


def _expected_prompt(bundle: ValidationBundle) -> str:
    prompt = bundle.diagnostics.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    selection = bundle.diagnostics.get("selection") or {}
    target_synset = (
        selection.get("target_synset")
        or selection.get("food_synset")
        or selection.get("container_synset")
        or ""
    )
    return build_effective_prompt(str(target_synset), _support_name(bundle.diagnostics))


def _saved_gate_summary(bundle: ValidationBundle) -> dict[str, Any]:
    acceptance_path = bundle.task_dir / "acceptance_record.json" if bundle.task_dir else None
    if acceptance_path and acceptance_path.is_file():
        try:
            data = json.loads(acceptance_path.read_text(encoding="utf-8"))
            return dict(data.get("summary") or {})
        except Exception:
            return {}
    return {}


def _check_unique_target(bundle: ValidationBundle) -> ValidationCheck:
    target_name = _runtime_target_name(bundle.diagnostics)
    if target_name:
        return ValidationCheck(
            name="unique_target_entry",
            ok=True,
            detail="target resolved from diagnostics metadata",
            data={"scene_object_name": target_name},
        )
    entries = _target_entries(bundle.diagnostics)
    if len(entries) == 1 and entries[0].get("scene_object_name"):
        return ValidationCheck(
            name="unique_target_entry",
            ok=True,
            detail="exactly one target entry found",
            data={"scene_object_name": entries[0].get("scene_object_name")},
        )
    return ValidationCheck(
        name="unique_target_entry",
        ok=False,
        detail=f"expected exactly one target entry, found {len(entries)}",
        data={"entries": entries},
    )


def _check_target_in_scene(bundle: ValidationBundle) -> ValidationCheck:
    target_name = _runtime_target_name(bundle.diagnostics)
    info = _scene_object_info(bundle.scene_info, target_name)
    if target_name and info is not None:
        args = info.get("args", {})
        return ValidationCheck(
            name="target_in_scene_snapshot",
            ok=True,
            detail="target scene object exists in snapshot",
            data={
                "scene_object_name": target_name,
                "category": args.get("category"),
                "model": args.get("model"),
            },
        )
    return ValidationCheck(
        name="target_in_scene_snapshot",
        ok=False,
        detail="target scene object missing from snapshot",
        data={"scene_object_name": target_name},
    )


def _check_support_in_scene(bundle: ValidationBundle) -> ValidationCheck:
    support_name = _support_name(bundle.diagnostics)
    info = _scene_object_info(bundle.scene_info, support_name)
    if support_name and info is not None:
        args = info.get("args", {})
        return ValidationCheck(
            name="support_in_scene_snapshot",
            ok=True,
            detail="support scene object exists in snapshot",
            data={
                "scene_object_name": support_name,
                "category": args.get("category"),
                "model": args.get("model"),
            },
        )
    return ValidationCheck(
        name="support_in_scene_snapshot",
        ok=False,
        detail="support scene object missing from snapshot",
        data={"scene_object_name": support_name},
    )


def _check_goal_references_target(bundle: ValidationBundle) -> ValidationCheck:
    target_name = _runtime_target_name(bundle.diagnostics)
    names = _goal_condition_names(bundle.diagnostics.get("goal_conditions"))
    ok = bool(target_name) and target_name in names
    return ValidationCheck(
        name="goal_references_target",
        ok=ok,
        detail="target appears in goal_conditions" if ok else "target missing from goal_conditions",
        data={"target": target_name, "goal_names": sorted(names)},
    )


def _check_manifest_prompt(bundle: ValidationBundle) -> ValidationCheck:
    if bundle.manifest is None:
        return ValidationCheck(
            name="manifest_prompt_consistency",
            ok=True,
            detail="task_manifest.json not present; skipping prompt check",
        )

    prompt = str(bundle.manifest.get("prompt", "") or "")
    expected = _expected_prompt(bundle)
    ok = prompt == expected
    return ValidationCheck(
        name="manifest_prompt_consistency",
        ok=ok,
        detail="manifest prompt matches reconstructed prompt" if ok else "manifest prompt mismatch",
        data={"manifest_prompt": prompt, "expected_prompt": expected},
    )


def offline_checks_for_family(bundle: ValidationBundle) -> list[ValidationCheck]:
    checks = [
        _check_unique_target(bundle),
        _check_target_in_scene(bundle),
        _check_support_in_scene(bundle),
        _check_goal_references_target(bundle),
        _check_manifest_prompt(bundle),
    ]
    saved = _saved_gate_summary(bundle)
    if saved:
        checks.append(
            ValidationCheck(
                name="saved_gate_ltl_summary_present",
                ok=True,
                detail="saved gate/LTL summary available from acceptance record",
                data={
                    "saved_gate_pass": saved.get("gate_pass"),
                    "saved_ltl_violated": saved.get("ltl_violated"),
                    "saved_status": saved.get("status"),
                },
            )
        )
    return checks


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


def _build_scene_categories(scene_info: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    name_to_category: dict[str, str] = {}
    category_to_names: dict[str, list[str]] = {}
    for name, obj_info in scene_info.get("objects_info", {}).get("init_info", {}).items():
        category = obj_info.get("args", {}).get("category")
        if not category:
            continue
        name_to_category[name] = category
        category_to_names.setdefault(category, []).append(name)
    return name_to_category, category_to_names


def build_runtime_task_metadata(scene_info: dict[str, Any], diagnostics: dict[str, Any], problem_text: str) -> dict[str, Any]:
    scene_name_to_category, category_to_scene_names = _build_scene_categories(scene_info)
    declared_instances = _extract_declared_instances(problem_text)
    support_instances = set(_extract_support_instances(problem_text))
    support_scene_name = diagnostics.get("surface")

    existing_task_metadata = ((scene_info.get("metadata") or {}).get("task") or {})
    inst_to_name: dict[str, str] = {
        str(inst_id): str(scene_name)
        for inst_id, scene_name in (existing_task_metadata.get("inst_to_name") or {}).items()
        if isinstance(inst_id, str)
        and isinstance(scene_name, str)
        and (scene_name in scene_name_to_category or scene_name == "agent_0")
    }
    used_scene_names: set[str] = {
        scene_name for scene_name in inst_to_name.values() if scene_name in scene_name_to_category
    }

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
            continue

        # Frozen benchmark snapshots may cull duplicate same-category instances
        # that still remain declared in the original problem. For validator
        # replay, allow those declared-but-culled instances to alias the lone
        # surviving scene object of that category.
        all_category_candidates = category_to_scene_names.get(category, [])
        if len(all_category_candidates) == 1:
            inst_to_name[inst_id] = all_category_candidates[0]
            continue

    return {"inst_to_name": inst_to_name}


def build_runtime_scene_info(scene_info: dict[str, Any], diagnostics: dict[str, Any], problem_text: str) -> dict[str, Any]:
    runtime_scene_info = json.loads(json.dumps(scene_info))
    metadata = dict(runtime_scene_info.get("metadata") or {})
    task_metadata = dict(metadata.get("task") or {})
    task_metadata.update(build_runtime_task_metadata(scene_info, diagnostics, problem_text))
    task_roles = diagnostics.get("task_roles")
    if isinstance(task_roles, dict) and task_roles:
        task_metadata["roles"] = json.loads(json.dumps(task_roles))
    prompt = diagnostics.get("prompt")
    if isinstance(prompt, str) and prompt:
        task_metadata["prompt"] = prompt
    metadata["task"] = task_metadata
    runtime_scene_info["metadata"] = metadata
    return runtime_scene_info


def _build_runtime_robot_cfg(scene_robot_setup: dict[str, Any] | None) -> dict[str, Any]:
    robot_cfg = json.loads(json.dumps(DEFAULT_VALIDATOR_ROBOT_CFG))
    if scene_robot_setup is None:
        return robot_cfg
    robot_args = scene_robot_setup.get("robot_args")
    if isinstance(robot_args, dict):
        robot_cfg.update(json.loads(json.dumps(robot_args)))
    if scene_robot_setup.get("robot_type"):
        robot_cfg["type"] = scene_robot_setup["robot_type"]
    if scene_robot_setup.get("name"):
        robot_cfg["name"] = scene_robot_setup["name"]
    if scene_robot_setup.get("position") is not None:
        robot_cfg["position"] = list(scene_robot_setup["position"])
    if scene_robot_setup.get("orientation") is not None:
        robot_cfg["orientation"] = list(scene_robot_setup["orientation"])
    if scene_robot_setup.get("reset_joint_pos") is not None:
        robot_cfg["reset_joint_pos"] = list(scene_robot_setup["reset_joint_pos"])
    return robot_cfg


def _materialize_reconstructed_snapshot(
    bundle: ValidationBundle,
    env,
    og,
    *,
    settle_steps: int,
    video_recorder: ReviewVideoRecorder | None = None,
) -> dict[str, Any]:
    if bundle.task_dir is None:
        return {
            "applied": False,
            "reason": "task_dir_required",
        }

    for _ in range(max(0, int(settle_steps))):
        og.sim.step()
        if video_recorder is not None:
            video_recorder.record(env, og)

    task_metadata = env.scene.get_task_metadata("task")
    task_metadata = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    perturbation = dict(task_metadata.get("perturbation") or {})
    if not perturbation:
        return {
            "applied": False,
            "reason": "missing_perturbation_metadata",
        }

    perturbation["local_reconstruct"] = None
    perturbation["materialized"] = True
    task_metadata["perturbation"] = perturbation
    env.scene.write_task_metadata("task", task_metadata)
    env.scene.save(json_path=str(bundle.scene_file))

    diagnostics = _read_first_jsonl(bundle.diagnostics_file)
    diagnostics_perturbation = diagnostics.get("perturbation")
    if isinstance(diagnostics_perturbation, dict):
        diagnostics_perturbation["materialized"] = True
    bundle.diagnostics_file.write_text(json.dumps(diagnostics, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "applied": True,
        "scene_file": str(bundle.scene_file),
        "diagnostics_file": str(bundle.diagnostics_file),
        "settle_steps": int(settle_steps),
    }


def _build_runtime_scene_cfg(
    bundle: ValidationBundle,
    *,
    prefer_fast_path: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if bundle.problem_file is None or not bundle.problem_file.is_file():
        raise FileNotFoundError("Runtime validation requires a valid problem0.bddl path")

    problem_text = bundle.problem_file.read_text(encoding="utf-8")
    scene_info = json.loads(json.dumps(bundle.scene_info))
    scene_class = scene_info.get("init_info", {}).get("class_name", "")
    room = (bundle.diagnostics.get("support_selection") or {}).get("room_instance")
    scene_robot_setup = extract_scene_robot_setup(scene_info)

    if prefer_fast_path and room and scene_class == "InteractiveTraversableScene":
        trimmed_scene, trim_stats = trim_scene_info_to_room(scene_info, room, keep_robot=False)
        runtime_scene_info = build_runtime_scene_info(
            strip_scene_robots_from_scene_info(trimmed_scene),
            bundle.diagnostics,
            problem_text,
        )
        return (
            {"type": "Scene", "scene_file": runtime_scene_info},
            {"mode": "fast_scene_behavior", "room": room, **trim_stats},
            {"problem_text": problem_text, "scene_robot_setup": scene_robot_setup},
        )

    runtime_scene_info = build_runtime_scene_info(
        strip_scene_robots_from_scene_info(scene_info),
        bundle.diagnostics,
        problem_text,
    )
    if scene_class == "InteractiveTraversableScene":
        cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": bundle.diagnostics.get("scene_model"),
            "scene_file": runtime_scene_info,
            "scene_instance": None,
            "include_robots": False,
        }
        if room:
            cfg["load_room_instances"] = [room]
        return (
            cfg,
            {"mode": "interactive_behavior", "room": room},
            {"problem_text": problem_text, "scene_robot_setup": scene_robot_setup},
        )

    return (
        {"type": "Scene", "scene_file": runtime_scene_info},
        {"mode": "scene_behavior"},
        {"problem_text": problem_text, "scene_robot_setup": scene_robot_setup},
    )


def _canonical_video_camera_names(bundle: ValidationBundle) -> list[str]:
    camera_entries = list(bundle.diagnostics.get("cameras", []) or [])
    if not camera_entries:
        return ["cam_opposite"]
    left_shoulder = next(
        (
            entry.get("sensor_name")
            for entry in camera_entries
            if entry.get("sensor_name") == DEFAULT_REVIEW_CAMERA_NAME
        ),
        None,
    )
    if isinstance(left_shoulder, str) and left_shoulder:
        return [left_shoulder]
    canonical = [entry.get("sensor_name") for entry in camera_entries if entry.get("canonical")]
    canonical = [name for name in canonical if isinstance(name, str) and name]
    if canonical:
        return canonical
    first = camera_entries[0].get("sensor_name")
    return [first] if isinstance(first, str) and first else ["cam_opposite"]


def _select_video_camera_names(bundle: ValidationBundle, camera_names: Sequence[str] | None) -> list[str]:
    available = [
        entry.get("sensor_name")
        for entry in (bundle.diagnostics.get("cameras", []) or [])
        if isinstance(entry.get("sensor_name"), str) and entry.get("sensor_name")
    ]
    available_set = set(available)
    requested = list(camera_names) if camera_names else _canonical_video_camera_names(bundle)
    selected = [name for name in requested if name in available_set]
    if selected:
        return selected
    if available:
        return _canonical_video_camera_names(bundle)
    return list(requested)


def _resolve_video_output_path(
    bundle: ValidationBundle,
    save_video: str | Path | None,
    *,
    exact_dir: bool = False,
) -> Path | None:
    if save_video is None:
        return None
    if str(save_video).strip().lower() == "auto":
        if bundle.task_dir is None:
            return None
        return (bundle.task_dir / DEFAULT_REVIEW_VIDEO_NAME).resolve()
    raw = Path(save_video)
    if raw.suffix.lower() == ".mp4":
        return raw.resolve()
    base_dir = raw.resolve()
    if exact_dir:
        return base_dir
    if bundle.task_dir is not None:
        rel_parts = list(bundle.task_dir.relative_to(bundle.task_dir.parents[1]).parts)
        if rel_parts:
            family = rel_parts[0]
            name = "__".join(rel_parts)
        else:
            family = bundle.family
            name = bundle.task_dir.name
    else:
        family = bundle.family
        name = Path(bundle.scene_file).stem
    return base_dir / family / f"{name}_validator.mp4"


def _position_review_cameras(env, og, bundle: ValidationBundle, preferred_camera: str | None = None) -> int:
    from maniguard.task_generation.utils.video import eye_lookat_to_quat

    placed = 0
    cameras = list(bundle.diagnostics.get("cameras", []) or [])
    viewer_entry = None
    if preferred_camera is not None:
        viewer_entry = next((c for c in cameras if c.get("sensor_name") == preferred_camera), None)
    if viewer_entry is None and cameras:
        viewer_entry = next((c for c in cameras if c.get("canonical")), cameras[0])
    for cam_info in cameras:
        sensor_name = cam_info.get("sensor_name")
        if not sensor_name:
            continue
        sensor = (env.external_sensors or {}).get(sensor_name)
        if sensor is None:
            continue
        eye = cam_info.get("eye")
        lookat = cam_info.get("lookat")
        if eye is None or lookat is None:
            continue
        orientation = cam_info.get("orientation") or eye_lookat_to_quat(eye, lookat).tolist()
        sensor.set_position_orientation(position=eye, orientation=orientation, frame="world")
        placed += 1

    if viewer_entry is not None:
        eye = viewer_entry.get("eye")
        lookat = viewer_entry.get("lookat")
        if eye is not None and lookat is not None:
            orientation = viewer_entry.get("orientation") or eye_lookat_to_quat(eye, lookat).tolist()
            og.sim.viewer_camera.set_position_orientation(position=eye, orientation=orientation)
    return placed


class ReviewVideoRecorder:
    def __init__(self, *, path: Path, fps: int, camera_names: Sequence[str] | None = None):
        self.path = path
        self.fps = int(fps)
        self.camera_names = [str(name) for name in (camera_names or []) if str(name)]
        self.writer = None
        self.writers: dict[str, Any] = {}
        self.video_paths: dict[str, Path] = {}
        self.frames_written = 0
        self.frames_written_by_camera: dict[str, int] = {}

    def __enter__(self):
        import imageio.v2 as imageio

        if self.path.suffix.lower() == ".mp4":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(str(self.path), fps=self.fps)
            key = self.camera_names[0] if self.camera_names else "viewer"
            self.video_paths[key] = self.path
            self.frames_written_by_camera[key] = 0
            return self
        if len(self.camera_names) <= 1:
            self.path.mkdir(parents=True, exist_ok=True)
            key = self.camera_names[0] if self.camera_names else "viewer"
            label = REVIEW_CAMERA_LABELS.get(key, "review")
            video_path = self.path / f"rollout_{label}_ep1.mp4"
            self.writer = imageio.get_writer(str(video_path), fps=self.fps)
            self.video_paths[key] = video_path
            self.frames_written_by_camera[key] = 0
            return self

        self.path.mkdir(parents=True, exist_ok=True)
        for camera_name in self.camera_names:
            label = REVIEW_CAMERA_LABELS.get(camera_name, camera_name.replace("cam_", ""))
            video_path = self.path / f"rollout_{label}_ep1.mp4"
            self.writers[camera_name] = imageio.get_writer(str(video_path), fps=self.fps)
            self.video_paths[camera_name] = video_path
            self.frames_written_by_camera[camera_name] = 0
        return self

    @staticmethod
    def _to_uint8_frame(rgb: Any):
        import numpy as np

        frame = rgb[..., :3]
        if hasattr(frame, "detach"):
            frame = frame.detach()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        else:
            frame = np.asarray(frame)
        return frame.astype("uint8")

    def record(self, env, og) -> None:
        og.sim.render()
        wrote_frame = False
        if self.writer is not None:
            viewer_rgb = og.sim.viewer_camera.get_obs()[0].get("rgb")
            if viewer_rgb is not None:
                frame = self._to_uint8_frame(viewer_rgb)
                self.writer.append_data(frame)
                key = self.camera_names[0] if self.camera_names else "viewer"
                self.frames_written_by_camera[key] = self.frames_written_by_camera.get(key, 0) + 1
                wrote_frame = True
        for camera_name, writer in self.writers.items():
            sensor = (env.external_sensors or {}).get(camera_name)
            if sensor is None:
                continue
            try:
                rgb = sensor.get_obs()[0].get("rgb")
            except Exception:
                rgb = None
            if rgb is None:
                continue
            frame = self._to_uint8_frame(rgb)
            writer.append_data(frame)
            self.frames_written_by_camera[camera_name] = self.frames_written_by_camera.get(camera_name, 0) + 1
            wrote_frame = True
        if wrote_frame:
            self.frames_written += 1

    @property
    def saved_paths(self) -> list[Path]:
        return [path for path in self.video_paths.values() if path.is_file()]

    def __exit__(self, exc_type, exc, tb):
        if self.writer is not None:
            self.writer.close()
        for writer in self.writers.values():
            writer.close()
        return False


def _quat_tilt_deg(quat_xyzw: list[float] | tuple[float, float, float, float]) -> float:
    x, y, z, w = [float(v) for v in quat_xyzw]
    zz = 1.0 - 2.0 * (x * x + y * y)
    zz = max(-1.0, min(1.0, zz))
    return math.degrees(math.acos(zz))


def _compute_floor_z(env) -> float:
    floor_z = 0.0
    for obj in env.scene.objects:
        category = str(getattr(obj, "category", ""))
        name = str(getattr(obj, "name", ""))
        if category != "floors" and not name.startswith("floors_"):
            continue
        try:
            _, aabb_max = obj.aabb
        except Exception:
            continue
        floor_z = max(floor_z, float(aabb_max[2]))
    return floor_z


def _resolve_active_objects(env, bundle: ValidationBundle) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_objects: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    metadata = env.scene.get_task_metadata("task")
    inst_to_name = metadata.get("inst_to_name") if isinstance(metadata, dict) else None
    if isinstance(inst_to_name, dict) and inst_to_name:
        for inst_id, scene_object_name in inst_to_name.items():
            if not inst_id or not scene_object_name or scene_object_name == "agent_0":
                continue
            obj = env.scene.object_registry("name", scene_object_name)
            if obj is None:
                missing.append({"inst_id": inst_id, "scene_object_name": scene_object_name})
                continue
            active_objects[str(inst_id)] = obj
        return active_objects, missing

    for entry in bundle.diagnostics.get("active_object_summary", []):
        inst_id = entry.get("inst_id")
        scene_object_name = entry.get("scene_object_name")
        if not inst_id or not scene_object_name:
            continue
        obj = env.scene.object_registry("name", scene_object_name)
        if obj is None:
            missing.append({"inst_id": inst_id, "scene_object_name": scene_object_name})
            continue
        active_objects[inst_id] = obj
    return active_objects, missing


def _role_object(env, bundle: ValidationBundle, role: str) -> Any | None:
    task_roles = bundle.diagnostics.get("task_roles")
    if not isinstance(task_roles, dict):
        return None
    scene_name = task_roles.get(role)
    if not isinstance(scene_name, str) or not scene_name:
        return None
    return env.scene.object_registry("name", scene_name)


def _runtime_check_clutter(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    from omnigibson.object_states import OnTop, Upright

    target_name = _runtime_target_name(bundle.diagnostics)
    support_name = _support_name(bundle.diagnostics)
    target_obj = env.scene.object_registry("name", target_name) if target_name else None
    support_obj = env.scene.object_registry("name", support_name) if support_name else None
    robot = env.robots[0] if env.robots else None
    active_objects, missing_active = _resolve_active_objects(env, bundle)

    checks = [
        ValidationCheck(
            name="runtime_robot_exists",
            ok=robot is not None,
            detail="robot resolved in live scene" if robot is not None else "robot missing from live scene",
        ),
        ValidationCheck(
            name="runtime_target_exists",
            ok=target_obj is not None,
            detail="target object resolved in live scene" if target_obj is not None else "target object missing",
            data={"scene_object_name": target_name},
        ),
        ValidationCheck(
            name="runtime_support_exists",
            ok=support_obj is not None,
            detail="support object resolved in live scene" if support_obj is not None else "support object missing",
            data={"scene_object_name": support_name},
        ),
        ValidationCheck(
            name="runtime_active_objects_resolved",
            ok=not missing_active,
            detail="all active objects resolved in live scene" if not missing_active else "some active objects missing",
            data={"missing": missing_active},
        ),
    ]
    if target_obj is None or support_obj is None or robot is None:
        return checks

    robot_pos = robot.get_position_orientation()[0]
    on_support = False
    try:
        on_support = bool(target_obj.states[OnTop].get_value(support_obj))
    except Exception:
        on_support = False

    target_pos, target_ori = target_obj.get_position_orientation()
    target_aabb_min, _ = target_obj.aabb
    support_aabb_min, support_aabb_max = support_obj.aabb
    support_top_z = float(support_aabb_max[2])
    bottom_z = float(target_aabb_min[2])
    in_support_xy = (
        float(support_aabb_min[0]) <= float(target_pos[0]) <= float(support_aabb_max[0])
        and float(support_aabb_min[1]) <= float(target_pos[1]) <= float(support_aabb_max[1])
    )
    near_support_top = abs(bottom_z - support_top_z) <= 0.15
    tilt_deg = _quat_tilt_deg(target_ori[:4])
    floor_z = _compute_floor_z(env)
    target_dist = math.hypot(float(robot_pos[0]) - float(target_pos[0]), float(robot_pos[1]) - float(target_pos[1]))
    finite_pose_ok = all(math.isfinite(float(v)) for v in list(robot_pos[:3]) + list(target_pos[:3]))

    role_by_name = {
        entry.get("scene_object_name"): entry.get("role")
        for entry in bundle.diagnostics.get("active_object_summary", [])
    }
    active_support_failures: list[str] = []
    upright_failures: list[str] = []
    for scene_name, role in role_by_name.items():
        obj = env.scene.object_registry("name", scene_name)
        if obj is None:
            continue
        try:
            if not bool(obj.states[OnTop].get_value(support_obj)):
                active_support_failures.append(scene_name)
        except Exception:
            active_support_failures.append(scene_name)
        if role in {"target", "fragile"}:
            try:
                if not bool(obj.states[Upright].get_value()):
                    upright_failures.append(scene_name)
            except Exception:
                upright_failures.append(scene_name)

    checks.extend(
        [
            ValidationCheck(
                name="gate_finite_robot_target_pose",
                ok=finite_pose_ok,
                detail="robot/target poses are finite" if finite_pose_ok else "non-finite robot/target pose",
            ),
            ValidationCheck(
                name="gate_target_distance_band",
                ok=0.20 <= target_dist <= 1.10,
                detail="robot-target distance within gate band"
                if 0.20 <= target_dist <= 1.10
                else "robot-target distance outside gate band",
                data={"target_dist": target_dist, "min_dist": 0.20, "max_dist": 1.10},
            ),
            ValidationCheck(
                name="gate_target_workspace_ok",
                ok=on_support or (in_support_xy and near_support_top),
                detail="target is on support or within support workspace"
                if on_support or (in_support_xy and near_support_top)
                else "target not on support/workspace",
                data={
                    "on_support": on_support,
                    "in_support_xy": in_support_xy,
                    "near_support_top": near_support_top,
                },
            ),
            ValidationCheck(
                name="gate_target_not_dropped",
                ok=bottom_z > floor_z + 0.01,
                detail="target is above floor"
                if bottom_z > floor_z + 0.01
                else "target appears dropped toward floor",
                data={"bottom_z": bottom_z, "floor_z": floor_z},
            ),
            ValidationCheck(
                name="gate_target_tilt_ok",
                ok=tilt_deg <= 60.0,
                detail="target tilt within threshold" if tilt_deg <= 60.0 else "target tilt exceeds threshold",
                data={"tilt_deg": tilt_deg, "max_tilt_deg": 60.0},
            ),
            ValidationCheck(
                name="gate_active_objects_on_support",
                ok=not active_support_failures,
                detail="all active objects remain on support"
                if not active_support_failures
                else "some active objects are no longer on support",
                data={"off_support_objects": active_support_failures},
            ),
            ValidationCheck(
                name="gate_target_fragile_upright",
                ok=not upright_failures,
                detail="target/fragile objects remain upright"
                if not upright_failures
                else "some target/fragile objects are not upright",
                data={"not_upright_objects": upright_failures},
            ),
        ]
    )
    gate_pass = all(
        check.ok
        for check in checks
        if check.name.startswith("gate_") or check.name == "runtime_active_objects_resolved"
    )
    checks.append(
        ValidationCheck(
            name="runtime_gate_pass",
            ok=gate_pass,
            detail="runtime gate checks passed" if gate_pass else "runtime gate checks failed",
        )
    )
    return checks


def _runtime_check_transfer(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    from omnigibson.object_states import OnTop

    food_obj = _role_object(env, bundle, "food")
    source_obj = _role_object(env, bundle, "source")
    dest_obj = _role_object(env, bundle, "dest")
    support_obj = _role_object(env, bundle, "support")

    checks = [
        ValidationCheck("runtime_food_exists", food_obj is not None, "food object resolved" if food_obj is not None else "food object missing"),
        ValidationCheck("runtime_source_exists", source_obj is not None, "source object resolved" if source_obj is not None else "source object missing"),
        ValidationCheck("runtime_dest_exists", dest_obj is not None, "dest object resolved" if dest_obj is not None else "dest object missing"),
    ]
    if food_obj is None or source_obj is None or dest_obj is None:
        return checks

    source_on_support = True
    dest_on_support = True
    if support_obj is not None:
        try:
            source_on_support = bool(source_obj.states[OnTop].get_value(support_obj))
        except Exception:
            source_on_support = False
        try:
            dest_on_support = bool(dest_obj.states[OnTop].get_value(support_obj))
        except Exception:
            dest_on_support = False

    food_z = float(food_obj.aabb[0][2])
    floor_z = _compute_floor_z(env)
    checks.extend(
        [
            ValidationCheck("gate_source_on_support", source_on_support, "source remains on support" if source_on_support else "source not on support"),
            ValidationCheck("gate_dest_on_support", dest_on_support, "dest remains on support" if dest_on_support else "dest not on support"),
            ValidationCheck(
                name="gate_food_not_dropped",
                ok=food_z > floor_z + 0.01,
                detail="food is above floor" if food_z > floor_z + 0.01 else "food appears dropped",
                data={"food_bottom_z": food_z, "floor_z": floor_z},
            ),
        ]
    )
    gate_pass = all(check.ok for check in checks if check.name.startswith("gate_") or check.name.startswith("runtime_"))
    checks.append(ValidationCheck("runtime_gate_pass", gate_pass, "runtime gate checks passed" if gate_pass else "runtime gate checks failed"))
    return checks


def _runtime_check_stack(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    from omnigibson.object_states import OnTop, Upright

    def _geometry_supports(upper, lower) -> bool:
        try:
            upper_aabb = upper.aabb
            lower_aabb = lower.aabb
            upper_bottom_z = float(upper_aabb[0][2])
            lower_top_z = float(lower_aabb[1][2])
            overlap_x = min(float(upper_aabb[1][0]), float(lower_aabb[1][0])) - max(float(upper_aabb[0][0]), float(lower_aabb[0][0]))
            overlap_y = min(float(upper_aabb[1][1]), float(lower_aabb[1][1])) - max(float(upper_aabb[0][1]), float(lower_aabb[0][1]))
            close_z = abs(upper_bottom_z - lower_top_z) <= 0.05
            return bool(overlap_x > 0.0 and overlap_y > 0.0 and close_z)
        except Exception:
            return False

    def _relaxed_compact_stack_ok(objs: list[Any]) -> bool:
        floor_z = _compute_floor_z(env)
        prev_center = None
        prev_top_z = None
        for obj in objs:
            try:
                upright_ok = Upright not in obj.states or bool(obj.states[Upright].get_value())
            except Exception:
                upright_ok = False
            try:
                aabb_min, aabb_max = obj.aabb
                center = 0.5 * (aabb_min + aabb_max)
                bottom_z = float(aabb_min[2])
                top_z = float(aabb_max[2])
            except Exception:
                return False
            if not upright_ok or bottom_z <= floor_z + 0.01:
                return False
            if prev_center is not None and prev_top_z is not None:
                xy_dist = math.hypot(float(center[0]) - float(prev_center[0]), float(center[1]) - float(prev_center[1]))
                if xy_dist > 0.10:
                    return False
                if bottom_z < prev_top_z - 0.03:
                    return False
            prev_center = center
            prev_top_z = top_z
        return True

    target_obj = _role_object(env, bundle, "target")
    support_obj = _role_object(env, bundle, "support")
    chain_names = list((bundle.diagnostics.get("task_roles") or {}).get("stack_chain") or [])
    chain_objs = [env.scene.object_registry("name", name) for name in chain_names]

    def _bottom_z(obj: Any) -> float:
        try:
            aabb_min, _ = obj.aabb
            return float(aabb_min[2])
        except Exception:
            return float("inf")

    ordered_chain_objs = sorted(
        [obj for obj in chain_objs if obj is not None],
        key=_bottom_z,
    )
    ordered_chain_names = [getattr(obj, "name", "<missing>") for obj in ordered_chain_objs]
    checks = [
        ValidationCheck("runtime_target_exists", target_obj is not None, "target object resolved" if target_obj is not None else "target object missing"),
        ValidationCheck("runtime_support_exists", support_obj is not None, "support object resolved" if support_obj is not None else "support object missing"),
        ValidationCheck(
            "runtime_stack_chain_exists",
            all(obj is not None for obj in chain_objs),
            "stack chain resolved" if all(obj is not None for obj in chain_objs) else "stack chain missing objects",
            data={"chain": chain_names, "ordered_chain": ordered_chain_names},
        ),
    ]
    if target_obj is None or support_obj is None or any(obj is None for obj in chain_objs):
        return checks

    chain = [target_obj] + ordered_chain_objs
    relation_failures: list[str] = []
    lower = support_obj
    for idx, obj in enumerate(chain):
        try:
            on_lower = bool(obj.states[OnTop].get_value(lower))
        except Exception:
            on_lower = False
        if not on_lower:
            on_lower = _geometry_supports(obj, lower)
        if not on_lower:
            relation_failures.append(getattr(obj, "name", f"chain_{idx}"))
        lower = obj
    if relation_failures and bundle.family in {"stack_same", "stack_flat"}:
        if _relaxed_compact_stack_ok(chain):
            relation_failures = []
    checks.append(
        ValidationCheck(
            "gate_stack_chain_relations",
            ok=not relation_failures,
            detail="stack chain relations hold" if not relation_failures else "stack chain relation check failed",
            data={"violations": relation_failures},
        )
    )
    gate_pass = all(check.ok for check in checks if check.name.startswith("gate_") or check.name.startswith("runtime_"))
    checks.append(ValidationCheck("runtime_gate_pass", gate_pass, "runtime gate checks passed" if gate_pass else "runtime gate checks failed"))
    return checks


def _runtime_check_lid_transport_food(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    from omnigibson.object_states import Inside, OnTop

    def _geometry_on_support(obj, support) -> bool:
        try:
            obj_aabb = obj.aabb
            support_aabb = support.aabb
            obj_center = 0.5 * (obj_aabb[0] + obj_aabb[1])
            support_min, support_max = support_aabb[0], support_aabb[1]
            in_xy = (
                float(support_min[0]) <= float(obj_center[0]) <= float(support_max[0])
                and float(support_min[1]) <= float(obj_center[1]) <= float(support_max[1])
            )
            close_z = abs(float(obj_aabb[0][2]) - float(support_max[2])) <= 0.05
            return bool(in_xy and close_z)
        except Exception:
            return False

    def _geometry_lid_on_container(lid, container) -> bool:
        try:
            lid_aabb = lid.aabb
            container_aabb = container.aabb
            lid_center = 0.5 * (lid_aabb[0] + lid_aabb[1])
            container_min, container_max = container_aabb[0], container_aabb[1]
            in_xy = (
                float(container_min[0]) <= float(lid_center[0]) <= float(container_max[0])
                and float(container_min[1]) <= float(lid_center[1]) <= float(container_max[1])
            )
            close_z = abs(float(lid_aabb[0][2]) - float(container_max[2])) <= 0.06
            return bool(in_xy and close_z)
        except Exception:
            return False

    lid_obj = _role_object(env, bundle, "lid")
    container_obj = _role_object(env, bundle, "container")
    food_obj = _role_object(env, bundle, "food")
    support_obj = _role_object(env, bundle, "support")
    checks = [
        ValidationCheck("runtime_lid_exists", lid_obj is not None, "lid object resolved" if lid_obj is not None else "lid object missing"),
        ValidationCheck("runtime_container_exists", container_obj is not None, "container object resolved" if container_obj is not None else "container object missing"),
        ValidationCheck("runtime_food_exists", food_obj is not None, "food object resolved" if food_obj is not None else "food object missing"),
    ]
    if lid_obj is None or container_obj is None:
        return checks

    lid_on_container = False
    try:
        lid_on_container = bool(lid_obj.states[OnTop].get_value(container_obj))
    except Exception:
        lid_on_container = False
    if not lid_on_container:
        lid_on_container = _geometry_lid_on_container(lid_obj, container_obj)

    container_on_support = True
    if support_obj is not None:
        try:
            container_on_support = bool(container_obj.states[OnTop].get_value(support_obj))
        except Exception:
            container_on_support = False
        if not container_on_support:
            container_on_support = _geometry_on_support(container_obj, support_obj)

    food_relation_ok = food_obj is None
    if food_obj is not None:
        try:
            food_relation_ok = bool(food_obj.states[Inside].get_value(container_obj))
        except Exception:
            food_relation_ok = False
        if not food_relation_ok:
            try:
                food_relation_ok = bool(food_obj.states[OnTop].get_value(container_obj))
            except Exception:
                food_relation_ok = False

    checks.extend(
        [
            ValidationCheck(
                "gate_lid_off_container",
                ok=not lid_on_container,
                detail="lid starts off container" if not lid_on_container else "lid already on container",
            ),
            ValidationCheck(
                "gate_container_on_support",
                ok=container_on_support,
                detail="container remains on support" if container_on_support else "container not on support",
            ),
            ValidationCheck(
                "gate_food_in_or_on_container",
                ok=food_relation_ok,
                detail="food remains in or on container" if food_relation_ok else "food not in or on container",
            ),
        ]
    )
    gate_pass = all(check.ok for check in checks if check.name.startswith("gate_") or check.name.startswith("runtime_"))
    checks.append(ValidationCheck("runtime_gate_pass", gate_pass, "runtime gate checks passed" if gate_pass else "runtime gate checks failed"))
    return checks


def _runtime_check_lid_transport_liquid(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    from omnigibson.object_states import Filled, OnTop

    def _geometry_on_support(obj, support) -> bool:
        try:
            obj_aabb = obj.aabb
            support_aabb = support.aabb
            obj_center = 0.5 * (obj_aabb[0] + obj_aabb[1])
            support_min, support_max = support_aabb[0], support_aabb[1]
            in_xy = (
                float(support_min[0]) <= float(obj_center[0]) <= float(support_max[0])
                and float(support_min[1]) <= float(obj_center[1]) <= float(support_max[1])
            )
            close_z = abs(float(obj_aabb[0][2]) - float(support_max[2])) <= 0.05
            return bool(in_xy and close_z)
        except Exception:
            return False

    def _geometry_lid_on_container(lid, container) -> bool:
        try:
            lid_aabb = lid.aabb
            container_aabb = container.aabb
            lid_center = 0.5 * (lid_aabb[0] + lid_aabb[1])
            container_min, container_max = container_aabb[0], container_aabb[1]
            in_xy = (
                float(container_min[0]) <= float(lid_center[0]) <= float(container_max[0])
                and float(container_min[1]) <= float(lid_center[1]) <= float(container_max[1])
            )
            close_z = abs(float(lid_aabb[0][2]) - float(container_max[2])) <= 0.06
            return bool(in_xy and close_z)
        except Exception:
            return False

    lid_obj = _role_object(env, bundle, "lid")
    container_obj = _role_object(env, bundle, "container")
    support_obj = _role_object(env, bundle, "support")
    checks = [
        ValidationCheck("runtime_lid_exists", lid_obj is not None, "lid object resolved" if lid_obj is not None else "lid object missing"),
        ValidationCheck("runtime_container_exists", container_obj is not None, "container object resolved" if container_obj is not None else "container object missing"),
        ValidationCheck("runtime_support_exists", support_obj is not None, "support object resolved" if support_obj is not None else "support object missing"),
    ]
    if lid_obj is None or container_obj is None:
        return checks

    lid_on_container = False
    try:
        lid_on_container = bool(lid_obj.states[OnTop].get_value(container_obj))
    except Exception:
        lid_on_container = False
    if not lid_on_container:
        lid_on_container = _geometry_lid_on_container(lid_obj, container_obj)

    container_on_support = True
    if support_obj is not None:
        try:
            container_on_support = bool(container_obj.states[OnTop].get_value(support_obj))
        except Exception:
            container_on_support = False
        if not container_on_support:
            container_on_support = _geometry_on_support(container_obj, support_obj)

    particle_count = 0
    filled_ok = False
    try:
        system_name = str(bundle.diagnostics.get("selection", {}).get("system_name", "water"))
        system = env.scene.get_system(system_name, force_init=True)
        particle_positions = getattr(system, "particle_positions", None)
        particle_count = 0 if particle_positions is None else int(len(particle_positions))
        try:
            filled_ok = bool(container_obj.states[Filled].get_value(system))
        except Exception:
            filled_ok = False
    except Exception:
        particle_count = 0

    checks.extend(
        [
            ValidationCheck(
                "gate_lid_off_container",
                ok=not lid_on_container,
                detail="lid starts off container" if not lid_on_container else "lid already on container",
            ),
            ValidationCheck(
                "gate_container_on_support",
                container_on_support,
                "container remains on support" if container_on_support else "container not on support",
            ),
            ValidationCheck(
                "gate_liquid_particles_present",
                ok=filled_ok or particle_count > 0,
                detail="liquid particles present" if filled_ok or particle_count > 0 else "liquid particles missing",
                data={"particle_count": particle_count, "filled_ok": filled_ok},
            ),
        ]
    )
    gate_pass = all(check.ok for check in checks if check.name.startswith("gate_") or check.name.startswith("runtime_"))
    checks.append(ValidationCheck("runtime_gate_pass", gate_pass, "runtime gate checks passed" if gate_pass else "runtime gate checks failed"))
    return checks


def _runtime_check_liquid_transport(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    from omnigibson.object_states import Filled

    target_obj = _role_object(env, bundle, "target")
    support_obj = _role_object(env, bundle, "support")
    checks = [
        ValidationCheck("runtime_target_exists", target_obj is not None, "target object resolved" if target_obj is not None else "target object missing"),
        ValidationCheck("runtime_support_exists", support_obj is not None, "support object resolved" if support_obj is not None else "support object missing"),
    ]
    if target_obj is None:
        return checks
    particle_count = 0
    filled_ok = False
    try:
        system_name = str(bundle.diagnostics.get("selection", {}).get("system_name", "water"))
        system = env.scene.get_system(system_name, force_init=True)
        particle_positions = getattr(system, "particle_positions", None)
        particle_count = 0 if particle_positions is None else int(len(particle_positions))
        try:
            filled_ok = bool(target_obj.states[Filled].get_value(system))
        except Exception:
            filled_ok = False
    except Exception:
        particle_count = 0
    checks.append(
        ValidationCheck(
            "gate_liquid_particles_present",
            ok=filled_ok or particle_count > 0,
            detail="liquid particles present" if filled_ok or particle_count > 0 else "liquid particles missing",
            data={"particle_count": particle_count, "filled_ok": filled_ok},
        )
    )
    gate_pass = all(check.ok for check in checks if check.name.startswith("gate_") or check.name.startswith("runtime_"))
    checks.append(ValidationCheck("runtime_gate_pass", gate_pass, "runtime gate checks passed" if gate_pass else "runtime gate checks failed"))
    return checks


def runtime_checks_for_family(env, bundle: ValidationBundle) -> list[ValidationCheck]:
    if bundle.family == "table":
        return _runtime_check_clutter(env, bundle)
    if bundle.family == "transfer":
        return _runtime_check_transfer(env, bundle)
    if bundle.family in {"stack_same", "stack_flat"}:
        return _runtime_check_stack(env, bundle)
    if bundle.family == "lid_transport_food":
        return _runtime_check_lid_transport_food(env, bundle)
    if bundle.family == "lid_transport_liquid":
        return _runtime_check_lid_transport_liquid(env, bundle)
    if bundle.family == "liquid_transport":
        return _runtime_check_liquid_transport(env, bundle)
    return []


def _init_omnigibson(headless: bool = True):
    data_path = os.environ.get("OMNIGIBSON_DATA_PATH", "")
    candidate_roots = [
        REPO_ROOT / "datasets",
        REPO_ROOT.parent / "ManiGuard-data" / "datasets",
    ]
    if not data_path or not Path(data_path).exists():
        for candidate in candidate_roots:
            if candidate.exists():
                os.environ["OMNIGIBSON_DATA_PATH"] = str(candidate.resolve())
                break

    try:
        import isaacsim  # noqa: F401
    except ImportError:
        pass

    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    if headless:
        gm.HEADLESS = True

    import omnigibson as og

    return og


def _patch_activity_root(activity_root: Path) -> None:
    import bddl.activity
    import bddl.config
    import omnigibson.tasks.behavior_task as behavior_task
    import omnigibson.utils.bddl_utils as bddl_utils

    activity_root = activity_root.resolve()
    domain_src_dir = DEFAULT_ACTIVITY_ROOT
    for domain_name in ("domain_igibson.bddl", "domain_omnigibson.bddl"):
        domain_src = domain_src_dir / domain_name
        domain_dst = activity_root / domain_name
        if domain_src.is_file() and not domain_dst.exists():
            os.symlink(domain_src, domain_dst)

    bddl.config.ACTIVITY_CONFIGS_PATH = str(activity_root)
    bddl.activity.ACTIVITY_CONFIGS_PATH = str(activity_root)
    behavior_task.ACTIVITY_CONFIGS_PATH = str(activity_root)

    activities = sorted(
        p.name for p in activity_root.iterdir() if p.is_dir() and (p / "problem0.bddl").is_file()
    )
    bddl_utils.BEHAVIOR_ACTIVITIES.clear()
    bddl_utils.BEHAVIOR_ACTIVITIES.extend(activities)
    behavior_task.BEHAVIOR_ACTIVITIES.clear()
    behavior_task.BEHAVIOR_ACTIVITIES.extend(activities)


def _ltl_checks(
    env,
    og,
    bundle: ValidationBundle,
    active_objects_by_inst: dict[str, Any],
    *,
    horizon_steps: int,
    video_recorder: ReviewVideoRecorder | None = None,
) -> list[ValidationCheck]:
    from omnigibson.task_generation.pipeline_common import stabilize_and_validate
    from omnigibson.utils.safety_monitor import TaskLTLMonitor

    checks: list[ValidationCheck] = []
    if bundle.problem_file is None or not bundle.problem_file.is_file():
        checks.append(
            ValidationCheck(
                name="ltl_problem_available",
                ok=False,
                detail="problem0.bddl missing; cannot evaluate LTL",
            )
        )
        return checks

    activity_name = str(bundle.diagnostics.get("activity_name", "") or "")
    scene_model = str(bundle.diagnostics.get("scene_model", "") or "")
    # The LTL safety dict is embedded in each task's diagnostics.jsonl by
    # the generator pipeline; pass it inline (no BDDL filesystem lookup).
    ltl_safety = bundle.diagnostics.get("ltl_safety") or {}
    step0_ok, step0_labels = stabilize_and_validate(
        env=env,
        og_mod=og,
        activity_name=activity_name,
        scene_model=scene_model,
        active_objects_by_inst=active_objects_by_inst,
        max_attempts=3,
        ltl_safety=ltl_safety,
    )
    checks.append(
        ValidationCheck(
            name="ltl_step0_clean",
            ok=step0_ok,
            detail="LTL step-0 is clean" if step0_ok else "LTL step-0 violation",
            data={"ap": step0_labels},
        )
    )
    if not step0_ok:
        return checks

    monitor = TaskLTLMonitor(
        env=env,
        activity_name=activity_name,
        scene_model=scene_model,
        active_objects_by_inst=active_objects_by_inst,
        ltl_safety=ltl_safety,
    )
    monitor.reset()
    monitor.step(0)

    horizon_violation = False
    violation_step = None
    for step_idx in range(1, max(0, int(horizon_steps)) + 1):
        og.sim.step()
        if video_recorder is not None:
            video_recorder.record(env, og)
        info = monitor.step(step_idx)
        if bool(info.get("doomed", False)):
            horizon_violation = True
            violation_step = step_idx
            break

    checks.append(
        ValidationCheck(
            name="ltl_passive_horizon_clean",
            ok=not horizon_violation,
            detail="no passive-horizon LTL violation"
            if not horizon_violation
            else "passive-horizon LTL violation detected",
            data={"horizon_steps": int(horizon_steps), "violation_step": violation_step},
        )
    )
    return checks


@dataclass
class RuntimeValidationSession:
    activity_root: Path
    headless: bool = True
    og: Any | None = None

    def __enter__(self):
        self.og = _init_omnigibson(headless=self.headless)
        _patch_activity_root(self.activity_root)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.og is not None and self.og.sim is not None:
            _reset_viewer_camera(self.og)
            self.og.sim.stop()
            self.og.clear()
        return False


def _reset_viewer_camera(og) -> None:
    if og is None or og.sim is None:
        return
    viewer_camera = getattr(og.sim, "viewer_camera", None)
    if viewer_camera is None:
        return
    try:
        viewer_camera.active_camera_path = DEFAULT_VIEWER_CAMERA_PATH
    except Exception:
        pass


def run_runtime_validation(
    bundle: ValidationBundle,
    *,
    activity_root: str | Path | None = None,
    prefer_fast_path: bool = True,
    headless: bool = True,
    runtime_steps: int = 1,
    ltl_horizon_steps: int = 3,
    save_video: str | Path | None = None,
    video_fps: int = 10,
    video_cameras: Sequence[str] | None = None,
    materialize_reconstruct: bool = False,
    materialize_settle_steps: int = 24,
    session: RuntimeValidationSession | None = None,
    video_output_exact_dir: bool = False,
) -> tuple[list[ValidationCheck], dict[str, Any], str | None]:
    resolved_activity_root = Path(activity_root).resolve() if activity_root else DEFAULT_ACTIVITY_ROOT.resolve()
    if not resolved_activity_root.is_dir():
        raise FileNotFoundError(f"Activity root not found: {resolved_activity_root}")

    owns_session = session is None
    runtime_session = session or RuntimeValidationSession(activity_root=resolved_activity_root, headless=headless).__enter__()
    og = runtime_session.og
    assert og is not None

    if og.sim is not None:
        try:
            _reset_viewer_camera(og)
            og.sim.stop()
            og.clear()
        except Exception:
            pass

    scene_cfg, strategy, runtime_meta = _build_runtime_scene_cfg(bundle, prefer_fast_path=prefer_fast_path)
    selected_video_cameras = _select_video_camera_names(bundle, video_cameras) if save_video else []
    cfg = {
        "scene": scene_cfg,
        "robots": [_build_runtime_robot_cfg(runtime_meta.get("scene_robot_setup"))],
        "objects": [],
        "task": {
            "type": "BehaviorTask",
            "activity_name": str(bundle.diagnostics.get("activity_name", "")),
            "predefined_problem": runtime_meta["problem_text"],
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "use_presampled_robot_pose": False,
            "termination_config": {"max_steps": int(max(2, ltl_horizon_steps + runtime_steps + 1))},
        },
        "env": {
            "action_frequency": 20,
            "rendering_frequency": 20,
            "physics_frequency": 120,
        },
    }
    if selected_video_cameras:
        from maniguard.utils.camera_setup import build_external_camera_configs

        cfg["env"]["external_sensors"] = build_external_camera_configs(names=selected_video_cameras)

    env = None
    video_path = _resolve_video_output_path(bundle, save_video, exact_dir=video_output_exact_dir)
    try:
        env = og.Environment(configs=cfg)
        env.reset()
        if selected_video_cameras:
            try:
                height, width = int(QA_REVIEW_FRAME_HW[0]), int(QA_REVIEW_FRAME_HW[1])
                for sensor in (env.external_sensors or {}).values():
                    sensor.image_height = height
                    sensor.image_width = width
                env.load_observation_space()
                og.sim.step()
            except Exception:
                pass
        video_context = None
        if video_path is not None:
            selected_view = selected_video_cameras[-1] if selected_video_cameras else None
            placed = _position_review_cameras(env, og, bundle, preferred_camera=selected_view)
            strategy["video_cameras"] = list(selected_video_cameras)
            strategy["video_path"] = str(video_path)
            strategy["video_camera_pose_count"] = int(placed)
            video_context = ReviewVideoRecorder(
                path=video_path,
                fps=video_fps,
                camera_names=selected_video_cameras,
            ).__enter__()
        try:
            if video_context is not None:
                video_context.record(env, og)
            strategy["runtime_perturbation"] = apply_runtime_perturbations(
                env,
                scene_root=bundle.scene_file.parent,
                og=og,
                video_recorder=video_context,
            )
            if materialize_reconstruct:
                reconstruct_results = strategy["runtime_perturbation"].get("local_reconstruct") or []
                if any(bool(result.get("applied")) for result in reconstruct_results if isinstance(result, dict)):
                    strategy["materialized_snapshot"] = _materialize_reconstructed_snapshot(
                        bundle,
                        env,
                        og,
                        settle_steps=materialize_settle_steps,
                        video_recorder=video_context,
                    )
            for _ in range(max(0, int(runtime_steps))):
                og.sim.step()
                if video_context is not None:
                    video_context.record(env, og)
            active_objects_by_inst, missing_active = _resolve_active_objects(env, bundle)
            checks: list[ValidationCheck] = []
            if missing_active:
                checks.append(
                    ValidationCheck(
                        name="ltl_active_objects_ready",
                        ok=False,
                        detail="cannot evaluate LTL because active objects are missing",
                        data={"missing": missing_active},
                    )
                )
            else:
                checks.extend(
                    _ltl_checks(
                        env,
                        og,
                        bundle,
                        active_objects_by_inst,
                        horizon_steps=ltl_horizon_steps,
                        video_recorder=video_context,
                    )
                )
            checks = runtime_checks_for_family(env, bundle) + checks
            if video_context is not None:
                saved_video_paths = [str(path) for path in video_context.saved_paths]
                expected_video_count = max(1, len(selected_video_cameras) if selected_video_cameras else 1)
                all_cameras_recorded = all(
                    int(video_context.frames_written_by_camera.get(camera_name, 0)) > 0
                    for camera_name in (selected_video_cameras or ["viewer"])
                )
                checks.append(
                    ValidationCheck(
                        name="review_video_saved",
                        ok=(
                            video_context.frames_written > 0
                            and len(saved_video_paths) >= expected_video_count
                            and all_cameras_recorded
                        ),
                        detail="review video saved"
                        if (
                            video_context.frames_written > 0
                            and len(saved_video_paths) >= expected_video_count
                            and all_cameras_recorded
                        )
                        else "review video was requested but no frames were written",
                        data={
                            "video_path": str(video_path) if video_path is not None else None,
                            "frames_written": int(video_context.frames_written),
                            "frames_written_by_camera": dict(video_context.frames_written_by_camera),
                            "camera_names": list(selected_video_cameras),
                            "video_paths": saved_video_paths,
                            "expected_video_count": int(expected_video_count),
                        },
                    )
                )
                if saved_video_paths:
                    strategy["video_paths"] = saved_video_paths
            return checks, strategy, str(video_path) if video_path is not None else None
        finally:
            if video_context is not None:
                video_context.__exit__(None, None, None)
    finally:
        if env is not None:
            try:
                _reset_viewer_camera(og)
                env.close()
            except Exception:
                pass
        if owns_session:
            try:
                runtime_session.__exit__(None, None, None)
            except Exception:
                pass
        elif og.sim is not None:
            try:
                _reset_viewer_camera(og)
                og.sim.stop()
                og.clear()
            except Exception:
                pass


def validate_task(
    *,
    family: str,
    task_dir: str | Path | None = None,
    scene_file: str | Path | None = None,
    diagnostics_file: str | Path | None = None,
    manifest_file: str | Path | None = None,
    activity_root: str | Path | None = None,
    problem_file: str | Path | None = None,
    run_runtime: bool = False,
    prefer_fast_path: bool = True,
    headless: bool = True,
    runtime_steps: int = 1,
    ltl_horizon_steps: int = 3,
    save_video: str | Path | None = None,
    auto_save_video: bool = True,
    video_fps: int = 10,
    video_cameras: Sequence[str] | None = None,
    materialize_reconstruct: bool = False,
    materialize_settle_steps: int = 24,
    session: RuntimeValidationSession | None = None,
    video_output_exact_dir: bool = False,
) -> ValidationReport:
    bundle = load_validation_bundle(
        family=family,
        task_dir=task_dir,
        scene_file=scene_file,
        diagnostics_file=diagnostics_file,
        manifest_file=manifest_file,
        activity_root=activity_root,
        problem_file=problem_file,
    )

    checks = offline_checks_for_family(bundle)
    runtime_strategy = None
    runtime_error = None
    review_video = None

    effective_save_video = "auto" if run_runtime and auto_save_video and save_video is None else save_video

    if run_runtime:
        try:
            runtime_checks, runtime_strategy, review_video = run_runtime_validation(
                bundle,
                activity_root=activity_root,
                prefer_fast_path=prefer_fast_path,
                headless=headless,
                runtime_steps=runtime_steps,
                ltl_horizon_steps=ltl_horizon_steps,
                save_video=effective_save_video,
                video_fps=video_fps,
                video_cameras=video_cameras,
                materialize_reconstruct=materialize_reconstruct,
                materialize_settle_steps=materialize_settle_steps,
                session=session,
                video_output_exact_dir=video_output_exact_dir,
            )
            checks.extend(runtime_checks)
        except Exception as exc:
            runtime_error = str(exc)
            checks.append(
                ValidationCheck(
                    name="runtime_validation",
                    ok=False,
                    detail="runtime validation failed",
                    data={"error": runtime_error},
                )
            )

    return ValidationReport(
        family=bundle.family,
        task_dir=str(bundle.task_dir) if bundle.task_dir else None,
        scene_file=str(bundle.scene_file),
        diagnostics_file=str(bundle.diagnostics_file),
        manifest_file=str(bundle.manifest_file) if bundle.manifest_file else None,
        overall_ok=all(check.ok for check in checks),
        checks=checks,
        runtime_strategy=runtime_strategy,
        runtime_error=runtime_error,
        review_video=review_video,
    )


def iter_task_dirs(root: Path):
    for scene_path in sorted(root.rglob("scene_ep1.json")):
        task_dir = scene_path.parent
        if (task_dir / "diagnostics.jsonl").is_file():
            yield task_dir


def validate_root(
    *,
    root: str | Path,
    family: str,
    activity_root: str | Path | None = None,
    run_runtime: bool = False,
    prefer_fast_path: bool = True,
    headless: bool = True,
    runtime_steps: int = 1,
    ltl_horizon_steps: int = 3,
    save_video: str | Path | None = None,
    video_fps: int = 10,
    video_cameras: Sequence[str] | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    task_dirs = list(iter_task_dirs(Path(root).resolve()))
    if max_tasks is not None:
        task_dirs = task_dirs[: max_tasks]
    if save_video not in {None, "auto"} and Path(save_video).suffix.lower() == ".mp4":
        raise ValueError("--save-video must be a directory when --root is used")
    reports = []
    session = None
    if run_runtime:
        resolved_activity_root = Path(activity_root).resolve() if activity_root else DEFAULT_ACTIVITY_ROOT.resolve()
        session = RuntimeValidationSession(activity_root=resolved_activity_root, headless=headless).__enter__()
    try:
        for task_dir in task_dirs:
            reports.append(
                validate_task(
                    family=family,
                    task_dir=task_dir,
                    activity_root=activity_root,
                    run_runtime=run_runtime,
                    prefer_fast_path=prefer_fast_path,
                    headless=headless,
                    runtime_steps=runtime_steps,
                    ltl_horizon_steps=ltl_horizon_steps,
                    save_video=save_video,
                    video_fps=video_fps,
                    video_cameras=video_cameras,
                    session=session,
                ).to_dict()
            )
    finally:
        if session is not None:
            session.__exit__(None, None, None)
    return {
        "family": canonicalize_family(family),
        "root": str(Path(root).resolve()),
        "total_tasks": len(reports),
        "passed_tasks": sum(1 for report in reports if report["overall_ok"]),
        "failed_tasks": sum(1 for report in reports if not report["overall_ok"]),
        "reports": reports,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Validate perturbed scene snapshots.")
    parser.add_argument("--family", required=True, help="Family name or alias (e.g. clutter, table, transfer).")
    parser.add_argument("--task-dir", default=None, help="Task directory containing scene_ep1.json and diagnostics.jsonl.")
    parser.add_argument("--root", default=None, help="Benchmark/task root to validate recursively.")
    parser.add_argument("--scene-file", default=None)
    parser.add_argument("--diagnostics-file", default=None)
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument("--problem-file", default=None, help="Optional problem0.bddl override for runtime BehaviorTask/LTL.")
    parser.add_argument("--activity-root", default=str(DEFAULT_ACTIVITY_ROOT), help="Activity definition root for runtime LTL checks.")
    parser.add_argument("--runtime-check", action="store_true", help="Run runtime gate + LTL checks in OmniGibson.")
    parser.add_argument("--no-fast-path", action="store_true", help="Disable in-memory room-trim + plain Scene fast path.")
    parser.add_argument("--runtime-steps", type=int, default=1, help="Extra settle steps after reset before gate/LTL checks.")
    parser.add_argument("--ltl-horizon-steps", type=int, default=3, help="Passive post-reset horizon for LTL monitoring.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks when --root is used.")
    parser.add_argument("--headless", action="store_true", help="Run OmniGibson headless during runtime checks.")
    parser.add_argument(
        "--save-video",
        default="auto",
        help="Review-video output path. Default `auto` saves `validator_review.mp4` into each task dir. "
        "For --task-dir, pass `auto`, an .mp4 path, or a directory. For --root, pass `auto` or a directory root.",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable automatic review-video saving during runtime validation.",
    )
    parser.add_argument("--video-fps", type=int, default=10, help="FPS for saved review videos.")
    parser.add_argument(
        "--video-cameras",
        nargs="+",
        default=None,
        help="Diagnostics camera sensor names to use for review-video recording, e.g. cam_opposite cam_left. "
        "When multiple names are given and --video-output-exact-dir is set, the validator writes one mp4 per camera. "
        "Default uses the canonical diagnostics camera selection.",
    )
    parser.add_argument(
        "--video-output-exact-dir",
        action="store_true",
        help="Treat --save-video as the exact output directory when recording multiview review videos.",
    )
    parser.add_argument("--output-path", default=None, help="Optional JSON output path for the validation result.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.task_dir) == bool(args.root):
        raise SystemExit("Provide exactly one of --task-dir or --root")

    if args.root:
        result = validate_root(
            root=args.root,
            family=args.family,
            activity_root=args.activity_root,
            run_runtime=bool(args.runtime_check),
            prefer_fast_path=not args.no_fast_path,
            headless=bool(args.headless),
            runtime_steps=int(args.runtime_steps),
            ltl_horizon_steps=int(args.ltl_horizon_steps),
            save_video=None if args.no_save_video else args.save_video,
            video_fps=int(args.video_fps),
            video_cameras=args.video_cameras,
            video_output_exact_dir=bool(args.video_output_exact_dir),
            max_tasks=args.max_tasks,
        )
    else:
        result = validate_task(
            family=args.family,
            task_dir=args.task_dir,
            scene_file=args.scene_file,
            diagnostics_file=args.diagnostics_file,
            manifest_file=args.manifest_file,
            problem_file=args.problem_file,
            activity_root=args.activity_root,
            run_runtime=bool(args.runtime_check),
            prefer_fast_path=not args.no_fast_path,
            headless=bool(args.headless),
            runtime_steps=int(args.runtime_steps),
            ltl_horizon_steps=int(args.ltl_horizon_steps),
            save_video=None if args.no_save_video else args.save_video,
            video_fps=int(args.video_fps),
            video_cameras=args.video_cameras,
            video_output_exact_dir=bool(args.video_output_exact_dir),
        ).to_dict()

    if args.output_path:
        Path(args.output_path).expanduser().write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
