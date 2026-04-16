"""Lightweight goal checking via OmniGibson object states.

No BDDL parsing needed — goal predicates are inferred from the
pipeline type + diagnostics selection fields. Each pipeline maps to a
set of (predicate, subject, reference) checks evaluated every step.

Supported predicates (from omnigibson.object_states):
    Inside   — e.g. potato Inside stockpot
    OnTop    — e.g. lid OnTop container
    Grasping — robot grasping target (via contact heuristic)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GoalCondition:
    predicate: str            # "Inside" | "OnTop" | "Grasping"
    subject_name: str         # scene object name
    reference_name: str       # scene object name or "robot"
    description: str = ""


@dataclass
class GoalChecker:
    conditions: List[GoalCondition]
    _resolved: bool = False
    _subject_objs: Dict[str, Any] = field(default_factory=dict)
    _reference_objs: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, env) -> None:
        """Bind object names to live OmniGibson objects."""
        scene = env.scene
        robot = env.robots[0] if env.robots else None
        for cond in self.conditions:
            for name in (cond.subject_name, cond.reference_name):
                if name == "robot":
                    continue
                if name not in self._subject_objs:
                    obj = scene.object_registry("name", name)
                    if obj is not None:
                        self._subject_objs[name] = obj
                        self._reference_objs[name] = obj
        if robot is not None:
            self._subject_objs["robot"] = robot
            self._reference_objs["robot"] = robot
        self._resolved = True

    def check(self, env) -> tuple[bool, dict]:
        """Evaluate all conditions. Returns (all_satisfied, detail_dict)."""
        if not self._resolved:
            self.resolve(env)

        results = {}
        for cond in self.conditions:
            subj = self._subject_objs.get(cond.subject_name)
            ref = self._reference_objs.get(cond.reference_name)
            if subj is None or ref is None:
                results[cond.description or f"{cond.predicate}({cond.subject_name},{cond.reference_name})"] = False
                continue

            satisfied = _eval_predicate(cond.predicate, subj, ref, env)
            key = cond.description or f"{cond.predicate}({cond.subject_name},{cond.reference_name})"
            results[key] = satisfied

        all_ok = bool(results) and all(results.values())
        return all_ok, results


def _eval_predicate(predicate: str, subject, reference, env) -> bool:
    """Evaluate a single predicate."""
    try:
        if predicate == "Inside":
            from omnigibson.object_states import Inside
            return bool(subject.states[Inside].get_value(reference))
        elif predicate == "OnTop":
            from omnigibson.object_states import OnTop
            return bool(subject.states[OnTop].get_value(reference))
        elif predicate == "Grasping":
            # Use contact-based grasping check
            robot = reference if hasattr(reference, "contact_list") else subject
            target = subject if robot is reference else reference
            try:
                contacts = robot.contact_list()
                for contact in contacts:
                    if target.name in str(contact):
                        return True
            except Exception:
                pass
            return False
        elif predicate == "Lifted":
            pos = subject.get_position_orientation()[0]
            return float(pos[2]) > 1.0
        else:
            print(f"[GoalChecker] Unknown predicate: {predicate}")
            return False
    except Exception as e:
        return False


def build_goal_checker(scene_info: dict) -> GoalChecker:
    """Build a GoalChecker from scene_info (pipeline + diagnostics).

    Infers goal conditions from the pipeline type and the selection
    fields in diagnostics. No BDDL needed.
    """
    pipeline = scene_info.get("pipeline", "")
    conditions = []

    # Read diagnostics for object names
    diag_path = Path(scene_info["scene_file"]).parent / "diagnostics.jsonl"
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8").strip().split("\n")[0])
    else:
        diag = {}
    sel = diag.get("selection", {})

    # Resolve object names from init_info
    scene_json = json.loads(Path(scene_info["scene_file"]).read_text(encoding="utf-8"))
    init_info = scene_json.get("objects_info", {}).get("init_info", {})

    def _find_obj_by_category(cat: str) -> Optional[str]:
        for name, info in init_info.items():
            if info.get("args", {}).get("category") == cat:
                return name
        return None

    if pipeline == "transfer":
        food_name = scene_info.get("target_name", "")
        dest_cat = sel.get("dest_synset", "").split(".n.")[0]
        dest_name = _find_obj_by_category(dest_cat)
        goal_pred = sel.get("goal_predicate", "inside").capitalize()
        if goal_pred.lower() == "inside":
            goal_pred = "Inside"
        elif goal_pred.lower() == "ontop":
            goal_pred = "OnTop"
        if food_name and dest_name:
            conditions.append(GoalCondition(
                predicate=goal_pred,
                subject_name=food_name,
                reference_name=dest_name,
                description=f"{food_name} {goal_pred} {dest_name}",
            ))

    elif pipeline in ("lid_transport_food", "lid_transport_liquid"):
        container_cat = sel.get("container_category", "")
        container_name = _find_obj_by_category(container_cat)
        lid_name = _find_obj_by_category("lid")
        if lid_name and container_name:
            conditions.append(GoalCondition(
                predicate="OnTop",
                subject_name=lid_name,
                reference_name=container_name,
                description=f"lid OnTop {container_name}",
            ))
            conditions.append(GoalCondition(
                predicate="Grasping",
                subject_name="robot",
                reference_name=container_name,
                description=f"robot Grasping {container_name}",
            ))

    elif pipeline in ("liquid_transport", "clutter", ""):
        target_name = scene_info.get("target_name", "")
        if target_name:
            conditions.append(GoalCondition(
                predicate="Grasping",
                subject_name="robot",
                reference_name=target_name,
                description=f"robot Grasping {target_name}",
            ))

    return GoalChecker(conditions=conditions)
