"""Goal checking via OmniGibson object states.

Goal conditions are stored in diagnostics.jsonl by the pipeline at
generation time, using actual scene object names (not BDDL synsets).
Eval reads them and evaluates using OmniGibson's native state API.

Flat format (list of AND'd predicates):
    "goal_conditions": [
        {"predicate": "inside", "subject": "potato_124", "reference": "stockpot_122"}
    ]

Compound format (AND/OR/NOT tree):
    "goal_conditions": {
        "op": "and",
        "terms": [
            {"predicate": "inside", "subject": "potato_124", "reference": "stockpot_122"},
            {"op": "not", "term": {"predicate": "touching", "subject": "robot", "reference": "wineglass_3"}}
        ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class GoalChecker:
    raw_conditions: Union[list, dict]
    _objects: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, env) -> None:
        """Bind object names to live OmniGibson objects."""
        scene = env.scene
        robot = env.robots[0] if env.robots else None
        if robot:
            self._objects["robot"] = robot
        # Collect all object names referenced in conditions
        for name in _collect_names(self.raw_conditions):
            if name == "robot":
                continue
            obj = scene.object_registry("name", name)
            if obj is not None:
                self._objects[name] = obj

    def check(self, env) -> tuple[bool, dict]:
        """Evaluate goal conditions. Returns (success, detail_dict)."""
        if not self._objects:
            self.resolve(env)
        success, detail = _eval_node(self.raw_conditions, self._objects, env)
        return success, detail


def _collect_names(node) -> set:
    """Recursively collect all object names from a condition tree."""
    names = set()
    if isinstance(node, list):
        for item in node:
            names |= _collect_names(item)
    elif isinstance(node, dict):
        if "subject" in node:
            names.add(node["subject"])
        if "reference" in node:
            names.add(node["reference"])
        if "terms" in node:
            for t in node["terms"]:
                names |= _collect_names(t)
        if "term" in node:
            names |= _collect_names(node["term"])
    return names


def _eval_node(node, objects, env) -> tuple[bool, dict]:
    """Evaluate a condition node (flat list or compound tree)."""
    # Flat list = implicit AND
    if isinstance(node, list):
        detail = {}
        for item in node:
            ok, sub = _eval_node(item, objects, env)
            detail.update(sub)
        return all(detail.values()) if detail else False, detail

    if not isinstance(node, dict):
        return False, {"error": f"unexpected node type: {type(node)}"}

    # Compound operator
    op = node.get("op")
    if op == "and":
        detail = {}
        for t in node.get("terms", []):
            ok, sub = _eval_node(t, objects, env)
            detail.update(sub)
        return all(detail.values()) if detail else False, detail
    elif op == "or":
        detail = {}
        for t in node.get("terms", []):
            ok, sub = _eval_node(t, objects, env)
            detail.update(sub)
        return any(detail.values()) if detail else False, detail
    elif op == "not":
        ok, sub = _eval_node(node["term"], objects, env)
        key = list(sub.keys())[0] if sub else "not"
        return not ok, {f"not({key})": not ok}

    # Leaf predicate
    predicate = node.get("predicate", "").lower()
    subject_name = node.get("subject", "")
    reference_name = node.get("reference", "")
    label = f"{predicate}({subject_name}, {reference_name})"

    subj = objects.get(subject_name)
    ref = objects.get(reference_name)
    if subj is None or ref is None:
        return False, {label: False}

    result = _eval_predicate(predicate, subj, ref)
    return result, {label: result}


def _eval_predicate(predicate: str, subject, reference) -> bool:
    """Evaluate a single predicate using OmniGibson object states."""
    try:
        if predicate == "inside":
            from omnigibson.object_states import Inside
            return bool(subject.states[Inside].get_value(reference))
        elif predicate == "ontop":
            from omnigibson.object_states import OnTop
            return bool(subject.states[OnTop].get_value(reference))
        elif predicate == "touching":
            from omnigibson.object_states import Touching
            return bool(subject.states[Touching].get_value(reference))
        elif predicate == "grasping":
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
        else:
            print(f"[GoalChecker] Unknown predicate: {predicate}")
            return False
    except Exception:
        return False


def build_goal_checker(scene_info: dict) -> Optional[GoalChecker]:
    """Build a GoalChecker from scene_info's goal_conditions field.

    Returns None if no goal_conditions are present.
    """
    conditions = scene_info.get("goal_conditions")
    if not conditions:
        return None
    return GoalChecker(raw_conditions=conditions)
