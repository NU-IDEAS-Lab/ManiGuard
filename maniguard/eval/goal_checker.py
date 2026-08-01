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

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from maniguard.utils.goal_region import GoalRegionSpec, object_intersects_goal_region, robot_holds_target

log = logging.getLogger(__name__)


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


@dataclass
class GoalRegionChecker:
    raw_region: GoalRegionSpec
    # lid_transport only (set by build_goal_checker when the diag carries ``lid_info``):
    # wide containers are legitimately carried by their WELDED lid (rim+lid sandwich or
    # lid-top grip) — grasping the lid of the assembly counts as holding the target.
    # None for every other family => behavior byte-identical.
    assembly_lid_name: str | None = None
    _objects: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, env) -> None:
        scene = env.scene
        robot = env.robots[0] if env.robots else None
        if robot is not None:
            self._objects["robot"] = robot
        target = scene.object_registry("name", self.raw_region.target_name)
        if target is not None:
            self._objects[self.raw_region.target_name] = target
        marker = scene.object_registry("name", self.raw_region.marker_name)
        if marker is not None:
            self._objects[self.raw_region.marker_name] = marker

    def check(self, env) -> tuple[bool, dict]:
        if not self._objects:
            self.resolve(env)
        target = self._objects.get(self.raw_region.target_name)
        if target is None:
            return False, {
                "mode": self.raw_region.mode,
                "target_resolved": False,
                "held": False,
                "intersects": False,
            }
        held = robot_holds_target(env, target)
        held_via = "target" if held else None
        if not held and self.assembly_lid_name:
            lid = env.scene.object_registry("name", self.assembly_lid_name)
            if lid is not None and robot_holds_target(env, lid):
                try:
                    from omnigibson.object_states import AttachedTo
                    if bool(lid.states[AttachedTo].get_value(target)):
                        held, held_via = True, "lid_assembly"
                except (KeyError, ImportError):
                    pass
        intersects = object_intersects_goal_region(target, self.raw_region)
        detail = {
            "mode": self.raw_region.mode,
            "target_name": self.raw_region.target_name,
            "marker_name": self.raw_region.marker_name,
            "held": bool(held),
            "held_via": held_via,
            "intersects": bool(intersects),
            "radius_m": float(self.raw_region.radius_m),
            "center_world": list(self.raw_region.center_world),
        }
        return bool(held and intersects), detail


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
    system_name = node.get("system", "")
    if system_name:
        label = f"{predicate}({subject_name}, system={system_name})"
    elif reference_name:
        label = f"{predicate}({subject_name}, {reference_name})"
    else:
        label = f"{predicate}({subject_name})"

    subj = objects.get(subject_name)
    if subj is None:
        return False, {label: False}
    # Unary predicates (e.g. open / closed) have no reference.
    if reference_name:
        ref = objects.get(reference_name)
        if ref is None:
            return False, {label: False}
    else:
        ref = None

    # ``joint_open_at_least`` is a LOWER BOUND on one articulated joint's position, used where
    # "open" is a matter of degree rather than a boolean: OmniGibson's ``Open`` state flips at its
    # own internal threshold, which can be far short of the opening a demonstration actually
    # achieves. The threshold here is calibrated from the demonstrations themselves (see
    # configs/firsthalf/), so the policy is asked to reproduce what it was shown.
    #
    # A lower bound, deliberately, NOT a band: the drawer has a hard stop, and a policy that pulls
    # it FURTHER than the demonstrations did has done better, not worse.
    if predicate == "joint_open_at_least":
        joint_name = node.get("joint", "")
        threshold = node.get("min_position")
        if not joint_name or threshold is None:
            return False, {f"{label}[malformed]": False}
        # The measured position goes in the KEY, never the value: ``_eval_node``'s and/or reduce
        # over ``detail.values()``, so a non-bool there would silently corrupt the conjunction.
        jlabel = f"{predicate}({subject_name}, {joint_name}>={float(threshold):.4f})"
        try:
            # Same index resolution the cabinet datagen uses: joints is an ordered mapping and
            # get_joint_positions() returns values in that order.
            jidx = list(subj.joints.keys()).index(joint_name)
            position = float(subj.get_joint_positions()[jidx])
        except Exception:
            return False, {f"{jlabel}[unreadable]": False}
        result = position >= float(threshold)
        return result, {f"{jlabel}[measured={position:.4f}]": result}

    # ``covered`` takes a particle-system name rather than an object
    # reference; resolve it via env.scene and pass through.
    if predicate == "covered":
        if not system_name:
            return False, {label: False}
        try:
            system = env.scene.get_system(system_name, force_init=False)
        except Exception:
            return False, {label: False}
        from omnigibson.object_states import Covered
        try:
            result = bool(subj.states[Covered].get_value(system))
        except Exception:
            result = False
        return result, {label: result}

    result = _eval_predicate(predicate, subj, ref)
    return result, {label: result}


def _eval_predicate(predicate: str, subject, reference) -> bool:
    """Evaluate a single predicate using OmniGibson object states.

    Raises if the state API crashes — callers should see real failures
    instead of silently counting them as ``False`` (which would make a
    broken goal checker look like an unachieved goal).
    """
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
        contacts = robot.contact_list()
        for contact in contacts:
            if target.name in str(contact):
                return True
        return False
    elif predicate == "open":
        # Unary on articulated objects (cabinets, drawers, doors).
        from omnigibson.object_states import Open
        return bool(subject.states[Open].get_value())
    elif predicate == "closed":
        from omnigibson.object_states import Open
        return not bool(subject.states[Open].get_value())
    else:
        raise ValueError(f"[GoalChecker] Unknown predicate: {predicate!r}")


def build_goal_checker(scene_info: dict) -> Optional[GoalChecker | GoalRegionChecker]:
    """Build a success checker from scene_info/dataset-level goal fields."""
    goal_region = scene_info.get("goal_region")
    if isinstance(goal_region, dict) and goal_region:
        assembly_lid = None
        if scene_info.get("lid_info"):
            # lid family: the lid instance is the ontop-subject of the goal conditions
            for g in scene_info.get("goal_conditions") or []:
                if isinstance(g, dict) and g.get("predicate") == "ontop":
                    assembly_lid = g.get("subject")
                    break
        return GoalRegionChecker(raw_region=GoalRegionSpec.from_json(goal_region),
                                 assembly_lid_name=assembly_lid)
    conditions = scene_info.get("goal_conditions")
    if not conditions:
        return None
    return GoalChecker(raw_conditions=conditions)


