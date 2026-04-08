"""Parser for extracting and normalizing manipulation task specifications from BDDL.

This module provides functionality to parse BDDL (Behavior Domain Definition Language)
task definitions and extract structured information about manipulation tasks, including:
- Target objects that need to be manipulated
- Fragile objects that require careful handling
- Support/container objects for placement
- Goal predicates describing task completion conditions
- Safety status rules for monitoring task execution

The parser is intentionally strict to reject malformed or unsupported tasks early.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bddl.activity import Conditions
from bddl.object_taxonomy import ObjectTaxonomy

# Goal predicates we currently allow in the manipulation MVP parser.
DEFAULT_ALLOWED_GOAL_PREDICATES = {
    "inside",
    "ontop",
    "nextto",
    "touching",
    "filled",
    "covered",
    "on_fire",
    "hot",
    "toggled_on",
    "grasped",
}

DEFAULT_SAFETY_STATUS_RULES = (
    "is_fragile_broken",      # Check if any fragile objects are broken
    "is_object_dropped",      # Check if objects are dropped unexpectedly
    "is_collision_risk",       # Check for potential collisions
    "is_spill_risk_event",    # Check for spill risks
)

# Global object taxonomy instance for querying object properties (e.g., breakability).
# Cached to avoid repeated initialization overhead.
_OBJECT_TAXONOMY = ObjectTaxonomy()


class ManipulationTaskSpecValidationError(ValueError):
    """Raised when a parsed manipulation task spec is incomplete or invalid."""


@dataclass(frozen=True)
class GoalPredicateSpec:
    """Specification for a single goal predicate extracted from BDDL goal conditions.
    
    Attributes:
        name: Predicate name, e.g. "ontop", "inside", "nextto", "touching".
        args: Predicate arguments as a tuple of strings, e.g. ("cup.n.01_1", "table.n.02_1")
            for "ontop(cup.n.01_1, table.n.02_1)".
        negated: Whether the predicate is negated. True if wrapped in "not", e.g.
            "not ontop(cup, table)" would have negated=True.
        quantified_vars: Tuple of (variable_name, synset) pairs for quantified variables.
            For example, ("?obj1", "cup.n.01") for "forall ?obj1 in cup.n.01: ...".
    """
    name: str
    args: Tuple[str, ...]
    negated: bool
    quantified_vars: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class ManipulationTaskSpec:
    task_name: str
    objects: Dict[str, Tuple[str, ...]]
    target_ids: Tuple[str, ...]
    fragile_ids: Tuple[str, ...]
    support_ids: Tuple[str, ...]
    goal_predicates: Tuple[GoalPredicateSpec, ...]
    safety_status_rules: Tuple[str, ...]


def build_manipulation_task_spec(
    activity_name: str,
    activity_definition_id: int = 0,
    simulator_name: str = "omnigibson",
    predefined_problem: Optional[str] = None,
    allowed_goal_predicates: Optional[Iterable[str]] = None,
    safety_status_rules: Sequence[str] = DEFAULT_SAFETY_STATUS_RULES,
) -> ManipulationTaskSpec:
    """
    Build a normalized manipulation task spec from BDDL Conditions.

    This parser is intentionally strict so malformed / unsupported tasks are rejected early.
    """
    # Parse BDDL conditions from the activity definition
    conditions = Conditions(
        behavior_activity=activity_name,
        activity_definition=activity_definition_id,
        simulator_name=simulator_name,
        predefined_problem=predefined_problem,
    )

    # Build object mappings: synset -> instances, and instance -> synset
    objects: Dict[str, Tuple[str, ...]] = {
        synset: tuple(sorted(instances)) for synset, instances in conditions.parsed_objects.items()
    }
    instance_to_synset = _build_instance_to_synset_map(objects)

    # Extract goal predicates from BDDL goal conditions
    goal_predicates = tuple(_extract_goal_predicates(conditions.parsed_goal_conditions))
    
    # Infer task components from parsed conditions
    target_ids = _infer_target_ids(goal_predicates=goal_predicates, objects=objects, instance_to_synset=instance_to_synset)
    support_ids = _infer_support_ids(
        init_conditions=conditions.parsed_initial_conditions,
        goal_predicates=goal_predicates,
        objects=objects,
        instance_to_synset=instance_to_synset,
    )
    fragile_ids = _infer_fragile_ids(instance_to_synset)

    # Construct the task specification
    spec = ManipulationTaskSpec(
        task_name=activity_name,
        objects=objects,
        target_ids=tuple(sorted(target_ids)),
        fragile_ids=tuple(sorted(fragile_ids)),
        support_ids=tuple(sorted(support_ids)),
        goal_predicates=goal_predicates,
        safety_status_rules=tuple(safety_status_rules),
    )

    # Validate the spec and raise error if invalid
    _validate_spec(
        spec=spec,
        allowed_goal_predicates=set(DEFAULT_ALLOWED_GOAL_PREDICATES if allowed_goal_predicates is None else allowed_goal_predicates),
    )
    return spec


def _build_instance_to_synset_map(objects: Dict[str, Tuple[str, ...]]) -> Dict[str, str]:
    return {instance: synset for synset, instances in objects.items() for instance in instances}


def _extract_goal_predicates(
    goal_conditions: Sequence,
    quantified_scope: Optional[Dict[str, str]] = None,
    negated: bool = False,
) -> List[GoalPredicateSpec]:
    """Recursively extract goal predicates from BDDL goal conditions.
    
    This function handles nested BDDL structures including:
    - "and" conjunctions: (and P1 P2 ...)
    - Quantifiers: (forall (?x synset) P) or (exists (?x synset) P)
    - Negations: (not P)
    - Base predicates: (predicate arg1 arg2 ...)
    
    Args:
        goal_conditions: Sequence of BDDL goal condition expressions (nested lists).
        quantified_scope: Dictionary mapping quantified variable names to their synsets.
            Used for resolving variables in nested scopes. Defaults to empty dict.
        negated: Whether the current scope is negated. Defaults to False.
    
    Returns:
        List of GoalPredicateSpec objects extracted from the goal conditions.
    """
    quantified_scope = {} if quantified_scope is None else dict(quantified_scope)
    out: List[GoalPredicateSpec] = []

    for cond in goal_conditions:
        if not isinstance(cond, list) or len(cond) == 0:
            continue

        head = cond[0]

        if head == "and":
            out.extend(_extract_goal_predicates(cond[1:], quantified_scope=quantified_scope, negated=negated))
            continue

        if head in {"forall", "exists"}:
            if len(cond) < 3 or not isinstance(cond[1], list) or len(cond[1]) < 3:
                continue
            scoped_quantifiers = dict(quantified_scope)
            var_name, synset = cond[1][0], cond[1][2]
            scoped_quantifiers[var_name] = synset
            out.extend(_extract_goal_predicates([cond[2]], quantified_scope=scoped_quantifiers, negated=negated))
            continue

        if head == "not":
            if len(cond) < 2 or not isinstance(cond[1], list):
                continue
            out.extend(_extract_goal_predicates([cond[1]], quantified_scope=quantified_scope, negated=not negated))
            continue

        args = tuple(str(arg) for arg in cond[1:])
        out.append(
            GoalPredicateSpec(
                name=str(head),
                args=args,
                negated=negated,
                quantified_vars=tuple(sorted(quantified_scope.items())),
            )
        )

    return out


def _infer_target_ids(
    goal_predicates: Sequence[GoalPredicateSpec],
    objects: Dict[str, Tuple[str, ...]],
    instance_to_synset: Dict[str, str],
) -> List[str]:
    movable_goal_preds = {"inside", "ontop", "nextto", "touching", "filled"}
    targets = set()

    for pred in goal_predicates:
        if pred.negated or len(pred.args) == 0:
            continue

        # For grasped(agent, obj), the target is the second argument.
        if pred.name == "grasped" and len(pred.args) >= 2:
            subject = pred.args[1]
        elif pred.name in movable_goal_preds:
            subject = pred.args[0]
        else:
            continue

        resolved_subjects = _resolve_argument_instances(
            argument=subject,
            quantified_vars=dict(pred.quantified_vars),
            objects=objects,
            instance_to_synset=instance_to_synset,
        )
        for subject_id in resolved_subjects:
            synset = instance_to_synset.get(subject_id)
            if synset in {"agent.n.01", "floor.n.01"}:
                continue
            targets.add(subject_id)

    return sorted(targets)


def _infer_support_ids(
    init_conditions: Sequence,
    goal_predicates: Sequence[GoalPredicateSpec],
    objects: Dict[str, Tuple[str, ...]],
    instance_to_synset: Dict[str, str],
) -> List[str]:
    supports = set()

    # Destination objects in init conditions for ontop / inside are strong support signals.
    # inroom also implies a scene object that acts as implicit support.
    for cond in init_conditions:
        if not isinstance(cond, list) or len(cond) < 2:
            continue
        if cond[0] in {"ontop", "inside"} and len(cond) >= 3:
            support = _normalize_instance_token(str(cond[2]), instance_to_synset)
            if support is not None:
                supports.add(support)
        elif cond[0] == "inroom" and len(cond) >= 2:
            support = _normalize_instance_token(str(cond[1]), instance_to_synset)
            if support is not None:
                supports.add(support)

    # Destination objects in positive relocation goals are also support signals.
    for pred in goal_predicates:
        if pred.negated or pred.name not in {"inside", "ontop"} or len(pred.args) < 2:
            continue
        destination = pred.args[1]
        resolved_destinations = _resolve_argument_instances(
            argument=destination,
            quantified_vars=dict(pred.quantified_vars),
            objects=objects,
            instance_to_synset=instance_to_synset,
        )
        supports.update(resolved_destinations)

    # Filter out agent / floor from supports.
    filtered = []
    for support_id in sorted(supports):
        synset = instance_to_synset.get(support_id)
        if synset in {"agent.n.01", "floor.n.01"}:
            continue
        filtered.append(support_id)
    return filtered


def _resolve_argument_instances(
    argument: str,
    quantified_vars: Dict[str, str],
    objects: Dict[str, Tuple[str, ...]],
    instance_to_synset: Dict[str, str],
) -> List[str]:
    # 1) If this is a concrete instance token (with or without accidental '?'), normalize it.
    normalized_instance = _normalize_instance_token(argument, instance_to_synset)
    if normalized_instance is not None:
        return [normalized_instance]

    # 2) If this is a quantified variable, expand to all instances of that synset.
    if argument.startswith("?") and argument in quantified_vars:
        synset = quantified_vars[argument]
        return list(objects.get(synset, ()))

    return []


def _normalize_instance_token(token: str, instance_to_synset: Dict[str, str]) -> Optional[str]:
    if token in instance_to_synset:
        return token
    if token.startswith("?"):
        stripped = token[1:]
        if stripped in instance_to_synset:
            return stripped
    return None


def _infer_fragile_ids(instance_to_synset: Dict[str, str]) -> List[str]:
    return sorted([instance for instance, synset in instance_to_synset.items() if _is_synset_breakable(synset)])


@lru_cache(maxsize=2048)
def _is_synset_breakable(synset: str) -> bool:
    try:
        abilities = _OBJECT_TAXONOMY.get_abilities(synset)
    except Exception:
        return False
    return "breakable" in abilities


def _validate_spec(spec: ManipulationTaskSpec, allowed_goal_predicates: set) -> None:
    if len(spec.target_ids) == 0:
        raise ManipulationTaskSpecValidationError(
            f"Task {spec.task_name} has no target objects inferred from goal predicates."
        )

    if len(spec.support_ids) == 0:
        raise ManipulationTaskSpecValidationError(
            f"Task {spec.task_name} has no support/container objects inferred from init/goal predicates."
        )

    unsupported = sorted({pred.name for pred in spec.goal_predicates if pred.name not in allowed_goal_predicates})
    if unsupported:
        raise ManipulationTaskSpecValidationError(
            f"Task {spec.task_name} has unsupported goal predicates: {unsupported}"
        )
