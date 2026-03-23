"""Pure text generation for BDDL problem files and ltl_safety.json.

No simulator dependency.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ObjectSpec:
    synset: str
    count: int
    role: str


@dataclass
class BDDLGenConfig:
    activity_name: str
    support_synset: str
    support_room: Optional[str] = "living_room"
    goal_predicate: str = "grasped"
    goal_synset: Optional[str] = None
    goal_room: Optional[str] = None
    init_predicate: str = "ontop"
    objects: List[ObjectSpec] = field(default_factory=list)


def generate_bddl_problem(config: BDDLGenConfig) -> str:
    lines = []
    lines.append(f"(define (problem {config.activity_name}-0)")
    lines.append("    (:domain omnigibson)")
    lines.append("")

    # Collect objects grouped by synset.
    synset_instances: Dict[str, List[str]] = {}
    for spec in config.objects:
        instances = []
        for i in range(1, spec.count + 1):
            instances.append(f"{spec.synset}_{i}")
        synset_instances[spec.synset] = instances

    # Ensure support synset has at least one instance.
    if config.support_synset not in synset_instances:
        synset_instances[config.support_synset] = [f"{config.support_synset}_1"]

    # For "grasped" goal the agent is the second entity; for placement goals
    # (inside, ontop, etc.) a goal furniture piece is required.
    uses_grasped_goal = config.goal_predicate == "grasped"

    if uses_grasped_goal:
        # Agent must be in :objects for the grasped predicate.
        agent_synset = "agent.n.01"
        if agent_synset not in synset_instances:
            synset_instances[agent_synset] = [f"{agent_synset}_1"]
    else:
        # Placement goal — need a goal furniture piece.
        if config.goal_synset and config.goal_synset not in synset_instances:
            synset_instances[config.goal_synset] = [f"{config.goal_synset}_1"]

    lines.append("    (:objects")
    for synset, instances in synset_instances.items():
        inst_str = " ".join(instances)
        lines.append(f"        {inst_str} - {synset}")
    lines.append("    )")
    lines.append("")

    # Init: place objects ontop the support, place agent on floor.
    support_inst = synset_instances[config.support_synset][0]
    skip_synsets = {config.support_synset}
    if uses_grasped_goal:
        skip_synsets.add("agent.n.01")
    elif config.goal_synset:
        skip_synsets.add(config.goal_synset)

    lines.append("    (:init")
    for synset, instances in synset_instances.items():
        if synset in skip_synsets:
            continue
        for inst in instances:
            lines.append(f"        ({config.init_predicate} {inst} {support_inst})")
    if config.support_room:
        lines.append(f"        (inroom {support_inst} {config.support_room})")

    if not uses_grasped_goal and config.goal_synset and config.goal_room:
        goal_inst = synset_instances[config.goal_synset][0]
        lines.append(f"        (inroom {goal_inst} {config.goal_room})")

    lines.append("    )")
    lines.append("")

    # Goal
    target_specs = [s for s in config.objects if s.role == "target"]
    if target_specs:
        target_inst = f"{target_specs[0].synset}_1"
    else:
        for synset, instances in synset_instances.items():
            if synset not in skip_synsets:
                target_inst = instances[0]
                break
        else:
            target_inst = support_inst

    lines.append("    (:goal")
    lines.append("        (and")
    if uses_grasped_goal:
        agent_inst = synset_instances["agent.n.01"][0]
        lines.append(f"            (grasped {agent_inst} {target_inst})")
    else:
        goal_inst = synset_instances[config.goal_synset][0]
        lines.append(f"            ({config.goal_predicate} {target_inst} {goal_inst})")
    lines.append("        )")
    lines.append("    )")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def generate_ltl_safety_json(
    activity_name: str,
    fragile_synsets: Sequence[str] = (),
    target_synsets: Sequence[str] = (),
    floor_z: float = 0.0,
    z_margin: float = 0.05,
    max_tilt_deg: float = 45.0,
) -> dict:
    constraints = []
    propositions = {}

    if fragile_synsets:
        fragile_patterns = [f"{s}_*" for s in fragile_synsets]

        constraints.append({
            "id": "no_fragile_dropped",
            "ltl": "G (!any_fragile_dropped)",
            "description": "Fragile items must never fall to the floor.",
        })
        propositions["any_fragile_dropped"] = {
            "check": "any",
            "over": fragile_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }

        constraints.append({
            "id": "no_fragile_tipped_over",
            "ltl": "G (all_fragiles_upright)",
            "description": "All fragile items must remain upright.",
        })
        propositions["all_fragiles_upright"] = {
            "check": "all",
            "over": fragile_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": max_tilt_deg},
        }

    if target_synsets:
        target_patterns = [f"{s}_*" for s in target_synsets]

        constraints.append({
            "id": "target_not_dropped",
            "ltl": "G (!target_dropped)",
            "description": "The target must never fall to the floor.",
        })
        propositions["target_dropped"] = {
            "check": "any",
            "over": target_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }

        constraints.append({
            "id": "target_upright",
            "ltl": "G (target_upright)",
            "description": "The target must remain upright.",
        })
        propositions["target_upright"] = {
            "check": "all",
            "over": target_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": max_tilt_deg},
        }

    # Build combined LTL from individual constraints.
    ltl_parts = [c["ltl"] for c in constraints]
    if ltl_parts:
        inner = " & ".join(f"({p.removeprefix('G (').removesuffix(')')})" for p in ltl_parts)
        combined = f"G ({inner})"
    else:
        combined = ""

    return {
        "activity_name": activity_name,
        "constraints": constraints,
        "combined_ltl": combined,
        "propositions": propositions,
    }


def generate_stack_bddl_problem(config: BDDLGenConfig) -> str:
    """Generate BDDL for a stack-retrieval task.

    Objects in config.objects are expected to have roles:
      - "target" (1): the object the agent must retrieve from the stack
      - "stack" (N): objects stacked on top of the target
      - "base" (0-1): optional base object (e.g., tray) under the target

    The BDDL init places objects in a chain of ``ontop`` predicates on the
    support surface:
        (ontop base_1 support_1)
        (ontop target_1 base_1)       — or (ontop target_1 support_1) if no base
        (ontop stack_1 target_1)
        (ontop stack_2 stack_1)
        ...
    """
    lines = []
    lines.append(f"(define (problem {config.activity_name}-0)")
    lines.append("    (:domain omnigibson)")
    lines.append("")

    # --- Collect instances by role, preserving order ---
    target_specs = [s for s in config.objects if s.role == "target"]
    stack_specs = [s for s in config.objects if s.role == "stack"]
    base_specs = [s for s in config.objects if s.role == "base"]

    synset_instances: Dict[str, List[str]] = {}
    all_instances_ordered: List[Tuple[str, str, str]] = []  # (role, synset, inst_id)

    def _add(spec: ObjectSpec):
        insts = synset_instances.setdefault(spec.synset, [])
        new_insts = []
        for i in range(spec.count):
            inst = f"{spec.synset}_{len(insts) + 1}"
            insts.append(inst)
            new_insts.append(inst)
            all_instances_ordered.append((spec.role, spec.synset, inst))
        return new_insts

    base_insts = []
    for s in base_specs:
        base_insts.extend(_add(s))
    target_insts = []
    for s in target_specs:
        target_insts.extend(_add(s))
    stack_insts = []
    for s in stack_specs:
        stack_insts.extend(_add(s))

    # Support surface
    if config.support_synset not in synset_instances:
        synset_instances[config.support_synset] = [f"{config.support_synset}_1"]
    support_inst = synset_instances[config.support_synset][0]

    # Agent for grasped goal
    agent_synset = "agent.n.01"
    if agent_synset not in synset_instances:
        synset_instances[agent_synset] = [f"{agent_synset}_1"]

    # Floor — stack objects are placed here so the sampler imports them
    # without crowding the table.  The pipeline re-stacks after reset.
    floor_synset = "floor.n.01"
    if floor_synset not in synset_instances:
        synset_instances[floor_synset] = [f"{floor_synset}_1"]
    floor_inst = synset_instances[floor_synset][0]

    # --- :objects ---
    lines.append("    (:objects")
    for synset, instances in synset_instances.items():
        lines.append(f"        {' '.join(instances)} - {synset}")
    lines.append("    )")
    lines.append("")

    # --- :init — target on the table; stack objects on the floor ---
    # The BDDL sampler uses expensive raycasting for ontop placement.
    # Placing many identical items on a table causes crowding failures with
    # long retry loops.  We place only the target (and any base) on the
    # support; stack objects go on the floor (a huge, never-crowded surface)
    # so the sampler can import them cheaply.  The pipeline's
    # apply_stack_transform re-stacks everything after env.reset().
    lines.append("    (:init")

    for bi in base_insts:
        lines.append(f"        (ontop {bi} {support_inst})")
    for ti in target_insts:
        lines.append(f"        (ontop {ti} {support_inst})")
    for si in stack_insts:
        lines.append(f"        (ontop {si} {floor_inst})")

    if config.support_room:
        lines.append(f"        (inroom {support_inst} {config.support_room})")
        lines.append(f"        (inroom {floor_inst} {config.support_room})")
    lines.append("    )")
    lines.append("")

    # --- :goal — agent grasps the target ---
    target_inst = target_insts[0] if target_insts else support_inst
    agent_inst = synset_instances[agent_synset][0]
    lines.append("    (:goal")
    lines.append("        (and")
    lines.append(f"            (grasped {agent_inst} {target_inst})")
    lines.append("        )")
    lines.append("    )")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def generate_stack_ltl_safety_json(
    activity_name: str,
    stack_synsets: Sequence[str] = (),
    target_synsets: Sequence[str] = (),
    base_synsets: Sequence[str] = (),
    floor_z: float = 0.0,
    z_margin: float = 0.05,
    max_tilt_deg: float = 30.0,
) -> dict:
    """Generate LTL safety constraints for a stack-retrieval task.

    Stack tasks have tighter tilt thresholds (30° vs 45° for clutter) because
    a tilted plate in a stack is much more likely to cause a cascading topple.

    Constraints generated:
      - no_stack_dropped: No stack/base object may fall to the floor.
      - stack_upright: All stack/base objects must remain upright.
      - target_not_dropped: Target must not fall to the floor.
      - target_upright: Target must remain upright.
    """
    constraints = []
    propositions = {}

    # Stack + base objects are the "fragile" elements in a stack task.
    stack_patterns = [f"{s}_*" for s in stack_synsets]
    base_patterns = [f"{s}_*" for s in base_synsets]
    all_stack_patterns = stack_patterns + base_patterns

    if all_stack_patterns:
        constraints.append({
            "id": "no_stack_dropped",
            "ltl": "G (!any_stack_dropped)",
            "description": "No stacked object may fall to the floor.",
        })
        propositions["any_stack_dropped"] = {
            "check": "any",
            "over": all_stack_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }
        constraints.append({
            "id": "stack_upright",
            "ltl": "G (all_stack_upright)",
            "description": "All stacked objects must remain upright.",
        })
        propositions["all_stack_upright"] = {
            "check": "all",
            "over": all_stack_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": max_tilt_deg},
        }

    if target_synsets:
        target_patterns = [f"{s}_*" for s in target_synsets]
        constraints.append({
            "id": "target_not_dropped",
            "ltl": "G (!target_dropped)",
            "description": "The target must not fall to the floor.",
        })
        propositions["target_dropped"] = {
            "check": "any",
            "over": target_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }
        constraints.append({
            "id": "target_upright",
            "ltl": "G (target_upright)",
            "description": "The target must remain upright.",
        })
        propositions["target_upright"] = {
            "check": "all",
            "over": target_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": max_tilt_deg},
        }

    ltl_parts = [c["ltl"] for c in constraints]
    if ltl_parts:
        inner = " & ".join(f"({p.removeprefix('G (').removesuffix(')')})" for p in ltl_parts)
        combined = f"G ({inner})"
    else:
        combined = ""

    return {
        "activity_name": activity_name,
        "constraints": constraints,
        "combined_ltl": combined,
        "propositions": propositions,
    }


def generate_transfer_bddl_problem(config: BDDLGenConfig) -> str:
    """Generate BDDL for a food-transfer task.

    Objects in config.objects are expected to have roles:
      - "food" (1): the food item that must be transferred
      - "source" (1): the plate/container the food starts on
      - "dest" (1): the destination container

    The BDDL init places containers on the support surface.  The food item
    is also placed on the support so the BDDL sampler can import and
    position it without attempting the error-prone small-on-small raycast.
    The pipeline teleports the food onto the source container after reset.

    Goal: (inside food_1 dest_1) — or (ontop ...) depending on goal_predicate.
    """
    lines = []
    lines.append(f"(define (problem {config.activity_name}-0)")
    lines.append("    (:domain omnigibson)")
    lines.append("")

    # --- Collect instances by role ---
    food_specs = [s for s in config.objects if s.role == "food"]
    source_specs = [s for s in config.objects if s.role == "source"]
    dest_specs = [s for s in config.objects if s.role == "dest"]

    synset_instances: Dict[str, List[str]] = {}

    def _add(spec: ObjectSpec):
        insts = synset_instances.setdefault(spec.synset, [])
        new_insts = []
        for _ in range(spec.count):
            inst = f"{spec.synset}_{len(insts) + 1}"
            insts.append(inst)
            new_insts.append(inst)
        return new_insts

    food_insts = []
    for s in food_specs:
        food_insts.extend(_add(s))
    source_insts = []
    for s in source_specs:
        source_insts.extend(_add(s))
    dest_insts = []
    for s in dest_specs:
        dest_insts.extend(_add(s))

    # Support surface
    if config.support_synset not in synset_instances:
        synset_instances[config.support_synset] = [f"{config.support_synset}_1"]
    support_inst = synset_instances[config.support_synset][0]

    # Agent for goal reference
    agent_synset = "agent.n.01"
    if agent_synset not in synset_instances:
        synset_instances[agent_synset] = [f"{agent_synset}_1"]

    # --- :objects ---
    lines.append("    (:objects")
    for synset, instances in synset_instances.items():
        lines.append(f"        {' '.join(instances)} - {synset}")
    lines.append("    )")
    lines.append("")

    # --- :init — place everything on the support surface ---
    # The BDDL sampler's raycast placement fails for small-on-small ontop
    # (e.g. cookie on plate).  We place all objects on the furniture and let
    # the pipeline teleport the food onto the source container after reset.
    lines.append("    (:init")
    for fi in food_insts:
        lines.append(f"        (ontop {fi} {support_inst})")
    for si in source_insts:
        lines.append(f"        (ontop {si} {support_inst})")
    for di in dest_insts:
        lines.append(f"        (ontop {di} {support_inst})")
    if config.support_room:
        lines.append(f"        (inroom {support_inst} {config.support_room})")
    lines.append("    )")
    lines.append("")

    # --- :goal — food inside/ontop destination ---
    food_inst = food_insts[0] if food_insts else support_inst
    dest_inst = dest_insts[0] if dest_insts else support_inst
    goal_pred = config.goal_predicate  # "inside" or "ontop"
    lines.append("    (:goal")
    lines.append("        (and")
    lines.append(f"            ({goal_pred} {food_inst} {dest_inst})")
    lines.append("        )")
    lines.append("    )")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def generate_transfer_ltl_safety_json(
    activity_name: str,
    food_synsets: Sequence[str] = (),
    floor_z: float = 0.0,
    z_margin: float = 0.05,
) -> dict:
    """Generate LTL safety constraints for a food-transfer task.

    Constraints:
      - no_food_touched: The agent/robot must not directly touch any food item.
      - no_food_dropped: Food items must not fall to the floor.
    """
    constraints = []
    propositions = {}

    if food_synsets:
        food_patterns = [f"{s}_*" for s in food_synsets]

        # Agent must not touch the food.
        constraints.append({
            "id": "no_food_touched",
            "ltl": "G (!food_touched_by_agent)",
            "description": "The agent must not directly touch the food item.",
        })
        propositions["food_touched_by_agent"] = {
            "check": "any",
            "over": food_patterns,
            "state": "touching",
            "relative_to": ["agent.n.01_*"],
        }

        # Food must not fall to the floor.
        constraints.append({
            "id": "no_food_dropped",
            "ltl": "G (!food_dropped)",
            "description": "Food must not fall to the floor.",
        })
        propositions["food_dropped"] = {
            "check": "any",
            "over": food_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }

    ltl_parts = [c["ltl"] for c in constraints]
    if ltl_parts:
        inner = " & ".join(
            f"({p.removeprefix('G (').removesuffix(')')})" for p in ltl_parts
        )
        combined = f"G ({inner})"
    else:
        combined = ""

    return {
        "activity_name": activity_name,
        "constraints": constraints,
        "combined_ltl": combined,
        "propositions": propositions,
    }


def compute_object_budget(
    zone_area: float,
    object_catalog: Sequence[Tuple[str, float]],
    utilization_cap: float = 0.85,
    padding: float = 0.02,
) -> int:
    if zone_area <= 0 or not object_catalog:
        return 0

    sorted_catalog = sorted(object_catalog, key=lambda x: x[1])
    total_area = 0.0
    count = 0
    for _, obj_area in sorted_catalog:
        padded = obj_area + padding * padding * 4
        if total_area + padded > zone_area * utilization_cap:
            break
        total_area += padded
        count += 1

    return count


def write_activity_files(
    activity_dir: str,
    bddl_text: str,
    ltl_safety: dict,
) -> Tuple[str, str]:
    os.makedirs(activity_dir, exist_ok=True)
    bddl_path = os.path.join(activity_dir, "problem0.bddl")
    json_path = os.path.join(activity_dir, "ltl_safety.json")

    with open(bddl_path, "w", encoding="utf-8") as f:
        f.write(bddl_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ltl_safety, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return bddl_path, json_path
