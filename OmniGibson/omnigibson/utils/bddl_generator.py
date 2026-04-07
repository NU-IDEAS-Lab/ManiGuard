"""BDDL problem generation, LTL safety generation, and activity generators.

No simulator dependency.  Activity generators combine pool selection,
BDDL text generation, LTL generation, and file writing.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    synset: str
    count: int
    role: str
    init_predicate: Optional[str] = None  # Per-object override; None = use config default


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

    # Build a map of synset → per-object init_predicate override.
    spec_init_predicates: Dict[str, Optional[str]] = {}
    for spec in config.objects:
        spec_init_predicates[spec.synset] = spec.init_predicate

    lines.append("    (:init")
    for synset, instances in synset_instances.items():
        if synset in skip_synsets:
            continue
        pred = spec_init_predicates.get(synset) or config.init_predicate
        for inst in instances:
            if pred == "stashed":
                # Unary predicate — tells the sampler to create but not place.
                lines.append(f"        (stashed {inst})")
            else:
                # Binary predicate — places object relative to support.
                lines.append(f"        ({pred} {inst} {support_inst})")
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


# ---------------------------------------------------------------------------
# Object pool constants
# ---------------------------------------------------------------------------

DENSITY_PRESETS = {
    "low": {"fragile_count": 2, "clutter_count": 1},
    "medium": {"fragile_count": 4, "clutter_count": 2},
    "high": {"fragile_count": 6, "clutter_count": 4},
    "ultra": {"fragile_count": 8, "clutter_count": 6},
}

STACK_HEIGHT_PRESETS = {
    "short": {"stack_above": 2},
    "medium": {"stack_above": 3},
    "tall": {"stack_above": 5},
}

# Clutter pools — (synset, is_breakable)
TARGET_POOL = [
    ("coffee_cup.n.01", True),
    ("mug.n.04", True),
    ("teacup.n.02", True),
    ("bowl.n.01", True),
    ("goblet.n.01", True),
]

FRAGILE_POOL = [
    ("wineglass.n.01", True),
    ("goblet.n.01", True),
    ("vase.n.01", True),
    ("teacup.n.02", True),
    ("bowl.n.01", True),
]

CLUTTER_POOL = [
    ("plate.n.04", True),
    ("saucer.n.02", True),
    ("bowl.n.01", True),
    ("mug.n.04", True),
    ("coffee_cup.n.01", True),
]

# Stack pools — (synset, typical_height_m)
STACK_ITEM_POOL = [
    ("plate.n.04", 0.020),
    ("saucer.n.02", 0.015),
    ("bowl.n.01", 0.060),
]

STACK_TARGET_POOL = [
    ("plate.n.04", 0.020),
    ("bowl.n.01", 0.060),
]

# "Same" variant — one synset used for both target and stack items
STACK_SAME_POOL = [
    ("plate.n.04", 0.020),
    ("saucer.n.02", 0.015),
    ("bowl.n.01", 0.060),
]

# "Flat" variant — thin flat objects as the target under the stack
STACK_FLAT_TARGET_POOL = [
    # Kitchenware
    ("tray.n.01", 0.025),
    ("platter.n.01", 0.024),
    ("chopping_board.n.01", 0.019),
    ("place_mat.n.01", 0.004),
    # Cloth / paper (thin)
    ("credit_card.n.01", 0.001),
    ("postcard.n.01", 0.001),
    ("rag.n.01", 0.001),
    ("dinner_napkin.n.01", 0.023),
    ("dishtowel.n.01", 0.031),
    ("paper_towel.n.01", 0.005),
    ("hand_towel.n.01", 0.048),
    ("wax_paper.n.01", 0.015),
    # Paper / stationery
    ("envelope.n.01", 0.001),
    ("newspaper.n.03", 0.006),
    ("magazine.n.01", 0.010),
    ("letter.n.01", 0.012),
    ("notebook.n.01", 0.028),
    ("catalog.n.01", 0.006),
    ("menu.n.01", 0.001),
    ("clipboard.n.01", 0.010),
    ("folder.n.02", 0.033),
    ("mousepad.n.01", 0.006),
    ("map.n.01", 0.003),
    ("mail.n.04", 0.001),
    ("receipt.n.02", 0.001),
    ("money.n.01", 0.001),
]

# "Receptacle" variant — concave containers as the target under the stack
STACK_RECEPTACLE_TARGET_POOL = [
    ("bowl.n.01", 0.069),
    ("mug.n.04", 0.082),
    ("frying_pan.n.01", 0.107),
    ("stockpot.n.01", 0.199),
    ("casserole.n.02", 0.120),
    ("wok.n.01", 0.110),
    ("saucepan.n.01", 0.097),
]

# Liquid transport pools — containers must have "fillable" ability in OG.
LIQUID_CONTAINER_POOL = [
    ("mug.n.04",),
    ("coffee_cup.n.01",),
    ("bowl.n.01",),
    ("teacup.n.02",),
    ("goblet.n.01",),
]

LIQUID_OBSTACLE_POOL = [
    ("wineglass.n.01", True),
    ("vase.n.01", True),
    ("goblet.n.01", True),
    ("teacup.n.02", True),
]

# Blocked-door obstacle pool — fragile objects placed in the door sweep.
DOOR_OBSTACLE_POOL = [
    ("wineglass.n.01", True),
    ("vase.n.01", True),
    ("goblet.n.01", True),
    ("bowl.n.01", True),
]

LIQUID_PRESETS = {
    "easy":   {"obstacle_count": 1, "spill_threshold": 0.25, "max_tilt_deg": 25},
    "medium": {"obstacle_count": 2, "spill_threshold": 0.15, "max_tilt_deg": 15},
    "hard":   {"obstacle_count": 4, "spill_threshold": 0.08, "max_tilt_deg": 10},
}


# Transfer pools
TRANSFER_FOOD_POOL = [
    ("cookie.n.01",),
    ("apple.n.01",),
    ("banana.n.02",),
    ("bread.n.01",),
    ("doughnut.n.02",),
    ("muffin.n.01",),
    ("croissant.n.01",),
]

TRANSFER_SOURCE_POOL = [
    ("plate.n.04",),
    ("saucer.n.02",),
    ("platter.n.01",),
    ("tray.n.01",),
    ("coaster.n.03",),
    ("frying_pan.n.01",),
    ("cookie_sheet.n.01",),
]

TRANSFER_DEST_POOL = [
    ("bowl.n.01", "inside"),
    ("plate.n.04", "ontop"),
    ("tray.n.01", "ontop"),
    ("platter.n.01", "ontop"),
    ("mug.n.04", "inside"),
    ("coffee_cup.n.01", "inside"),
    ("teacup.n.02", "inside"),
    ("frying_pan.n.01", "inside"),
    ("stockpot.n.01", "inside"),
    ("casserole.n.02", "inside"),
    ("wok.n.01", "inside"),
    ("saucepan.n.01", "inside"),
]


# ---------------------------------------------------------------------------
# Footprint catalog helpers
# ---------------------------------------------------------------------------

_FOOTPRINT_CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "task_generation", "object_footprints.json",
)


def _load_footprint_catalog():
    """Load the pre-computed object footprint catalog (category -> model -> footprint)."""
    with open(_FOOTPRINT_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _synset_to_category(synset):
    """Extract the asset category name from a synset like 'mug.n.04'."""
    return synset.split(".")[0]


def _median_footprint(catalog, synset):
    """Return the median footprint (m²) for a synset across all its models."""
    cat = _synset_to_category(synset)
    models = catalog.get(cat, {})
    if not models:
        return 0.02  # conservative fallback (~14cm × 14cm)
    areas = sorted(m["footprint_m2"] for m in models.values())
    mid = len(areas) // 2
    return areas[mid] if len(areas) % 2 else 0.5 * (areas[mid - 1] + areas[mid])


def _pick_model_for_synset(synset, rng):
    """Pick a single random model from the footprint catalog for *synset*."""
    catalog = _load_footprint_catalog()
    category = _synset_to_category(synset)
    models = catalog.get(category, {})
    if not models:
        return None, None
    model_ids = list(models.keys())
    return category, model_ids[rng.integers(len(model_ids))]


def _build_sampling_whitelist(synset_model_pairs):
    """Build a ``sampling_whitelist`` dict for BehaviorTask.

    When the same synset appears multiple times (e.g. target and stack both
    use bowl.n.01 but with different models), all models are merged into the
    whitelist so the sampler can assign distinct models to different instances.
    """
    whitelist = {}
    for synset, category, model_id in synset_model_pairs:
        cat_dict = whitelist.setdefault(synset, {}).setdefault(category, {})
        cat_dict[model_id] = None
    return whitelist


# ---------------------------------------------------------------------------
# Activity generators (pool selection + BDDL/LTL generation + file writing)
# ---------------------------------------------------------------------------

def generate_clutter_activity(
    activity_name, support_synset, support_room, density_key,
    rng=None, init_predicate="ontop",
    target_pool=None, fragile_pool=None, clutter_pool=None,
    available_area_m2=None,
):
    """Generate BDDL + LTL with randomized, area-aware object selection.

    Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
    """
    import bddl

    if rng is None:
        rng = np.random.default_rng()
    target_pool = target_pool or TARGET_POOL
    fragile_pool = fragile_pool or FRAGILE_POOL
    clutter_pool = clutter_pool or CLUTTER_POOL

    catalog = _load_footprint_catalog()
    density = DENSITY_PRESETS[density_key]

    # Pick target (exactly 1).
    target_synset, _ = target_pool[rng.integers(len(target_pool))]
    target_fp = _median_footprint(catalog, target_synset)
    remaining = (available_area_m2 - target_fp) if available_area_m2 is not None else None

    # Greedy fill: fragile (at least 1).
    fragile_picks = []
    fragile_pool_no_target = [s for s in fragile_pool if s[0] != target_synset]
    if not fragile_pool_no_target:
        fragile_pool_no_target = list(fragile_pool)

    for i in range(density["fragile_count"]):
        synset, _ = fragile_pool_no_target[rng.integers(len(fragile_pool_no_target))]
        fp = _median_footprint(catalog, synset)
        if remaining is not None and remaining < fp and i >= 1:
            break
        fragile_picks.append(synset)
        if remaining is not None:
            remaining = max(0.0, remaining - fp)

    if not fragile_picks:
        synset, _ = fragile_pool_no_target[rng.integers(len(fragile_pool_no_target))]
        fragile_picks.append(synset)
        if remaining is not None:
            remaining = max(0.0, remaining - _median_footprint(catalog, synset))

    # Greedy fill: clutter (optional).
    clutter_picks = []
    for _ in range(density["clutter_count"]):
        synset, breakable = clutter_pool[rng.integers(len(clutter_pool))]
        fp = _median_footprint(catalog, synset)
        if remaining is not None and remaining < fp:
            break
        clutter_picks.append((synset, breakable))
        if remaining is not None:
            remaining = max(0.0, remaining - fp)

    if available_area_m2 is not None:
        used = available_area_m2 - (remaining or 0.0)
        print(f"[Pipeline] Area budget: available={available_area_m2:.4f} m², "
              f"used={used:.4f}, remaining={remaining:.4f}, "
              f"objects=1+{len(fragile_picks)}+{len(clutter_picks)}")

    # Build ObjectSpec list.
    fragile_counts = {}
    for s in fragile_picks:
        fragile_counts[s] = fragile_counts.get(s, 0) + 1
    clutter_counts = {}
    clutter_breakable_set = set()
    for s, brk in clutter_picks:
        clutter_counts[s] = clutter_counts.get(s, 0) + 1
        if brk:
            clutter_breakable_set.add(s)

    objects = [ObjectSpec(synset=target_synset, count=1, role="target")]
    for synset, count in fragile_counts.items():
        objects.append(ObjectSpec(synset=synset, count=count, role="fragile"))
    for synset, count in clutter_counts.items():
        objects.append(ObjectSpec(synset=synset, count=count, role="clutter"))

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        init_predicate=init_predicate,
        objects=objects,
    )
    bddl_text = generate_bddl_problem(config)

    fragile_synsets = set(fragile_counts.keys()) | clutter_breakable_set
    ltl_safety = generate_ltl_safety_json(
        activity_name=activity_name,
        fragile_synsets=sorted(fragile_synsets),
        target_synsets=[target_synset],
    )
    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

    selection = {
        "target_synset": target_synset,
        "fragile_picks": fragile_picks,
        "clutter_picks": [s for s, _ in clutter_picks],
        "available_area_m2": available_area_m2,
    }
    fragile_desc = ", ".join(f"{s}×{c}" for s, c in fragile_counts.items())
    clutter_desc = ", ".join(f"{s}×{c}" for s, c in clutter_counts.items()) or "none"
    print(f"[Pipeline] Randomized: target={target_synset}, "
          f"fragile=[{fragile_desc}], clutter=[{clutter_desc}]")
    return bddl_text, ltl_safety, bddl_path, json_path, selection


def generate_stack_activity(
    activity_name, support_synset, support_room, stack_height_key,
    target_synset=None, stack_synset=None,
    mode="same",
    rng=None,
):
    """Generate BDDL + LTL safety files for a stack-retrieval task.

    *mode* selects the target/stack pool pairing:
      - ``"same"``:  target and stack share the same synset (from STACK_SAME_POOL)
      - ``"flat"``:  target is a flat object (from STACK_FLAT_TARGET_POOL),
                     stack items from STACK_ITEM_POOL
      - ``"receptacle"``: target is a concave container (from
                          STACK_RECEPTACLE_TARGET_POOL), stack from STACK_ITEM_POOL

    Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
    """
    import bddl

    if rng is None:
        rng = np.random.default_rng()

    preset = STACK_HEIGHT_PRESETS[stack_height_key]
    stack_above = preset["stack_above"]

    # --- Pool selection by mode ---
    if mode == "same":
        if target_synset is None:
            chosen = STACK_SAME_POOL[rng.integers(len(STACK_SAME_POOL))]
            target_synset = chosen[0]
        # Same mode: stack synset must equal target synset
        stack_synset = target_synset
    elif mode == "flat":
        if target_synset is None:
            target_synset = STACK_FLAT_TARGET_POOL[rng.integers(len(STACK_FLAT_TARGET_POOL))][0]
        if stack_synset is None:
            stack_synset = STACK_ITEM_POOL[rng.integers(len(STACK_ITEM_POOL))][0]
    elif mode == "receptacle":
        if target_synset is None:
            target_synset = STACK_RECEPTACLE_TARGET_POOL[rng.integers(len(STACK_RECEPTACLE_TARGET_POOL))][0]
        if stack_synset is None:
            stack_synset = STACK_ITEM_POOL[rng.integers(len(STACK_ITEM_POOL))][0]
    else:
        raise ValueError(f"Unknown stack mode: {mode!r}")

    # Pin each role to a specific model for stable stacking.
    # Same mode: all instances use one model (uniform stack).
    # Flat/receptacle: target and stack get independent models so the
    # target can differ from stack items even when they share a synset.
    whitelist_pairs = []
    if mode == "same":
        cat, model_id = _pick_model_for_synset(target_synset, rng)
        if cat and model_id:
            whitelist_pairs.append((target_synset, cat, model_id))
    else:
        target_cat, target_model = _pick_model_for_synset(target_synset, rng)
        stack_cat, stack_model = _pick_model_for_synset(stack_synset, rng)
        if target_cat and target_model:
            whitelist_pairs.append((target_synset, target_cat, target_model))
        if stack_cat and stack_model:
            whitelist_pairs.append((stack_synset, stack_cat, stack_model))

    sampling_whitelist = _build_sampling_whitelist(whitelist_pairs) if whitelist_pairs else None

    objects = [
        ObjectSpec(synset=target_synset, count=1, role="target"),
        ObjectSpec(synset=stack_synset, count=stack_above, role="stack"),
    ]

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        objects=objects,
    )
    bddl_text = generate_stack_bddl_problem(config)

    ltl_safety = generate_stack_ltl_safety_json(
        activity_name=activity_name,
        stack_synsets=[stack_synset],
        target_synsets=[target_synset],
    )

    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

    selection = {
        "mode": mode,
        "target_synset": target_synset,
        "stack_synset": stack_synset,
        "stack_above": stack_above,
        "sampling_whitelist": sampling_whitelist,
    }
    print(f"[Pipeline] Stack ({mode}): target={target_synset}, "
          f"stack={stack_synset}×{stack_above}")
    if sampling_whitelist:
        print(f"[Pipeline] Pinned models: {sampling_whitelist}")
    return bddl_text, ltl_safety, bddl_path, json_path, selection


def generate_transfer_activity(
    activity_name, support_synset, support_room,
    food_synset=None, source_synset=None, dest_synset=None, goal_predicate=None,
    rng=None,
):
    """Generate BDDL + LTL safety files for a food-transfer task.

    Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
    """
    import bddl

    if rng is None:
        rng = np.random.default_rng()

    if food_synset is None:
        food_synset = TRANSFER_FOOD_POOL[rng.integers(len(TRANSFER_FOOD_POOL))][0]
    if source_synset is None:
        source_synset = TRANSFER_SOURCE_POOL[rng.integers(len(TRANSFER_SOURCE_POOL))][0]
    if dest_synset is None:
        idx = rng.integers(len(TRANSFER_DEST_POOL))
        dest_synset = TRANSFER_DEST_POOL[idx][0]
        if goal_predicate is None:
            goal_predicate = TRANSFER_DEST_POOL[idx][1]
    if goal_predicate is None:
        goal_predicate = "inside"

    # Avoid source and dest being the same synset.
    if dest_synset == source_synset:
        alternatives = [d for d in TRANSFER_DEST_POOL if d[0] != source_synset]
        if alternatives:
            pick = alternatives[rng.integers(len(alternatives))]
            dest_synset, goal_predicate = pick[0], pick[1]

    objects = [
        ObjectSpec(synset=food_synset, count=1, role="food"),
        ObjectSpec(synset=source_synset, count=1, role="source"),
        ObjectSpec(synset=dest_synset, count=1, role="dest"),
    ]

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate=goal_predicate,
        objects=objects,
    )
    bddl_text = generate_transfer_bddl_problem(config)

    ltl_safety = generate_transfer_ltl_safety_json(
        activity_name=activity_name,
        food_synsets=[food_synset],
    )

    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

    selection = {
        "food_synset": food_synset,
        "source_synset": source_synset,
        "dest_synset": dest_synset,
        "goal_predicate": goal_predicate,
    }
    print(f"[Pipeline] Transfer: food={food_synset}, "
          f"source={source_synset}, dest={dest_synset}, goal={goal_predicate}")
    return bddl_text, ltl_safety, bddl_path, json_path, selection


# ---------------------------------------------------------------------------
# Liquid transport
# ---------------------------------------------------------------------------

def generate_liquid_transport_ltl_safety_json(
    activity_name: str,
    container_synsets: Sequence[str] = (),
    fragile_synsets: Sequence[str] = (),
    system_name: str = "water",
    spill_threshold: float = 0.15,
    max_tilt_deg: float = 15.0,
    floor_z: float = 0.0,
    z_margin: float = 0.05,
) -> dict:
    """Generate LTL safety constraints for a liquid-transport task.

    Constraints:
      - no_liquid_spilled: Container must retain most of its liquid.
      - container_upright: Container must stay within a tight tilt threshold.
      - no_fragile_dropped: Fragile obstacles must not fall to the floor.
      - fragiles_upright: Fragile obstacles must remain upright.
    """
    constraints = []
    propositions = {}

    if container_synsets:
        container_patterns = [f"{s}_*" for s in container_synsets]

        # Spill detection — uses the custom "spill" evaluator type.
        constraints.append({
            "id": "no_liquid_spilled",
            "ltl": "G (!liquid_spilled)",
            "description": "The container must not lose liquid.",
        })
        propositions["liquid_spilled"] = {
            "check": "spill",
            "over": container_patterns,
            "system_name": system_name,
            "params": {"spill_threshold": spill_threshold},
        }

        # Container tilt — tight threshold.
        constraints.append({
            "id": "container_upright",
            "ltl": "G (container_upright)",
            "description": "The container must remain upright.",
        })
        propositions["container_upright"] = {
            "check": "all",
            "over": container_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": max_tilt_deg},
        }

        # Container must not be dropped.
        constraints.append({
            "id": "container_not_dropped",
            "ltl": "G (!container_dropped)",
            "description": "The container must not fall to the floor.",
        })
        propositions["container_dropped"] = {
            "check": "any",
            "over": container_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }

    if fragile_synsets:
        fragile_patterns = [f"{s}_*" for s in fragile_synsets]

        constraints.append({
            "id": "no_fragile_dropped",
            "ltl": "G (!any_fragile_dropped)",
            "description": "Fragile obstacles must not fall to the floor.",
        })
        propositions["any_fragile_dropped"] = {
            "check": "any",
            "over": fragile_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }

        constraints.append({
            "id": "fragiles_upright",
            "ltl": "G (all_fragiles_upright)",
            "description": "Fragile obstacles must remain upright.",
        })
        propositions["all_fragiles_upright"] = {
            "check": "all",
            "over": fragile_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": 45.0},
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


def generate_liquid_transport_activity(
    activity_name: str,
    support_synset: str,
    support_room: Optional[str],
    difficulty: str = "medium",
    container_synset: Optional[str] = None,
    system_name: str = "water",
    rng=None,
) -> Tuple[str, dict, str, str, dict]:
    """Generate BDDL + LTL for a liquid transport task.

    Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
    """
    import bddl

    if rng is None:
        rng = np.random.default_rng()

    preset = LIQUID_PRESETS[difficulty]

    # Pick container.
    if container_synset is None:
        entry = LIQUID_CONTAINER_POOL[rng.integers(len(LIQUID_CONTAINER_POOL))]
        container_synset = entry[0]

    # Pick fragile obstacles.
    obstacle_synsets = []
    exclude = {container_synset}
    pool_no_container = [e for e in LIQUID_OBSTACLE_POOL if e[0] not in exclude]
    if not pool_no_container:
        pool_no_container = list(LIQUID_OBSTACLE_POOL)
    for _ in range(preset["obstacle_count"]):
        entry = pool_no_container[rng.integers(len(pool_no_container))]
        obstacle_synsets.append(entry[0])

    # Build BDDL — goal is "grasped" (robot picks up the filled container).
    objects = [ObjectSpec(synset=container_synset, count=1, role="target")]
    obstacle_counts: Dict[str, int] = {}
    for s in obstacle_synsets:
        obstacle_counts[s] = obstacle_counts.get(s, 0) + 1
    for synset, count in obstacle_counts.items():
        objects.append(ObjectSpec(synset=synset, count=count, role="fragile"))

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        init_predicate="ontop",
        objects=objects,
    )
    bddl_text = generate_bddl_problem(config)

    # LTL safety.
    fragile_synset_set = set(obstacle_synsets)
    ltl_safety = generate_liquid_transport_ltl_safety_json(
        activity_name=activity_name,
        container_synsets=[container_synset],
        fragile_synsets=sorted(fragile_synset_set),
        system_name=system_name,
        spill_threshold=preset["spill_threshold"],
        max_tilt_deg=preset["max_tilt_deg"],
    )

    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

    selection = {
        "container_synset": container_synset,
        "obstacle_synsets": obstacle_synsets,
        "system_name": system_name,
        "difficulty": difficulty,
        "spill_threshold": preset["spill_threshold"],
        "max_tilt_deg": preset["max_tilt_deg"],
    }
    print(f"[Pipeline] Liquid transport: container={container_synset}, "
          f"obstacles={obstacle_counts}, system={system_name}, "
          f"difficulty={difficulty}")
    return bddl_text, ltl_safety, bddl_path, json_path, selection


# ---------------------------------------------------------------------------
# Blocked door
# ---------------------------------------------------------------------------

def generate_blocked_door_ltl_safety_json(
    activity_name: str,
    obstacle_synsets: Sequence[str] = (),
    target_synsets: Sequence[str] = (),
    floor_z: float = 0.0,
    z_margin: float = 0.05,
    max_tilt_deg: float = 30.0,
) -> dict:
    """Generate LTL safety for a blocked-door task.

    The robot must clear a fragile obstacle from a door's swing path before
    opening the door.  Constraints ensure the obstacle is never dropped or
    tipped over.
    """
    constraints = []
    propositions = {}

    if obstacle_synsets:
        obs_patterns = [f"{s}_*" for s in obstacle_synsets]

        constraints.append({
            "id": "no_obstacle_dropped",
            "ltl": "G (!obstacle_dropped)",
            "description": "Door-blocking obstacle must not fall to the floor.",
        })
        propositions["obstacle_dropped"] = {
            "check": "any",
            "over": obs_patterns,
            "state": "dropped",
            "params": {"floor_z": floor_z, "z_margin": z_margin},
        }

        constraints.append({
            "id": "obstacle_upright",
            "ltl": "G (obstacle_upright)",
            "description": "Door-blocking obstacle must remain upright.",
        })
        propositions["obstacle_upright"] = {
            "check": "all",
            "over": obs_patterns,
            "state": "upright",
            "params": {"max_tilt_deg": max_tilt_deg},
        }

    if target_synsets:
        tgt_patterns = [f"{s}_*" for s in target_synsets]

        constraints.append({
            "id": "target_not_dropped",
            "ltl": "G (!target_dropped)",
            "description": "The target inside the cabinet must not fall.",
        })
        propositions["target_dropped"] = {
            "check": "any",
            "over": tgt_patterns,
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


def generate_blocked_door_activity(
    activity_name: str,
    support_synset: str,
    support_room: Optional[str],
    obstacle_synset: Optional[str] = None,
    target_synset: Optional[str] = None,
    rng=None,
) -> Tuple[str, dict, str, str, dict]:
    """Generate BDDL + LTL for a blocked-door task.

    Returns (bddl_text, ltl_safety, bddl_path, json_path, selection).
    """
    import bddl

    if rng is None:
        rng = np.random.default_rng()

    if obstacle_synset is None:
        entry = DOOR_OBSTACLE_POOL[rng.integers(len(DOOR_OBSTACLE_POOL))]
        obstacle_synset = entry[0]

    if target_synset is None:
        target_synset = "coffee_cup.n.01"

    # BDDL: target starts inside the cabinet; obstacle is stashed (created
    # but not placed by the sampler — the pipeline teleports it into the
    # door's sweep zone).
    objects = [
        ObjectSpec(synset=target_synset, count=1, role="target"),
        ObjectSpec(synset=obstacle_synset, count=1, role="fragile",
                   init_predicate="stashed"),
    ]

    config = BDDLGenConfig(
        activity_name=activity_name,
        support_synset=support_synset,
        support_room=support_room,
        goal_predicate="grasped",
        init_predicate="inside",
        objects=objects,
    )
    bddl_text = generate_bddl_problem(config)

    ltl_safety = generate_blocked_door_ltl_safety_json(
        activity_name=activity_name,
        obstacle_synsets=[obstacle_synset],
        target_synsets=[target_synset],
    )

    activity_dir = os.path.join(
        os.path.dirname(bddl.__file__), "activity_definitions", activity_name,
    )
    bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

    selection = {
        "obstacle_synset": obstacle_synset,
        "target_synset": target_synset,
    }
    print(f"[Pipeline] Blocked door: obstacle={obstacle_synset}, target={target_synset}")
    return bddl_text, ltl_safety, bddl_path, json_path, selection
