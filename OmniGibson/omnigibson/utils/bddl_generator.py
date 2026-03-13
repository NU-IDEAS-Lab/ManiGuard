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
    support_room: str
    goal_synset: str
    goal_room: str
    goal_predicate: str = "inside"
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

    # Ensure support and goal synsets have at least one instance.
    if config.support_synset not in synset_instances:
        synset_instances[config.support_synset] = [f"{config.support_synset}_1"]
    if config.goal_synset not in synset_instances:
        synset_instances[config.goal_synset] = [f"{config.goal_synset}_1"]

    lines.append("    (:objects")
    for synset, instances in synset_instances.items():
        inst_str = " ".join(instances)
        lines.append(f"        {inst_str} - {synset}")
    lines.append("    )")
    lines.append("")

    # Init: place all non-support, non-goal objects ontop the support.
    support_inst = synset_instances[config.support_synset][0]
    goal_inst = synset_instances[config.goal_synset][0]

    lines.append("    (:init")
    for synset, instances in synset_instances.items():
        if synset == config.support_synset or synset == config.goal_synset:
            continue
        for inst in instances:
            lines.append(f"        (ontop {inst} {support_inst})")
    lines.append(f"        (inroom {support_inst} {config.support_room})")
    lines.append(f"        (inroom {goal_inst} {config.goal_room})")
    lines.append("    )")
    lines.append("")

    # Goal: first target object goes to goal destination.
    target_specs = [s for s in config.objects if s.role == "target"]
    if target_specs:
        target_inst = f"{target_specs[0].synset}_1"
    else:
        # Fallback: first non-support, non-goal object.
        for synset, instances in synset_instances.items():
            if synset not in (config.support_synset, config.goal_synset):
                target_inst = instances[0]
                break
        else:
            target_inst = support_inst

    lines.append("    (:goal")
    lines.append("        (and")
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
