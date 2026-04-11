#!/usr/bin/env python3
"""
Repair/rebuild diagnostics.jsonl files in a benchmark directory
to match the current SentinelEnv registry format.

Reads scene_ep1.json + existing diagnostics + BDDL problem files,
then rebuilds active_object_summary with correct inst_id → scene_object_name mappings.

Usage:
    python tools/repair_benchmark_diagnostics.py \
        --benchmark-root outputs/local_eval_benchmark/clutter_all_scene_20260319 \
        --activity-root bddl3/bddl/activity_definitions
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Repair benchmark diagnostics.")
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--activity-root", required=True)
    p.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    return p.parse_args()


def load_scene_objects(scene_file: Path) -> dict:
    """Load scene objects: name → category mapping."""
    scene_info = json.loads(scene_file.read_text(encoding="utf-8"))
    init_info = scene_info.get("objects_info", {}).get("init_info", {})
    objects = {}
    for name, obj_info in init_info.items():
        category = obj_info.get("args", {}).get("category")
        class_module = str(obj_info.get("class_module", ""))
        # Skip robots
        if "robot" in class_module.lower():
            continue
        if category:
            objects[name] = category
    return objects


def parse_bddl_instances(problem_file: Path) -> list[str]:
    """Extract BDDL instance IDs from problem file."""
    text = problem_file.read_text(encoding="utf-8")
    pattern = re.compile(r"[A-Za-z0-9_.-]+\.[a-z]\.\d+_\d+")
    return list(dict.fromkeys(pattern.findall(text)))


def load_synset_to_categories(activity_root: Path) -> dict[str, list[str]]:
    """Load synset → [category, ...] mapping from bddl3 category_mapping.csv."""
    csv_path = activity_root.parent / "generated_data" / "category_mapping.csv"
    mapping: dict[str, list[str]] = {}
    if csv_path.is_file():
        import csv
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                synset = row.get("synset", "").strip()
                category = row.get("category", "").strip()
                if synset and category:
                    mapping.setdefault(synset, []).append(category)
    return mapping


def parse_bddl_target_from_goal(problem_file: Path) -> str | None:
    """Try to extract target instance from BDDL goal (usually the grasped object)."""
    text = problem_file.read_text(encoding="utf-8")
    # Look for (inhand ?target agent) or similar patterns
    grasp_match = re.search(r"\((?:inhand|holding|grasping)\s+(\S+)\s+", text)
    if grasp_match:
        return grasp_match.group(1)
    return None


def build_active_object_summary(
    scene_objects: dict[str, str],
    bddl_instances: list[str],
    diagnostics: dict,
    synset_to_categories: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Build active_object_summary by matching BDDL instances to scene objects."""
    synset_to_categories = synset_to_categories or {}

    # Use existing summary as base if available
    existing = {
        entry["inst_id"]: entry
        for entry in diagnostics.get("active_object_summary", [])
        if "inst_id" in entry
    }

    # Build category → scene_names mapping
    cat_to_names: dict[str, list[str]] = {}
    for name, cat in scene_objects.items():
        cat_to_names.setdefault(cat, []).append(name)

    # Track used scene names to avoid duplicates
    used_names: set[str] = set()
    for entry in existing.values():
        if "scene_object_name" in entry:
            used_names.add(entry["scene_object_name"])

    summary = []
    target_synset = diagnostics.get("selection", {}).get("target_synset", "")
    surface_name = diagnostics.get("surface", "")

    for inst_id in bddl_instances:
        if inst_id.startswith("agent.n."):
            continue

        # Use existing entry if available
        if inst_id in existing:
            summary.append(existing[inst_id])
            continue

        # Match by category, then by synset → category mapping
        synset = ".".join(inst_id.rsplit("_", 1)[0].split(".")[:3]) if ".n." in inst_id else ""
        category = inst_id.split(".n.")[0] if ".n." in inst_id else inst_id.rsplit("_", 1)[0]

        # Try direct category match first
        candidates = [n for n in cat_to_names.get(category, []) if n not in used_names]

        # If no match, try all categories that map to the same synset
        if not candidates and synset and synset in synset_to_categories:
            for alt_category in synset_to_categories[synset]:
                candidates = [n for n in cat_to_names.get(alt_category, []) if n not in used_names]
                if candidates:
                    category = alt_category  # use the matched category
                    break

        if not candidates:
            # Try matching surface
            if surface_name and surface_name not in used_names:
                surface_cat = scene_objects.get(surface_name, "")
                if surface_cat == category:
                    candidates = [surface_name]

        if candidates:
            scene_name = candidates[0]
            used_names.add(scene_name)

            # Determine role
            role = "fragile"
            if target_synset and inst_id.startswith(target_synset.split(".")[0]):
                # Check if this is the target synset
                inst_synset = ".".join(inst_id.rsplit("_", 1)[0].split(".")[:3]) if ".n." in inst_id else ""
                if inst_synset == target_synset:
                    role = "target"

            summary.append({
                "inst_id": inst_id,
                "scene_object_name": scene_name,
                "category": category,
                "role": role,
            })

    # Make sure target is marked — check all categories that map to target synset
    if target_synset and not any(e.get("role") == "target" for e in summary):
        target_cats = set()
        target_cats.add(target_synset.split(".n.")[0] if ".n." in target_synset else target_synset)
        if target_synset in synset_to_categories:
            target_cats.update(synset_to_categories[target_synset])
        # First try to mark an existing entry
        found_target = False
        for entry in summary:
            if entry.get("category") in target_cats:
                entry["role"] = "target"
                found_target = True
                break
        # If not in summary yet, find in scene objects and add (allow reuse)
        if not found_target:
            for obj_name, obj_cat in scene_objects.items():
                if obj_cat in target_cats:
                    summary.append({
                        "inst_id": f"{target_synset}_1",
                        "scene_object_name": obj_name,
                        "category": obj_cat,
                        "role": "target",
                    })
                    found_target = True
                    break

    return summary


def repair_scene(scene_dir: Path, activity_root: Path, dry_run: bool, synset_to_categories: dict | None = None) -> dict:
    """Repair diagnostics for one scene. Returns status dict."""
    scene_name = scene_dir.name
    scene_file = scene_dir / "scene_ep1.json"
    diag_file = scene_dir / "diagnostics.jsonl"

    if not scene_file.is_file() or not diag_file.is_file():
        return {"scene": scene_name, "status": "missing_files"}

    # Load existing diagnostics
    with diag_file.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return {"scene": scene_name, "status": "empty_diagnostics"}

    diagnostics = json.loads(lines[0])
    activity_name = diagnostics.get("activity_name", "")

    # Check BDDL problem file
    problem_file = activity_root / activity_name / "problem0.bddl"
    if not problem_file.is_file():
        return {"scene": scene_name, "status": "missing_bddl", "activity": activity_name}

    # Load scene objects
    scene_objects = load_scene_objects(scene_file)
    bddl_instances = parse_bddl_instances(problem_file)

    # Check if repair is needed
    existing_summary = diagnostics.get("active_object_summary", [])
    has_target = any(e.get("role") == "target" for e in existing_summary)

    if has_target and len(existing_summary) >= len([i for i in bddl_instances if not i.startswith("agent.n.")]):
        return {"scene": scene_name, "status": "ok"}

    # Build repaired summary
    new_summary = build_active_object_summary(scene_objects, bddl_instances, diagnostics, synset_to_categories)
    new_has_target = any(e.get("role") == "target" for e in new_summary)

    if not new_has_target:
        return {
            "scene": scene_name,
            "status": "no_target_found",
            "instances": len(bddl_instances),
            "matched": len(new_summary),
        }

    # Update diagnostics
    diagnostics["active_object_summary"] = new_summary

    if not dry_run:
        with diag_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(diagnostics, ensure_ascii=True) + "\n")

    return {
        "scene": scene_name,
        "status": "repaired",
        "objects_matched": len(new_summary),
        "target": next((e["scene_object_name"] for e in new_summary if e.get("role") == "target"), None),
    }


def main():
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    activity_root = Path(args.activity_root)

    if not benchmark_root.is_dir():
        print(f"Benchmark root not found: {benchmark_root}")
        return

    scene_dirs = sorted(p for p in benchmark_root.iterdir() if p.is_dir())
    print(f"Found {len(scene_dirs)} scene directories")

    synset_map = load_synset_to_categories(activity_root)
    print(f"Loaded {len(synset_map)} synset → category mappings")

    results = {"ok": 0, "repaired": 0, "failed": 0}
    for scene_dir in scene_dirs:
        status = repair_scene(scene_dir, activity_root, args.dry_run, synset_map)
        s = status["status"]
        prefix = "  ✓" if s == "ok" else "  ✓ REPAIRED" if s == "repaired" else "  ✗"
        print(f"{prefix} {status['scene']}: {s}")
        if "target" in status:
            print(f"      target: {status['target']}")

        if s == "ok":
            results["ok"] += 1
        elif s == "repaired":
            results["repaired"] += 1
        else:
            results["failed"] += 1

    print(f"\nSummary: {results['ok']} ok, {results['repaired']} repaired, {results['failed']} failed")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
