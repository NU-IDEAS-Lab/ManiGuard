#!/usr/bin/env python
"""Inspect a curated scene snapshot without regenerating the scene."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


_CATEGORY_ALIASES = {
    "chalice": "goblet",
    "cocktail_glass": "goblet",
}


def make_parser():
    parser = argparse.ArgumentParser(description="Inspect a curated clutter scene snapshot")
    parser.add_argument("--scene-model", required=True)
    parser.add_argument("--curation-manifest", required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument("--diagnostics-path", default=None)
    parser.add_argument("--support-name", default=None)
    parser.add_argument("--report-json", default=None)
    return parser


def build_expected_category_counts(selection: dict[str, Any] | None) -> Counter:
    counts: Counter = Counter()
    if not selection:
        return counts
    target = selection.get("target_synset")
    if target:
        counts[_normalize_category(_synset_to_category(target))] += 1
    for synset in list(selection.get("fragile_picks", ())) + list(selection.get("clutter_picks", ())):
        counts[_normalize_category(_synset_to_category(synset))] += 1
    return counts


def counter_missing(expected: Counter, actual: Counter) -> dict[str, int]:
    missing = {}
    for key, expected_count in expected.items():
        actual_count = int(actual.get(key, 0))
        if actual_count < expected_count:
            missing[key] = expected_count - actual_count
    return missing


def choose_diagnostics_entry(path: str | None, episode: int | None) -> dict[str, Any] | None:
    if not path:
        return None
    diag_path = Path(path)
    if not diag_path.is_file():
        return None
    entries = []
    for line in diag_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    if not entries:
        return None
    if episode is None:
        return entries[-1]
    for entry in entries:
        if entry.get("episode") == episode:
            return entry
    return entries[-1]


def _synset_to_category(synset: str) -> str:
    return synset.split(".", 1)[0]


def _normalize_category(category: str) -> str:
    return _CATEGORY_ALIASES.get(category, category)


def _resolve_snapshot_path(entry, args) -> str:
    if args.snapshot_path:
        return str(Path(args.snapshot_path).resolve())
    snapshot_path = entry.resolve_snapshot_path(episode=args.episode)
    if snapshot_path is None:
        raise RuntimeError(f"No snapshot found for scene '{args.scene_model}' episode={args.episode!r}")
    return snapshot_path


def _resolve_diagnostics_path(entry, args, snapshot_path: str) -> str | None:
    if args.diagnostics_path:
        return str(Path(args.diagnostics_path).resolve())
    sibling = Path(snapshot_path).with_name("diagnostics.jsonl")
    if sibling.is_file():
        return str(sibling.resolve())
    candidate = Path(entry.scene_dir) / "diagnostics.jsonl"
    if candidate.is_file():
        return str(candidate.resolve())
    return None


def _scene_category_counts(snapshot_path: str) -> Counter:
    data = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    init_info = data.get("objects_info", {}).get("init_info", {})
    counts: Counter = Counter()
    for info in init_info.values():
        args = info.get("args", {})
        if isinstance(args, dict) and args.get("category"):
            counts[_normalize_category(str(args["category"]))] += 1
    return counts


def _build_scene_only_config(scene_model: str, snapshot_path: str) -> dict[str, Any]:
    return {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "scene_file": snapshot_path,
            "scene_instance": None,
        },
        "task": {
            "type": "DummyTask",
        },
        "robots": [],
    }


def _compute_scene_floor_z(env) -> float:
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


def _inspect_loaded_snapshot(env, entry, snapshot_path: str, diagnostics_entry, support_name_override=None) -> dict[str, Any]:
    from omnigibson.object_states import OnTop
    from omnigibson.task_generation.pipeline_common import (
        discover_best_surface,
    )
    from omnigibson.utils.surface_discovery import is_table_like

    expected = build_expected_category_counts((diagnostics_entry or {}).get("selection"))
    expected_categories = set(expected)

    support_name = support_name_override
    if support_name is None and diagnostics_entry is not None:
        support_name = diagnostics_entry.get("surface")
    if support_name is None:
        support_name = entry.surface_name

    support_obj = None
    surface_info = None
    if support_name:
        support_obj = env.scene.object_registry("name", support_name)
    if support_obj is None:
        forced_name = entry.surface_name if entry.surface_name else None
        forced_category = entry.support_category if entry.support_category else None
        surface_info, support_obj = discover_best_surface(env, forced_name=forced_name, forced_category=forced_category)
        support_name = getattr(support_obj, "name", support_name)

    if support_obj is None:
        raise RuntimeError(f"Failed to resolve support surface for scene '{entry.scene_model}'")

    if surface_info is None:
        try:
            surface_info, _ = discover_best_surface(env, forced_name=support_name, forced_category=None)
        except Exception:
            surface_info = None

    floor_z = _compute_scene_floor_z(env)
    support_min, support_max = support_obj.aabb
    support_top_z = float(support_max[2])
    support_bounds_xy = (
        (float(support_min[0]), float(support_min[1])),
        (float(support_max[0]), float(support_max[1])),
    )

    candidate_surfaces = []
    for obj in env.scene.objects:
        category = str(getattr(obj, "category", ""))
        if not is_table_like(category):
            continue
        if getattr(obj, "name", None) == support_name:
            continue
        candidate_surfaces.append(obj)

    actual_scene_categories: Counter = Counter()
    task_objects = []
    for obj in env.scene.objects:
        category = _normalize_category(str(getattr(obj, "category", "")))
        if expected_categories and category not in expected_categories:
            continue
        if bool(getattr(obj, "fixed_base", False)):
            continue
        try:
            aabb_min, aabb_max = obj.aabb
        except Exception:
            continue
        actual_scene_categories[category] += 1
        bottom_z = float(aabb_min[2])
        top_z = float(aabb_max[2])
        on_support = False
        try:
            on_support = bool(obj.states[OnTop].get_value(support_obj))
        except Exception:
            on_support = False
        other_support = None
        if not on_support:
            for candidate in candidate_surfaces:
                try:
                    if obj.states[OnTop].get_value(candidate):
                        other_support = getattr(candidate, "name", None)
                        break
                except Exception:
                    continue
        floor_suspect = bottom_z <= floor_z + 0.05
        if on_support:
            status = "on_support"
        elif other_support:
            status = "wrong_surface_suspect"
        elif floor_suspect:
            status = "floor_suspect"
        else:
            status = "off_support"
        task_objects.append(
            {
                "scene_object_name": getattr(obj, "name", None),
                "category": category,
                "status": status,
                "on_support": on_support,
                "other_support_name": other_support,
                "bottom_z": bottom_z,
                "top_z": top_z,
                "center_xy": [
                    0.5 * (float(aabb_min[0]) + float(aabb_max[0])),
                    0.5 * (float(aabb_min[1]) + float(aabb_max[1])),
                ],
            }
        )

    scene_counts = _scene_category_counts(snapshot_path)

    return {
        "scene_model": entry.scene_model,
        "activity_name": entry.activity_name,
        "support_surface": {
            "name": support_name,
            "category": str(getattr(support_obj, "category", "")),
            "top_z": support_top_z,
            "bounds_xy": support_bounds_xy,
            "score": getattr(getattr(surface_info, "surface", None), "score", None),
        },
        "floor_z": floor_z,
        "diagnostics": diagnostics_entry,
        "expected_category_counts": dict(expected),
        "actual_scene_category_counts": dict(actual_scene_categories),
        "scene_category_counts_subset": {key: int(scene_counts.get(key, 0)) for key in expected},
        "missing_from_loaded_scene": counter_missing(expected, actual_scene_categories),
        "missing_from_scene_snapshot": counter_missing(expected, scene_counts),
        "task_objects": task_objects,
    }


def main():
    parser = make_parser()
    args = parser.parse_args()

    from omnigibson.task_generation.curation.curation_manifest import load_curation_manifest
    from omnigibson.task_generation.pipeline_common import pipeline_exit
    import omnigibson as og
    from omnigibson.macros import gm

    exit_code = 0
    manifest = load_curation_manifest(args.curation_manifest)
    entry = manifest.get_scene_entry(args.scene_model)
    episode = args.episode if args.episode is not None else entry.canonical_episode
    snapshot_path = _resolve_snapshot_path(entry, args)
    diagnostics_path = _resolve_diagnostics_path(entry, args, snapshot_path)
    diagnostics_entry = choose_diagnostics_entry(diagnostics_path, episode)

    print(f"[Inspect] Manifest: {manifest.source_path}")
    print(f"[Inspect] Scene: {args.scene_model} episode={episode!r}")
    print(f"[Inspect] Snapshot: {snapshot_path}")
    if diagnostics_path:
        print(f"[Inspect] Diagnostics: {diagnostics_path}")

    gm.ENABLE_OBJECT_STATES = True
    cfg = _build_scene_only_config(args.scene_model, snapshot_path)

    env = og.Environment(configs=cfg)
    try:
        env.reset()
        og.sim.step()
        report = _inspect_loaded_snapshot(
            env,
            entry,
            snapshot_path=snapshot_path,
            diagnostics_entry=diagnostics_entry,
            support_name_override=args.support_name,
        )
        print(json.dumps(report, indent=2))
        if args.report_json:
            report_path = Path(args.report_json).resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"[Inspect] Report written: {report_path}")
    except Exception:
        exit_code = 1
        raise
    finally:
        print("[Inspect] Shutdown simulator.")
        pipeline_exit(exit_code)


if __name__ == "__main__":
    main()
