"""Lightweight scene discovery — no OmniGibson / torch / imageio deps.

Importable from any Python env (system, conda, venv) to list benchmark
scenes without triggering heavy GPU/simulation imports. Used by both
benchmark.py and scripts/run_benchmark_all_scenes.sh.
"""

from __future__ import annotations

import json
from pathlib import Path
import logging

from maniguard.utils.goal_region import build_task_prompt

log = logging.getLogger(__name__)


def _iter_scene_dirs(root: Path):
    """Yield every directory under ``root`` that looks like a scene."""
    if not root.is_dir():
        return
    stack = [(root, 0)]
    max_depth = 3
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError as exc:
            log.warning("scene dir iteration failed for %s: %s", current, exc)
            continue
        is_scene = (current / "scene_ep1.json").is_file() and (current / "diagnostics.jsonl").is_file()
        if is_scene:
            yield current
            continue
        if depth >= max_depth:
            continue
        for entry in entries:
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                stack.append((entry, depth + 1))


def _scene_key(root: Path, scene_dir: Path) -> str:
    try:
        rel = scene_dir.relative_to(root)
    except ValueError:
        return scene_dir.name
    return rel.as_posix()


STRUCTURAL_CATEGORIES = {"walls", "floors", "ceilings", "door", "window"}


def _match_category(init_info: dict, category: str):
    """First scene-object name whose category == ``category`` (else None)."""
    if not category:
        return None
    for obj_name, obj_info in init_info.items():
        if obj_info.get("args", {}).get("category") == category:
            return obj_name
    return None


def _category_from_synset(synset: str) -> str:
    """'potato.n.01' -> 'potato'; '' or a non-synset string -> ''."""
    return synset.split(".n.")[0] if ".n." in synset else ""


def discover_scenes(benchmark_root: str, scene_names=None, max_scenes=None):
    """Discover valid benchmark scenes with diagnostics.

    Handles both 1-level layouts (``<root>/<scene>/``) and 2-level
    layouts (``<root>/<task_family>/<scene>/``).

    Returns a list of dicts, each with keys: name, scene_file,
    scene_model, target_name, surface_name, target_rooms, prompt,
    pipeline, activity_name.
    """
    root = Path(benchmark_root)
    scenes = []
    for scene_dir in _iter_scene_dirs(root):
        scene_file = scene_dir / "scene_ep1.json"
        diag_file = scene_dir / "diagnostics.jsonl"
        scene_key = _scene_key(root, scene_dir)
        # Match a requested name exactly ("task_0000/base"), as a task-dir
        # prefix ("task_0000" -> task_0000/base, task_0000/env), or by leaf
        # dir name — so `--scenes task_0000` selects a single task ergonomically.
        if scene_names and not any(
            scene_key == n or scene_key.startswith(f"{n}/") or scene_dir.name == n
            for n in scene_names
        ):
            continue

        # diagnostics.jsonl holds one JSON record; it may be a single line OR a
        # pretty-printed multi-line object (older pipelines, e.g. dusty). Decode
        # the first complete JSON value rather than assuming a single line.
        diag = json.JSONDecoder().raw_decode(
            diag_file.read_text(encoding="utf-8").lstrip()
        )[0]
        scene_info_json = json.loads(scene_file.read_text(encoding="utf-8"))
        init_info = scene_info_json.get("objects_info", {}).get("init_info", {})
        sel = diag.get("selection", {})
        pipeline = diag.get("pipeline", "")
        surface_name = diag.get("surface", "")
        surface_label = (
            init_info.get(surface_name, {}).get("args", {}).get("category", "")
            or surface_name or "table"
        )

        target_name = None
        prompt = str(diag.get("prompt") or "").strip() or None

        # Resolve the manipulation target per 6fam-base family, keyed on the
        # diagnostics `pipeline` field. Families with sub-variants (clutter,
        # lid, stack) group their variant names into one branch; the names do
        # not overlap across families. else = safety skip only.
        if pipeline == "dusty_transfer":
            # Target = transferred food. The clean dustify batch carries
            # food_synset (+ a baked prompt); the merged food_transfer remnants
            # have empty categories / no synset / no prompt — skip them
            # (see incomplete_source_note.txt).
            food_synset = sel.get("food_synset", "")
            if not food_synset:
                print(f"  Skipping {scene_key}: dusty degraded merge batch (no synset/prompt)")
                continue
            target_name = _match_category(init_info, _category_from_synset(food_synset))

        elif pipeline == "jar_transport":
            # Target = the jar (lid closed, then carried).
            target_name = _match_category(init_info, sel.get("jar_category", ""))

        elif pipeline == "cabinet_pickup":
            # Target = the object placed into the drawer.
            target_name = _match_category(init_info, sel.get("target_category", ""))

        elif pipeline in ("liquid_transport", "table"):
            # clutter: pick the target out of the clutter (a liquid-filled
            # container, or a plain table pick). Both carry target_synset.
            target_name = _match_category(init_info, _category_from_synset(sel.get("target_synset", "")))

        elif pipeline in ("lid_transport_food", "lid_transport_liquid"):
            # lid: place the lid on the container, then move the container.
            # liquid-mode diags carry a STALE selection.container_category (the
            # pre-respawn pick, e.g. "can"); the actually spawned container is
            # the spawn_specs entry with role=="target" (e.g. hingeless_jar) —
            # prefer it, fall back to container_category (the food-mode truth;
            # food diags have no role=="target" spawn spec).
            _tgt_spec = next(
                (s for s in (sel.get("spawn_specs") or []) if s.get("role") == "target"),
                None,
            )
            target_name = None
            if _tgt_spec:
                target_name = _match_category(init_info, _tgt_spec.get("category", ""))
            if not target_name:
                target_name = _match_category(init_info, sel.get("container_category", ""))

        elif pipeline in ("stack_same", "stack_flat"):
            # stack: retrieve the bottom object from the stack.
            target_name = _match_category(init_info, _category_from_synset(sel.get("target_synset", "")))

        else:
            # Safety: a pipeline that belongs to no 6fam-base family.
            print(f"  Skipping {scene_key}: unrecognized pipeline '{pipeline}' (not a 6fam-base family)")
            continue

        if not target_name:
            print(f"  Skipping {scene_key}: could not resolve target object (pipeline={pipeline})")
            continue
        # Every 6fam-base family bakes its prompt into diagnostics; rebuild from
        # task fields only as a fallback so a missing prompt stays in-distribution
        # rather than dropping to the generic surface line.
        if prompt is None:
            try:
                prompt = build_task_prompt(scene_info_json, diag, goal_region=diag.get("goal_region"))
            except Exception:
                prompt = None
        if not prompt:
            prompt = f"Manipulate the objects on the {surface_label}."

        target_rooms = set()
        diag_room = diag.get("support_selection", {}).get("room_instance")
        if diag_room:
            target_rooms.add(diag_room)
        target_obj_info = init_info.get(target_name, {})
        target_rooms.update(target_obj_info.get("args", {}).get("in_rooms", []))
        if surface_name and surface_name in init_info:
            target_rooms.update(init_info[surface_name].get("args", {}).get("in_rooms", []))
        target_rooms = list(target_rooms)

        scenes.append({
            "name": scene_key,
            "scene_file": str(scene_file),
            "scene_model": diag.get("scene_model", scene_dir.name),
            "target_name": target_name,
            "surface_name": surface_name,
            "target_rooms": target_rooms,
            "prompt": prompt,
            "pipeline": pipeline,
            "activity_name": diag.get("activity_name", ""),
            "cameras": diag.get("cameras", []),
            "goal_conditions": diag.get("goal_conditions", []),
            "goal_region": diag.get("goal_region"),
            "ltl_safety": diag.get("ltl_safety") or {},
        })

    if max_scenes:
        scenes = scenes[:max_scenes]
    return scenes
