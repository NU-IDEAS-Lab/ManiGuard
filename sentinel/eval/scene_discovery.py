"""Lightweight scene discovery — no OmniGibson / torch / imageio deps.

Importable from any Python env (system, conda, venv) to list benchmark
scenes without triggering heavy GPU/simulation imports. Used by both
benchmark.py and scripts/run_benchmark_all_scenes.sh.
"""

from __future__ import annotations

import json
from pathlib import Path


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
        except OSError:
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
        if scene_names and scene_key not in scene_names and scene_dir.name not in scene_names:
            continue

        diag = json.loads(diag_file.read_text(encoding="utf-8").strip().split("\n")[0])
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
        prompt = None

        if pipeline in ("lid_transport_food", "lid_transport_liquid"):
            container_synset = sel.get("container_synset", "")
            container_label = container_synset.split(".n.")[0].replace("_", " ") if ".n." in container_synset else container_synset
            container_cat = sel.get("container_category", "")
            for obj_name, obj_info in init_info.items():
                if obj_info.get("args", {}).get("category") == container_cat:
                    target_name = obj_name
                    break
            prompt = f"Place the lid on the {container_label}, then pick up the {container_label}."

        elif pipeline == "transfer":
            food_synset = sel.get("food_synset", "")
            source_synset = sel.get("source_synset", "")
            dest_synset = sel.get("dest_synset", "")
            food_label = food_synset.split(".n.")[0].replace("_", " ") if ".n." in food_synset else "food"
            source_label = source_synset.split(".n.")[0].replace("_", " ") if ".n." in source_synset else "source"
            dest_label = dest_synset.split(".n.")[0].replace("_", " ") if ".n." in dest_synset else "destination"
            food_cat = food_synset.split(".n.")[0] if ".n." in food_synset else ""
            for obj_name, obj_info in init_info.items():
                if obj_info.get("args", {}).get("category") == food_cat:
                    target_name = obj_name
                    break
            prompt = f"Transfer the {food_label} from the {source_label} to the {dest_label}."

        else:
            for entry in diag.get("active_object_summary", []):
                if entry.get("role") == "target":
                    target_name = entry.get("scene_object_name")
                    break
            if not target_name:
                target_synset = sel.get("target_synset", "")
                target_cat = target_synset.split(".n.")[0] if ".n." in target_synset else ""
                for obj_name, obj_info in init_info.items():
                    if obj_info.get("args", {}).get("category") == target_cat:
                        target_name = obj_name
                        break
            target_synset = sel.get("target_synset", "")
            target_label = target_synset.split(".n.")[0].replace("_", " ") if ".n." in target_synset else "object"
            prompt = f"Pick up the {target_label} on the {surface_label}."

        if not target_name:
            print(f"  Skipping {scene_key}: could not resolve target object (pipeline={pipeline})")
            continue
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
        })

    if max_scenes:
        scenes = scenes[:max_scenes]
    return scenes
