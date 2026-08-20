"""Parse a ManiGuard base-task dump — Layer-1 primitive (family-agnostic).

A base task is a folder with ``scene_ep<N>.json`` (the OmniGibson scene snapshot)
+ ``diagnostics.jsonl`` (the bench's per-episode metadata). This module turns that
pair into the three things the scene builder needs: the diagnostics row, the scene
snapshot, and the ordered list of task-object names ``[support, obj1, obj2, ...]``
plus a ``DatasetObject`` config for each.

OmniGibson-free on purpose (pure JSON parsing → fast to unit-test without a sim).
Replicated clean from the reference reader
``maniguard/data/curobo/replay_empty_from_dataset.py`` — datagen does NOT import
the old curobo reference tree, so the loaders live here. Cleanup vs the reference:
dropped the unused ``_SCHEME_B_PATTERN`` and the family-specific dusty sponge-snap
hack (that belongs in the Layer-2 dusty skeleton, not a Layer-1 primitive).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_diagnostics_row(task_dir: Path, episode: int) -> dict[str, Any]:
    """Load the diagnostics entry for ``episode``.

    Tolerates both layouts the bench emits: a single pretty-printed JSON object
    (e.g. dusty) and one-JSON-object-per-line JSONL keyed by ``episode``.
    """
    path = Path(task_dir) / "diagnostics.jsonl"
    txt = path.read_text()
    try:
        row = json.loads(txt)
        if isinstance(row, dict):
            return row
    except json.JSONDecodeError:
        pass
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if int(row.get("episode", 0)) == episode:
            return row
    raise ValueError(f"No diagnostics entry for episode {episode} in {path}")


def load_scene_info(task_dir: Path, episode: int) -> dict[str, Any]:
    """Load the scene snapshot for ``episode`` (``scene_ep<N>.json``, with the
    ``..._replay.json`` variant as a fallback)."""
    task_dir = Path(task_dir)
    primary = task_dir / f"scene_ep{episode}.json"
    if primary.is_file():
        return json.loads(primary.read_text())
    fallback = task_dir / f"scene_ep{episode}_replay.json"
    if fallback.is_file():
        return json.loads(fallback.read_text())
    raise FileNotFoundError(
        f"No scene_ep{episode}.json or scene_ep{episode}_replay.json in {task_dir}"
    )


# Task objects are auto-named ``<category>_<digits>`` (e.g. ``lid_43``), distinct
# from scene furniture ``<category>_<model>_<id>``.
_TASK_OBJ_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_\d+$")


def _is_task_object_name(name: str, category: str) -> bool:
    """True if ``name`` is a spawned task object of ``category``.

    Scheme A: auto-named by category, prefix == category (``lid_43``).
    Scheme B/C: role-prefixed names where the category tokens appear as a
    contiguous run before the trailing ``_<idx>`` (``target_milk_carton_ep1_1``
    with category ``milk_carton``; dusty's ``dust_sponge_0`` with category
    ``sponge``). Callers further gate on spawn-spec categories, so scene
    furniture is already excluded.
    """
    if not _TASK_OBJ_PATTERN.match(name):
        return False
    prefix, _, tail = name.rpartition("_")
    if not tail.isdigit():
        return False
    if prefix == category:
        return True
    cat_toks = category.split("_")
    name_toks = name.split("_")
    for i in range(len(name_toks) - len(cat_toks) + 1):
        if name_toks[i:i + len(cat_toks)] == cat_toks:
            return True
    return False


def identify_task_objects(
    scene_info: dict[str, Any],
    diagnostics: dict[str, Any],
) -> list[str]:
    """Return ``[support, task_obj_1, task_obj_2, ...]``.

    ``spawn_specs`` records the pipeline's *intent*, but a spawn can fail silently
    (placement / physics gate), so the snapshot may hold fewer instances than
    requested. Trust the snapshot: keep every object matching a spawn-spec category
    AND the task-object name pattern.
    """
    init_info = scene_info["objects_info"]["init_info"]
    surface = _resolve_surface_name(scene_info, diagnostics)

    # Trust the snapshot: spawn every DatasetObject it holds (surface first), skipping the
    # robot. ``diagnostics['surface']`` may be the object NAME (clutter) or a ``cat/model``
    # (cabinet, named ``support_surface``); the robot is a non-DatasetObject class (e.g.
    # FrankaPanda) and is added separately by the scene builder, so it is excluded here.
    names: list[str] = [surface]
    for n, info in init_info.items():
        if n == surface:
            continue
        if info.get("class_name") != "DatasetObject":
            continue
        names.append(n)
    return names


def _resolve_surface_name(scene_info: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    """The support-surface object name. ``diagnostics['surface']`` is the object name in some
    families (clutter) and a ``category/model`` in others (cabinet, surface named
    ``support_surface``); resolve robustly via name, the conventional ``support_surface`` key,
    then category match."""
    init_info = scene_info["objects_info"]["init_info"]
    s = str(diagnostics.get("surface", ""))
    if s in init_info:
        return s
    if "support_surface" in init_info:
        return "support_surface"
    cat = s.split("/")[0] if "/" in s else (diagnostics.get("surface_info") or {}).get("category", "")
    for n, info in init_info.items():
        if info.get("args", {}).get("category") == cat and info.get("class_name") == "DatasetObject":
            return n
    raise ValueError(f"Surface {s!r} not found in scene snapshot")


def build_object_cfg(
    name: str,
    scene_info: dict[str, Any],
    fixed_base: bool | None = None,
) -> dict[str, Any]:
    """Build an OmniGibson ``DatasetObject`` config from the snapshot, at the object's dumped
    pose. ``fixed_base`` defaults to the snapshot's own per-object value (the surface AND
    furniture like a cabinet are fixed; manipulable objects are free)."""
    init_info = scene_info["objects_info"]["init_info"][name]
    reg = scene_info["state"]["registry"]["object_registry"][name]
    args = init_info.get("args", {})
    scale = args.get("scale")
    scale = [float(v) for v in scale] if scale is not None else [1.0, 1.0, 1.0]
    fb = bool(args.get("fixed_base", False)) if fixed_base is None else bool(fixed_base)
    return {
        "type": "DatasetObject",
        "name": name,
        "category": args["category"],
        "model": args["model"],
        "scale": scale,
        "fixed_base": fb,
        "position": [float(v) for v in reg["root_link"]["pos"]],
        "orientation": [float(v) for v in reg["root_link"]["ori"]],
    }
