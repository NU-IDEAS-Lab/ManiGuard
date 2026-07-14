"""Augment ``placeable_surfaces_v1.json`` with per-instance applied scale.

The runtime support surface in each B1K scene carries a per-instance
``scale`` (3-vector) from the scene's USD ``init_info``. The placeable
catalog records every (category, model) → list of scenes/rooms it
appears in, but stops short of recording each instance's scale.
Without the scale, offline_pack can't size the world-frame pack region
without loading the env and reading ``support_obj.scale`` at runtime —
which is exactly the dependency we want to break so multi-episode
liquid pipelines can pre-spawn task objects via the env config (avoiding
the GPU-dynamics mid-run ``add_object`` race).

This script reads each entry's scene JSON
(``behavior-1k/datasets/behavior-1k-assets/scenes/<scene_model>/json/<scene_model>_best.json``),
finds the instance with the matching category + model + room_instance,
and records its ``scale_xyz`` and ``instance_name``. When more than one
matching instance exists in the same room, the FIRST one in iteration
order is recorded (mirroring what runtime ``next(o for o in env.scene.objects
if o.category == cat and ...)`` would pick).

Run:
    conda activate behavior  # only for the path; this script is pure Python
    python -m maniguard.task_generation.utils.build_placeable_surface_scales
"""
from __future__ import annotations

import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PLACEABLE_PATH = os.path.join(_HERE, "placeable_surfaces_v1.json")
_SCENES_DIR = os.path.join(
    _REPO, "behavior-1k", "datasets", "behavior-1k-assets", "scenes",
)


def _scene_init_info(scene_model: str) -> Optional[dict]:
    """Load ``<scene_model>_best.json`` and return ``objects_info.init_info``.

    Returns None if the scene file is missing (we tolerate dataset
    sparsity rather than failing the whole build).
    """
    path = os.path.join(_SCENES_DIR, scene_model, "json", f"{scene_model}_best.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    return ((doc.get("objects_info") or {}).get("init_info")) or None


def _find_instance(init_info: dict, category: str, model: str,
                   room_instance: str) -> Optional[tuple[str, list[float]]]:
    """Return (instance_name, scale_xyz) of the first matching instance.

    Match criteria: USD init_info args.category == category AND
    args.model == model AND ``room_instance`` ∈ args.in_rooms. The
    first match in iteration order wins (dicts preserve insertion order
    on Py3.7+, so this is deterministic per scene file).
    """
    for name, info in init_info.items():
        args = (info or {}).get("args") or {}
        if args.get("category") != category:
            continue
        if args.get("model") != model:
            continue
        in_rooms = args.get("in_rooms") or []
        if isinstance(in_rooms, str):
            in_rooms = [in_rooms]
        if room_instance not in in_rooms:
            continue
        scale = args.get("scale")
        if scale is None:
            continue
        return name, [float(s) for s in scale]
    return None


def main():
    with open(_PLACEABLE_PATH) as f:
        doc = json.load(f)

    by_model = doc.get("by_model") or {}
    init_cache: dict[str, Optional[dict]] = {}

    n_entries = 0
    n_filled = 0
    n_missing_scene_file: list[str] = []
    n_missing_instance: list[str] = []

    for category, models in by_model.items():
        for model, info in models.items():
            scenes = info.get("scenes") or []
            for entry in scenes:
                n_entries += 1
                scene_model = entry["scene_model"]
                room_instance = entry["room_instance"]
                if scene_model not in init_cache:
                    init_cache[scene_model] = _scene_init_info(scene_model)
                init_info = init_cache[scene_model]
                if init_info is None:
                    n_missing_scene_file.append(scene_model)
                    continue
                hit = _find_instance(init_info, category, model, room_instance)
                if hit is None:
                    n_missing_instance.append(
                        f"{category}/{model}@{scene_model}/{room_instance}"
                    )
                    continue
                instance_name, scale_xyz = hit
                entry["instance_name"] = instance_name
                entry["scale_xyz"] = scale_xyz
                n_filled += 1

    # Update metadata so consumers can see when the augmentation ran.
    source = doc.get("source") or {}
    source["augmented_with_scale"] = {
        "tool": "build_placeable_surface_scales.py",
        "scenes_dir": os.path.relpath(_SCENES_DIR, _REPO),
        "entries_total": n_entries,
        "entries_filled": n_filled,
        "missing_scene_files": sorted(set(n_missing_scene_file)),
        "missing_instances": sorted(set(n_missing_instance)),
    }
    doc["source"] = source

    with open(_PLACEABLE_PATH, "w") as f:
        json.dump(doc, f, indent=2)

    rel = os.path.relpath(_PLACEABLE_PATH, _REPO)
    print(f"wrote {rel}: filled {n_filled}/{n_entries} scene entries with "
          f"instance_name + scale_xyz")
    if n_missing_scene_file:
        uniq = sorted(set(n_missing_scene_file))
        print(f"  WARNING: {len(uniq)} scene_best.json missing — "
              f"{uniq[:5]}{'...' if len(uniq) > 5 else ''}")
    if n_missing_instance:
        print(f"  WARNING: {len(n_missing_instance)} entries with no "
              f"matching instance — first: {n_missing_instance[:3]}")


if __name__ == "__main__":
    main()
