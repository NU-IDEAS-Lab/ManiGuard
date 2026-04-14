"""Shared loaders and scene-selection helpers around placeable_surfaces_v1.json.

The JSON itself lives at utils/placeable_surfaces_v1.json and is the consumer-
facing index of raycast-validated, ready-filtered support surfaces. Pipelines
use `pick_scene_from_placeable` to select a (scene, model) pair for their
support surface, either across all scenes or within a user-specified one.
"""

from __future__ import annotations

import json
import os


_PLACEABLE_SURFACES_PATH = os.path.join(
    os.path.dirname(__file__), "placeable_surfaces_v1.json",
)
_placeable_doc_cache = None


def load_placeable_surfaces():
    """Load placeable_surfaces_v1.json (module-cached)."""
    global _placeable_doc_cache
    if _placeable_doc_cache is None:
        with open(_PLACEABLE_SURFACES_PATH, "r", encoding="utf-8") as f:
            _placeable_doc_cache = json.load(f)
    return _placeable_doc_cache


def _model_area_index():
    doc = load_placeable_surfaces()
    model_area = {}
    for s in doc["surfaces"]:
        key = (s["category"], s["model"])
        model_area[key] = model_area.get(key, 0.0) + s["area_m2"]
    return doc, model_area


def pick_scene_from_placeable(
    rng,
    required_area_m2,
    scene_model=None,
    required_category=None,
    required_model=None,
    require_scene=True,
    weighted_by_area=False,
):
    """Pick (scene_model, category, model, room_instance, area) from placeable.

    Source of truth is placeable_surfaces_v1.json. Each by_model entry carries:
      - summed usable region area (from surfaces[])
      - scenes: list of (scene_model, room_instance) where this model appears

    Filters (all optional):
      - required_area_m2: minimum summed usable area
      - scene_model: restrict to entries in this scene
      - required_category / required_model: pin category / model
      - require_scene: if True (default), only models with >=1 scene entry are
        eligible; if False, models with no scene still pass and are returned
        with scene_model=room_instance=None (use for empty-scene pipelines).

    Selection: uniform random over all (model, scene) pairs passing the filter,
    or area-weighted if weighted_by_area=True.

    Returns a dict: {scene_model, category, model, room_instance, area_m2}.
    Raises RuntimeError if nothing matches.
    """
    doc, model_area = _model_area_index()
    by_model = doc["by_model"]

    eligible = []
    for (cat, model), area in model_area.items():
        if area < required_area_m2:
            continue
        if required_category and cat != required_category:
            continue
        if required_model and model != required_model:
            continue
        scenes = (by_model.get(cat) or {}).get(model, {}).get("scenes") or []
        if not scenes:
            if require_scene:
                continue
            eligible.append({
                "scene_model": None,
                "category": cat,
                "model": model,
                "room_instance": None,
                "area_m2": float(area),
            })
            continue
        for sc in scenes:
            if scene_model is not None and sc["scene_model"] != scene_model:
                continue
            eligible.append({
                "scene_model": sc["scene_model"],
                "category": cat,
                "model": model,
                "room_instance": sc["room_instance"],
                "area_m2": float(area),
            })

    if not eligible:
        extras = []
        if scene_model:
            extras.append(f"scene={scene_model!r}")
        if required_category:
            extras.append(f"category={required_category!r}")
        if required_model:
            extras.append(f"model={required_model!r}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        raise RuntimeError(
            f"No (scene, model) pair in placeable_surfaces_v1.json with "
            f"usable area >= {required_area_m2:.3f} m²{suffix}."
        )
    if weighted_by_area:
        import numpy as _np
        weights = _np.array([e["area_m2"] for e in eligible], dtype=float)
        probs = weights / weights.sum()
        return eligible[int(rng.choice(len(eligible), p=probs))]
    return eligible[rng.integers(len(eligible))]
