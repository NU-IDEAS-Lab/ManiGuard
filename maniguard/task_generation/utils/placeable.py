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


def _surface_entry(s, height_m):
    """Flatten a surfaces[] row into a self-contained pickable dict."""
    return {
        "category": s["category"],
        "model": s["model"],
        "region_id": s["region_id"],
        "xy_min": list(s["xy_min"]),
        "xy_max": list(s["xy_max"]),
        "top_plane_z_local": float(s["top_plane_z_local"]),
        "area_m2": float(s["area_m2"]),
        "reachable_edge_labels": list(s.get("reachable_edge_labels") or ()),
        "height_m": float(height_m) if height_m is not None else None,
    }


def list_eligible_surfaces(
    required_area_m2=0.0,
    required_category=None,
    required_model=None,
):
    """Return all placeable regions passing the area / category / model filter.

    Operates at REGION granularity: each region is an independent candidate
    (a 2-region model contributes two entries), so smaller regions can be
    picked on their own merit. Returns a list of self-contained surface
    dicts; scene info is not included here -- use pick_scene_from_placeable
    for that.
    """
    doc = load_placeable_surfaces()
    by_model = doc.get("by_model") or {}
    out = []
    for s in doc["surfaces"]:
        if s["area_m2"] < required_area_m2:
            continue
        if required_category and s["category"] != required_category:
            continue
        if required_model and s["model"] != required_model:
            continue
        height_m = (by_model.get(s["category"]) or {}).get(s["model"], {}).get("height_m")
        out.append(_surface_entry(s, height_m))
    return out


def _choose_from(candidates, rng, weighted_by_area):
    if weighted_by_area:
        import numpy as _np
        weights = _np.array([c["area_m2"] for c in candidates], dtype=float)
        probs = weights / weights.sum()
        return candidates[int(rng.choice(len(candidates), p=probs))]
    return candidates[rng.integers(len(candidates))]


def pick_surface_from_placeable(
    rng,
    required_area_m2,
    required_category=None,
    required_model=None,
    weighted_by_area=False,
):
    """Pick one placeable region by area (no scene constraint).

    Each region is its own candidate (2-region models contribute two entries).
    Intended for empty-scene pipelines that don't care which B1K scene the
    model came from.

    Returns a surface dict with category / model / region_id / xy_min / xy_max /
    top_plane_z_local / area_m2 / reachable_edge_labels / height_m.
    Raises RuntimeError if no region passes the filter.
    """
    candidates = list_eligible_surfaces(
        required_area_m2=required_area_m2,
        required_category=required_category,
        required_model=required_model,
    )
    if not candidates:
        extras = []
        if required_category:
            extras.append(f"category={required_category!r}")
        if required_model:
            extras.append(f"model={required_model!r}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        raise RuntimeError(
            f"No placeable region with area >= {required_area_m2:.3f} m²{suffix}."
        )
    return _choose_from(candidates, rng, weighted_by_area)


def pick_scene_from_placeable(
    rng,
    required_area_m2,
    scene_model=None,
    required_category=None,
    required_model=None,
    weighted_by_area=False,
):
    """Pick one (placeable region, scene, room) triple.

    Filters regions by area / category / model, then expands each kept region
    against the model's scenes[] list. Restricts to scene_model if specified.
    Each (region, scene, room) combination is an independent candidate.

    Returns a dict merging the surface fields (see pick_surface_from_placeable)
    with scene_model and room_instance. Raises RuntimeError on no match.
    """
    doc = load_placeable_surfaces()
    by_model = doc.get("by_model") or {}
    eligible = []
    for surface in list_eligible_surfaces(
        required_area_m2=required_area_m2,
        required_category=required_category,
        required_model=required_model,
    ):
        scenes = (by_model.get(surface["category"]) or {}).get(surface["model"], {}).get("scenes") or []
        for sc in scenes:
            if scene_model is not None and sc["scene_model"] != scene_model:
                continue
            entry = dict(surface)
            entry["scene_model"] = sc["scene_model"]
            entry["room_instance"] = sc["room_instance"]
            # Pass through the augmented per-instance scale + name so
            # callers can size the world-frame pack region offline
            # without reading ``support_obj.scale`` from a live env.
            if "scale_xyz" in sc:
                entry["scale_xyz"] = list(sc["scale_xyz"])
            if "instance_name" in sc:
                entry["instance_name"] = sc["instance_name"]
            eligible.append(entry)

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
            f"No (region, scene) pair in placeable_surfaces_v1.json with "
            f"region area >= {required_area_m2:.3f} m²{suffix}."
        )
    return _choose_from(eligible, rng, weighted_by_area)


def _scene_entry_for(surface_dict, scene_model):
    """Return the ``by_model[*].scenes[]`` row matching surface + scene.

    ``surface_dict`` is a row from ``list_eligible_surfaces`` (or the
    flatter ``pick_*`` outputs); ``scene_model`` selects the scene.
    Returns None if no row matches.
    """
    doc = load_placeable_surfaces()
    by_model = doc.get("by_model") or {}
    scenes = (by_model.get(surface_dict["category"]) or {}).get(
        surface_dict["model"], {}
    ).get("scenes") or []
    for sc in scenes:
        if sc.get("scene_model") == scene_model:
            return sc
    return None


def applied_scale_for(category, model, scene_model):
    """Look up the per-instance applied scale (3-vector) for a support.

    Reads ``by_model[category][model].scenes[*]`` for the row whose
    ``scene_model`` matches, returning its ``scale_xyz``. The
    augmentation that fills these fields lives in
    ``build_placeable_surface_scales.py``; if the entry is missing
    (older catalog), this returns None and callers must fall back to
    reading ``support_obj.scale`` at runtime.
    """
    doc = load_placeable_surfaces()
    by_model = doc.get("by_model") or {}
    scenes = (by_model.get(category) or {}).get(model, {}).get("scenes") or []
    for sc in scenes:
        if sc.get("scene_model") != scene_model:
            continue
        sx = sc.get("scale_xyz")
        if sx is not None:
            return tuple(float(v) for v in sx)
    return None
