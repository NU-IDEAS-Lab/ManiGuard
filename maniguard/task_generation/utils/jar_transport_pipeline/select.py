"""Selection helpers for the jar_transport pipeline.

Picks one of the four ``hinged_jar`` models in BEHAVIOR-1K plus a
graspable item that fits through the jar's opening.

Fit rule (from the task spec): ``item.extent_xyz.max() < jar.extent_xyz.min()``
— i.e. the longest axis of the item must be strictly shorter than the
shortest axis of the jar's bbox. We also apply an additive margin so
the item doesn't graze the rim.

Item candidates come from ``table_obstacle_pool.json`` (~1946 graspable
models). Items are *not* pre-filtered for fragility / dropability; LTL
safety constraints catch those at rollout time.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[3]
_FOOTPRINTS_PATH = (
    _PROJECT_ROOT
    / "maniguard" / "task_generation" / "utils" / "object_footprints.json"
)
_OBSTACLE_POOL_PATH = (
    _PROJECT_ROOT
    / "maniguard" / "task_generation" / "utils" / "clutter_pipeline"
    / "table_obstacle_pool.json"
)
_JAR_ASSETS_DIR = (
    _PROJECT_ROOT
    / "behavior-1k" / "datasets" / "behavior-1k-assets" / "objects"
)

JAR_CATEGORY = "hinged_jar"
JAR_SYNSET = "hinged_jar.n.01"
# All four hinged_jar models shipping with BEHAVIOR-1K.
JAR_MODELS = ("gqtsam", "jnjtrl", "kijnrj", "vzwhbg")
# Inset (m) from the jar body's outer xy bbox to its inner opening —
# accounts for the glass / plastic wall thickness. 1 cm is a safe
# default across all four hinged_jar models inspected.
_DEFAULT_WALL_INSET_M = 0.010
# Preference order when locating the "body" link for opening sizing.
_BODY_LINK_CANDIDATES = ("glass", "body", "container", "base_link")

_footprint_cache: dict | None = None
_obstacle_pool_cache: dict | None = None
_jar_meta_cache: dict[str, dict] = {}


def _load_footprints() -> dict:
    global _footprint_cache
    if _footprint_cache is None:
        with open(_FOOTPRINTS_PATH) as fh:
            _footprint_cache = json.load(fh)
    return _footprint_cache


def _load_obstacle_pool() -> dict:
    global _obstacle_pool_cache
    if _obstacle_pool_cache is None:
        with open(_OBSTACLE_POOL_PATH) as fh:
            _obstacle_pool_cache = json.load(fh)
    return _obstacle_pool_cache


def jar_min_dim(jar_model: str) -> float:
    """Return min(extent_xyz) of the named hinged_jar's FULL bounding
    box (m). Kept for backward compatibility / diagnostics; the actual
    fit gate uses :func:`jar_opening_min_dim` because the full bbox
    includes the lid and the outer wall, both wider than the cavity
    opening the item must clear.
    """
    fp = _load_footprints().get(JAR_CATEGORY, {}).get(jar_model)
    if fp is None:
        raise RuntimeError(
            f"No footprint for {JAR_CATEGORY}/{jar_model} — regenerate "
            "object_footprints.json against the current BEHAVIOR assets."
        )
    return float(min(fp["extent_xyz"]))


def _load_jar_metadata(jar_model: str) -> dict:
    if jar_model not in _jar_meta_cache:
        path = (_JAR_ASSETS_DIR / JAR_CATEGORY / jar_model
                / "misc" / "metadata.json")
        _jar_meta_cache[jar_model] = json.loads(path.read_text())
    return _jar_meta_cache[jar_model]


def jar_opening_min_dim(jar_model: str,
                        wall_inset_m: float = _DEFAULT_WALL_INSET_M) -> float:
    """Return the smallest xy edge of the jar's *cavity opening* (m).

    Approach: pick the body link (``glass`` / ``body`` / ``container``
    / ``base_link`` — first present in that priority order), take the
    smaller of its two xy extents (the outer rim), then subtract
    ``2 * wall_inset_m`` to approximate the *inner* opening — accounting
    for the jar's wall thickness.

    This is the correct dimension to compare against an item's max
    bbox extent when deciding "does it fit through the opening".
    """
    meta = _load_jar_metadata(jar_model)
    boxes = meta.get("link_bounding_boxes", {})
    for ln in _BODY_LINK_CANDIDATES:
        lb = boxes.get(ln)
        if lb is None:
            continue
        aa = lb.get("collision", {}).get("axis_aligned", {})
        ext = aa.get("extent")
        if not ext:
            continue
        outer = float(min(float(ext[0]), float(ext[1])))
        return max(0.0, outer - 2.0 * float(wall_inset_m))
    # Fallback: full bbox min minus a wall inset (lower bound).
    return max(0.0, jar_min_dim(jar_model) - 2.0 * float(wall_inset_m))


def select_jar(
    rng: np.random.Generator,
    jar_model: str | None = None,
) -> tuple[str, str, str]:
    """Pick a hinged_jar model. Returns (synset, category, model)."""
    if jar_model is not None:
        if jar_model not in JAR_MODELS:
            raise RuntimeError(
                f"Unknown hinged_jar model {jar_model!r}; "
                f"available: {JAR_MODELS}"
            )
        return JAR_SYNSET, JAR_CATEGORY, jar_model
    return JAR_SYNSET, JAR_CATEGORY, JAR_MODELS[int(rng.integers(len(JAR_MODELS)))]


def candidate_items(
    jar_model: str,
    *,
    fit_margin_m: float = 0.015,
    min_extent_m: float = 0.0,
    wall_inset_m: float = _DEFAULT_WALL_INSET_M,
    exclude_cats: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Return {category: [models]} of graspable items whose:

      * longest bbox axis is strictly less than the jar's *opening*
        min-xy minus ``fit_margin_m`` (so the item clears the cavity
        opening). The opening size comes from
        :func:`jar_opening_min_dim` — the body link's inner xy
        diameter, not the full jar bbox.
      * every bbox axis is at least ``min_extent_m`` (so the item is
        big enough to actually see in renders).
    """
    max_budget = jar_opening_min_dim(jar_model, wall_inset_m=wall_inset_m)
    max_budget -= float(fit_margin_m)
    if max_budget <= 0.0:
        raise RuntimeError(
            f"jar_model={jar_model!r} opening smaller than margin "
            f"({fit_margin_m}); no items can fit."
        )
    min_floor = float(min_extent_m)
    if min_floor > max_budget:
        raise RuntimeError(
            f"min_extent_m={min_floor:.3f} exceeds jar opening budget "
            f"{max_budget:.3f} for jar_model={jar_model!r} — no items "
            f"can satisfy both constraints."
        )

    footprints = _load_footprints()
    pool = _load_obstacle_pool()
    exclude = set(exclude_cats) | {JAR_CATEGORY}

    eligible: dict[str, list[str]] = {}
    for cat, entry in pool.items():
        if cat == "metadata" or cat in exclude:
            continue
        cat_fp = footprints.get(cat, {})
        kept = []
        for model in entry["models"]:
            fp = cat_fp.get(model)
            if fp is None:
                continue
            extent = fp["extent_xyz"]
            if float(max(extent)) >= max_budget:
                continue
            if min_floor > 0.0 and float(min(extent)) < min_floor:
                continue
            kept.append(model)
        if kept:
            eligible[cat] = kept
    return eligible


def select_item(
    rng: np.random.Generator,
    jar_model: str,
    *,
    item_category: str | None = None,
    item_model: str | None = None,
    fit_margin_m: float = 0.015,
    min_extent_m: float = 0.0,
    wall_inset_m: float = _DEFAULT_WALL_INSET_M,
    exclude_cats: Sequence[str] = (),
) -> tuple[str, str, str]:
    """Pick a graspable item that fits through the jar's opening AND
    is at least ``min_extent_m`` on every axis (so it's not a sliver
    that vanishes in renders).

    Sampling is uniform-by-category → uniform-by-model so categories
    with one model (e.g. one-off graspables) aren't drowned out.
    Returns (synset, category, model).
    """
    fit_pool = candidate_items(jar_model, fit_margin_m=fit_margin_m,
                               min_extent_m=min_extent_m,
                               wall_inset_m=wall_inset_m,
                               exclude_cats=exclude_cats)
    if not fit_pool:
        raise RuntimeError(
            f"No graspable items fit hinged_jar/{jar_model} with "
            f"fit_margin {fit_margin_m} m and min_extent {min_extent_m} m."
        )

    if item_category is not None:
        if item_category not in fit_pool:
            raise RuntimeError(
                f"override item_category={item_category!r} doesn't fit "
                f"hinged_jar/{jar_model}; eligible: "
                f"{sorted(fit_pool.keys())[:10]}..."
            )
        models = fit_pool[item_category]
        if item_model is not None:
            if item_model not in models:
                raise RuntimeError(
                    f"override item_model={item_model!r} not in "
                    f"category {item_category!r} (have: {models})."
                )
            return _resolve_synset(item_category), item_category, item_model
        chosen_model = models[int(rng.integers(len(models)))]
        return _resolve_synset(item_category), item_category, chosen_model

    cats = sorted(fit_pool.keys())
    cat = cats[int(rng.integers(len(cats)))]
    model = fit_pool[cat][int(rng.integers(len(fit_pool[cat])))]
    return _resolve_synset(cat), cat, model


def _resolve_synset(category: str) -> str:
    """Look up the category's synset via the obstacle pool entry, falling
    back to ``<category>.n.01`` if the pool doesn't record one.
    """
    pool = _load_obstacle_pool()
    entry = pool.get(category) or {}
    return entry.get("synset") or f"{category}.n.01"


def select_jar_and_item(
    rng: np.random.Generator,
    *,
    jar_model: str | None = None,
    item_category: str | None = None,
    item_model: str | None = None,
    fit_margin_m: float = 0.015,
    min_extent_m: float = 0.0,
    wall_inset_m: float = _DEFAULT_WALL_INSET_M,
) -> dict:
    """Convenience picker used by the pipeline's ``select_objects``."""
    jar_synset, jar_cat, jar_m = select_jar(rng, jar_model=jar_model)
    item_synset, item_cat, item_m = select_item(
        rng, jar_m,
        item_category=item_category, item_model=item_model,
        fit_margin_m=fit_margin_m, min_extent_m=min_extent_m,
        wall_inset_m=wall_inset_m,
    )
    return {
        "jar_synset": jar_synset,
        "jar_category": jar_cat,
        "jar_model": jar_m,
        "jar_min_dim_m": jar_min_dim(jar_m),
        "jar_opening_min_dim_m": jar_opening_min_dim(jar_m, wall_inset_m),
        "item_synset": item_synset,
        "item_category": item_cat,
        "item_model": item_m,
    }
