"""Selection helpers for the clutter pipeline.

Wraps the two consumer-facing JSONs in this folder:

  * ``clutter_target_pool.json``  — graspable categories with
    ``clutter_target ∈ {perfect, possible}``.
  * ``table_obstacle_pool.json``  — graspable categories with
    ``table_obstacle ∈ {perfect, possible}``.

Both are sampled **uniform-by-category → uniform-by-model**, so categories
with one model (e.g. ``goblet``) are not drowned out by categories with
many (e.g. ``hardback`` ×251).

Returned identifiers are always ``(synset, category, model)`` triples so
the caller can pin a specific graspable model in the spawn spec.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TARGET_PATH = _HERE / "clutter_target_pool.json"
_OBSTACLE_PATH = _HERE / "table_obstacle_pool.json"
_FRAGILE_PATH = _HERE / "fragile_pool.json"
_FILLABLE_PATH = _HERE / "fillable_container_pool.json"

_target_cache = None
_obstacle_cache = None
_fragile_cache = None
_fillable_cache = None


def _load(path, cache_attr):
    cur = globals()[cache_attr]
    if cur is None:
        with open(path) as f:
            cur = json.load(f)
        globals()[cache_attr] = cur
    return cur


def load_target_pool():
    return _load(_TARGET_PATH, "_target_cache")


def load_obstacle_pool():
    return _load(_OBSTACLE_PATH, "_obstacle_cache")


def load_fragile_pool():
    return _load(_FRAGILE_PATH, "_fragile_cache")


def load_fillable_pool():
    return _load(_FILLABLE_PATH, "_fillable_cache")


def _entries(doc):
    return {k: v for k, v in doc.items() if k != "metadata"}


def _pick(pool_entries, rng, exclude_cats=()):
    """Uniform-by-category → uniform-by-model over the given pool."""
    cats = [c for c in pool_entries if c not in exclude_cats]
    if not cats:
        raise RuntimeError(
            f"Pool empty after exclude_cats={sorted(exclude_cats)}; "
            f"check pool JSONs in clutter_pipeline/."
        )
    cat = cats[rng.integers(len(cats))]
    entry = pool_entries[cat]
    models = entry["models"]
    model = models[rng.integers(len(models))]
    return entry["synset"], cat, model


def select_target(rng):
    """Pick one clutter target uniformly from clutter_target_pool.

    Returns (synset, category, model).
    """
    return _pick(_entries(load_target_pool()), rng)


def select_obstacle(rng, exclude_cats=()):
    """Pick one tabletop obstacle uniformly from table_obstacle_pool.

    ``exclude_cats`` lets the caller forbid the target's category so the
    target and an obstacle don't share an asset model. Returns
    (synset, category, model).
    """
    return _pick(_entries(load_obstacle_pool()), rng, exclude_cats=exclude_cats)


def select_fragile(rng, exclude_cats=()):
    """Pick one fragile (tall, tippable, graspable) object uniformly.

    Source: ``fragile_pool.json`` (regenerate via
    ``build_fragile_pool.py`` if the graspability CSV or the footprint
    catalog changes). Returns (synset, category, model).
    """
    return _pick(_entries(load_fragile_pool()), rng, exclude_cats=exclude_cats)


def select_fillable_container(rng, exclude_cats=()):
    """Pick one graspable + fillable container uniformly.

    Source: ``fillable_container_pool.json`` (regenerate via
    ``build_fillable_pool.py``). Used by liquid_transport / wet_transport
    pipelines as the carried-container target. Returns (synset,
    category, model).
    """
    return _pick(_entries(load_fillable_pool()), rng, exclude_cats=exclude_cats)
