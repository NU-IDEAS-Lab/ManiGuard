"""Selection helpers for the liquid_transport pipeline.

Wraps ``liquid_fragile_pool.json`` — a GPU-dynamics-safe subset of the
clutter pipeline's fragile pool. See ``build_liquid_fragile_pool.py``
for the rationale on why a separate pool is needed.

Sampling is uniform-by-category → uniform-by-model, mirroring the
clutter selectors so category counts with one model don't get drowned
out by categories with many.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIQUID_FRAGILE_PATH = _HERE / "liquid_fragile_pool.json"

_liquid_fragile_cache = None


def _load(path, cache_attr):
    cur = globals()[cache_attr]
    if cur is None:
        with open(path) as f:
            cur = json.load(f)
        globals()[cache_attr] = cur
    return cur


def load_liquid_fragile_pool():
    return _load(_LIQUID_FRAGILE_PATH, "_liquid_fragile_cache")


def _entries(doc):
    return {k: v for k, v in doc.items() if k != "metadata"}


def _pick(pool_entries, rng, exclude_cats=()):
    """Uniform-by-category → uniform-by-model over the given pool."""
    cats = [c for c in pool_entries if c not in exclude_cats]
    if not cats:
        raise RuntimeError(
            f"Pool empty after exclude_cats={sorted(exclude_cats)}; "
            f"check pool JSONs in liquid_transport/."
        )
    cat = cats[rng.integers(len(cats))]
    entry = pool_entries[cat]
    models = entry["models"]
    model = models[rng.integers(len(models))]
    return entry["synset"], cat, model


def select_liquid_fragile(rng, exclude_cats=()):
    """Pick one fragile (tall, tippable, graspable) object uniformly,
    excluding any category whose BEHAVIOR abilities would trigger
    particle-system init under GPU dynamics. Returns
    (synset, category, model).
    """
    return _pick(_entries(load_liquid_fragile_pool()), rng,
                 exclude_cats=exclude_cats)
