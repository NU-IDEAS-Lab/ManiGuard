"""Selection helpers for the wet_transport pipeline.

Wraps ``water_sensitive_pool.json`` — curated water-sensitive
categories (books, papers, electronics) at category-keyed
``{synset, models}`` granularity. ``select_water_sensitive`` samples
uniformly by category, then uniformly by model — same pattern the
clutter pipeline's selectors use, so the hardback-heavy raw count (251
graspable models vs ~3-17 for other categories) doesn't dominate.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PATH = _HERE / "water_sensitive_pool.json"
_cache = None


def load_water_sensitive_pool():
    global _cache
    if _cache is None:
        with open(_PATH) as f:
            _cache = json.load(f)
    return _cache


def _entries(doc):
    return {k: v for k, v in doc.items() if k != "metadata"}


def select_water_sensitive(rng, exclude_cats=()):
    """Pick one graspable water-sensitive model uniformly.

    Source: ``water_sensitive_pool.json`` (regenerate via
    ``build_water_sensitive_pool.py``). ``exclude_cats`` is a set of
    categories to skip — e.g. when chaining picks you usually want to
    exclude the target's category to avoid duplicates. Returns
    ``(synset, category, model)``.
    """
    entries = _entries(load_water_sensitive_pool())
    eligible = [c for c in entries if c not in exclude_cats]
    if not eligible:
        raise RuntimeError(
            "select_water_sensitive: no categories left after excluding "
            f"{sorted(exclude_cats)} from "
            f"{sorted(entries)}."
        )
    cat = eligible[int(rng.integers(len(eligible)))]
    info = entries[cat]
    model = info["models"][int(rng.integers(len(info["models"])))]
    return info["synset"], cat, model
