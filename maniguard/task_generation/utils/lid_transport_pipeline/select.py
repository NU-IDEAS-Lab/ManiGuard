"""Selection helpers for lid-transport pipelines.

Wraps the two consumer-facing JSONs in this folder:

  * ``lid_cap_container_pairs.json`` — full inventory of every
    (lid|cap, container) attachment pair with per-side status + verdict.
    Used by the **liquid** variant: pick uniformly from admitted pairs.

  * ``lid_transport_food_compat.json`` — admitted pairs joined with the
    ``transfer_compatibility.json`` food list per container. Used by the
    **food** variant: pick (item, container, food_cat, food_model)
    uniformly with category-first sampling on foods.

Selection is uniform-by-pair (then uniform-by-food-category, then
uniform-by-food-model). Caller pre-resolves everything; the activity
generator in ``task_spec`` only receives the final identifiers.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PAIRS_PATH = _HERE / "lid_cap_container_pairs.json"
_FOOD_COMPAT_PATH = _HERE / "lid_transport_food_compat.json"

_pairs_cache = None
_food_compat_cache = None

# Verdicts that admit a pair into the kept set.
KEEP_VERDICTS = frozenset({"kept", "kept_via_relax"})
MANUAL_STATUS = "manually_added"

def load_pairs():
    """Load lid_cap_container_pairs.json (cached)."""
    global _pairs_cache
    if _pairs_cache is None:
        with open(_PAIRS_PATH) as f:
            _pairs_cache = json.load(f)
    return _pairs_cache


def load_food_compat():
    """Load lid_transport_food_compat.json (cached)."""
    global _food_compat_cache
    if _food_compat_cache is None:
        with open(_FOOD_COMPAT_PATH) as f:
            _food_compat_cache = json.load(f)
    return _food_compat_cache


def _is_admitted(item_status, container_entry):
    """Stage-1 admission rule: kept verdicts OR a hand-set ``manually_added``
    on either side."""
    if item_status == MANUAL_STATUS:
        return True
    if container_entry.get("status") == MANUAL_STATUS:
        return True
    return container_entry.get("verdict") in KEEP_VERDICTS


def iter_admitted_pairs(item_categories=("lid", "cap")):
    """Yield admitted (item, container) pairs from the stage-1 inventory.

    Each yielded dict carries ``item_category``, ``item_model``,
    ``container_category``, ``container_model``.
    """
    doc = load_pairs()
    for item_cat in item_categories:
        if item_cat not in doc:
            continue
        for item_model, info in doc[item_cat].items():
            for c in info["containers"]:
                if _is_admitted(info["item_status"], c):
                    yield {
                        "item_category": item_cat,
                        "item_model": item_model,
                        "container_category": c["category"],
                        "container_model": c["model"],
                    }


def select_pair_for_liquid(rng,
                           item_categories=("lid", "cap"),
                           item_model=None):
    """Uniform random pick from admitted (item, container) pairs.

    Used by the liquid variant. Raises RuntimeError if the filtered pool
    is empty.
    """
    pool = list(iter_admitted_pairs(item_categories))
    if item_model is not None:
        pool = [p for p in pool if p["item_model"] == item_model]
    if not pool:
        raise RuntimeError(
            f"No admitted lid/cap-container pairs (item_categories={item_categories}, "
            f"item_model={item_model}). "
            f"Run build_lid_cap_container_pairs.py to refresh."
        )
    return pool[rng.integers(len(pool))]


def select_pair_for_food(rng,
                         item_categories=("lid", "cap"),
                         item_model=None):
    """Pick (item, container, food_cat, food_model) uniformly from food compat.

    Three sampling stages:
      1. Pick (item_category, item_model, container) uniformly across all
         (item, container) pairs that have food data.
      2. Within that container, pick a food category uniformly.
      3. Pick a food model uniformly from that category's models list.

    The category-first food sampling avoids bias toward food categories
    with many models (e.g. ``apple`` has ~10, single-model categories
    would be drowned out by per-model uniform).

    Returns dict with: ``item_category``, ``item_model``,
    ``container_category``, ``container_model``, ``food_category``,
    ``food_model``.
    """
    doc = load_food_compat()
    triples = []  # flat list of (item_cat, item_model, container_entry)
    for item_cat in item_categories:
        if item_cat not in doc:
            continue
        for im, containers in doc[item_cat].items():
            if item_model is not None and im != item_model:
                continue
            for c in containers:
                triples.append((item_cat, im, c))
    if not triples:
        raise RuntimeError(
            f"No (item, container) pairs with food data "
            f"(item_categories={item_categories}, item_model={item_model}). "
            f"Run build_lid_transport_food_compat.py to refresh."
        )
    item_cat, im, c = triples[rng.integers(len(triples))]
    foods = c["foods"]
    food_cats = list(foods.keys())
    food_cat = food_cats[rng.integers(len(food_cats))]
    food_model = foods[food_cat][rng.integers(len(foods[food_cat]))]
    return {
        "item_category": item_cat,
        "item_model": im,
        "container_category": c["container_category"],
        "container_model": c["container_model"],
        "food_category": food_cat,
        "food_model": food_model,
    }
