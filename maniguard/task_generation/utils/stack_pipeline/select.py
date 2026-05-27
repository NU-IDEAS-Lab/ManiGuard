"""Shared target/stack-item selection for stack-retrieval pipelines.

Both ``stack_scene_pipeline`` (in-scene) and ``empty_scene_pipeline``
(empty-stage) use this. The three modes:

  * ``same``       — verified self-stack pool: target IS the stack item.
                     Source: ``stack_same_pool.json`` (physics-tested
                     3-copy + shake stability + graspable + no unresolved
                     complaints).
  * ``flat``       — geometric compat matrix: ``max(item.bbox_xy) <=
                     target.square_at_z_max_side_m``. Source:
                     ``stack_flat_compatibility.json``.
  * ``receptacle`` — geometric compat matrix on the cavity floor:
                     ``max(item.bbox_xy) <= target.square_at_z_min_side_m``
                     plus a ``z_range_m`` lower bound. Source:
                     ``stack_recep_compatibility.json``.

Selection is uniform over categories first, then over models within a
category, then over the model's ``items`` list. This avoids bias toward
categories with many models (e.g. ``hardback`` had 245 entries before
the category-first restructure).
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_STACK_SAME_POOL_PATH = _HERE / "stack_same_pool.json"
_STACK_FLAT_COMPAT_PATH = _HERE / "stack_flat_compatibility.json"
_STACK_RECEP_COMPAT_PATH = _HERE / "stack_recep_compatibility.json"

_stack_same_pool_cache = None
_stack_flat_compat_cache = None
_stack_recep_compat_cache = None


def load_stack_same_pool():
    """Load the verified stack-same pool, ``{category: [{model, ...}, ...]}``."""
    global _stack_same_pool_cache
    if _stack_same_pool_cache is None:
        with open(_STACK_SAME_POOL_PATH) as f:
            _stack_same_pool_cache = json.load(f)
    return _stack_same_pool_cache


def load_stack_flat_compat():
    """Load the stack-flat compatibility matrix, category-keyed."""
    global _stack_flat_compat_cache
    if _stack_flat_compat_cache is None:
        with open(_STACK_FLAT_COMPAT_PATH) as f:
            _stack_flat_compat_cache = json.load(f)
    return _stack_flat_compat_cache


def load_stack_recep_compat():
    """Load the stack-receptacle compatibility matrix, category-keyed."""
    global _stack_recep_compat_cache
    if _stack_recep_compat_cache is None:
        with open(_STACK_RECEP_COMPAT_PATH) as f:
            _stack_recep_compat_cache = json.load(f)
    return _stack_recep_compat_cache


def select_stack_objects(mode, rng, target_model=None, stack_model=None):
    """Pick (target, stack_item) for a stack task.

    Parameters
    ----------
    mode : {'same', 'flat', 'receptacle'}
    rng  : numpy Generator
    target_model : optional model id to pin the target.
    stack_model  : optional model id to pin the stack item.

    Returns
    -------
    dict with keys: ``required_area_m2``, ``target_synset``,
    ``target_category``, ``target_model``, ``stack_synset``,
    ``stack_category``, ``stack_model``.

    The returned dict is the same shape ``stack_scene_pipeline`` and
    ``empty_scene_pipeline`` both consume.
    """
    from maniguard.utils.task_spec import _load_footprint_catalog

    if mode == "same":
        pool = load_stack_same_pool()
        if target_model:
            hits = [
                (cat, e["model"])
                for cat, entries in pool.items()
                for e in entries
                if e["model"] == target_model
            ]
            if not hits:
                raise RuntimeError(
                    f"--target-model {target_model!r} not found in "
                    f"stack_same_pool.json. Run "
                    f"maniguard/task_generation/utils/stack_pipeline/"
                    f"build_stack_same_pool.py to refresh, or pick a model "
                    f"that passed the self-stack stability test."
                )
            t_cat, t_model = hits[0]
        else:
            cats = list(pool.keys())
            t_cat = cats[rng.integers(len(cats))]
            entries = pool[t_cat]
            t_model = entries[rng.integers(len(entries))]["model"]
        s_cat, s_model = t_cat, t_model
        t_synset = f"{t_cat}.n.01"
        s_synset = t_synset
        catalog = _load_footprint_catalog()
        # Exact per-model footprint (target == stack item in "same" mode).
        required = catalog[t_cat][t_model]["footprint_m2"]
        return {
            "required_area_m2": required,
            "target_synset": t_synset,
            "target_category": t_cat,
            "target_model": t_model,
            "stack_synset": s_synset,
            "stack_category": s_cat,
            "stack_model": s_model,
        }

    if mode in ("receptacle", "flat"):
        if mode == "receptacle":
            compat = load_stack_recep_compat()
            source = "stack_recep_compatibility.json"
            build_script = "build_stack_recep_compat.py"
        else:
            compat = load_stack_flat_compat()
            source = "stack_flat_compatibility.json"
            build_script = "build_stack_flat_compat.py"

        cats = list(compat.keys())
        if not cats:
            raise RuntimeError(
                f"{source} has no entries. Run "
                f"maniguard/task_generation/utils/stack_pipeline/"
                f"{build_script} to refresh."
            )

        if target_model:
            hit_cat, hit_entry = None, None
            for c, entries in compat.items():
                for e in entries:
                    if e["model"] == target_model:
                        hit_cat, hit_entry = c, e
                        break
                if hit_entry is not None:
                    break
            if hit_entry is None:
                raise RuntimeError(
                    f"--target-model {target_model!r} not found in {source}. "
                    f"Pick a model that passes the geometry filter, or "
                    f"rebuild the matrix."
                )
            t_cat, t_entry = hit_cat, hit_entry
        else:
            t_cat = cats[rng.integers(len(cats))]
            entries = compat[t_cat]
            t_entry = entries[rng.integers(len(entries))]
        t_model = t_entry["model"]
        t_synset = f"{t_cat}.n.01"

        items = t_entry["items"]
        if stack_model:
            s_hits = [it for it in items if it["model"] == stack_model]
            if not s_hits:
                raise RuntimeError(
                    f"--stack-model {stack_model!r} is not compatible with "
                    f"target {t_cat}/{t_model} (geometry / readiness "
                    f"filter). Try without --stack-model or pick a "
                    f"different target."
                )
            pick = s_hits[0]
        else:
            pick = items[rng.integers(len(items))]
        s_cat, s_model = pick["category"], pick["model"]
        s_synset = f"{s_cat}.n.01"

        catalog = _load_footprint_catalog()
        # Exact per-model footprint of the larger of the two participants.
        required = max(catalog[t_cat][t_model]["footprint_m2"],
                       catalog[s_cat][s_model]["footprint_m2"])
        return {
            "required_area_m2": required,
            "target_synset": t_synset,
            "target_category": t_cat,
            "target_model": t_model,
            "stack_synset": s_synset,
            "stack_category": s_cat,
            "stack_model": s_model,
        }

    raise ValueError(f"Unknown stack mode: {mode!r}")
