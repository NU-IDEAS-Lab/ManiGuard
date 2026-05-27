"""Build ``jar_transport_food_compat.json`` — per hinged_jar model, the
graspable food categories + models whose longest axis fits through the
jar's shortest opening.

Inputs:
  * ``docs/graspability_classified.csv`` — Franka graspability + food
    suitability classification (same source the food_transfer and
    lid_transport pipelines use). We keep rows where
    ``status == "graspable"`` AND ``food_transfer_target ∈ {perfect,
    possible}``.
  * ``utils/object_footprints.json`` — unscaled bounding-box extents
    for every (category, model) in BEHAVIOR-1K.

Fit rule (matching ``select.candidate_items``):
    ``food.extent_xyz.max() < jar.extent_xyz.min() - fit_margin_m``

Schema:
  {
    "metadata": {
      "source_graspability": ".../graspability_classified.csv",
      "source_footprints": ".../object_footprints.json",
      "fit_margin_m": 0.015,
      "jar_models": ["gqtsam", "jnjtrl", "kijnrj", "vzwhbg"],
      "n_food_models_eligible": int,
      "n_food_categories_eligible": int
    },
    "hinged_jar": {
      "<jar_model>": {
        "jar_extent_xyz": [x, y, z],
        "jar_min_dim_m": float,
        "fit_margin_m": float,
        "n_food_categories": int,
        "n_food_models": int,
        "foods": {"<food_cat>": ["<food_model>", ...], ...}
      },
      ...
    }
  }

Regenerate with::

    python -m maniguard.task_generation.utils.jar_transport_pipeline.build_jar_transport_food_compat
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[3]
_OUT_PATH = _HERE / "jar_transport_food_compat.json"
_FOOTPRINTS_PATH = (
    _PROJECT_ROOT
    / "maniguard" / "task_generation" / "utils" / "object_footprints.json"
)
_GRASPABILITY_PATH = _PROJECT_ROOT / "docs" / "graspability_classified.csv"

_JAR_CATEGORY = "hinged_jar"
_USABLE_FOOD_SUITABILITY = frozenset({"perfect", "possible"})
_DEFAULT_WALL_INSET_M = 0.010


def _eligible_food_rows() -> List[dict]:
    """Filter graspability_classified.csv to graspable foods that are
    valid transfer targets — same predicate as the food_transfer builder.
    """
    rows = []
    with open(_GRASPABILITY_PATH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("status", "").strip() != "graspable":
                continue
            if r.get("food_transfer_target", "").strip() not in _USABLE_FOOD_SUITABILITY:
                continue
            rows.append(r)
    return rows


def build(fit_margin_m: float = 0.015,
          wall_inset_m: float = _DEFAULT_WALL_INSET_M) -> dict:
    # Import inside to avoid an import-time dependency on the select
    # module from the build script.
    from maniguard.task_generation.utils.jar_transport_pipeline.select import (
        jar_opening_min_dim,
    )

    footprints = json.loads(_FOOTPRINTS_PATH.read_text())
    jar_entries = footprints.get(_JAR_CATEGORY, {})
    jar_models = sorted(jar_entries.keys())
    if not jar_models:
        raise RuntimeError(
            f"No {_JAR_CATEGORY} footprints found in object_footprints.json. "
            "Regenerate the footprint catalog first."
        )

    food_rows = _eligible_food_rows()
    print(f"[build] {len(food_rows)} eligible food (cat, model) rows "
          f"({len({r['category'] for r in food_rows})} categories)")

    per_jar: Dict[str, dict] = {}
    eligible_food_models: set[tuple[str, str]] = set()
    for jar_model in jar_models:
        jar_extent = list(jar_entries[jar_model]["extent_xyz"])
        opening = float(jar_opening_min_dim(jar_model, wall_inset_m=wall_inset_m))
        budget = opening - float(fit_margin_m)
        foods: Dict[str, List[str]] = {}
        n_models = 0
        for r in food_rows:
            cat = r["category"]
            model = r["model"]
            fp = footprints.get(cat, {}).get(model)
            if fp is None:
                continue
            if float(max(fp["extent_xyz"])) >= budget:
                continue
            foods.setdefault(cat, []).append(model)
            eligible_food_models.add((cat, model))
            n_models += 1
        for cat in foods:
            foods[cat].sort()
        per_jar[jar_model] = {
            "jar_extent_xyz": [float(v) for v in jar_extent],
            "jar_full_min_dim_m": float(min(jar_extent)),
            "jar_opening_min_dim_m": opening,
            "wall_inset_m": float(wall_inset_m),
            "fit_margin_m": float(fit_margin_m),
            "budget_m": budget,
            "n_food_categories": len(foods),
            "n_food_models": n_models,
            "foods": dict(sorted(foods.items())),
        }
        print(f"[build] {jar_model}: opening={opening:.3f} m, "
              f"budget={budget:.3f} m → {len(foods)} categories, "
              f"{n_models} models")

    payload = {
        "metadata": {
            "source_graspability": str(_GRASPABILITY_PATH.relative_to(_PROJECT_ROOT)),
            "source_footprints": str(_FOOTPRINTS_PATH.relative_to(_PROJECT_ROOT)),
            "fit_margin_m": float(fit_margin_m),
            "wall_inset_m": float(wall_inset_m),
            "fit_rule": ("item.extent_xyz.max() < "
                         "jar_opening_min_dim - fit_margin_m"),
            "jar_models": jar_models,
            "n_food_models_eligible": len(eligible_food_models),
            "n_food_categories_eligible": len({c for c, _ in eligible_food_models}),
        },
        "hinged_jar": per_jar,
    }
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-margin-m", type=float, default=0.015,
                   help="Additive margin (m) between food.max_dim and jar.min_dim.")
    p.add_argument("--out", type=Path, default=_OUT_PATH)
    args = p.parse_args()

    payload = build(fit_margin_m=args.fit_margin_m)
    args.out.write_text(json.dumps(payload, indent=2))
    meta = payload["metadata"]
    print(f"\n[build] wrote {args.out}")
    print(f"[build] eligible food: {meta['n_food_models_eligible']} models "
          f"in {meta['n_food_categories_eligible']} categories")


if __name__ == "__main__":
    main()
