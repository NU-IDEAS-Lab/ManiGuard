"""Export tableware-like object specs to sim_to_real.json.

For each category in the tableware pool, reads:
  - avg_category_specs.json (mass, volume, density)
  - per-model metadata.json (bbox_size, base_link_offset)
and writes a consolidated JSON that can be used to source real-world
counterparts (IKEA, etc.).

Usage:
    python -m sentinel.task_generation.build_sim_to_real \
        [--categories mug,plate,bowl] [--output sim_to_real.json]
"""

import argparse
import json
import os
import sys
import logging

log = logging.getLogger(__name__)

# Default tableware categories to export.
DEFAULT_CATEGORIES = [
    # Cups and glasses
    "mug", "coffee_cup", "teacup", "goblet", "water_glass",
    "beer_glass", "beaker", "measuring_cup",
    # Bowls and plates
    "bowl", "mixing_bowl", "plate", "saucer", "platter", "tray",
    "coaster", "china",
    # Cookware
    "saucepan", "frying_pan", "wok", "casserole", "kettle",
    # Pitchers / bottles
    "pitcher", "carafe", "wine_bottle",
    # Other containers
    "gravy_boat", "watering_can", "lid", "chopping_board",
]

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
AVG_SPECS_PATH = os.path.join(
    REPO_ROOT, "OmniGibson", "omnigibson", "configs", "avg_category_specs.json"
)
OBJECTS_ROOT = os.path.join(REPO_ROOT, "datasets", "behavior-1k-assets", "objects")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def read_model_metadata(category, model_id):
    """Read a model's metadata.json. Returns dict or None."""
    path = os.path.join(OBJECTS_ROOT, category, model_id, "misc", "metadata.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("read_model_metadata(%s) failed: %s", path, exc)
        return None


def list_models(category):
    cat_dir = os.path.join(OBJECTS_ROOT, category)
    if not os.path.isdir(cat_dir):
        return []
    return sorted(
        m for m in os.listdir(cat_dir)
        if os.path.isdir(os.path.join(cat_dir, m))
    )


def build_entry(category, avg_specs):
    """Build the sim_to_real entry for one category."""
    spec = avg_specs.get(category)
    if spec is None:
        return None

    models = list_models(category)
    model_entries = []
    for m in models:
        meta = read_model_metadata(category, m)
        if meta is None:
            continue
        bbox = meta.get("bbox_size")
        if not bbox:
            continue
        model_entries.append({
            "model": m,
            "bbox_size_m": [round(float(v), 4) for v in bbox],
            "bbox_volume_m3": round(
                float(bbox[0]) * float(bbox[1]) * float(bbox[2]), 6),
        })

    if not model_entries:
        return None

    return {
        "category": category,
        "mass_kg": round(float(spec["mass"]), 4),
        "volume_m3": round(float(spec["volume"]), 6),
        "density_kg_m3": round(float(spec["density"]), 2),
        "n_models": len(model_entries),
        "models": model_entries,
        "sim_to_real_notes": {
            "typical_retailer": "",  # to be filled in manually (IKEA, Target, etc.)
            "product_url": "",
            "real_mass_kg": None,
            "real_dimensions_m": None,
            "notes": "",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", default=None,
                   help="Comma-separated list (default: tableware pool)")
    p.add_argument("--output", default=None,
                   help="Output path (default: sim_to_real.json next to script)")
    args = p.parse_args()

    categories = (
        args.categories.split(",") if args.categories else DEFAULT_CATEGORIES
    )

    with open(AVG_SPECS_PATH) as f:
        avg_specs = json.load(f)

    missing = [c for c in categories if c not in avg_specs]
    if missing:
        log(f"Warning: categories not in avg_specs: {missing}")

    entries = {}
    skipped = []
    for cat in categories:
        entry = build_entry(cat, avg_specs)
        if entry is None:
            skipped.append(cat)
            continue
        entries[cat] = entry
        log(f"  {cat}: mass={entry['mass_kg']}kg, "
            f"density={entry['density_kg_m3']}kg/m³, "
            f"n_models={entry['n_models']}")

    if skipped:
        log(f"\nSkipped (no models or no avg spec): {skipped}")

    output = {
        "source": {
            "avg_specs": "behavior-1k/OmniGibson/omnigibson/configs/avg_category_specs.json",
            "metadata": "datasets/behavior-1k-assets/objects/<category>/<model>/misc/metadata.json",
        },
        "units": {
            "mass": "kg",
            "volume": "m^3",
            "density": "kg/m^3",
            "bbox": "m",
        },
        "categories": entries,
    }

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "sim_to_real.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    log(f"\nWrote {len(entries)} categories to {out_path}")


if __name__ == "__main__":
    main()
