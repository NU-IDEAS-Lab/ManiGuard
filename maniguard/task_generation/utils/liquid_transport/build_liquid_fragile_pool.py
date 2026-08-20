"""Build ``liquid_fragile_pool.json`` — the clutter-pipeline fragile pool,
minus categories that crash under GPU dynamics.

Same geometric filter as ``clutter_pipeline.build_fragile_pool`` (tall,
narrow, column-like graspable objects), with one additional exclusion:

  * BEHAVIOR taxonomy abilities must NOT include any of
    ``{particleSource, particleSink, particleApplier, particleRemover}``.

Why exclude them: the liquid pipeline runs ``gm.USE_GPU_DYNAMICS = True``.
Under GPU dynamics, ``ParticleApplier._initialize`` (called eagerly when
any of those abilities is present) reads ``self.obj.aabb`` to size the
particle spawn grid. For a just-added prim physx hasn't yet written its
pose into the GPU buffer, so ``obj.aabb`` collapses to a single point
and the upstream "too small to sample any particle of radius X" assert
fires. The clutter pipeline runs CPU dynamics where the particle system
isn't force-initialized, so it never hits this — those bottles
(``hot_sauce_bottle``, ``*_atomizer``, ``soy_sauce_bottle``, …) are
perfectly valid clutter there and remain in the clutter fragile pool.

Output schema matches the clutter pools (category-keyed → synset +
model list) so ``select.py`` can use the same uniform-by-category →
uniform-by-model sampling.

Run:
    conda activate behavior
    python -m maniguard.task_generation.utils.liquid_transport.build_liquid_fragile_pool
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

from bddl.object_taxonomy import ObjectTaxonomy

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
CSV_PATH = os.path.join(_REPO, "docs", "graspability_classified.csv")
FOOTPRINTS_PATH = os.path.join(_HERE, "..", "object_footprints.json")
OUT_PATH = os.path.join(_HERE, "liquid_fragile_pool.json")

# Mirror ``clutter_pipeline.build_fragile_pool`` thresholds. Keep these
# in sync; the only divergence is the particle-modifier exclusion below.
Z_MIN_M = 0.10
ASPECT_MIN = 3.0
MIN_THICKNESS_M = 0.03
MAX_XY_RATIO = 3.0

_PARTICLE_MODIFIER_ABILITIES = (
    "particleSource", "particleSink", "particleApplier", "particleRemover",
)


def main():
    tax = ObjectTaxonomy()
    with open(FOOTPRINTS_PATH) as f:
        footprints = json.load(f)

    graspable: set[tuple[str, str]] = set()
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "graspable":
                graspable.add((row["category"], row["model"]))

    pool: dict[str, list[str]] = defaultdict(list)
    dropped_particle_mod: list[str] = []
    for cat, model in graspable:
        info = footprints.get(cat, {}).get(model)
        if info is None:
            continue
        ex, ey, ez = info["extent_xyz"]
        min_xy, max_xy = min(ex, ey), max(ex, ey)
        if ez < Z_MIN_M or min_xy < MIN_THICKNESS_M:
            continue
        if max_xy / min_xy > MAX_XY_RATIO:
            continue
        if ez / min_xy < ASPECT_MIN:
            continue
        pool[cat].append(model)

    out: dict[str, dict] = {}
    skipped_no_synset: list[str] = []
    for cat in sorted(pool):
        synset = tax.get_synset_from_category(cat)
        if synset is None:
            skipped_no_synset.append(cat)
            continue
        abilities = set(tax.get_abilities(synset).keys())
        if abilities & set(_PARTICLE_MODIFIER_ABILITIES):
            dropped_particle_mod.append(cat)
            continue
        out[cat] = {"synset": synset, "models": sorted(pool[cat])}

    payload = {
        "metadata": {
            "source_csv": os.path.relpath(CSV_PATH, _REPO),
            "footprint_catalog": os.path.relpath(FOOTPRINTS_PATH, _REPO),
            "filter": (
                f"status=graspable AND extent_z >= {Z_MIN_M} AND "
                f"min(extent_x, extent_y) >= {MIN_THICKNESS_M} AND "
                f"max(extent_x, extent_y) / min(extent_x, extent_y) <= "
                f"{MAX_XY_RATIO} AND "
                f"extent_z / min(extent_x, extent_y) >= {ASPECT_MIN} AND "
                f"BEHAVIOR abilities ∩ {list(_PARTICLE_MODIFIER_ABILITIES)} == ∅"
            ),
            "z_min_m": Z_MIN_M,
            "aspect_min": ASPECT_MIN,
            "min_thickness_m": MIN_THICKNESS_M,
            "max_xy_ratio": MAX_XY_RATIO,
            "particle_modifier_abilities_excluded": list(_PARTICLE_MODIFIER_ABILITIES),
            "categories": len(out),
            "models": sum(len(v["models"]) for v in out.values()),
            "dropped_particle_modifier": sorted(dropped_particle_mod),
            "skipped_no_synset": sorted(skipped_no_synset),
        },
    }
    payload.update(out)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    rel = os.path.relpath(OUT_PATH, _REPO)
    print(f"wrote {rel}: {payload['metadata']['categories']} categories, "
          f"{payload['metadata']['models']} models "
          f"(dropped {len(dropped_particle_mod)} particle-modifier cats)")


if __name__ == "__main__":
    main()
