"""Visualize the offline max-rectangles pack for a given object set + surface.

Reads object AABBs from ``object_footprints.json`` and surface bounds from
``placeable_surfaces_v1.json`` so the visualization uses exactly the same
inputs the pipeline does. Saves a PNG showing:

  * The full surface region (light grey).
  * The 2 cm shrunken pack region (dashed black outline).
  * Each placed object as a coloured rectangle (target = red, fragile =
    orange, clutter = blue), labelled with ``{category}/{model}``.
  * Padded AABB (faded outline) so you can see the clearance budget.
  * Unplaced objects listed in the figure title.

Run examples
------------
# Episode 1 of the validation sim (Wainscott_0_garden countertop tpuwys):
python tools/viz_offline_pack.py \\
    --surface-cat countertop --surface-model tpuwys \\
    --target shampoo_dispenser/nrthyl \\
    --fragile vase/hkwtnf goblet/nawrfs wineglass/adiwil wineglass/akusda \\
    --clutter scoop/oaghrf mug/yxaapv

# Episode 4 (the bottle_of_beer failure case):
python tools/viz_offline_pack.py \\
    --surface-cat breakfast_table --surface-model iaritq \\
    --target bottle_of_beer/ikgezm \\
    --fragile vase/hkwtnf vase/nuqzjs wineglass/adiwil wineglass/akusda \\
    --clutter box_of_granola_bars/bqeeki hardback/qomarm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentinel.utils.maxrects_pack import PackInputDescriptor, solve_pack  # noqa: E402

_ROLE_COLORS = {"target": "#d62728", "fragile": "#ff7f0e", "clutter": "#1f77b4"}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _parse_picks(items):
    out = []
    for item in items or []:
        cat, model = item.split("/", 1)
        out.append((cat, model))
    return out


def _lookup_extent(footprints, cat, model):
    entry = footprints.get(cat, {}).get(model)
    if entry is None:
        raise SystemExit(f"object_footprints.json has no {cat}/{model}; "
                         "run build_object_footprints to refresh.")
    return tuple(entry["extent_xyz"])


def _lookup_surface(surfaces_doc, surface_cat, surface_model, region_id):
    for entry in surfaces_doc["surfaces"]:
        if (entry["category"] == surface_cat
                and entry["model"] == surface_model
                and entry["region_id"] == region_id):
            return entry
    raise SystemExit(
        f"placeable_surfaces_v1.json has no {surface_cat}/{surface_model}/{region_id}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--surface-cat", required=True)
    parser.add_argument("--surface-model", required=True)
    parser.add_argument("--surface-region", default="region_00")
    parser.add_argument("--target", required=True, help="cat/model")
    parser.add_argument("--fragile", nargs="*", default=[], help="cat/model …")
    parser.add_argument("--clutter", nargs="*", default=[], help="cat/model …")
    parser.add_argument("--min-clearance", type=float, default=0.008)
    parser.add_argument("--edge-buffer", type=float, default=0.02)
    parser.add_argument("--out", default="/tmp/offline_pack_viz.png")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    footprints_path = os.path.join(repo_root, "sentinel/task_generation/utils/object_footprints.json")
    surfaces_path = os.path.join(repo_root, "sentinel/task_generation/utils/placeable_surfaces_v1.json")
    footprints = _load_json(footprints_path)
    surfaces = _load_json(surfaces_path)

    surf = _lookup_surface(surfaces, args.surface_cat, args.surface_model, args.surface_region)
    (sx0, sy0), (sx1, sy1) = surf["xy_min"], surf["xy_max"]
    surface_w, surface_h = sx1 - sx0, sy1 - sy0

    # Build descriptors. Use uniform bottom_offset_z=0 (z is irrelevant for the
    # 2D viz). Inst ids match the format the pipeline would generate.
    target_cat, target_model = args.target.split("/", 1)
    fragiles = _parse_picks(args.fragile)
    clutters = _parse_picks(args.clutter)

    counters = {}
    def _next(cat):
        counters[cat] = counters.get(cat, 0) + 1
        return f"{cat}_{counters[cat]}"

    descriptors = []
    target_inst = _next(target_cat)
    descriptors.append(PackInputDescriptor(
        inst_id=target_inst, role="target",
        extent_xy=_lookup_extent(footprints, target_cat, target_model)[:2],
        bottom_offset_z=0.0,
    ))
    for cat, model in fragiles:
        descriptors.append(PackInputDescriptor(
            inst_id=_next(cat), role="fragile",
            extent_xy=_lookup_extent(footprints, cat, model)[:2],
            bottom_offset_z=0.0,
        ))
    for cat, model in clutters:
        descriptors.append(PackInputDescriptor(
            inst_id=_next(cat), role="clutter",
            extent_xy=_lookup_extent(footprints, cat, model)[:2],
            bottom_offset_z=0.0,
        ))

    # Shrink by edge-buffer on every side.
    bx0, by0 = sx0 + args.edge_buffer, sy0 + args.edge_buffer
    bx1, by1 = sx1 - args.edge_buffer, sy1 - args.edge_buffer
    region_w, region_h = max(0.0, bx1 - bx0), max(0.0, by1 - by0)

    sol = solve_pack(
        descriptors=descriptors,
        region_bounds=((0.0, 0.0), (region_w, region_h)),
        min_clearance=args.min_clearance,
        target_inst_id=target_inst,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 11))

    # Full surface bounds (light grey fill).
    ax.add_patch(Rectangle((sx0, sy0), surface_w, surface_h,
                           edgecolor="#888", facecolor="#eee", linewidth=1.2,
                           label="surface bounds"))
    # 2 cm shrunken pack region (dashed outline).
    ax.add_patch(Rectangle((bx0, by0), region_w, region_h,
                           edgecolor="black", facecolor="none",
                           linewidth=1.0, linestyle="--",
                           label=f"pack region (−{args.edge_buffer*100:.0f} cm buffer)"))

    # Region origin = pack origin in world coords (region centroid).
    cx_world = 0.5 * (bx0 + bx1)
    cy_world = 0.5 * (by0 + by1)

    # Each placement, drawn at world coords.
    for p in sol.placements:
        # Find the descriptor for extents.
        d = next(d for d in descriptors if d.inst_id == p.inst_id)
        w, h = d.extent_xy
        # Apply yaw (only 0 or pi/2 from the solver).
        if abs(p.yaw - math.pi / 2) < 0.01:
            w, h = h, w
        # World center of the placement.
        wx = cx_world + p.cx
        wy = cy_world + p.cy
        # Padded outline (faded).
        pad = 0.5 * args.min_clearance
        ax.add_patch(Rectangle((wx - w/2 - pad, wy - h/2 - pad),
                               w + 2*pad, h + 2*pad,
                               edgecolor=_ROLE_COLORS[p.role], facecolor="none",
                               linewidth=0.8, linestyle=":", alpha=0.5))
        # Actual AABB (solid).
        ax.add_patch(Rectangle((wx - w/2, wy - h/2), w, h,
                               edgecolor=_ROLE_COLORS[p.role],
                               facecolor=_ROLE_COLORS[p.role], alpha=0.35,
                               linewidth=1.5))
        # Label.
        ax.annotate(f"{d.inst_id}", (wx, wy), ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_aspect("equal")
    ax.set_xlim(sx0 - 0.05, sx1 + 0.05)
    ax.set_ylim(sy0 - 0.05, sy1 + 0.05)
    ax.set_xlabel("x (m, object-local frame)")
    ax.set_ylabel("y (m, object-local frame)")
    title = (f"{args.surface_cat}/{args.surface_model}/{args.surface_region}  "
             f"({surface_w:.2f}×{surface_h:.2f} m)\n"
             f"placed {len(sol.placements)}/{len(descriptors)}")
    if sol.unplaced:
        title += f"  unplaced: {sol.unplaced}"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")
    print(f"placed={len(sol.placements)}/{len(descriptors)}, unplaced={sol.unplaced}")


if __name__ == "__main__":
    main()
