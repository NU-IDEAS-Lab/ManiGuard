"""Render a per-episode offline-pack visualization to a PNG.

Used by ``clutter_scene_pipeline.place_objects`` to save what the
max-rectangles solver actually planned for the episode — for comparing
against what the sim ends up with after settle.

Coordinates are region-centred (the same frame the solver returns), so
the figure can be read directly against any other region-centred
diagnostic.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

_ROLE_COLORS = {
    "target": "#d62728",   # red
    "fragile": "#ff7f0e",  # orange
    "clutter": "#1f77b4",  # blue
}


def save_pack_viz(
    out_path: str,
    *,
    surface_bounds_xy,   # ((sx0, sy0), (sx1, sy1)) — world frame
    pack_region_bounds,  # ((bx0, by0), (bx1, by1)) — world frame (2cm-shrunken)
    placements,          # Iterable[PackPlacement] from maxrects_pack
    descriptors,         # Iterable[PackInputDescriptor] (for extent lookup)
    unplaced: Sequence[str],
    min_clearance: float,
    episode_label: str,
    scene_label: str,
    surface_label: str,
):
    """Render the planned pack to a PNG. Lazy-imports matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    (sx0, sy0), (sx1, sy1) = surface_bounds_xy
    (bx0, by0), (bx1, by1) = pack_region_bounds
    cx_world = 0.5 * (bx0 + bx1)
    cy_world = 0.5 * (by0 + by1)
    surface_w, surface_h = sx1 - sx0, sy1 - sy0

    extents = {d.inst_id: d.extent_xy for d in descriptors}
    roles = {d.inst_id: d.role for d in descriptors}

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.add_patch(Rectangle((sx0, sy0), surface_w, surface_h,
                           edgecolor="#888", facecolor="#eee",
                           linewidth=1.2, label="surface bounds"))
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                           edgecolor="black", facecolor="none",
                           linewidth=1.0, linestyle="--",
                           label="pack region (−2 cm buffer)"))

    pad = 0.5 * min_clearance
    for p in placements:
        w, h = extents[p.inst_id]
        if abs(p.yaw - math.pi / 2) < 0.01:
            w, h = h, w
        wx = cx_world + p.cx
        wy = cy_world + p.cy
        color = _ROLE_COLORS.get(roles.get(p.inst_id, ""), "#999")
        # padded outline (clearance budget)
        ax.add_patch(Rectangle((wx - w/2 - pad, wy - h/2 - pad),
                               w + 2*pad, h + 2*pad,
                               edgecolor=color, facecolor="none",
                               linewidth=0.8, linestyle=":", alpha=0.5))
        # actual AABB
        ax.add_patch(Rectangle((wx - w/2, wy - h/2), w, h,
                               edgecolor=color, facecolor=color,
                               alpha=0.35, linewidth=1.5))
        ax.annotate(p.inst_id, (wx, wy), ha="center", va="center",
                    fontsize=8, color="black")

    placed_count = sum(1 for _ in placements)
    total = placed_count + len(unplaced)
    title = (f"{episode_label} — {scene_label} / {surface_label}  "
             f"({surface_w:.2f}×{surface_h:.2f} m)\n"
             f"placed {placed_count}/{total}")
    if unplaced:
        title += f"  unplaced: {list(unplaced)}"
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(sx0 - 0.05, sx1 + 0.05)
    ax.set_ylim(sy0 - 0.05, sy1 + 0.05)
    ax.set_xlabel("x (m, world frame)")
    ax.set_ylabel("y (m, world frame)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
