"""Shared matplotlib helpers for point-cloud + grasp-pose PNG dumps.

Used by:
  - maniguard.rl.grasps.inspect_mesh         (point cloud only)
  - maniguard.rl.grasps.visualize_grasps      (grasps overlaid on cloud)
  - maniguard.rl.grasps.render_grasps         (per-object PNGs alongside
                                              the success/failure MP4)

We render with matplotlib's Agg backend because it's already installed
in the behavior conda env and renders without a GPU display, which keeps
us off the cuRobo / OG GPU path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_VIEWS = (
    ("top",   dict(elev=89, azim=-90)),
    ("iso",   dict(elev=25, azim=-60)),
    ("front", dict(elev=0,  azim=-90)),
    ("side",  dict(elev=0,  azim=0)),
)


def render_point_cloud_views(pts: np.ndarray, out_dir: Path, stem: str,
                             image_size: int = 720,
                             suffix: str = "pcd") -> list[Path]:
    """Save 4 fixed-camera PNGs of an (N, 3) point cloud.

    Color is by Z so depth structure reads in flat 2-D projections.
    File names: ``{stem}_{suffix}_{view}.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    half = (bbox_max - bbox_min).max() * 0.55

    z_norm = (pts[:, 2] - bbox_min[2]) / max(bbox_max[2] - bbox_min[2], 1e-9)

    paths = []
    for label, view in _VIEWS:
        fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=z_norm, cmap="viridis",
                   s=2.0, alpha=0.85, edgecolors="none")
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(**view)
        ax.set_axis_off()
        path = out_dir / f"{stem}_{suffix}_{label}.png"
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0,
                    facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def render_grasp_views(pts: np.ndarray, eef_pos: np.ndarray,
                       approach: np.ndarray, scores: np.ndarray,
                       out_dir: Path, stem: str,
                       image_size: int = 720,
                       approach_len: float = 0.04,
                       suffix: str = "grasps") -> list[Path]:
    """Save 4 fixed-camera PNGs of grasps overlaid on the point cloud.

    Each grasp is drawn as one line segment along its approach axis,
    colored by ``scores`` (viridis). The cloud is gray so the grasp
    overlay reads clearly. File names: ``{stem}_{suffix}_{view}.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    half = (bbox_max - bbox_min).max() * 0.65

    seg_starts = eef_pos
    seg_ends = eef_pos + approach * approach_len
    segments = np.stack([seg_starts, seg_ends], axis=1)

    score_norm = (scores - scores.min()) / max(
        scores.max() - scores.min(), 1e-9)
    colors = cm.viridis(score_norm)

    paths = []
    for label, view in _VIEWS:
        fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c="gray", s=1.0, alpha=0.25, edgecolors="none")
        lc = Line3DCollection(segments, colors=colors, linewidths=0.7)
        ax.add_collection3d(lc)
        ax.scatter(seg_ends[:, 0], seg_ends[:, 1], seg_ends[:, 2],
                   c=colors, s=2.0, edgecolors="none")
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(**view)
        ax.set_axis_off()
        path = out_dir / f"{stem}_{suffix}_{label}.png"
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0,
                    facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths
