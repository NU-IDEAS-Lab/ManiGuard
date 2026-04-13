from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from omnigibson.macros import gm
from omnigibson.task_generation.pipeline_common import (
    append_jsonl,
    pipeline_exit,
    set_viewer_camera_pose,
)
from omnigibson.task_generation.support_surface_profiles import (
    Bounds2D,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROFILE_PATH,
    bounds_area,
    connected_components,
    current_timestamp_utc,
    evaluate_region_reachability,
    load_support_surface_profiles,
    mask_to_usable_regions,
    normalize_bounds,
    region_span_xy,
    save_support_surface_profiles,
    select_dominant_top_plane_height,
    set_support_surface_profile,
    transform_local_point_to_world,
    transform_world_point_to_local,
)
from omnigibson.utils.surface_discovery import is_table_like


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SURFACE_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "surface_catalog.json")


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy(), dtype=float)
    return np.asarray(values, dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile support-surface usable regions for BEHAVIOR assets.")
    parser.add_argument("--category", default=None, help="Specific support category to profile")
    parser.add_argument("--model", default=None, help="Specific model to profile (requires --category)")
    parser.add_argument("--max-models", type=int, default=None, help="Limit number of models processed")
    parser.add_argument("--grid-step-m", type=float, default=0.03)
    parser.add_argument("--top-band-size-m", type=float, default=0.02)
    parser.add_argument("--top-band-min-count-ratio", type=float, default=0.2)
    parser.add_argument("--normal-min-z", type=float, default=0.9)
    parser.add_argument("--min-component-cells", type=int, default=4)
    parser.add_argument("--min-region-area-m2", type=float, default=0.01)
    parser.add_argument("--ray-start-margin-m", type=float, default=0.15)
    parser.add_argument("--ray-end-margin-m", type=float, default=0.20)
    parser.add_argument("--mount-gap-m", type=float, default=0.03)
    parser.add_argument("--robot-radius-m", type=float, default=0.24)
    parser.add_argument("--robot-half-extent-x", type=float, default=0.15)
    parser.add_argument("--robot-half-extent-y", type=float, default=0.15)
    parser.add_argument("--edge-margin-m", type=float, default=0.08)
    parser.add_argument("--reachable-min-m", type=float, default=0.45)
    parser.add_argument("--reachable-max-m", type=float, default=1.10)
    parser.add_argument("--spawn-z", type=float, default=1.50)
    parser.add_argument("--highlight-thickness-m", type=float, default=0.008)
    parser.add_argument("--highlight-alpha", type=float, default=0.45)
    parser.add_argument("--review-image-width", type=int, default=1280)
    parser.add_argument("--review-image-height", type=int, default=960)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-json", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--debug-jsonl", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--showcase-gui", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _load_surface_catalog() -> dict:
    if not os.path.exists(_SURFACE_CATALOG_PATH):
        return {}
    with open(_SURFACE_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_run_dir(args) -> None:
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = args.category or "all"
        args.run_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"profile_{label}_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.debug_jsonl is None:
        args.debug_jsonl = os.path.join(args.run_dir, "summary.jsonl")


def _list_profile_targets(category: Optional[str], model: Optional[str], max_models: Optional[int]) -> List[Tuple[str, str]]:
    from omnigibson.utils.asset_utils import get_all_object_categories, get_all_object_category_models

    if model and not category:
        raise ValueError("--model requires --category")

    targets: List[Tuple[str, str]] = []
    categories = [category] if category else [cat for cat in get_all_object_categories() if is_table_like(cat)]
    for cat in sorted(categories):
        models = [model] if model else get_all_object_category_models(cat)
        for mod in sorted(models):
            targets.append((cat, mod))
            if max_models is not None and len(targets) >= max_models:
                return targets
    return targets


def _make_support_cfg(category: str, model: str, spawn_z: float) -> dict:
    return {
        "type": "DatasetObject",
        "name": "profile_support",
        "category": category,
        "model": model,
        "fixed_base": True,
        "position": [0.0, 0.0, float(spawn_z)],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


def _make_env_config(category: str, model: str, spawn_z: float, viewer_width: int, viewer_height: int) -> dict:
    return {
        "scene": {"type": "Scene"},
        "robots": [],
        "objects": [_make_support_cfg(category, model, spawn_z)],
        "task": {"type": "DummyTask"},
        "render": {
            "viewer_width": int(viewer_width),
            "viewer_height": int(viewer_height),
        },
    }


def _make_grid_edges(lo: float, hi: float, step_m: float) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    span = max(hi - lo, 1e-6)
    n_cells = max(1, int(np.ceil(span / float(step_m))))
    return np.linspace(lo, hi, num=n_cells + 1, dtype=float)


def _support_hit_prefixes(support_obj) -> Tuple[str, ...]:
    prefixes = set()
    prim_path = getattr(support_obj, "prim_path", None)
    if prim_path:
        prefixes.add(str(prim_path))
    root_link = getattr(support_obj, "root_link", None)
    if root_link is not None:
        root_prim = getattr(root_link, "prim_path", None)
        if root_prim:
            prefixes.add(str(root_prim))
    links = getattr(support_obj, "links", None)
    if isinstance(links, dict):
        for link in links.values():
            link_prim = getattr(link, "prim_path", None)
            if link_prim:
                prefixes.add(str(link_prim))
    return tuple(sorted(prefixes))


def _hit_matches_support(hit: dict, support_prefixes: Sequence[str]) -> bool:
    rigid_body = str(hit.get("rigidBody", "") or "")
    collision = str(hit.get("collision", "") or "")
    for prefix in support_prefixes:
        if rigid_body.startswith(prefix) or collision.startswith(prefix):
            return True
    return False


def _cell_area_sum(mask: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> float:
    widths = np.diff(x_edges)
    heights = np.diff(y_edges)
    cell_areas = heights[:, None] * widths[None, :]
    return float(cell_areas[mask].sum())


def _local_aabb_xy_from_world_bounds(aabb_min, aabb_max, position, orientation, scale) -> Bounds2D:
    local_corners = np.asarray(
        [
            transform_world_point_to_local(
                (float(aabb_min[0]), float(aabb_min[1]), float(position[2])),
                position_xyz_world=position,
                orientation_xyzw_world=orientation,
                scale_xyz=scale,
            ),
            transform_world_point_to_local(
                (float(aabb_min[0]), float(aabb_max[1]), float(position[2])),
                position_xyz_world=position,
                orientation_xyzw_world=orientation,
                scale_xyz=scale,
            ),
            transform_world_point_to_local(
                (float(aabb_max[0]), float(aabb_min[1]), float(position[2])),
                position_xyz_world=position,
                orientation_xyzw_world=orientation,
                scale_xyz=scale,
            ),
            transform_world_point_to_local(
                (float(aabb_max[0]), float(aabb_max[1]), float(position[2])),
                position_xyz_world=position,
                orientation_xyzw_world=orientation,
                scale_xyz=scale,
            ),
        ],
        dtype=float,
    )
    return normalize_bounds(
        (
            (float(np.min(local_corners[:, 0])), float(np.min(local_corners[:, 1]))),
            (float(np.max(local_corners[:, 0])), float(np.max(local_corners[:, 1]))),
        )
    )


def _save_review_plots(
    artifact_dir: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    raw_mask: np.ndarray,
    top_mask: np.ndarray,
    regions: Sequence[dict],
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.patches import Rectangle

    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    color_cycle = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    mask_fill_color = "#6e6e6e"
    mask_cmap = ListedColormap(["#ffffff", mask_fill_color])

    def _plot_mask(mask: np.ndarray, title: str, path: str, draw_regions: bool, mask_label: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        mask_count = int(mask.sum())
        total_cells = int(mask.size)
        ax.imshow(
            mask.astype(np.uint8),
            origin="lower",
            extent=extent,
            cmap=mask_cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            alpha=1.0,
        )
        legend_handles = [
            Patch(facecolor=mask_fill_color, edgecolor="#4d4d4d", alpha=0.92, label=mask_label),
        ]
        if draw_regions:
            for idx, region in enumerate(regions):
                x0, y0 = region["xy_min"]
                x1, y1 = region["xy_max"]
                color = color_cycle[idx % len(color_cycle)]
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=True,
                    facecolor=color,
                    alpha=0.28,
                    linewidth=1.8,
                    edgecolor=color,
                )
                ax.add_patch(rect)
                ax.text(
                    x0,
                    y1,
                    region.get("region_id", ""),
                    fontsize=7,
                    color=color,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.5},
                )
            legend_handles.append(
                Patch(facecolor="tab:red", edgecolor="tab:red", alpha=0.28, label="usable placement sub-rectangle")
            )
        ax.set_title(f"{title}\n{mask_count}/{total_cells} hit cells")
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("local y (m)")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    raw_path = os.path.join(artifact_dir, "mask_raw_hits.png")
    top_path = os.path.join(artifact_dir, "mask_top_plane.png")
    rect_path = os.path.join(artifact_dir, "mask_regions_overlay.png")
    _plot_mask(raw_mask, "Raw Support Hits", raw_path, draw_regions=False, mask_label="support raycast hit cells")
    _plot_mask(top_mask, "Dominant Top Plane", top_path, draw_regions=False, mask_label="dominant top-plane cells")
    _plot_mask(top_mask, "Usable Region Rectangles", rect_path, draw_regions=True, mask_label="dominant top-plane cells")
    return {
        "mask_raw_hits_png": os.path.relpath(raw_path, _PROJECT_ROOT),
        "mask_top_plane_png": os.path.relpath(top_path, _PROJECT_ROOT),
        "mask_regions_overlay_png": os.path.relpath(rect_path, _PROJECT_ROOT),
    }


def _local_region_center(region: dict, top_plane_z_local: float, local_thickness: float, local_epsilon: float) -> np.ndarray:
    x0, y0 = (float(v) for v in region["xy_min"])
    x1, y1 = (float(v) for v in region["xy_max"])
    return np.asarray(
        [
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
            float(top_plane_z_local) + 0.5 * float(local_thickness) + float(local_epsilon),
        ],
        dtype=float,
    )


def _save_rgb_image(path: str, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path)


def _capture_view_image(path: str, eye: Sequence[float], lookat: Sequence[float]) -> np.ndarray:
    import omnigibson as og

    set_viewer_camera_pose(eye, lookat)
    og.sim.render()
    og.sim.render()
    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    _save_rgb_image(path, rgb)
    return rgb


def _review_camera_views(local_aabb_xy, top_plane_z_local, position_np, orientation_np, scale_np) -> dict:
    (x0, y0), (x1, y1) = local_aabb_xy
    span_x_local = max(1e-3, float(x1 - x0))
    span_y_local = max(1e-3, float(y1 - y0))
    span_x_world = span_x_local * abs(float(scale_np[0]))
    span_y_world = span_y_local * abs(float(scale_np[1]))
    max_span = max(span_x_world, span_y_world, 0.3)
    local_center = np.asarray([0.5 * (x0 + x1), 0.5 * (y0 + y1), float(top_plane_z_local)], dtype=float)
    world_center = transform_local_point_to_world(
        local_center,
        position_xyz_world=position_np,
        orientation_xyzw_world=orientation_np,
        scale_xyz=scale_np,
    )

    top_eye = world_center + np.asarray([0.0, 0.0, max(0.9, 2.2 * max_span + 0.35)], dtype=float)
    oblique_eye = world_center + np.asarray(
        [-1.25 * max_span, -1.05 * max_span, max(0.75, 1.15 * max_span + 0.25)],
        dtype=float,
    )
    oblique_lookat = world_center + np.asarray([0.0, 0.0, 0.03], dtype=float)
    return {
        "topdown": {
            "eye": [float(v) for v in top_eye],
            "lookat": [float(v) for v in world_center],
        },
        "oblique": {
            "eye": [float(v) for v in oblique_eye],
            "lookat": [float(v) for v in oblique_lookat],
        },
    }


def _create_region_overlays(
    env,
    regions: Sequence[dict],
    *,
    top_plane_z_local: float,
    position_np: np.ndarray,
    orientation_np: np.ndarray,
    scale_np: np.ndarray,
    thickness_m: float,
    alpha: float,
):
    from omnigibson.objects.primitive_object import PrimitiveObject

    overlays = []
    scale_z = max(abs(float(scale_np[2])), 1e-6)
    local_thickness = float(thickness_m) / scale_z
    local_epsilon = 0.002 / scale_z
    palette = (
        th.tensor([1.0, 0.15, 0.15, alpha], dtype=th.float32),
        th.tensor([0.10, 0.80, 0.95, alpha], dtype=th.float32),
        th.tensor([0.95, 0.80, 0.10, alpha], dtype=th.float32),
        th.tensor([0.65, 0.20, 0.95, alpha], dtype=th.float32),
    )

    for idx, region in enumerate(regions):
        span_xy = region_span_xy(region) or (0.0, 0.0)
        overlay = PrimitiveObject(
            relative_prim_path=f"/support_region_overlay_{idx:02d}",
            name=f"support_region_overlay_{idx:02d}",
            primitive_type="Cube",
            size=1.0,
            scale=[
                max(1e-3, float(span_xy[0]) * abs(float(scale_np[0])) * 0.995),
                max(1e-3, float(span_xy[1]) * abs(float(scale_np[1])) * 0.995),
                max(1e-3, float(thickness_m)),
            ],
            fixed_base=True,
            visual_only=True,
            rgba=palette[idx % len(palette)],
        )
        env.scene.add_object(overlay)
        center_world = transform_local_point_to_world(
            _local_region_center(region, top_plane_z_local, local_thickness, local_epsilon),
            position_xyz_world=position_np,
            orientation_xyzw_world=orientation_np,
            scale_xyz=scale_np,
        )
        overlay.set_position_orientation(
            position=[float(v) for v in center_world],
            orientation=[float(v) for v in orientation_np],
        )
        overlays.append(overlay)
    return overlays


def _remove_overlays(env, overlays) -> None:
    for overlay in overlays:
        try:
            env.scene.remove_object(overlay)
        except Exception:
            continue


def _image_difference_stats(before: np.ndarray, after: np.ndarray) -> dict:
    if before.shape != after.shape:
        raise ValueError("before/after image shapes must match")
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    changed = np.any(diff > 8, axis=2)
    return {
        "changed_pixel_count": int(changed.sum()),
        "changed_fraction": round(float(changed.mean()), 6),
        "mean_abs_rgb_delta": round(float(diff.mean()), 6),
        "max_abs_rgb_delta": int(diff.max()),
    }


def _render_review_artifacts(
    args,
    artifact_dir: str,
    env,
    *,
    usable_regions: Sequence[dict],
    local_aabb_xy: Bounds2D,
    top_plane_z_local: float,
    position_np: np.ndarray,
    orientation_np: np.ndarray,
    scale_np: np.ndarray,
) -> Tuple[dict, dict]:
    import omnigibson as og

    views = _review_camera_views(local_aabb_xy, top_plane_z_local, position_np, orientation_np, scale_np)

    plain_top_path = os.path.join(artifact_dir, "sim_topdown.png")
    plain_oblique_path = os.path.join(artifact_dir, "sim_oblique.png")
    top_plain = _capture_view_image(plain_top_path, views["topdown"]["eye"], views["topdown"]["lookat"])
    oblique_plain = _capture_view_image(plain_oblique_path, views["oblique"]["eye"], views["oblique"]["lookat"])

    artifacts = {
        "sim_topdown_png": os.path.relpath(plain_top_path, _PROJECT_ROOT),
        "sim_oblique_png": os.path.relpath(plain_oblique_path, _PROJECT_ROOT),
    }
    checks = {
        "region_count": int(len(usable_regions)),
        "viewer_resolution": [int(args.review_image_width), int(args.review_image_height)],
        "highlight_visible": False,
    }

    if not usable_regions:
        return artifacts, checks

    overlays = _create_region_overlays(
        env,
        usable_regions,
        top_plane_z_local=top_plane_z_local,
        position_np=position_np,
        orientation_np=orientation_np,
        scale_np=scale_np,
        thickness_m=args.highlight_thickness_m,
        alpha=float(args.highlight_alpha),
    )
    try:
        og.sim.step()
        highlight_top_path = os.path.join(artifact_dir, "sim_topdown_regions.png")
        highlight_oblique_path = os.path.join(artifact_dir, "sim_oblique_regions.png")
        top_highlight = _capture_view_image(highlight_top_path, views["topdown"]["eye"], views["topdown"]["lookat"])
        oblique_highlight = _capture_view_image(highlight_oblique_path, views["oblique"]["eye"], views["oblique"]["lookat"])
    finally:
        _remove_overlays(env, overlays)
        og.sim.step()

    top_diff = _image_difference_stats(top_plain, top_highlight)
    oblique_diff = _image_difference_stats(oblique_plain, oblique_highlight)
    artifacts.update({
        "sim_topdown_regions_png": os.path.relpath(highlight_top_path, _PROJECT_ROOT),
        "sim_oblique_regions_png": os.path.relpath(highlight_oblique_path, _PROJECT_ROOT),
    })
    checks.update({
        "highlight_visible": bool(top_diff["changed_pixel_count"] > 0 and oblique_diff["changed_pixel_count"] > 0),
        "topdown_diff": top_diff,
        "oblique_diff": oblique_diff,
    })
    return artifacts, checks


def _profile_single_model(args, category: str, model: str, surface_catalog: dict) -> dict:
    import omnigibson as og
    from omnigibson.utils.sampling_utils import raytest_batch

    artifact_dir = os.path.join(args.run_dir, category, model)
    os.makedirs(artifact_dir, exist_ok=True)
    env = None
    try:
        env = og.Environment(
            configs=_make_env_config(
                category,
                model,
                args.spawn_z,
                viewer_width=args.review_image_width,
                viewer_height=args.review_image_height,
            )
        )
        env.reset()
        for _ in range(3):
            og.sim.step()
        support_obj = env.scene.object_registry("name", "profile_support")
        if support_obj is None:
            raise RuntimeError("support object not found after env init")

        aabb_min, aabb_max = support_obj.aabb
        position, orientation = support_obj.get_position_orientation()
        scale = getattr(support_obj, "scale", (1.0, 1.0, 1.0))
        scale_np = _to_numpy(scale)
        position_np = _to_numpy(position)
        orientation_np = _to_numpy(orientation)
        local_aabb_xy = _local_aabb_xy_from_world_bounds(aabb_min, aabb_max, position_np, orientation_np, scale_np)

        x_edges_world = _make_grid_edges(float(aabb_min[0]), float(aabb_max[0]), args.grid_step_m)
        y_edges_world = _make_grid_edges(float(aabb_min[1]), float(aabb_max[1]), args.grid_step_m)
        x_centers_world = 0.5 * (x_edges_world[:-1] + x_edges_world[1:])
        y_centers_world = 0.5 * (y_edges_world[:-1] + y_edges_world[1:])

        starts = []
        ends = []
        grid_index = []
        for row, y in enumerate(y_centers_world):
            for col, x in enumerate(x_centers_world):
                starts.append((float(x), float(y), float(aabb_max[2]) + args.ray_start_margin_m))
                ends.append((float(x), float(y), float(aabb_min[2]) - args.ray_end_margin_m))
                grid_index.append((row, col))

        hits = raytest_batch(starts, ends, only_closest=True)
        support_prefixes = _support_hit_prefixes(support_obj)

        raw_mask = np.zeros((len(y_centers_world), len(x_centers_world)), dtype=bool)
        local_points: dict[Tuple[int, int], np.ndarray] = {}
        local_hit_heights: List[float] = []
        for hit, (row, col) in zip(hits, grid_index):
            if not hit.get("hit", False):
                continue
            if not _hit_matches_support(hit, support_prefixes):
                continue
            normal = _to_numpy(hit.get("normal", [0.0, 0.0, 0.0]))
            if normal.shape[0] != 3 or float(normal[2]) < args.normal_min_z:
                continue
            local = transform_world_point_to_local(
                hit["position"],
                position_xyz_world=position_np,
                orientation_xyzw_world=orientation_np,
                scale_xyz=scale_np,
            )
            raw_mask[row, col] = True
            local_points[(row, col)] = local
            local_hit_heights.append(float(local[2]))

        top_plane_z_local, dominant_count = select_dominant_top_plane_height(
            local_hit_heights,
            band_size_m=args.top_band_size_m,
            min_count_ratio=args.top_band_min_count_ratio,
        )
        if top_plane_z_local is None:
            top_plane_z_local = 0.0

        top_mask = np.zeros_like(raw_mask, dtype=bool)
        for (row, col), point_local in local_points.items():
            if abs(float(point_local[2]) - float(top_plane_z_local)) <= float(args.top_band_size_m):
                top_mask[row, col] = True

        x_edges_local = np.asarray(
            [
                transform_world_point_to_local(
                    (float(x_edge), float(position_np[1]), float(position_np[2])),
                    position_xyz_world=position_np,
                    orientation_xyzw_world=orientation_np,
                    scale_xyz=scale_np,
                )[0]
                for x_edge in x_edges_world
            ],
            dtype=float,
        )
        y_edges_local = np.asarray(
            [
                transform_world_point_to_local(
                    (float(position_np[0]), float(y_edge), float(position_np[2])),
                    position_xyz_world=position_np,
                    orientation_xyzw_world=orientation_np,
                    scale_xyz=scale_np,
                )[1]
                for y_edge in y_edges_world
            ],
            dtype=float,
        )
        if top_mask.shape != (len(y_edges_local) - 1, len(x_edges_local) - 1):
            raise RuntimeError("local edge construction drifted from world grid shape")

        usable_regions = mask_to_usable_regions(
            top_mask,
            x_edges_local,
            y_edges_local,
            region_id_prefix="region",
            min_component_cells=args.min_component_cells,
            min_region_area_m2=args.min_region_area_m2,
            source="raycast_maximal_rectangles",
        )

        reachable_region_ids: List[str] = []
        reachable_edge_labels = set()
        for region in usable_regions:
            reach = evaluate_region_reachability(
                (tuple(region["xy_min"]), tuple(region["xy_max"])),
                local_aabb_xy,
                reachable_distance_range=(args.reachable_min_m, args.reachable_max_m),
                desired_gap_m=args.mount_gap_m,
                robot_radius_m=args.robot_radius_m,
                edge_margin_m=args.edge_margin_m,
                robot_half_extent_xy=(args.robot_half_extent_x, args.robot_half_extent_y),
            )
            span_xy = region_span_xy(region)
            region["reachable"] = bool(reach["reachable"])
            region["reachable_edge_labels"] = list(reach["edge_labels"])
            if span_xy is not None:
                region["span_xy_m"] = [round(float(span_xy[0]), 6), round(float(span_xy[1]), 6)]
            if reach["reachable"]:
                reachable_region_ids.append(region["region_id"])
                reachable_edge_labels.update(reach["edge_labels"])

        raw_hit_count = int(raw_mask.sum())
        top_hit_count = int(top_mask.sum())
        occupancy_area_m2 = _cell_area_sum(top_mask, x_edges_local, y_edges_local)
        effective_area_m2 = float(sum(float(region["area_m2"]) for region in usable_regions))
        aabb_area_m2 = bounds_area(local_aabb_xy)
        component_count = len([coords for coords in connected_components(top_mask) if len(coords) >= args.min_component_cells])
        region_count = len(usable_regions)
        largest_region = max(usable_regions, key=lambda region: float(region.get("area_m2", 0.0)), default=None)
        largest_region_area_m2 = float(largest_region.get("area_m2", 0.0)) if largest_region else 0.0
        largest_region_span_xy_m = (
            [round(float(v), 6) for v in (region_span_xy(largest_region) or (0.0, 0.0))]
            if largest_region else []
        )

        exclusion_reasons: List[str] = []
        if raw_hit_count == 0:
            exclusion_reasons.append("no_support_hits")
        if top_hit_count == 0:
            exclusion_reasons.append("no_dominant_top_plane")
        if not usable_regions:
            exclusion_reasons.append("no_usable_regions")

        candidate_for_generation = bool(usable_regions)
        review_artifacts = {}
        if not args.skip_plots:
            review_artifacts.update(
                _save_review_plots(artifact_dir, x_edges_local, y_edges_local, raw_mask, top_mask, usable_regions)
            )
        sim_artifacts, review_checks = _render_review_artifacts(
            args,
            artifact_dir,
            env,
            usable_regions=usable_regions,
            local_aabb_xy=local_aabb_xy,
            top_plane_z_local=float(top_plane_z_local),
            position_np=position_np,
            orientation_np=orientation_np,
            scale_np=scale_np,
        )
        review_artifacts.update(sim_artifacts)

        source_catalog = (surface_catalog.get(category) or {}).get(model)
        entry = {
            "category": category,
            "model": model,
            "nominal_scale": [1.0, 1.0, 1.0],
            "top_plane_z_local": round(float(top_plane_z_local), 6),
            "aabb_xy_local": {
                "xy_min": [round(float(local_aabb_xy[0][0]), 6), round(float(local_aabb_xy[0][1]), 6)],
                "xy_max": [round(float(local_aabb_xy[1][0]), 6), round(float(local_aabb_xy[1][1]), 6)],
            },
            "aabb_area_m2": round(float(aabb_area_m2), 6),
            "occupancy_area_m2": round(float(occupancy_area_m2), 6),
            "usable_regions": usable_regions,
            "effective_area_m2": round(float(effective_area_m2), 6),
            "candidate_for_generation": bool(candidate_for_generation),
            "exclusion_reasons": exclusion_reasons,
            "reachability": {
                "franka_mounted": {
                    "reachable": bool(reachable_region_ids),
                    "edge_labels": sorted(reachable_edge_labels),
                    "reachable_region_ids": reachable_region_ids,
                    "reachable_distance_range": [float(args.reachable_min_m), float(args.reachable_max_m)],
                }
            },
            "review_status": "auto_pending",
            "review_artifacts": review_artifacts,
            "review_checks": review_checks,
            "diagnostics": {
                "grid_step_m": float(args.grid_step_m),
                "normal_min_z": float(args.normal_min_z),
                "raw_support_hit_count": raw_hit_count,
                "dominant_top_plane_hit_count": top_hit_count,
                "dominant_top_plane_band_count": int(dominant_count),
                "component_count": int(component_count),
                "region_count": int(region_count),
                "reachable_region_count": len(reachable_region_ids),
                "largest_region_area_m2": round(float(largest_region_area_m2), 6),
                "largest_region_span_xy_m": largest_region_span_xy_m,
                "usable_to_aabb_ratio": round(float(effective_area_m2 / max(aabb_area_m2, 1e-6)), 6),
                "occupancy_to_aabb_ratio": round(float(occupancy_area_m2 / max(aabb_area_m2, 1e-6)), 6),
                "region_cover_to_occupancy_ratio": round(float(effective_area_m2 / max(occupancy_area_m2, 1e-6)), 6),
                "dropped_occupancy_area_m2": round(max(0.0, float(occupancy_area_m2 - effective_area_m2)), 6),
                "artifact_dir": os.path.relpath(artifact_dir, _PROJECT_ROOT),
            },
            "provenance": {
                "method": "support_surface_profiler_v2",
                "computed_at": current_timestamp_utc(),
                "source_catalog": source_catalog,
            },
        }

        with open(os.path.join(artifact_dir, "profile_summary.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=True)
            f.write("\n")

        return entry
    finally:
        if env is not None:
            env.close()
        try:
            import omnigibson as og

            if og.sim is not None:
                if not og.sim.is_stopped():
                    og.sim.stop()
                og.clear()
        except Exception:
            pass


def main():
    args = parse_args()
    _resolve_run_dir(args)

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_OBJECT_STATES = True
    gm.RENDER_VIEWER_CAMERA = True
    gm.HEADLESS = not args.showcase_gui

    surface_catalog = _load_surface_catalog()
    document = load_support_surface_profiles(args.output_json)
    targets = _list_profile_targets(args.category, args.model, args.max_models)
    if not targets:
        print("[Profiler] No targets found.")
        pipeline_exit(1)

    print(f"[Profiler] Targets: {len(targets)}")
    print(f"[Profiler] Output JSON: {os.path.abspath(args.output_json)}")
    print(f"[Profiler] Run dir: {args.run_dir}")
    sys.stdout.flush()

    exit_code = 0
    success_count = 0
    for idx, (category, model) in enumerate(targets, start=1):
        existing = load_support_surface_profiles(args.output_json).get("profiles", {}).get(category, {}).get(model)
        if existing is not None and not args.overwrite:
            print(f"[Profiler] Skip existing {idx}/{len(targets)}: {category}/{model}")
            append_jsonl(args.debug_jsonl, {
                "event": "skip_existing",
                "category": category,
                "model": model,
            })
            continue

        print(f"[Profiler] Profiling {idx}/{len(targets)}: {category}/{model}")
        sys.stdout.flush()
        try:
            entry = _profile_single_model(args, category, model, surface_catalog)
            document = load_support_surface_profiles(args.output_json)
            document = set_support_surface_profile(document, category, model, entry)
            save_support_surface_profiles(document, args.output_json)
            append_jsonl(args.debug_jsonl, {
                "event": "profile_success",
                "category": category,
                "model": model,
                "candidate_for_generation": entry["candidate_for_generation"],
                "effective_area_m2": entry["effective_area_m2"],
                "usable_region_count": len(entry["usable_regions"]),
                "exclusion_reasons": list(entry["exclusion_reasons"]),
                "highlight_visible": entry.get("review_checks", {}).get("highlight_visible"),
            })
            success_count += 1
        except Exception as exc:
            exit_code = 1
            print(f"[Profiler] ERROR {category}/{model}: {exc}")
            traceback.print_exc()
            append_jsonl(args.debug_jsonl, {
                "event": "profile_error",
                "category": category,
                "model": model,
                "error": str(exc),
            })

    print(f"[Profiler] Completed: success={success_count}/{len(targets)}")
    sys.stdout.flush()
    pipeline_exit(exit_code)


if __name__ == "__main__":
    main()
