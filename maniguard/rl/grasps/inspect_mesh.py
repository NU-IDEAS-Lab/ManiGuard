#!/usr/bin/env python3
"""Render the raw collision/visual mesh of one or more BEHAVIOR-1K objects
as flat-shaded PNGs (top + side views), no sim camera, no robot, no
texture — so you can eyeball whether the mesh has a real cavity for
the gripper.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m maniguard.rl.grasps.inspect_mesh \\
            --targets alarm_clock:cvknrh alarm_clock:trwyaq alarm_clock:vqwovi
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", required=True,
                   help="One or more 'category:model' pairs.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_datasets/survey/inspect"))
    p.add_argument("--use-collision", action="store_true",
                   help="Render the collision mesh instead of the visual mesh.")
    p.add_argument("--image-size", type=int, default=720)
    p.add_argument("--also-save-ply", action="store_true",
                   help="Also dump the trimesh as a .ply alongside the PNGs.")
    p.add_argument("--point-cloud", type=int, default=0, metavar="N",
                   help="If >0, also sample N points via "
                        "trimesh.sample.sample_surface (same path GraspGen "
                        "client uses) and render them as scatter PNGs "
                        "({stem}_pcd_{view}.png).")
    return p.parse_args()


def _parse_targets(raw):
    out = []
    for s in raw:
        if ":" not in s:
            raise SystemExit(f"bad --targets {s!r}")
        c, m = s.split(":", 1)
        out.append((c.strip(), m.strip()))
    return out


def _build_env_config():
    """Empty-ish scene for cheap mesh extraction. No robot needed."""
    return dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[],
        objects=[],
        task=dict(type="DummyTask"),
    )


def _render_mesh_views(mesh, out_dir: Path, stem: str, image_size: int):
    """Save top (looking down -Z) and side (looking +Y) flat-shaded PNGs.

    Uses matplotlib Poly3DCollection because it's already installed in the
    behavior env and renders without a GPU display. The mesh is auto-fit
    in a 1.05× bbox cube so all three views share the same scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tris = verts[faces]  # (F, 3, 3)

    # Per-triangle simple lambertian shading from a fixed +Z light.
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    light = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    shading = np.clip((norm @ light + 1.0) * 0.5, 0.15, 1.0)
    rgba = np.stack([0.6 * shading, 0.7 * shading, 0.85 * shading,
                     np.ones_like(shading)], axis=1)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    half = (bbox_max - bbox_min).max() * 0.55  # 1.1× margin

    views = [
        ("top", dict(elev=89, azim=-90)),    # looking straight down (-Z)
        ("side", dict(elev=10, azim=-60)),   # 3/4 perspective
        ("front", dict(elev=0, azim=-90)),   # camera on +Y looking -Y
    ]
    paths = []
    for label, view in views:
        fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        coll = Poly3DCollection(tris, facecolors=rgba, edgecolors="none",
                                linewidths=0.0)
        ax.add_collection3d(coll)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(**view)
        ax.set_axis_off()
        path = out_dir / f"{stem}_{label}.png"
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0,
                    facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    targets = _parse_targets(args.targets)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og
    import torch as th
    from omnigibson.objects import DatasetObject

    from maniguard.rl.grasps.mesh import mesh_from_og_object

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG ...", flush=True)
    env = og.Environment(configs=_build_env_config())
    env.reset()

    for cat, mdl in targets:
        stem = f"{cat}_{mdl}_{'col' if args.use_collision else 'vis'}"
        print(f"\n[{time.strftime('%H:%M:%S')}] {cat}/{mdl}", flush=True)
        name = f"inspect_{cat}_{mdl}"
        try:
            obj = DatasetObject(name=name, category=cat, model=mdl)
            env.scene.add_object(obj)
            obj.set_position_orientation(
                position=th.tensor([0.0, 0.0, 1.0], dtype=th.float32),
                orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
            )
            for _ in range(2):
                og.sim.step()
        except Exception as e:  # noqa: BLE001
            print(f"  spawn failed: {e}", flush=True)
            continue

        try:
            mesh = mesh_from_og_object(obj, use_visual=not args.use_collision)
            print(f"  mesh: V={len(mesh.vertices)} F={len(mesh.faces)} "
                  f"extents={[round(float(x), 3) for x in mesh.extents]}",
                  flush=True)
            if args.also_save_ply:
                ply_path = out_dir / f"{stem}.ply"
                mesh.export(str(ply_path))
                print(f"  wrote {ply_path.name}", flush=True)
            paths = _render_mesh_views(mesh, out_dir, stem, args.image_size)
            for p in paths:
                print(f"  wrote {p.name}", flush=True)
            if args.point_cloud > 0:
                import trimesh
                from maniguard.rl.grasps._viz_helpers import (
                    render_point_cloud_views,
                )
                pts, _ = trimesh.sample.sample_surface(mesh, args.point_cloud)
                pts = np.asarray(pts, dtype=np.float32)
                print(f"  point cloud: N={len(pts)}", flush=True)
                pcd_paths = render_point_cloud_views(
                    pts, out_dir, stem, args.image_size)
                for p in pcd_paths:
                    print(f"  wrote {p.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ! {cat}/{mdl} failed: {exc}", flush=True)
        finally:
            try:
                env.scene.remove_object(obj)
            except Exception:  # noqa: BLE001
                pass

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. dir={out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
