#!/usr/bin/env python3
"""Spin up OG, load a benchmark scene, extract the target object's mesh to PLY.

Also runs the antipodal grasp sampler with Franka params and saves a visualization
PNG, so we can eyeball whether the sampler produces sensible candidates on real
meshes (not synthetic).

Usage:
    cd /home/nu-ideas-4080/Desktop/projects/SENTINEL-Lite
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \
        python scripts/export_target_mesh.py \
            --scene-dir datasets/safety-benchmark/clutter_goblet_00 \
            --output-dir outputs/grasp_sampler_viz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Export target object mesh + antipodal viz.")
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_sampler_viz"))
    p.add_argument("--num-candidates-viz", type=int, default=80)
    p.add_argument("--use-collision-mesh", action="store_true",
                   help="Use the object's collision mesh instead of the visual mesh.")
    return p.parse_args()


def render_grasp_viz(mesh, grasps, gripper, out_dir: Path, subset: int = 80,
                     gripper_overlay_count: int = 6, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grasps), size=min(subset, len(grasps)), replace=False)
    show = grasps[idx]

    # Pick gripper-overlay candidates spanning different approach directions
    # via farthest-point sampling on the approach unit-vector, then keep only
    # those close to the object (small distance between fingertip and mesh
    # surface) so fingers actually straddle geometry.
    origins = grasps[:, :3, 3]
    approach = grasps[:, :3, 2]  # local +Z rotated into object frame
    # Fingertip estimate (eef origin + finger_offset along approach)
    fingertip = origins + 0.10 * approach
    # Distance from fingertip to nearest mesh surface
    closest_pts, surface_dist, _ = mesh.nearest.on_surface(fingertip)
    # Keep candidates whose fingertip is within 1cm of surface
    near_surface = np.where(surface_dist < 0.01)[0]
    if len(near_surface) < gripper_overlay_count:
        near_surface = np.argsort(surface_dist)[: max(gripper_overlay_count * 4, 20)]

    # Farthest-point sampling on approach direction unit-vector so the overlays
    # don't all look alike.
    pool = near_surface.copy()
    picked = [pool[int(rng.integers(len(pool)))]]
    for _ in range(gripper_overlay_count - 1):
        # min distance from each pool candidate to any already-picked approach
        dir_dists = np.min(
            np.linalg.norm(approach[pool][:, None, :] - approach[picked][None, :, :], axis=-1),
            axis=1,
        )
        picked.append(pool[int(np.argmax(dir_dists))])
    overlay_grasps = grasps[picked]
    print(f"  gripper overlay: picked {len(picked)} grasps (surface_dist range: "
          f"{surface_dist[picked].min():.3f}-{surface_dist[picked].max():.3f}m)")

    # Distinct color per overlay for clarity
    overlay_colors = ['orange', 'limegreen', 'violet', 'gold', 'deepskyblue', 'coral',
                     'yellow', 'magenta'][:len(overlay_grasps)]

    def plot_frame(ax, T, length=0.012, lw=0.7):
        o = T[:3, 3]; x, y, z = T[:3, 0]*length, T[:3, 1]*length, T[:3, 2]*length
        ax.plot([o[0], o[0]+x[0]], [o[1], o[1]+x[1]], [o[2], o[2]+x[2]], c='r', lw=lw)
        ax.plot([o[0], o[0]+y[0]], [o[1], o[1]+y[1]], [o[2], o[2]+y[2]], c='g', lw=lw)
        ax.plot([o[0], o[0]+z[0]], [o[1], o[1]+z[1]], [o[2], o[2]+z[2]], c='b', lw=lw)

    def plot_gripper(ax, T, color='orange', alpha=0.5):
        """Transform the gripper triangles by T and draw them."""
        # Homogeneous transform of vertices
        verts = gripper.vertices
        homog = np.hstack([verts, np.ones((len(verts), 1))])
        verts_world = (homog @ T.T)[:, :3]
        # Build triangle array using face indices
        tris = verts_world[gripper.faces]
        ax.add_collection3d(Poly3DCollection(
            tris, alpha=alpha, edgecolor='darkorange', linewidth=0.3, facecolor=color,
        ))

    def render(view_angle, fname, *, overlay: bool):
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(111, projection='3d')
        poly = Poly3DCollection(mesh.triangles, alpha=0.25, edgecolor='gray',
                                linewidth=0.1, facecolor='lightblue')
        ax.add_collection3d(poly)
        if overlay:
            for T, col in zip(overlay_grasps, overlay_colors):
                plot_gripper(ax, T, color=col, alpha=0.55)
                plot_frame(ax, T, length=0.025, lw=1.2)
        else:
            for T in show:
                plot_frame(ax, T)
        b = mesh.bounds
        pad = max(0.10, 0.5 * float(np.max(mesh.extents)))
        ax.set_xlim(b[0, 0]-pad, b[1, 0]+pad)
        ax.set_ylim(b[0, 1]-pad, b[1, 1]+pad)
        ax.set_zlim(b[0, 2]-0.02, b[1, 2]+pad)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        title = (f"{'gripper overlay' if overlay else 'coord frames'}  "
                 f"(shown={len(overlay_grasps) if overlay else len(show)} / "
                 f"total={len(grasps)})")
        ax.set_title(title)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        return fname

    return [
        render((15, 30), str(out_dir / "real_oblique.png"), overlay=False),
        render((0, 0),   str(out_dir / "real_side.png"),    overlay=False),
        render((85, 0),  str(out_dir / "real_top.png"),     overlay=False),
        render((15, 30), str(out_dir / "real_oblique_gripper.png"), overlay=True),
        render((0, 0),   str(out_dir / "real_side_gripper.png"),    overlay=True),
        render((85, 0),  str(out_dir / "real_top_gripper.png"),     overlay=True),
    ]


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Fail fast if OG unavailable — the import is heavy.
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og
    import trimesh

    from sentinel.rl.config import build_config
    from sentinel.rl.resets.grasp_sampler import AntipodalConfig, sample_antipodal_grasps
    from sentinel.rl.resets.mesh_loader import (
        franka_gripper_params, gripper_mesh_local_to_eef, mesh_from_og_object,
    )

    print(f"Building config from {args.scene_dir}")
    cfg = build_config(args.scene_dir)
    target_name = cfg["task"]["obj_name"]
    print(f"Target: {target_name}")

    print("Starting OmniGibson sim (this takes ~30s)...")
    env = og.Environment(configs=cfg)
    env.reset()

    target = env.scene.object_registry("name", target_name)
    if target is None:
        raise RuntimeError(f"Target {target_name!r} not found in scene.")

    print(f"Extracting {'visual' if not args.use_collision_mesh else 'collision'} mesh from {target.name}...")
    mesh = mesh_from_og_object(target, use_visual=not args.use_collision_mesh)
    print(f"  vertices={len(mesh.vertices)}  faces={len(mesh.faces)}  "
          f"extents={mesh.extents.tolist()}  bounds={mesh.bounds.tolist()}")

    ply_path = out / f"{target_name}.ply"
    mesh.export(str(ply_path))
    print(f"  wrote {ply_path}")

    print("Running antipodal grasp sampler (Franka params)...")
    gp = franka_gripper_params()
    acfg = AntipodalConfig(
        num_surface_samples=64, num_orientations=16, num_standoff_samples=8,
        top_bias=False, **gp,
    )
    grasps = sample_antipodal_grasps(mesh, acfg)
    print(f"  total candidates: {len(grasps)} (upper bound {acfg.num_surface_samples*acfg.num_orientations*acfg.num_standoff_samples})")

    npy_path = out / f"{target_name}_grasps_local.npy"
    np.save(str(npy_path), grasps)
    print(f"  wrote {npy_path}")

    print("Extracting real Franka gripper mesh (panda_hand + fingers)...")
    robot = env.robots[0]
    gripper = gripper_mesh_local_to_eef(robot, use_visual=True)
    print(f"  gripper vertices={len(gripper.vertices)}  faces={len(gripper.faces)}  extents={gripper.extents.tolist()}")
    gripper_ply = out / "franka_gripper.ply"
    gripper.export(str(gripper_ply))
    print(f"  wrote {gripper_ply}")

    print("Rendering viz...")
    files = render_grasp_viz(mesh, grasps, gripper, out, subset=args.num_candidates_viz)
    for f in files:
        print(f"  wrote {f}")

    print("\nDone. Exiting.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
