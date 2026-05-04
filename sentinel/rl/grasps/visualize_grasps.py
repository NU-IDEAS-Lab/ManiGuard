#!/usr/bin/env python3
"""Render the GraspGen-predicted grasp poses for one or more BEHAVIOR-1K
objects as scatter+frustum overlays on the sampled point cloud.

For each ``category:model`` target this script:
  1. boots a minimal OG to extract the visual mesh,
  2. samples the same 8000-point cloud the GraspGen client uses,
  3. queries the GraspGen ZMQ server for ``--num-grasps`` candidates and
     keeps the top ``--topk`` by discriminator confidence,
  4. renders top/iso/front/side PNGs of the cloud (gray) with each grasp
     as a short line along its approach axis (color = confidence,
     viridis), saved next to the point-cloud PNGs.

Why a separate script rather than a flag on ``inspect_mesh``: this needs
the live ZMQ client and adds enough plotting machinery that bolting it
on to the mesh-only inspector would clutter both code paths. Sharing the
boot path costs ~30s either way.

Usage::

    DISPLAY=:0 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.grasps.visualize_grasps \\
            --targets alphabet_abacus:wojwvu --topk 1000
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
                   default=Path("outputs/grasp_datasets/inspect_grasps"))
    p.add_argument("--n-points", type=int, default=8000,
                   help="Surface points sampled from the mesh and sent to "
                        "GraspGen (matches the render pipeline default).")
    p.add_argument("--num-grasps", type=int, default=2000,
                   help="Diffusion samples drawn server-side per inference.")
    p.add_argument("--topk", type=int, default=1000,
                   help="Top-K by discriminator confidence to keep.")
    p.add_argument("--hand-to-eef-offset", type=float, default=0.1034,
                   help="Forward shift applied to GraspGen's panda_hand-frame "
                        "pose origins (matches sample_graspgen_grasps).")
    p.add_argument("--approach-len", type=float, default=0.04,
                   help="Length (m) of the approach-axis line drawn per "
                        "grasp. 0.04 ≈ Franka's default fingertip depth.")
    p.add_argument("--image-size", type=int, default=900)
    p.add_argument("--graspgen-host", type=str, default=None)
    p.add_argument("--graspgen-port", type=int, default=None)
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
    return dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[],
        objects=[],
        task=dict(type="DummyTask"),
    )


from sentinel.rl.grasps._viz_helpers import render_grasp_views as _render_grasps


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
    import trimesh
    from omnigibson.objects import DatasetObject

    from sentinel.rl.grasps.mesh import mesh_from_og_object
    from sentinel.rl.grasps.graspgen_sampler import _get_client

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG ...", flush=True)
    env = og.Environment(configs=_build_env_config())
    env.reset()

    host = args.graspgen_host or os.environ.get("GRASPGEN_HOST", "localhost")
    port = args.graspgen_port or int(os.environ.get("GRASPGEN_PORT", "5556"))
    client = _get_client(host, port)
    print(f"[{time.strftime('%H:%M:%S')}] GraspGen client ready: "
          f"{client.server_metadata}", flush=True)

    rng = np.random.default_rng(0)

    for cat, mdl in targets:
        stem = f"{cat}_{mdl}"
        print(f"\n[{time.strftime('%H:%M:%S')}] {cat}/{mdl}", flush=True)
        name = f"viz_{cat}_{mdl}"
        try:
            obj = DatasetObject(name=name, category=cat, model=mdl)
            env.scene.add_object(obj)
            obj.set_position_orientation(
                position=th.tensor([0.0, 0.0, 1.0], dtype=th.float32),
                orientation=th.tensor([0.0, 0.0, 0.0, 1.0],
                                      dtype=th.float32),
            )
            for _ in range(2):
                og.sim.step()
        except Exception as e:  # noqa: BLE001
            print(f"  spawn failed: {e}", flush=True)
            continue

        try:
            mesh = mesh_from_og_object(obj, use_visual=True)
            print(f"  mesh: V={len(mesh.vertices)} F={len(mesh.faces)}",
                  flush=True)

            pts, _ = trimesh.sample.sample_surface(mesh, args.n_points)
            pts = np.asarray(pts, dtype=np.float32)

            grasps, scores = client.infer(
                pts, num_grasps=args.num_grasps, topk_num_grasps=args.topk)
            if len(grasps) == 0:
                print("  GraspGen returned 0 grasps", flush=True)
                continue

            grasps = grasps.copy()
            if args.hand_to_eef_offset != 0.0:
                approach = grasps[:, :3, 2]
                grasps[:, :3, 3] += approach * args.hand_to_eef_offset

            order = np.argsort(-scores)
            grasps = grasps[order]
            scores = scores[order]

            eef_pos = grasps[:, :3, 3]
            approach = grasps[:, :3, 2]

            print(f"  GraspGen: {len(grasps)} grasps, "
                  f"score range={scores.min():.3f}-{scores.max():.3f}",
                  flush=True)

            paths = _render_grasps(
                pts, eef_pos, approach, scores, out_dir, stem,
                args.image_size, args.approach_len)
            for p in paths:
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
