#!/usr/bin/env python3
"""Collect a physically-validated grasp dataset for a target object.

Pipeline:
  1. Load the benchmark scene via ``sentinel.rl.config.build_config``.
  2. Extract the target's visual mesh (object-local frame).
  3. Run antipodal grasp sampling on the mesh.
  4. For each candidate, IK + teleport arm + close gripper + OmniReset-style
     shake test; keep only grasps that hold physically (no sticky).
  5. Save as ``grasps_<target>.pt`` alongside the mesh PLY.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python scripts/collect_grasps.py \\
            --scene-dir datasets/safety-benchmark/clutter_goblet_00 \\
            --output-dir outputs/grasp_datasets/clutter_goblet_00 \\
            --num-target-grasps 100 --max-attempts 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Collect validated grasps for a benchmark scene.")
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_datasets"))
    p.add_argument("--num-target-grasps", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--num-surface-samples", type=int, default=64)
    p.add_argument("--num-orientations", type=int, default=16)
    p.add_argument("--num-standoff-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-collision-mesh", action="store_true",
                   help="Sample on the object's collision mesh (coarser) instead of visual.")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # OG boot
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og

    from sentinel.rl.config import build_config
    from sentinel.rl.resets.grasp_collector import (
        GraspCollectorConfig, collect_valid_grasps, save_grasp_dataset,
    )
    from sentinel.rl.resets.grasp_sampler import AntipodalConfig, sample_antipodal_grasps
    from sentinel.rl.resets.mesh_loader import franka_gripper_params, mesh_from_og_object

    print(f"[{time.strftime('%H:%M:%S')}] Building config from {args.scene_dir}")
    cfg = build_config(args.scene_dir)
    target_name = cfg["task"]["obj_name"]
    print(f"  target: {target_name}")

    print(f"[{time.strftime('%H:%M:%S')}] Starting OmniGibson (~30s)...")
    env = og.Environment(configs=cfg)
    env.reset()

    target = env.scene.object_registry("name", target_name)
    if target is None:
        raise RuntimeError(f"Target {target_name!r} not found in scene.")

    # Confirm we're NOT using sticky — physical grasp is what gives a real validity test.
    robot = env.robots[0]
    print(f"  robot: {robot.name}  grasping_mode={robot.grasping_mode}")
    if robot.grasping_mode == "sticky":
        print("  ! WARNING: grasping_mode=sticky would accept trivial contacts. "
              "Expected 'physical' for a meaningful physics validation.")

    print(f"[{time.strftime('%H:%M:%S')}] Extracting {'collision' if args.use_collision_mesh else 'visual'} mesh...")
    mesh = mesh_from_og_object(target, use_visual=not args.use_collision_mesh)
    print(f"  vertices={len(mesh.vertices)} faces={len(mesh.faces)} extents={mesh.extents.tolist()}")

    print(f"[{time.strftime('%H:%M:%S')}] Sampling antipodal candidates...")
    gp = franka_gripper_params()
    acfg = AntipodalConfig(
        num_surface_samples=args.num_surface_samples,
        num_orientations=args.num_orientations,
        num_standoff_samples=args.num_standoff_samples,
        top_bias=False,
        **gp,
    )
    candidates = sample_antipodal_grasps(mesh, acfg, rng=np.random.default_rng(args.seed))
    print(f"  {len(candidates)} candidates (cap {acfg.num_surface_samples * acfg.num_orientations * acfg.num_standoff_samples})")

    # Shuffle so we don't keep trying similar surface points / standoffs in order.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(candidates))
    candidates = candidates[perm]

    print(f"[{time.strftime('%H:%M:%S')}] Validating via physical grasp + shake test...")
    ccfg = GraspCollectorConfig(
        num_target_grasps=args.num_target_grasps,
        max_attempts=args.max_attempts,
    )
    t0 = time.time()
    valid = collect_valid_grasps(env, target, candidates, cfg=ccfg, rng=rng, verbose=True)
    dt = time.time() - t0
    print(f"  validated {len(valid)} grasps in {dt:.1f}s "
          f"(pass rate {len(valid) / max(1, min(ccfg.max_attempts or len(candidates), len(candidates))):.1%})")

    if not valid:
        print("  ! no valid grasps — try more candidates or relax thresholds.")
        os._exit(1)

    ply_path = out / f"{target_name}.ply"
    mesh.export(str(ply_path))
    pt_path = out / f"grasps_{target_name}.pt"
    save_grasp_dataset(valid, pt_path, target_name=target_name)
    print(f"  wrote {ply_path}")
    print(f"  wrote {pt_path}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
