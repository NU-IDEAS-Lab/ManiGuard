#!/usr/bin/env python3
"""Collect Franka grasps for a target object in the actual benchmark scene.

Mirrors ``maniguard.rl.grasps.render_grasps`` but loads the trained scene
(robot + desk + distractors + target at their training poses) instead of
the empty floating-object setup. The resulting ``arm_joint_pos`` values
are valid for the training scene's robot+object geometry, so cached-mode
``GraspDatasetResetter`` actually places the gripper on the target.

Usage:
    PYTHONPATH=/data/Projects/ManiGuard \\
    OMNI_KIT_ACCEPT_EULA=yes \\
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        /home/simonzhan/anaconda3/envs/behavior/bin/python \\
            tools/collect_grasps_in_scene.py \\
            --scene-file <path>/scene_ep1.joint.json \\
            --diagnostics-file <path>/diagnostics.jsonl \\
            --target-name cocktail_glass_178 \\
            --category cocktail_glass --model xevdnl \\
            --output-dir outputs/grasp_datasets/task0000_inscene \\
            --num-target-grasps 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Apply maniguard patches (longfinger asset bundle, AG state aliases, etc.)
import maniguard  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-file", type=Path, required=True)
    p.add_argument("--diagnostics-file", type=Path, required=True)
    p.add_argument("--target-name", type=str, required=True)
    p.add_argument("--category", type=str, required=True,
                   help="Used to name the output .pt: grasps_<cat>_<model>.pt")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-target-grasps", type=int, default=200)
    p.add_argument("--graspgen-num-grasps", type=int, default=1000)
    p.add_argument("--graspgen-topk", type=int, default=800)
    p.add_argument("--max-candidates", type=int, default=800)
    p.add_argument("--per-object-timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-reach", type=float, default=0.95)
    p.add_argument("--graspgen-host", type=str, default=None)
    p.add_argument("--graspgen-port", type=int, default=None)
    p.add_argument("--settle-open-steps", type=int, default=8)
    p.add_argument("--close-steps", type=int, default=20)
    p.add_argument("--gravity-hold-steps", type=int, default=30)
    p.add_argument("--max-obj-to-eef-after-hold", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read room + scene_model from diagnostics for partial load.
    with args.diagnostics_file.open() as f:
        diag = json.loads(f.readline())
    room = (diag.get("support_selection") or {}).get("room_instance")
    scene_model = diag.get("scene_model")

    # OG macros (must precede `import omnigibson`).
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    # Same AG-window patch as the original render_grasps.py — fires AG
    # after ~2 stable contact steps instead of waiting for the default
    # 333 ms commit window.
    from omnigibson.robots.manipulation_robot import m as _ag_macros
    _ag_macros.GRASP_WINDOW = 1.0 / 300.0
    _ag_macros.RELEASE_WINDOW = 1.0 / 300.0

    import omnigibson as og
    import torch as th

    # Build env: actual scene + InteractiveTraversableScene with room filter.
    scene_cfg = {
        "type": "InteractiveTraversableScene",
        "scene_model": scene_model,
        "scene_file": str(args.scene_file),
    }
    if room:
        scene_cfg["load_room_instances"] = [room]

    cfg = dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=scene_cfg,
        robots=[],   # robot is in the scene_file
        objects=[],  # target is in the scene_file
        task=dict(type="DummyTask"),
    )

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG: scene={scene_model} "
          f"room={room} ...", flush=True)
    env = og.Environment(configs=cfg)
    env.reset()

    # Locate target + robot.
    target_obj = env.scene.object_registry("name", args.target_name)
    if target_obj is None:
        names = [o.name for o in env.scene.objects]
        raise SystemExit(
            f"Target {args.target_name!r} not in scene. Have: {names}"
        )
    if not env.robots:
        raise SystemExit("No robot found in scene")
    robot = env.robots[0]

    bundle = getattr(robot, "_franka_panda_asset_bundle", "<unset>")
    print(f"  target {target_obj.name} @ "
          f"{target_obj.get_position_orientation()[0].cpu().numpy()}", flush=True)
    print(f"  robot {robot.name} @ "
          f"{robot.get_position_orientation()[0].cpu().numpy()}", flush=True)
    print(f"  asset bundle: {bundle}", flush=True)

    # Snapshot the target's scene-init pose (what the resetter will restore to).
    init_pos, init_quat = target_obj.get_position_orientation()
    init_pos = init_pos.detach().clone()
    init_quat = init_quat.detach().clone()

    # Float the target so the gripper can approach without PhysX kicking it
    # around. Hard-pinning is handled by ``_phase1_step`` inside the collector.
    target_obj.root_link.set_linear_velocity(th.zeros(3))
    target_obj.root_link.set_angular_velocity(th.zeros(3))
    target_obj.root_link.disable_gravity()

    from maniguard.rl.grasps.collector import (
        GraspCollectorConfig,
        _phase1_step,
        collect_valid_grasps,
        save_grasp_dataset,
    )
    from maniguard.rl.grasps.graspgen_sampler import sample_graspgen_grasps
    from maniguard.rl.grasps.mesh import mesh_from_og_object
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )

    print(f"[{time.strftime('%H:%M:%S')}] initialising cuRobo ...", flush=True)
    primitives = StarterSemanticActionPrimitives(
        env, robot, enable_head_tracking=False
    )
    print(f"[{time.strftime('%H:%M:%S')}] cuRobo ready.", flush=True)

    # Brief pinned settle so PhysX/USD initialise the target prim cleanly.
    for _ in range(8):
        _phase1_step(env, robot, target_obj, init_pos, init_quat, hard_pin=True)

    # Extract mesh for GraspGen.
    print("Extracting target mesh ...", flush=True)
    mesh = mesh_from_og_object(target_obj, use_visual=True)

    # Query GraspGen for candidates.
    print("Querying GraspGen ...", flush=True)
    rng = np.random.default_rng(args.seed)
    candidates, scores = sample_graspgen_grasps(
        mesh,
        num_grasps=args.graspgen_num_grasps,
        topk_num_grasps=args.graspgen_topk,
        confidence_threshold=0.0,
        host=args.graspgen_host,
        port=args.graspgen_port,
        rng=rng,
    )
    if len(candidates) == 0:
        raise SystemExit("GraspGen returned 0 candidates")
    if len(candidates) > args.max_candidates:
        candidates = candidates[: args.max_candidates]
        scores = scores[: args.max_candidates]
    print(f"  {len(candidates)} candidates (score range "
          f"{scores[-1]:.3f}-{scores[0]:.3f})", flush=True)

    # Tell cuRobo to ignore the target when planning approaches.
    primitives._motion_generator.update_obstacles(ignore_objects=[target_obj])

    # Phase A: validate.
    collector_cfg = GraspCollectorConfig(
        num_target_grasps=args.num_target_grasps,
        settle_open_steps=args.settle_open_steps,
        close_steps=args.close_steps,
        gravity_hold_steps=args.gravity_hold_steps,
        max_reach=args.max_reach,
        max_obj_to_eef_after_hold=args.max_obj_to_eef_after_hold,
    )

    n_held = [0]

    def _on_progress(ci, result):
        if result is not None:
            n_held[0] += 1
            print(f"  cand {ci}: HELD "
                  f"(obj_to_eef={result['obj_to_eef']:.3f}m, "
                  f"traj_len={len(result['approach_traj'])}) "
                  f"[{n_held[0]}/{collector_cfg.num_target_grasps}]",
                  flush=True)

    deadline = time.time() + args.per_object_timeout
    held = collect_valid_grasps(
        env, robot, primitives, target_obj,
        init_pos, init_quat,
        candidates_local=candidates,
        cfg=collector_cfg,
        deadline=deadline,
        on_progress=_on_progress,
        verbose=True,
    )

    if not held:
        print("\nFAILED: 0 valid grasps after full sweep.", flush=True)
        sys.exit(1)

    # Save dataset (same format as render_grasps.py output).
    pt_path = args.output_dir / f"grasps_{args.category}_{args.model}.pt"
    save_grasp_dataset(held, pt_path,
                       target_name=f"{args.category}_{args.model}")
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. {len(held)} grasps saved to "
          f"{pt_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
