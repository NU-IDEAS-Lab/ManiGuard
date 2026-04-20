#!/usr/bin/env python3
"""Smoke-test: reset-from-grasp dataset integration in PickAndLiftTask.

Boots an empty OG Scene + FrankaMounted + breakfast_table + one soda_cup,
installs PickAndLiftTask with ``grasp_dataset_path`` pointing at a
previously-collected ``grasps_soda_cup_fsfsas.pt``, then calls ``env.reset()``
N times. For each reset we record:
  - whether the IK-based grasp reset succeeded (gripper at saved pose)
  - eef-to-target distance (should be < ~5 cm if reset worked)
  - gripper finger-joint positions (should match saved gripper_qpos)

Usage:
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.grasps.smoke_test_reset \\
            --category soda_cup --model fsfsas --num-resets 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="soda_cup")
    p.add_argument("--model", default="fsfsas")
    p.add_argument("--grasp-dataset-dir", type=Path,
                   default=Path("outputs/grasp_datasets/batch"))
    p.add_argument("--num-resets", type=int, default=10)
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_reset_smoke"))
    return p.parse_args()




def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target_name = f"target_{args.category}_{args.model}"

    grasp_path = args.grasp_dataset_dir / f"grasps_{args.category}_{args.model}.pt"
    if not grasp_path.exists():
        raise SystemExit(f"Missing grasp dataset: {grasp_path}\n"
                         f"Run `python -m sentinel.rl.grasps.collect_batch` first.")

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og
    import torch as th
    from omnigibson.object_states import Touching
    # Side-effect: registers PickAndLiftTask with OG's task registry.
    import sentinel.rl.tasks  # noqa: F401
    from sentinel.rl.envs.grasp_reset_scene import build_config

    cfg = build_config(
        target_name=target_name,
        category=args.category,
        model=args.model,
        grasp_dataset_path=grasp_path,
        obs_modalities=[],
        visualize_goal=False,
        flatten_obs=False,
        flatten_action=False,
        max_steps=500,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Booting OG...", flush=True)
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    arm = robot.default_arm

    target_obj = env.scene.object_registry("name", target_name)
    if target_obj is None:
        raise RuntimeError(f"Target {target_name!r} not in scene.")

    # Load saved grasps for ground-truth comparison.
    data = th.load(str(grasp_path), map_location="cpu")
    rel_positions = data["rel_position"].float()
    rel_orientations = data["rel_orientation_xyzw"].float()
    saved_gripper_qpos = data["gripper_qpos"].float()
    num_grasps = len(rel_positions)
    print(f"  {num_grasps} saved grasps in {grasp_path.name}", flush=True)

    # N calls to env.reset() and record diagnostics.
    results = []
    for i in range(args.num_resets):
        env.reset()
        eef_pos, eef_quat = robot.eef_links[arm].get_position_orientation()
        eef_pos = eef_pos.detach().cpu().numpy()
        tgt_pos, tgt_quat = target_obj.get_position_orientation()
        tgt_pos = tgt_pos.detach().cpu().numpy()
        gripper_idx = robot.gripper_control_idx[arm]
        gripper_q = robot.get_joint_positions()[gripper_idx].detach().cpu().numpy()

        eef_to_tgt = float(np.linalg.norm(eef_pos - tgt_pos))
        q_diff = (saved_gripper_qpos.numpy() - gripper_q).reshape(num_grasps, -1)
        min_q_err = float(np.min(np.linalg.norm(q_diff, axis=1)))

        # Is the gripper ACTUALLY touching the target post-reset? This is the
        # definitive "did the reset configure a real grasp?" check.
        touching = bool(robot.states[Touching].get_value(target_obj))

        res = {
            "reset_idx": i,
            "eef_world_pos": eef_pos.tolist(),
            "target_world_pos": tgt_pos.tolist(),
            "eef_to_target_dist_m": eef_to_tgt,
            "gripper_qpos": gripper_q.tolist(),
            "min_gripper_qpos_error_vs_saved": min_q_err,
            "touching_target": touching,
        }
        results.append(res)
        print(
            f"  reset {i + 1:2d}/{args.num_resets}: "
            f"eef→tgt={eef_to_tgt * 100:.2f} cm, "
            f"gripper_qpos_err={min_q_err:.4f} m, "
            f"touching={touching}",
            flush=True,
        )

    # Summary
    dists = np.asarray([r["eef_to_target_dist_m"] for r in results])
    q_errs = np.asarray([r["min_gripper_qpos_error_vs_saved"] for r in results])
    touching_count = sum(1 for r in results if r["touching_target"])
    summary = {
        "num_resets": args.num_resets,
        "num_saved_grasps": num_grasps,
        "eef_to_target_mean_m": float(dists.mean()),
        "eef_to_target_min_m": float(dists.min()),
        "eef_to_target_max_m": float(dists.max()),
        "eef_to_target_n_above_10cm": int(np.sum(dists > 0.10)),
        "gripper_qpos_err_mean": float(q_errs.mean()),
        "gripper_qpos_err_max": float(q_errs.max()),
        "touching_after_reset_count": touching_count,
        "touching_after_reset_rate": touching_count / max(1, args.num_resets),
    }
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)

    with open(out / f"{args.category}_{args.model}_summary.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nWrote {out / f'{args.category}_{args.model}_summary.json'}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
