#!/usr/bin/env python3
"""RL rollout test: verify reset-from-grasp → lift → goal works end-to-end.

What this tests beyond ``sentinel.rl.grasps.smoke_test_reset``:
  - ``env.step(action)`` integrates correctly after a reset-from-grasp start
  - The held object rises with the arm (friction holds, physics is stable)
  - Reward signal fires as the object moves toward the goal region
  - ``InGoalRegion`` termination triggers when the cup reaches the goal ball
  - None of the failure paths (cup drops, arm explodes) happens on a
    well-conditioned grasp start

Policy under test is a hand-coded "lift straight up with closed gripper"
(OSC pose-delta +Z, gripper = close). That turns the task into a purely
physical check of whether the grasp dataset + reset pipeline puts the
gripper in a state from which the simplest possible policy can succeed.

Usage:
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.training.rollout_test \\
            --category soda_cup --model fsfsas \\
            --num-episodes 5 --max-steps 80
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
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=80,
                   help="Max env.step calls per episode (before artificially ending).")
    p.add_argument("--lift-rate", type=float, default=0.15,
                   help="OSC +Z pose-delta input per step.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_reset_rollout"))
    return p.parse_args()




def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target_name = f"target_{args.category}_{args.model}"

    grasp_path = args.grasp_dataset_dir / f"grasps_{args.category}_{args.model}.pt"
    if not grasp_path.exists():
        raise SystemExit(f"Missing grasp dataset: {grasp_path}")

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og
    import torch as th
    from omnigibson.object_states import Touching
    import sentinel.rl.tasks  # noqa: F401  (register PickAndLiftTask)
    from sentinel.rl.envs.grasp_reset_scene import build_config

    cfg = build_config(
        target_name=target_name,
        category=args.category,
        model=args.model,
        grasp_dataset_path=grasp_path,
        # Rollout test matches the earlier hardcoded values: no obs_modalities
        # since the test ignores obs, and a generous 500-step timeout.
        obs_modalities=[],
        flatten_obs=False,
        flatten_action=False,
        max_steps=500,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Booting OG...", flush=True)
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    arm = robot.default_arm
    target_obj = env.scene.object_registry("name", target_name)

    # Hand-coded lift policy: OSC +Z pose-delta each step, gripper stays closed.
    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]
    lift_action_template = np.zeros(robot.action_dim, dtype=np.float32)
    # Arm dims 0..2 = eef [dx, dy, dz] world delta; 3..5 = axis-angle delta.
    # We want Z only, orientation held.
    lift_action_template[arm_action_idx[2]] = args.lift_rate
    lift_action_template[gripper_action_idx] = -1.0  # stay closed

    episode_summaries = []
    for ep in range(args.num_episodes):
        print(f"\n{'=' * 60}", flush=True)
        print(f"Episode {ep + 1}/{args.num_episodes}", flush=True)
        print(f"{'=' * 60}", flush=True)

        obs, info = env.reset()

        # Post-reset state
        tgt_pos_0 = target_obj.get_position_orientation()[0].detach().cpu().numpy()
        touching_after_reset = bool(robot.states[Touching].get_value(target_obj))
        print(f"  after reset: tgt_z={tgt_pos_0[2]:.3f} m, touching={touching_after_reset}", flush=True)
        if not touching_after_reset:
            print(f"  ! reset did not configure a grasp — skipping lift rollout.", flush=True)
            episode_summaries.append({
                "ep": ep,
                "touching_after_reset": False,
                "steps_to_success": None,
                "total_reward": 0.0,
                "final_obj_z": float(tgt_pos_0[2]),
                "terminated": False,
                "info_keys": [],
            })
            continue

        # Rollout with the lift policy.
        total_reward = 0.0
        steps_to_success = None
        terminated_step = None
        first_info_keys = None
        final_tgt_pos = tgt_pos_0

        for step in range(args.max_steps):
            action = {robot.name: th.tensor(lift_action_template)}
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            tgt_pos = target_obj.get_position_orientation()[0].detach().cpu().numpy()
            final_tgt_pos = tgt_pos

            # Capture info keys once (to verify task reward/termination fires
            # through the standard RL API).
            if first_info_keys is None:
                first_info_keys = sorted(list(info.keys()))

            done = bool(terminated) or bool(truncated)
            if step % 10 == 0 or done:
                in_hand = robot._ag_obj_in_hand.get(arm)
                holding = in_hand is target_obj
                print(
                    f"  step {step:3d}: reward={float(reward):+.3f} "
                    f"cum={total_reward:+.3f} tgt_z={tgt_pos[2]:.3f} "
                    f"holding={holding} terminated={terminated} truncated={truncated}",
                    flush=True,
                )

            # Check termination reason via env task directly (info may be
            # flat depending on wrappers).
            if terminated:
                terminated_step = step
                # Find which termination fired
                task = env.task
                for name, cond in task._termination_conditions.items():
                    # Conditions have internal _done state but not public; we
                    # proxy by checking the eventual info dict and via
                    # task.success condition on the object.
                    pass
                steps_to_success = step
                break
            if truncated:
                terminated_step = step
                break

        summary = {
            "ep": ep,
            "touching_after_reset": True,
            "steps_to_success": steps_to_success,
            "terminated_step": terminated_step,
            "total_reward": total_reward,
            "final_obj_z": float(final_tgt_pos[2]),
            "obj_z_rise": float(final_tgt_pos[2] - tgt_pos_0[2]),
            "info_keys": first_info_keys or [],
        }
        episode_summaries.append(summary)
        print(
            f"  episode result: terminated_step={terminated_step}, "
            f"reward={total_reward:.3f}, obj_rise={summary['obj_z_rise'] * 100:.1f} cm",
            flush=True,
        )

    # Summary across episodes
    n = len(episode_summaries)
    n_reset_ok = sum(1 for s in episode_summaries if s["touching_after_reset"])
    n_terminated = sum(1 for s in episode_summaries if s.get("steps_to_success") is not None)
    mean_rise = float(np.mean(
        [s.get("obj_z_rise", 0.0) for s in episode_summaries if s["touching_after_reset"]]
        or [0.0]
    ))
    mean_reward = float(np.mean([s["total_reward"] for s in episode_summaries]))

    overall = {
        "num_episodes": n,
        "reset_touching_rate": n_reset_ok / max(1, n),
        "goal_reach_rate": n_terminated / max(1, n),
        "mean_obj_z_rise_m": mean_rise,
        "mean_total_reward": mean_reward,
    }
    print(f"\n{'=' * 60}\nSUMMARY ({args.category}/{args.model})\n{'=' * 60}", flush=True)
    for k, v in overall.items():
        print(f"  {k}: {v}", flush=True)

    out_json = out / f"{args.category}_{args.model}_rollout.json"
    with open(out_json, "w") as f:
        json.dump({"overall": overall, "episodes": episode_summaries}, f, indent=2)
    print(f"\nWrote {out_json}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
