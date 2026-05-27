#!/usr/bin/env python3
"""Open-loop validation of the joint-action route.

Replays the absolute joint-position actions recovered from an SFT rollout
(``maniguard.data.lerobot.joint_actions.extract_joint_trajectory``) straight through a
JointController in the demo's own scene -- NO eef->joint IK shim. Loads the
exact recorded start state (``states[0]``) so the replay is comparable to the
demo. Validates:

  * eef tracking: achieved relative-eef vs the recorded eef trajectory
  * grasp: target object lift (z rise) under assisted grasping

If the demo replays faithfully (eef tracks, object lifts), the joint-action
eval path is sound and we can re-export + train on joint actions.

Usage:
  conda run -n behavior python -m tools.validate_joint_replay \
    --hdf5 outputs/variants_n10x2_first25/task_0000/seed_00/variant_01/rollout.hdf5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from maniguard.data.lerobot.joint_actions import extract_joint_trajectory
from maniguard.eval import benchmark as B
from maniguard.eval.eval_config import load_eval_config
from maniguard.eval.scene_discovery import discover_scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--config", default="configs/eval/pnp_clutter_3cam.yaml")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full episode")
    args = ap.parse_args()

    import h5py
    with h5py.File(args.hdf5, "r") as f:
        task_dir = f.attrs["task_dir"]
        target_name = str(f.attrs["target_name"])
        states0 = np.asarray(f["data/demo_0"]["states"][0], dtype=np.float32)
    task_dir = task_dir.decode() if isinstance(task_dir, bytes) else str(task_dir)
    # scene name relative to the clutter_pickup family root: e.g. "task_0000/base"
    family_root = str(Path(task_dir).parents[1])
    scene_name = str(Path(task_dir).relative_to(family_root))
    print(f"[replay] task_dir={task_dir}\n[replay] target={target_name} scene={scene_name}", flush=True)

    traj = extract_joint_trajectory(args.hdf5)
    jaction, eef_rec = traj["joint_action"], traj["eef"]
    N = len(jaction) if args.max_steps <= 0 else min(args.max_steps, len(jaction))

    cfg = load_eval_config(args.config)
    cfg.headless = True  # metrics only
    B._init_omnigibson(cfg)
    import omnigibson as og

    scenes = discover_scenes(family_root, scene_names=[scene_name])
    if not scenes:
        raise SystemExit(f"scene {scene_name} not found under {family_root}")
    scene_info = scenes[0]
    env = og.Environment(configs=B.build_og_config(scene_info, cfg))
    robot = env.robots[0]
    env.reset()

    # JointController (position drive) -- same override the benchmark builds.
    from maniguard.envs.frozen_task_runtime import CONTROLLER_PRESETS
    cc = json.loads(json.dumps(CONTROLLER_PRESETS[cfg.controller_preset]))
    a0 = cc["arm_0"]
    if cfg.joint_pos_kp is not None and a0.get("name") == "JointController":
        kp = float(cfg.joint_pos_kp)
        a0["use_impedances"] = False
        a0["isaac_kp"] = kp
        a0["isaac_kd"] = 2.0 * (kp ** 0.5)
    robot.reload_controllers(cc)

    # NOTE: og.sim.load_state(states0) fails across sessions (object UUIDs
    # differ — the known cross-session UUID issue). The eef-tracking check is
    # object-independent (depends only on robot joints), so instead set the
    # robot to the recorded start joint config and settle. (Object placement
    # then comes from the base scene, so the grasp/lift number is only
    # informational — it won't match the perturbed demo placement.)
    arm = robot.default_arm
    asp = robot.action_space
    arm_idx = robot.arm_control_idx[arm]
    g_idx = robot.gripper_control_idx[arm]
    js0 = traj["joint_state"][0]
    dev = robot.get_joint_positions().device
    robot.set_joint_positions(torch.tensor(js0[:7], dtype=torch.float32, device=dev),
                              indices=arm_idx, drive=False)
    robot.set_joint_positions(torch.full((len(g_idx),), float(js0[7]), dtype=torch.float32, device=dev),
                              indices=g_idx, drive=False)
    # settle the controller goal by holding the start joint target a few steps
    hold = np.clip(np.asarray(jaction[0], dtype=np.float32), asp.low, asp.high)
    hold[:7] = js0[:7]
    for _ in range(5):
        env.step(torch.from_numpy(hold).unsqueeze(0))

    def target_obj_z():
        obj = env.scene.object_registry("name", target_name)
        return float(obj.get_position_orientation()[0][2]) if obj is not None else float("nan")

    z0 = target_obj_z()
    eef_errs, zs = [], []
    for t in range(N):
        act = np.clip(np.asarray(jaction[t], dtype=np.float32), asp.low, asp.high)
        env.step(torch.from_numpy(act).unsqueeze(0))
        eef = robot.get_relative_eef_position().cpu().numpy()
        # action t targets joints[t+1] -> compare to recorded eef[t+1]
        ref = eef_rec[min(t + 1, len(eef_rec) - 1), :3]
        eef_errs.append(float(np.linalg.norm(eef - ref)))
        zs.append(target_obj_z())
        if t % 100 == 0:
            print(f"  step {t}/{N}  eef_err={eef_errs[-1]:.4f}  obj_z={zs[-1]:.3f}", flush=True)

    eef_errs = np.array(eef_errs)
    zs = np.array(zs)
    print("\n==== JOINT-REPLAY VALIDATION ====", flush=True)
    print(f"steps replayed: {N}")
    print(f"eef tracking err (m): mean {eef_errs.mean():.4f}  median {np.median(eef_errs):.4f}  max {eef_errs.max():.4f}")
    print(f"target obj z: start {z0:.3f}  max {zs.max():.3f}  end {zs[-1]:.3f}  lift {zs.max()-z0:+.3f}")
    print(f"GRASP/LIFT (>3cm): {'YES' if (zs.max()-z0) > 0.03 else 'no'}")
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
