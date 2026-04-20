#!/usr/bin/env python3
"""Measure and print Franka / FrankaMounted gripper geometry.

For each robot we record:
  - gripper joint name(s), lower/upper limits
  - derived max aperture   = sum(joint_upper - joint_lower) across finger joints
  - actual fingertip-to-fingertip world distance at closed / open configurations
  - eef_link -> fingertip length (robot.eef_to_fingertip_lengths)

These are the numbers that should seed the grasp sampler's hardcoded
gripper_max_aperture + finger_offset instead of the current magic values.

Usage:
    conda activate behavior
    OMNI_KIT_ACCEPT_EULA=yes VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.grasps.measure_gripper
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _robot_cfg(robot_type: str, name: str, x: float) -> dict:
    return dict(
        type=robot_type,
        name=name,
        obs_modalities=[],
        grasping_mode="physical",
        self_collisions=False,
        position=[x, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0, 1.0],
        controller_config={
            "arm_0": {"name": "JointController"},
            "gripper_0": {"name": "MultiFingerGripperController"},
        },
    )


def _measure(robot) -> dict:
    """Probe gripper joint limits, aperture, and fingertip geometry."""
    import torch as th

    arm = robot.default_arm
    finger_names = list(robot.finger_link_names[arm])
    eef_link_name = robot.eef_link_names[arm]
    gripper_idx = robot.gripper_control_idx[arm]

    gripper_joint_names = list(robot.finger_joint_names[arm])
    lower = robot.joint_lower_limits[gripper_idx].detach().cpu().numpy().tolist()
    upper = robot.joint_upper_limits[gripper_idx].detach().cpu().numpy().tolist()
    derived_aperture = float(sum(u - l for u, l in zip(upper, lower)))

    # Measure actual world fingertip distance at closed and open joint targets.
    import omnigibson as og

    def _tip_positions():
        return {
            fn: robot.links[fn].get_position_orientation()[0].detach().cpu().numpy().tolist()
            for fn in finger_names
        }

    def _distance(positions):
        import numpy as np
        if len(positions) != 2:
            return None
        p = list(positions.values())
        return float(np.linalg.norm(np.asarray(p[0]) - np.asarray(p[1])))

    def _finger_aabbs():
        """Return (aabb_low, aabb_high) per finger link in world frame."""
        import numpy as np
        aabbs = {}
        for fn in finger_names:
            low, high = robot.links[fn].aabb
            aabbs[fn] = (
                np.asarray(low.detach().cpu() if hasattr(low, "detach") else low).tolist(),
                np.asarray(high.detach().cpu() if hasattr(high, "detach") else high).tolist(),
            )
        return aabbs

    def _inner_surface_gap(aabbs):
        """Compute y-axis gap between the two blades' inner faces (blade closing axis).

        Franka closes along world-Y when spawned with identity orientation. The
        two finger AABBs are symmetric around y=0; the inner surfaces are the
        faces facing each other. Gap = min(|aabb_high_y|, |aabb_low_y|) for each
        blade, then sum.
        """
        if len(aabbs) != 2:
            return None
        vals = list(aabbs.values())
        # finger 1's inner face is the face closer to y=0 (i.e., the max of low_y or the min of high_y, whichever is closer to zero)
        low0, high0 = vals[0]
        low1, high1 = vals[1]
        # Each blade occupies a Y range. Distance between the two nearest y-edges:
        inner_edge_0 = high0[1] if abs(high0[1]) < abs(low0[1]) else low0[1]
        inner_edge_1 = high1[1] if abs(high1[1]) < abs(low1[1]) else low1[1]
        return float(abs(inner_edge_0 - inner_edge_1))

    # Measure via CONTROLLER action (gripper_cmd = -1 close, +1 open) and let
    # physics settle. This is what the RL / collector pipeline actually sees:
    # joints converge to joint limits via the PD controller, subject to any
    # blade-blade contact or friction that may stop them short of the limit.
    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]

    def _apply_gripper(cmd_value: float, n_steps: int = 60):
        action = th.zeros(robot.action_dim, dtype=th.float32)
        # Hold arm at current joint positions (delta 0 for position-mode arm controller).
        # Franka arm controller is JointController position-mode; zero arm cmd works.
        action[arm_action_idx] = 0.0
        action[gripper_action_idx] = cmd_value
        for _ in range(n_steps):
            robot.apply_action(action)
            og.sim.step()

    # Close via controller cmd = -1
    _apply_gripper(-1.0, n_steps=60)
    closed_joint_positions = robot.get_joint_positions()[gripper_idx].detach().cpu().numpy().tolist()
    closed_positions = _tip_positions()
    closed_distance = _distance(closed_positions)
    closed_aabbs = _finger_aabbs()
    closed_inner_gap = _inner_surface_gap(closed_aabbs)

    # Open via controller cmd = +1
    _apply_gripper(1.0, n_steps=60)
    open_joint_positions = robot.get_joint_positions()[gripper_idx].detach().cpu().numpy().tolist()
    open_positions = _tip_positions()
    open_distance = _distance(open_positions)
    open_aabbs = _finger_aabbs()
    open_inner_gap = _inner_surface_gap(open_aabbs)

    eef_to_tip = {
        fn: float(v) for fn, v in robot.eef_to_fingertip_lengths[arm].items()
    }

    return {
        "robot_type": type(robot).__name__,
        "name": robot.name,
        "eef_link": eef_link_name,
        "finger_links": finger_names,
        "gripper_joint_names": gripper_joint_names,
        "joint_lower_limits": lower,
        "joint_upper_limits": upper,
        "derived_max_aperture_m": derived_aperture,
        "closed_joint_positions": closed_joint_positions,
        "open_joint_positions": open_joint_positions,
        "fingertip_distance_closed_m": closed_distance,
        "fingertip_distance_open_m": open_distance,
        "inner_surface_gap_closed_m": closed_inner_gap,
        "inner_surface_gap_open_m": open_inner_gap,
        "eef_to_fingertip_lengths_m": eef_to_tip,
    }


def main():
    out_dir = Path.cwd() / "outputs" / "gripper_measurements"
    out_dir.mkdir(parents=True, exist_ok=True)

    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = False
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og

    cfg = dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[
            _robot_cfg("FrankaPanda", "franka_0", x=0.0),
            _robot_cfg("FrankaMounted", "franka_mounted_0", x=1.5),
        ],
        objects=[],
        task=dict(type="DummyTask"),
    )

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG with FrankaPanda + FrankaMounted...", flush=True)
    env = og.Environment(configs=cfg)
    env.reset()

    results = {}
    for robot in env.robots:
        print(f"\n{'=' * 60}\nMeasuring {type(robot).__name__} ({robot.name})\n{'=' * 60}", flush=True)
        r = _measure(robot)
        results[r["robot_type"]] = r
        print(f"  eef_link:               {r['eef_link']}")
        print(f"  finger_links:           {r['finger_links']}")
        print(f"  gripper joints:         {r['gripper_joint_names']}")
        print(f"  joint_lower_limits:     {r['joint_lower_limits']}")
        print(f"  joint_upper_limits:     {r['joint_upper_limits']}")
        print(f"  derived max aperture:   {r['derived_max_aperture_m']:.6f} m")
        print(f"  joint @ close cmd(-1):  {r['closed_joint_positions']}")
        print(f"  joint @ open  cmd(+1):  {r['open_joint_positions']}")
        print(f"  fingertip dist closed:  {r['fingertip_distance_closed_m']:.6f} m  (link origin-to-origin)")
        print(f"  fingertip dist open:    {r['fingertip_distance_open_m']:.6f} m  (link origin-to-origin)")
        print(f"  inner-face gap closed:  {r['inner_surface_gap_closed_m']:.6f} m  (AABB inner-to-inner = MIN graspable)")
        print(f"  inner-face gap open:    {r['inner_surface_gap_open_m']:.6f} m  (AABB inner-to-inner = MAX graspable)")
        print(f"  eef→fingertip lengths:  {r['eef_to_fingertip_lengths_m']}")

    out_path = out_dir / "gripper_measurements.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
