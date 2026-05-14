#!/usr/bin/env python3
"""Apply each cached grasp from a .pt file in the actual training scene.

For each grasp index, restores the target to its scene-init pose, sets
the arm + gripper joints to the cached values, settles a few steps,
captures the viewer frame, and logs the distance from the gripper eef
to the target object. Stitches frames into an MP4 for inspection.

Lets you visually verify whether the cached arm_joint_pos values place
the eef on the target (collection geometry == training geometry) or
off by some delta (geometries don't match).

Usage:
    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python tools/render_cached_grasps_in_scene.py \\
            --scene-file datasets/.../task_0000/base/scene_ep1.joint.json \\
            --diagnostics-file datasets/.../task_0000/base/diagnostics.jsonl \\
            --grasp-pt outputs/grasp_datasets/task0000_200/cocktail_glass_xevdnl_success/grasps_cocktail_glass_xevdnl.pt \\
            --target-name cocktail_glass_178 \\
            --num-grasps 10 \\
            --output-dir outputs/grasp_datasets/task0000_200/cached_check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Import sentinel up-front so _apply_omnigibson_patches() runs (longfinger
# bundle resolver, Dropped/Upright states, etc.) before any OmniGibson code
# constructs a Franka. Without this the rendered robot uses the stock
# (short-finger) asset, not the longfinger geometry the .pt was collected on.
import sentinel  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-file", type=Path, required=True)
    p.add_argument("--diagnostics-file", type=Path, required=True)
    p.add_argument("--grasp-pt", type=Path, required=True)
    p.add_argument("--target-name", type=str, required=True)
    p.add_argument("--num-grasps", type=int, default=10,
                   help="How many grasp indices to render (evenly spaced).")
    p.add_argument("--settle-steps", type=int, default=30,
                   help="Sim steps after applying joints before capturing.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_datasets/task0000_200/cached_check"))
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument("--camera-eye", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Override viewer eye (defaults to a wide 3/4 view).")
    p.add_argument("--camera-lookat", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Override viewer lookat (defaults to the goblet).")
    return p.parse_args()


def _eye_lookat_to_quat(eye, lookat):
    import math
    import torch as th
    import omnigibson.utils.transform_utils as T
    d = np.asarray(lookat, dtype=np.float32) - np.asarray(eye, dtype=np.float32)
    d = d / max(1e-6, np.linalg.norm(d))
    return T.euler2quat(th.tensor([
        math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))),
        0.0,
        float(np.arctan2(-d[0], d[1])),
    ], dtype=th.float32)).tolist()


def _capture_frame(viewer_cam, target_hw):
    obs = viewer_cam.get_obs()[0]
    rgb = obs.get("rgb")
    if rgb is None:
        return None
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr.astype(np.uint8)
    if arr.shape[:2] != tuple(target_hw):
        from PIL import Image
        arr = np.asarray(Image.fromarray(arr).resize(
            (target_hw[1], target_hw[0]), Image.BILINEAR
        ))
    return arr


def _annotate(arr, lines):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    y = 8
    for line in lines:
        # Text shadow for readability
        draw.text((10, y + 1), line, fill=(0, 0, 0), font=font)
        draw.text((9, y), line, fill=(0, 0, 0), font=font)
        draw.text((11, y), line, fill=(0, 0, 0), font=font)
        draw.text((10, y - 1), line, fill=(0, 0, 0), font=font)
        draw.text((10, y), line, fill=(255, 255, 0), font=font)
        y += 22
    return np.asarray(img)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read diagnostics for room + camera
    with args.diagnostics_file.open() as f:
        diag = json.loads(f.readline())
    room = (diag.get("support_selection") or {}).get("room_instance")
    scene_model = diag.get("scene_model")

    # Camera: CLI override > 3/4 wide-angle default. The diagnostics
    # canonical camera is too tight to see the gripper.
    if args.camera_eye is not None and args.camera_lookat is not None:
        camera_eye = list(args.camera_eye)
        camera_lookat = list(args.camera_lookat)
    else:
        # Wide 3/4 view from robot's right side, elevated. Robot at (8.13,
        # 1.88, 0.72) facing +y; goblet on desk in front of it at (8.12,
        # 2.37, 0.83). This eye-position gives a clear view of arm + gripper
        # + goblet + desk.
        camera_eye = [9.2, 1.6, 1.6]
        camera_lookat = [8.13, 2.25, 0.85]

    # OG macros must be set before importing omnigibson
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    # Apply same AG window patch as collection time so AG fires consistently.
    from omnigibson.robots.manipulation_robot import m as _ag_macros
    _ag_macros.GRASP_WINDOW = 1.0 / 300.0
    _ag_macros.RELEASE_WINDOW = 1.0 / 300.0

    import omnigibson as og
    import torch as th

    # Build scene config (mirrors grasp_reset_scene.build_config branch with
    # InteractiveTraversableScene + load_room_instances).
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
        robots=[],
        objects=[],
        task=dict(type="DummyTask"),
    )

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG with scene "
          f"{scene_model} room {room} ...", flush=True)
    env = og.Environment(configs=cfg)
    env.reset()

    # Find target + robot
    target_obj = env.scene.object_registry("name", args.target_name)
    if target_obj is None:
        names = [o.name for o in env.scene.objects]
        raise SystemExit(
            f"Target {args.target_name!r} not in scene. Available: {names}"
        )
    robot = env.robots[0] if env.robots else None
    if robot is None:
        # The robot is in the scene via the scene_file
        raise SystemExit("No robot found in scene")

    print(f"  target {target_obj.name} world pos: "
          f"{target_obj.get_position_orientation()[0].cpu().numpy()}")
    print(f"  robot {robot.name} world pos: "
          f"{robot.get_position_orientation()[0].cpu().numpy()}")
    bundle = getattr(robot, "_franka_panda_asset_bundle", "<unset>")
    longfinger = bundle == "franka_panda_longfinger"
    print(f"  robot asset bundle: {bundle} "
          f"({'LONG' if longfinger else 'STANDARD'} fingers)")

    # Set viewer camera
    quat = _eye_lookat_to_quat(camera_eye, camera_lookat)
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(camera_eye, dtype=th.float32),
        orientation=th.tensor(quat, dtype=th.float32),
    )
    og.sim.step()
    viewer_cam = og.sim.viewer_camera

    # Load grasps
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading grasps from {args.grasp_pt}",
          flush=True)
    data = th.load(str(args.grasp_pt), map_location="cpu", weights_only=False)
    rel_pos = data["rel_position"].float()
    rel_ori = data["rel_orientation_xyzw"].float()
    arm_joint_pos = data["arm_joint_pos"].float()
    gripper_qpos = data["gripper_qpos"].float()
    N = arm_joint_pos.shape[0]
    print(f"  loaded N={N} grasps")

    # Snapshot initial poses
    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    initial_joint_pos = robot.get_joint_positions().clone()
    init_tgt_pos, init_tgt_quat = target_obj.get_position_orientation()
    init_tgt_pos = init_tgt_pos.detach().clone()
    init_tgt_quat = init_tgt_quat.detach().clone()

    # Pick evenly-spaced indices
    n = min(args.num_grasps, N)
    indices = np.linspace(0, N - 1, n, dtype=int).tolist()

    target_hw = (args.video_height, args.video_width)
    frames = []
    rows = []  # (idx, eef_world, dist_to_obj)

    for k, idx in enumerate(indices):
        # Restore target pose, zero velocities
        target_obj.set_position_orientation(
            position=init_tgt_pos, orientation=init_tgt_quat
        )
        target_obj.root_link.set_linear_velocity(th.zeros(3))
        target_obj.root_link.set_angular_velocity(th.zeros(3))

        # Apply cached arm + gripper joints
        robot.set_joint_positions(arm_joint_pos[idx], arm_control_idx)
        robot.set_joint_positions(gripper_qpos[idx], gripper_control_idx)
        for ctrl in robot._controllers.values():
            ctrl._goal = None

        # Build a hold_action that actively commands the gripper to close
        # — required for AG's contact-ray logic to fire (a static gripper
        # configuration produces no "squeeze" event). Mirrors what
        # GraspDatasetResetter does during its settle loop.
        hold_action = th.zeros(robot.action_dim, dtype=th.float32)
        hold_action[robot.arm_action_idx[arm]] = th.zeros(
            len(robot.arm_action_idx[arm]), dtype=th.float32
        )
        hold_action[robot.gripper_action_idx[arm]] = -1.0  # close

        # Settle: zero-gravity on target while gripper actively closes.
        target_obj.root_link.disable_gravity()
        try:
            for _ in range(args.settle_steps):
                robot.apply_action(hold_action)
                og.sim.step()
        finally:
            target_obj.root_link.enable_gravity()

        # Post-gravity hold steps to verify physical grasp under gravity.
        for _ in range(3):
            robot.apply_action(hold_action)
            og.sim.step()

        # Capture eef + obj world poses
        eef_pos = robot.eef_links[arm].get_position_orientation()[0].cpu().numpy()
        obj_pos = target_obj.get_position_orientation()[0].cpu().numpy()
        dist = float(np.linalg.norm(eef_pos - obj_pos))
        try:
            from omnigibson.controllers.controller_base import IsGraspingState
            ag_state = robot.is_grasping(arm, target_obj) == IsGraspingState.TRUE
        except Exception:
            ag_state = False

        rows.append({"idx": int(idx), "eef": eef_pos.tolist(),
                     "obj": obj_pos.tolist(), "dist": dist, "ag": bool(ag_state)})

        # Capture frame, annotate
        frame = _capture_frame(viewer_cam, target_hw)
        if frame is not None:
            lines = [
                f"grasp {idx:3d}/{N}  ({k+1}/{n})",
                f"eef:   ({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f})",
                f"obj:   ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})",
                f"dist:  {dist*100:.1f} cm   AG: {'YES' if ag_state else 'NO'}",
            ]
            frame = _annotate(frame, lines)
            frames.append(frame)

            png = args.output_dir / f"grasp_{idx:03d}.png"
            from PIL import Image
            Image.fromarray(frame).save(str(png))

        print(f"  grasp {idx:3d}: eef={eef_pos.round(3)} obj={obj_pos.round(3)} "
              f"dist={dist*100:.1f}cm ag={ag_state}", flush=True)

        # Cleanup AG bond before next iter
        try:
            robot.release_grasp_immediately(arm)
        except Exception:
            pass

    # Reset everything
    robot.set_joint_positions(initial_joint_pos)
    target_obj.set_position_orientation(position=init_tgt_pos, orientation=init_tgt_quat)

    # Write MP4
    if frames:
        import imageio
        mp4 = args.output_dir / "cached_grasps_in_scene.mp4"
        writer = imageio.get_writer(str(mp4), fps=args.fps, codec="libx264",
                                    macro_block_size=1, quality=7)
        try:
            for fr in frames:
                # Hold each for ~fps seconds (so 1 frame = 1 sec at fps=1)
                writer.append_data(fr)
        finally:
            writer.close()
        print(f"\nMP4: {mp4}")

    # Summary stats
    dists = np.array([r["dist"] for r in rows])
    ag_count = sum(1 for r in rows if r["ag"])
    print(f"\nSummary across {len(rows)} grasps:")
    print(f"  eef-to-obj dist: min={dists.min()*100:.1f}cm "
          f"max={dists.max()*100:.1f}cm mean={dists.mean()*100:.1f}cm")
    print(f"  AG holding: {ag_count}/{len(rows)}")

    # JSON log
    log = args.output_dir / "results.json"
    log.write_text(json.dumps(rows, indent=2))
    print(f"  results: {log}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
