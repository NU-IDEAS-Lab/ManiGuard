"""
GELLO Leader Arm → Franka Teleop (joint-mirroring)

Teleoperates a FrankaPanda in OmniGibson by reading 7 calibrated joint
angles directly from a GELLO leader (no IK on the follower side; the
leader's kinematics are 1:1 with the Franka). Loads a pipeline-generated
scene snapshot and optionally records a trajectory via DataCollectionWrapper.

Differs from so101_franka_teleop.py in three ways:
  - Follower uses `JointController` (mode=position, absolute) — actions
    are 7 raw joint radians, no IK solver in the loop.
  - Leader is the bundled joylo `DynamixelRobot`; we read calibrated
    joints directly via `get_joint_state()` and skip the higher-level
    GelloAgent wrapper (which adds force-feedback / cooldown machinery
    we don't need for sim-only data collection).
  - Gripper has no physical leader yet — bind to SPACE key (toggle
    open/close) so a single operator can drive both the arm via GELLO
    and the gripper via keyboard.

Usage:
    python -m sentinel.teleop.gello_franka_teleop \
        --snapshot outputs/teleop_scenes/table/scene_ep0000.json \
        --output-hdf5 outputs/jixing_teleop2_hdf5/table/scene_ep0000.hdf5

Hotkeys (need GUI focus on the OmniGibson viewport):
    SPACE = toggle gripper open/close
    S     = toggle success flag (recording is saved iff success when --only-successes)
    C     = save checkpoint
    R     = rollback to last checkpoint
    Q     = clean exit (writes HDF5)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Vulkan ICD must be set BEFORE OmniGibson imports trigger Isaac Sim's
# renderer init. Use setdefault so an explicit shell override still wins
# (e.g. if the user has a non-NVIDIA / different driver path).
os.environ.setdefault(
    "VK_ICD_FILENAMES",
    "/usr/share/vulkan/icd.d/nvidia_icd.json",
)

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs import DataCollectionWrapper
from omnigibson.utils.constants import LightingMode
from omnigibson.utils.ui_utils import KeyboardEventHandler


# ---------------------------------------------------------------------------
# joylo on sys.path (we don't pip install it because its setup.py pulls in
# telemoma / pyglm / joycon / pybullet / etc which we don't need)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_JOYLO = _REPO_ROOT / "behavior-1k" / "joylo"
if str(_JOYLO) not in sys.path:
    sys.path.insert(0, str(_JOYLO))

from gello.robots.dynamixel import DynamixelRobot  # noqa: E402

# Long-finger Franka assets are now eagerly patched in via
# sentinel/_omnigibson_patches.py:_patch_franka_longfinger() at OmniGibson
# init time, so this entry no longer needs to install anything.


# ---------------------------------------------------------------------------
# GELLO calibration constants (from `gello_get_offset.py`, 2026-04-27)
# Re-run gello_get_offset.py and update these if you re-flash IDs / replace
# servos / change finger geometry.
# ---------------------------------------------------------------------------
GELLO_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNJP-if00-port0"
GELLO_JOINT_IDS = (1, 2, 3, 4, 5, 6, 7)
GELLO_JOINT_OFFSETS = [
    3 * np.pi / 2,   # J1
    2 * np.pi / 2 + (1.7628 - np.pi / 4),   # J2  (trim: GELLO physical max-back -> Franka -π/4 instead of joint limit -1.7628, for a relaxed rest pose)
    0 * np.pi / 2,   # J3
    3 * np.pi / 2 - (3.0718 - np.pi / 4 - np.pi / 9),   # J4  (trim: GELLO physical max-forward -> Franka ~-65° (-π/4 - 20°), slightly more bent than -π/4)
    0 * np.pi / 2,   # J5
    2 * np.pi / 2,   # J6
    4 * np.pi / 2 - np.pi / 4,   # J7  (-π/4 trim: GELLO J7 mounting offset, calibration script rounds to π/2)
]
GELLO_JOINT_SIGNS = (1, -1, 1, 1, 1, 1, 1)
GELLO_GRIPPER_CONFIG = None  # no physical gripper attached yet — keyboard takes over


# ---------------------------------------------------------------------------
# External cameras (mirrors so101_franka_teleop)
# ---------------------------------------------------------------------------
from sentinel.utils.camera_setup import build_external_camera_configs  # noqa: E402

_EXTERNAL_CAMERAS = build_external_camera_configs()


# ---------------------------------------------------------------------------
# Snapshot loader (same shape as so101 but swaps IK -> JointController)
# ---------------------------------------------------------------------------
def _build_from_snapshot(
    snapshot_path,
    robot_type="FrankaPanda",
    grasping_mode="physical",
):
    """Rewrite snapshot's robot entry for joint-position teleop.

    grasping_mode selects OmniGibson's grasping semantics — same options as
    so101_franka_teleop: 'physical' / 'assisted' / 'sticky'.
    """
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    robot_key = None
    for k, info in snap.get("objects_info", {}).get("init_info", {}).items():
        if "Franka" in info.get("class_name", ""):
            robot_key = k
            break
    if robot_key is None:
        raise RuntimeError(f"No Franka robot found in {snapshot_path}")

    entry = snap["objects_info"]["init_info"][robot_key]
    old_class = entry["class_name"]
    entry["class_module"] = "omnigibson.robots.franka"
    entry["class_name"] = robot_type
    entry["args"].pop("expected_file_hash", None)
    # Absolute joint position control — actions are raw radians, not deltas
    # and not normalized. Matches what GELLO publishes directly.
    entry["args"]["action_normalize"] = False
    entry["args"]["controller_config"] = {
        "arm_0": {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": None,
            "command_output_limits": None,
            "use_delta_commands": False,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": None,
            "mode": "binary",
        },
    }
    entry["args"]["grasping_mode"] = grasping_mode

    # Drop saved controller goals: the snapshot was taken with a different
    # controller stack (OperationalSpace / IK) whose goal-state shape doesn't
    # match JointController. Both BaseController._load_state and
    # MultiFingerGripperController._load_state guard their reads behind
    # `if self._goal is not None`, so a null goal sidesteps the mismatch.
    robot_state = (
        snap.get("state", {})
        .get("registry", {})
        .get("object_registry", {})
        .get(robot_key)
    )
    if robot_state is not None:
        robot_state["controllers"] = {
            "arm_0": {"goal_is_valid": False, "goal": None},
            "gripper_0": {"goal_is_valid": False, "goal": None},
        }

    # Lift the base only when swapping a floor-mounted FrankaMounted snapshot
    # to FrankaPanda. Snapshots already saved as FrankaPanda (HF furnished
    # scenes with the robot desk-mounted) are at the correct height.
    needs_lift = old_class == "FrankaMounted" and robot_type == "FrankaPanda"
    if needs_lift:
        if robot_state is not None:
            root_link = robot_state.get("root_link")
            if root_link is not None and "pos" in root_link:
                root_link["pos"][2] = float(root_link["pos"][2]) + 0.5
        args_pos = entry.get("args", {}).get("position")
        if args_pos is not None:
            args_pos[2] = float(args_pos[2]) + 0.5

    stem, ext = os.path.splitext(snapshot_path)
    teleop_path = f"{stem}_gello_teleop{ext}"
    with open(teleop_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    n_objs = len(snap["objects_info"]["init_info"]) - 1
    print(f"[Gello] Loaded snapshot: {snapshot_path}")
    print(f"[Gello] Robot {old_class} -> {robot_type} (JointController), "
          f"temp snapshot: {teleop_path}")
    print(f"[Gello] {n_objs} non-robot objects + 3 external cameras")

    return dict(
        scene={"type": "Scene", "scene_file": teleop_path},
        task={"type": "DummyTask"},
        env={"external_sensors": _EXTERNAL_CAMERAS},
    )


# ---------------------------------------------------------------------------
# Camera placement (copied from so101_franka_teleop.main; same logic)
# ---------------------------------------------------------------------------
def _setup_cameras_for_scene(env, robot, args):
    import omnigibson.utils.transform_utils as _T
    support_obj = env.scene.object_registry("name", "support_surface")

    if support_obj is not None:
        target_obj = next(
            (obj for obj in env.scene.objects
             if obj is not robot and obj is not support_obj),
            support_obj,
        )
        from sentinel.task_generation.utils.video import (
            build_video_view_specs, setup_cameras,
        )
        views = build_video_view_specs(args, robot, target_obj, support_obj=support_obj)
        setup_cameras(env, views)
        print(f"[Gello] Camera mode = support_surface; target={target_obj.name}")
        return

    # Robot-frame fallback for HF furnished scenes.
    from sentinel.task_generation.utils.video import setup_cameras

    rp_t, rq_t = robot.get_position_orientation()
    rp = np.asarray(rp_t.cpu().numpy() if hasattr(rp_t, "cpu") else rp_t,
                    dtype=np.float32)
    rmat_t = _T.quat2mat(rq_t)
    rmat = np.asarray(rmat_t.cpu().numpy() if hasattr(rmat_t, "cpu") else rmat_t,
                      dtype=np.float32)
    forward = rmat[:, 0].copy()
    forward[2] = 0.0
    n = float(np.linalg.norm(forward))
    forward = forward / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)

    cam_height_off = 0.9
    back_off = 1.2
    side_off = 1.0
    side_forward_off = 0.2

    workspace = rp + forward * 0.45 + np.array([0, 0, 0.05], dtype=np.float32)
    opp_eye = rp - forward * back_off + np.array([0, 0, cam_height_off], dtype=np.float32)
    left_eye = rp + left * side_off + forward * side_forward_off \
                  + np.array([0, 0, cam_height_off], dtype=np.float32)
    right_eye = rp - left * side_off + forward * side_forward_off \
                   + np.array([0, 0, cam_height_off], dtype=np.float32)
    views = [
        {"label": "opposite_side_front", "eye": opp_eye.tolist(),  "lookat": workspace.tolist()},
        {"label": "left_overview",       "eye": left_eye.tolist(), "lookat": workspace.tolist()},
        {"label": "right_overview",      "eye": right_eye.tolist(),"lookat": workspace.tolist()},
    ]
    setup_cameras(env, views)
    print(f"[Gello] Camera mode = robot-frame; "
          f"robot_pos=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}), "
          f"forward=({forward[0]:.2f},{forward[1]:.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GELLO -> Franka Teleop (joint mirroring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Path to a pipeline scene snapshot JSON (scene_ep*.json)")
    parser.add_argument("--output-hdf5", type=str, default=None,
                        help="If set, wrap env in DataCollectionWrapper and write trajectory here.")
    parser.add_argument("--only-successes", action="store_true",
                        help="Only persist successful episodes (toggled with S key)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Maximum sim steps before forced exit")
    parser.add_argument("--gello-port", type=str, default=GELLO_PORT,
                        help="USB serial path for the GELLO leader")
    parser.add_argument("--invert-gripper", action="store_true",
                        help="Swap which SPACE state means open vs close")
    parser.add_argument("--start-gripper-open", action="store_true",
                        help="Begin with the gripper OPEN (default starts CLOSED)")
    # ---- Gripper / asset knobs (same shape as so101_franka_teleop) ----
    parser.add_argument("--grasping-mode", choices=["physical", "assisted", "sticky"],
                        default="physical",
                        help="OmniGibson grasping semantics. 'physical' = pure Coulomb "
                             "contact. 'assisted' = AG raycast + FixedJoint weld when both "
                             "fingers contact. 'sticky' = any finger contact + close welds. "
                             "Use 'assisted' for thin/flat objects when slip is the bottleneck.")
    parser.add_argument("--gpu-dynamics", action="store_true",
                        help="Enable PhysX GPU dynamics (gm.USE_GPU_DYNAMICS=True). "
                             "REQUIRED for fluid / particle / cloth simulation — "
                             "without it liquid scenes render empty (no particles "
                             "spawned, no physics). Costs extra VRAM; only enable "
                             "when the scene actually has fluids/particles/cloth.")
    args = parser.parse_args()

    # Apply global macros BEFORE creating og.Environment — simulator.py reads
    # gm.USE_GPU_DYNAMICS during _init_physx and won't pick up later changes.
    if args.gpu_dynamics:
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        print("[Gello] gm.USE_GPU_DYNAMICS = True (fluids/particles/cloth enabled)")

    # ----- Build env from snapshot -----
    cfg = _build_from_snapshot(
        args.snapshot,
        grasping_mode=args.grasping_mode,
    )
    print(f"[Gello] grasping_mode = {args.grasping_mode}")
    env = og.Environment(configs=cfg)
    env.reset()

    # ----- Lighting (mirrors so101) -----
    og.sim.add_skybox()
    try:
        og.sim._skybox.intensity = 12000
    except Exception as _exc:
        print(f"[Gello] Could not bump skybox intensity ({_exc}); using default 2500.")
    og.sim.set_lighting_mode(LightingMode.CAMERA)
    print("[Gello] Skybox dome added; viewport lighting = CAMERA.")

    # ----- HDF5 recording (mirrors so101) -----
    recorder = None
    if args.output_hdf5:
        recorder = DataCollectionWrapper(
            env=env,
            output_path=args.output_hdf5,
            only_successes=args.only_successes,
            flush_every_n_traj=1,
        )
        env = recorder
        print(f"[Gello] Recording HDF5 to {args.output_hdf5} "
              f"(only_successes={args.only_successes})")

    robot = env.robots[0]

    # ----- Cameras -----
    _setup_cameras_for_scene(env, robot, args)

    # ----- Connect GELLO -----
    print(f"[Gello] Connecting to leader at {args.gello_port}")
    leader = DynamixelRobot(
        joint_ids=GELLO_JOINT_IDS,
        joint_offsets=list(GELLO_JOINT_OFFSETS),
        joint_signs=list(GELLO_JOINT_SIGNS),
        port=args.gello_port,
        real=True,
        gripper_config=GELLO_GRIPPER_CONFIG,
        start_joints=None,
    )
    n_dofs = leader.num_dofs()
    print(f"[Gello] Connected. DOFs={n_dofs} (expected 7 for arm-only)")
    if n_dofs != 7:
        print(f"[Gello] WARNING: expected 7 DOFs but got {n_dofs}. "
              f"If gripper is now wired, update GELLO_GRIPPER_CONFIG and re-run.")

    # ----- Hotkeys -----
    KeyboardEventHandler.initialize()
    state = {
        "quit": False,
        "gripper_open": bool(args.start_gripper_open),
        "success": False,
    }

    def _on_quit():
        state["quit"] = True
        print("[Gello] Quit requested -- flushing and shutting down...")

    def _on_space():
        state["gripper_open"] = not state["gripper_open"]
        print(f"[Gello] Gripper -> {'OPEN' if state['gripper_open'] else 'CLOSE'}")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Q, callback_fn=_on_quit,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.SPACE, callback_fn=_on_space,
    )
    if recorder is not None:
        def _on_checkpoint():
            recorder.update_checkpoint()
            print(f"[Gello] Checkpoint saved (total={len(recorder.checkpoint_states)})")

        def _on_rollback():
            if not recorder.checkpoint_states:
                print("[Gello] Rollback: no checkpoint yet.")
                return
            recorder.rollback_to_checkpoint()
            print(f"[Gello] Rolled back ({len(recorder.checkpoint_states)} remain)")

        def _on_success():
            state["success"] = not state["success"]
            tag = "SUCCESS" if state["success"] else "not-success"
            print(f"[Gello] Episode marked {tag}")

        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C, callback_fn=_on_checkpoint,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R, callback_fn=_on_rollback,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S, callback_fn=_on_success,
        )

    # ----- Action assembly -----
    arm_idx = robot.controller_action_idx["arm_0"]
    gripper_idx = robot.gripper_action_idx["0"]
    action_dim = robot.action_dim
    open_value = -1.0 if args.invert_gripper else 1.0
    close_value = 1.0 if args.invert_gripper else -1.0

    print("\n" + "=" * 50)
    print(f"GELLO → Franka Teleop Ready  [{args.snapshot}]")
    print("Move the GELLO arm to drive Franka joints 1:1")
    print("SPACE = toggle gripper   Q = clean exit")
    if recorder is not None:
        print("S = mark success   C = checkpoint   R = rollback")
    print("=" * 50 + "\n")

    try:
        for step in range(args.steps):
            if state["quit"]:
                break
            joints = leader.get_joint_state()  # 7-dim numpy (radians, calibrated)
            action = th.zeros(action_dim, dtype=th.float32)
            action[arm_idx] = th.tensor(np.asarray(joints, dtype=np.float32))
            action[gripper_idx] = th.tensor(
                [open_value if state["gripper_open"] else close_value],
                dtype=th.float32,
            )
            env.step(action)

            if step % 300 == 0 and step > 0:
                grip_str = "OPEN" if state["gripper_open"] else "CLOSE"
                print(f"step {step}/{args.steps} gripper={grip_str} "
                      f"first_joint={float(joints[0]):+.3f} rad")

    except KeyboardInterrupt:
        print("\nStopping teleop (SIGINT)...")
    finally:
        if recorder is not None:
            inner_env = recorder.env
            if hasattr(inner_env, "task") and inner_env.task is not None:
                inner_env.task._success = bool(state["success"])
            print(f"[Gello] Saving with success={state['success']} "
                  f"(only_successes={args.only_successes})")
            recorder.save_data()
            print(f"[Gello] HDF5 written: {args.output_hdf5}")
        og.clear()


if __name__ == "__main__":
    main()
