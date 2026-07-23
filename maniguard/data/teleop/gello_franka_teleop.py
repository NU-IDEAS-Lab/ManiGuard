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
    python -m maniguard.data.teleop.gello_franka_teleop \
        --snapshot outputs/teleop_scenes/table/scene_ep0000.json \
        --output-hdf5 outputs/teleop/table/scene_ep0000.hdf5

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
_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOYLO = _REPO_ROOT / "behavior-1k" / "joylo"
if str(_JOYLO) not in sys.path:
    sys.path.insert(0, str(_JOYLO))

from gello.robots.dynamixel import DynamixelRobot  # noqa: E402

# Long-finger Franka assets are now eagerly patched in via
# maniguard/_omnigibson_patches.py:_patch_franka_longfinger() at OmniGibson
# init time, so this entry no longer needs to install anything.

# Reuse so101's diagnostics-jsonl reader for goal_checker auto-success.
from maniguard.data.teleop.so101_franka_teleop import _read_first_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# GELLO calibration constants (from `gello_get_offset.py`, 2026-04-27)
# Re-run gello_get_offset.py and update these if you re-flash IDs / replace
# servos / change finger geometry.
# ---------------------------------------------------------------------------
GELLO_PORT = "/dev/ttyUSB0"  # override per machine with --gello-port (see `ls /dev/serial/by-id/`)
GELLO_JOINT_IDS = (1, 2, 3, 4, 5, 6, 7)
GELLO_JOINT_OFFSETS = [
    # 2026-05-10 recal: ran gello_get_offset.py with --start-joints 0 0 0 0 0 0 0.
    # Same calibration target as the 2026-05-05 run — only the raw multiples
    # of π/2 changed because three servos (J3, J5, J7) wrapped to different
    # turn counts on this power-up. Trims (J2, J4, J6) are unchanged.
    # Per-joint: delta_offset = -CALIB_POSE[i] * sign[i].
    3 * np.pi / 2,                                    # J1: no trim (CALIB=0)
    2 * np.pi / 2 - np.pi / 4,                        # J2: sign=-1, want -π/4 → δ=-π/4
    4 * np.pi / 2,                                    # J3: no trim (CALIB=0; was 8π/2 last recal — wrapped back 4π)
    1 * np.pi / 2 + np.pi / 4 + np.pi / 9,            # J4: sign=+1, want -π/4-π/9 → δ=+π/4+π/9
    -4 * np.pi / 2,                                   # J5: no trim (CALIB=0; servo wrapped to negative this time)
    1 * np.pi / 2 + 0.0175,                           # J6: sign=+1, want -0.0175 → δ=+0.0175
    4 * np.pi / 2,                                    # J7: no trim (CALIB=0; was 0π/2 — wrapped 4π more)
]
GELLO_JOINT_SIGNS = (1, -1, 1, 1, 1, 1, 1)
GELLO_GRIPPER_CONFIG = None  # no physical gripper attached yet — keyboard takes over

# Franka joint pose that corresponds to GELLO held at the calibration
# reference pose (the physical pose used during gello_get_offset.py),
# AFTER applying the trims baked into GELLO_JOINT_OFFSETS. This is what
# we seed Franka at on env.reset() — deterministic across launches and
# independent of where GELLO physically is at startup. The teleop loop
# then ramps Franka from this pose to GELLO's actual current reading
# over GELLO_RAMP_STEPS steps so there's no jolt.
GELLO_CALIBRATION_FRANKA_POSE = (
    0.0,                                # J1 (calibrated to 0)
    -np.pi / 4,                         # J2 (raw -1.7628 + trim 0.978 = -π/4)
    0.0,                                # J3 (calibrated to 0)
    -np.pi / 4 - np.pi / 9,             # J4 (raw -3.0718 + trim 2.286 = -π/4 - π/9 ≈ -65°)
    0.0,                                # J5 (calibrated to 0)
    -0.0175,                            # J6 (raw J6 lower limit)
    0.0,                                # J7 (raw 0 + trim -π/4 + actual -45° spin = 0)
)
GELLO_RAMP_STEPS = 60                   # ~2 s at 30 Hz to drive Franka from calibration pose to GELLO current


# ---------------------------------------------------------------------------
# External cameras (mirrors so101_franka_teleop)
# ---------------------------------------------------------------------------
from maniguard.utils.camera_setup import build_external_camera_configs  # noqa: E402

_EXTERNAL_CAMERAS = build_external_camera_configs()


# ---------------------------------------------------------------------------
# Snapshot loader (same shape as so101 but swaps IK -> JointController)
# ---------------------------------------------------------------------------
def _build_from_snapshot(
    snapshot_path,
    robot_type="FrankaPanda",
    grasping_mode="physical",
    initial_joint_pos=None,
):
    """Rewrite snapshot's robot entry for joint-position teleop.

    grasping_mode selects OmniGibson's grasping semantics — same options as
    so101_franka_teleop: 'physical' / 'assisted' / 'sticky'.

    initial_joint_pos (None or 7-element sequence): if given, overwrite the
    snapshot's saved arm joint_pos[0:7] with these values. Used to seed the
    follower at the operator's current GELLO pose so env.reset() doesn't
    snap the arm to the snapshot's saved pose (which can be 100° off from
    where GELLO currently is). Gripper joints (7:9) are left untouched.
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
        if initial_joint_pos is not None:
            saved_jp = robot_state.get("joint_pos")
            if saved_jp is not None and len(saved_jp) >= 7:
                for i in range(7):
                    saved_jp[i] = float(initial_joint_pos[i])
                # Also zero arm joint velocities so env.reset() doesn't
                # restore a non-zero v that would jolt the arm at startup.
                jv = robot_state.get("joint_vel")
                if jv is not None and len(jv) >= 7:
                    for i in range(7):
                        jv[i] = 0.0
                print(f"[Gello] Snapshot arm joint_pos overridden with GELLO leader pose:")
                print(f"        {[round(float(v), 3) for v in initial_joint_pos[:7]]} rad")

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
    """Place the external cameras at the task's PRESET poses (the shared
    load-side rule, camera_setup.place_recorded_task_cameras): the
    diagnostics.jsonl next to --snapshot carries the recorded ``cameras``
    poses — the same views datagen/eval/playback use — with the canonical
    robot-frame recompute as the (warned) fallback for legacy snapshots."""
    from maniguard.utils.camera_setup import place_recorded_task_cameras

    diagnostics = None
    diag_path = os.path.join(os.path.dirname(args.snapshot), "diagnostics.jsonl")
    if os.path.isfile(diag_path):
        diagnostics = _read_first_jsonl(diag_path)
    place_recorded_task_cameras(env, diagnostics, set_viewer=True)


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
    parser.add_argument("--no-lid-snap", action="store_true",
                        help="Disable the eager lid/cap → container "
                             "snap-attach. By default, when an eligible "
                             "pair is loaded, the lid auto-attaches when "
                             "the lid is touching the container and the "
                             "gripper has released it.")
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

    # ----- Connect GELLO before building the env so DynamixelRobot
    #       failures (port busy, no power, …) surface fast — before the
    #       30-90 s OmniGibson startup. We don't seed the snapshot from
    #       the leader's current reading anymore: GELLO is passive when
    #       not actively held, so a "startup snapshot of the leader" is
    #       non-deterministic. Instead we seed Franka at a fixed pose
    #       (GELLO_CALIBRATION_FRANKA_POSE) below, then ramp toward
    #       leader.get_joint_state() over GELLO_RAMP_STEPS steps in the
    #       main loop.
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

    # ----- Build env from snapshot, seeded at the deterministic
    #       calibration-reference pose (NOT the leader's current reading) -----
    cfg = _build_from_snapshot(
        args.snapshot,
        grasping_mode=args.grasping_mode,
        initial_joint_pos=np.asarray(GELLO_CALIBRATION_FRANKA_POSE, dtype=np.float32),
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

    # Eager (lid|cap, container) snap-attach. No-op when no eligible pair
    # is loaded. Discovery runs once, hooked after each env.step below.
    from maniguard.utils.lid_attach import LidSnapper
    lid_snapper = None if args.no_lid_snap else LidSnapper(env)

    # ----- Cameras -----
    _setup_cameras_for_scene(env, robot, args)

    # ----- Hotkeys -----
    KeyboardEventHandler.initialize()
    state = {
        "quit": False,
        "gripper_open": bool(args.start_gripper_open),
        "success": False,            # authoritative success flag (auto OR manual)
        "manual_override": False,    # operator-forced success via S key
        # When recording, the trajectory only begins once the operator presses
        # B. Until then the loop still steps (physics/viewport stay live and
        # GELLO mirroring works while you reposition the camera), but every
        # frame is discarded so the saved trajectory has no idle lead-in.
        # Nothing to trim when not writing an HDF5 -> treat as already started.
        "recording_started": recorder is None,
    }

    # Auto-success: if a sibling diagnostics.jsonl exists (HF furnished
    # scenes ship one with goal_region + goal_conditions), build a goal
    # checker that fires success the moment the green-sphere goal region
    # is satisfied — no need for the operator to press S.
    success_checker = None
    success_detail = {}
    task_prompt = None
    task_target = None
    diagnostics_path = os.path.join(os.path.dirname(args.snapshot), "diagnostics.jsonl")
    if os.path.isfile(diagnostics_path):
        diagnostics = _read_first_jsonl(diagnostics_path)
        task_prompt = diagnostics.get("prompt")
        task_target = diagnostics.get("goal_region", {}).get("target_name")
        from maniguard.eval.goal_checker import build_goal_checker
        success_checker = build_goal_checker(
            {
                "goal_region": diagnostics.get("goal_region"),
                "goal_conditions": diagnostics.get("goal_conditions", []),
            }
        )
        if success_checker is not None:
            success_checker.resolve(env)
            print("[Gello] Loaded success checker from diagnostics.")

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
            state["manual_override"] = not state["manual_override"]
            tag = "FORCE-SUCCESS" if state["manual_override"] else "auto-checker"
            print(f"[Gello] Success mode -> {tag}")

        def _on_begin():
            if state["recording_started"]:
                print("[Gello] Recording already started.")
                return
            state["recording_started"] = True
            # Drop the pre-warm frames so the saved trajectory's first frame is
            # this instant. Keep step_count in sync (mirrors rollback's
            # bookkeeping) and clear any checkpoint taken before the real start.
            dropped = len(recorder.current_traj_history)
            recorder.step_count -= dropped
            recorder.current_traj_history.clear()
            if hasattr(recorder, "checkpoint_states"):
                recorder.checkpoint_states.clear()
            if hasattr(recorder, "checkpoint_step_idxs"):
                recorder.checkpoint_step_idxs.clear()
            print(f"[Gello] ▶ Recording STARTED (discarded {dropped} pre-warm frames).")

        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C, callback_fn=_on_checkpoint,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R, callback_fn=_on_rollback,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S, callback_fn=_on_success,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.B, callback_fn=_on_begin,
        )

    # ----- Action assembly -----
    arm_idx = robot.controller_action_idx["arm_0"]
    gripper_idx = robot.gripper_action_idx["0"]
    action_dim = robot.action_dim
    open_value = -1.0 if args.invert_gripper else 1.0
    close_value = 1.0 if args.invert_gripper else -1.0

    print("\n" + "=" * 50)
    print(f"GELLO → Franka Teleop Ready  [{args.snapshot}]")
    if task_prompt:
        print(f"  TASK   : {task_prompt}")
    if task_target:
        print(f"  TARGET : {task_target}")
    print("Move the GELLO arm to drive Franka joints 1:1")
    print("SPACE = toggle gripper   Q = clean exit")
    if recorder is not None:
        print("B = begin recording (discards idle lead-in)")
        print("S = mark success   C = checkpoint   R = rollback")
    print("=" * 50 + "\n")

    # Capture Franka's actual starting joint pose for the ramp source.
    # env.reset() already loaded the calibration pose into the snapshot,
    # so this should match GELLO_CALIBRATION_FRANKA_POSE within numerical
    # noise — but reading the live value is robust to URDF joint-limit
    # clamping etc.
    ramp_source = np.asarray(GELLO_CALIBRATION_FRANKA_POSE, dtype=np.float32)
    try:
        live = robot.get_joint_positions().cpu().numpy()
        ramp_source = live[:7].astype(np.float32)
    except Exception as _exc:
        print(f"[Gello] Could not read live arm pose for ramp source ({_exc}); "
              f"falling back to GELLO_CALIBRATION_FRANKA_POSE.")
    print(f"[Gello] Ramp: {ramp_source.round(3).tolist()} -> live GELLO over "
          f"{GELLO_RAMP_STEPS} steps")

    try:
        for step in range(args.steps):
            if state["quit"]:
                break
            target = np.asarray(leader.get_joint_state(), dtype=np.float32)[:7]
            if step < GELLO_RAMP_STEPS:
                alpha = (step + 1) / GELLO_RAMP_STEPS
                arm_action = (1.0 - alpha) * ramp_source + alpha * target
            else:
                arm_action = target
            action = th.zeros(action_dim, dtype=th.float32)
            action[arm_idx] = th.tensor(arm_action, dtype=th.float32)
            action[gripper_idx] = th.tensor(
                [open_value if state["gripper_open"] else close_value],
                dtype=th.float32,
            )
            env.step(action)

            if lid_snapper is not None:
                lid_snapper.try_snap(robot=robot)

            # Pre-warm: until B is pressed, discard every recorded frame so the
            # saved trajectory starts the instant the operator begins. Steps
            # still run (viewport + GELLO mirroring stay live for repositioning
            # the camera). Skipping the rest also blocks any premature success.
            if not state["recording_started"]:
                if recorder is not None:
                    recorder.step_count -= len(recorder.current_traj_history)
                    recorder.current_traj_history.clear()
                if step % 150 == 0:
                    print("[Gello] (pre-warm — press B to begin recording)")
                continue

            # Auto-success — break the loop the moment the goal region fires.
            if success_checker is not None:
                auto_success, success_detail = success_checker.check(env)
                if auto_success:
                    state["success"] = True
                    print(f"[Gello] Success satisfied: {success_detail}")
                    break
            # Manual override still wins (S-key force).
            if state["manual_override"]:
                state["success"] = True
                break

            if step % 300 == 0 and step > 0:
                grip_str = "OPEN" if state["gripper_open"] else "CLOSE"
                ramp_str = "ramp" if step < GELLO_RAMP_STEPS else "live"
                print(f"step {step}/{args.steps} gripper={grip_str} "
                      f"first_joint={float(target[0]):+.3f} rad ({ramp_str}) "
                      f"success={state['success']}")

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
