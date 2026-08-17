"""
SO-101 Leader Arm → Franka Teleop

Teleoperates a FrankaPanda in OmniGibson using an SO-101 leader arm.
The SO-101 end-effector deltas are mapped to Franka IK targets. Loads a
pipeline-generated scene snapshot and optionally records a trajectory via
DataCollectionWrapper.

Usage:
    python -m maniguard.data.teleop.so101_franka_teleop \
        --snapshot outputs/pipeline_runs/<run>/scene_ep1.json

Prerequisites:
    Terminal 1 (lerobot venv):
        python teleop_bridge/so101_server.py --mock          # mock mode
        python teleop_bridge/so101_server.py --port /dev/ttyACM0  # real hardware

    Terminal 2 (behavior conda):
        python -m maniguard.data.teleop.so101_franka_teleop \
            --snapshot <path>
"""

import argparse
import json
import os

import omnigibson as og
from omnigibson import lazy
from omnigibson.envs import DataCollectionWrapper
from omnigibson.utils.ui_utils import KeyboardEventHandler

from maniguard.data.teleop.so101_teleop import SO101TeleopAgent, SO101TeleopConfig

# ---------------------------------------------------------------------------
# Robot config helper
# ---------------------------------------------------------------------------

def _robot_cfg(robot_type="FrankaPanda"):
    """Franka arm with IK controller for teleop.

    Defaults to FrankaPanda (full reach, free base); pass "FrankaMounted"
    to match BasePipeline's task-generation robot (fixed chassis, reduced
    workspace).
    """
    return {
        "type": robot_type,
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "physical",
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": None,
                "mode": "binary",
            },
        },
    }


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

from maniguard.utils.camera_setup import build_external_camera_configs

_EXTERNAL_CAMERAS = build_external_camera_configs()


def _read_first_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


def _build_from_snapshot(
    snapshot_path,
    robot_type="FrankaPanda",
    grasping_mode="physical",
):
    """Load an og.sim.save()-style scene snapshot via scene_file.

    Rewrites the robot entry in a copy of the snapshot so the loaded
    robot uses `robot_type` (default FrankaPanda for full reach) with an
    IK controller, then hands that rewritten copy to scene_file.
    OmniGibson reconstructs everything else (support surface, objects,
    poses, dynamics) from the snapshot directly.

    FrankaPanda and FrankaMounted share the same 9-DOF joint layout (7
    arm + 2 gripper), so the saved joint_pos transfers cleanly; only
    the chassis mesh + base mount differ.
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
    entry["args"]["controller_config"] = {
        "arm_0": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": None,
            "mode": "binary",
        },
    }
    entry["args"]["grasping_mode"] = grasping_mode

    # Replace the saved controller state with a minimal null-goal entry
    # per controller. The snapshot was taken with an
    # OperationalSpaceController (no control_filter field), so its state
    # shape doesn't match our swapped-in IK controller. Both
    # InverseKinematicsController._load_state and MultiFingerGripperController
    # ._load_state guard their filter-state reads behind
    # `if self._goal is not None`, so a null goal sidesteps the mismatch
    # entirely. joint_pos / joint_vel / root_link are kept so the arm
    # appears in its saved pose.
    robot_state = snap.get("state", {}).get("registry", {}).get("object_registry", {}).get(robot_key)
    if robot_state is not None:
        robot_state["controllers"] = {
            "arm_0": {"goal_is_valid": False, "goal": None},
            "gripper_0": {"goal_is_valid": False, "goal": None},
        }
    # Lift the base by 0.5m only when swapping a floor-mounted FrankaMounted
    # snapshot to FrankaPanda — without this the arm would sit on the floor.
    # Snapshots already saved as FrankaPanda (e.g. HF furnished scenes where
    # the robot is desk-mounted) are at the correct height.
    needs_lift = old_class == "FrankaMounted" and robot_type == "FrankaPanda"
    if needs_lift:
        if robot_state is not None:
            root_link = robot_state.get("root_link")
            if root_link is not None and "pos" in root_link:
                root_link["pos"][2] = float(root_link["pos"][2]) + 0.5
        args_pos = entry.get("args", {}).get("position")
        if args_pos is not None:
            args_pos[2] = float(args_pos[2]) + 0.5

    # Write the rewritten snapshot next to the original.
    stem, ext = os.path.splitext(snapshot_path)
    teleop_path = f"{stem}_teleop{ext}"
    with open(teleop_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    n_objs = len(snap["objects_info"]["init_info"]) - 1
    print(f"[Teleop] Loaded snapshot: {snapshot_path}")
    print(f"[Teleop] Robot {old_class} -> {robot_type} (IK), "
          f"temp snapshot: {teleop_path}")
    print(f"[Teleop] {n_objs} non-robot objects + 3 external cameras")

    return dict(
        scene={
            "type": "Scene",
            "scene_file": teleop_path,
        },
        task={"type": "DummyTask"},
        env={"external_sensors": _EXTERNAL_CAMERAS},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SO-101 → Franka Teleop Demo")
    parser.add_argument("--zmq-port", type=int, default=5557, help="ZMQ server port")
    parser.add_argument("--zmq-host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--pos-scale", type=float, default=5.0, help="Position scaling factor")
    parser.add_argument("--rot-scale", type=float, default=1.0, help="Rotation scaling factor")
    parser.add_argument("--steps", type=int, default=10000, help="Number of sim steps")
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Path to a pipeline scene snapshot JSON (scene_ep*.json)")
    parser.add_argument("--output-hdf5", type=str, default=None,
                        help="If set, wraps env in DataCollectionWrapper and writes "
                             "the teleop trajectory (actions + states + transitions) "
                             "to this HDF5 file. Use with DataPlaybackWrapper later "
                             "to regenerate obs datasets at any sensor config.")
    parser.add_argument("--only-successes", action="store_true",
                        help="Only save successful episodes when recording to HDF5")
    parser.add_argument("--gripper-threshold", type=float, default=0.5,
                        help="SO-101 gripper value above which the sim gripper opens (0-1)")
    parser.add_argument("--invert-gripper", action="store_true",
                        help="Invert gripper open/close mapping (use if your leader "
                             "reports CLOSED near the top of its calibrated range)")
    parser.add_argument("--debug-gripper", action="store_true",
                        help="Print the raw gripper value each step")
    parser.add_argument("--grasping-mode", choices=["physical", "assisted", "sticky"],
                        default="physical",
                        help="OmniGibson grasping semantics. 'physical' (default) "
                             "= pure Coulomb contact physics. 'assisted' = once "
                             "both fingers contact an object between the AG "
                             "raycast endpoints, OG welds the object to the "
                             "gripper via a force-limited FixedJoint. 'sticky' = "
                             "any finger contact + close command welds. Use "
                             "assisted for teleop demos when slip on thin/flat "
                             "objects is the bottleneck; physical preserves "
                             "real grasp dynamics for downstream training.")
    parser.add_argument("--no-lid-snap", action="store_true",
                        help="Disable the eager lid/cap → container "
                             "snap-attach. By default, when an eligible "
                             "pair is loaded, the lid auto-attaches when "
                             "the lid is touching the container and the "
                             "gripper has released it.")
    args = parser.parse_args()

    cfg = _build_from_snapshot(
        args.snapshot,
        grasping_mode=args.grasping_mode,
    )
    print(f"[Teleop] grasping_mode = {args.grasping_mode}")

    # Create environment
    env = og.Environment(configs=cfg)
    env.reset()

    # Optionally wrap in DataCollectionWrapper to record the teleop trajectory.
    # Observations are NOT recorded here (too heavy); use DataPlaybackWrapper
    # on the output HDF5 to materialize obs datasets afterwards.
    recorder = None
    if args.output_hdf5:
        recorder = DataCollectionWrapper(
            env=env,
            output_path=args.output_hdf5,
            only_successes=args.only_successes,
            flush_every_n_traj=1,
        )
        env = recorder
        print(f"[Teleop] Recording HDF5 to {args.output_hdf5} "
              f"(only_successes={args.only_successes})")

    robot = env.robots[0]

    # Lid-transport assist: discover any (lid|cap, container) pairs in the
    # loaded scene and eagerly snap-attach when the operator places one
    # within range and releases the gripper. No-op when no eligible pairs
    # are loaded — safe to enable globally.
    from maniguard.utils.lid_attach import LidSnapper
    lid_snapper = None if args.no_lid_snap else LidSnapper(env)

    # Position the external cameras (and viewer) at the task's PRESET poses:
    # the shared load-side rule (camera_setup.place_recorded_task_cameras)
    # applies the ``cameras`` recorded in the snapshot's sibling
    # diagnostics.jsonl — the same views datagen/eval/playback use — with the
    # canonical robot-frame recompute as the (warned) fallback for legacy
    # snapshots without recorded poses.
    from maniguard.utils.camera_setup import place_recorded_task_cameras

    _diag_path = os.path.join(os.path.dirname(args.snapshot), "diagnostics.jsonl")
    _cam_diag = _read_first_jsonl(_diag_path) if os.path.isfile(_diag_path) else None
    place_recorded_task_cameras(env, _cam_diag, set_viewer=True)

    # Initialize SO-101 teleop agent
    teleop_cfg = SO101TeleopConfig(
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        position_scale=args.pos_scale,
        rotation_scale=args.rot_scale,
        gripper_threshold=args.gripper_threshold,
        gripper_invert=args.invert_gripper,
        gripper_debug=args.debug_gripper,
    )
    agent = SO101TeleopAgent(config=teleop_cfg)

    # Hotkeys. Q = clean shutdown (needed because Isaac Sim's carb layer
    # installs its own SIGINT handler that bypasses Python try/finally --
    # Ctrl+C leaves the HDF5 as a truncated 96B header). C/R wired only
    # when recording.
    KeyboardEventHandler.initialize()
    exit_flag = {"quit": False}

    def _on_quit():
        exit_flag["quit"] = True
        print("[Teleop] Quit requested -- flushing and shutting down...")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Q, callback_fn=_on_quit,
    )

    success_flag = {"ok": False}
    manual_success_override = {"ok": False}
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
            print("[Teleop] Loaded success checker from diagnostics.")

    if recorder is not None:
        def _on_checkpoint():
            recorder.update_checkpoint()
            print(f"[Teleop] Checkpoint saved "
                  f"(total={len(recorder.checkpoint_states)})")

        def _on_rollback():
            if not recorder.checkpoint_states:
                print("[Teleop] Rollback: no checkpoint yet.")
                return
            recorder.rollback_to_checkpoint()
            print(f"[Teleop] Rolled back to last checkpoint "
                  f"({len(recorder.checkpoint_states)} remain)")

        def _on_success():
            manual_success_override["ok"] = not manual_success_override["ok"]
            state = "FORCE-SUCCESS" if manual_success_override["ok"] else "auto-checker"
            print(f"[Teleop] Success mode -> {state}")

        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C, callback_fn=_on_checkpoint,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R, callback_fn=_on_rollback,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S, callback_fn=_on_success,
        )

    label = args.snapshot
    print("\n" + "=" * 50)
    print(f"SO-101 → Franka Teleop Ready  [{label}]")
    if task_prompt:
        print(f"  TASK   : {task_prompt}")
    if task_target:
        print(f"  TARGET : {task_target}")
    print("Move the SO-101 leader arm to control Franka")
    if recorder is not None:
        print("C = save checkpoint   R = rollback to last checkpoint")
        print("S = toggle manual success override for the current episode")
    print("Q = clean exit (saves HDF5)")
    print("=" * 50 + "\n")

    # One-time probe: where in the action array does the gripper live, and
    # what's the current joint_pos / control_limits the controller will use?
    gripper_ctrl = robot._controllers.get("gripper_0")
    gripper_idx = robot.gripper_action_idx["0"]
    print(f"[Teleop] gripper_ctrl={type(gripper_ctrl).__name__}, "
          f"mode={getattr(gripper_ctrl, '_mode', '?')}, "
          f"inverted={getattr(gripper_ctrl, '_inverted', '?')}, "
          f"input_limits={getattr(gripper_ctrl, '_command_input_limits', '?')}, "
          f"output_limits={getattr(gripper_ctrl, '_command_output_limits', '?')}, "
          f"action_idx={gripper_idx}, dof_idx={getattr(gripper_ctrl, 'dof_idx', '?')}")

    try:
        for step in range(args.steps):
            if exit_flag["quit"]:
                break
            action = agent.get_action(robot)
            if args.debug_gripper and step % 5 == 0:
                gv = action[gripper_idx].tolist() if hasattr(action[gripper_idx], "tolist") else action[gripper_idx]
                print(f"[Action] gripper -> {gv}")
            env.step(action)

            if lid_snapper is not None:
                lid_snapper.try_snap(robot=robot)

            if success_checker is not None:
                auto_success, success_detail = success_checker.check(env)
                if auto_success:
                    success_flag["ok"] = True
                    print(f"[Teleop] Success satisfied: {success_detail}")
                    break
            if manual_success_override["ok"]:
                success_flag["ok"] = True
                break

            if step % 300 == 0 and step > 0:
                status = "connected" if agent.is_connected else "waiting for SO-101 data..."
                print(f"Step {step}/{args.steps} — {status} — success={success_flag['ok']}")

    except KeyboardInterrupt:
        # Usually bypassed by carb's SIGINT handler; kept as a fallback.
        print("\nStopping teleop (SIGINT)...")
    finally:
        if recorder is not None:
            # Force the task success flag so DataCollectionWrapper's
            # should_save_current_episode uses the user's S-key choice.
            # DummyTask's _success is otherwise always False.
            inner_env = recorder.env
            if hasattr(inner_env, "task") and inner_env.task is not None:
                inner_env.task._success = bool(success_flag["ok"])
            print(f"[Teleop] Saving with success={success_flag['ok']} "
                  f"(only_successes={args.only_successes})")
            recorder.save_data()
            print(f"[Teleop] HDF5 written: {args.output_hdf5}")
        agent.stop()
        og.clear()


if __name__ == "__main__":
    main()
