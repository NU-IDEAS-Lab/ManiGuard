"""
SO-101 Leader Arm → Franka Teleop

Teleoperates a FrankaPanda in OmniGibson using an SO-101 leader arm.
The SO-101 end-effector deltas are mapped to Franka IK targets. Loads a
pipeline-generated scene snapshot and optionally records a trajectory via
DataCollectionWrapper.

Usage:
    python -m sentinel.teleop.so101_franka_teleop \
        --snapshot outputs/pipeline_runs/<run>/scene_ep1.json

Prerequisites:
    Terminal 1 (lerobot venv):
        python teleop_bridge/so101_server.py --mock          # mock mode
        python teleop_bridge/so101_server.py --port /dev/ttyACM0  # real hardware

    Terminal 2 (behavior conda):
        python -m sentinel.teleop.so101_franka_teleop \
            --snapshot <path>
"""

import argparse
import json
import os
import sys

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs import DataCollectionWrapper
from sentinel.teleop.so101_teleop import SO101TeleopAgent, SO101TeleopConfig
from omnigibson.utils.ui_utils import KeyboardEventHandler


# ---------------------------------------------------------------------------
# Long-finger asset patch
# ---------------------------------------------------------------------------

def _install_longfinger_franka_patch():
    """Repoint FrankaPanda's asset paths to the long-finger bundle.

    The long-finger bundle lives next to stock panda under
    ``omnigibson-robot-assets/models/franka/franka_panda_longfinger/`` and
    keeps panda's link / joint / eef_link contract — only finger
    geometry differs. We override the path properties on the FrankaPanda
    class without touching ``model_name`` so the curobo / urdf asserts
    in upstream ``omnigibson.robots.franka`` keep passing.

    Scope: this patch is installed only by ``so101_franka_teleop.main()``
    (i.e. ``scripts/run_teleop_batch.sh``). Other entry points (task
    generation, RL, eval) don't import this module and aren't affected.
    """
    from omnigibson.robots.franka import FrankaPanda
    from omnigibson.utils.asset_utils import get_dataset_path

    if getattr(FrankaPanda, "_sentinel_longfinger_patched", False):
        return

    root = os.path.join(
        get_dataset_path("omnigibson-robot-assets"),
        "models/franka/franka_panda_longfinger",
    )
    paths = {
        "usd":    os.path.join(root, "usd",    "franka_panda_longfinger.usda"),
        "urdf":   os.path.join(root, "urdf",   "franka_panda_longfinger.urdf"),
        "curobo": os.path.join(root, "curobo", "franka_panda_longfinger_description_curobo_default.yaml"),
    }
    for kind, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"[Teleop] long-finger {kind} asset missing: {p}\n"
                f"Expected the long-finger bundle under {root}, or pass "
                f"--stock-franka to fall back to stock BEHAVIOR FrankaPanda."
            )

    # Only swap for the default 'gripper' end_effector. allegro/leap/inspire
    # variants keep upstream behavior (model_name differs there).
    def _make_swap(orig_prop, swap_path):
        def _getter(self):
            if self._model_name == "franka_panda":
                return swap_path
            return orig_prop.fget(self)
        return property(_getter)

    FrankaPanda.usd_path    = _make_swap(FrankaPanda.usd_path,    paths["usd"])
    FrankaPanda.urdf_path   = _make_swap(FrankaPanda.urdf_path,   paths["urdf"])
    FrankaPanda.curobo_path = _make_swap(FrankaPanda.curobo_path, paths["curobo"])

    # Delegate AG grasping points to ManipulationRobot's auto-inference so
    # raycast endpoints land at the actual long-finger fingertip instead of
    # the stock 0.045 m offset (which would be ~28% along the long finger,
    # well below the pinch zone). Mirrors FrankaMounted's strategy
    # (franka_mounted.py:43-48) which uses the same long fingers.
    def _make_ag_swap(orig_prop):
        def _getter(self):
            if self._model_name == "franka_panda":
                return None  # -> base class auto-inference via _infer_finger_properties
            return orig_prop.fget(self)
        return property(_getter)

    FrankaPanda._assisted_grasp_start_points = _make_ag_swap(FrankaPanda._assisted_grasp_start_points)
    FrankaPanda._assisted_grasp_end_points   = _make_ag_swap(FrankaPanda._assisted_grasp_end_points)

    FrankaPanda._sentinel_longfinger_patched = True

    print("[Teleop] FrankaPanda assets repointed to long-finger bundle:")
    print(f"           USD:    {paths['usd']}")
    print(f"           URDF:   {paths['urdf']}")
    print(f"           cuRobo: {paths['curobo']}")
    print("[Teleop] AG grasping points -> auto-inferred from long-finger geometry.")


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

from sentinel.utils.camera_setup import build_external_camera_configs

_EXTERNAL_CAMERAS = build_external_camera_configs()


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
    parser.add_argument("--stock-franka", action="store_true",
                        help="Fall back to BEHAVIOR-1K's stock FrankaPanda asset. "
                             "Default loads the long-finger variant from "
                             "omnigibson-robot-assets/models/franka/franka_panda_longfinger/.")
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
    args = parser.parse_args()

    if not args.stock_franka:
        _install_longfinger_franka_patch()
    else:
        print("[Teleop] --stock-franka set; using stock BEHAVIOR FrankaPanda assets.")

    cfg = _build_from_snapshot(
        args.snapshot,
        grasping_mode=args.grasping_mode,
    )
    print(f"[Teleop] grasping_mode = {args.grasping_mode}")

    # Create environment
    env = og.Environment(configs=cfg)
    env.reset()

    # Lighting:
    #   - Skybox dome (intensity bumped to ~12000) lights every camera in
    #     the stage including the external_sensors → recorded HDF5 frames
    #     are bright enough.
    #   - Viewport switched to LightingMode.CAMERA so the operator's view
    #     always feels lit-from-where-they're-looking (the same effect as
    #     toggling "Camera Light" in the Kit viewport top bar).
    #   add_skybox() is idempotent for the LightObject creation; we then
    #   try to bump its intensity in case OmniGibson's default 2500 isn't
    #   strong enough for furnished BEHAVIOR rooms (silently no-op if the
    #   intensity setter isn't exposed).
    from omnigibson.utils.constants import LightingMode
    og.sim.add_skybox()
    try:
        og.sim._skybox.intensity = 12000
    except Exception as _light_exc:
        print(f"[Teleop] Could not bump skybox intensity ({_light_exc}); "
              f"using default 2500.")
    og.sim.set_lighting_mode(LightingMode.CAMERA)
    print("[Teleop] Skybox dome added; viewport lighting = CAMERA "
          "(camera-attached light, like the Kit viewport toggle).")

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

    # Position the 3 external cameras (and viewer = opposite side). Two
    # scene conventions are handled:
    #   1. Pipeline-synthesized scenes (clutter / stack / mug_into_bowl …)
    #      contain an object literally named "support_surface" → use the
    #      original AABB-derived placement which is tuned to the table edge.
    #   2. HF-shipped BEHAVIOR scenes (7_fam_*, full furnished rooms) have
    #      no support_surface AND object-derived placement risks putting a
    #      camera inside a wall (e.g. liquid_transport rooms with 220
    #      objects). Instead, derive cameras in the robot's own local frame:
    #      Franka's body +X axis is its "forward" (workspace direction), so
    #      placing cameras behind / left / right of the robot lands them in
    #      the room's open area — the robot was placed by the pipeline
    #      specifically to face an unobstructed workspace.
    import numpy as _np
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
        print(f"[Teleop] Camera mode = support_surface; target={target_obj.name}")
    else:
        # Robot-frame fallback.
        import omnigibson.utils.transform_utils as _T
        from sentinel.task_generation.utils.video import setup_cameras

        rp_t, rq_t = robot.get_position_orientation()
        rp = _np.asarray(rp_t.cpu().numpy() if hasattr(rp_t, "cpu") else rp_t,
                         dtype=_np.float32)
        rmat_t = _T.quat2mat(rq_t)
        rmat = _np.asarray(rmat_t.cpu().numpy() if hasattr(rmat_t, "cpu") else rmat_t,
                           dtype=_np.float32)
        forward = rmat[:, 0].copy()
        forward[2] = 0.0
        n = float(_np.linalg.norm(forward))
        forward = forward / n if n > 1e-6 else _np.array([1.0, 0.0, 0.0], dtype=_np.float32)
        # Robot's left in world = +Z × forward (right-hand rule with +Z up).
        left = _np.cross(_np.array([0.0, 0.0, 1.0], dtype=_np.float32), forward)

        cam_height_off = 0.9        # meters above robot base
        back_off = 1.2              # how far behind the robot the opp cam sits
        side_off = 1.0              # left/right distance
        side_forward_off = 0.2      # nudge side cams slightly toward workspace

        workspace = rp + forward * 0.45 + _np.array([0, 0, 0.05], dtype=_np.float32)

        opp_eye = rp - forward * back_off + _np.array([0, 0, cam_height_off], dtype=_np.float32)
        left_eye = rp + left * side_off + forward * side_forward_off \
                      + _np.array([0, 0, cam_height_off], dtype=_np.float32)
        right_eye = rp - left * side_off + forward * side_forward_off \
                       + _np.array([0, 0, cam_height_off], dtype=_np.float32)

        views = [
            {"label": "opposite_side_front", "eye": opp_eye.tolist(),  "lookat": workspace.tolist()},
            {"label": "left_overview",       "eye": left_eye.tolist(), "lookat": workspace.tolist()},
            {"label": "right_overview",      "eye": right_eye.tolist(),"lookat": workspace.tolist()},
        ]
        setup_cameras(env, views)
        print(f"[Teleop] Camera mode = robot-frame (no support_surface); "
              f"robot_pos=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}), "
              f"forward=({forward[0]:.2f},{forward[1]:.2f})")

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
            success_flag["ok"] = not success_flag["ok"]
            state = "SUCCESS" if success_flag["ok"] else "not-success"
            print(f"[Teleop] Episode marked {state}")

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
    print("Move the SO-101 leader arm to control Franka")
    if recorder is not None:
        print("C = save checkpoint   R = rollback to last checkpoint")
        print("S = toggle success flag for the current episode")
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

            if step % 300 == 0 and step > 0:
                status = "connected" if agent.is_connected else "waiting for SO-101 data..."
                print(f"Step {step}/{args.steps} — {status}")

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
