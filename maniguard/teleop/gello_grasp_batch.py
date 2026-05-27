"""
GELLO Leader Arm → Franka Grasp Teleop (batch over CSV)

Mash-up of:
  - maniguard.rl.grasps.render_grasps : per-object outer loop, floor-only env,
    object pin + disable_gravity, .pt grasp dataset save format.
  - maniguard.teleop.gello_franka_teleop : GELLO joint mirroring, calibration
    pose + ramp, SPACE-toggle gripper.

Per object: spawn target → pin to --object-xyz → disable gravity → reset
robot to calibration pose + ramp to GELLO live. Operator drives Franka via
GELLO, presses S to capture each successful grasp, N to advance, R to retry,
K to skip, Q to quit. On N/Q the captured grasps for the current object are
written to ``grasps_{cat}_{model}.pt`` in the format consumed by
``maniguard.rl.grasps.reset.GraspDatasetResetter``.

Resume: rows whose .pt already exists are skipped.

Usage:
    python -m maniguard.teleop.gello_grasp_batch \\
        --csv maniguard/task_generation/utils/franka_graspability.csv \\
        --limit 50 \\
        --output-dir outputs/grasp_datasets/teleop/tensors

Hotkeys (need GUI focus on the OmniGibson viewport):
    SPACE = toggle gripper open/close
    S     = capture current frame as a grasp (append to per-object buffer)
    N     = next object (writes .pt if buffer non-empty)
    R     = retry current object (resets robot + target, clears in-flight traj)
    K     = skip current object (no save)
    Q     = quit (writes .pt for current object if buffer non-empty)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Vulkan ICD must be set BEFORE OmniGibson imports trigger Isaac Sim's
# renderer init. Use setdefault so an explicit shell override still wins.
os.environ.setdefault(
    "VK_ICD_FILENAMES",
    "/usr/share/vulkan/icd.d/nvidia_icd.json",
)

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.constants import LightingMode
from omnigibson.utils.ui_utils import KeyboardEventHandler

# joylo on sys.path (not pip-installed).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_JOYLO = _REPO_ROOT / "behavior-1k" / "joylo"
if str(_JOYLO) not in sys.path:
    sys.path.insert(0, str(_JOYLO))

from gello.robots.dynamixel import DynamixelRobot  # noqa: E402

from maniguard.rl.grasps.collector import (  # noqa: E402
    _mat_to_pose,
    _phase1_step,
    _pose_to_mat,
    _reset_controller_goals,
    save_grasp_dataset,
)
from maniguard.rl.grasps.render_grasps import (  # noqa: E402
    _read_graspable,
    _rpy_deg_to_quat_xyzw,
)
from maniguard.teleop.gello_franka_teleop import (  # noqa: E402
    GELLO_CALIBRATION_FRANKA_POSE,
    GELLO_GRIPPER_CONFIG,
    GELLO_JOINT_IDS,
    GELLO_JOINT_OFFSETS,
    GELLO_JOINT_SIGNS,
    GELLO_PORT,
    GELLO_RAMP_STEPS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="GELLO -> Franka grasp teleop, batched over the survey CSV."
    )
    p.add_argument("--csv", type=Path,
                   default=Path("maniguard/task_generation/utils/franka_graspability.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/grasp_datasets/teleop/tensors"),
                   help="Where ``grasps_{cat}_{model}.pt`` files land.")
    p.add_argument("--limit", type=int, default=50,
                   help="Cap on number of pending objects (0 = no cap).")
    p.add_argument("--exclude-statuses", type=str, default="too_large",
                   help="Comma-separated CSV statuses to skip. Default skips "
                        "only 'too_large' so we attempt the full graspability "
                        "spectrum (graspable + GraspGen failures).")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Optional 'category:model' list overriding the CSV.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-do objects whose .pt already exists.")
    # Scene geometry — mirrors render_grasps so .pt grasps are interchangeable.
    p.add_argument("--object-xyz", type=float, nargs=3, default=[0.55, 0.0, 0.55],
                   help="Where to drop each target. Z should be above the "
                        "table top so the object settles onto the surface.")
    p.add_argument("--target-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="Spawn-orientation override (degrees, intrinsic ZYX).")
    p.add_argument("--franka-xy", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--franka-z", type=float, default=0.72)
    p.add_argument("--table-top-z", type=float, default=0.50,
                   help="World Z of the tabletop surface. The cube tabletop "
                        "is added to the env; targets settle on top under "
                        "gravity, no manual pinning.")
    p.add_argument("--table-size", type=float, nargs=2, default=[0.8, 0.8],
                   metavar=("X", "Y"),
                   help="Tabletop plan size (xy).")
    # GELLO + grasping.
    p.add_argument("--gello-port", type=str, default=GELLO_PORT)
    p.add_argument("--invert-gripper", action="store_true")
    p.add_argument("--start-gripper-open", action="store_true",
                   help="Start each object with the gripper OPEN (default CLOSED).")
    p.add_argument("--grasping-mode", choices=["physical", "assisted", "sticky"],
                   default="assisted",
                   help="OmniGibson grasping semantics. 'assisted' is the "
                        "default since we want AG-fired holds to count.")
    p.add_argument("--gpu-dynamics", action="store_true",
                   help="Enable PhysX GPU dynamics (only needed for fluids).")
    p.add_argument("--debug-ag", action="store_true",
                   help="Set gm.DEBUG=True so manipulation_robot draws small "
                        "green spheres at each AG raycast endpoint (2 per "
                        "finger). Lets you see whether the rays are inside "
                        "the object when the gripper closes. Side effect: "
                        "OG internal log level rises to DEBUG.")
    return p.parse_args()


def _build_env_config(args, franka_base_z: float) -> dict:
    """Floor + tabletop Scene + FrankaPanda (JointController). The
    tabletop is a fixed_base cube; targets are spawned just above its
    surface and settle under gravity (no per-step pin)."""
    table_top_z = float(args.table_top_z)
    table_thickness = 0.05
    table_center_z = table_top_z - table_thickness / 2
    return dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[dict(
            type="FrankaPanda",
            name="agent_0",
            obs_modalities=["rgb"],
            action_type="continuous",
            action_normalize=False,
            grasping_mode=args.grasping_mode,
            self_collisions=True,
            position=[args.franka_xy[0], args.franka_xy[1], franka_base_z],
            orientation=[0.0, 0.0, 0.0, 1.0],
            controller_config={
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
            },
        )],
        objects=[dict(
            type="PrimitiveObject",
            name="grasp_table",
            primitive_type="Cube",
            rgba=[0.55, 0.4, 0.3, 1.0],
            size=1.0,
            scale=[args.table_size[0], args.table_size[1], table_thickness],
            position=[args.object_xyz[0], 0.0, table_center_z],
            fixed_base=True,
        )],
        task=dict(type="DummyTask"),
    )


def _capture_grasp(robot, obj, arm_control_idx, gripper_control_idx,
                   approach_traj):
    """Build the saved-grasp dict for the current frame. Mirrors the tail of
    ``maniguard.rl.grasps.collector.run_grasp_attempt`` so the resulting .pt
    is bit-compatible with what the survey pipeline writes."""
    arm = robot.default_arm
    eef_pos_t, eef_quat_t = robot.eef_links[arm].get_position_orientation()
    tgt_pos_t, tgt_quat_t = obj.get_position_orientation()
    T_eef_w = _pose_to_mat(eef_pos_t.cpu().numpy(), eef_quat_t.cpu().numpy())
    T_tgt_w = _pose_to_mat(tgt_pos_t.cpu().numpy(), tgt_quat_t.cpu().numpy())
    T_eef_local = np.linalg.inv(T_tgt_w) @ T_eef_w
    rel_pos, rel_quat = _mat_to_pose(T_eef_local)

    full_q = robot.get_joint_positions().cpu().numpy()
    arm_idx_np = arm_control_idx.cpu().numpy()
    grip_idx_np = gripper_control_idx.cpu().numpy()
    traj_np = (np.stack(approach_traj, axis=0).astype(np.float32)
               if approach_traj else np.zeros((0, len(arm_idx_np)),
                                              dtype=np.float32))
    return {
        "rel_position": rel_pos.astype(np.float32),
        "rel_orientation_xyzw": rel_quat.astype(np.float32),
        "gripper_qpos": full_q[grip_idx_np].astype(np.float32),
        "arm_joint_pos": full_q[arm_idx_np].astype(np.float32),
        "approach_traj": traj_np,
    }


def _reset_for_object(env, robot, obj, init_pos, init_quat,
                      initial_joint_pos):
    """Reset robot + target to the per-object starting condition.

    Drops the target at ``init_pos`` above the tabletop and lets gravity
    settle it. Robot returns to the calibration anchor pose."""
    import omnigibson as og

    arm = robot.default_arm
    try:
        robot.release_grasp_immediately(arm)
    except Exception:  # noqa: BLE001
        pass
    robot.set_joint_positions(initial_joint_pos)
    obj.set_position_orientation(position=init_pos, orientation=init_quat)
    obj.root_link.set_linear_velocity(th.zeros(3))
    obj.root_link.set_angular_velocity(th.zeros(3))
    _reset_controller_goals(robot)
    # Settle: gravity pulls the target onto the table, robot stays at home.
    for _ in range(20):
        og.sim.step()


def _teleop_object(env, robot, leader, obj, args, state, hotkeys,
                   arm_control_idx, gripper_control_idx,
                   arm_action_idx, gripper_action_idx, action_dim,
                   open_value, close_value, init_pos, init_quat,
                   initial_joint_pos):
    """Drive one object's teleop session. Returns ('next' | 'skip' | 'quit',
    held_grasps_list).

    Loop terminates on N (next, save), K (skip, no save), or Q (quit, save).
    R triggers an in-place reset (robot + target back to initial, traj cleared).
    """
    from omnigibson.controllers.controller_base import IsGraspingState

    arm = robot.default_arm
    held: list[dict] = []
    approach_traj: list[np.ndarray] = []
    # `released` flips True the first step is_grasping returns TRUE: pin
    # disengages and gravity turns back on so the operator can lift+verify.
    # Reset to False on every attempt (object entry, R-retry).
    released = False
    # AG-fail diagnostic: every DBG_AG_EVERY steps while gripper is closed
    # and no AG yet, print which stage rejected the grasp this frame.
    DBG_AG_EVERY = 30  # ~1 Hz at 30 Hz action frequency
    step_count = 0
    finger_links = robot.finger_links[arm]
    finger_prim_paths = {l.prim_path for l in finger_links}
    obj_prim = obj.prim_path

    # Per-object reset + ramp source capture (so the gripper-cam isn't jolted
    # when we hand the arm over to GELLO at a fresh object).
    _reset_for_object(env, robot, obj, init_pos, init_quat, initial_joint_pos)
    ramp_source = robot.get_joint_positions().cpu().numpy()[
        arm_control_idx.cpu().numpy()
    ].astype(np.float32)
    ramp_step = 0

    print("  Drive GELLO to grasp. Hotkeys: SPACE=gripper  S=save  "
          "N=next  R=retry  K=skip  Q=quit", flush=True)

    while True:
        if hotkeys["quit"]:
            return "quit", held
        if hotkeys["next"]:
            hotkeys["next"] = False
            return "next", held
        if hotkeys["skip"]:
            hotkeys["skip"] = False
            return "skip", held
        if hotkeys["retry"]:
            hotkeys["retry"] = False
            print(f"  retry: clearing in-flight traj "
                  f"({len(approach_traj)} samples), kept {len(held)} "
                  f"saved grasp(s).", flush=True)
            _reset_for_object(env, robot, obj, init_pos, init_quat,
                              initial_joint_pos)
            ramp_source = robot.get_joint_positions().cpu().numpy()[
                arm_control_idx.cpu().numpy()
            ].astype(np.float32)
            ramp_step = 0
            approach_traj.clear()
            released = False  # _reset_for_object re-disables gravity
            continue
        if hotkeys["save"]:
            hotkeys["save"] = False
            try:
                grasp = _capture_grasp(robot, obj, arm_control_idx,
                                       gripper_control_idx, approach_traj)
                held.append(grasp)
                print(f"  saved grasp #{len(held)} "
                      f"(traj_len={len(grasp['approach_traj'])})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  save failed: {exc}", flush=True)

        # GELLO → Franka joint command.
        target = np.asarray(leader.get_joint_state(), dtype=np.float32)[:7]
        if ramp_step < GELLO_RAMP_STEPS:
            alpha = (ramp_step + 1) / GELLO_RAMP_STEPS
            arm_cmd_np = (1.0 - alpha) * ramp_source + alpha * target
        else:
            arm_cmd_np = target
        ramp_step += 1

        action = th.zeros(action_dim, dtype=th.float32)
        action[arm_action_idx] = th.tensor(arm_cmd_np, dtype=th.float32)
        action[gripper_action_idx] = th.tensor(
            [open_value if state["gripper_open"] else close_value],
            dtype=th.float32,
        )
        env.step(action)
        step_count += 1

        # No pinning: the object sits on the tabletop under gravity. We
        # still print a one-shot "AG engaged" line so the operator gets
        # confirmation, but otherwise the simulator handles physics.
        if not released and \
                robot.is_grasping(arm, obj) == IsGraspingState.TRUE:
            released = True
            print("  AG engaged → lift to verify (S to save grasp).",
                  flush=True)

        # Diagnose why AG isn't firing — only when gripper is closing and
        # roughly 1 Hz so we don't spam. Walks the same gates
        # _calculate_in_hand_object_rigid uses (manipulation_robot.py:1109+).
        if (not released and not state["gripper_open"]
                and step_count % DBG_AG_EVERY == 0):
            try:
                contacts_set, contact_links = \
                    robot._find_gripper_contacts(arm=arm)
                rc_set = robot._find_gripper_raycast_collisions(arm=arm)
                inter = contacts_set & rc_set
                target_in_contacts = any(
                    obj_prim in p for p in contacts_set)
                target_in_rc = any(obj_prim in p for p in rc_set)
                target_in_inter = any(obj_prim in p for p in inter)
                # Per-finger touch on target (assisted needs ≥2).
                # contact_links: dict[str prim_path -> set[str finger_prim_paths]]
                n_target_fingers = 0
                for fp in finger_prim_paths:
                    for tp, link_paths in contact_links.items():
                        if obj_prim in tp and fp in link_paths:
                            n_target_fingers += 1
                            break
                print(f"  [ag-debug step={step_count}] "
                      f"contacts={len(contacts_set)} "
                      f"(target_hit={target_in_contacts}) | "
                      f"raycast={len(rc_set)} "
                      f"(target_hit={target_in_rc}) | "
                      f"∩={len(inter)} "
                      f"(target_hit={target_in_inter}) | "
                      f"target_fingers_touching={n_target_fingers}/2",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ag-debug] introspect failed: {exc}",
                      flush=True)

        # Record arm joint pos for approach_traj. Reading after step() so the
        # waypoint reflects the actual physics state, not the command.
        approach_traj.append(
            robot.get_joint_positions().cpu().numpy()[
                arm_control_idx.cpu().numpy()
            ].astype(np.float32)
        )


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.targets:
        rows = []
        for s in args.targets:
            if ":" not in s:
                raise SystemExit(f"bad --targets {s!r}; expected category:model")
            c, m = s.split(":", 1)
            rows.append((c.strip(), m.strip()))
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} explicit targets "
              f"(--targets), ignoring CSV.", flush=True)
    else:
        csv_path = args.csv.resolve()
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        excl = tuple(s.strip() for s in args.exclude_statuses.split(",") if s.strip())
        rows = list(_read_graspable(csv_path, exclude_statuses=excl))
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} rows in CSV "
              f"(excluded statuses: {excl}).", flush=True)

    if not args.overwrite:
        before = len(rows)
        rows = [(c, m) for (c, m) in rows
                if not (out_dir / f"grasps_{c}_{m}.pt").exists()]
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} pending "
              f"({before - len(rows)} already done in {out_dir}).", flush=True)

    if args.limit > 0:
        rows = rows[:args.limit]
        print(f"  capped at --limit {args.limit}", flush=True)
    if not rows:
        print("Nothing to teleop.", flush=True)
        return

    # Apply global macros BEFORE og.Environment — simulator.py reads
    # gm.USE_GPU_DYNAMICS during _init_physx and won't pick up later changes.
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False

    # Drop AG grasp window from default 1/30s (10 action steps of consistent
    # contact, ~333ms) to one physics step. Teleop contact tends to flicker
    # under operator wobble — the default window means AG rarely fires
    # without a perfectly steady hand. 2 action steps (~67ms) of contact
    # is enough for AG to commit. Cost: brushing past an object briefly
    # can fire AG; for teleop verification that's fine.
    from omnigibson.robots.manipulation_robot import m as _ag_macros
    _ag_macros.GRASP_WINDOW = 1.0 / 300.0
    _ag_macros.RELEASE_WINDOW = 1.0 / 300.0
    if args.gpu_dynamics:
        gm.USE_GPU_DYNAMICS = True
        print("[GraspBatch] gm.USE_GPU_DYNAMICS = True", flush=True)
    if args.debug_ag:
        # Read once at robot _post_load (manipulation_robot.py:427) — must
        # be set BEFORE og.Environment(configs=...) constructs the robot.
        gm.DEBUG = True
        print("[GraspBatch] gm.DEBUG = True (AG raycast endpoints visualized "
              "as green spheres; expect verbose OG logs)", flush=True)
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    # Connect GELLO before booting OG so DynamixelRobot failures (port busy /
    # no power / wrong cable) surface fast — before the 30-90 s OG startup.
    print(f"[GraspBatch] Connecting to GELLO at {args.gello_port}", flush=True)
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
    print(f"[GraspBatch] GELLO connected. DOFs={n_dofs}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG ...", flush=True)
    env = og.Environment(
        configs=_build_env_config(args, franka_base_z=float(args.franka_z))
    )
    env.reset()

    og.sim.add_skybox()
    try:
        og.sim._skybox.intensity = 12000
    except Exception as exc:  # noqa: BLE001
        print(f"[GraspBatch] Could not bump skybox intensity ({exc}).", flush=True)
    og.sim.set_lighting_mode(LightingMode.CAMERA)

    robot = env.robots[0]
    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    arm_action_idx = robot.controller_action_idx["arm_0"]
    gripper_action_idx = robot.gripper_action_idx["0"]
    action_dim = robot.action_dim
    open_value = -1.0 if args.invert_gripper else 1.0
    close_value = 1.0 if args.invert_gripper else -1.0

    # Seed Franka at the deterministic GELLO calibration pose so we can ramp
    # from a known anchor regardless of the leader's startup position.
    initial_arm = th.tensor(GELLO_CALIBRATION_FRANKA_POSE, dtype=th.float32)
    full_q = robot.get_joint_positions().clone()
    full_q[arm_control_idx] = initial_arm
    robot.set_joint_positions(full_q)
    initial_joint_pos = robot.get_joint_positions().clone()

    # ----- Hotkeys -----
    KeyboardEventHandler.initialize()
    state = {"gripper_open": bool(args.start_gripper_open)}
    hotkeys = {"quit": False, "next": False, "retry": False,
               "skip": False, "save": False}

    def _on_space():
        state["gripper_open"] = not state["gripper_open"]
        print(f"[GraspBatch] Gripper -> "
              f"{'OPEN' if state['gripper_open'] else 'CLOSE'}", flush=True)

    def _on_save():
        hotkeys["save"] = True

    def _on_next():
        hotkeys["next"] = True

    def _on_retry():
        hotkeys["retry"] = True

    def _on_skip():
        hotkeys["skip"] = True

    def _on_quit():
        hotkeys["quit"] = True
        print("[GraspBatch] Quit requested.", flush=True)

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.SPACE, callback_fn=_on_space)
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.S, callback_fn=_on_save)
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.N, callback_fn=_on_next)
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.R, callback_fn=_on_retry)
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.K, callback_fn=_on_skip)
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Q, callback_fn=_on_quit)

    print("\n" + "=" * 56)
    print(f"GELLO Grasp Batch  [{len(rows)} objects, out -> {out_dir}]")
    print("SPACE=gripper  S=save  N=next  R=retry  K=skip  Q=quit")
    print("=" * 56 + "\n")

    init_pos = th.tensor(args.object_xyz, dtype=th.float32)
    init_quat = th.tensor(_rpy_deg_to_quat_xyzw(args.target_rpy),
                          dtype=th.float32)

    from omnigibson.objects import DatasetObject

    n_done = n_saved = n_skipped = 0
    try:
        for idx, (cat, mdl) in enumerate(rows):
            if hotkeys["quit"]:
                break
            stem = f"{cat}_{mdl}"
            pt_path = out_dir / f"grasps_{stem}.pt"
            print(f"\n[{time.strftime('%H:%M:%S')}] "
                  f"({idx+1}/{len(rows)}) {cat}/{mdl}", flush=True)

            name = f"teleop_target_{cat}_{mdl}"
            try:
                obj = DatasetObject(name=name, category=cat, model=mdl)
                env.scene.add_object(obj)
            except Exception as exc:  # noqa: BLE001
                # Match render_grasps' defensive cleanup of half-spawned prims.
                try:
                    existing = env.scene.object_registry("name", name)
                    if existing is not None:
                        env.scene.remove_object(existing)
                except Exception:  # noqa: BLE001
                    pass
                print(f"  spawn failed: {type(exc).__name__}: {str(exc)[:160]}",
                      flush=True)
                continue

            try:
                action, held = _teleop_object(
                    env, robot, leader, obj, args, state, hotkeys,
                    arm_control_idx, gripper_control_idx,
                    arm_action_idx, gripper_action_idx, action_dim,
                    open_value, close_value, init_pos, init_quat,
                    initial_joint_pos,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  ! teleop crashed: {exc}", flush=True)
                # Same OG bug render_grasps documents: closing the viewport,
                # an unlucky add/remove cycle, or a bad apply_action can null
                # the articulation_view. Once dead, every subsequent
                # get_joint_positions() returns None → infinite cascade. Bail
                # out so the operator can rerun (resume skips finished .pt's).
                msg = str(exc)
                if (("NoneType" in msg and "view" in msg)
                        or "articulation_view" in msg.lower()):
                    print(f"[{time.strftime('%H:%M:%S')}] FATAL: OG "
                          f"articulation state corrupted — exiting. "
                          f"Re-run to resume from already-saved .pt files.",
                          flush=True)
                    sys.stdout.flush()
                    sys.exit(2)
                action, held = "skip", []

            # Persist before cleanup so a remove-time failure can't lose the
            # operator's saved grasps.
            if held and action in ("next", "quit"):
                try:
                    save_grasp_dataset(held, pt_path, target_name=stem)
                    n_saved += 1
                    print(f"  wrote {pt_path.name} (N={len(held)})",
                          flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  save failed: {exc}", flush=True)
            elif action == "skip":
                n_skipped += 1
                print(f"  skipped (no .pt written)", flush=True)
            elif not held:
                print(f"  no grasps captured; nothing to save", flush=True)

            n_done += 1

            # Cleanup — match render_grasps:497-518 verbatim. The
            # release_action loop matters: if we remove the prim while the
            # gripper is still applying close torque on it, articulation_view
            # gets confused and nulls out a few cycles later.
            try:
                robot.release_grasp_immediately(arm)
            except Exception:  # noqa: BLE001
                pass
            try:
                from maniguard.rl.grasps.collector import _build_action
                zero_arm = th.zeros(len(arm_action_idx), dtype=th.float32)
                release_action = _build_action(robot, zero_arm, gripper_cmd=+1.0)
                for _ in range(6):
                    robot.apply_action(release_action)
                    og.sim.step()
            except Exception:  # noqa: BLE001
                pass
            try:
                env.scene.remove_object(obj)
            except Exception:  # noqa: BLE001
                pass
            try:
                robot.set_joint_positions(initial_joint_pos)
                _reset_controller_goals(robot)
                og.sim.step()
            except Exception:  # noqa: BLE001
                pass

            if action == "quit":
                break
    finally:
        print(f"\n[{time.strftime('%H:%M:%S')}] DONE. "
              f"done={n_done} saved={n_saved} skipped={n_skipped} "
              f"dir={out_dir}", flush=True)
        sys.stdout.flush()
        og.clear()


if __name__ == "__main__":
    main()
