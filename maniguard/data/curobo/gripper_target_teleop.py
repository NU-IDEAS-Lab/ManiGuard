"""Click-to-target gripper teleop with cuRobo motion planning.

Spawns 4 translucent boxes that visually approximate the Franka panda
gripper (palm + 2 fingers + AG-zone swept volume) attached to a small
"anchor" cube. You drag the anchor in the viewport via Isaac's gizmo
(W/E/R), watch the ghost gripper preview the target pose, then press
Enter — cuRobo motion-plans from the current joint state to the
anchor's pose and the real Franka follows. Space toggles the gripper.
Q quits.

This is the manual/iterative alternative to the OBB grasp sampler:
when the sampler can't find a good candidate, you eyeball one yourself
and let cuRobo solve the IK + path.

Usage::

    DISPLAY=:1 conda run -n behavior --no-capture-output \\
      python -m maniguard.data.curobo.gripper_target_teleop \\
        --task-dir datasets/6fam-base-20260513/lid_transport/task_0003/base \\
        --episode 1 --lid-at-edge 0.6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


# Franka panda gripper geometry (copied from maniguard/rl/grasps/obb_sampler.py).
# Coordinate frame: perp = X, closing = Y, approach = Z. Origin = eef_link.
_FRANKA_MAX_OPENING        = 0.070
_FRANKA_FINGER_LEN         = 0.145
_FRANKA_FINGER_BREAD       = 0.038
_FRANKA_FINGER_THICK       = 0.040
_FRANKA_EEF_TO_TIP         = 0.098
_FRANKA_PALM_HALF_LEN      = 0.030
_FRANKA_PALM_HALF_WIDTH    = 0.046
_FRANKA_PALM_HALF_BREAD    = 0.032
_FRANKA_AG_Z_FROM_EEF_LOW  = 0.045 - 0.0466
_FRANKA_AG_Z_FROM_EEF_HIGH = 0.140 - 0.0466


def _plan_to_eef_target(motion_gen, robot, eef_link_name,
                         eef_goal_pos, eef_goal_quat, arm_control_idx,
                         timeout: float):
    """Plan a single cuRobo trajectory to an eef goal pose, returning a
    (T, n_arm_dof) torch tensor of arm joint waypoints, or None on failure.

    Bypasses OG's path_to_joint_trajectory() which crashes with
    ``ValueError: 'panda_finger_joint1' is not in list`` when cuRobo
    returns a trajectory whose joint_names doesn't include the gripper
    finger joints. Here we read joint_state.position directly and index
    only the arm joints.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    emb = CuRoboEmbodimentSelection.DEFAULT
    bs = motion_gen.batch_size
    target_pos = {eef_link_name: th.stack([eef_goal_pos] * bs)}
    target_quat = {eef_link_name: th.stack([eef_goal_quat] * bs)}
    # Use return_full_result=True so we can salvage trajectories that
    # cuRobo's trajopt marks as `success=False` but actually converged
    # near the goal (pos_err≈0, rot_err≈0). The OG wrapper's success
    # tensor follows trajopt's strict convergence criterion, but
    # `result.interpolated_plan` is populated whenever cuRobo produced
    # *any* feasible interpolated path. For interactive teleop, a
    # slightly-sub-optimal-but-feasible trajectory is fine.
    full = motion_gen.compute_trajectories(
        target_pos=target_pos, target_quat=target_quat,
        initial_joint_pos=None, is_local=False,
        max_attempts=12, timeout=timeout,
        ik_fail_return=5, enable_finetune_trajopt=True,
        finetune_attempts=2, return_full_result=True,
        success_ratio=1.0 / bs,
        attached_obj=None, attached_obj_scale=None,
        motion_constraint=None, skip_obstacle_update=True,
        ik_only=False, ik_world_collision_check=True,
        emb_sel=emb,
    )

    def _flt_at(e, j):
        """Pull scalar at batch index j from a per-batch tensor (or
        accept a plain scalar)."""
        if e is None:
            return None
        try:
            return float(e[j])
        except Exception:
            try:
                return float(e)
            except Exception:
                return None

    POS_TOL = 0.02   # 2 cm — generous for click-to-target teleop
    ROT_TOL = 0.10   # ~5.7° rotation error

    js = None
    chosen = None
    for i, r in enumerate(full):
        status = getattr(r, "status", "?")
        try:
            paths = r.get_paths()  # list of JointState, length = bs
        except Exception as exc:
            print(f"[plan] iter#{i}: get_paths() raised "
                  f"{type(exc).__name__}: {exc}", flush=True)
            paths = []
        succ_t = getattr(r, "success", None)
        for j in range(len(paths)):
            is_succ = False
            if succ_t is not None:
                try:
                    is_succ = bool(succ_t[j].item())
                except Exception:
                    try:
                        is_succ = bool(succ_t[j])
                    except Exception:
                        is_succ = False
            pos_err = _flt_at(getattr(r, "position_error", None), j)
            rot_err = _flt_at(getattr(r, "rotation_error", None), j)
            path_js = paths[j]
            if path_js is None:
                print(f"[plan]  iter#{i} batch#{j}: path is None "
                      f"(status={status!r})", flush=True)
                continue
            # Accept if cuRobo flagged success, OR if it produced a
            # plan that lands inside tolerance even if trajopt's
            # internal "success" criterion was strict.
            accepted = is_succ or (
                (pos_err is None or pos_err < POS_TOL)
                and (rot_err is None or rot_err < ROT_TOL)
            )
            tag = "accept" if accepted else "reject"
            print(f"[plan]  iter#{i} batch#{j}: {tag} cuRobo_success="
                  f"{is_succ} status={status!r} pos_err={pos_err} "
                  f"rot_err={rot_err}", flush=True)
            if accepted:
                js = path_js
                chosen = (i, j)
                break
        if js is not None:
            break
    if js is None:
        print(f"[plan] cuRobo compute_trajectories: no usable path "
              f"across {len(full)} iter(s)", flush=True)
        return None
    print(f"[plan] chosen path: iter#{chosen[0]} batch#{chosen[1]}",
          flush=True)
    # js.position is (T, n_dof_planned). js.joint_names lists the dofs.
    # We want the arm dofs only. Map by name → index in js.joint_names,
    # then by arm_control_idx-from-robot via robot.joint_names ordering.
    pos = js.position  # (T, k)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    js_names = list(js.joint_names)
    # OG's FrankaPanda exposes arm joint names as `arm_joint_names[arm]` —
    # there's no top-level `joint_names`. Default arm = "0" or "panda_arm";
    # accept either by querying via robot.default_arm.
    arm_full_names = list(robot.arm_joint_names[robot.default_arm])
    try:
        arm_cols = [js_names.index(n) for n in arm_full_names]
    except ValueError as exc:
        print(f"[plan] joint-name mismatch: {exc}; js_names={js_names}",
              flush=True)
        return None
    arm_traj = pos[:, arm_cols].cpu().contiguous()
    print(f"[plan] cuRobo OK: {len(arm_traj)} waypoints, "
          f"dof_planned={len(js_names)}", flush=True)
    return arm_traj


def _ghost_specs():
    """Return list of (name, local_pos_xyz, local_half_extents_xyz, rgba).

    All offsets / extents in eef_link frame; rgba alpha tuned per box so
    the swept volume + fingers stay visible against the lid mesh.
    """
    fb = _FRANKA_FINGER_BREAD * 0.5
    ft = _FRANKA_FINGER_THICK * 0.5
    fl = _FRANKA_FINGER_LEN * 0.5
    open_half = _FRANKA_MAX_OPENING * 0.5
    # Finger center along closing axis: open_half + ft (slab thickness/2).
    finger_y = open_half + ft
    # Finger center along approach: midpoint of [eef_to_tip - FL, eef_to_tip].
    finger_z = _FRANKA_EEF_TO_TIP - fl
    palm_y_extent = _FRANKA_PALM_HALF_WIDTH
    palm_x_extent = _FRANKA_PALM_HALF_BREAD
    palm_z_extent = _FRANKA_PALM_HALF_LEN
    palm_z_center = -palm_z_extent  # palm sits BEHIND eef_link along -approach
    # Swept-volume box: between fingers, in AG raycast band.
    sw_z_low, sw_z_high = _FRANKA_AG_Z_FROM_EEF_LOW, _FRANKA_AG_Z_FROM_EEF_HIGH
    sw_z_center = 0.5 * (sw_z_low + sw_z_high)
    sw_z_extent = 0.5 * (sw_z_high - sw_z_low)
    sw_x_extent = fb            # same perp span as finger
    sw_y_extent = open_half     # spans the corridor between finger inner faces
    # Opaque (alpha=1.0): OG's translucency rendering didn't reliably show
    # alpha < 1 in earlier marker tests; fully opaque keeps the gripper
    # ghost clearly visible. Distinct colors per part = readable preview.
    return [
        ("ghost_palm",
         (0.0, 0.0, palm_z_center),
         (palm_x_extent, palm_y_extent, palm_z_extent),
         (0.30, 0.30, 0.30, 1.0)),
        ("ghost_lfinger",
         (0.0, -finger_y, finger_z),
         (fb, ft, fl),
         (0.20, 0.85, 0.20, 1.0)),
        ("ghost_rfinger",
         (0.0,  finger_y, finger_z),
         (fb, ft, fl),
         (0.20, 0.85, 0.20, 1.0)),
        ("ghost_swept",
         (0.0, 0.0, sw_z_center),
         (sw_x_extent, sw_y_extent, sw_z_extent),
         (0.20, 0.50, 0.95, 1.0)),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lid-at-edge", type=float, default=None, metavar="OVERHANG_FRAC",
                   help="Move the lid to the near edge of the support before teleop.")
    p.add_argument("--lid-edge-pause", type=float, default=0.0,
                   help="Seconds to idle after placing the lid at the edge.")
    p.add_argument("--initial-z-offset", type=float, default=0.30,
                   help="Anchor starts this many meters above the lid AABB center.")
    p.add_argument("--transport-timeout", type=float, default=60.0,
                   help="cuRobo per-attempt timeout when planning to a target.")
    p.add_argument("--gripper-toggle-key", default="space",
                   help="Key that toggles gripper open/close.")
    p.add_argument("--lid-mass", type=float, default=None,
                   help="If set, override the lid's root-link mass (kg) "
                        "after env build. Useful for A/B testing whether "
                        "PD tracking degradation during held-object "
                        "transport is mass-induced torque saturation.")
    # Saving: each ENTER → replay cycle is one LeRobot episode plus a
    # sim_state.json snapshot of the segment-start. Variants can later
    # be regenerated per-segment by restoring the snapshot.
    p.add_argument("--save-out-dir", type=Path, default=None,
                   help="Directory where per-segment sim-state snapshots "
                        "(JSON) are written. Enables saving when set.")
    p.add_argument("--lerobot-repo-id", default=None,
                   help="HF repo id for the LeRobot v2.1 dataset; passed "
                        "to create_or_open_dataset. Required when "
                        "--save-out-dir is set.")
    p.add_argument("--lerobot-root", type=Path, default=None,
                   help="Local LeRobot dataset root (overrides default).")
    p.add_argument("--lerobot-prompt", default="teleop-driven manipulation",
                   help="Language instruction used as the episode prompt.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--record-resolution", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    from tools.pick_and_place_from_dataset import _build_env, _init_omnigibson, _replay_holding, _solve_one_segment
    from tools.pick_up_lid_from_dataset import _place_lid_at_edge
    from tools.lid_transport_from_dataset import _identify_lid_container
    from tools.replay_empty_from_dataset import _load_scene_info
    from maniguard.utils.lid_attach import LidSnapper

    task_dir = args.task_dir.resolve()
    if not (task_dir / "diagnostics.jsonl").is_file():
        raise SystemExit(f"diagnostics.jsonl missing at {task_dir}")
    if not (task_dir / f"scene_ep{args.episode}.json").is_file():
        if (task_dir / f"scene_ep{args.episode}_replay.json").is_file():
            (task_dir / f"scene_ep{args.episode}.json").symlink_to(
                f"scene_ep{args.episode}_replay.json")

    _init_omnigibson(headless=False)
    import torch as th
    import numpy as np
    import omnigibson.utils.transform_utils as T
    from omnigibson.objects.primitive_object import PrimitiveObject
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )

    saving = args.save_out_dir is not None
    if saving and not args.lerobot_repo_id:
        raise SystemExit("--save-out-dir requires --lerobot-repo-id")
    lerobot_dataset = None
    if saving:
        # Install the wrist camera patch BEFORE env build (the patch
        # injects a wrist VisionSensor when the robot config is loaded).
        from tools._sft_recorder import install_wrist_camera_patch
        install_wrist_camera_patch()
        from maniguard.data.lerobot.lerobot_writer import create_or_open_dataset
        lerobot_dataset = create_or_open_dataset(
            repo_id=args.lerobot_repo_id, root=args.lerobot_root,
            fps=args.video_fps, resolution=args.record_resolution,
        )
        print(f"[gtt] LeRobot dataset open at {lerobot_dataset.root}  "
              f"(starting from episode {lerobot_dataset.meta.total_episodes})",
              flush=True)
        args.save_out_dir.mkdir(parents=True, exist_ok=True)

    print("[gtt] Building env...", flush=True)
    env, og_mod, diagnostics, _ = _build_env(
        task_dir, args.episode, no_distractors=False,
        record_sft=saving, record_resolution=args.record_resolution,
    )
    scene_info = _load_scene_info(task_dir, args.episode)
    lid_name, container_name = _identify_lid_container(scene_info, diagnostics)
    lid = env.scene.object_registry("name", lid_name)
    container = env.scene.object_registry("name", container_name)
    print(f"[gtt] lid={lid_name}  container={container_name}", flush=True)
    if args.lid_mass is not None:
        try:
            old_mass = float(lid.root_link.mass)
        except Exception:
            old_mass = None
        try:
            lid.root_link.mass = float(args.lid_mass)
            print(f"[gtt] lid mass override: {old_mass} -> "
                  f"{float(args.lid_mass)} kg", flush=True)
        except Exception as exc:
            print(f"[gtt] lid-mass override raised "
                  f"{type(exc).__name__}: {exc}", flush=True)
    if args.lid_at_edge is not None:
        _place_lid_at_edge(env, lid, container, diagnostics, args)

    robot = env.robots[0]
    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]
    eef_link_name = robot.eef_link_names[arm]


    print("[gtt] Initializing cuRobo...", flush=True)
    primitives = StarterSemanticActionPrimitives(
        env, robot, enable_head_tracking=False)
    motion_gen = primitives._motion_generator
    print("[gtt] cuRobo ready.", flush=True)

    # LidSnapper: contact-triggered M-link/F-link alignment. Discovers
    # pair candidates at init by scanning the scene; on every tick once
    # the gripper has released the lid (or even during transport with
    # contacts), it checks whether the held lid's F-link is aligned
    # with the container's M-link and snaps them together via OG's
    # AttachedTo state. Without this, the teleop release behavior
    # would leave the lid resting unattached.
    print("[gtt] Initializing LidSnapper...", flush=True)
    snapper = LidSnapper(env)
    print("[gtt] LidSnapper ready.", flush=True)

    # ── Spawn the anchor + 4 ghost boxes ──────────────────────────────────
    lo, hi = lid.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, dtype=np.float64)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, dtype=np.float64)
    init_anchor_pos = np.array([
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        hi[2] + float(args.initial_z_offset),
    ], dtype=np.float64)

    # The anchor is a small bright cube the user actually grabs in the
    # viewport. We track its pose every tick and re-pose the 4 ghost boxes.
    anchor_name = "gtt_anchor"
    existing = env.scene.object_registry("name", anchor_name)
    if existing is not None:
        try:
            env.scene.remove_object(existing)
        except Exception:
            env.scene.remove_object(obj=existing)
    anchor = PrimitiveObject(
        relative_prim_path=f"/{anchor_name}",
        name=anchor_name,
        category="gtt_anchor",
        primitive_type="Cube",
        scale=th.tensor([0.025, 0.025, 0.025], dtype=th.float32),  # 2.5cm cube
        fixed_base=True,
        visual_only=True,
        rgba=(1.0, 0.4, 0.1, 1.0),  # opaque orange
    )
    env.scene.add_object(anchor)
    # Initial orientation: 180° about world x so the gripper's approach axis
    # (eef +z) points DOWN toward the table — the natural top-down grasp
    # pose. Without this, the identity quat leaves the gripper pointing up
    # (palm to ceiling) and every plan fails.
    anchor.set_position_orientation(
        position=th.as_tensor(init_anchor_pos, dtype=th.float32),
        orientation=th.tensor([1.0, 0.0, 0.0, 0.0], dtype=th.float32),
    )

    # OG's get_position_orientation() reads from a cached physics-body
    # state that doesn't sync back when the user drags via Isaac viewport
    # gizmo (the gizmo edits the USD xform directly). We instead compute
    # the anchor's local-to-world transform via UsdGeom.Xformable's
    # ComputeLocalToWorldTransform, which handles ANY xform-op layout
    # (translate+orient pair, single matrix op, scale, etc.).
    #
    # pxr must be imported AFTER Isaac Sim has fully booted (post env build)
    # — importing it earlier races Isaac's extension loader and brings
    # down the entire sim with "extension class wrapper not created yet".
    from pxr import UsdGeom, Gf, Usd
    anchor_prim = anchor.prim
    anchor_xformable = UsdGeom.Xformable(anchor_prim)
    # set_position_orientation only updates OG's cached pose, not the USD
    # xformOps that ComputeLocalToWorldTransform reads. Initialize the USD
    # ops to the same pose so the marker actually sits where we placed it
    # (and the gizmo subsequently edits these same ops).
    _anchor_ops = anchor_xformable.GetOrderedXformOps()
    _anchor_translate_op = next(
        (op for op in _anchor_ops
         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None,
    )
    if _anchor_translate_op is None:
        _anchor_translate_op = anchor_xformable.AddTranslateOp()
    _anchor_orient_op = next(
        (op for op in _anchor_ops
         if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None,
    )
    if _anchor_orient_op is None:
        _anchor_orient_op = anchor_xformable.AddOrientOp()
    _anchor_translate_op.Set(Gf.Vec3d(
        float(init_anchor_pos[0]),
        float(init_anchor_pos[1]),
        float(init_anchor_pos[2])))
    # Initial orient: 180° about world x, so gripper +z points down.
    # Quatd is scalar-first (w, x, y, z).
    _anchor_orient_op.Set(Gf.Quatd(0.0, 1.0, 0.0, 0.0))

    # Ghosts are raw UsdGeom.Cube prims under a non-scene path
    # (/World/_gtt_ghosts/...) — explicitly OUTSIDE OG's scene object
    # registry. Earlier iterations used PrimitiveObject(visual_only=True)
    # but og.sim.dump_state/load_state (the articulation rebind we run
    # after every replay) has side-effects on visual_only scene objects
    # that hide the rendered mesh even though USD attributes
    # (visibility, purpose, active, xformOpOrder including scale) all
    # check out. Raw USD prims under a non-scene path aren't visited
    # by dump_state/load_state at all, so they survive cleanly.
    from pxr import Vt, Sdf  # noqa: F401 (Sdf imported for completeness)
    stage = anchor.prim.GetStage()
    ghost_root_path = "/World/_gtt_ghosts"
    stage.DefinePrim(ghost_root_path, "Xform")

    class _GhostHandle:
        __slots__ = ("name", "prim")
        def __init__(self, name, prim):
            self.name = name
            self.prim = prim

    ghost_objs = []
    for name, local_pos, half_extents, rgba in _ghost_specs():
        prim_path = f"{ghost_root_path}/{name}"
        # Wipe any leftover from a prior run.
        existing = stage.GetPrimAtPath(prim_path)
        if existing and existing.IsValid():
            stage.RemovePrim(prim_path)
        cube = UsdGeom.Cube.Define(stage, prim_path)
        # Unit cube; xformOp:scale handles dimension.
        cube.GetSizeAttr().Set(1.0)
        cube_prim = cube.GetPrim()
        # displayColor + displayOpacity primvars give us the rgba look
        # without needing a material binding.
        UsdGeom.PrimvarsAPI(cube_prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
                Vt.Vec3fArray([(float(rgba[0]), float(rgba[1]),
                                float(rgba[2]))]))
        UsdGeom.PrimvarsAPI(cube_prim).CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray).Set(
                Vt.FloatArray([float(rgba[3])]))
        xf = UsdGeom.Xformable(cube_prim)
        translate_op = xf.AddTranslateOp(
            precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op = xf.AddOrientOp(
            precision=UsdGeom.XformOp.PrecisionDouble)
        scale_op = xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(
            2.0 * float(half_extents[0]),
            2.0 * float(half_extents[1]),
            2.0 * float(half_extents[2]),
        ))
        g = _GhostHandle(name, cube_prim)
        ghost_objs.append(
            (g, np.asarray(local_pos, dtype=np.float64),
             translate_op, orient_op),
        )

    def _refresh_xform_handles():
        """Re-fetch the cached USD xformOp handles for the anchor and
        every ghost. Necessary after og.sim.dump_state()/load_state()
        roundtrip because load_state rebuilds USD xform stacks and any
        cached xformOp handle becomes a phantom — writes to it land in
        memory but the prim's rendered transform reads from the new ops.

        Also force the xformOpOrder to ['translate', 'orient'] so the
        ops we hold are guaranteed to be the ones the renderer reads.
        load_state can insert a different op (matrix or transform) that
        would otherwise override our translate/orient writes.
        """
        nonlocal anchor_xformable, _anchor_translate_op, _anchor_orient_op

        def _rebind(prim):
            # Refresh the Xformable wrapper from the prim and re-fetch
            # the existing translate / orient op handles. Do NOT clear
            # xformOpOrder — that drops the existing xformOp:scale and
            # collapses the ghost cubes back to default 1 m³ dimensions
            # (so they look gone because they engulf the camera). Only
            # add ops that are genuinely missing, and match precision.
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            t_op = next(
                (op for op in ops
                 if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None,
            )
            if t_op is None:
                t_op = xf.AddTranslateOp(
                    precision=UsdGeom.XformOp.PrecisionDouble)
            o_op = next(
                (op for op in ops
                 if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None,
            )
            if o_op is None:
                o_op = xf.AddOrientOp(
                    precision=UsdGeom.XformOp.PrecisionDouble)
            return xf, t_op, o_op

        anchor_xformable, _anchor_translate_op, _anchor_orient_op = _rebind(
            anchor.prim)
        for idx, (g, local_pos, _t_op, _o_op) in enumerate(ghost_objs):
            _, t_op, o_op = _rebind(g.prim)
            ghost_objs[idx] = (g, local_pos, t_op, o_op)
        # Diagnostic + corrective: print the post-rebind xformOpOrder
        # AND inspect visibility / active state. Force visibility to
        # "inherited" and prim to active in case load_state hid them.
        try:
            for idx, (g, _lp, _t, _o) in enumerate(ghost_objs):
                p = g.prim
                imageable = UsdGeom.Imageable(p)
                vis_attr = imageable.GetVisibilityAttr()
                cur_vis = vis_attr.Get()
                purp = imageable.GetPurposeAttr().Get()
                active = p.IsActive()
                # Force visible (inherited) + active
                try:
                    vis_attr.Set(UsdGeom.Tokens.inherited)
                except Exception:
                    pass
                try:
                    p.SetActive(True)
                except Exception:
                    pass
                # Force purpose=default (so the prim is rendered in the
                # main camera, not just guide/proxy).
                try:
                    imageable.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
                except Exception:
                    pass
                if idx == 0:
                    order_attr = p.GetAttribute("xformOpOrder")
                    order = list(order_attr.Get() or [])
                    print(f"[gtt] post-rebind ghost[{idx}] name={g.name} "
                          f"order={order} visibility={cur_vis} "
                          f"purpose={purp} active={active}", flush=True)
        except Exception as exc:
            print(f"[gtt] post-rebind probe raised "
                  f"{type(exc).__name__}: {exc}", flush=True)

    def _update_ghost_poses():
        """Sync the 4 ghost boxes to the anchor pose + their local offsets.

        Reads the anchor's local-to-world transform from USD via
        ComputeLocalToWorldTransform (gizmo-edited). Then writes each
        ghost's xformOp:translate / :orient.
        """
        # ComputeLocalToWorldTransform returns a Gf.Matrix4d — decompose
        # into translation + rotation. Matrix4d is row-major:
        # the bottom row holds the translation when the convention is
        # "row vectors right-multiply by the matrix" (USD convention).
        m4 = anchor_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        a_pos = np.array([float(m4[3][0]),
                          float(m4[3][1]),
                          float(m4[3][2])], dtype=np.float64)
        rot = m4.ExtractRotation()
        q = rot.GetQuat()  # Gf.Quatd, scalar-first (w, x, y, z)
        a_quat = np.array([
            float(q.GetImaginary()[0]),
            float(q.GetImaginary()[1]),
            float(q.GetImaginary()[2]),
            float(q.GetReal()),
        ], dtype=np.float64)
        qx, qy, qz, qw = float(a_quat[0]), float(a_quat[1]), float(a_quat[2]), float(a_quat[3])
        n = qx * qx + qy * qy + qz * qz + qw * qw
        s = 2.0 / n if n > 0 else 0.0
        xs, ys, zs = qx * s, qy * s, qz * s
        wx, wy, wz = qw * xs, qw * ys, qw * zs
        xx, xy, xz = qx * xs, qx * ys, qx * zs
        yy, yz, zz = qy * ys, qy * zs, qz * zs
        R = np.array([
            [1.0 - (yy + zz),       xy - wz,            xz + wy],
            [xy + wz,               1.0 - (xx + zz),    yz - wx],
            [xz - wy,               yz + wx,            1.0 - (xx + yy)],
        ], dtype=np.float64)
        # USD Gf.Quatd is (w, x, y, z) — scalar-first — while OG uses (x, y, z, w).
        # The orient xformOp on USD prims defaults to double precision (Quatd),
        # so set with matching type or Set raises a Type mismatch error.
        gf_quat = Gf.Quatd(qw, qx, qy, qz)
        for g, local_pos, translate_op, orient_op in ghost_objs:
            world_pos = a_pos + R @ local_pos
            translate_op.Set(Gf.Vec3d(float(world_pos[0]),
                                      float(world_pos[1]),
                                      float(world_pos[2])))
            orient_op.Set(gf_quat)
        return a_pos, a_quat

    _update_ghost_poses()

    # ── Keyboard listener (pynput, X11-safe; no Kit focus needed) ─────────
    from pynput import keyboard as pkb

    want_execute = False
    want_quit = False
    want_toggle_gripper = False
    want_plan_to_current = False
    gripper_open = True

    def _on_press(key):
        nonlocal want_execute, want_quit, want_toggle_gripper
        nonlocal want_plan_to_current
        if isinstance(key, pkb.KeyCode):
            c = (key.char or "").lower()
            if c == "q":
                print("[gtt] key Q pressed — quit", flush=True)
                want_quit = True
            elif c == "p":
                # Sanity-check: plan to the CURRENT eef pose. If cuRobo
                # fails even here, it's a convention bug (quat handedness,
                # wrong target link, etc.) — not a reachability problem.
                print("[gtt] key P pressed — plan to CURRENT eef pose",
                      flush=True)
                want_plan_to_current = True
                want_execute = True
        elif key == pkb.Key.enter:
            print("[gtt] key ENTER pressed — plan to anchor target",
                  flush=True)
            want_execute = True
        elif key == pkb.Key.space:
            want_toggle_gripper = True

    listener = pkb.Listener(on_press=_on_press)
    listener.daemon = True
    listener.start()

    print("\n[gtt] === gripper target teleop ===\n"
          "  - drag the ORANGE anchor cube in the viewport via W/E/R gizmo;\n"
          "    the 4 ghost boxes are the gripper preview.\n"
          "  - ENTER  : cuRobo plan & move Franka to the ghost target.\n"
          "  - SPACE  : toggle gripper open / close.\n"
          "  - P      : sanity-check — plan to CURRENT eef pose. If cuRobo\n"
          "            fails even here, the frame/quat convention is wrong\n"
          "            (not a reachability issue).\n"
          "  - Q      : quit.\n", flush=True)

    tick = 0
    last_anchor_xyz = None
    last_hold_action = None
    # Edge-triggered articulation health: print only when state flips
    # (working → broken or broken → working), not every tick. The
    # post-replay broken state can last hundreds of ticks; spamming would
    # bury everything else.
    art_was_broken = False
    art_broken_since_tick = None
    try:
        while not want_quit:
            og_mod.sim.step()
            a_pos, a_quat = _update_ghost_poses()

            # Hold gripper at current open/closed command between waypoints.
            # No arm motion in idle — env.step requires an action; the OG
            # JointController with impedance + absolute targets holds current
            # joints when commanded with the current joint positions.
            #
            # robot.get_joint_positions() occasionally returns None right
            # after a failed cuRobo attempt (Isaac articulation goes through
            # a transient un-initialized state). Reuse the previous action
            # rather than crash on those ticks.
            action = th.zeros(robot.action_dim, dtype=th.float32)
            try:
                current_q = robot.get_joint_positions()
                arm_action_idx_local = robot.arm_action_idx[arm]
                for i, ci in enumerate(arm_action_idx_local.tolist()):
                    action[ci] = current_q[arm_control_idx[i]].float()
                last_hold_action = action.clone()
                if art_was_broken:
                    # Recovered. Emit once.
                    duration = (tick - art_broken_since_tick
                                if art_broken_since_tick is not None else "?")
                    print(f"[gtt] articulation RECOVERED at tick={tick} "
                          f"(broken for {duration} ticks)", flush=True)
                    art_was_broken = False
                    art_broken_since_tick = None
            except (AttributeError, RuntimeError, TypeError) as exc:
                # Articulation transiently de-init'd. Edge-triggered:
                # print once on entry, then stay quiet until recovery.
                if not art_was_broken:
                    print(f"[gtt] articulation BROKEN at tick={tick} "
                          f"({type(exc).__name__}: {exc}); reusing last "
                          f"hold action until it recovers", flush=True)
                    art_was_broken = True
                    art_broken_since_tick = tick
                if last_hold_action is not None:
                    action = last_hold_action.clone()
                # else: zeros — JointController treats as no-op delta.
            action[gripper_action_idx] = 1.0 if gripper_open else -1.0

            if want_toggle_gripper:
                gripper_open = not gripper_open
                print(f"[gtt] gripper -> {'OPEN' if gripper_open else 'CLOSE'}",
                      flush=True)
                want_toggle_gripper = False
                action[gripper_action_idx] = 1.0 if gripper_open else -1.0

            env.step(action)

            # LidSnapper tick: fires on the exact step the M/F-link pair
            # comes into alignment and snaps via OG's AttachedTo state.
            # Cheap to call every tick; returns the lid name only on the
            # firing step (otherwise None).
            try:
                snapper.try_snap(robot=robot, verbose=False)
            except Exception as exc:
                print(f"[gtt] snapper.try_snap raised "
                      f"{type(exc).__name__}: {exc}", flush=True)

            if want_execute:
                want_execute = False
                # Always print the current eef pose so we can compare with
                # the requested target — large mismatch = wrong frame.
                cur_eef_pos_t, cur_eef_quat_t = (
                    robot.eef_links[arm].get_position_orientation())
                cur_eef_pos = np.asarray(
                    cur_eef_pos_t.cpu() if hasattr(cur_eef_pos_t, "cpu")
                    else cur_eef_pos_t, dtype=np.float64)
                cur_eef_quat = np.asarray(
                    cur_eef_quat_t.cpu() if hasattr(cur_eef_quat_t, "cpu")
                    else cur_eef_quat_t, dtype=np.float64)
                if want_plan_to_current:
                    want_plan_to_current = False
                    print(f"[gtt] PLAN-TO-CURRENT (sanity): using eef "
                          f"pose=({cur_eef_pos[0]:.3f},{cur_eef_pos[1]:.3f},"
                          f"{cur_eef_pos[2]:.3f}) "
                          f"quat=({cur_eef_quat[0]:.3f},{cur_eef_quat[1]:.3f},"
                          f"{cur_eef_quat[2]:.3f},{cur_eef_quat[3]:.3f})",
                          flush=True)
                    target_pos = cur_eef_pos
                    target_quat = cur_eef_quat
                else:
                    print(f"[gtt] PLAN to target "
                          f"pos=({a_pos[0]:.3f},{a_pos[1]:.3f},{a_pos[2]:.3f}) "
                          f"quat=({a_quat[0]:.3f},{a_quat[1]:.3f},"
                          f"{a_quat[2]:.3f},{a_quat[3]:.3f})", flush=True)
                    print(f"[gtt]   (current eef "
                          f"pos=({cur_eef_pos[0]:.3f},{cur_eef_pos[1]:.3f},"
                          f"{cur_eef_pos[2]:.3f}) "
                          f"quat=({cur_eef_quat[0]:.3f},{cur_eef_quat[1]:.3f},"
                          f"{cur_eef_quat[2]:.3f},{cur_eef_quat[3]:.3f}))",
                          flush=True)
                    target_pos = a_pos
                    target_quat = a_quat
                t_plan = time.time()
                eef_target_pos = th.as_tensor(target_pos, dtype=th.float32)
                eef_target_quat = th.as_tensor(target_quat, dtype=th.float32)
                motion_gen.update_obstacles(ignore_objects=[])
                arm_traj = _plan_to_eef_target(
                    motion_gen, robot, eef_link_name,
                    eef_target_pos, eef_target_quat,
                    arm_control_idx, args.transport_timeout)
                plan_wall = time.time() - t_plan
                if arm_traj is None:
                    print(f"[gtt] cuRobo FAILED ({plan_wall:.1f}s) — "
                          f"target unreachable. Move the anchor and retry.",
                          flush=True)
                else:
                    print(f"[gtt] cuRobo OK ({plan_wall:.1f}s, "
                          f"{len(arm_traj)} waypoints). Replaying...",
                          flush=True)
                    # Per-segment recorder. One LeRobot episode per ENTER
                    # cycle so each segment is independently restoreable
                    # for later variant generation.
                    segment_recorder = None
                    segment_writer = None
                    segment_idx = None
                    if saving:
                        from tools._sft_recorder import SFTRecorder
                        from maniguard.data.lerobot.lerobot_writer import (
                            LeRobotEpisodeWriter,
                        )
                        segment_idx = (
                            lerobot_dataset.meta.total_episodes
                        )
                        # Dump segment-start sim state. serialized=True
                        # yields a flat 1-D torch tensor (round-trippable
                        # via og.sim.load_state(state, serialized=True))
                        # which we persist as .npy for stable I/O.
                        seg_state_path = (args.save_out_dir /
                                          f"segment_{segment_idx:04d}_start_state.npy")
                        try:
                            seg_state_t = og_mod.sim.dump_state(
                                serialized=True)
                            seg_state_np = (
                                seg_state_t.cpu().numpy()
                                if hasattr(seg_state_t, "cpu")
                                else np.asarray(seg_state_t))
                            np.save(seg_state_path, seg_state_np)
                            print(f"[gtt] saved segment-start state -> "
                                  f"{seg_state_path} "
                                  f"(shape={seg_state_np.shape})",
                                  flush=True)
                        except Exception as exc:
                            print(f"[gtt] segment-start state dump raised "
                                  f"{type(exc).__name__}: {exc}", flush=True)
                        segment_writer = LeRobotEpisodeWriter(lerobot_dataset)
                        segment_recorder = SFTRecorder(
                            args.save_out_dir / f"segment_{segment_idx:04d}",
                            resolution=args.record_resolution,
                            fps=args.video_fps,
                            lerobot_writer=segment_writer,
                            lerobot_prompt=args.lerobot_prompt,
                        )
                        segment_recorder.attach(env, robot)
                    ok = _replay_holding(
                        env, og_mod, robot, lid, arm_traj,
                        deadline=time.time() + max(args.transport_timeout * 4, 60.0),
                        sft_recorder=segment_recorder,
                        segment_breakdown=[("gtt_target", len(arm_traj))],
                        gripper_cmd=(1.0 if gripper_open else -1.0),
                    )
                    print(f"[gtt] replay done (ok={ok})", flush=True)
                    if segment_recorder is not None:
                        try:
                            attached_now = bool(
                                lid.states[__import__(
                                    "omnigibson.object_states",
                                    fromlist=["AttachedTo"]).AttachedTo
                                ].get_value(container))
                        except Exception:
                            attached_now = False
                        try:
                            segment_recorder.finalize(
                                success=bool(ok),
                                attrs={
                                    "task_dir": str(task_dir),
                                    "segment_idx": segment_idx,
                                    "target_pos": list(map(float, target_pos)),
                                    "target_quat": list(map(float, target_quat)),
                                    "gripper_open": bool(gripper_open),
                                    "lid_attached_to_container": attached_now,
                                },
                            )
                            print(f"[gtt] saved segment {segment_idx} "
                                  f"(success={bool(ok)}, "
                                  f"attached={attached_now})", flush=True)
                        except Exception as exc:
                            print(f"[gtt] segment finalize raised "
                                  f"{type(exc).__name__}: {exc}", flush=True)
                    # Re-bind the articulation handle. _replay_holding's
                    # rapid env.step loop sometimes invalidates Isaac's
                    # ArticulationView. pnp_from_dataset works around this
                    # via sim.load_state between variants; we do the same
                    # here. Two-tier strategy:
                    #   (1) step the sim a few times to let Isaac recover
                    #       naturally, then dump_state/load_state roundtrip
                    #       (cheap, ~50 ms);
                    #   (2) if (1) fails, og.sim.stop() + og.sim.play()
                    #       — heavy (~1 s pause) but definitively re-
                    #       binds everything.
                    rebound = False
                    for _ in range(20):
                        og_mod.sim.step()
                    try:
                        snapshot = og_mod.sim.dump_state()
                        og_mod.sim.load_state(snapshot)
                        og_mod.sim.step()
                        print("[gtt] articulation re-bound via "
                              "dump_state/load_state roundtrip",
                              flush=True)
                        rebound = True
                    except Exception as exc:
                        print(f"[gtt] dump/load_state failed "
                              f"({type(exc).__name__}: {exc}); "
                              "falling back to sim.stop()+play()",
                              flush=True)
                    if not rebound:
                        try:
                            og_mod.sim.stop()
                            og_mod.sim.play()
                            og_mod.sim.step()
                            print("[gtt] articulation re-bound via "
                                  "sim.stop()+play()", flush=True)
                            rebound = True
                        except Exception as exc:
                            print(f"[gtt] sim.stop()+play() also FAILED: "
                                  f"{type(exc).__name__}: {exc}",
                                  flush=True)
                    # The dump/load (or stop/play) snaps every scene
                    # object back to its physics-root pose at dump time
                    # AND rebuilds the USD xform stacks under each prim.
                    # That invalidates our cached translate/orient op
                    # handles — writes to them land but the prim's
                    # rendered transform reads from the new ops. Refetch
                    # handles before re-applying the anchor-driven pose.
                    try:
                        _refresh_xform_handles()
                    except Exception as exc:
                        print(f"[gtt] _refresh_xform_handles raised "
                              f"{type(exc).__name__}: {exc}", flush=True)
                    # Re-seed the anchor's USD ops with the pose the user
                    # had it at before the replay, so the gizmo's local
                    # edits aren't lost. We pull from a_pos / a_quat
                    # captured at the start of this iteration.
                    try:
                        _anchor_translate_op.Set(Gf.Vec3d(
                            float(a_pos[0]), float(a_pos[1]), float(a_pos[2])))
                        _anchor_orient_op.Set(Gf.Quatd(
                            float(a_quat[3]),
                            float(a_quat[0]),
                            float(a_quat[1]),
                            float(a_quat[2])))
                    except Exception as exc:
                        print(f"[gtt] re-seed anchor xform raised "
                              f"{type(exc).__name__}: {exc}", flush=True)
                    _update_ghost_poses()

            tick += 1
            cur_xyz = (round(float(a_pos[0]), 3),
                       round(float(a_pos[1]), 3),
                       round(float(a_pos[2]), 3))
            # Live: every 30 ticks, print anchor pose + each ghost USD-pose
            # so we can verify the user's gizmo drag is propagating, and
            # whether the ghosts are tracking. Only emits when something
            # actually changed since the last print.
            if tick % 30 == 0 and cur_xyz != last_anchor_xyz:
                from pxr import Gf as _Gf  # noqa
                ghost_lines = []
                for g, local_pos, translate_op, _ in ghost_objs:
                    val = translate_op.Get()
                    ghost_lines.append(
                        f"{g.name}=({float(val[0]):.3f},"
                        f"{float(val[1]):.3f},{float(val[2]):.3f})")
                print(f"[gtt-debug] tick={tick}  anchor={cur_xyz}  "
                      f"gripper={'OPEN' if gripper_open else 'CLOSE'}  "
                      f"ghosts: {' | '.join(ghost_lines)}", flush=True)
                last_anchor_xyz = cur_xyz
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
