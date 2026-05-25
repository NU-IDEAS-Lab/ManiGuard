"""Lid-transport pipeline driven by a replay-empty task dump.

Given a single lid_transport task folder containing ``scene_ep<N>.json`` +
``diagnostics.jsonl`` (produced by ``LidTransportPipeline``), this script:

  1. Rebuilds the task in a bare OmniGibson Scene: floor + fixed support
     + lid + container + (optional) food + Franka at dump pose + green
     ``goal_region`` marker + diagnostics cameras. ``LidSnapper`` is
     installed for automatic lid → container attachment when the lid
     touches the container after release.
  2. **Phase 1A** — Grasps the lid (OBB sampler + cuRobo + AG).
  3. **Phase 1B** — Plans + replays a 3-segment cuRobo transport of the
     lid to just above the container's F (female attach) meta-link.
  4. **Phase 1C** — Opens the gripper and retreats vertically; ``LidSnapper``
     repositions + welds the lid to the container's F-link.
  5. **Phase 2A** — Grasps the container (cuRobo treats the now-attached
     lid as a separate obstacle, so candidates that collide with it are
     filtered).
  6. **Phase 2B** — Plans + replays a 3-segment cuRobo transport of the
     container to ``goal_region.center_world``.

The full bimanual flow is recorded into ONE HDF5 + 3 review MP4s when
``--record-sft`` is on. State + action format match
``tools/pick_and_place_from_dataset.py``: 8D state, 7D EEF-delta action,
gripper_cmd ∈ {+1 (open), −1 (close)}.

Success = container AABB ∩ goal sphere AND robot still grasping container
AND lid still ``AttachedTo`` container.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m tools.lid_transport_from_dataset \\
            --task-dir datasets/6fam-base-20260513/lid_transport/task_0000/base \\
            --episode 1 --seed 0 --record-sft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-dir", type=Path, required=True,
                   help="Task folder (must contain scene_ep<N>.json + diagnostics.jsonl)")
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <task-dir>/lid_transport)")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--gui", action="store_true")
    # Phase A (pick) — shared params for both lid and container picks
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None,
                   help="Multi-attempt mode: single int, range 'A-B', or "
                        "comma list 'A,B,C'. Reuses one env across all "
                        "seeds; resets to a post-edge-placement snapshot "
                        "between attempts. Stops at --max-successes.")
    p.add_argument("--max-successes", type=int, default=5,
                   help="Stop the --seeds loop once this many succeed.")
    p.add_argument("--pick-timeout", type=float, default=300.0,
                   help="Wall-clock budget for EACH pick phase (lid, container).")
    p.add_argument("--max-reach", type=float, default=0.95)
    p.add_argument("--max-obj-to-eef-after-hold", type=float, default=0.20)
    # Phase B (transport)
    p.add_argument("--transport-timeout", type=float, default=60.0,
                   help="Wall-clock budget for each transport-plan cuRobo call.")
    p.add_argument("--ik-precheck", action="store_true",
                   help="Phase A ik_only precheck (skip unreachable candidates).")
    p.add_argument("--single-stage-grasp", action="store_true",
                   help="Skip the standoff + gripper-collision-disabled linear "
                        "servo. Plan ONE cuRobo trajectory directly to the "
                        "OBB-sampled grasp pose with full collision (gripper "
                        "links included). Rejects up-front the phantom plans "
                        "that the two-stage approach produces for thin lids.")
    # Lid-specific
    p.add_argument("--lid-at-edge", type=float, default=None, metavar="OVERHANG_FRAC",
                   help="Move the lid to the near edge of the support surface "
                        "(closest to robot, local x = lx0) before Phase 1A. "
                        "OVERHANG_FRAC in [0,1] = fraction of the lid's local "
                        "half-extent along robot-local x that hangs past the "
                        "edge (0.5 = half overhangs the table).")
    p.add_argument("--lid-edge-pause", type=float, default=0.0,
                   help="Seconds to idle (with og.sim.step()) after moving the "
                        "lid to the edge — useful with --gui so you can see "
                        "the placement before Phase 1A starts.")
    p.add_argument("--lid-clearance-above-container", type=float, default=0.03,
                   help="Place the lid this far above the container's F-link "
                        "during Phase 1B descent. LidSnapper takes over from "
                        "there: it rep-aligns + welds when the lid touches.")
    p.add_argument("--release-steps", type=int, default=20,
                   help="OSC steps to open gripper + retreat after lid drop.")
    p.add_argument("--release-retreat-z", type=float, default=0.10,
                   help="Vertical retreat (m) after opening the gripper, to "
                        "clear the lid before settling for LidSnapper.")
    p.add_argument("--post-release-settle", type=int, default=60,
                   help="Sim steps to settle physics after release; "
                        "LidSnapper.try_snap is invoked on every step.")
    # SFT recording
    p.add_argument("--record-sft", action="store_true",
                   help="Capture (image_left, image_right, wrist_image, state, "
                        "action) per env.step across Phase 1+2 and emit "
                        "rollout.hdf5 + 3 MP4s into --out-dir on success.")
    p.add_argument("--record-resolution", type=int, default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Identify lid / container / food from spawn_specs
# ---------------------------------------------------------------------------


def _identify_lid_container(scene_info, diagnostics):
    """Return (lid_name, container_name) — the actual scene-object names
    matching the spawn-spec roles. Walks init_info to find an instance
    with category matching each role's category.
    """
    spawn_specs = diagnostics["selection"]["spawn_specs"]
    role_to_cat = {s["role"]: s["category"] for s in spawn_specs}
    if "lid" not in role_to_cat or "container" not in role_to_cat:
        raise RuntimeError(
            f"spawn_specs missing lid or container role: roles={list(role_to_cat)}")
    init_info = scene_info["objects_info"]["init_info"]

    # Match using the same pattern as _is_task_object_name (category_<digits>).
    import re
    _TASK_OBJ_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_\d+$")

    def _find(cat):
        for n, info in init_info.items():
            if not _TASK_OBJ_PATTERN.match(n):
                continue
            if info.get("args", {}).get("category") == cat:
                prefix, _, tail = n.rpartition("_")
                if prefix == cat and tail.isdigit():
                    return n
        return None

    lid = _find(role_to_cat["lid"])
    container = _find(role_to_cat["container"])
    if lid is None or container is None:
        raise RuntimeError(
            f"Could not find lid={role_to_cat['lid']!r} or "
            f"container={role_to_cat['container']!r} in scene snapshot")
    return lid, container


# ---------------------------------------------------------------------------
# Lid-above-container goal pose
# ---------------------------------------------------------------------------


def _lid_goal_world_pose(lid, container, *, clearance_z: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """World-frame target POSE for the LID ROOT LINK so that lid's M-link
    frame coincides with container's F-link frame, plus ``clearance_z``
    in world z.

    Returns (pos_xyz, quat_xyzw). Mirrors the transform in
    ``sentinel.utils.lid_attach.reposition_lid_onto_F`` but does NOT
    mutate the lid pose. The quat encodes the lid orientation needed
    for snap-attach to fire cleanly — without it the lid lands tilted
    and never makes contact with the container rim.
    """
    import omnigibson.utils.transform_utils as T
    from sentinel.utils.lid_attach import find_M_link, find_F_link

    m = find_M_link(lid)
    if m is None:
        raise RuntimeError(f"lid {lid.name} has no M-link")
    f = find_F_link(container, m.meta_link_id)
    if f is None:
        raise RuntimeError(
            f"container {container.name} has no matching F-link for "
            f"{m.meta_link_id}")

    parent_pos, parent_quat = f.get_position_orientation()
    child_pos, child_quat = m.get_position_orientation()
    child_root_pos, child_root_quat = lid.get_position_orientation()

    rel_pos, rel_quat = T.mat2pose(
        T.pose2mat((parent_pos, parent_quat))
        @ T.pose_inv(T.pose2mat((child_pos, child_quat)))
    )
    new_root_pos, new_root_quat = T.pose_transform(
        rel_pos, rel_quat, child_root_pos, child_root_quat
    )
    pos = np.asarray(
        [float(new_root_pos[0]), float(new_root_pos[1]),
         float(new_root_pos[2]) + float(clearance_z)],
        dtype=np.float64,
    )
    quat = np.asarray([float(new_root_quat[i]) for i in range(4)],
                      dtype=np.float64)
    return pos, quat


# ---------------------------------------------------------------------------
# Release: open gripper + retreat vertically
# ---------------------------------------------------------------------------


def _release_and_retreat(env, og, robot, *, sft_recorder=None,
                         retreat_z: float, n_open_steps: int = 6,
                         n_retreat_steps: int = 14):
    """OSC commands to (1) open the gripper, then (2) retreat ``retreat_z``
    in world +z while keeping the gripper open. Records each step into the
    SFT recorder if provided.
    """
    import torch as th

    arm = robot.default_arm
    arm_action_idx = robot.arm_action_idx[arm]
    gripper_action_idx = robot.gripper_action_idx[arm]

    n_open = max(1, int(n_open_steps))
    n_retreat = max(1, int(n_retreat_steps))
    dz_per_step = float(retreat_z) / n_retreat

    def _step(arm_delta, gripper_cmd):
        action = th.zeros(robot.action_dim, dtype=th.float32)
        action[arm_action_idx] = th.tensor(
            [arm_delta[0], arm_delta[1], arm_delta[2], 0.0, 0.0, 0.0],
            dtype=th.float32,
        )
        action[gripper_action_idx] = float(gripper_cmd)
        env.step(action)
        if sft_recorder is not None:
            act7 = np.zeros(7, dtype=np.float32)
            act7[:3] = np.asarray(arm_delta, dtype=np.float32)
            act7[6] = float(gripper_cmd)
            sft_recorder.record_step(act7, done=False)

    # Open gripper in place.
    for _ in range(n_open):
        _step((0.0, 0.0, 0.0), gripper_cmd=1.0)

    # Retreat in +z, gripper open.
    for _ in range(n_retreat):
        _step((0.0, 0.0, dz_per_step), gripper_cmd=1.0)


def _parse_seeds_arg(args) -> list[int]:
    """Resolve --seeds (priority) or --seed (fallback) into a list of ints.

    --seeds supports: single int, range 'A-B' (inclusive), comma list
    'A,B,C', or None (falls back to --seed).
    """
    if args.seeds is None:
        return [int(args.seed)]
    s = args.seeds.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"task-dir not found: {task_dir}")

    out_dir = (args.out_dir or (task_dir / "lid_transport")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse machinery from the pnp pipeline.
    from tools.pick_and_place_from_dataset import (
        _init_omnigibson,
        _build_env,
        _phase_a_pick,
        _plan_transport,
        _replay_holding,
        _record_phase_a_replay,
    )

    og = _init_omnigibson(headless=not args.gui)

    if args.record_sft:
        from tools._sft_recorder import install_wrist_camera_patch
        install_wrist_camera_patch()

    env, og, diagnostics, camera_names = _build_env(
        task_dir, args.episode,
        record_sft=args.record_sft, record_resolution=args.record_resolution,
        grasping_mode="sticky",
    )

    # Multi-seed mode reuses a single env across attempts. The
    # SFTRecorder and result_payload are PER-ATTEMPT (created inside
    # the seed loop below). In single-seed mode the loop runs once.
    lid_name = ""
    container_name = ""

    try:
        from omnigibson.controllers.controller_base import IsGraspingState
        from sentinel.utils.goal_region import (
            GoalRegionSpec,
            object_intersects_goal_region,
            robot_holds_target,
            target_or_gripper_in_goal,
        )
        from sentinel.utils.lid_attach import LidSnapper
        from omnigibson.object_states import AttachedTo

        # Goal region (always centered on the container per pipeline_common).
        gr = diagnostics["goal_region"]
        goal_spec = GoalRegionSpec.from_json(gr)
        container_name = goal_spec.target_name
        # Pull lid name via spawn_specs (NOT in goal_region).
        with open(task_dir / f"scene_ep{args.episode}_replay.json"
                  if (task_dir / f"scene_ep{args.episode}_replay.json").is_file()
                  else task_dir / f"scene_ep{args.episode}.json", "r") as fh:
            scene_info = json.load(fh)
        lid_name, container_name_ss = _identify_lid_container(scene_info, diagnostics)
        # Sanity: goal_region.target_name should match the container we found.
        if container_name != container_name_ss:
            print(f"[Lid] WARNING goal target_name={container_name} vs "
                  f"spawn-spec container={container_name_ss}; using "
                  f"{container_name}", flush=True)

        robot = env.robots[0]
        lid = env.scene.object_registry("name", lid_name)
        container = env.scene.object_registry("name", container_name)
        if lid is None or container is None:
            raise RuntimeError(
                f"missing lid={lid_name!r} or container={container_name!r}")
        goal_center = np.asarray(goal_spec.center_world, dtype=np.float64)
        goal_radius = float(goal_spec.radius_m)

        # ---- Optional: move lid to the near edge of the support surface ----
        if args.lid_at_edge is not None:
            # Reuse the SAME local↔world transforms used to write
            # support_bounds_robot_local_xy in the first place. Those
            # are full-quaternion rotations, not yaw-only.
            from sentinel.utils.goal_region import (
                _local_xy_to_world, _world_to_local,
            )
            sb = diagnostics.get("goal_region", {}).get(
                "support_bounds_robot_local_xy")
            if sb is None:
                raise RuntimeError("--lid-at-edge requires goal_region."
                                   "support_bounds_robot_local_xy in diagnostics")
            (lx0, ly0), (lx1, ly1) = sb  # robot-local AABB

            r_pos_t, r_quat_t = robot.get_position_orientation()
            r_pos = tuple(float(v) for v in (r_pos_t.cpu()
                          if hasattr(r_pos_t, "cpu") else r_pos_t))
            r_quat = tuple(float(v) for v in (r_quat_t.cpu()
                           if hasattr(r_quat_t, "cpu") else r_quat_t))

            # Build the lid's AABB in robot-local frame by transforming
            # all 8 world AABB corners. The world AABB is axis-aligned
            # in WORLD, so its corners give the tightest local bbox.
            l_lo_t, l_hi_t = lid.aabb
            l_lo = [float(v) for v in (l_lo_t.cpu()
                    if hasattr(l_lo_t, "cpu") else l_lo_t)]
            l_hi = [float(v) for v in (l_hi_t.cpu()
                    if hasattr(l_hi_t, "cpu") else l_hi_t)]
            xs_l, ys_l = [], []
            for x in (l_lo[0], l_hi[0]):
                for y in (l_lo[1], l_hi[1]):
                    for z in (l_lo[2], l_hi[2]):
                        lx, ly, _lz = _world_to_local(r_pos, r_quat, (x, y, z))
                        xs_l.append(float(lx)); ys_l.append(float(ly))
            lid_local_xmin, lid_local_xmax = min(xs_l), max(xs_l)
            lid_local_ymin, lid_local_ymax = min(ys_l), max(ys_l)
            lid_local_half_x = 0.5 * (lid_local_xmax - lid_local_xmin)
            lid_local_cx = 0.5 * (lid_local_xmin + lid_local_xmax)
            lid_local_cy = 0.5 * (lid_local_ymin + lid_local_ymax)

            # Near edge = lx0 (smallest local x, closest to the robot).
            # OVERHANG_FRAC = fraction of the lid that hangs past the edge:
            #   0.0 -> lid flush with the edge, fully on the table
            #         center_x = lx0 + lid_local_half_x
            #   0.5 -> half of the lid past the edge
            #         center_x = lx0
            #   1.0 -> entire lid past the edge
            #         center_x = lx0 - lid_local_half_x
            # Linear interp: center_x = lx0 + (1 - 2*overhang) * half_x.
            overhang = float(args.lid_at_edge)
            target_local_x = lx0 + (1.0 - 2.0 * overhang) * lid_local_half_x
            # Keep current local-y center so we don't drift sideways.
            target_local_y = lid_local_cy

            # Diagnostic: the support's live world AABB vs the placeable
            # region bounds, so we can see how much margin the region
            # has on the near side (placeable lx0 is usually inside the
            # physical desk edge).
            support_name = diagnostics.get("surface", "")
            support_obj = env.scene.object_registry("name", support_name)
            if support_obj is None:
                raise RuntimeError(
                    f"--lid-at-edge: surface {support_name!r} not in scene")
            s_lo_t, s_hi_t = support_obj.aabb
            s_lo_w = [float(v) for v in (s_lo_t.cpu()
                      if hasattr(s_lo_t, "cpu") else s_lo_t)]
            s_hi_w = [float(v) for v in (s_hi_t.cpu()
                      if hasattr(s_hi_t, "cpu") else s_hi_t)]
            xs_s, ys_s = [], []
            for sx in (s_lo_w[0], s_hi_w[0]):
                for sy in (s_lo_w[1], s_hi_w[1]):
                    lxs, lys, _ = _world_to_local(
                        r_pos, r_quat, (sx, sy, s_hi_w[2]))
                    xs_s.append(float(lxs)); ys_s.append(float(lys))
            print(f"[Lid]   support world AABB -> local: "
                  f"x=[{min(xs_s):.3f},{max(xs_s):.3f}] "
                  f"y=[{min(ys_s):.3f},{max(ys_s):.3f}]  "
                  f"(placeable near-edge margin on x: "
                  f"{lx0 - min(xs_s):+.3f} m)", flush=True)

            # Delta in local frame -> world delta via the same transform.
            delta_local = (target_local_x - lid_local_cx,
                           target_local_y - lid_local_cy)
            # _local_xy_to_world returns absolute world xy; we want a
            # delta, so subtract robot pos and undo the rotation by
            # applying _quat_rotate. Simpler: transform two points and
            # subtract.
            wx_a, wy_a = _local_xy_to_world(r_pos, r_quat, 0.0, 0.0)
            wx_b, wy_b = _local_xy_to_world(r_pos, r_quat,
                                            delta_local[0], delta_local[1])
            dwx, dwy = wx_b - wx_a, wy_b - wy_a

            current_pos, current_quat = lid.get_position_orientation()
            current_pos_np = np.asarray(current_pos.cpu()
                                        if hasattr(current_pos, "cpu")
                                        else current_pos, dtype=np.float64)
            new_root = current_pos_np.copy()
            new_root[0] += dwx
            new_root[1] += dwy
            print(f"[Lid] --lid-at-edge: support_bounds_local_xy="
                  f"[({lx0:.3f},{ly0:.3f}),({lx1:.3f},{ly1:.3f})] "
                  f"overhang_frac={overhang:.2f}  (near edge = lx0)", flush=True)
            print(f"[Lid]   lid local AABB: x=[{lid_local_xmin:.3f},"
                  f"{lid_local_xmax:.3f}] y=[{lid_local_ymin:.3f},"
                  f"{lid_local_ymax:.3f}] half_x={lid_local_half_x:.3f}",
                  flush=True)
            print(f"[Lid]   target local center=({target_local_x:.3f},"
                  f"{target_local_y:.3f}); world delta=({dwx:+.3f},{dwy:+.3f})",
                  flush=True)
            print(f"[Lid]   lid root: {current_pos_np.tolist()} -> "
                  f"{new_root.tolist()}", flush=True)
            import torch as th
            lid.set_position_orientation(
                position=th.as_tensor(new_root, dtype=th.float32),
                orientation=current_quat,
            )
            lid.root_link.set_linear_velocity(th.zeros(3))
            lid.root_link.set_angular_velocity(th.zeros(3))
            # Settle physics.
            for _ in range(30):
                og.sim.step()
            # Optional idle pause for visual inspection.
            if args.lid_edge_pause > 0:
                import time as _t
                t_end = _t.time() + args.lid_edge_pause
                print(f"[Lid] --lid-edge-pause: idling {args.lid_edge_pause}s "
                      "for visual inspection ...", flush=True)
                while _t.time() < t_end:
                    og.sim.step()

        # cuRobo primitives.
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        primitives = StarterSemanticActionPrimitives(env, env.robots[0],
                                                     enable_head_tracking=False)

        # LidSnapper — discovers pair(s) at init, fires on release.
        snapper = LidSnapper(env)

        # ------------------------------------------------------------------
        # Multi-attempt loop: snapshot once after edge placement, then for
        # each seed restore state + reseed and retry. Saves the per-attempt
        # env-startup cost. Stop once --max-successes successes collected.
        # ------------------------------------------------------------------
        seeds = _parse_seeds_arg(args)
        multi = args.seeds is not None
        base_out_dir = out_dir
        snapshot = og.sim.dump_state()
        print(f"[Lid] seeds={seeds}  "
              f"max_successes={args.max_successes if multi else 1}",
              flush=True)

        def _attempt_body(result_payload, sft_recorder):
            """Run PHASE 1A → 2B with current ``args.seed``.
            Mutates ``result_payload`` and ``sft_recorder`` in place.
            Early-bail returns are bare ``return`` (None); the success
            flag at the end lives in ``result_payload['phase_2b']['success']``.
            """
            # =====================================================================
            # PHASE 1A — Pick the lid
            # =====================================================================
            print("\n[Lid] === PHASE 1A: pick lid ===", flush=True)
            t0 = time.time()
            pick_deadline_1 = t0 + args.pick_timeout
            home_joint_q = robot.get_joint_positions().clone()
            lid_init_pos, lid_init_quat = lid.get_position_orientation()
            lid_init_pos = lid_init_pos.clone(); lid_init_quat = lid_init_quat.clone()
            grasp_lid, lid_timings = _phase_a_pick(
                env, og, primitives, lid, args, pick_deadline_1,
            )
            result_payload["phase_1a"]["wall_s"] = round(time.time() - t0, 2)
            result_payload["phase_1a"]["cands_tried"] = len(lid_timings)
            if grasp_lid is None:
                print("[Lid] Phase 1A FAILED — no graspable lid", flush=True)
                result_payload["fail_step"] = "phase_1a"
                return
            result_payload["phase_1a"]["held"] = True
            result_payload["phase_1a"]["obj_to_eef"] = float(grasp_lid["obj_to_eef"])

            # SFT: replay successful Phase 1A pick.
            if sft_recorder is not None:
                replay_ok = _record_phase_a_replay(
                    env, og, robot, lid,
                    home_joint_q=home_joint_q,
                    target_init_pos=lid_init_pos,
                    target_init_quat=lid_init_quat,
                    approach_traj=grasp_lid["approach_traj"],
                    sft_recorder=sft_recorder,
                    deadline=time.time() + args.pick_timeout,
                )
                if not replay_ok:
                    result_payload.setdefault("fail_step",
                                              "phase_1a_replay_lost_grip")
                    return

            # =====================================================================
            # PHASE 1B — Transport lid above container
            # =====================================================================
            print("\n[Lid] === PHASE 1B: plan + execute lid → above container ===",
                  flush=True)
            lid_goal_xyz, lid_goal_quat = _lid_goal_world_pose(
                lid, container,
                clearance_z=args.lid_clearance_above_container,
            )
            # Hover-z anchored to the container's world AABB top (+10 cm
            # clearance) so the lid passes above ANY container variant
            # (steamer baskets are taller than canisters). Convert to the
            # relative offset _plan_transport expects:
            #   hover_z_arg = (container_aabb_max_z + 0.10) - lid_now_z.
            import torch as th
            _, c_hi_t = container.aabb
            c_hi_z = float(c_hi_t.cpu()[2] if hasattr(c_hi_t, "cpu") else c_hi_t[2])
            lid_now_z = float(
                lid.get_position_orientation()[0].cpu()[2]
                if hasattr(lid.get_position_orientation()[0], "cpu")
                else lid.get_position_orientation()[0][2]
            )
            hover_z_rel = (c_hi_z + 0.10) - lid_now_z
            # Lift target z MUST match hover target z — otherwise hover has
            # to both translate AND change height in one cuRobo segment,
            # which trajopt struggles with. So lift_z is whatever offset
            # brings the lid to (container_top + 0.10) — possibly negative
            # if the grasped lid is already above the hover height (the
            # "lift" then becomes a small descent to reach hover height,
            # which trajopt handles cleanly).
            lift_z_rel = hover_z_rel
            print(f"[Lid] lid_goal_xyz={lid_goal_xyz.tolist()}  "
                  f"lid_goal_quat={lid_goal_quat.tolist()}", flush=True)
            print(f"[Lid] container.aabb_max_z={c_hi_z:.3f}  lid_now_z="
                  f"{lid_now_z:.3f}  hover_z_rel={hover_z_rel:+.3f}  "
                  f"lift_z_rel={lift_z_rel:+.3f}  "
                  f"(target_hover_z_world={c_hi_z + 0.10:.3f})", flush=True)
            seg_timings_1: list = []
            t_plan_1 = time.time()
            seg_pairs_1 = _plan_transport(
                primitives, robot, lid, lid_goal_xyz,
                transport_timeout=args.transport_timeout,
                lift_z=lift_z_rel,
                hover_z=hover_z_rel,
                goal_target_quat_world=lid_goal_quat,
                skip_descend=True,
                seg_timings=seg_timings_1,
            )
            result_payload["phase_1b"]["plan_wall_s"] = round(time.time() - t_plan_1, 2)
            result_payload["phase_1b"]["plan_segments"] = seg_timings_1
            if seg_pairs_1 is None:
                print("[Lid] Phase 1B PLAN FAILED", flush=True)
                failing = next((s["label"] for s in seg_timings_1 if not s["ok"]), "?")
                result_payload["fail_step"] = f"phase_1b_plan:{failing}"
                return
            result_payload["phase_1b"]["planned"] = True

            # Concatenate the 3 segments' eef-base trajectories.
            import torch as th
            transport_arm_traj_1 = th.cat([arm for _, arm, _ in seg_pairs_1], dim=0)
            replay_deadline_1 = time.time() + max(args.transport_timeout * 4, 240.0)
            t_exec_1 = time.time()
            ok_1 = _replay_holding(
                env, og, robot, lid, transport_arm_traj_1,
                deadline=replay_deadline_1,
                sft_recorder=sft_recorder,
            )
            result_payload["phase_1b"]["execute_wall_s"] = round(
                time.time() - t_exec_1, 2)
            result_payload["phase_1b"]["executed"] = bool(ok_1)
            if not ok_1:
                result_payload.setdefault("fail_step", "phase_1b_execute")

            # =====================================================================
            # PHASE 1C — Release + retreat → LidSnapper attaches
            # =====================================================================
            print("\n[Lid] === PHASE 1C: release + retreat + snap-attach ===",
                  flush=True)
            # collector._try_grasp_candidate leaves gravity DISABLED on the
            # target after a successful Phase 1A grasp (the `finally` after
            # the gravity-hold verification turns it off again so the next
            # candidate starts clean). Re-enable here so the lid actually
            # falls onto the container at release.
            lid.root_link.enable_gravity()
            _release_and_retreat(
                env, og, robot, sft_recorder=sft_recorder,
                retreat_z=args.release_retreat_z,
                n_open_steps=6,
                n_retreat_steps=max(1, args.release_steps - 6),
            )
            # Settle physics so the lid drops onto the container, then run the
            # snapper a few times. Print snap diagnostics at start, midway,
            # and end of the settle window so we can see why it isn't firing
            # without flooding the log.
            from sentinel.utils.lid_attach import find_M_link, find_F_link
            f_link = find_F_link(container,
                                 find_M_link(lid).meta_link_id)
            f_pos_t, _ = f_link.get_position_orientation()
            f_pos_np = np.asarray(f_pos_t.cpu()
                                  if hasattr(f_pos_t, "cpu") else f_pos_t,
                                  dtype=np.float64)
            print(f"[Lid] Phase 1C: container F-link world pos = "
                  f"{f_pos_np.tolist()}", flush=True)
            attached = False
            settle_n = int(args.post_release_settle)
            verbose_steps = {0, settle_n // 4, settle_n // 2,
                             (3 * settle_n) // 4, settle_n - 1}
            for i in range(settle_n):
                og.sim.step()
                verbose = i in verbose_steps
                if verbose:
                    lid_pos_t, _ = lid.get_position_orientation()
                    lid_pos_np = np.asarray(lid_pos_t.cpu()
                                            if hasattr(lid_pos_t, "cpu")
                                            else lid_pos_t, dtype=np.float64)
                    m_link = find_M_link(lid)
                    m_pos_t, _ = m_link.get_position_orientation()
                    m_pos_np = np.asarray(m_pos_t.cpu()
                                          if hasattr(m_pos_t, "cpu") else m_pos_t,
                                          dtype=np.float64)
                    m_to_f = m_pos_np - f_pos_np
                    print(f"[Lid] Phase 1C settle step {i+1}/{settle_n}: "
                          f"lid_root_xyz={lid_pos_np.tolist()}  "
                          f"M-link_xyz={m_pos_np.tolist()}  "
                          f"M-to-F delta=({m_to_f[0]:+.3f},{m_to_f[1]:+.3f},"
                          f"{m_to_f[2]:+.3f})", flush=True)
                snapper.try_snap(robot=robot, verbose=verbose)
                # Authoritative check via OmniGibson's AttachedTo state:
                # try_snap returns the lid name only on the EXACT step the
                # attach fires, but subsequent steps return None (the pair
                # is "already-attached"). Read the state directly so we
                # don't miss the event when the firing step is non-verbose.
                if lid.states[AttachedTo].get_value(container):
                    attached = True
                    print(f"[Lid] Phase 1C: ATTACHED detected at step {i+1}",
                          flush=True)
                    break
            result_payload["phase_1c"]["lid_attached"] = bool(attached)
            if not attached:
                print("[Lid] Phase 1C: LidSnapper did not attach within settle window",
                      flush=True)
                result_payload.setdefault("fail_step", "phase_1c_no_attach")

            # =====================================================================
            # PHASE 2A — Pick the container
            # =====================================================================
            print("\n[Lid] === PHASE 2A: pick container ===", flush=True)
            # Refresh cuRobo obstacle world: now that the lid is attached, we
            # don't want it in the obstacle world (it's effectively part of
            # the container assembly and grasp candidates would otherwise be
            # rejected for "colliding" with the lid that sits on top).
            primitives._motion_generator.update_obstacles(ignore_objects=[lid])

            t1 = time.time()
            pick_deadline_2 = t1 + args.pick_timeout
            home_joint_q_2 = robot.get_joint_positions().clone()
            c_init_pos, c_init_quat = container.get_position_orientation()
            c_init_pos = c_init_pos.clone(); c_init_quat = c_init_quat.clone()
            grasp_c, c_timings = _phase_a_pick(
                env, og, primitives, container, args, pick_deadline_2,
            )
            result_payload["phase_2a"]["wall_s"] = round(time.time() - t1, 2)
            result_payload["phase_2a"]["cands_tried"] = len(c_timings)
            if grasp_c is None:
                print("[Lid] Phase 2A FAILED — no graspable container", flush=True)
                result_payload.setdefault("fail_step", "phase_2a")
                return
            result_payload["phase_2a"]["held"] = True
            result_payload["phase_2a"]["obj_to_eef"] = float(grasp_c["obj_to_eef"])

            if sft_recorder is not None:
                replay_ok = _record_phase_a_replay(
                    env, og, robot, container,
                    home_joint_q=home_joint_q_2,
                    target_init_pos=c_init_pos,
                    target_init_quat=c_init_quat,
                    approach_traj=grasp_c["approach_traj"],
                    sft_recorder=sft_recorder,
                    deadline=time.time() + args.pick_timeout,
                )
                if not replay_ok:
                    result_payload.setdefault("fail_step",
                                              "phase_2a_replay_lost_grip")
                    return

            # =====================================================================
            # PHASE 2B — Transport container to goal
            # =====================================================================
            print("\n[Lid] === PHASE 2B: plan + execute container → goal ===",
                  flush=True)
            seg_timings_2: list = []
            t_plan_2 = time.time()
            seg_pairs_2 = _plan_transport(
                primitives, robot, container, goal_center,
                transport_timeout=args.transport_timeout,
                seg_timings=seg_timings_2,
            )
            result_payload["phase_2b"]["plan_wall_s"] = round(time.time() - t_plan_2, 2)
            result_payload["phase_2b"]["plan_segments"] = seg_timings_2
            if seg_pairs_2 is None:
                print("[Lid] Phase 2B PLAN FAILED", flush=True)
                failing = next((s["label"] for s in seg_timings_2 if not s["ok"]), "?")
                result_payload.setdefault("fail_step", f"phase_2b_plan:{failing}")
                return
            result_payload["phase_2b"]["planned"] = True

            transport_arm_traj_2 = th.cat([arm for _, arm, _ in seg_pairs_2], dim=0)
            replay_deadline_2 = time.time() + max(args.transport_timeout * 4, 240.0)
            t_exec_2 = time.time()
            ok_2 = _replay_holding(
                env, og, robot, container, transport_arm_traj_2,
                deadline=replay_deadline_2,
                sft_recorder=sft_recorder,
            )
            result_payload["phase_2b"]["execute_wall_s"] = round(
                time.time() - t_exec_2, 2)
            result_payload["phase_2b"]["executed"] = bool(ok_2)

            # Final success check: container intersects goal AND robot still
            # holds container AND lid still attached to container.
            c_final_pos, _ = container.get_position_orientation()
            c_final_np = c_final_pos.cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(c_final_np - goal_center))
            intersects = object_intersects_goal_region(container, goal_spec)
            pos_ok, which = target_or_gripper_in_goal(env, container, goal_spec)
            still_held = robot_holds_target(env, container)
            lid_still_attached = bool(
                lid.states[AttachedTo].get_value(container)
                if AttachedTo in lid.states else False
            )
            success = bool(ok_2 and pos_ok and still_held and lid_still_attached)

            result_payload["phase_2b"]["final_container_world"] = c_final_np.tolist()
            result_payload["phase_2b"]["final_container_to_goal_m"] = dist
            result_payload["phase_2b"]["target_intersects_goal"] = bool(intersects)
            result_payload["phase_2b"]["gripper_or_target_in_goal"] = bool(pos_ok)
            result_payload["phase_2b"]["pos_check_which"] = which
            result_payload["phase_2b"]["robot_holds_container"] = bool(still_held)
            result_payload["phase_2b"]["lid_still_attached"] = lid_still_attached
            result_payload["phase_2b"]["success"] = success
            print(f"[Lid] Phase 2B → "
                  f"dist={dist:.3f}m radius={goal_radius:.3f}m "
                  f"pos_ok={pos_ok}(by={which!r}) AG_held={still_held} "
                  f"lid_attached={lid_still_attached} → "
                  f"{'SUCCESS' if success else 'MISS'}", flush=True)
            if not success and "fail_step" not in result_payload:
                if not lid_still_attached:
                    result_payload["fail_step"] = "lid_detached_during_transport"
                elif not still_held:
                    result_payload["fail_step"] = "lost_grip_container"
                elif not pos_ok:
                    result_payload["fail_step"] = "goal_not_intersected"
                else:
                    result_payload["fail_step"] = "unknown"

            # Final breakdown.
            print("\n[Lid] === STEP BREAKDOWN ===", flush=True)
            for k, label in [("phase_1a", "Phase 1A pick lid       "),
                             ("phase_1b", "Phase 1B lid transport  "),
                             ("phase_1c", "Phase 1C release+snap   "),
                             ("phase_2a", "Phase 2A pick container "),
                             ("phase_2b", "Phase 2B container→goal ")]:
                d = result_payload.get(k, {})
                wall = (d.get("wall_s")
                        or (d.get("plan_wall_s", 0) + d.get("execute_wall_s", 0)))
                print(f"[Lid]   {label}: {wall:6.1f}s", flush=True)
            print(f"[Lid]   fail_step               : "
                  f"{result_payload.get('fail_step', '-')}", flush=True)
            print(f"[Lid]   success                 : {success}", flush=True)

        successes = 0
        last_result_payload: dict = {}
        for _attempt_seed in seeds:
            args.seed = int(_attempt_seed)
            attempt_dir = (base_out_dir / f"seed_{_attempt_seed}"
                           if multi else base_out_dir)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[Lid] ========== ATTEMPT seed={_attempt_seed} "
                  f"({successes}/{args.max_successes if multi else 1}) "
                  f"==========", flush=True)
            og.sim.load_state(snapshot)
            og.sim.step()

            result_payload: dict = {
                "task_dir": str(task_dir),
                "seed": int(_attempt_seed),
                "phase_1a": {"held": False},
                "phase_1b": {"planned": False, "executed": False},
                "phase_1c": {"lid_attached": False},
                "phase_2a": {"held": False},
                "phase_2b": {"planned": False, "executed": False,
                             "success": False},
                "lid_name": lid_name,
                "container_name": container_name,
                "goal_center_world": goal_center.tolist(),
                "goal_radius_m": goal_radius,
            }
            sft_recorder = None
            if args.record_sft:
                from tools._sft_recorder import SFTRecorder
                sft_recorder = SFTRecorder(attempt_dir,
                                           resolution=args.record_resolution,
                                           fps=args.video_fps)
                sft_recorder.attach(env, env.robots[0])

            try:
                _attempt_body(result_payload, sft_recorder)
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                print(f"[Lid] attempt seed={_attempt_seed} raised "
                      f"{type(exc).__name__}: {exc}", flush=True)
                _tb.print_exc()
                result_payload.setdefault(
                    "fail_step", f"exception:{type(exc).__name__}")
            finally:
                if sft_recorder is not None:
                    sft_success = bool(result_payload.get(
                        "phase_2b", {}).get("success"))
                    sft_recorder.finalize(success=sft_success, attrs={
                        "task_dir": str(task_dir),
                        "lid_name": str(lid_name),
                        "container_name": str(container_name),
                        "seed": int(args.seed),
                        "phase_1a_held": bool(result_payload.get(
                            "phase_1a", {}).get("held")),
                        "phase_1c_lid_attached": bool(result_payload.get(
                            "phase_1c", {}).get("lid_attached")),
                        "phase_2a_held": bool(result_payload.get(
                            "phase_2a", {}).get("held")),
                    })
                (attempt_dir / "result.json").write_text(
                    json.dumps(result_payload, indent=2))
                print(f"[Lid] wrote {attempt_dir / 'result.json'}", flush=True)
            last_result_payload = result_payload

            if result_payload.get("phase_2b", {}).get("success"):
                successes += 1
                if multi and successes >= args.max_successes:
                    print(f"[Lid] reached --max-successes={args.max_successes}; "
                          f"stopping seed loop.", flush=True)
                    break
        print(f"[Lid] sweep done: {successes} successes across "
              f"{len(seeds)} seeds", flush=True)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[Lid] FAIL: {exc}", flush=True)
        traceback.print_exc()
        raise
