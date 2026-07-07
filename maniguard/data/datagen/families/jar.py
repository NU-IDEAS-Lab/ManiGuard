"""Jar family skeleton — the ONLY jar-specific code.

jar_transport = ``close_lid(hinge) -> grasp(jar_body, side) -> transport(jar -> goal)``. Phase A
closes the articulated lid by pivoting the OPEN gripper about the hinge axis (a chain of short SERVO
arc segments resolved by the family ``arc_about_hinge`` compute tag), past the vertical tipping
point, then retreats so gravity seats the lid at the ``lower_limit`` (closed). Phase B is the clutter
boxy skeleton restricted to SIDE grasps (keeps the jar upright, off the just-closed lid). The generic
executor plans / executes / gates / records everything; the engine is unchanged."""
from __future__ import annotations

import numpy as np

from maniguard.data.datagen.executor.contracts import (
    FamilySkeleton, GraspCand, Grip, Mode, MotionSegment, SampleParams, TaskContext,
)
from maniguard.data.datagen.families import jar_hinge as JH
from maniguard.data.datagen.grasp_db import load_db, target_grasps_world


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, float)


class JarSkeleton(FamilySkeleton):
    name = "jar"

    def __init__(self, db: dict | None = None, *, grip_settle_steps: int = 6):
        self._db = db
        self.grip_settle_steps = int(grip_settle_steps)
        self._h: dict = {}                       # per-target cached hinge plan (also the unit-test seam)

    # --- per-task hinge read + arc plan (idempotent; a unit test may pre-populate self._h) ---
    def _prepare(self, ctx: TaskContext) -> None:
        if ctx.target_name in self._h:
            return
        hf = JH.read_hinge(ctx.target)
        e = hf.ext_dir                                        # TRUE radial extension (from lid_quat, robust)
        # LID-RIDE (the user's teleop maneuver), parameterized from the lid link's MEASURED hull:
        # a finger bar in the wedge UNDER the flopped lid, then ONE straight fixed-orientation ride
        # that lifts the lid past its tipping point — the lid rests on the bar under gravity and
        # pivots about its own hinge, the contact sliding freely (unilateral: nothing drags the jar).
        base_p = _np(ctx.robot.get_position_orientation()[0])
        hull = _np(JH.lid_link(ctx.target).collision_boundary_points_world)
        lid_c = hull.mean(axis=0)
        s = 1.0 if float(np.dot(hf.axis, base_p - lid_c)) >= 0.0 else -1.0
        # adaptive bar elevation: on SHORT jars the under-lid wedge sits near the desk and the
        # wrist/palm of the default 12deg-below-the-lid bar lands inside the desk's collision
        # inflation (IK "fails" with tiny pose errors = collision-rejected). Raise the bar toward
        # the lid line until the LOWEST approach point clears the desk by a safe margin.
        from maniguard.data.datagen.executor import geometry as G
        desk_top = float(G.surface_top_z(ctx.support)) if ctx.support is not None else -np.inf
        # a support with raised parts (task_0014's desk has a privacy divider) reports its AABB top,
        # 30cm above the actual sitting plane, and the palm-floor checks then veto EVERY ride pose.
        # The jar's own bottom IS the sitting plane — clamp to it (identical on flat supports).
        jar_bottom = float(G.aabb_lo_hi(ctx.target)[0][2])
        desk_top = min(desk_top, jar_bottom)
        if ctx.support is not None:
            slo, shi = G.aabb_lo_hi(ctx.support)
            desk_xy = (float(slo[0]) - 0.02, float(slo[1]) - 0.02,
                       float(shi[0]) + 0.02, float(shi[1]) + 0.02)
        else:
            desk_xy = None
        ride_start = ride_q = ride_end = fN = None
        for open_deg in (12.0, 8.0, 5.0, 2.0):
            ride_start, ride_q, ride_end, fN = JH.ride_plan(hf.anchor, hf.axis, e, hull, s,
                                                            open_deg=open_deg)
            pre_z = float(ride_start[2] + 0.04 * fN[2])          # the lowest pose = the 4cm-below pre
            if pre_z >= desk_top + 0.09:
                if open_deg != 12.0:
                    print(f"[datagen.jar] shallow wedge: bar raised to {open_deg:.0f}deg below the lid "
                          f"(pre_z={pre_z:.3f} desk={desk_top:.3f})", flush=True)
                break
        self._h[ctx.target_name] = {
            "hf": hf, "e": e, "ride_start": ride_start, "ride_q": ride_q,
            "ride_end": ride_end, "fN": fN, "hull": hull, "s": s, "desk_top": desk_top,
            "desk_xy": desk_xy}

    # --- Phase B grasps: side grasps of the jar body, from the annotation DB ---
    @staticmethod
    def _side_filter(cands: list[GraspCand]) -> list[GraspCand]:
        from maniguard.data.datagen.executor.grasp_select import is_top_down
        side = [c for c in cands if not is_top_down(c.eef_quat)]
        return side if side else cands           # never yield zero (fall back to all)

    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        db = self._db if self._db is not None else load_db()
        obj_pos, obj_quat = ctx.target.get_position_orientation()
        world = target_grasps_world(db, ctx.target_key, _np(obj_pos), _np(obj_quat))
        cands = [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                           approach=g["approach"]) for g in world]
        cands = self._side_filter(cands)
        ok = (self._h.get(ctx.target_name) or {}).get("goal_ok_ids")
        if ok:                                    # keep only grasps whose transport endpoint IK'd (set
            kept = [c for c in cands if c.id in ok]   # in select_grasps); never filter down to zero
            if kept:
                return kept
        return cands

    def score_margin_floor(self) -> float:
        # 0.15 rad (~8.6deg): each jar has only a handful of SIDE grasps, and edge placements can
        # land the best one at ~0.17 — the shared 0.2 floor then yields 0 attempts (task_0025 after
        # its yaw surgery). Still above the wrist-at-limit regime the floor guards against.
        return 0.15

    def select_grasps(self, ctx: TaskContext, world, robot) -> None:
        self._prepare(ctx)                       # geometric default first
        import torch as th
        from maniguard.data.datagen.primitives.curobo_seg import solve_ik, solve_segment
        h = self._h[ctx.target_name]
        hf, e = h["hf"], h["e"]
        q0 = robot.get_joint_positions()
        world.update_obstacles()
        # --- ride-pose variant ladder: extreme placements (far / high / flat lid) leave the default
        # orientation 18-48deg outside the wrist envelope (IK pos_err~mm, rot_err huge). The wrist
        # ROLL about the bar is FREE (finger plane stays ⊥ lid plane), so IK-test both roll branches
        # across the elevation ladder and keep the FIRST feasible pre-pose. ---
        chosen = None
        ax_u = np.asarray(hf.axis, float) / (np.linalg.norm(np.asarray(hf.axis, float)) + 1e-12)
        w_rob = h["s"] * ax_u                                # unit toward the robot side
        for skew in (0.0, 20.0, 40.0):
            for open_deg in (12.0, 8.0, 5.0, 2.0):
                for roll, bar in ((False, False), (True, False), (False, True), (True, True)):
                    rs, rq, re_, fN = JH.ride_plan(hf.anchor, hf.axis, e, h["hull"], h["s"],
                                                   open_deg=open_deg, roll_flip=roll, bar_flip=bar,
                                                   skew_deg=skew)
                    pre = np.asarray(rs, float) + 0.10 * w_rob + 0.02 * np.asarray(fN, float)
                    dxy = h.get("desk_xy")
                    over = dxy is None or (dxy[0] <= float(pre[0]) <= dxy[2]
                                           and dxy[1] <= float(pre[1]) <= dxy[3])
                    if over and float(pre[2]) < h["desk_top"] + 0.05:   # palm floor: only OVER the
                        continue                                        # support (edge-overhang = air)
                    res = None
                    for _try in range(2):                      # stochastic solver — retry once.
                        # screen with the SAME solve as the real lid_under (collision-aware FREE
                        # plan, jar in-world): what passes here actually plans at execution
                        res = solve_segment(
                            world.motion_gen, robot, th.as_tensor(pre, dtype=th.float32),
                            th.as_tensor(np.asarray(rq, float), dtype=th.float32), q0, timeout=3.0,
                            label=f"ride:s{skew:.0f}:{open_deg:.0f}:r{int(roll)}b{int(bar)}")
                        if res is not None:
                            break
                    if res is not None:
                        chosen = (rs, rq, re_, fN, open_deg, roll, bar, skew)
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen:
            rs, rq, re_, fN, od, roll, bar, skew = chosen
            h.update({"ride_start": rs, "ride_q": rq, "ride_end": re_, "fN": fN})
            print(f"[datagen.jar] ride variant: open_deg={od:.0f} roll_flip={roll} bar_flip={bar} "
                  f"skew={skew:.0f}", flush=True)
        else:
            print("[datagen.jar] ride variant: NONE feasible, keeping geometric default", flush=True)
        # --- goal-endpoint margin DIAGNOSTIC (env-gated, OFF by default; NO filtering). As a hard
        # filter this dropped the empirically-winning grasp on 5/11 passing tasks (endpoint margin
        # is measured on ONE IK branch; the transport has configuration freedom, so low margin does
        # NOT predict failure — task_0001 succeeded first-try with a 0.01-margin grasp). Gated even
        # as a log: the solves perturb the flaky solver's RNG stream and marginal tasks (0009) roll
        # different dice — keep the default call sequence identical to the validated one. ---
        import os
        if os.environ.get("DATAGEN_DIAG_GOAL_MARGINS") != "1":
            return
        from maniguard.data.datagen.executor.grasp_select import joint_margin
        goal = np.asarray(ctx.goal_center, float)
        arm_idx = robot.arm_control_idx[robot.default_arm]
        lo_lim = np.asarray(robot.joint_lower_limits)[arm_idx]
        hi_lim = np.asarray(robot.joint_upper_limits)[arm_idx]
        margins = {}
        for c in self.grasp_candidates(ctx):
            zt = max(float(c.eef_pos[2]) + 0.07, goal[2] + 0.03)
            res = solve_ik(world.motion_gen, robot,
                           th.as_tensor(np.array([goal[0], goal[1], zt]), dtype=th.float32),
                           th.as_tensor(np.asarray(c.eef_quat, float), dtype=th.float32), q0,
                           timeout=2.0, ik_collision=False, label=f"goal_ik:g{c.id}")
            if res is not None:
                q = res.arm_traj[-1].detach().cpu().numpy().reshape(-1)
                margins[c.id] = joint_margin(q, lo_lim, hi_lim)
        if margins:
            print(f"[datagen.jar] goal-endpoint margins (diagnostic): "
                  f"{ {g: round(m, 2) for g, m in sorted(margins.items())} }", flush=True)
        else:
            print("[datagen.jar] goal-endpoint: NONE IK-viable (diagnostic)", flush=True)

    # --- the skeleton: Phase A (close lid) then Phase B (side-grasp transport) ---
    def derive_segments(self, ctx: TaskContext, grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        from scipy.spatial.transform import Rotation as Rot
        self._prepare(ctx)
        h = self._h[ctx.target_name]
        ride_start, ride_q, ride_end, fN = h["ride_start"], h["ride_q"], h["ride_end"], h["fN"]
        steps = self.grip_settle_steps

        # --- Phase A: LID-RIDE — finger bar under the lid, one straight fixed-orientation ride
        # lifts the lid past its tipping point; gravity closes the rest; retreat retraces the ride.
        # The gripper stays OPEN and never closes on the lid (no coupling that could drag the jar). ---
        # standoff style ALTERNATES per attempt for zero regression on already-passing tasks:
        #   even draws -> the ORIGINAL 4cm-below-the-lid pre (validated by the 20/26 sweep);
        #   odd draws  -> sideways 10cm toward the robot ALONG the hinge axis (exits cuRobo's obstacle
        #                 inflation without diving toward the desk) — the extra way out for tasks whose
        #                 below-the-lid pre sits inside the inflation ("IK_FAIL" with mm pose errors)
        if params.draw_index % 2 == 0:
            pre_p = np.asarray(ride_start, float) + 0.04 * np.asarray(fN, float)
        else:
            ax = np.asarray(h["hf"].axis, float)
            w_rob = h["s"] * ax / (np.linalg.norm(ax) + 1e-12)
            pre_p = (np.asarray(ride_start, float) + 0.10 * w_rob + 0.02 * np.asarray(fN, float))
        segs: list[MotionSegment] = [
            MotionSegment("lid_under", pre_p, ride_q, mode=Mode.FREE, grip=Grip.OPEN,
                          grip_steps=steps,    # jar KEPT in the cuRobo world: route AROUND the lid
                          rot_relax=np.radians(10.0), pos_relax=0.008),   # ride pose is tolerance-
            #             insensitive (a rolled/offset support bar lifts the same); some jar poses
            #             sit 2-6deg past cuRobo's strict IK threshold at the workspace edge
            MotionSegment("lid_slip", np.asarray(ride_start, float), ride_q, mode=Mode.SERVO,
                          grip=Grip.HOLD, ignore_objects=(ctx.target_name,), servo_spw=2,
                          compute="ride_delta",
                          extra={"delta": (np.asarray(ride_start, float) - pre_p)}),
            MotionSegment("lid_ride", np.asarray(ride_end, float), ride_q, mode=Mode.SERVO,
                          grip=Grip.HOLD, ignore_objects=(ctx.target_name,),
                          servo_spw=5, servo_step_m=0.004,    # slow, fine: the lid rides the bar
                          compute="ride_delta",
                          extra={"delta": (np.asarray(ride_end, float) - np.asarray(ride_start, float))}),
            #             ride_delta: translate by the DESIGNED vector from the LIVE arrived pose,
            #             keeping the arrived orientation — the ride mechanics tolerate the relaxed
            #             lid_under arrival (+-10deg), but an ABSOLUTE re-target does not (strict
            #             per-waypoint IK can't absorb the residual -> "stuck at the pre-close spot")
            MotionSegment("lid_back", np.asarray(ride_start, float), ride_q, mode=Mode.SERVO,
                          grip=Grip.HOLD, replay_reverse=True),   # retrace the ride path back out
        ]

        # --- Phase B: side-grasp transport (the clutter boxy skeleton) ---
        R_g = Rot.from_quat(grasp.eef_quat).as_matrix()
        appr = R_g[:, 2] / (np.linalg.norm(R_g[:, 2]) + 1e-9)
        q = np.asarray(grasp.eef_quat, float)
        pre_grasp = np.asarray(grasp.eef_pos, float) - params.standoff_m * appr
        goal = np.asarray(ctx.goal_center, float)
        # close-in goals leave no elbow room for a straight carry at grasp height (the servo IK dies
        # mid-line near the base pillar and the cuRobo upright-hold fallback is flaky in this fork) —
        # carry HIGHER when the goal sits close to the robot base
        close_in = False
        if ctx.robot is not None:
            base_xy = _np(ctx.robot.get_position_orientation()[0])[:2]
            close_in = float(np.linalg.norm(goal[:2] - base_xy)) < 0.44   # 0.48 would also flip task_0023
            #                                                              (goal_d .45) which PASSES at 7cm
        lift_dz = 0.14 if close_in else 0.07
        segs += [
            MotionSegment("side_pre_grasp", pre_grasp, q, mode=Mode.FREE, grip=Grip.OPEN,
                          grip_steps=steps),
            #             jar KEPT in the cuRobo world: a plan that ignores a 13-27cm jar routes
            #             THROUGH it and physically smacks it over (upright violation)
            MotionSegment("descend", np.asarray(grasp.eef_pos, float), q, mode=Mode.LINEAR,
                          grip=Grip.CLOSE, grip_steps=steps, require_attach=True,
                          ignore_objects=(ctx.target_name,), ignore_clutter=True),
            #             require_attach: if AG did not magnetize the jar, fail fast (never carry air)
            MotionSegment("lift", np.asarray(grasp.eef_pos, float) + np.array([0.0, 0.0, lift_dz]), q,
                          mode=Mode.SERVO, attach=True, grip=Grip.HOLD, servo_spw=3),
            #             ORIENTATION-LOCKED straight lift (SERVO): the held jar stays upright by
            #             construction. Fixed LOW +7cm (content rides INSIDE the jar, so the generic
            #             clearance check can't be satisfied; jar scenes have no clutter)
            MotionSegment("transport",
                          np.array([goal[0], goal[1], max(grasp.eef_pos[2] + lift_dz, goal[2] + 0.03)]),
                          q, mode=Mode.SERVO, attach=True, grip=Grip.HOLD, servo_spw=3,
                          free_fallback=True),
            #             ORIENTATION-LOCKED straight carry, z aligned to the GOAL height (shelf goals
            #             sit far above the grasp height); if the straight line can't reach, the engine
            #             falls back to cuRobo WITH the UPRIGHT_HOLD constraint
            MotionSegment("to_goal", goal.copy(), q, mode=Mode.SERVO, attach=True,
                          grip=Grip.HOLD, servo_spw=3, compute="aim_to_goal_center",
                          free_fallback=True),
        ]
        return segs

    def debug_state(self, ctx: TaskContext) -> str:
        """Live lid joint angle + jar root height + finger-vs-disc-plane signed distances + live eef
        axes (DATAGEN_DEBUG_STATE) — shows whether the lid tracks, whether the jar is dragged, whether
        the fingers actually STRADDLE the disc (opposite signs) and whether the arm tracks the roll."""
        from scipy.spatial.transform import Rotation as Rot
        hf = JH.read_hinge(ctx.target)
        root_z = float(_np(ctx.target.get_position_orientation()[0])[2])
        out = (f"lid={np.degrees(hf.angle):.1f}deg (closed={JH.is_closed(hf.angle, hf.lower, hf.upper)}) "
               f"jar_z={root_z:.3f}")
        f = JH.face_normal(hf.axis, hf.ext_dir)
        try:
            dl, dr = (float(np.dot(_np(ctx.robot.links[n].get_position_orientation()[0]) - hf.anchor, f))
                      for n in ("panda_leftfinger", "panda_rightfinger"))
            arm = ctx.robot.default_arm
            eq = _np(ctx.robot.eef_links[arm].get_position_orientation()[1])
            M = Rot.from_quat(eq).as_matrix()
            out += (f" | finger_dplane L={dl * 100:+.1f}cm R={dr * 100:+.1f}cm "
                    f"({'STRADDLE' if dl * dr < 0 else 'SAME-SIDE'}) "
                    f"| eefZ={np.round(M[:, 2], 2)} eefY={np.round(M[:, 1], 2)}")
        except Exception as ex:  # noqa: BLE001
            out += f" | dbg_err={ex}"
        return out

    def resolve_compute(self, tag: str, seg: MotionSegment, ctx: TaskContext):
        from scipy.spatial.transform import Rotation as Rot
        h = self._h[ctx.target_name]
        if tag == "arc_about_hinge":
            hf = JH.read_hinge(ctx.target)                            # live anchor/axis (jar body may micro-shift)
            if int(seg.extra.get("k", 0)) == 0:
                # arc base = the LIVE settled eef pose right after the grasp (NOT the nominal grasp
                # pose): AG attached wherever the eef actually settled, and rotating the nominal pose
                # would drag the rigidly-held lid by that constant offset — hauling the light jar.
                arm = ctx.robot.default_arm
                bp, bq = (_np(v) for v in ctx.robot.eef_links[arm].get_position_orientation())
                h["arc_base"] = (bp, bq)
            bp, bq = h["arc_base"]
            return JH.rotate_pose_about_axis(bp, bq, hf.anchor, hf.axis, float(seg.extra["theta"]))
        arm = ctx.robot.default_arm
        ep, eq = (_np(v) for v in ctx.robot.eef_links[arm].get_position_orientation())
        if tag == "hold":                                             # in-place gripper action (release)
            return ep, eq
        if tag == "ride_delta":                                       # translate by the designed delta
            return ep + np.asarray(seg.extra["delta"], float), eq     # from the LIVE pose (orientation
        #                                                               kept as-arrived; ride-tolerant)
        if tag == "lid_retreat":
            back = Rot.from_quat(eq).as_matrix()[:, 2]                # eef +Z (∥ hinge axis, roll-invariant)
            return ep - JH.RETREAT_M * back, eq                       # slide back out along the axis
        raise ValueError(f"JarSkeleton has no resolve_compute for tag {tag!r}")
