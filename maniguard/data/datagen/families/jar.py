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
            "ride_end": ride_end, "fN": fN}

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
        return self._side_filter(cands)

    def select_grasps(self, ctx: TaskContext, world, robot) -> None:
        self._prepare(ctx)                       # cache the hinge before the variant loop

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
        pre_p = np.asarray(ride_start, float) + 0.04 * np.asarray(fN, float)   # 4cm further below the lid
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
        segs += [
            MotionSegment("side_pre_grasp", pre_grasp, q, mode=Mode.FREE, grip=Grip.OPEN,
                          grip_steps=steps),
            #             jar KEPT in the cuRobo world: a plan that ignores a 13-27cm jar routes
            #             THROUGH it and physically smacks it over (upright violation)
            MotionSegment("descend", np.asarray(grasp.eef_pos, float), q, mode=Mode.LINEAR,
                          grip=Grip.CLOSE, grip_steps=steps, require_attach=True,
                          ignore_objects=(ctx.target_name,), ignore_clutter=True),
            #             require_attach: if AG did not magnetize the jar, fail fast (never carry air)
            MotionSegment("lift", np.asarray(grasp.eef_pos, float) + np.array([0.0, 0.0, 0.07]), q,
                          mode=Mode.SERVO, attach=True, grip=Grip.HOLD, servo_spw=3),
            #             ORIENTATION-LOCKED straight lift (SERVO): the held jar stays upright by
            #             construction. Fixed LOW +7cm (content rides INSIDE the jar, so the generic
            #             clearance check can't be satisfied; jar scenes have no clutter)
            MotionSegment("transport",
                          np.array([goal[0], goal[1], max(grasp.eef_pos[2] + 0.07, goal[2] + 0.03)]),
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
