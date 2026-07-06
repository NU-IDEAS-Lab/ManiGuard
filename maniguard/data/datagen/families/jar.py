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
        e = JH.lid_extension_dir(hf.anchor, hf.axis, hf.lid_tip)
        insert_p, insert_q = JH.insert_pose(hf.anchor, hf.axis, hf.lid_tip)
        hints = ctx.diagnostics.get("datagen_hints") or {}
        margin_deg = float(hints.get("close_margin_deg", JH.CLOSE_MARGIN_DEG))
        total = JH.arc_close_angle(e, hf.axis, np.radians(margin_deg))
        self._h[ctx.target_name] = {
            "hf": hf, "e": e, "insert_p": insert_p, "insert_q": insert_q, "total": total}

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
        hf, insert_p, insert_q, total = h["hf"], h["insert_p"], h["insert_q"], h["total"]
        steps = self.grip_settle_steps

        # --- Phase A: close the lid ---
        approach = Rot.from_quat(insert_q).as_matrix()[:, 2]          # eef +Z, down the lid
        pre_p = insert_p - JH.LID_STANDOFF_M * approach                # standoff back along approach
        segs: list[MotionSegment] = [
            MotionSegment("lid_pre", pre_p, insert_q, mode=Mode.FREE, grip=Grip.OPEN,
                          grip_steps=steps, ignore_objects=(ctx.target_name,)),
            MotionSegment("lid_insert", np.asarray(insert_p, float), insert_q, mode=Mode.SERVO,
                          grip=Grip.OPEN, ignore_objects=(ctx.target_name,), servo_spw=2),
        ]
        wps = JH.arc_waypoints(insert_p, insert_q, hf.anchor, hf.axis, total)
        for k, (theta, pos_k, quat_k) in enumerate(wps):
            segs.append(MotionSegment(
                f"lid_arc_{k:03d}", np.asarray(pos_k, float), np.asarray(quat_k, float),
                mode=Mode.SERVO, grip=Grip.OPEN, compute="arc_about_hinge",
                ignore_objects=(ctx.target_name,), servo_spw=4, extra={"theta": float(theta)}))
        segs.append(MotionSegment("lid_retreat", np.asarray(insert_p, float), insert_q,
                                  mode=Mode.SERVO, grip=Grip.OPEN, compute="lid_retreat"))

        # --- Phase B: side-grasp transport (the clutter boxy skeleton) ---
        R_g = Rot.from_quat(grasp.eef_quat).as_matrix()
        appr = R_g[:, 2] / (np.linalg.norm(R_g[:, 2]) + 1e-9)
        q = np.asarray(grasp.eef_quat, float)
        pre_grasp = np.asarray(grasp.eef_pos, float) - params.standoff_m * appr
        goal = np.asarray(ctx.goal_center, float)
        segs += [
            MotionSegment("side_pre_grasp", pre_grasp, q, mode=Mode.FREE, grip=Grip.OPEN,
                          grip_steps=steps, ignore_objects=(ctx.target_name,)),
            MotionSegment("descend", np.asarray(grasp.eef_pos, float), q, mode=Mode.LINEAR,
                          grip=Grip.CLOSE, grip_steps=steps,
                          ignore_objects=(ctx.target_name,), ignore_clutter=True),
            MotionSegment("lift", np.asarray(grasp.eef_pos, float), q, mode=Mode.LINEAR,
                          attach=True, grip=Grip.HOLD, compute="lift_to_clearance",
                          min_clearance_m=params.min_clearance_m,
                          target_clearance_m=params.min_clearance_m * params.lift_clearance_mult),
            MotionSegment("transport", np.array([goal[0], goal[1], grasp.eef_pos[2]]), q,
                          mode=Mode.FREE, attach=True, grip=Grip.HOLD, compute="over_goal",
                          reach_fallback=True),
            MotionSegment("to_goal", goal.copy(), q, mode=Mode.LINEAR, attach=True,
                          grip=Grip.HOLD, compute="aim_to_goal_center"),
        ]
        return segs

    def resolve_compute(self, tag: str, seg: MotionSegment, ctx: TaskContext):
        from scipy.spatial.transform import Rotation as Rot
        h = self._h[ctx.target_name]
        if tag == "arc_about_hinge":
            hf = JH.read_hinge(ctx.target)                            # live anchor/axis (jar body may micro-shift)
            return JH.rotate_pose_about_axis(h["insert_p"], h["insert_q"], hf.anchor, hf.axis,
                                             float(seg.extra["theta"]))
        if tag == "lid_retreat":
            arm = ctx.robot.default_arm
            ep, eq = (_np(v) for v in ctx.robot.eef_links[arm].get_position_orientation())
            back = Rot.from_quat(eq).as_matrix()[:, 2]                # eef +Z approach -> retreat opposite
            return ep - JH.RETREAT_M * back, eq
        raise ValueError(f"JarSkeleton has no resolve_compute for tag {tag!r}")
