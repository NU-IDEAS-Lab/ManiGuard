"""Clutter family skeleton — the ONLY clutter-specific code.

clutter = ``grasp(target) -> transport(target -> goal)`` via the boxy top-down waypoints
(§4.1/§4.3). It implements ``FamilySkeleton`` and nothing else: where the grasps come from
(the annotation DB) and how to turn a chosen grasp + goal into the ordered MotionSegment
list. It never touches cuRobo / execution / the gate / the recorder — the generic
``executor`` owns those.

The "boxy" sequence (each line = one MotionSegment)::

    pre_grasp   FREE,   open    standoff back along the grasp approach axis
    descend     LINEAR, close   straight in to the annotated grasp pose, close on the object
    lift        LINEAR, hold    straight up until the held object's lowest point clears the
                                tallest clutter by >= min_clearance  (compute=lift_to_clearance)
    transport   FREE,   hold    over the goal at the cleared height           (compute=over_goal)
    to_goal     LINEAR, hold    drive the held object's CENTRE to the goal-sphere centre
                                (compute=aim_to_goal_center; the §4.3 endpoint redundancy)

The clearance lift, over-goal move and aim-to-centre target are RUNTIME-resolved by the
engine from the live post-grasp state (``compute`` tags), so ``derive_segments`` stays a
pure function of (grasp, params, goal) — no sim reads, trivially unit-testable.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.executor.contracts import (
    FamilySkeleton, GraspCand, Grip, Mode, MotionSegment, SampleParams, TaskContext,
)
from maniguard.data.datagen.grasp_db import load_db, target_grasps_world

WORLD_UP = np.array([0.0, 0.0, 1.0])


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


class ClutterSkeleton(FamilySkeleton):
    name = "clutter"

    def __init__(self, db: dict | None = None, *, grip_settle_steps: int = 6):
        self._db = db
        self.grip_settle_steps = int(grip_settle_steps)

    # --- where grasps come from: the annotation DB, transformed to world via live pose ---
    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        db = self._db if self._db is not None else load_db()
        obj_pos, obj_quat = ctx.target.get_position_orientation()
        world = target_grasps_world(db, ctx.target_key, _np(obj_pos), _np(obj_quat))
        return [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                          approach=g["approach"]) for g in world]

    # --- the boxy skeleton: (grasp, params, goal) -> ordered MotionSegments (pure) ---
    def derive_segments(self, ctx: TaskContext, grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        R = Rot.from_quat(grasp.eef_quat).as_matrix()
        approach_dir = R[:, 2]                                 # eef +Z world = fingertips toward object
        approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)
        q = np.asarray(grasp.eef_quat, float)

        # pre-grasp = standoff back along the approach axis (+ optional lateral jitter for diversity)
        dx, dy = params.jitter.get("above_xy", (0.0, 0.0))
        pre_pos = grasp.eef_pos - params.standoff_m * approach_dir + np.array([dx, dy, 0.0])

        steps = self.grip_settle_steps
        goal = np.asarray(ctx.goal_center, float)

        return [
            MotionSegment("pre_grasp", pre_pos, q, mode=Mode.FREE,
                          grip=Grip.OPEN, grip_steps=steps,
                          ignore_objects=(ctx.target_name,)),
            MotionSegment("descend", np.asarray(grasp.eef_pos, float), q, mode=Mode.LINEAR,
                          grip=Grip.CLOSE, grip_steps=steps,
                          ignore_objects=(ctx.target_name,), ignore_clutter=True),
            MotionSegment("lift", np.asarray(grasp.eef_pos, float), q, mode=Mode.LINEAR,
                          attach=True, grip=Grip.HOLD, compute="lift_to_clearance",
                          min_clearance_m=params.min_clearance_m,                       # verify floor (3cm)
                          target_clearance_m=params.min_clearance_m * params.lift_clearance_mult),
            MotionSegment("transport", np.array([goal[0], goal[1], grasp.eef_pos[2]]), q,
                          mode=Mode.FREE, attach=True, grip=Grip.HOLD, compute="over_goal"),
            MotionSegment("to_goal", goal.copy(), q, mode=Mode.LINEAR,
                          attach=True, grip=Grip.HOLD, compute="aim_to_goal_center"),
        ]
