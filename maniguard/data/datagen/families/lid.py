"""lid_transport family skeleton.

lid_transport = ``pick(lid) -> place onto the container's F meta-link -> release ->
LidSnapper auto-weld -> pick(container body / rim+lid sandwich) -> transport -> END
HOLDING inside the green goal sphere`` (no place-down; teleop termination semantics).

Success = shared GoalRegionChecker (held ∧ AABB∩sphere; assembly-aware held for the
sandwich/lid-grip cases, gated on diag ``lid_info``) ∧ ``success_extra`` (lid still
``AttachedTo`` container). LTL: ``(container_on_support U lid_on_container) &
G(!container_dropped)``.

Three geometry classes drive the lid-place approach:
  plain  (~14 tasks): top knob grasp, vertical descend onto F, vertical retreat.
  handle (kettle/teapot x4): SIDE grasp (annotated), horizontal insertion under the
         overhead handle arch, replay-reverse exit back out of the arch.
  cap    (x8): small cap onto a tall bottle/carton mouth, vertical.

Conventions (audited 2026-07-09): grasp approach axis = **eef +Z** (grasp_select
convention); annotated positions are EEF-LINK poses (fingertips ~0.104 m further along
+Z). Spec: docs/superpowers/specs/2026-07-09-lid-transport-datagen-design.md
"""
from __future__ import annotations

import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.executor.contracts import (
    FamilyAbort,
    FamilySkeleton,
    GraspCand,
    Grip,
    Mode,
    MotionSegment,
    SampleParams,
    SegmentSkip,
    TaskContext,
)
from maniguard.data.datagen.grasp_db import load_db, target_grasps_world

HANDLE_CONTAINERS = frozenset({"kettle", "teapot"})
CAP_CATEGORIES = frozenset({"cap"})

RELEASE_CLEAR_Z_M = 0.008     # release the lid's M-link this far above the F-link (8mm:
#                               15mm drop impulse tips the hypersensitive gqwnfv canister)
PRE_STANDOFF_M = 0.10         # pre-grasp standoff back along the approach axis
LIFT_M = 0.10                 # straight lift after the lid grasp
PLACE_STANDOFF_M = 0.10       # transit standoff before the final place servo
CONT_LIFT_M = 0.08            # fixed lift of the lidded assembly before to_goal (the
#                               clearance-based lift is 0 in lid scenes -- nothing else on
#                               the table -- which reads as dragging the object across it)
ALIGN_TOL_M = 0.012           # pre-release M<->F horizontal alignment gate (a crooked
#                               drop onto a narrow mouth TIPS light canisters: 0008/0010).
#                               12mm: healthy placements measure 9.9-10.7mm (0010), tip-over
#                               drops were the >15mm class — 8mm rejected good seats.
SNAP_MAX_STEPS = 60           # try_snap budget after release before FamilyAbort
SIDE_TILT_MIN_DEG = 60.0      # geometric side-grasp threshold (arch insertion)
FAT_MESH_LIMIT = 32           # cuRobo _attach_objects_to_robot hard cap on collision prims;
#                               containers over it (carton=40) carry with attach=False
#                               (SERVO is collision-off anyway; the arm-only model plans fine)
CONT_STEP_DISP_MAX_M = 0.06   # violent container shove during Phase L (dusty gate)
CONT_TOTAL_DISP_MAX_M = 0.15  # cumulative Phase-L container drift backstop


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, th.Tensor) else np.asarray(x)


def geometry_class(container_category: str, lid_category: str) -> str:
    if container_category in HANDLE_CONTAINERS:
        return "handle"
    if lid_category in CAP_CATEGORIES:
        return "cap"
    return "plain"


def approach_axis(eef_quat_xyzw) -> np.ndarray:
    """World approach direction of a grasp = eef +Z (grasp_select convention)."""
    return Rot.from_quat(np.asarray(eef_quat_xyzw, float)).apply([0.0, 0.0, 1.0])


def approach_tilt_deg(eef_quat_xyzw) -> float:
    """Tilt of the approach axis from straight-down (0 = pure top-down)."""
    a = approach_axis(eef_quat_xyzw)
    return float(np.degrees(np.arccos(np.clip(-a[2], -1.0, 1.0))))


def insertion_dir_from_grasp(eef_quat_xyzw) -> np.ndarray:
    """World horizontal insertion direction = the approach axis projected onto XY.
    Near-vertical approaches have no insertion direction (raises ValueError)."""
    a = approach_axis(eef_quat_xyzw)
    horiz = np.array([a[0], a[1], 0.0])
    n = float(np.linalg.norm(horiz))
    if n < np.sin(np.deg2rad(15.0)):
        raise ValueError("grasp approach is near-vertical: no horizontal insertion dir")
    return horiz / n


def release_pose_for_lid(f_pos, eef_pos, m_pos, clear_z: float = RELEASE_CLEAR_Z_M) -> np.ndarray:
    """eef world target that puts the HELD lid's M-link ``clear_z`` above the F-link
    (rigid grasp => the eef-to-M offset is constant while held)."""
    off = np.asarray(eef_pos, float) - np.asarray(m_pos, float)
    return np.asarray(f_pos, float) + off + np.array([0.0, 0.0, clear_z])


def arch_open_dirs(container_mesh_verts_local: np.ndarray, f_local: np.ndarray,
                   lid_radius: float) -> list[np.ndarray]:
    """For an overhead-handle container: the two horizontal unit directions
    PERPENDICULAR to the handle-arch span (the only corridors a held lid can enter
    through), ordered by corridor clearance (best first), in the CONTAINER's local
    frame. Arch = mesh vertices above the F-link."""
    v = np.asarray(container_mesh_verts_local, float)
    above = v[v[:, 2] > f_local[2] + 0.01]
    if len(above) < 8:
        return []
    xy = above[:, :2] - f_local[:2]
    # principal axis of the arch footprint = the span the arch crosses the mouth along
    cov = np.cov(xy.T)
    w, vec = np.linalg.eigh(cov)
    span = vec[:, int(np.argmax(w))]
    perp = np.array([-span[1], span[0]])
    dirs = []
    for sgn in (1.0, -1.0):
        d = sgn * perp / (np.linalg.norm(perp) + 1e-9)
        # corridor clearance: nearest container vertex to the corridor line (at lid
        # height band around F z), sampled outside the mouth
        pts = np.stack([f_local[:2] + d * r for r in np.linspace(lid_radius,
                                                                 lid_radius + 0.15, 8)])
        band = v[np.abs(v[:, 2] - f_local[2]) < 0.05]
        if len(band) == 0:
            dirs.append((np.inf, d))
            continue
        dmin = min(float(np.min(np.linalg.norm(band[:, :2] - p, axis=1))) for p in pts)
        dirs.append((dmin, d))
    dirs.sort(key=lambda t: -t[0])
    return [np.array([d[1][0], d[1][1], 0.0]) for d in dirs]


class LidSkeleton(FamilySkeleton):
    name = "lid"

    def __init__(self):
        self._db = None
        self._snapper = None
        self._lid_obj = None
        self._lid_grasps: list[GraspCand] = []
        self._gclass = "plain"
        self._cont_home_xy = None
        self._cur_lid_grasp: GraspCand | None = None
        self._AttachedTo = None

    def grasping_mode(self) -> str:
        return "sticky"          # thin lid rims defeat assisted raycast attach (jar evidence)

    # ---- one-time setup ---------------------------------------------------------------
    def select_grasps(self, ctx: TaskContext, world, robot) -> None:
        from omnigibson.object_states import AttachedTo

        from maniguard.data.datagen.executor import grasp_select
        from maniguard.utils.lid_attach import LidSnapper, find_M_link

        self._AttachedTo = AttachedTo
        self._snapper = LidSnapper(ctx.env)
        pair = next((p for p in self._snapper.pairs
                     if p.container.name == ctx.target_name), None)
        if pair is None:
            raise RuntimeError(f"LidSnapper found no (lid, {ctx.target_name}) pair "
                               f"(pairs={[(p.lid.name, p.container.name) for p in self._snapper.pairs]})")
        self._lid_obj = pair.lid
        self._gclass = geometry_class(ctx.target.category, self._lid_obj.category)
        self._cont_home_xy = _np(ctx.target.get_position_orientation()[0])[:2].copy()
        n_coll = sum(len(getattr(l, "collision_meshes", {}) or {})
                     for l in ctx.target.links.values())
        self._fat_target = n_coll > FAT_MESH_LIMIT
        if self._fat_target:
            print(f"[datagen.lid] target has {n_coll} collision prims > {FAT_MESH_LIMIT} "
                  f"-> transport with attach=False (cuRobo attach cap)", flush=True)
        # food-mode tasks: the food RIDES INSIDE the lidded container — it must never
        # count as "other clutter" for clearance, nor as a static cuRobo obstacle while
        # the assembly is attached (it moves with the carry).
        sel = {x.get("role"): x for x in
               (ctx.diagnostics.get("selection", {}) or {}).get("spawn_specs", [])}
        self._food_name = None
        fspec = sel.get("food")
        if fspec:
            self._food_name = next(
                (o.name for o in ctx.env.scene.objects
                 if getattr(o, "category", None) == fspec["category"]
                 and getattr(o, "model", None) == fspec["model"]), None)
        self._m_id = find_M_link(self._lid_obj).meta_link_id

        db = self._db or load_db()
        self._db = db
        lid_key = f"{self._lid_obj.category}/{self._lid_obj.model}"
        lp, lq = ctx.env.scene.object_registry("name", self._lid_obj.name).get_position_orientation()
        gs = target_grasps_world(db, lid_key, _np(lp), _np(lq))
        cands = [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                           approach=g["approach"]) for g in gs]
        if self._gclass == "handle":
            cands = [c for c in cands if approach_tilt_deg(c.eef_quat) >= SIDE_TILT_MIN_DEG]
            if not cands:
                raise RuntimeError(f"{lid_key}: handle-class task has no SIDE lid grasps "
                                   f"(tilt >= {SIDE_TILT_MIN_DEG} deg) in the annotation DB")
        self._arch_dirs = []
        if self._gclass == "handle":
            import json as _json

            import trimesh

            from maniguard.utils.lid_attach import find_F_link
            mesh_db = _json.load(open("outputs/grasp_annotation/mesh_db.json"))["objects"]
            ent = mesh_db.get(ctx.target_key) or {}
            mesh_rel = ent.get("mesh_bare") or ent.get("mesh")
            f = find_F_link(ctx.target, self._m_id)
            cp, cq = (_np(v) for v in ctx.target.get_position_orientation())
            f_local = Rot.from_quat(cq).inv().apply(_np(f.get_position_orientation()[0]) - cp)
            lid_r = float(max(_np(self._lid_obj.aabb_extent)[:2]) / 2.0)
            if mesh_rel:
                m = trimesh.load(f"outputs/grasp_annotation/{mesh_rel}", force="mesh")
                dirs_local = arch_open_dirs(np.asarray(m.vertices), f_local, lid_r)
                yaw = Rot.from_quat(cq)
                self._arch_dirs = [yaw.apply(d) for d in dirs_local]
                self._arch_dirs = [d / (np.linalg.norm(d[:2]) + 1e-9) for d in self._arch_dirs]
                print(f"[datagen.lid] arch open dirs (world): "
                      f"{[list(np.round(d[:2], 2)) for d in self._arch_dirs]}", flush=True)
        scored = grasp_select.score_grasps(world, robot, self._lid_obj, cands)
        self._lid_grasps = [c for c in scored if c.reachable] or scored[:1]
        print(f"[datagen.lid] setup: class={self._gclass} lid={self._lid_obj.name} "
              f"({lid_key}) lid_grasps={len(self._lid_grasps)} reachable="
              f"{sum(c.reachable for c in self._lid_grasps)}", flush=True)

    # ---- container grasp candidates (driver-iterated + scored) -------------------------
    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        db = self._db or load_db()
        self._db = db
        obj_pos, obj_quat = ctx.target.get_position_orientation()
        gs = target_grasps_world(db, ctx.target_key, _np(obj_pos), _np(obj_quat))
        return [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                          approach=g["approach"]) for g in gs]

    def score_drop_extra(self, ctx: TaskContext) -> list:
        # the driver scores container grasps at BOOT (lid still on the table). SANDWICH
        # grasps close over rim+lid TOGETHER — at boot there is no lid there yet, and at
        # execution time the descend ignores both; nothing extra to drop.
        return []

    # ---- the manip skeleton -------------------------------------------------------------
    def derive_segments(self, ctx: TaskContext, grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        rng = np.random.default_rng((int(params.seed) & 0xFFFFFFFF) ^ 0x11D)
        gl = self._lid_grasps[int(rng.integers(len(self._lid_grasps)))]
        self._cur_lid_grasp = gl
        self._cur_ins_dir = None
        if self._gclass == "handle" and self._arch_dirs:
            self._cur_ins_dir = self._arch_dirs[int(params.draw_index) % len(self._arch_dirs)]
        self._cont_disp_accum = 0.0
        self._cont_last_xy = None

        a_l = approach_axis(gl.eef_quat)
        lid_pre = np.asarray(gl.eef_pos, float) - PRE_STANDOFF_M * a_l
        lid_lift = np.asarray(gl.eef_pos, float) + np.array([0.0, 0.0, LIFT_M])
        gq_l = np.asarray(gl.chosen_quat if gl.chosen_quat is not None else gl.eef_quat, float)

        ride_along = tuple(n for n in (self._lid_obj.name, self._food_name) if n)
        a_c = approach_axis(grasp.eef_quat)
        cont_pre = np.asarray(grasp.eef_pos, float) - PRE_STANDOFF_M * a_c
        gq_c = np.asarray(grasp.chosen_quat if grasp.chosen_quat is not None else grasp.eef_quat, float)
        Z = np.zeros(3)

        return [
            # ---- Phase L: lid pick -> place -> release -> weld --------------------------
            MotionSegment("lid_pre", lid_pre, gq_l, mode=Mode.FREE),
            # ignore_clutter: the thin lid lies flat on the support — fingertips end
            # mm above the table, which a support-collision plan always refuses
            # (clutter/dusty descend precedent; physics + AG + LTL gate own execution).
            MotionSegment("lid_descend", np.asarray(gl.eef_pos, float), gq_l,
                          mode=Mode.LINEAR,
                          ignore_objects=(self._lid_obj.name,
                                          getattr(ctx.support, "name", "")),
                          grip=Grip.CLOSE, grip_steps=24),
            # held_name: Phase L carries the LID — without it the engine attaches
            # ctx.target (the CONTAINER) to the cuRobo model: wrong collision body for
            # every plan, and a >32-collision-mesh container (carton=40) trips cuRobo's
            # attach cap outright.
            MotionSegment("lid_lift", lid_lift, gq_l, mode=Mode.SERVO, attach=True,
                          extra={"held_name": self._lid_obj.name}),
            MotionSegment("lid_transit", Z, gq_l, mode=Mode.FREE, attach=True,
                          compute="lid_transit",
                          extra={"held_name": self._lid_obj.name}),
            MotionSegment("lid_place", Z, gq_l, mode=Mode.SERVO, attach=True,
                          compute="lid_place",
                          servo_step_m=0.004, servo_spw=1,
                          extra={"held_name": self._lid_obj.name}),
            MotionSegment("lid_align_verify", Z, gq_l, compute="align_verify"),
            MotionSegment("lid_release", Z, gq_l, mode=Mode.SERVO, attach=True,
                          compute="lid_place", grip=Grip.OPEN, grip_steps=20,
                          servo_step_m=0.004, servo_spw=1,
                          extra={"held_name": self._lid_obj.name}),
            MotionSegment("lid_retreat", Z, gq_l, replay_reverse=True),
            MotionSegment("snap_verify", Z, gq_l, compute="snap_verify"),
            # ---- Phase C: container pick -> transport to the goal sphere ---------------
            MotionSegment("cont_pre", cont_pre, gq_c, mode=Mode.FREE),
            MotionSegment("cont_descend", np.asarray(grasp.eef_pos, float), gq_c,
                          mode=Mode.LINEAR,
                          ignore_objects=(ctx.target_name, self._lid_obj.name),
                          grip=Grip.CLOSE, grip_steps=24),
            MotionSegment("cont_lift",
                          np.asarray(grasp.eef_pos, float) + np.array([0.0, 0.0, CONT_LIFT_M]),
                          gq_c, mode=Mode.SERVO, attach=not self._fat_target,
                          carry_closed=True,
                          min_clearance_m=params.min_clearance_m,
                          ignore_objects=ride_along,
                          clearance_exclude=ride_along),
            MotionSegment("goal_over", Z, gq_c, mode=Mode.SERVO,
                          attach=not self._fat_target, carry_closed=True,
                          compute="goal_over", ignore_objects=ride_along),
            MotionSegment("to_goal", Z, gq_c, mode=Mode.SERVO,
                          attach=not self._fat_target, carry_closed=True,
                          compute="aim_to_goal_center", reach_fallback=True,
                          ignore_objects=ride_along),
        ]

    # ---- per-segment runtime checks --------------------------------------------------------
    def on_segment(self, seg: MotionSegment, ctx: TaskContext) -> None:
        if seg.name == "lid_retreat":
            # EARLY SNAP: weld at first touch, right after the release settle — the
            # crooked-settle window between release and snap_verify is when a slightly
            # off-centre lid pushes light containers over (0010). Best-effort only;
            # snap_verify remains the hard gate.
            import omnigibson as og
            for _ in range(10):
                if self._snapper.try_snap(robot=ctx.robot):
                    break
                og.sim.step()
        if seg.name == "lid_pre":
            # RESET HYGIENE: the attach patch disables lid<->container collision at weld
            # time and that filter OUTLIVES the scene reset — the next attempt's released
            # lid falls THROUGH the container (0012: lid z 0.97 -> floor, not-touching
            # x70). Re-enable the pair at the start of every attempt.
            st = self._lid_obj.states.get(self._AttachedTo)
            if st is not None and ctx.target in getattr(st, "parents_disabled_collisions", set()):
                st.parents_disabled_collisions.discard(ctx.target)
                for cl in self._lid_obj.links.values():
                    for pl in ctx.target.links.values():
                        try:
                            cl.remove_filtered_collision_pair(pl)
                        except Exception:  # noqa: BLE001
                            pass
                print("[datagen.lid] re-enabled lid<->container collision (post-reset)",
                      flush=True)
        # wrong-grab guards (dusty food_grabbed precedent): sticky can attach a NEIGHBOR
        # (0002: the lid grasp closed onto the steamer basket) — verify at the first
        # post-close segment of each phase, abort cleanly so the driver retries.
        if seg.name == "lid_lift" and not self._is_grasped(ctx, self._lid_obj):
            raise FamilyAbort("wrong_grab_lid", held=self._held_name(ctx))
        if seg.name == "cont_lift":
            ok = (self._is_grasped(ctx, ctx.target)
                  or (self._is_grasped(ctx, self._lid_obj)
                      and self._attached(ctx)))          # sandwich: lid-of-assembly counts
            if not ok:
                raise FamilyAbort("wrong_grab_container", held=self._held_name(ctx))

    def _is_grasped(self, ctx: TaskContext, obj) -> bool:
        from omnigibson.controllers.controller_base import IsGraspingState
        return ctx.robot.is_grasping(candidate_obj=obj) == IsGraspingState.TRUE

    def _attached(self, ctx: TaskContext) -> bool:
        try:
            return bool(self._lid_obj.states[self._AttachedTo].get_value(ctx.target))
        except Exception:  # noqa: BLE001
            return False

    def _held_name(self, ctx: TaskContext) -> str:
        arm = ctx.robot.default_arm
        obj = (getattr(ctx.robot, "_ag_obj_in_hand", {}) or {}).get(arm)
        return getattr(obj, "name", "none")

    # ---- runtime-resolved targets --------------------------------------------------------
    def resolve_compute(self, tag: str, seg: MotionSegment, ctx: TaskContext):
        if tag == "lid_transit":
            release, quat = self._live_release_pose(ctx)
            if self._gclass == "handle":
                ins = (self._cur_ins_dir if self._cur_ins_dir is not None
                       else insertion_dir_from_grasp(self._cur_lid_grasp.eef_quat))
                target = release - PLACE_STANDOFF_M * ins
            else:
                target = release + np.array([0.0, 0.0, PLACE_STANDOFF_M])
            return target, quat
        if tag == "lid_place":
            self._check_cont_disp(ctx, seg.name)
            release, quat = self._live_release_pose(ctx)
            return release, quat
        if tag == "goal_over":
            from maniguard.data.datagen.executor import geometry
            pos, quat = geometry.aim_to_center_eef(ctx.robot, ctx.target,
                                                   np.asarray(ctx.goal_center, float))
            eef_z = float(_np(ctx.robot.eef_links[ctx.robot.default_arm]
                              .get_position_orientation()[0])[2])
            target = np.array([float(pos[0]), float(pos[1]), eef_z])
            return target, np.asarray(quat, float)
        if tag == "align_verify":
            from maniguard.utils.lid_attach import find_F_link, find_M_link
            f = find_F_link(ctx.target, self._m_id)
            m = find_M_link(self._lid_obj)
            dxy = (_np(m.get_position_orientation()[0])[:2]
                   - _np(f.get_position_orientation()[0])[:2])
            err = float(np.linalg.norm(dxy))
            if err > ALIGN_TOL_M:
                raise FamilyAbort("misaligned", err_m=round(err, 4))
            raise SegmentSkip(seg.name)
        if tag == "snap_verify":
            self._check_cont_disp(ctx, seg.name)
            self._run_snap(ctx)
            raise SegmentSkip(seg.name)
        raise ValueError(f"LidSkeleton has no resolve_compute for tag {tag!r}")

    def _live_release_pose(self, ctx: TaskContext):
        from maniguard.utils.lid_attach import find_F_link, find_M_link
        f = find_F_link(ctx.target, self._m_id)
        m = find_M_link(self._lid_obj)
        arm = ctx.robot.default_arm
        eef_pos, eef_quat = (_np(v) for v in
                             ctx.robot.eef_links[arm].get_position_orientation())
        f_pos = _np(f.get_position_orientation()[0])
        m_pos = _np(m.get_position_orientation()[0])
        # handle class with a chosen arch corridor: rotate the LIVE grip about world z
        # so the approach azimuth lines up with the corridor (round lids -> the regrip
        # is invisible); the eef->M offset rotates with it.
        off = eef_pos - m_pos
        quat = eef_quat
        if self._gclass == "handle" and self._cur_ins_dir is not None:
            a = Rot.from_quat(eef_quat).apply([0.0, 0.0, 1.0])
            az_now = float(np.arctan2(a[1], a[0]))
            az_want = float(np.arctan2(self._cur_ins_dir[1], self._cur_ins_dir[0]))
            rz = Rot.from_euler("z", az_want - az_now)
            off = rz.apply(off)
            quat = (rz * Rot.from_quat(eef_quat)).as_quat()
        target = np.asarray(f_pos, float) + off + np.array([0.0, 0.0, RELEASE_CLEAR_Z_M])
        return target, np.asarray(quat, float)

    def _run_snap(self, ctx: TaskContext) -> None:
        import os

        import omnigibson as og
        dbg = bool(os.environ.get("DATAGEN_DEBUG_STATE"))
        for i in range(SNAP_MAX_STEPS):
            self._snapper.try_snap(robot=ctx.robot,
                                   verbose=dbg and i >= SNAP_MAX_STEPS - 10)
            og.sim.step()
            try:
                if self._lid_obj.states[self._AttachedTo].get_value(ctx.target):
                    print(f"[datagen.lid] snap_verify: ATTACHED after {i + 1} steps", flush=True)
                    return
            except KeyError:
                pass
        raise FamilyAbort("snap_fail", steps=SNAP_MAX_STEPS)

    def _check_cont_disp(self, ctx: TaskContext, seg_name: str) -> None:
        xy = _np(ctx.target.get_position_orientation()[0])[:2]
        total = float(np.linalg.norm(xy - self._cont_home_xy))
        step = (float(np.linalg.norm(xy - self._cont_last_xy))
                if self._cont_last_xy is not None else 0.0)
        self._cont_last_xy = xy.copy()
        # NOTE: no ``seg=`` in FamilyAbort detail — the engine injects the current
        # segment name itself (a duplicate kwarg crashes the conversion).
        if step > CONT_STEP_DISP_MAX_M:
            raise FamilyAbort("container_displaced",
                              disp_m=round(step, 4), kind="violent")
        if total > CONT_TOTAL_DISP_MAX_M:
            raise FamilyAbort("container_displaced",
                              disp_m=round(total, 4), kind="cumulative")

    # ---- verdicts ------------------------------------------------------------------------
    def success_extra(self, ctx: TaskContext) -> bool:
        try:
            ok = bool(self._lid_obj.states[self._AttachedTo].get_value(ctx.target))
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            print("[datagen.lid] success_extra: lid NOT attached at end -> fail", flush=True)
        return ok

    def debug_state(self, ctx: TaskContext) -> str:
        lp = _np(self._lid_obj.get_position_orientation()[0])
        cp = _np(ctx.target.get_position_orientation()[0])
        try:
            att = bool(self._lid_obj.states[self._AttachedTo].get_value(ctx.target))
        except Exception:  # noqa: BLE001
            att = False
        gd = float(np.linalg.norm(cp - np.asarray(ctx.goal_center, float)))
        return (f"lid={lp.round(3)} cont={cp.round(3)} attached={att} "
                f"goal_d={gd:.3f} class={self._gclass}")
