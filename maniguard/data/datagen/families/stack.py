"""Stack-retrieve family skeleton — the ONLY stack-specific manip code.

``stack_retrieve`` = unstack the 3 IDENTICAL top objects onto ONE right-side re-stack pile, then
retrieve the exposed bottom target left into the goal sphere (held). Each object is moved by a
"double-gate": lift to a FIXED safe transfer height ``H_safe`` → translate over the pick point →
cuRobo-descend to the grasp → close (sticky) → lift back to ``H_safe`` → translate over the dest pile
→ descend onto the growing pile → release. Only the two grasp descents use cuRobo (``Mode.FREE``);
every transfer is pure-IK (``Mode.SERVO``). The target tail reuses clutter's ``over_goal`` +
``aim_to_goal_center`` and keeps the gripper CLOSED through the goal (success = ``held_intersection``).

Per-task geometry (stack order, ``z_top0``, ``dest_xy``, per-instance grasps) is captured once in
``select_grasps``/``_prepare`` (needs the live world — filled in Task 4). ``derive_segments`` is a PURE
function of that captured state + the target grasp + the variant params. ``resolve_compute`` (Task 5)
resolves the 3 family compute tags — ``safe_up`` (raise to H_safe, live eef), ``over_dest`` (held CENTRE
over dest at H_safe), ``lower_to_dest_pile`` (descend onto the live dest-pile top) — from the live state;
the ``over``/``descend`` waypoints are absolute (the captured grasp pose is valid every pristine variant).

Spec/plan: docs/superpowers/{specs,plans}/2026-07-01-stack-retrieve-datagen.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.executor.contracts import (
    FamilySkeleton, GraspCand, Grip, Mode, MotionSegment, SampleParams, TaskContext,
)
from maniguard.data.datagen.grasp_db import load_db, target_grasps_world


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


@dataclass
class _StackItem:
    """One relocated stack instance, captured on the pristine scene by ``_prepare`` (Task 4)."""

    name: str
    obj: object = None
    grasp_pos: np.ndarray | None = None    # world eef grasp position (3,)
    grasp_quat: np.ndarray | None = None   # world eef grasp orientation xyzw (4,)
    pre_pos: np.ndarray | None = None      # pre-grasp standoff position (3,)
    descend_pos: np.ndarray | None = None  # SHALLOW FREE-descend target (grasp_pos retracted along
    #                                        -approach so the pick clears the object below); None => grasp_pos
    below_obj: object = None               # the instance/target directly UNDER this one when it is grasped
    upright_quat: np.ndarray | None = None  # the object's ABSOLUTE upright world orientation (grasp-DB
    #                                        upright_orientation_xyzw; verified true upright — opening +Z —
    #                                        for bowls). NOT the spawn pose (a nested stack spawns tilted).
    place_quat: np.ndarray | None = None   # eef orientation that holds THIS object upright for the re-stack
    #                                        place (measured in-hand at reorient); None => keep the grasp quat


class StackSkeleton(FamilySkeleton):
    name = "stack"

    PLACE_GAP = 0.01          # drop-height above the current dest-pile top before release
    CLEARANCE_BASE = 0.08     # H_safe term 1: max(z_top0, max grasp eef z) + this. Keeps a margin above the
    #                           grasp poses themselves; fine when the grasp eef sits well above the pack top.
    FINGER_MARGIN = 0.04      # H_safe term 2 (the real guarantee): the LOWEST gripper part (fingertips/rail,
    #                           via gripper_drop_below_eef) must clear the pack top z_top0 by this. Term 1 alone
    #                           constrains only eef_link, so for a RIM grasp of a TALL object (nested bowls:
    #                           grasp eef only ~2cm above the rim) the open fingers hang at rim height and sweep
    #                           the stack over during the H_safe transit. Term 2 lifts H_safe until the whole
    #                           gripper clears — plates/boxes already satisfy it, so only tall rim-grasps rise.
    GAP = 0.065               # 5-8cm separation between the pack's right edge and the re-stack pile
    REACH_MAX = 0.85          # coarse Franka horizontal reach clamp for the dest (bring-up refines)
    REACH_COMFORT = 0.72      # Fix 4: pull the dest toward the robot until within this reach (narrow/arc
    #                           table) so the pure-IK carry can actually solve (REACH_MAX passed servo-IK-fail)
    GENTLE_STEP = 0.004       # SERVO eef step for the lift: small step + spw=1 => slow continuous glide, so
    GENTLE_SPW = 1            # peeling the top object off a stack doesn't jerk/tip the ones below (nested bowls)
    SHALLOW_MARGIN = 0.004    # geometric shallow-grab: clear the object BELOW by this (m) ...
    SHALLOW_CONTACT_TOL = 0.012  # ... while still contacting the grasped TOP object within this (m) ...
    SHALLOW_D_MAX = 0.05      # ... retracting at most this far along -approach (else raise: gripper too thick)
    MULTIGRAB_XY = 0.05       # a pick that moves >1 object by more than this (m, xy) grabbed the one below
    CARRY_IK_TIMEOUT_S = 2.0  # per-grasp downstream carry-reachability IK probe timeout (Fix 3)

    def __init__(self, db: dict | None = None, *, grip_settle_steps: int = 6):
        self._db = db
        self.grip_settle_steps = int(grip_settle_steps)
        # per-task state, populated by _prepare/select_grasps (Task 4)
        self._prepared_for: str | None = None
        self._stack: list[_StackItem] = []
        self._z_top0: float = 0.0
        self._dest_xy: np.ndarray | None = None
        self._stack_half: float = 0.0
        # multi-grab acceptance gate (per-attempt; reset on the round-0 up segment)
        self._round_baseline: dict[int, dict] = {}
        self._multigrab: bool = False
        self._logged_h_safe: bool = False     # one H_safe log line per task

    # --- family mode overrides ---
    def grasping_mode(self) -> str:
        return "sticky"                       # flat/wide slabs (plates, boards) can't force-close

    def relocate_prefer_top_down(self) -> bool:
        return True                           # everything is picked straight down

    def score_drop_extra(self, ctx: TaskContext) -> list:
        """The bottom target is scored by the driver while still BURIED under the stack. It is grasped
        LAST (after all 3 stack objects are relocated), so drop them for its reachability scoring — else
        a fully-covered target (nested same-mode stack) scores 0-reachable and no variants are sampled."""
        self._prepare(ctx)
        return [it.obj for it in self._stack]

    # --- multi-grab acceptance gate (FAMILY-SPECIFIC, not the shared safety monitor) ---
    def _obj_xy(self, ctx: TaskContext) -> dict:
        """{name: xy} of the 3 stack instances + the bottom target, live."""
        objs = [it.obj for it in self._stack] + [ctx.target]
        return {getattr(o, "name", f"o{k}"): _np(o.get_position_orientation()[0])[:2]
                for k, o in enumerate(objs)}

    def _measure_place_quat(self, i: int, ctx: TaskContext) -> None:
        """Measure instance i's in-hand rotation NOW (held over the dest) and set the eef orientation that
        would hold it UPRIGHT: eef_place = R_up * (R_obj^-1 * R_eef). Rigid grasp => R_obj^-1*R_eef is the
        fixed eef-in-object rotation, so releasing at eef_place lands the object at its world upright R_up."""
        it = self._stack[i]
        if it.upright_quat is None:
            it.place_quat = np.asarray(it.grasp_quat, float)      # fallback: no DB upright => keep grasp pose
            return
        arm = ctx.robot.default_arm
        _, eq = ctx.robot.eef_links[arm].get_position_orientation()
        _, oq = it.obj.get_position_orientation()
        R_eo = Rot.from_quat(_np(oq)).inv() * Rot.from_quat(_np(eq))   # eef-in-object (fixed by the grasp)
        it.place_quat = (Rot.from_quat(np.asarray(it.upright_quat, float)) * R_eo).as_quat()

    def on_segment(self, seg: MotionSegment, ctx: TaskContext) -> None:
        """Snapshot object positions at each round's ``s{i}_up`` and, at the matching ``s{i}_place``,
        flag the demo if the pick moved MORE THAN ONE object (it grabbed the one below). The round-0
        up also resets the gate, so it clears between attempts on a reused skeleton."""
        name = getattr(seg, "name", "") or ""
        if not name.startswith("s") or "_" not in name:
            return                                          # target (t_*) + non-round segments: ignore
        prefix, kind = name.split("_", 1)
        if not prefix[1:].isdigit():
            return
        i = int(prefix[1:])
        if kind == "up":
            if i == 0:                                      # first segment of a rollout -> reset per-attempt
                self._multigrab = False
                self._round_baseline = {}
            self._round_baseline[i] = self._obj_xy(ctx)
        elif kind == "reorient":
            self._measure_place_quat(i, ctx)                # capture the upright-hold eef pose over the dest
        elif kind == "place":
            base = self._round_baseline.get(i)
            if not base:
                return
            now = self._obj_xy(ctx)
            moved = sum(1 for nm, xy0 in base.items()
                        if nm in now and float(np.linalg.norm(now[nm] - xy0)) > self.MULTIGRAB_XY)
            if moved > 1:
                self._multigrab = True

    def success_extra(self, ctx: TaskContext) -> bool:
        """Reject the demo if any pick this rollout moved >1 object (contaminated multi-grab)."""
        return not self._multigrab

    # --- grasp sourcing: the BOTTOM target's DB grasps -> world, TOP-DOWN ONLY (clutter + a filter) ---
    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        from maniguard.data.datagen.executor.grasp_select import is_top_down
        db = self._db if self._db is not None else load_db()
        obj_pos, obj_quat = ctx.target.get_position_orientation()
        world = target_grasps_world(db, ctx.target_key, _np(obj_pos), _np(obj_quat))
        cands = [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                           approach=g["approach"]) for g in world]
        # stack is a straight-down family — drop any side grasps (the user may annotate side poses on some
        # objects for diversity; here they must never be picked). Geometric test, not the stored hint.
        td = [c for c in cands if is_top_down(c.eef_quat)]
        return td if td else cands

    # --- per-task state capture (once, on the pristine scene) ---
    @staticmethod
    def _stack_specs(diag: dict) -> set:
        """The relocated stack objects' ``(category, model)`` from ``selection.spawn_specs`` (role=stack)."""
        sel = diag.get("selection") or {}
        return {(s.get("category"), s.get("model")) for s in sel.get("spawn_specs", [])
                if s.get("role") == "stack" and s.get("category") and s.get("model")}

    def _prepare(self, ctx: TaskContext) -> None:
        """Enumerate the 3 stack instances (top→down), the fixed transfer height, the stack footprint,
        and the right-side re-stack destination. Idempotent; grasp poses are filled in select_grasps."""
        if self._prepared_for == ctx.target_name:
            return
        from maniguard.data.datagen.executor import geometry as G
        from maniguard.data.datagen.families import stack_geom as SG
        from maniguard.data.datagen.families import stack_grasp_depth as SD

        env, diag = ctx.env, ctx.diagnostics
        want = self._stack_specs(diag)
        insts = [o for o in env.scene.objects
                 if (getattr(o, "category", None), getattr(o, "model", None)) in want
                 and getattr(o, "name", None) != ctx.target_name]
        insts.sort(key=lambda o: G.top_z(o), reverse=True)          # topmost first = removed first
        if len(insts) != 3:
            raise ValueError(f"stack: expected 3 stack instances, found {len(insts)} "
                             f"for {sorted(want)} (names={[getattr(o, 'name', '?') for o in insts]})")

        four = insts + [ctx.target]
        aabbs = [G.aabb_lo_hi(o) for o in four]
        self._z_top0 = SG.transfer_height(aabbs)
        pack_lo, pack_hi = SG.combined_xy_aabb(aabbs)
        lo0, hi0 = G.aabb_lo_hi(insts[0])                            # stack footprint half (larger horiz axis)
        self._stack_half = 0.5 * float(max(hi0[0] - lo0[0], hi0[1] - lo0[1]))

        goal_xy = np.asarray(ctx.goal_center, float)[:2]
        pack_c = 0.5 * (pack_lo + pack_hi)
        right = pack_c - goal_xy                                     # goal is LEFT of the pack -> pack-goal points RIGHT
        n = float(np.linalg.norm(right))
        if n < 1e-6:                                                 # degenerate (goal ~over pack): robot-frame right
            _, rq = ctx.robot.get_position_orientation()
            yaw = float(Rot.from_quat(_np(rq)).as_euler("zyx")[0])
            right = np.array([np.sin(yaw), -np.cos(yaw)])            # -Y_base (robot's right) in world
        else:
            right = right / n

        si = diag.get("surface_info") or {}
        b = si.get("bounds_xy")
        surf_lo = np.asarray(b[0], float) if b else pack_lo - 1.0
        surf_hi = np.asarray(b[1], float) if b else pack_hi + 1.0
        robot_xy = _np(ctx.robot.get_position_orientation()[0])[:2]
        self._right = right                                         # source->dest unit dir (Fix 1/3/4)
        self._dest_geom = (pack_lo, pack_hi, surf_lo, surf_hi, robot_xy)   # replayed to finalise the dest
        # Fix 4: pull the dest toward the robot until within REACH_COMFORT (0 for normal wide tables where
        # the un-pulled dest is already close). Estimated from the un-pulled provisional dest.
        base = SG.dest_center(pack_lo, pack_hi, right, self._stack_half, gap=self.GAP, surf_lo_xy=surf_lo,
                              surf_hi_xy=surf_hi, robot_xy=robot_xy, reach_max=self.REACH_MAX,
                              rail_half=SD.rail_half(), eef_off=self._stack_half)
        self._pull_robot_ward = (0.0 if base is None
                                 else max(0.0, float(np.linalg.norm(base - robot_xy)) - self.REACH_COMFORT))
        # PROVISIONAL dest with the worst-case rail reach (eef_off=stack_half) — used by score_drop_extra's
        # reachability probe; select_grasps re-finalises it from the CHOSEN grasps' actual eef offsets.
        dest = SG.dest_center(pack_lo, pack_hi, right, self._stack_half, gap=self.GAP,
                              surf_lo_xy=surf_lo, surf_hi_xy=surf_hi, robot_xy=robot_xy,
                              reach_max=self.REACH_MAX, rail_half=SD.rail_half(), eef_off=self._stack_half,
                              pull_robot_ward=self._pull_robot_ward)
        if dest is None:
            raise ValueError(f"stack: no on-table + in-reach re-stack destination for {ctx.target_name} "
                             f"(pack right edge + {self.GAP}m gap exceeds surface/reach) — report this task")
        self._dest_xy = dest
        self._stack = [_StackItem(name=getattr(o, "name", f"stk_{i}"), obj=o) for i, o in enumerate(insts)]
        self._logged_h_safe = False
        self._prepared_for = ctx.target_name

    def select_grasps(self, ctx: TaskContext, world, robot) -> None:
        """Score each stack instance's TOP-DOWN grasps once (cuRobo reachability) and cache the best
        reachable world grasp pose on its ``_StackItem`` (the driver only scores the target grasp)."""
        from maniguard.data.datagen.executor.grasp_select import is_top_down, score_grasps
        from maniguard.data.datagen.executor import geometry as G
        from maniguard.data.datagen.families import stack_grasp_depth as SD

        self._prepare(ctx)
        db = self._db if self._db is not None else load_db()
        shallow_ds: list[float] = []
        for i, it in enumerate(self._stack):
            above = [self._stack[j].obj for j in range(i)]     # removed before instance i is grasped
            key = f"{it.obj.category}/{it.obj.model}"
            op, oq = it.obj.get_position_orientation()
            wg = target_grasps_world(db, key, _np(op), _np(oq))
            cands = [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                               approach=g["approach"]) for g in wg if is_top_down(g["eef_quat"])]
            # multi-drop: score in instance i's WILL-BE-EXPOSED state (drop i itself + the above ones,
            # gone by then) — score_grasps drops every object in the list target from the collision world.
            scored = score_grasps(world, robot, [it.obj, *above], cands, prefer_top_down=True)
            # Fix 3: among the pick-reachable grasps, PREFER the DEST-side ones (shorter carry -> easier
            # transfer IK), then take the first whose CARRY to over-dest is ALSO IK-reachable (bounded).
            reach = [c for c in scored if c.reachable] or scored
            c_xy = G.object_center(it.obj)[:2]
            reach.sort(key=lambda cc: float(np.dot(_np(cc.eef_pos)[:2] - c_xy, self._right)), reverse=True)
            best = reach[0] if reach else None
            for cc in reach[:4]:                               # cap the downstream IK probes
                if self._carry_reachable(ctx, world, robot, cc):
                    best = cc
                    break
            if best is None:
                raise ValueError(f"stack: no top-down grasp for stack instance {it.name} ({key})")
            it.grasp_pos = _np(best.eef_pos)
            it.grasp_quat = _np(best.chosen_quat if best.chosen_quat is not None else best.eef_quat)
            uq = (db["objects"].get(key) or {}).get("upright_orientation_xyzw")
            it.upright_quat = _np(uq) if uq is not None else None   # ABSOLUTE upright target for the place

            # geometric shallow-grab: retract the FREE descend along THIS pose's approach axis so the
            # gripper clears the object directly below (else the sticky long fingers grab/drag two).
            it.below_obj = self._stack[i + 1].obj if i + 1 < len(self._stack) else ctx.target
            below_key = (ctx.target_key if it.below_obj is ctx.target
                         else f"{it.below_obj.category}/{it.below_obj.model}")
            R = Rot.from_quat(it.grasp_quat).as_matrix()
            approach = R[:, 2]                                  # eef +Z = approach (grasp_obb: approach=Z)
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = it.grasp_pos
            bp, bq = it.below_obj.get_position_orientation()
            d = SD.instance_descend_offset(key, T, approach, below_key, (_np(bp), _np(bq)),
                                           (_np(op), _np(oq)), margin=self.SHALLOW_MARGIN,
                                           contact_tol=self.SHALLOW_CONTACT_TOL, d_max=self.SHALLOW_D_MAX)
            if d is None:
                raise ValueError(f"stack: no shallow grasp for {it.name} ({key}) — gripper thicker than "
                                 f"the stack gap above {below_key} (raise the annotated grasp or skip task)")
            it.descend_pos = it.grasp_pos - float(d) * (approach / float(np.linalg.norm(approach)))
            shallow_ds.append(round(float(d), 3))

        # Fix 1: finalise the dest from the CHOSEN grasps' ACTUAL eef offset toward the source. The rail
        # only clips group1 when a grasp points the eef (hence the rail) at it; a wide / centre grasp keeps
        # eef_off~0 => offset falls back to stack_half => the dest is NOT over-pushed (task_0000 restored).
        from maniguard.data.datagen.executor import geometry as G
        from maniguard.data.datagen.families import stack_geom as SG
        eef_off = 0.0
        for it in self._stack:
            R_obj = Rot.from_quat(_np(it.obj.get_position_orientation()[1]))       # object world orient
            off_obj = R_obj.inv().apply(np.asarray(it.grasp_pos, float) - G.object_center(it.obj))
            R_up = (Rot.from_quat(np.asarray(it.upright_quat, float))
                    if it.upright_quat is not None else R_obj)                     # eef offset at upright place
            off_place_xy = R_up.apply(off_obj)[:2]
            eef_off = max(eef_off, float(-np.dot(off_place_xy, self._right)))      # component toward source
        pack_lo, pack_hi, surf_lo, surf_hi, robot_xy = self._dest_geom
        dest = SG.dest_center(pack_lo, pack_hi, self._right, self._stack_half, gap=self.GAP,
                              surf_lo_xy=surf_lo, surf_hi_xy=surf_hi, robot_xy=robot_xy,
                              reach_max=self.REACH_MAX, rail_half=SD.rail_half(), eef_off=eef_off,
                              pull_robot_ward=self._pull_robot_ward)
        if dest is not None:
            self._dest_xy = dest                                                  # else keep the provisional
        gzs = [round(float(np.asarray(it.grasp_pos)[2]), 3) for it in self._stack]
        print(f"[datagen.stack] prepared: {len(self._stack)} instances top->down, "
              f"z_top0={self._z_top0:.3f}, grasp_eef_z={gzs}, shallow_d={shallow_ds}, eef_off={eef_off:.3f}, "
              f"dest_xy={np.round(self._dest_xy, 3)}, stack_half={self._stack_half:.3f}", flush=True)

    # --- the double-gate skeleton: (captured state, target grasp, params) -> MotionSegments (pure) ---
    def derive_segments(self, ctx: TaskContext, grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        self._safe_dz = float(params.jitter.get("safe_dz", 0.0))   # jitter added to the LIVE z_safe
        self._logged_h_safe = False
        gap = self.PLACE_GAP
        steps = self.grip_settle_steps
        dxy = np.asarray(self._dest_xy, float)
        h_ph = self._z_top0 + 0.3        # placeholder z for stored over/dest waypoints (every one is a
        #                                  compute segment, so resolve_compute overrides this with live z)
        segs: list[MotionSegment] = []
        for i, it in enumerate(self._stack):
            segs += self._gate_relocate(i, it, h_ph, gap, steps, dxy)
        segs += self._gate_target(ctx, grasp, h_ph, steps)
        return segs

    def _gate_relocate(self, i, it, h_ph, gap, steps, dxy) -> list[MotionSegment]:
        gpos = np.asarray(it.grasp_pos, float)
        gq = np.asarray(it.grasp_quat, float)
        # SHALLOW descend: stop short along -approach so the gripper clears the object below (Task 3);
        # None => not computed (pure unit tests) => full grasp depth.
        dpos = np.asarray(it.descend_pos, float) if it.descend_pos is not None else gpos.copy()
        over = np.array([gpos[0], gpos[1], h_ph])
        dest_over = np.array([dxy[0], dxy[1], h_ph])
        gz = float(gpos[2])
        ph = {"gz": gz, "gq": gq}                          # phase grasp info for the LIVE z_safe
        phi = {"inst": i, **ph}                            # + held-object index (attached segments)
        return [
            MotionSegment(f"s{i}_up", gpos.copy(), gq, mode=Mode.SERVO, grip=Grip.OPEN,
                          grip_steps=steps, compute="safe_up", extra=dict(ph)),
            MotionSegment(f"s{i}_over", over, gq, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=steps,
                          compute="over_at_hsafe", extra=dict(ph)),
            MotionSegment(f"s{i}_descend", dpos, gq, mode=Mode.FREE, grip=Grip.CLOSE,
                          grip_steps=steps, ignore_clutter=True),
            MotionSegment(f"s{i}_lift", gpos.copy(), gq, mode=Mode.SERVO, grip=Grip.HOLD,
                          attach=True, compute="safe_up", extra=dict(phi),
                          servo_step_m=self.GENTLE_STEP, servo_spw=self.GENTLE_SPW),
            MotionSegment(f"s{i}_carry", dest_over, gq, mode=Mode.SERVO, grip=Grip.HOLD,
                          attach=True, compute="over_dest", extra=dict(phi)),
            # reorient the held object to its UPRIGHT world pose over the dest (so every stack object is
            # released at the SAME orientation — identical concentric+upright objects then nest/stack
            # together instead of perching mis-aligned and toppling).
            MotionSegment(f"s{i}_reorient", dest_over, gq, mode=Mode.SERVO, grip=Grip.HOLD,
                          attach=True, compute="reorient_place", extra=dict(phi)),
            # AFTER reorient, re-align the (now upright) object CENTRE onto dest_xy — the reorient rotation
            # shifts the centre, so aligning before it (in carry) leaves ~cm of drift. Doing it here makes
            # every stack object's xy ABSOLUTELY coincident, so the place below is a pure vertical descent.
            MotionSegment(f"s{i}_realign", dest_over, gq, mode=Mode.SERVO, grip=Grip.HOLD,
                          attach=True, compute="align_xy", extra=dict(phi)),
            MotionSegment(f"s{i}_place", dest_over, gq, mode=Mode.SERVO, grip=Grip.OPEN,
                          grip_steps=steps, attach=True, compute="lower_to_dest_pile",
                          extra={"inst": i, "gap": gap}),
        ]

    def _gate_target(self, ctx, grasp, h_ph, steps) -> list[MotionSegment]:
        tpos = np.asarray(grasp.eef_pos, float)
        tq = np.asarray(grasp.eef_quat, float)
        over = np.array([tpos[0], tpos[1], h_ph])
        goal = np.asarray(ctx.goal_center, float)
        ph = {"gz": float(tpos[2]), "gq": tq}              # target-phase grasp info for the LIVE z_safe
        return [
            MotionSegment("t_up", tpos.copy(), tq, mode=Mode.SERVO, grip=Grip.OPEN,
                          grip_steps=steps, compute="safe_up", extra=dict(ph)),
            MotionSegment("t_over", over, tq, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=steps,
                          compute="over_at_hsafe", extra=dict(ph)),
            MotionSegment("t_descend", tpos.copy(), tq, mode=Mode.FREE, grip=Grip.CLOSE,
                          grip_steps=steps, ignore_clutter=True),
            MotionSegment("t_lift", tpos.copy(), tq, mode=Mode.SERVO, grip=Grip.HOLD,
                          attach=True, compute="safe_up", extra=dict(ph)),
            # target TRANSPORT to the goal: require a CLEAN cuRobo solve (no colliding salvage) with more
            # seeds — a winding salvaged path would knock the just-built re-stack pile (task_0021 unsafe).
            # (t_place, the final LINEAR descent INTO the goal sphere, KEEPS salvage: its partial-pose query
            # is a known-buggy cuRobo path that relies on the endpoint-tol recovery to place at all.)
            MotionSegment("t_transport", np.array([goal[0], goal[1], tpos[2]]), tq, mode=Mode.FREE,
                          grip=Grip.HOLD, attach=True, compute="over_goal", reach_fallback=True,
                          no_salvage=True, plan_tries=8),
            MotionSegment("t_place", goal.copy(), tq, mode=Mode.LINEAR, grip=Grip.HOLD,
                          attach=True, compute="aim_to_goal_center"),
        ]

    # --- per-phase LIVE transfer height (drops as objects are removed; unblocks tall stacks) ---
    def _live_h_safe(self, ctx: TaskContext, held, grasp_z, drop) -> float:
        """H_safe from the LIVE scene: clear the tallest OTHER object (held excluded) by ``drop`` (the
        lowest point that hangs below the eef — the gripper, or, when carrying, the held object)."""
        from maniguard.data.datagen.executor import geometry as G
        from maniguard.data.datagen.families import stack_geom as SG
        st = G.surface_top_z(ctx.support)
        other, _ = G.max_other_top_z(ctx.env, exclude=[held] if held is not None else [],
                                     robots=ctx.env.robots, support_top=st)
        if not np.isfinite(other):
            other = st if st is not None else 0.0
        h = SG.live_h_safe(other, float(grasp_z), float(drop),
                           clearance=self.CLEARANCE_BASE, finger_margin=self.FINGER_MARGIN)
        return h + float(getattr(self, "_safe_dz", 0.0))

    def _seg_z_safe(self, seg: MotionSegment, ctx: TaskContext) -> float:
        from maniguard.data.datagen.executor import geometry as G
        from maniguard.data.datagen.families import stack_grasp_depth as SD
        x = seg.extra
        drop = SD.gripper_drop_below_eef(np.asarray(x["gq"], float))   # gripper's lowest point below eef
        held = None
        if seg.attach:                                         # holding an object -> exclude it from the max
            inst = x.get("inst")                               # AND lift IT above the others: it hangs BELOW
            held = self._stack[int(inst)].obj if inst is not None else ctx.target   # the gripper fingertips
            ep_z = float(_np(ctx.robot.eef_links[ctx.robot.default_arm].get_position_orientation()[0])[2])
            drop = max(drop, ep_z - float(G.lowest_z(held)))   # held object's bottom below eef (rigid grasp)
        z = self._live_h_safe(ctx, held, x["gz"], drop)
        if not self._logged_h_safe:
            print(f"[datagen.stack] H_safe(live)={z:.3f} @ {seg.name} (drop={drop:.3f})", flush=True)
            self._logged_h_safe = True
        return z

    def _carry_reachable(self, ctx: TaskContext, world, robot, cand) -> bool:
        """Fix 3: is the CARRY (hold this grasp's object over the dest @ H_safe) IK-reachable? A pick that
        can't be transferred to the dest is worthless — reject it before it wastes an attempt. Bounded
        single-pose IK probe (uses the PROVISIONAL dest; the finalised dest is within ~cm)."""
        from maniguard.data.datagen.primitives.curobo_seg import solve_ik
        from maniguard.data.datagen.families import stack_grasp_depth as SD
        import torch as th
        quat = np.asarray(cand.chosen_quat if cand.chosen_quat is not None else cand.eef_quat, float)
        gz = float(_np(cand.eef_pos)[2])
        zc = self._live_h_safe(ctx, None, gz, SD.gripper_drop_below_eef(quat))
        dxy = np.asarray(self._dest_xy, float)
        res = solve_ik(world.motion_gen, robot,
                       th.tensor([dxy[0], dxy[1], zc], dtype=th.float32),
                       th.tensor(quat, dtype=th.float32), robot.get_joint_positions(),
                       timeout=self.CARRY_IK_TIMEOUT_S, ik_collision=False, label=f"carrycheck:{cand.id}")
        return res is not None

    # --- family compute tags resolved from the LIVE state (engine owns over_goal / aim_to_goal_center) ---
    def resolve_compute(self, tag: str, seg: MotionSegment, ctx: TaskContext):
        from maniguard.data.datagen.executor import geometry as G
        from maniguard.data.datagen.families import stack_geom as SG

        arm = ctx.robot.default_arm
        ep, eq = ctx.robot.eef_links[arm].get_position_orientation()
        ep, eq = _np(ep), _np(eq)
        x = seg.extra
        if tag == "safe_up":                                   # raise straight up to live H_safe (keep xy+orient)
            return np.array([ep[0], ep[1], self._seg_z_safe(seg, ctx)]), eq
        if tag == "over_at_hsafe":                             # lateral to the STORED over-xy @ live H_safe
            return np.array([seg.eef_pos[0], seg.eef_pos[1], self._seg_z_safe(seg, ctx)]), eq
        if tag == "over_dest":                                 # lateral @ H_safe: held CENTRE xy -> dest_xy
            held = self._stack[int(x["inst"])].obj             # (still at the grasp orientation here)
            c = G.object_center(held)
            dxy = np.asarray(self._dest_xy, float)
            return np.array([ep[0] + (dxy[0] - c[0]), ep[1] + (dxy[1] - c[1]), self._seg_z_safe(seg, ctx)]), eq
        if tag == "reorient_place":                            # rotate in place @ H_safe to the upright hold
            pq = self._stack[int(x["inst"])].place_quat
            return (np.array([ep[0], ep[1], self._seg_z_safe(seg, ctx)]),
                    np.asarray(pq, float) if pq is not None else eq)
        if tag == "align_xy":                                  # @ H_safe: shift so the UPRIGHT held centre
            it = self._stack[int(x["inst"])]                   # -> dest_xy (rigid: centre lands exactly on it)
            c = G.object_center(it.obj)
            dxy = np.asarray(self._dest_xy, float)
            pq = it.place_quat
            return (np.array([ep[0] + (dxy[0] - c[0]), ep[1] + (dxy[1] - c[1]), self._seg_z_safe(seg, ctx)]),
                    np.asarray(pq, float) if pq is not None else eq)
        if tag == "lower_to_dest_pile":                        # descend so held bottom = dest-pile top + gap,
            inst = int(x["inst"])                              # holding the UPRIGHT place orientation, so an
            it = self._stack[inst]                             # identical concentric object nests/stacks in
            held = it.obj
            placed = [G.aabb_lo_hi(self._stack[j].obj) for j in range(inst)]      # already-placed instances
            st = G.surface_top_z(ctx.support)
            pile = SG.dest_pile_top(placed, self._dest_xy, self._stack_half,
                                    support_top=0.0 if st is None else st)
            dz = (pile + float(x["gap"])) - G.lowest_z(held)
            pq = it.place_quat
            return np.array([ep[0], ep[1], ep[2] + dz]), (np.asarray(pq, float) if pq is not None else eq)
        raise ValueError(f"StackSkeleton has no resolve_compute for tag {tag!r}")
