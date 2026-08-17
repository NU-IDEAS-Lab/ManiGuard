"""Dusty family skeleton — the ONLY dusty-specific code.

dusty = ``pnp(sponge) -> wipe(dest dust) -> return(sponge) -> pnp(source, food riding) ->
tilt-pour(food -> dest)``. It implements ``FamilySkeleton`` and nothing else: grasps come
from the annotation DB (source rim grasps filtered by a food-distance margin; sponge
top-down grasps loaded as a second DB read), and the ordered MotionSegment list encodes
the five phases. Dynamic-length behaviors (the wipe tour, the tilt ramp) are fixed-budget
segment chains whose ``compute`` tags resolve against LIVE sim state and become no-ops
once their objective is met — the generic executor plans/executes/gates/records.

Family-hard gates (user's laws): dust must be 100% removed before phase 3+
(``wipe_incomplete``; checked as n_particles == 0, STRICTER than the bench's
NOT-Covered which flips with residual particles), and the episode ends in the tilted
pose once the food lands (teleop termination semantics — no untilt / place-back).

Spike-0 findings baked in (2026-07-07, task_0000): the datagen boot DROPS the saved dust
group (attachment-uuid mismatch) -> ``_ensure_dust`` restores it from scene_ep1.json;
particle removal clears only the sponge FOOTPRINT on contact (14 -> 8 then stuck at any
hover height without lateral motion) -> the live wipe tour is load-bearing; the heavy
dest barely moves under light touch (0.0 mm) but the displacement gate stays as a guard.

Grasping mode = **sticky** (user decision). Sticky AG attaches whatever the closing
gripper touches, so mis-grabbing the FOOD at the source grasp would silently ruin the
pour ("food rides the gripper, never falls"). Guard chain (each layer independent):
  1. candidate filter — grasp XY distance to food >= dynamic margin (food half-extent +
     finger clearance, floored at FOOD_MARGIN_M);
  2. the food stays a LIVE cuRobo obstacle through src_pre AND src_descend (dusty does
     NOT use the blanket ignore_clutter there — only the target + support are dropped);
  3. per-step LTL ``food_touched_by_agent`` fails any finger brush instantly;
  4. after the close, ``require_attach`` verifies the SOURCE attached and an explicit
     AG check at src_lift fails the attempt if the FOOD is grasped ("food_grabbed");
  5. ``pour_no_drop`` backstops anything that still pins the food to the source.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.executor import geometry
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

H_WIPE_M = 0.004          # press depth: sponge-bottom gap above the TARGET particle's z.
#                           2mm presses hard (2cm nudge/press); 6mm under-removes (re-probe:
#                           remaining 17-18 on bowls) — 4mm is the removal/nudge sweet spot
WIPE_HOP_CLEAR_M = 0.03   # peck-wipe hop height above the particle plane: lateral moves happen
#                           IN AIR at this clearance — any bottom-contact lateral glide (even a
#                           4-5mm nominal gap, runs 6-7) friction-drags the pot ~1-3cm per hop,
#                           while pure VERTICAL presses never moved it (descend + Spike 0)
WIPE_OVER_TOL_M = 0.02    # "sponge is over the target particle" XY tolerance
FOOD_MARGIN_M = 0.08      # FLOOR of the dynamic grasp->food XY margin (no-touch + mis-grab guard)
FOOD_CLEAR_M = 0.05       # finger/palm clearance added to the food half-extent for that margin
WIPE_BUDGET_K = 20        # wipe_step segment budget: peck-wipe spends ~3 segments per press
#                           cycle (lift -> swing -> press); no-op segments are ~20 cheap sim steps
DEST_STEP_DISP_MAX_M = 0.06   # ONE wipe step shoving the dest this far = a violent hit -> fail
#                               (real rams measure 16-25cm; slick tables slide 4-5cm on a GENTLE press).
DEST_TOTAL_DISP_MAX_M = 0.15  # cumulative drift backstop. Gentle presses nudge a LIGHT bowl
#                               ~1-2cm each and legitimately walk it 5-9cm over a full wipe;
#                               every consumer (tour, pour stance, goal) tracks the dest LIVE,
#                               so slow drift is harmless — only sudden shoves are real faults
SPONGE_HALF_DIAG_M = 0.064  # worst-case sponge half-diagonal (10.5x7cm) + 1mm: mouth-clamp margin
# Gripper width profile in the eef frame (+z toward the object), measured from
# gripper_longfinger.glb: (z_lo, z_hi, xy half-diagonal). The wide finger-carriage block
# (z <= CARRIAGE_Z, half-diag 0.109) may NEVER dip below a deep dest's rim — it barely
# fits the widest mouth dead-centred, which is what dragged the pot through runs 2-9.
GRIP_BANDS = (
    (0.06, 0.104, 0.050),   # fingertips closed on the sponge
    (0.01, 0.06, 0.075),    # finger bodies
    (-0.03, 0.01, 0.082),   # knuckle block
)
CARRIAGE_Z = -0.03
RIM_MARGIN_M = 0.005
BOOM_MIN_OFF_M, BOOM_MAX_OFF_M = 0.03, 0.065  # boom-swing sponge-grasp offset window
NEED_BOOM_R_M = 0.07      # particles beyond this radius need the boom (centred footprint limit)
POUR_STEPS_K = 12         # tilt increments (~4 deg each: finer ramp = less angular kick
#                           at release; the bounce-outs track the release energy)
POUR_ALPHA_MAX_DEG = 95.0
H_POUR_M = 0.02           # source bottom above dest top at the pour stance (sweep-1:
#                           29/74 pour_missed + 5 landed-then-bounced-out => cut arrival energy)
CROSSBAR_CLEAR_M = 0.06   # door-frame crossbar clearance above the tallest scene object


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


def source_cat_model(diagnostics: dict) -> tuple[str, str]:
    """(category, model) of the dusty SOURCE carrier (the grasp/transport target; the
    goal_conditions reference is the DEST, which is never grasped)."""
    specs = (diagnostics.get("selection") or {}).get("spawn_specs") or []
    for s in specs:
        if s.get("role") == "source" and s.get("category") and s.get("model"):
            return s["category"], s["model"]
    raise ValueError("dusty diagnostics has no role='source' spawn spec")


def filter_grasps_by_food_margin(grasps_xy, food_xy, margin_m) -> list[int]:
    """Indices of grasps whose XY distance to the food is >= margin (LTL no-touch guard)."""
    d = np.linalg.norm(np.asarray(grasps_xy, float)[:, :2]
                       - np.asarray(food_xy, float)[None, :2], axis=1)
    return [i for i in range(len(d)) if d[i] >= margin_m]


def dynamic_food_margin(food_lo, food_hi, clear_m=FOOD_CLEAR_M, floor_m=FOOD_MARGIN_M) -> float:
    """Food-size-aware grasp margin: the fixed floor only covers a small food; a large one
    (e.g. a long potato) needs its half-extent + finger clearance so the closing fingers
    can never reach it (sticky AG would silently stick it)."""
    lo, hi = np.asarray(food_lo, float), np.asarray(food_hi, float)
    half = float(max(hi[0] - lo[0], hi[1] - lo[1])) / 2.0
    return max(float(floor_m), half + float(clear_m))


def wipe_next_xy(remaining_xy, cur_xy):
    """Nearest remaining particle (by XY) from the current sponge XY; None when clean.
    Accepts (N,2) or (N,3) rows and returns the matching full row — the caller uses the
    particle's OWN z to ride the curved container bottom instead of pinning to the
    global minimum (which digs the sponge edge in near the walls and drags the dest)."""
    rem = np.asarray(remaining_xy, float)
    if rem.shape[0] == 0:
        return None
    d = np.linalg.norm(rem[:, :2] - np.asarray(cur_xy, float)[None, :2], axis=1)
    return rem[int(np.argmin(d))]


def pour_stance_and_axis(grasp_xy, src_center_xy, dest_center_xy, r_src):
    """Pour stance: put the source's FAR edge (opposite the gripper) over the dest centre.
    far_dir = unit(src_center - grasp); stance (source-CENTRE xy) = dest_center - far_dir*r_src.
    Tilt axis = horizontal, perpendicular to far_dir, signed so the far edge DIPS
    (axis = up x far_dir; right-hand rotation about it lowers +far_dir points)."""
    far = np.asarray(src_center_xy, float) - np.asarray(grasp_xy, float)
    far = far / (np.linalg.norm(far) + 1e-9)
    stance = np.asarray(dest_center_xy, float) - far * float(r_src)
    axis = np.array([far[1], -far[0], 0.0])          # up x far_dir, sign so far edge dips
    # verify sign: rotating +far by small +angle about axis must lower z
    test = Rot.from_rotvec(axis * 0.1).apply(np.array([far[0], far[1], 0.0]))
    if test[2] > 0:
        axis = -axis
    return stance, axis


class DustySkeleton(FamilySkeleton):
    name = "dusty"

    def __init__(self, db: dict | None = None, *, grip_settle_steps: int = 6):
        self._db = db
        self.grip_settle_steps = int(grip_settle_steps)
        # per-task state (set once in select_grasps)
        self._sponge_name = None
        self._sponge_grasps: list[GraspCand] = []
        self._sponge_home = None          # (pos(3,), quat(4,)) at task start
        self._sponge_home_bottom = None   # sponge lowest z at task start
        self._dest_name = None
        self._food_name = None
        self._goal_pred = "inside"
        self._dest_home_xy = None
        self._src_rest_thick = None
        self._src_rest_width = None
        # per-variant state (reset in on_segment at the first segment)
        self._pour_axis = None
        self._pour_alpha = 0.0

    # ---- shared handles -------------------------------------------------------------
    def grasping_mode(self) -> str:
        return "sticky"                   # user decision for this family (see mis-grab guard chain)

    def _obj(self, ctx, name):
        return ctx.env.scene.object_registry("name", name)

    def _eef(self, ctx):
        arm = ctx.robot.default_arm
        p, q = ctx.robot.eef_links[arm].get_position_orientation()
        return _np(p), _np(q)

    # ---- dust helpers ----------------------------------------------------------------
    def _system(self, ctx):
        return ctx.env.scene.get_system(ctx.diagnostics["dust_system"], force_init=True)

    def _dust_positions(self, ctx) -> np.ndarray:
        """World positions (N,3) of the REMAINING dust particles (N shrinks as wiped)."""
        sys_ = self._system(ctx)
        n = int(sys_.dump_state(serialized=False).get("n_particles") or 0)
        if n == 0:
            return np.zeros((0, 3))
        pos, _ = sys_.get_particles_position_orientation()
        return _np(pos).reshape(-1, 3)

    def _ensure_dust(self, ctx) -> None:
        """Boot guard: the reloaded scene DROPS the saved dust group (attachment-uuid
        mismatch — same failure replay_empty_from_dataset.py:352 documents; Spike 0
        confirmed it on the datagen boot path). Restore the EXACT saved group from the
        task's scene_ep1.json, repointing it at the live dest. No-op when dust survived."""
        import copy
        import json
        import os

        import torch as th
        sys_ = self._system(ctx)
        if int(sys_.dump_state(serialized=False).get("n_particles") or 0) > 0:
            return
        tdir = ctx.diagnostics.get("_task_dir")
        if not tdir:
            print("[dusty] WARN: dust empty and no _task_dir — task would be dust-free", flush=True)
            return
        scene_info = json.load(open(os.path.join(tdir, "scene_ep1.json")))
        dust_src = (scene_info.get("state", {}).get("registry", {})
                    .get("system_registry", {}) or {}).get(ctx.diagnostics["dust_system"])
        dest = self._obj(ctx, self._dest_name)
        if not dust_src or dest is None:
            print("[dusty] WARN: dust restore skipped (no saved group / no dest)", flush=True)
            return
        st = copy.deepcopy(dust_src)
        for g in (st.get("groups") or {}).values():    # repoint the group at the rebuilt dest
            g["particle_attached_obj_uuid"] = dest.uuid
        # tensorize EVERY array field: OG's _load_state stores min/max_scale verbatim as
        # system attrs, and a later og.sim.dump_state(serialized=True) th.cat()s them —
        # a leftover json list crashes that dump (hit on the first driver run).
        for k in ("positions", "orientations", "scales", "min_scale", "max_scale"):
            if k in st:
                st[k] = th.tensor(st[k], dtype=th.float32)
        sys_.load_state(st, serialized=False)
        import omnigibson as og
        og.sim.step()
        print(f"[dusty] restored {st.get('n_particles')} dust particles onto {dest.name}",
              flush=True)

    def _sponge_xy_off(self, ctx, ep) -> np.ndarray:
        """LIVE eef->sponge XY offset. The sticky grasp hangs the sponge several cm off the
        eef axis (annotation offset + attach drift) — run 5 measured 5.7cm on task_0000, which
        wedged an 'eef-centred' descend against the pot wall. All wipe targets command the
        SPONGE CENTRE; add this offset to get the eef command (rigid hold => constant)."""
        sc = geometry.object_center(self._obj(ctx, self._sponge_name))
        return ep[:2] - sc[:2]

    def _rim_top(self, ctx) -> float:
        return float(geometry.aabb_lo_hi(self._obj(ctx, self._dest_name))[1][2])

    def _mouth_c(self, ctx) -> np.ndarray:
        """Live mouth-axis xy: original particle centroid, ridden along with any dest drift."""
        dxy = _np(self._obj(ctx, self._dest_name).get_position_orientation()[0])[:2]
        return self._mouth_c0 + (dxy - self._dest_home_xy)

    def _lateral_budget(self, eef_z: float, rim: float) -> float:
        """Max |eef_xy - mouth_axis| at this eef height: every gripper band whose lowest
        point dips below the rim plane must keep its half-diagonal inside the mouth."""
        if not self._dest_deep:
            return float("inf")
        b = float("inf")
        for _zlo, zhi, half in GRIP_BANDS:
            if eef_z - zhi < rim:                       # band's lowest world point below the rim
                b = min(b, self._wipe_r_in - half - RIM_MARGIN_M)
        return max(b, 0.0)

    def _boom_solve(self, ctx, ep, sponge_xy, s_star, eef_z):
        """Place the eef column (within budget of the mouth axis) + yaw the boom so the
        hanging sponge lands on s_star. Returns (eef_xy, dyaw)."""
        c = self._mouth_c(ctx)
        o_vec = np.asarray(sponge_xy, float) - ep[:2]
        o = float(np.linalg.norm(o_vec))
        budget = self._lateral_budget(eef_z, self._rim_top(ctx))
        v = np.asarray(s_star, float) - c
        r_t = float(np.linalg.norm(v))
        dhat = v / r_t if r_t > 1e-9 else np.array([1.0, 0.0])
        if o >= 0.02:
            # SIGNED placement: a target inside the boom ring needs the eef pulled to the
            # FAR side (e_r < 0) so the boom tip lands at r_t < o (run 12: the unsigned
            # solve pointed the boom at a near-centre target and overshot to the far wall)
            e_r = float(np.clip(r_t - o, -budget, budget))
        else:
            e_r = float(np.clip(r_t, 0.0, budget))
        e = c + dhat * e_r
        if o < 0.02:                                    # centred grasp: no boom, no yaw
            return e, 0.0
        want = np.asarray(s_star, float) - e            # boom must point from e toward s_star
        dyaw = float(np.arctan2(want[1], want[0]) - np.arctan2(o_vec[1], o_vec[0]))
        dyaw = float((dyaw + np.pi) % (2 * np.pi) - np.pi)
        return e, dyaw

    def _check_dest_disp(self, ctx) -> None:
        dest = self._obj(ctx, self._dest_name)
        xy = _np(dest.get_position_orientation()[0])[:2]
        last = getattr(self, "_dest_last_xy", None)
        self._dest_last_xy = xy
        if last is not None:
            step_d = float(np.linalg.norm(xy - last))
            if step_d > DEST_STEP_DISP_MAX_M:            # violent single-step hit
                raise FamilyAbort("dest_displaced", disp_m=round(step_d, 4), kind="violent")
        total = float(np.linalg.norm(xy - self._dest_home_xy))
        if total > DEST_TOTAL_DISP_MAX_M:                # runaway-drift backstop
            raise FamilyAbort("dest_displaced", disp_m=round(total, 4), kind="cumulative")

    def _landed(self, ctx) -> bool:
        """Has the food arrived in/on the dest (pour early-stop + verify)?"""
        food, dest = self._obj(ctx, self._food_name), self._obj(ctx, self._dest_name)
        try:
            from omnigibson.object_states import Inside, OnTop
            state = Inside if self._goal_pred == "inside" else OnTop
            if bool(food.states[state].get_value(dest)):
                return True
        except Exception:  # noqa: BLE001
            pass
        # geometric fallback: food centre within the dest mouth footprint, below its rim
        lo, hi = geometry.aabb_lo_hi(dest)
        c = geometry.object_center(food)
        return bool(lo[0] < c[0] < hi[0] and lo[1] < c[1] < hi[1] and c[2] < hi[2])

    # ---- per-task setup ---------------------------------------------------------------
    def select_grasps(self, ctx: TaskContext, world, robot) -> None:
        self._world, self._robot = world, robot   # for the lift-reach probe in grasp_candidates
        diag = ctx.diagnostics
        # exact instance names from the goal conditions (subject=food, reference=dest)
        for t in diag.get("goal_conditions") or []:
            if isinstance(t, dict) and t.get("predicate") in ("inside", "ontop"):
                self._goal_pred = t["predicate"]
                self._food_name, self._dest_name = t["subject"], t["reference"]
                break
        else:
            raise ValueError("dusty diagnostics has no inside/ontop goal condition")
        self._sponge_name = diag["sponge_name"]

        self._ensure_dust(ctx)

        sponge = self._obj(ctx, self._sponge_name)
        sp, sq = sponge.get_position_orientation()
        self._sponge_home = (_np(sp), _np(sq))
        self._sponge_home_bottom = float(geometry.lowest_z(sponge))
        dest = self._obj(ctx, self._dest_name)
        self._dest_home_xy = _np(dest.get_position_orientation()[0])[:2]
        # source rest extents (pre-grasp, level): AABB attitude inversion for src_level —
        # the held object's QUATERNION reads stale under AG, but its AABB tracks reality
        slo, shi = geometry.aabb_lo_hi(ctx.target)
        self._src_rest_thick = float(shi[2] - slo[2])
        self._src_rest_width = float(max(shi[0] - slo[0], shi[1] - slo[1]))
        # wipe geometry from the ORIGINAL particle spread (particles sit ON the inner
        # bottom, out to the wall base): mouth axis = particle centroid (dest AABB centre
        # is skewed by handles), inner radius = max radial spread. Only DEEP dests have
        # walls/rim to respect — the 3 flat ontop dests skip all of it.
        dlo, dhi = geometry.aabb_lo_hi(dest)
        self._dest_deep = bool((dhi[2] - dlo[2]) > 0.06)
        rem0 = self._dust_positions(ctx)
        if len(rem0):
            self._mouth_c0 = np.mean(rem0[:, :2], axis=0)
            r0 = np.linalg.norm(rem0[:, :2] - self._mouth_c0[None, :], axis=1)
            self._wipe_r_in = float(np.max(r0)) + 0.006
            self._need_boom = bool(self._dest_deep and float(np.max(r0)) > NEED_BOOM_R_M)
        else:
            self._mouth_c0 = geometry.object_center(dest)[:2]
            self._wipe_r_in, self._need_boom = 0.10, False

        db = self._db if self._db is not None else load_db()
        self._db = db
        key = f"sponge/{diag['sponge_model']}"
        gs = target_grasps_world(db, key, _np(sp), _np(sq))
        self._sponge_grasps = [GraspCand(id=g["id"], eef_pos=g["eef_pos"],
                                         eef_quat=g["eef_quat"], approach=g["approach"])
                               for g in gs]
        if not self._sponge_grasps:
            raise ValueError(f"no sponge grasps annotated for {key}")
        print(f"[dusty] setup: food={self._food_name} dest={self._dest_name} "
              f"pred={self._goal_pred} sponge={self._sponge_name} "
              f"sponge_grasps={len(self._sponge_grasps)}", flush=True)

    # ---- source grasp candidates (driver-iterated) -------------------------------------
    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        db = self._db if self._db is not None else load_db()
        self._db = db
        obj_pos, obj_quat = ctx.target.get_position_orientation()
        world = target_grasps_world(db, ctx.target_key, _np(obj_pos), _np(obj_quat))
        cands = [GraspCand(id=g["id"], eef_pos=g["eef_pos"], eef_quat=g["eef_quat"],
                           approach=g["approach"]) for g in world]
        food = self._obj(ctx, self._food_name)
        fxy = _np(food.get_position_orientation()[0])[:2]
        flo, fhi = geometry.aabb_lo_hi(food)
        margin = dynamic_food_margin(flo, fhi)
        keep = filter_grasps_by_food_margin(
            np.array([c.eef_pos[:2] for c in cands]), fxy, margin)
        print(f"[dusty] food margin={margin:.3f}m -> {len(keep)}/{len(cands)} grasps kept",
              flush=True)
        # lift-reach probe: rim grasps on the robot's FAR side pass the grasp-pose IK but
        # hit the arm's kinematic ceiling on the vertical carry lift — every such attempt
        # dies deterministically at clearance -0.085 (runs 17-19). Provably infeasible for
        # the required lift -> drop (grasp-set law compliant).
        if getattr(self, "_world", None) is not None and keep:
            import torch as th

            from maniguard.data.datagen.primitives.curobo_seg import solve_ik, solve_segment
            dest_obj = self._obj(ctx, self._dest_name)
            self._world.update_obstacles(ignore_objects=[dest_obj] if dest_obj else [])
            q0 = self._robot.get_joint_positions()
            manip_idx = self._robot.arm_control_idx[self._robot.default_arm]
            lifted = []
            for i in keep:
                c = cands[i]
                quat = th.as_tensor(np.asarray(c.eef_quat, float), dtype=th.float32)
                # CHAINED probe: solve the GRASP pose first, then the lifted pose seeded
                # from that arm config — mirrors the real servo chain. An isolated lifted-
                # pose IK can succeed in a different arm branch the servo can't reach
                # (run 20: g5 passed the naive probe, then stalled at the same ceiling).
                g_res = solve_ik(self._world.motion_gen, self._robot,
                                 th.as_tensor(np.asarray(c.eef_pos, float), dtype=th.float32),
                                 quat, q0, timeout=3.0, ik_collision=False,
                                 label=f"dusty:graspprobe:g{c.id}")
                if g_res is None:
                    continue
                q_seed = q0.clone()
                q_seed[manip_idx] = g_res.arm_traj[-1].to(q_seed.device, q_seed.dtype)
                res = solve_ik(self._world.motion_gen, self._robot,
                               th.as_tensor(np.asarray(c.eef_pos, float)
                                            + np.array([0.0, 0.0, 0.18]), dtype=th.float32),
                               quat, q_seed, timeout=3.0, ik_collision=False,
                               label=f"dusty:liftprobe:g{c.id}")
                if res is None:
                    continue
                # carry-STANCE probe: a COLLISION-AWARE plan attempt (sweep-1: solve_ik
                # passed stances that the real carry's collision-aware fallback then
                # refused 6/6 on task_0001 — plan_fail src_carry burned whole tasks).
                # No attach model at select time (the board is still on the table), but
                # the arm-vs-scene reachability is the deterministic killer being probed.
                e_xy, z_abs, _, _ = self._carry_solution(ctx, np.asarray(c.eef_pos, float))
                q_seed2 = q0.clone()
                q_seed2[manip_idx] = res.arm_traj[-1].to(q_seed2.device, q_seed2.dtype)
                res2 = solve_segment(self._world.motion_gen, self._robot,
                                     th.as_tensor(np.array([e_xy[0], e_xy[1], z_abs]),
                                                  dtype=th.float32),
                                     quat, q_seed2, timeout=3.0, max_attempts=4,
                                     label=f"dusty:stanceprobe:g{c.id}")
                if res2 is not None:
                    lifted.append(i)
            if lifted:
                if len(lifted) < len(keep):
                    print(f"[dusty] lift-reach probe dropped "
                          f"{sorted(set(keep) - set(lifted))} -> {len(lifted)} grasps", flush=True)
                keep = lifted
            else:
                print("[dusty] WARN: lift-reach probe emptied grasps; keeping unprobed set",
                      flush=True)
        if not keep:                       # degenerate: keep farthest-half rather than none
            d = [float(np.linalg.norm(np.asarray(c.eef_pos[:2]) - fxy)) for c in cands]
            keep = list(np.argsort(d)[::-1][: max(1, len(cands) // 2)])
            print(f"[dusty] WARN: food-margin emptied grasps; keeping farthest {len(keep)}",
                  flush=True)
        return [cands[i] for i in keep]

    # ---- the 5-phase segment plan -------------------------------------------------------
    def derive_segments(self, ctx: TaskContext, grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        rng = np.random.default_rng((int(params.seed) & 0xFFFFFFFF) ^ 0xD057)
        # attempt-alternating stance lift (jar's alternating-standoff pattern): odd draws
        # carry +6cm higher — trades a little pour energy for carry-plan feasibility
        # (re-sweep: plan_fail@src_carry x15 deterministic on big/awkward sources)
        self._stance_lift = 0.06 * (params.draw_index % 2)
        # sponge grasp pool by wipe mode: deep+wide dests take a BOOM grasp (3-6.5cm
        # offset — the eef column stays on the mouth axis and yaw swings the hanging
        # sponge out to the wall); deep+narrow take a CENTRED grasp (eef column and
        # sponge coaxial — any off-axis hang parks the finger blade over the rim ring,
        # runs 5-9); flat dests have no walls -> full pool for diversity.
        home_xy = self._sponge_home[0][:2]
        offs = [float(np.linalg.norm(np.asarray(g.eef_pos[:2]) - home_xy))
                for g in self._sponge_grasps]
        if self._dest_deep and self._need_boom:
            pool = [g for g, o in zip(self._sponge_grasps, offs)
                    if BOOM_MIN_OFF_M <= o <= BOOM_MAX_OFF_M] \
                or [self._sponge_grasps[int(np.argmax(offs))]]
        elif self._dest_deep:
            pool = [g for g, o in zip(self._sponge_grasps, offs) if o <= 0.02] \
                or [self._sponge_grasps[int(np.argmin(offs))]]
        else:
            pool = self._sponge_grasps
        sg = pool[int(rng.integers(len(pool)))]
        sR = Rot.from_quat(sg.eef_quat).as_matrix()
        s_app = sR[:, 2] / (np.linalg.norm(sR[:, 2]) + 1e-9)
        s_pre = np.asarray(sg.eef_pos, float) - min(params.standoff_m, 0.08) * s_app
        sq = np.asarray(sg.eef_quat, float)

        gR = Rot.from_quat(grasp.eef_quat).as_matrix()
        g_app = gR[:, 2] / (np.linalg.norm(gR[:, 2]) + 1e-9)
        g_pre = np.asarray(grasp.eef_pos, float) - params.standoff_m * g_app
        gq = np.asarray(grasp.eef_quat, float)

        steps = self.grip_settle_steps
        sponge, dest = self._sponge_name, self._dest_name
        held_s = {"held_name": sponge}
        Z = np.array([0.0, 0.0, 0.0])      # placeholder pos for compute-resolved segments

        segs = [
            # ---- phase 1: pick sponge ----
            MotionSegment("sponge_pre", s_pre, sq, mode=Mode.FREE,
                          grip=Grip.OPEN, grip_steps=steps,
                          ignore_objects=(sponge,), extra=dict(held_s)),
            MotionSegment("sponge_descend", np.asarray(sg.eef_pos, float), sq,
                          mode=Mode.LINEAR, grip=Grip.CLOSE, grip_steps=steps,
                          ignore_objects=(sponge,), ignore_clutter=True,
                          require_attach=True, extra=dict(held_s)),
            MotionSegment("sponge_lift", Z, sq, mode=Mode.LINEAR, attach=True,
                          compute="rise_crossbar", extra=dict(held_s)),
            # ---- phase 2: wipe ----
            MotionSegment("wipe_over", Z, sq, mode=Mode.FREE, attach=True,
                          compute="wipe_over", extra=dict(held_s)),
            MotionSegment("wipe_descend", Z, sq, mode=Mode.LINEAR, attach=True,
                          compute="wipe_descend", ignore_objects=(dest,),
                          extra=dict(held_s)),
            *[MotionSegment(f"wipe_step_{i}", Z, sq, mode=Mode.SERVO, attach=True,
                            compute="wipe_next", servo_step_m=0.01, servo_spw=6,
                            orient_slerp=True, extra=dict(held_s))
              for i in range(WIPE_BUDGET_K)],
            MotionSegment("wipe_verify", Z, sq, mode=Mode.SERVO, attach=True,
                          compute="wipe_verify", extra=dict(held_s)),
            MotionSegment("wipe_ascend", Z, sq, mode=Mode.LINEAR, attach=True,
                          compute="rise_crossbar", ignore_objects=(dest,),
                          extra=dict(held_s)),
            # ---- phase 3: return sponge ----
            MotionSegment("sponge_home_over", Z, sq, mode=Mode.FREE, attach=True,
                          compute="sponge_home_over", extra=dict(held_s)),
            # ignore ONLY the support (it blocks the 3mm-above-table place target). Never
            # ignore_clutter here, and never ignore the HELD sponge itself: cuRobo's
            # attach pulls the attached object's mesh from the world config — dropping it
            # crashes the merged-mesh build (runs 13-14).
            MotionSegment("sponge_home_place", Z, sq, mode=Mode.LINEAR, attach=True,
                          grip=Grip.OPEN, grip_steps=steps, compute="sponge_home_place",
                          ignore_objects=tuple(n for n in (
                              getattr(ctx.support, "name", None),) if n),
                          extra=dict(held_s)),
            MotionSegment("sponge_retreat", Z, sq, mode=Mode.LINEAR,
                          compute="rise_crossbar", ignore_objects=(sponge,)),
            # ---- phase 4: pick source (food riding; grasp point is food-margin-filtered) ----
            MotionSegment("src_pre", g_pre, gq, mode=Mode.FREE,
                          grip=Grip.OPEN, grip_steps=steps,
                          ignore_objects=(ctx.target_name,)),
            # NO blanket ignore_clutter here: the FOOD must stay a live cuRobo obstacle
            # through the descend (sticky mis-grab guard #2) — only target + support drop.
            # +4mm on the grasp z: fingertips pressing the rim see-saws the board on the
            # table at close/lift-off — the transient starts the round food ROLLING and it
            # exits mid-transport even after the board levels out (run 30 frames). Long
            # settle lets any residual rock damp out before the lift.
            MotionSegment("src_descend",
                          np.asarray(grasp.eef_pos, float) + np.array([
                              0.0, 0.0,
                              # sticky: +4mm no-press hover (see-saw guard); assisted: the
                              # finger raycast needs the thin rim EXACTLY between the pads
                              0.004 if getattr(ctx.robot, "grasping_mode", "sticky") != "assisted"
                              else 0.0]), gq,
                          mode=Mode.LINEAR, grip=Grip.CLOSE, grip_steps=24,
                          ignore_objects=tuple(n for n in (
                              ctx.target_name, getattr(ctx.support, "name", None)) if n),
                          require_attach=True),
            # ---- phase 5: ONE-SEGMENT carry (lift+level+transport merged) + tilt-pour ----
            # Every segment boundary brakes (re-command + settle) and the jolt kicks the
            # round food into a roll on the smooth board (run 31: both drops came right
            # AFTER a boundary, on a LEVEL board, in slow motion). One 3D straight-line
            # UNIFORM glide (4mm/sim-step, spw=1 — the gentle drawer-close recipe) from
            # the grasp to the pour stance, the AABB droop correction folded into the same
            # segment's orient_slerp: ZERO internal stops. dest ignored: the stance
            # intentionally hangs the far edge over the mouth (height/angle budget
            # guarantees the physical clearance).
            # 2mm/sim-step (~0.06 m/s): a free-ROLLING food keeps the carry velocity when
            # the tray stops and coasts v^2/(2*rolling-resistance) past the arrival brake —
            # 0.12 m/s coasts ~3.6cm (off a flat tray's edge, run 36-era telemetry on 0015),
            # 0.06 m/s coasts <1cm and stays aboard even with no lip.
            MotionSegment("src_carry", Z, gq, mode=Mode.SERVO, attach=True,
                          compute="src_carry", orient_slerp=True,
                          servo_step_m=0.002, servo_spw=1,
                          free_fallback=True, ignore_objects=(self._dest_name,)),
            *[MotionSegment(f"pour_step_{i}", Z, gq, mode=Mode.SERVO, attach=True,
                            compute="pour_next", orient_slerp=True, servo_spw=6)
              for i in range(POUR_STEPS_K)],
            MotionSegment("pour_verify", Z, gq, mode=Mode.SERVO, attach=True,
                          compute="pour_verify"),
            MotionSegment("pour_settle", Z, gq, mode=Mode.SERVO, attach=True,
                          compute="pour_hold"),
        ]
        return segs

    # ---- end-of-rollout quality gate ------------------------------------------------------
    def success_extra(self, ctx: TaskContext) -> bool:
        """The dest must still stand upright at the end: a hard food impact can tip a light
        bowl and 'inside a tipped-over bowl' must not count as success (user review). The
        dest is never AG-held, so its quaternion is reliable."""
        dest = self._obj(ctx, self._dest_name)
        dq = _np(dest.get_position_orientation()[1])
        up_z = float(Rot.from_quat(dq).apply(np.array([0.0, 0.0, 1.0]))[2])
        if up_z < np.cos(np.deg2rad(20.0)):
            print(f"[dusty] success_extra: dest tipped (up_z={up_z:.2f}) -> fail", flush=True)
            return False
        # END-state dest displacement: the tight-mouth tupperware can get HOOKED by the
        # rising gripper after the wipe (task_0006 kept demo: dest lifted 17cm + carried
        # 30cm, episode still "succeeded" because every consumer tracks the dest live).
        # Such polluted-but-lucky episodes must NOT enter the dataset.
        dxy = _np(dest.get_position_orientation()[0])[:2] - self._dest_home_xy
        if float(np.linalg.norm(dxy)) > DEST_TOTAL_DISP_MAX_M:
            print(f"[dusty] success_extra: dest displaced {np.linalg.norm(dxy):.2f}m -> fail",
                  flush=True)
            return False
        return True

    # ---- DATAGEN_DEBUG_STATE probe (engine prints it after every segment) -----------------
    def debug_state(self, ctx: TaskContext) -> str:
        sponge = self._obj(ctx, self._sponge_name)
        dest = self._obj(ctx, self._dest_name)
        lo, hi = geometry.aabb_lo_hi(dest)
        sc = geometry.object_center(sponge)
        dr = _np(dest.get_position_orientation()[0])
        extra = ""
        if self._src_rest_thick is not None and self._food_name:
            tlo, thi = geometry.aabb_lo_hi(ctx.target)
            food_c = geometry.object_center(self._obj(ctx, self._food_name))
            extra = (f" src_sag={thi[2] - tlo[2] - self._src_rest_thick:+.3f}"
                     f" food={np.round(food_c, 3)}")
        return (f"sponge_c={np.round(sc, 3)} sponge_bot={geometry.lowest_z(sponge):.3f} "
                f"dest_root={np.round(dr, 3)} rim={hi[2]:.3f} "
                f"n_dust={len(self._dust_positions(ctx))} "
                f"alpha={np.rad2deg(self._pour_alpha):.0f}{extra}")

    # ---- per-variant state reset ---------------------------------------------------------
    def on_segment(self, seg: MotionSegment, ctx: TaskContext) -> None:
        if seg.name == "sponge_pre":               # first segment of every variant
            self._pour_axis = None
            self._pour_alpha = 0.0
            self._pour_alpha_max = None
            self._wipe_zoff = None
            self._dest_last_xy = None
            # _stance_lift is set per-variant in derive_segments (draw parity)
        if seg.name == "src_carry":                # runs AFTER src_descend's CLOSE: sticky
            #                                        mis-grab guard #4 — the FOOD must not be AG-held
            food = self._obj(ctx, self._food_name)
            try:
                ag = ctx.robot.is_grasping(ctx.robot.default_arm, food)
            except Exception:  # noqa: BLE001
                ag = None
            if ag is not None and int(ag) == 1:
                raise FamilyAbort("food_grabbed")


    def _carry_solution(self, ctx, ep):
        """Shared carry-stance solver (resolve + candidate prescreen use the SAME math):
        returns (stance_eef_xy, stance_eef_z_abs, pour_axis, alpha_max) for an eef at ep."""
        src, dest = ctx.target, self._obj(ctx, self._dest_name)
        src_c = geometry.object_center(src)
        dest_c = self._mouth_c(ctx) if self._dest_deep else geometry.object_center(dest)[:2]
        lo, hi = geometry.aabb_lo_hi(src)
        ep = np.asarray(ep, float)
        far = src_c[:2] - ep[:2]
        far = far / (np.linalg.norm(far) + 1e-9)
        half = np.array([(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0])
        r_far = float(abs(far[0]) * half[0] + abs(far[1]) * half[1])
        _, axis = pour_stance_and_axis(ep[:2], src_c[:2], dest_c, max(r_far - 0.02, 0.0))
        dlo, dhi = geometry.aabb_lo_hi(dest)
        dest_top, floor = float(dhi[2]), float(dlo[2]) + 0.005
        width = 2.0 * r_far
        # FLAT-BOTTOMED foods (half garlic/berry) don't slide at ~50 deg — they need a
        # ~65-deg budget (conf2: 0025 pour_no_drop x4 at its 51-deg cap). They are light
        # and land dead, so the taller stance costs little bounce energy.
        food_cat = getattr(self._obj(ctx, self._food_name), "category", "")
        a_tilt = 65.0 if food_cat in ("garlic_clove", "half_blackberry") else 50.0
        want_bottom = max(dest_top + H_POUR_M,
                          floor + 0.015 + width * np.sin(np.deg2rad(a_tilt)))
        want_bottom += getattr(self, "_stance_lift", 0.0) or 0.0
        alpha_max = float(np.arcsin(np.clip(
            (want_bottom - floor - 0.010) / max(width, 1e-6), 0.2, 0.995)))
        dz = want_bottom - float(geometry.lowest_z(src))
        flat_src = ctx.target.category in ("chopping_board", "cutting_board")
        a_slide = 12.0 if flat_src else 35.0
        if food_cat in ("garlic_clove", "half_blackberry"):
            a_slide = 55.0                    # flat-bottomed food releases LATE (55-60 deg):
            #                                   the edge has retracted far more at exit
        if self._dest_deep:
            mouth_r = self._wipe_r_in
        else:                                   # flat dest (ontop): corridor = its real
            dlo2, dhi2 = geometry.aabb_lo_hi(self._obj(ctx, self._dest_name))
            mouth_r = 0.7 * min(dhi2[0] - dlo2[0], dhi2[1] - dlo2[1]) / 2.0
        # flat boards release FAST with ~±5-8cm dispersion -> aim the mouth CENTRE (the
        # far wall backstops the roll flight); lipped sources release slow -> near-rim
        # aim keeps their gentler exit inside the corridor.
        reach = width * np.cos(np.deg2rad(a_slide)) + (0.0 if flat_src
                                                       else max(mouth_r - 0.03, 0.0))
        food = self._obj(ctx, self._food_name)
        t_hat = np.array([-far[1], far[0]])
        lat = float(np.dot(geometry.object_center(food)[:2] - src_c[:2], t_hat))
        e_xy = np.asarray(dest_c, float) - far * reach - t_hat * lat
        return e_xy, float(ep[2] + dz), axis, alpha_max

    # ---- runtime compute tags --------------------------------------------------------------
    def resolve_compute(self, tag: str, seg: MotionSegment, ctx: TaskContext):
        ep, eq = self._eef(ctx)

        if tag == "rise_crossbar":
            held = (seg.extra or {}).get("held_name") or (ctx.target_name if seg.attach else None)
            held_obj = self._obj(ctx, held) if held else None
            exclude = [held_obj] if held_obj is not None else []
            st = geometry.surface_top_z(ctx.support)
            top, _ = geometry.max_other_top_z(ctx.env, exclude=exclude,
                                              robots=ctx.env.robots, support_top=st)
            cross = (top if np.isfinite(top) else (st or ep[2])) + CROSSBAR_CLEAR_M
            base = float(geometry.lowest_z(held_obj)) if held_obj is not None else float(ep[2])
            dz = max(0.0, cross - base)
            return np.array([ep[0], ep[1], ep[2] + dz]), eq

        if tag == "wipe_over":
            # deep dest: pre-align eef + boom yaw so the hanging sponge lands CORNER-SAFE
            # over the dust centroid (a raw mouth-axis park leaves the boom sponge at its
            # full offset ring — its corners overlap the wall and the first press stalls,
            # run 11); flat dest: centre the SPONGE over the dust centroid.
            c = self._mouth_c(ctx)
            if self._dest_deep:
                rem = self._dust_positions(ctx)
                s0 = np.mean(rem[:, :2], axis=0) if len(rem) else c
                v = s0 - c
                r_cap = self._wipe_r_in - SPONGE_HALF_DIAG_M - 0.002
                r0 = float(np.linalg.norm(v))
                if r0 > r_cap:
                    s0 = c + v * (r_cap / max(r0, 1e-9))
                sponge_xy = geometry.object_center(self._obj(ctx, self._sponge_name))[:2]
                e, dyaw = self._boom_solve(ctx, ep, sponge_xy, s0,
                                           self._rim_top(ctx) - (-CARRIAGE_Z) + RIM_MARGIN_M)
                q = (Rot.from_rotvec(np.array([0.0, 0.0, dyaw])) * Rot.from_quat(eq)).as_quat()
                return np.array([e[0], e[1], ep[2]]), q
            off = self._sponge_xy_off(ctx, ep)
            return np.array([c[0] + off[0], c[1] + off[1], ep[2]]), eq

        if tag == "wipe_descend":
            # PURE-Z first press at the current XY: a mixed XY+Z target breaks the LINEAR
            # partial-pose constraint (silent unconstrained fallback that swings into the
            # rim). Press eef-z floored so the carriage NEVER dips below the rim.
            rem = self._dust_positions(ctx)
            dest = self._obj(ctx, self._dest_name)
            plane = float(np.min(rem[:, 2])) if len(rem) else float(geometry.aabb_lo_hi(dest)[0][2]) + 0.01
            bottom = float(geometry.lowest_z(self._obj(ctx, self._sponge_name)))
            # cache the eef->sponge-bottom offset while the sponge hangs FREE: the AG
            # attach is compliant, so re-deriving it from a PRESSED state shrinks it and
            # every subsequent press commands deeper — a positive-feedback ram that
            # curling-drags light bowls across the table (task_0008 probe).
            self._wipe_zoff = ep[2] - bottom
            z = ep[2] + (plane + H_WIPE_M) - bottom
            if self._dest_deep:
                z = max(z, self._rim_top(ctx) - (-CARRIAGE_Z) + RIM_MARGIN_M)
            return np.array([ep[0], ep[1], z]), eq

        if tag == "wipe_next":
            self._check_dest_disp(ctx)
            rem = self._dust_positions(ctx)
            sponge = self._obj(ctx, self._sponge_name)
            sponge_xy = geometry.object_center(sponge)[:2]
            nxt = wipe_next_xy(rem, sponge_xy)
            if nxt is None:
                raise SegmentSkip                    # clean — drop the leftover budget segments
                #                                      (executing them froze the demo for seconds)
            bottom = float(geometry.lowest_z(sponge))
            # free-hanging eef->bottom offset (cached at wipe_descend; live zoff shrinks
            # under press compression and runs the depth control into positive feedback)
            zoff = getattr(self, "_wipe_zoff", None) or (ep[2] - bottom)
            rim = self._rim_top(ctx) if self._dest_deep else -np.inf
            press_z = float(nxt[2]) + H_WIPE_M + zoff
            hop_z = float(nxt[2]) + WIPE_HOP_CLEAR_M + zoff
            if self._dest_deep:
                press_z = max(press_z, rim - (-CARRIAGE_Z) + RIM_MARGIN_M)  # carriage floor
                # swings happen FULLY ABOVE the rim: a boom yaw below rim level sweeps the
                # sponge corners through the wall ring no matter the endpoint clamps
                # (swing radius ≈ boom length > corner-safe radius — run 12). Above the
                # rim the whole gripper clears the pot and lateral/yaw motion is free.
                hop_z = max(hop_z, rim + RIM_MARGIN_M + zoff)
            # clamp the desired SPONGE landing spot CORNER-SAFE: the sponge yaws with the
            # boom, so its worst-case half-DIAGONAL must clear the wall at every press AND
            # every swing endpoint (a short-half-only cap let the corners sideswipe the
            # wall ring and drag the pot — run 11). The swept band still reaches
            # r_cap + half-diag ≈ the wall base.
            c = self._mouth_c(ctx)
            v = nxt[:2] - c
            r_t = float(np.linalg.norm(v))
            r_cap = self._wipe_r_in - SPONGE_HALF_DIAG_M - 0.002
            s_star = (c + v * (r_cap / max(r_t, 1e-9))) if (self._dest_deep and r_t > r_cap) else nxt[:2]
            e, dyaw = self._boom_solve(ctx, ep, sponge_xy, s_star, press_z)
            # predicted sponge landing after the swing (eef -> e, boom rotated by dyaw)
            o_vec = sponge_xy - ep[:2]
            cs, sn = np.cos(dyaw), np.sin(dyaw)
            s_hat = e + np.array([cs * o_vec[0] - sn * o_vec[1], sn * o_vec[0] + cs * o_vec[1]])
            aligned = (float(np.linalg.norm(sponge_xy - s_hat)) < WIPE_OVER_TOL_M
                       and abs(dyaw) < 0.06)
            # --- boom peck-wipe state machine: swing/travel ONLY in air, contact ONLY vertical
            if aligned:                               # 3) PRESS straight down at the current spot
                return np.array([ep[0], ep[1], press_z]), eq
            if bottom < (hop_z - zoff) - 0.005:       # 1) LIFT pure-z clear (deep: above the rim)
                return np.array([ep[0], ep[1], hop_z]), eq
            # 2) SWING in air: eef to its budgeted spot + boom yaw (orient_slerp segment)
            q = (Rot.from_rotvec(np.array([0.0, 0.0, dyaw])) * Rot.from_quat(eq)).as_quat()
            return np.array([e[0], e[1], hop_z]), q

        if tag == "wipe_verify":
            self._check_dest_disp(ctx)
            n = len(self._dust_positions(ctx))
            if n > 0:                                 # user's law: 100% removed, no tolerance
                raise FamilyAbort("wipe_incomplete", remaining=int(n))
            return ep, eq

        if tag == "sponge_home_over":
            hx, hy = self._sponge_home[0][:2]
            return np.array([hx, hy, ep[2]]), eq

        if tag == "sponge_home_place":
            # PURE-Z (sponge_home_over already put us over the home XY) — same LINEAR
            # single-axis rule as wipe_descend.
            bottom = float(geometry.lowest_z(self._obj(ctx, self._sponge_name)))
            dz = (self._sponge_home_bottom + 0.003) - bottom
            return np.array([ep[0], ep[1], ep[2] + dz]), eq

        if tag == "src_carry":
            e_xy, z_abs, axis, a_max = self._carry_solution(ctx, ep)
            self._pour_axis = axis
            self._pour_alpha_max = a_max
            # fold the droop correction into the carry's orient_slerp: AABB attitude
            # inversion (the held object's quaternion reads STALE under AG)
            q_out = eq
            src_c = geometry.object_center(ctx.target)
            sag = float(geometry.aabb_lo_hi(ctx.target)[1][2]
                        - geometry.aabb_lo_hi(ctx.target)[0][2]) - self._src_rest_thick
            if sag > 0.02:
                droop = min(float(np.arcsin(np.clip(
                    sag / max(self._src_rest_width, 1e-6), 0.0, 0.98))), np.deg2rad(40.0))
                _, dax = pour_stance_and_axis(ep[:2], src_c[:2],
                                              src_c[:2] + (src_c[:2] - ep[:2]), 0.1)
                q_out = (Rot.from_rotvec(np.asarray(dax) * (-droop)) * Rot.from_quat(eq)).as_quat()
                print(f"[dusty] src_carry: folding {np.rad2deg(droop):.1f}deg droop correction",
                      flush=True)
            return np.array([e_xy[0], e_xy[1], z_abs]), q_out

        if tag == "pour_next":
            if self._landed(ctx):
                raise SegmentSkip                     # food is in — drop the leftover tilt segments
            if self._pour_axis is None:               # safety: recompute if transport skipped it
                src, dest = ctx.target, self._obj(ctx, self._dest_name)
                _, self._pour_axis = pour_stance_and_axis(
                    ep[:2], geometry.object_center(src)[:2],
                    geometry.object_center(dest)[:2], 0.1)
            a_max = getattr(self, "_pour_alpha_max", None) or np.deg2rad(POUR_ALPHA_MAX_DEG)
            if self._pour_alpha >= a_max - 1e-6:
                return ep, eq                         # geometric tilt budget spent — hold
            da = a_max / POUR_STEPS_K
            self._pour_alpha += da
            q = (Rot.from_rotvec(np.asarray(self._pour_axis) * da) * Rot.from_quat(eq)).as_quat()
            return ep, q

        if tag == "pour_verify":
            if not self._landed(ctx):
                # distinguish "never slid off" from "slid but missed the dest" (run 16)
                food = self._obj(ctx, self._food_name)
                slo, shi = geometry.aabb_lo_hi(ctx.target)
                fc = geometry.object_center(food)
                aboard = bool(slo[0] - 0.02 < fc[0] < shi[0] + 0.02
                              and slo[1] - 0.02 < fc[1] < shi[1] + 0.02
                              and fc[2] > slo[2] - 0.02)
                raise FamilyAbort("pour_no_drop" if aboard else "pour_missed",
                                  alpha_deg=round(float(np.rad2deg(self._pour_alpha)), 1))
            return ep, eq

        if tag == "pour_hold":
            return ep, eq                             # zero-motion settle; engine rest_settle follows

        raise ValueError(f"dusty: unknown compute tag {tag!r}")
