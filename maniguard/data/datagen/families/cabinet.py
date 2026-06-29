"""Cabinet family skeleton — the only cabinet-specific code.

Task: a sliding-drawer cabinet on the table + a target (goes INSIDE the drawer) + an obstacle.
Success (gate) = ``inside(target, cabinet) & closed(cabinet)``; temporal safety (clear the path
before opening) is enforced implicitly by the LTL upright gate (opening into an un-moved object
knocks it over → violation → demo voided).

Two motion building blocks (the engine owns both — see ``executor.contracts.Mode``):
  * ``Mode.FREE``  — cuRobo collision-aware transit. ALL free-space moves (to a grasp, to a place,
                     to the handle pre-grasp). Avoids the cabinet + every other object.
  * ``Mode.SERVO`` — pure straight-line Cartesian IK (collision off, nearest-seed so a straight
                     servo does NOT skew the wrist). EVERY deterministic straight contact: the pick /
                     place descents + lifts, and the open/close drawer push-pull.
Cabinet deliberately avoids ``Mode.LINEAR``: this old cuRobo fork's partial-pose (LINEAR_SERVO)
constraint query always fails here and silently falls back to an UNCONSTRAINED salvage solve that
drifts the eef off the straight line (e.g. carrying the relocated object forward off the table edge,
or missing the grasp). SERVO's pure IK has no such dependency, so all straight moves use it.

The 4-phase flow (the bench spawns the drawer CLOSED, Method 2 — no redundant initial close)::

    1  relocate blockers        FREE pick → SERVO descend → SERVO lift → FREE transit → SERVO place
    2  open drawer              FREE → handle pre-grasp / SERVO grasp / SERVO pull to MAX / open
    3  place target in drawer    FREE → SERVO deep grasp / SERVO inverted-U (↑lift →over upright ↓lower)
    4  close drawer (final)     FREE → handle pre-grasp / SERVO push shut / replay back

Only segment generation lives here; the generic engine plans / executes / gates / records. Runtime
targets (grasps at live poses, drawer pull/push, the place up-over-down) resolve through
``resolve_compute``. Geometry comes from :mod:`cabinet_geom`; grasp poses from the annotation DB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.executor.contracts import (
    FamilySkeleton, GraspCand, Grip, Mode, MotionSegment, SampleParams, TaskContext,
)
from maniguard.data.datagen.families import cabinet_geom as CG
from maniguard.data.datagen.grasp_db import load_db

CAB_GEOM_PATH = Path("outputs/grasp_annotation/cabinet_geom.json")
STANDOFF = 0.10            # pre-grasp / pre-handle standoff along the approach axis
OVER_HANDLE_FRAC = 0.55    # close approach: SERVO this fraction of the way from the cavity centre toward the
#                            close-handle pre-grasp XY (z held, in the open space above the rim), then let the
#                            COLLISION-AWARE close_pre cuRobo do the short final reach — the gripper is already
#                            near the drawer front, so cuRobo has an easy, door-clear plan (full=contorted, none=long winding).
PLACE_Z_MARGIN = 0.01     # drop height above the table / drawer floor
RETREAT_DZ = 0.12         # straight-up retreat after releasing a relocated blocker
HANDLE_BACK_DIST = 0.10   # after opening, retreat the gripper this far along +open so fingers clear the handle
CLOSE_PRE_ROT_RELAX = 0.10  # close_pre IK rotation_threshold relax (rad, ≈5.7°). The FINAL close re-grasps the
#                            handle at its pulled-OUT (open) position = the far edge of reach; the only reachable
#                            wrist there is a few degrees (obs. 0.043–0.057 rad) off the grasp orientation — a miss
#                            on the ROLL-SYMMETRIC handle bar (HARMLESS).
CLOSE_PRE_POS_RELAX = 0.015  # close_pre IK position_threshold relax (m, 1.5 cm). At that far open-handle pre-grasp
#                            STANDOFF the arm bottoms out its reach ~5.2-5.8 mm short of the exact point (> the 5 mm
#                            gate) — diagnosed as the BINDING constraint (pos, not rot, IK_FAILs first on the worst
#                            tasks). Harmless at a standoff: close_grasp SERVOs onto the handle from its LIVE pose,
#                            absorbing the offset. cuRobo's baked 0.005-m IK gate else rejects it (IK_FAIL, no traj).
#                            Widen BOTH gates for THIS plan only; a genuinely-bad grasp is still caught by close_grasp
#                            + the physical success gate. See [[project_maniguard_cabinet_family]] SESSION-5 Approach A.
CLOSE_FRACTION = 0.12     # close to this fraction of the spawn opening (≈88% shut; << Open threshold)
DRAWER_OPEN_MARGIN = 0.02   # stop the open pull this far off the drawer's hard stop (full stroke)
DRAWER_OPEN_FRACS = (1.0, 0.85, 0.72, 0.6)   # search the open distance from full-stroke DOWN; the widest
#                            whose open-END the arm can still reach (handle grasp clear there) is the max
LIFT_CLEAR = 0.18         # relocate: lift the held blocker this HIGH above the table before the FREE transit
#                           — a low (~5 cm) lift leaves the object hugging the table next to the tall cabinet,
#                           so the cabinet-avoiding cuRobo transit plans a hard low path (~50% plan_fail on this
#                           fork); lifting it well clear first gives the transit a roomy high lane = reliable plan
RIM_CLEARANCE = 0.06      # place: lift the target's bottom this far above the drawer rim before going over
# Per-demo diversity bands (sampled from the variant seed in derive_segments; small so the long-horizon
# task still reliably completes). Dim 1: how far above the rim the target's bottom is carried (the lift
# height). Dim 2: how far the relocated TARGET slides along +opening on the near edge (where it lands).
RIM_CLEAR_BAND = (0.01, 0.02)   # COMMANDED carry clearance above the rim — kept SMALL. The old 0.10-0.13
# compensated for the joint_position_impedance WRIST sag (~0.07 m droop); since the switch to the RIGID
# joint_position_raw controller (903c0277) there is NO droop, so the commanded clearance is achieved directly.
# A large clearance also OVERSHOOTS the arm's TOP-DOWN orientation-reachability ceiling at the cavity (IK-verified:
# the rim-crossing eef must stay LOW enough that the wrist can still point straight down at the extended cavity
# pose — 0.12 clearance -> eef 1.26 = FAIL; ~0.02 -> eef ~1.16 = REACH). So command only ~1-2 cm above the rim.
RIM_CLEAR_HARD = 0.005          # HARD floor: the held bottom MUST end at least this far above the rim ("clearly
# lifted over the drawer wall" — verify_held_above_z checks THIS object height, not an exact eef pose). The rigid
# controller tracks the small commanded clearance closely, so this 0.5 cm floor is comfortably met. Upright stays gated by LTL.
PLACE_NEAR_EDGE_BIAS = 0.07     # place: bias the rim-crossing / drop point this far toward the robot (+p = the
# near-robot edge of the open cavity) instead of the exact cavity centre. The FAR cavity centre is at the arm's
# TOP-DOWN orientation-reachability ceiling for a TALL object (the eef must be HIGH to clear the rim AND extended
# FORWARD to the centre, where the wrist can no longer point straight down). Pulling +p toward the robot cuts the
# forward extension and recovers the reach (IK-verified on task_0007: cavity-centre z=1.15 is knife-edge; +0.03..+0.18
# all REACH with margin). Clamped to keep the object fully inside the drawer interior (off the near wall). Harmless
# for short objects (already reachable; the drop just shifts a few cm), rescues tall ones -> family-wide default.
PLACE_WALL_MARGIN = 0.02        # keep the placed object's near edge this far inside the open-drawer near wall
#                                 (clamp in _carry_target_xy: pc <= p_hi - obj_half - PLACE_WALL_MARGIN).
TARGET_D_SHIFT_BAND = (0.18, 0.30)
LOWER_IN_MAX = 0.35       # place: cap the final straight descent into the OPEN drawer. This chest's drawer
#                           is DEEP (rim 0.734, interior floor ~0.47), so the 0.21 m target needs a ~0.26 m
#                           descent to sit fully below the rim (else it jams the close) — NOT the ~10 cm
#                           first assumed. The descent stops physically on the interior floor; no free-fall.
REACH_TOL = 0.05          # eef-reached-target tolerance for the place lift/over stuck check
# Drawer prismatic-joint resistance. Default (damping 5.0 / friction 0.30) stalls a position-controlled
# push; soften it for the whole demo so open + close both move. STIFFEN it only while the arm reaches
# OVER the open drawer to place (Phase 3), so a stray brush can't shove the soft drawer shut.
DRAWER_DAMPING, DRAWER_FRICTION = 0.1, 0.01
DRAWER_HOLD_DAMPING, DRAWER_HOLD_FRICTION = 5.0, 0.5


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


def _pose_to_mat(pos, quat):
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


class CabinetSkeleton(FamilySkeleton):
    name = "cabinet"

    def __init__(self, db: dict | None = None):
        self._db = db if db is not None else load_db()
        self._geom = json.load(open(CAB_GEOM_PATH))
        self._p: dict | None = None         # per-task cache (one task per process)

    def relocate_prefer_top_down(self) -> bool:
        """Phase-1 relocate (target + obstacle) prefers real top-down grasps: the straight vertical
        relocate lift stalls a side grasp's wrist at a limit, where a top-down grasp lifts cleanly."""
        return True

    def grasping_mode(self) -> str:
        """Sticky AG for this family: the target slabs (e.g. fruitcake, bbox 13.9x8.9x4.9cm) are
        wider than the 8cm gripper in BOTH horizontal axes, so on the table they can only be
        edge-pinched (one finger on the top face, one on a side face) — a non-antipodal grasp whose
        lower finger just shoves the slab away (assisted AG returns -1, no attach). Sticky attaches
        on first finger contact, so relocate/place teach the SEQUENCE. Cabinet eval must match."""
        return "sticky"

    def relocate_open_dir(self, ctx: TaskContext):
        """Drawer-OPEN world xy (``layout.d``): relocate grasps trail the wrist toward +open so the
        approach leans toward the cabinet (-open) and the arm body stays on the open side, off the
        closed cabinet — picking from the cabinet side jams the wrist/arm against it (the g6 failure)."""
        return np.asarray(self._prepare(ctx)["layout"].d, float)

    # ---- per-task preparation (live env reads; cabinet/table/robot are fixed) ---------
    def _prepare(self, ctx: TaskContext) -> dict:
        if self._p is not None:
            return self._p
        env, diag = ctx.env, ctx.diagnostics
        ci = diag["cabinet_info"]
        cab = env.scene.object_registry("name", ci["name"])
        cp, cq = cab.get_position_orientation()
        rp, _ = ctx.robot.get_position_orientation()
        # j_current is left at the diagnostics' open_fraction CALIBRATION CONSTANT (not the live joint):
        # open_distance and the cavity-centre prediction were tuned around it, and the demo's open/place
        # geometry stays identical to the long-stable 0.2-open pipeline. (Method 2 only spawns the drawer
        # closed and drops the redundant Phase-1 close; it must NOT retune the open stroke.)
        layout = CG.build_layout(diag, self._geom, cab_pos=_np(cp), cab_quat=_np(cq),
                                 robot_xy=_np(rp)[:2])

        obstacle = env.scene.object_registry("name", diag["obstacle_info"]["name"])
        # Phase-1 relocation list, in a role-FIXED order: the OBSTACLE is cleared FIRST (it parks at
        # the base's perpendicular foot on the near-base edge), THEN the target (it parks +d-staggered
        # along the SAME near-base edge, where Phase 3 re-picks it — both stay in front of the cabinet,
        # in reach). Only objects actually in the drawer's opening sweep are moved — the
        # ``in_path`` flag (bench diagnostics, §0b reliable) is the per-role "needs relocating" test —
        # so an object already clear of the path is skipped. This covers all bench layouts: target-only
        # blocks, obstacle-only blocks, or both. (e.g. task_0000: obstacle already clear → skipped, the
        # list is just [target].)
        blockers = []
        if diag["obstacle_info"]["placement"]["in_path"]:
            blockers.append(("obstacle", obstacle))
        if diag["target_info"]["placement"]["in_path"]:
            blockers.append(("target", ctx.target))

        j = cab.joints[ci["joint"]]
        try:
            j.damping, j.friction = DRAWER_DAMPING, DRAWER_FRICTION
            print(f"[datagen.cab] drawer joint softened: damping={j.damping} friction={j.friction}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[datagen.cab] drawer-soften failed: {e}", flush=True)

        try:                                            # drawer top rim z (target must clear it to drop in)
            drawer_top_z = float(_np(cab.links[ci["link"]].aabb[1])[2])
        except Exception:  # noqa: BLE001
            drawer_top_z = float(_np(cab.aabb[1])[2])
        self._p = {
            "layout": layout, "cab": cab, "cab_name": ci["name"], "obstacle": obstacle,
            "obstacle_key": self._obj_key(env, diag["obstacle_info"]["name"]),
            "blockers": blockers, "drawer_top_z": drawer_top_z,
        }
        print(f"[datagen.cab] prepared: blockers={[r for r, _ in blockers]} rim={drawer_top_z:.3f}", flush=True)
        return self._p

    @staticmethod
    def _obj_key(env, name):
        o = env.scene.object_registry("name", name)
        return f"{o.category}/{o.model}"

    def _local_grasps(self, key: str) -> list[dict]:
        """Prefer the programmatic straight-down (sticky) grasps when present — the object grasps used
        for relocate + place, which hang the object level over the CoM (no tilt -> no upright violation,
        no off-centre wrist sag). Fall back to all annotations when there is no topdown_gen entry (e.g.
        the drawer-HANDLE grasps, which keep their side approach). See generate_topdown_grasps.py."""
        grasps = self._db.get("objects", {}).get(key, {}).get("grasps", [])
        td = [g for g in grasps if g.get("source") == "topdown_gen"]
        return td if td else grasps

    # ---- grasp candidates = the TARGET's annotated grasps (driver scores + varies; these are
    # ---- the grasps used to RELOCATE the target in Phase 1; the Phase-3 PLACE grasp is chosen
    # ---- separately in select_grasps) ------------------------------------------------
    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        self._prepare(ctx)
        tp, tq = ctx.target.get_position_orientation()
        T_obj = _pose_to_mat(_np(tp), _np(tq))
        out = []
        for g in self._local_grasps(ctx.target_key):
            T = T_obj @ _pose_to_mat(g["position"], g["orientation_xyzw"])
            out.append(GraspCand(id=int(g["id"]), eef_pos=T[:3, 3],
                                 eef_quat=Rot.from_matrix(T[:3, :3]).as_quat(),
                                 approach=g.get("approach_hint", "top_down")))
        return out

    # ---- per-task pre-selection of the family-internal handle / obstacle / place grasps ----
    def select_grasps(self, ctx: TaskContext, world, robot) -> None:
        """cuRobo-score the family-internal aux grasps once before the variant loop (the driver only
        scores the target grasp the sampler iterates): the drawer handle (open + close), the movable
        obstacle, and the Phase-3 PLACE grasp on the target."""
        P = self._prepare(ctx)
        hkey = f"{self._geom['category']}/{self._geom['model']}"
        # A handle grasp must be COLLISION-FREE + reachable at the 3 binding poses. The handle is reached
        # in Phase 2 (open) and Phase 4 (close) — AFTER the in-path blockers are relocated aside — so it is
        # scored in that POST-relocate world (same principle as the place grasp): drop the cabinet (the
        # handle rides it) AND the in-path blockers (`P["blockers"]`, gone by then). A genuinely wedged,
        # NON-in-path obstacle is NOT in P["blockers"], so it stays present and correctly rejects a handle
        # point that would ram it. (Scoring with the in-path blockers present made clear=[] on both-mode
        # tasks like task_0002, where the tall obstacle straddles the drawer front at scoring time.)
        #   pre     = close PRE-pose  (live joint, 0.10 standoff, +Y-most reach)
        #   contact = handle CONTACT  (live joint, no standoff)  <- rejects wedged-obstacle-colliding points
        #   close   = push ENDPOINT   (target joint, no standoff, -Y-most reach)
        handle_ignore = [P["cab"], *(obj for _, obj in P["blockers"])]
        pre = self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=handle_ignore,
                              on_handle=True, joint=None, standoff_m=0.10)
        contact = set(self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=handle_ignore,
                                      on_handle=True, joint=None, standoff_m=0.0))
        close = set(self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=handle_ignore,
                                    on_handle=True, joint=self.close_target_joint(), standoff_m=0.0))
        # WIDEST reliable open: the chosen handle grasp must ALSO be reachable at the open-END (the
        # handle pulled out by d_open), not only at the closed pose — else the straight SERVO pull winds
        # the wrist past a limit and the drawer half-opens (the 4/5 undershoot failures). Search the open
        # distance from the full stroke DOWN; the widest d_open whose open-end keeps a clear handle grasp
        # is THIS cabinet+robot geometry's max reliable open (auto-adapts to the base<->cabinet reach).
        L = P["layout"]
        clear, P["open_dist"] = [], 0.0
        for d_open in (max(0.0, L.stroke - DRAWER_OPEN_MARGIN) * f for f in DRAWER_OPEN_FRACS):
            open_end = set(self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=handle_ignore,
                                           on_handle=True, joint=float(d_open), standoff_m=0.0))
            # the FINAL close must RE-GRASP the handle at the open-end (close_pre, standoff STANDOFF). A
            # full-open handle gets pushed to the far +p reach edge where that re-grasp IK-fails on
            # orientation (the close_pre_final plan_fail), so ALSO require the close PRE-grasp reachable at
            # d_open. The widest d_open clearing BOTH the open pull (open_end, contact) AND the close
            # re-grasp (open_pre, standoff) = the max open the demo can still close from — adapts per cabinet.
            open_pre = set(self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=handle_ignore,
                                           on_handle=True, joint=float(d_open), standoff_m=STANDOFF))
            cand = [g for g in pre if g in contact and g in close and g in open_end and g in open_pre]
            if cand:
                clear, P["open_dist"] = cand, float(d_open)
                break
        if not clear:                                  # no open-end-reachable grasp -> legacy behaviour
            clear = [g for g in pre if g in contact and g in close]
            P["open_dist"] = CG.open_distance(0.0, L.remaining_travel, np.random.default_rng(0))
        print(f"[datagen.cab] max reliable open_dist={P['open_dist']:.3f} of stroke {L.stroke:.3f} "
              f"({len(clear)} handle grasps clear at the open-end)", flush=True)
        # among the clear-reachable handle points, prefer the one CLOSEST to the robot base = farthest
        # from the in-path blocker, where an open-gripper finger reliably catches the drawer front.
        rp = _np(ctx.robot.get_position_orientation()[0])[:2]

        def _d2base(gid):
            g = self._grasp_record(hkey, gid)
            T = self._handle_grasp_T(P["cab"], ctx, _pose_to_mat(g["position"], g["orientation_xyzw"]))
            return float(np.linalg.norm(T[:2, 3] - rp))

        P["handle_gids"] = sorted(clear, key=_d2base)
        print(f"[datagen.cab] handle gids: clear={clear} -> base-nearest-first={P['handle_gids']}", flush=True)
        if any(role == "obstacle" for role, _ in P["blockers"]):
            P["obstacle_gids"] = self._score_aux(world, robot, ctx, obj=P["obstacle"],
                                                 key=P["obstacle_key"], ignore=P["obstacle"],
                                                 on_handle=False, prefer_top_down=True,
                                                 prefer_wrist_dir=self.relocate_open_dir(ctx))
            print(f"[datagen.cab] obstacle reachable gids (best-first): {P['obstacle_gids']}", flush=True)
        self._select_place_grasp(ctx, world, robot, P)

    def _select_place_grasp(self, ctx, world, robot, P) -> None:
        """Cache the Phase-3 PLACE-grasp candidate LIST, ADAPTIVELY filtered by whether the grasp can
        actually clear the drawer rim AND still reach the cavity. Two gates, both cuRobo-measured (no
        magic constants), top-down AND side grasps eligible:
          1. reachable to PICK the target at its PREDICTED relocated location;
          2. the over-cavity PLACE pose is IK-solvable. Each grasp grips the object at a different
             height; the eef must rise to ``rim + clearance + (eef height above the object's bottom)``
             to clear the rim, then translate over the cavity AT THAT HEIGHT. A grasp gripping high on
             the object forces the eef toward the arm's vertical-reach ceiling over the (often far)
             cavity → ``place_over`` IK-fails. So we cuRobo-IK each candidate's actual over-cavity pose
             and keep only the ones that solve, DEEPEST-FIRST (lowest lift = most reach margin).
        The deepest grasp is preferred but the surviving list keeps diversity; the rollout + gate are
        still the final judge. If none clear within reach, keep the single deepest as a best-effort.
        Caches ``P['place_gids']``."""
        import torch as th

        from maniguard.data.datagen.executor.grasp_select import MARGIN_FLOOR, joint_margin, roll_variants
        from maniguard.data.datagen.primitives.curobo_seg import solve_ik

        key = ctx.target_key
        if not self._local_grasps(key):
            return
        obj_pose = self._predicted_target_pose(ctx, P)
        cands = self._score_aux(world, robot, ctx, obj=ctx.target, key=key,
                                ignore=ctx.target, on_handle=False, obj_pose=obj_pose, return_cands=True)
        reach = [c for c in cands if getattr(c, "reachable", False)]
        if not reach:                                  # fall back to spawn-pose reachability
            cands = self._score_aux(world, robot, ctx, obj=ctx.target, key=key,
                                    ignore=ctx.target, on_handle=False, return_cands=True)
            reach = [c for c in cands if getattr(c, "reachable", False)]
        if not reach:
            print("[datagen.cab] place grasp: none reachable -> fallback to target_grasp at runtime", flush=True)
            return

        # --- adaptive height filter: can each grasp clear the rim AND reach the open cavity? ---
        L = P["layout"]
        open_dist = P["open_dist"]                     # max reliable open (reachability search, select_grasps)
        cav_xy = self._predicted_cavity_xy(P, open_dist, self._obj_width(ctx.target_key) / 2.0)
        rim = float(P["drawer_top_z"])
        rim_clear = RIM_CLEAR_BAND[1]                  # conservative: highest lift any draw will ask for
        obj_bottom = float(_np(ctx.target.aabb[0])[2])  # the held object's resting bottom (upright AABB)
        init_q = robot.get_joint_positions()
        world.update_obstacles(ignore_objects=[ctx.target, P["cab"]])   # place_over ignores the cabinet too
        arm_idx = robot.arm_control_idx[robot.default_arm]
        lo = np.asarray(robot.joint_lower_limits)[arm_idx]
        hi = np.asarray(robot.joint_upper_limits)[arm_idx]

        def _pose_margin(pos, quat):
            res = solve_ik(world.motion_gen, robot,
                           th.as_tensor(np.asarray(pos, float), dtype=th.float32),
                           th.as_tensor(np.asarray(quat, float), dtype=th.float32),
                           init_q, timeout=3.0, label="place_carry")
            if res is None:
                return float("-inf")
            q = res.arm_traj[0].detach().cpu().numpy().reshape(-1)   # 1-waypoint IK config (7,)
            return joint_margin(q, lo, hi)

        # Score each grasp on the WORST joint-margin across the two place-carry poses it must hold the
        # eef-rigid object through: the straight lift OVER THE RIM (place_lift — where the palm-flip side
        # grasp stalled) and the horizontal OVER THE CAVITY (place_over). Take the better roll variant.
        scored = []                                    # (worst place margin, lift_z, gid, rolled)
        for c in reach:
            eef_above_bottom = float(c.eef_pos[2]) - obj_bottom
            lift_z = rim + rim_clear + eef_above_bottom
            over_rim = (float(c.eef_pos[0]), float(c.eef_pos[1]), lift_z)   # straight lift over the rim
            over_cav = (float(cav_xy[0]), float(cav_xy[1]), lift_z)         # then horizontal over the cavity
            best = None                                # (worst-pose margin, rolled)
            for roll_idx, quat in enumerate(roll_variants(c.eef_quat)):
                m = min(_pose_margin(over_rim, quat), _pose_margin(over_cav, quat))
                if best is None or m > best[0]:
                    best = (m, bool(roll_idx))
            scored.append((best[0], lift_z, int(c.id), best[1]))
            print(f"[datagen.cab] place-carry g{c.id} ({c.approach}): eef_above_bottom={eef_above_bottom:.3f} "
                  f"lift_z={lift_z:.3f} place_margin={best[0]:.3f} roll={best[1]} "
                  f"ok={best[0] >= MARGIN_FLOOR}", flush=True)
        viable = [(z, gid, roll) for (m, z, gid, roll) in scored if m >= MARGIN_FLOOR]
        if not viable:                                 # none clears the singularity floor → best-effort
            if scored:
                _, _, gid, roll = max(scored, key=lambda s: s[0])   # highest-margin (most slack) best-effort
            else:                                      # nothing IK-solved at all → deepest best-effort
                d = min(reach, key=lambda c: c.eef_pos[2])
                gid, roll = int(d.id), False
            P["place_gids"], P["place_roll"] = [gid], {gid: roll}
            print(f"[datagen.cab] place grasp: NONE clears MARGIN_FLOOR (rim={rim:.3f}, cav_xy="
                  f"{np.round(cav_xy, 3)}) -> best-effort g{gid} roll={roll}", flush=True)
            return
        # Rule 1 — among the singularity-safe place grasps (the MARGIN_FLOOR gate above dropped the
        # palm-flip ones), prefer the one nearest the object's CENTRE along its long axis: an end-grasp on
        # an elongated object slips / swings (the observed below_z + tilt). Weight by aspect ratio so this
        # only bites for long thin objects; lift_z (depth) breaks ties.
        long_local, long_len, short_len = self._object_footprint(key)
        aspect = long_len / max(short_len, 1e-6)
        projs = [float(np.dot(g["position"], long_local)) for g in self._local_grasps(key)]
        center_proj = float(np.mean(projs)) if projs else 0.0

        def _balance(gid):
            g = self._grasp_record(key, gid)
            return abs(float(np.dot(g["position"], long_local)) - center_proj) * aspect

        viable.sort(key=lambda t: (round(_balance(t[1]), 3), t[0]))   # (centre-balance, then deepest)
        P["place_gids"] = [gid for _, gid, _ in viable]
        P["place_roll"] = {int(gid): roll for _, gid, roll in viable}
        print(f"[datagen.cab] place grasp viable (balance-then-deep): "
              f"{[(g, round(_balance(g), 3), round(z, 3), r) for z, g, r in viable]}", flush=True)

    def _carry_target_xy(self, L, open_val: float, obj_half: float):
        """The place rim-crossing / drop XY: the exposed-cavity centre along the slide (cabinet front + half
        the open span), BIASED toward the robot (+p near edge) by PLACE_NEAR_EDGE_BIAS so the HIGH rim-crossing
        stays inside the arm's top-down reach, clamped to keep the object fully inside the open-drawer interior
        (off the near wall). Used by BOTH the place-grasp reachability check (_predicted_cavity_xy) and the
        runtime ``over_cavity``, so the selection and the executed carry aim at the SAME point. If the drawer is
        too narrow in p for the bias to fit, the clamp falls back to the cavity centre (no bias)."""
        dc = (L.d_front - L.j_current) + 0.5 * float(open_val)      # exposed-cavity centre along the slide
        pc = min(L.p_center + PLACE_NEAR_EDGE_BIAS, L.p_hi - obj_half - PLACE_WALL_MARGIN)   # toward robot, off near wall
        pc = max(pc, L.p_center)                                    # never bias AWAY from the robot
        return L.to_world(dc, pc)

    def _predicted_cavity_xy(self, P, open_dist: float, obj_half: float):
        """The place rim-crossing / drop xy used by the place-grasp reachability check — the near-edge-biased
        carry point (see ``_carry_target_xy``), matching the runtime ``over_cavity`` EXACTLY so the selection and
        the executed carry aim at the SAME point (else selection rejects every grasp it would actually use)."""
        return np.asarray(self._carry_target_xy(P["layout"], open_dist, obj_half)[:2], float)

    def _predicted_target_pose(self, ctx, P):
        """Where the target will sit when Phase 3 picks it: its Phase-1 relocated (base-side) xy if it
        is in the drawer's path, else its spawn pose. Orientation + z kept from spawn (the relocate
        sets it back upright on the table). Mirrors the Phase-2 ``blocker_placement`` call exactly
        (``open_distance`` is deterministic, target placement is rng-independent) so the prediction
        matches the actual relocation."""
        tp, tq = (_np(v) for v in ctx.target.get_position_orientation())
        if not any(role == "target" for role, _ in P["blockers"]):
            return tp, tq                              # not in path -> picked at spawn
        L = P["layout"]
        open_dist = P["open_dist"]                     # max reliable open (reachability search, select_grasps)
        place_xy = CG.blocker_placement(L, tp[:2], self._obj_width(ctx.target_key) / 2, "target",
                                        open_dist)
        if place_xy is None:
            return tp, tq
        return np.array([place_xy[0], place_xy[1], tp[2]]), tq

    def _score_aux(self, world, robot, ctx, *, obj, key, ignore, on_handle: bool,
                   joint=None, standoff_m: float = 0.10, obj_pose=None,
                   return_cands: bool = False, prefer_top_down: bool = False, prefer_wrist_dir=None):
        """Build world-frame GraspCands for ``key``'s annotation grasps (handle ones ride the drawer
        link at ``joint``; other objects sit at ``obj_pose`` if given, else their live pose),
        cuRobo-score them at ``standoff_m``, and return the reachable ids best-first. With
        ``return_cands`` return the full scored GraspCand list instead (callers that need the world
        eef poses — e.g. the place-height reach filter — read ``c.eef_pos`` / ``c.eef_quat``)."""
        from maniguard.data.datagen.executor.grasp_select import score_grasps
        if on_handle:
            base = None
        elif obj_pose is not None:
            base = _pose_to_mat(_np(obj_pose[0]), _np(obj_pose[1]))
        else:
            base = _pose_to_mat(*[_np(v) for v in obj.get_position_orientation()])
        cands = []
        for g in self._local_grasps(key):
            glocal = _pose_to_mat(g["position"], g["orientation_xyzw"])
            T = self._handle_grasp_T(obj, ctx, glocal, joint=joint) if on_handle else base @ glocal
            cands.append(GraspCand(id=int(g["id"]), eef_pos=T[:3, 3],
                                   eef_quat=Rot.from_matrix(T[:3, :3]).as_quat(),
                                   approach=g.get("approach_hint", "side" if on_handle else "top_down")))
        if not cands:
            return [] if not return_cands else []
        # roll-disambiguation OFF for the handle (its own contact/close gates already filter it, and the
        # handle grasp is not propagated through _grasp_pose's roll branch); ON for the object grasps.
        score_grasps(world, robot, ignore, cands, standoff_m=standoff_m,
                     roll_disambig=not on_handle, prefer_top_down=prefer_top_down,
                     prefer_wrist_dir=prefer_wrist_dir)   # reachable + chosen_roll + wrist-open + sort
        if not on_handle:                          # cache the winning roll + top-down so selection threads it
            roll_cache = self._p.setdefault("roll_by_key_gid", {})
            td_cache = self._p.setdefault("topdown_by_key_gid", {})
            for c in cands:
                roll_cache[(key, int(c.id))] = bool(getattr(c, "chosen_roll", False))
                td_cache[(key, int(c.id))] = bool(getattr(c, "is_top_down", False))
        if return_cands:
            return cands
        return [c.id for c in cands if c.reachable]

    # ---- the full 5-phase sequence ---------------------------------------------------
    def derive_segments(self, ctx: TaskContext, target_grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        P = self._prepare(ctx)
        L = P["layout"]
        rng = np.random.default_rng(params.seed)
        open_dist = P["open_dist"]                     # max reliable open (reachability search, select_grasps)
        # per-demo diversity draws (deterministic per variant seed)
        d_shift = float(rng.uniform(*TARGET_D_SHIFT_BAND))     # dim 2: target's landing point along the edge
        rim_clear = float(rng.uniform(*RIM_CLEAR_BAND))        # dim 1: carry height above the rim
        place_gids = P.get("place_gids") or []                 # dim 3: reachable place grasp, balance-then-deep
        k_draw = int(params.seed) % 1000                       # canonical draw 0 -> the most robust (most central,
        place_gid = (int(place_gids[0]) if k_draw == 0          # deepest) grasp; diversity draws sample the rest
                     else int(place_gids[int(rng.integers(len(place_gids)))])) if place_gids else None
        print(f"[datagen.cab] diversity: d_shift={d_shift:.3f} rim_clear={rim_clear:.3f} place_gid={place_gid}", flush=True)

        segs: list[MotionSegment] = []
        # The bench spawns the drawer CLOSED (Method 2), so there is no redundant initial close — the
        # demo is the natural relocate → open → place → close. (A tall in-path obstacle used to block the
        # old Phase-1 handle reach; spawning closed removes that failure mode entirely.)

        # Phase 1 — move each path-blocking object aside (clean pick-and-place). Resolve the TARGET's
        # destination FIRST (deterministic) so the obstacle can park clear of its parked bbox.
        target_dc = target_dh = None
        if any(role == "target" for role, _ in P["blockers"]):
            t_ph, t_dh = self._relocate_halves(ctx.target_key)
            t_place = CG.blocker_placement(L, _np(ctx.target.get_position_orientation()[0])[:2],
                                           self._obj_width(ctx.target_key) / 2, "target", open_dist,
                                           d_shift=d_shift, p_half=t_ph, d_half=t_dh)
            target_dc, target_dh = float(np.asarray(t_place) @ L.d), t_dh
        for role, obj in P["blockers"]:
            key = ctx.target_key if role == "target" else P["obstacle_key"]
            gid = target_grasp.id if role == "target" else self._best_grasp_id(key)
            # singularity-aware roll for this relocate grasp: the target carries it on the scored cand;
            # the obstacle's was cached by select_grasps' _score_aux when it scored the obstacle grasps.
            roll = (bool(getattr(target_grasp, "chosen_roll", False)) if role == "target"
                    else bool(P.get("roll_by_key_gid", {}).get((key, int(gid)), False)))
            ph, dh = self._relocate_halves(key)
            place_xy = CG.blocker_placement(
                L, _np(obj.get_position_orientation()[0])[:2], self._obj_width(key) / 2, role,
                open_dist, d_shift=d_shift, p_half=ph, d_half=dh,
                avoid_dc=(target_dc if role == "obstacle" else None),
                avoid_half=(target_dh or 0.0 if role == "obstacle" else 0.0))
            if place_xy is None:                       # no room → a "third-type" task to report
                return []
            print(f"[datagen.cab] relocate[{role}] "
                  f"obj_xy={_np(obj.get_position_orientation()[0])[:2].round(3)} "
                  f"-> place_xy={np.asarray(place_xy).round(3)} roll={roll}", flush=True)
            segs += self._relocate_blocker(ctx, key, gid, role, place_xy, params, roll=roll)

        # Phase 2 — open the drawer to its widest.
        segs += self._open_drawer(ctx, dist=open_dist)

        # Phase 3 — pick the target (now at its moved spot) + place it INTO the open drawer.
        segs += self._place_in_drawer(ctx, target_grasp, params, rim_clear, place_gid)

        # Phase 4 — close the drawer.
        segs += self._close_drawer(ctx, tag="final")
        return segs

    # ---- segment builders ------------------------------------------------------------
    def _relocate_blocker(self, ctx, key, gid, role, place_xy, params, *, roll: bool = False) -> list[MotionSegment]:
        """Phase 1: a clean pick-and-place (the boxy clutter pattern). FREE cuRobo to the grasp
        standoff, SERVO straight down + close, SERVO small lift off the table, FREE cuRobo transit to
        above the place spot (collision-aware — avoids the cabinet + the other blocker), SERVO
        straight down, open, lift off. The straight descents/lifts are SERVO (pure IK) — LINEAR's
        partial-pose query is broken on this fork and drifts the carried object off the place spot.
        The blockers relocate toward the robot / along the table front, so the FREE transit clears
        the cabinet with a short collision-aware path."""
        obj_name = ctx.target_name if role == "target" else ctx.diagnostics["obstacle_info"]["name"]
        cab = self._prepare(ctx)["cab_name"]
        e = {"obj": role, "key": key, "grasp_id": gid, "grasp_roll": roll}
        held = {"held_name": obj_name}
        q0 = np.array([0.0, 0.0, 0.0, 1.0])            # placeholder; resolve_compute fills the real target
        xy = {"xy": [float(place_xy[0]), float(place_xy[1])]}
        support_top = float(_np(ctx.support.aabb[1])[2]) if ctx.support is not None else 0.0
        segs = [
            # FREE to the grasp standoff ABOVE the blocker. Ignore NOTHING — cuRobo must AVOID the blocker's
            # body AND the cabinet (§4 Flaw-1: a FREE connector avoids the cabinet + the live drawer). Ignoring
            # the blocker routed the approach straight through + knocked it; only the final straight descent
            # ignores the blocker, once the gripper is poised to enclose it.
            MotionSegment("pick_pre", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**e, "standoff": params.standoff_m}, path_begin=True),
            MotionSegment("pick_descend", q0[:3], q0, mode=Mode.SERVO, grip=Grip.CLOSE, grip_steps=8,
                          compute="grasp", extra={**e, "standoff": 0.0},
                          ignore_objects=(obj_name,), ignore_clutter=True,
                          reach_tol_m=REACH_TOL),   # don't close on air: if the SERVO descend stalls short
                          # (e.g. the arm jams on the cabinet, eef_err >> tol), fail "stuck" BEFORE the close
                          # so the driver retries another grasp — same guard place_descend already carries.
            # verify the blocker ACTUALLY left the table — a failed grip (the eef lifts empty) otherwise
            # leaves it in place, and the drawer knocks it over later with no early signal.
            MotionSegment("pick_lift", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lift_above_support", extra={**held, "clear": LIFT_CLEAR},
                          verify_held_above_z=support_top + 0.05, ignore_objects=(cab,)),
            MotionSegment("move_transit", q0[:3], q0, mode=Mode.FREE, attach=True, grip=Grip.HOLD,
                          compute="move_to", extra={**xy, **held}),     # avoid ALL (incl. cabinet)
        ]
        # Only ELONGATED blockers need re-orienting: roll the held object so its LONG axis lies ∥ the
        # table edge before setting down → a long handle ends up flush ALONG the edge, not jutting into
        # the opening corridor / toward the open drawer where the re-pick gripper would collide. Compact
        # / symmetric objects (a shaker) skip this — force-rotating a tippy symmetric object only tips it.
        if self._is_elongated(key):
            segs.append(MotionSegment("move_edge_yaw", q0[:3], q0, mode=Mode.SERVO, attach=True,
                                      grip=Grip.HOLD, compute="edge_yaw", extra={**e, **held},
                                      ignore_objects=(cab,)))
        segs += [
            MotionSegment("move_place", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lower_to_support", extra={**xy, **held},   # RE-TARGET place_xy (correct any
                          ignore_objects=(cab,)),                             # carry/reorient drift) + lower the
                          #                                                     held bottom onto the table
            MotionSegment("move_release", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=6,
                          compute="hold", ignore_objects=(cab,)),       # open in place (gripper wraps the object)
            MotionSegment("move_retreat", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="up", extra={"dz": RETREAT_DZ}, ignore_objects=(cab,)),
            # Return toward HOME by RETRACING the WHOLE relocate joint path backwards (it was just executed,
            # so reversing it joint-by-joint is drift-free + collision-free — no replan). Replaying ~80% of it
            # leaves the arm ~20% into the path = near the HOME config the FIRST pick solved from. The next FREE
            # plan (re-pick target / reach handle) then starts from that good config instead of the folded
            # post-place pose cuRobo trajopt can't route out of (it has no graph search to find a detour).
            MotionSegment("move_return", q0[:3], q0, replay_reverse_path=True, replay_frac=0.8,
                          grip=Grip.OPEN, ignore_objects=(cab,)),
        ]
        return segs

    def _open_drawer(self, ctx, *, dist: float) -> list[MotionSegment]:
        """Phase 2: cuRobo-reach the handle pre-pose, CLOSE on the handle, a straight pure-IK SERVO
        PULL (+slide) drags it open by ``dist`` (the softened joint lets the pull move it), open the
        gripper, retreat along +open so the fingers clear the handle before the next recenter."""
        cab = (self._prepare(ctx)["cab_name"],)        # grasping the handle = don't avoid the cabinet
        e = {"obj": "handle"}
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        return [
            MotionSegment("handle_pre_open", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**e, "standoff": STANDOFF}, ignore_objects=cab),
            MotionSegment("handle_grasp_open", q0[:3], q0, mode=Mode.SERVO, grip=Grip.CLOSE, grip_steps=8,
                          compute="grasp", extra={**e, "standoff": 0.0}, ignore_objects=cab),
            MotionSegment("drawer_open", q0[:3], q0, mode=Mode.SERVO, grip=Grip.HOLD, carry_closed=True,
                          compute="drawer", extra={"to": "open", "dist": float(dist)}, ignore_objects=cab),
            MotionSegment("handle_release_open", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=6,
                          compute="hold", ignore_objects=cab),
            # clear the fingers off the (horizontal-bar) handle by lifting UP, NOT sliding further +open:
            # after a FULL open the handle sits at the arm's forward reach edge, so a +open retreat
            # servo_ik_fails — lifting straight up off the bar clears the fingers and stays reachable.
            MotionSegment("handle_back_open", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="up", extra={"dz": HANDLE_BACK_DIST}, ignore_objects=cab),
        ]

    def _place_in_drawer(self, ctx, target_grasp, params, rim_clear: float = RIM_CLEARANCE,
                         place_gid: int | None = None) -> list[MotionSegment]:
        """Phase 3 — pick the target with the sampled place grasp, then carry it into the open drawer
        along the inverted-"门" of 4 must-reach waypoints (all-SERVO straight, pure IK nearest-seed →
        the held object only translates):
          D_grasp   `place_descend`  — grasp at the relocated target
          D_up_src  `place_lift`     — ↑ straight UP, target bottom hard-verified above the rim
          D_up_dst  `place_across`   — → HORIZONTAL to above the exposed-cavity geometric centre I
          D_place   `place_lower`    — ↓ straight DOWN into the drawer, then release (no free-fall)
        Orientation (fit / upright + finger-rail ⊥ opening) is done where the wrist has ROOM — a SERVO
        segment snaps its goal orientation at the FIRST step, so folding it into the across would rotate
        on the tight source side. ELONGATED: the ~90° fit roll LOW/early (`place_reorient`, off the
        table, before the lift) — at rim height cuRobo could only roll by dropping below the rim.
        COMPACT: the small upright + rail-clear at D_up_dst (`place_upright`/`place_rail_clear`, the roomy
        end). The singularity-aware chosen roll threads through all of them (`grasp_roll`)."""
        P = self._prepare(ctx)
        cab, tname = P["cab_name"], ctx.target_name
        gid = place_gid if place_gid is not None else target_grasp.id   # sampled place grasp (id 0 is valid!)
        roll = bool(P.get("place_roll", {}).get(int(gid), False))      # singularity-aware roll for this place grasp
        e = {"obj": "target", "key": ctx.target_key, "grasp_id": gid, "grasp_roll": roll}
        held = {"held_name": tname}
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        rim = float(P["drawer_top_z"])                 # open-drawer wall top — the target bottom must clear THIS
        # ELONGATED target → roll its long axis ∥ the wider exposed cavity axis (fit_yaw); COMPACT target
        # → its yaw is irrelevant to fit, so keep the finger-rail ⊥ the opening (rail_clear) to clear the
        # upper drawer's handle on descent (preserves the task_0000 behaviour).
        elongated = self._is_elongated(ctx.target_key)
        print(f"[datagen.cab] place_in_drawer: target grasp id={gid} rim={rim:.3f} "
              f"{'elongated->fit_yaw' if elongated else 'compact->rail_clear'}", flush=True)
        head = [
            # FREE to the deep-grasp standoff ABOVE the target. Ignore NOTHING — cuRobo must AVOID the
            # target's body AND the cabinet + the live OPEN drawer (§4 Flaw-1: this is the re-pick approach
            # AFTER the drawer opened; ignoring the cabinet dropped the slid-out drawer link from the world,
            # so cuRobo routed the approach straight THROUGH the open drawer). The standoff sits above the
            # target so cuRobo still reaches it while avoiding the body; only the final straight descent
            # (place_descend) ignores the target, once the gripper is poised to enclose it.
            MotionSegment("place_pre_grasp", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**e, "standoff": params.standoff_m}),
            MotionSegment("place_descend", q0[:3], q0, mode=Mode.SERVO, grip=Grip.CLOSE, grip_steps=8,
                          compute="grasp", extra={**e, "standoff": 0.0},
                          ignore_objects=(tname,), ignore_clutter=True),
        ]
        # ↑ lift the target bottom above the rim + clearance, HARD-VERIFY it cleared before ANY lateral
        # move (a lift that stays below the rim catches it + rams the drawer).
        # Success = the held bottom CLEARED the rim by RIM_CLEAR_HARD (the functional "lifted over the wall"
        # truth), NOT an exact eef pose: the asymmetric sticky slab sags the compliant wrist ~0.07 m, so a
        # reach_tol on the eef would fail a lift that actually cleared. Upright is still gated by the LTL gate.
        place_lift = MotionSegment("place_lift", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                                   compute="lift_over_rim", extra={**held, "clearance": rim_clear},
                                   verify_held_above_z=rim + RIM_CLEAR_HARD, ignore_objects=(cab,))
        if elongated:
            # ELONGATED: the fit roll is ~90° (relocate parks ∥ edge, the cavity wants ∥ width). Do it LOW
            # (just off the table) where the arm has wrist room — at the rim-clearing height (z≈1 m, the
            # reach ceiling) cuRobo can only achieve the roll by DROPPING the eef, which drags the held
            # bottom back below the rim. So: small lift off the table → fit roll (FREE, smooth) → THEN lift
            # the already-fitted object straight over the rim → over the cavity (toward the robot).
            mid = [
                MotionSegment("place_clear_lift", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                              compute="lift_above_support", extra={**held, "clear": 0.12},
                              ignore_objects=(cab,)),
                # FREE reorient (object attached) — AVOID the cabinet + open drawer (§4/§5: the held object
                # rolls in free space just off the table; cuRobo plans the roll around the cabinet).
                MotionSegment("place_reorient", q0[:3], q0, mode=Mode.FREE, attach=True, grip=Grip.HOLD,
                              compute="fit_yaw", extra={**e, **held}),
                place_lift,
                MotionSegment("place_across", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                              compute="over_cavity", extra={**e, **held},
                              reach_tol_m=REACH_TOL, reach_xy_only=True, verify_held_above_z=rim, ignore_objects=(cab,)),
            ]
        else:
            # COMPACT: lift over the rim, over the cavity CENTRE, square to vertical, then finger-rail ⊥
            # opening to clear the upper drawer's handle on descent (the original task_0000 placement).
            mid = [
                place_lift,
                MotionSegment("place_across", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                              compute="over_cavity", extra={**held},
                              reach_tol_m=REACH_TOL, reach_xy_only=True, verify_held_above_z=rim, ignore_objects=(cab,)),
                MotionSegment("place_upright", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                              compute="uprightify", extra={**e, **held}, ignore_objects=(cab,)),
                MotionSegment("place_rail_clear", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                              compute="rail_clear", extra={**held}, ignore_objects=(cab,)),
            ]
        tail = [
            # ↓ DOWN ~10 cm to set the bottom near the drawer floor, upright the WHOLE way (no free-fall —
            # a tall object dropped 0.3-0.5 m tumbles on impact).
            MotionSegment("place_lower", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lower_to_floor", extra={**held, "max_dz": LOWER_IN_MAX},
                          ignore_objects=(cab,), path_begin=True),   # record the descent so the empty gripper
            #                                                          retraces it back out (reverse-replay below)
            MotionSegment("place_release", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=10,
                          compute="hold", ignore_objects=(cab,)),
            # RETRACE the recorded lower-in REVERSED — the empty (open) gripper rises back out along the
            # exact entry lane to the descent-START config ABOVE the cavity centre. That start config was
            # reached cleanly by the carry, so the following over_handle + close cuRobo plan from there is
            # short + reliable — vs lifting straight up out of the deep place pose, which left a contorted
            # config cuRobo had to route around in a 90+-waypoint winding path (then timed out at the push).
            MotionSegment("place_lift_out", q0[:3], q0, replay_reverse_path=True, replay_frac=1.0,
                          grip=Grip.OPEN, ignore_objects=(cab,)),
            # PARTIAL translate toward the close-handle pre-grasp XY (OVER_HANDLE_FRAC of the way, z held):
            # gets the empty gripper out of the cavity centre into the open space toward the drawer front so
            # the following COLLISION-AWARE close_pre has a short, door-clear reach. (Full=contorted config,
            # none=long winding reach; the close approach itself avoids the cabinet — close_pre no longer
            # ignores it — so the gripper no longer clips the drawer door on the way in.)
            MotionSegment("place_toward_handle", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="over_handle", extra={"standoff": STANDOFF, "frac": OVER_HANDLE_FRAC},
                          ignore_objects=(cab,)),
        ]
        return head + mid + tail

    def close_target_joint(self) -> float:
        """Target drawer joint for a 'closed' demo: a small fraction of the spawn opening
        (``j_extract``), comfortably below OmniGibson's Open threshold (5% of stroke) so the
        ``closed`` predicate holds, while leaving a sliver so we never ram the drawer fully shut."""
        return float(self._geom["j_extract"]) * CLOSE_FRACTION

    def _close_drawer(self, ctx, *, tag: str) -> list[MotionSegment]:
        """Phase 4: close the drawer by GRASPING the handle and sliding it shut — the exact INVERSE of the
        open pull (as in teleop), NOT an open-finger push. cuRobo-reach a collision-aware handle pre-pose
        (gripper OPEN), SERVO straight onto the handle + CLOSE on it, then a straight pure-IK SERVO slides
        the GRIPPED handle to the closed joint: the gripper is rigidly coupled to the handle, so the soft
        drawer follows it continuously.

        Why not the old open-finger push: a flat finger pushing the sliding drawer FRONT is an unstable
        contact — under the rigid controller it stick-slips (stall→slip→jolt), and each jolt impacts the
        drawer and topples the just-placed object inside (measured: close_push velocity stutters, object
        tips). A grasp removes the contact entirely. Reachability holds: the close grasps the handle at its
        OPEN (pulled-out, near-robot = easy) position and slides it back to the CLOSED position — which is
        exactly where the OPEN sequence first grasped it, so both endpoints are known-reachable.

        The handle is released in place; the drawer stays shut = the task success state (``inside &
        closed``). Method-2 closes once (tag='final'); the engine's end-of-run settle + success check
        then confirm it on the settled state."""
        cab = (self._prepare(ctx)["cab_name"],)         # gripping the handle = don't avoid the cabinet
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        h = {"obj": "handle"}
        tj = self.close_target_joint()                  # ~88% closed (a sliver below the Open threshold)
        return [
            # COLLISION-AWARE approach to the handle pre-grasp (gripper OPEN, NO cab-ignore): cuRobo MUST
            # avoid the cabinet door/drawer on the way in to the STANDOFF pre-pose at the live (open) handle.
            MotionSegment(f"close_pre_{tag}", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**h, "standoff": STANDOFF},
                          rot_relax=CLOSE_PRE_ROT_RELAX, pos_relax=CLOSE_PRE_POS_RELAX),
            # SERVO straight onto the handle + CLOSE on it (mirror of handle_grasp_open).
            MotionSegment(f"close_grasp_{tag}", q0[:3], q0, mode=Mode.SERVO, grip=Grip.CLOSE, grip_steps=8,
                          compute="grasp", extra={**h, "standoff": 0.0}, ignore_objects=cab,
                          servo_step_m=0.0025, servo_spw=1),
            # SERVO slide the GRIPPED handle shut to tj (mirror of drawer_open, to=close): gripper HELD closed
            # (carry_closed), the softened drawer follows the handle. FINE waypoints (2.5 mm) + servo_spw=1 =>
            # a continuous UNIFORM-velocity glide (no slam-then-idle stutter) so the gripped handle / drawer
            # doesn't jerk-tip the marginally-stable placed object. Duration-neutral vs the old 1 cm / spw=4.
            MotionSegment(f"close_push_{tag}", q0[:3], q0, mode=Mode.SERVO, grip=Grip.HOLD, carry_closed=True,
                          compute="drawer", extra={"to": "close", "joint": tj}, ignore_objects=cab,
                          servo_step_m=0.0025, servo_spw=1),
            # release the handle in place; the drawer stays shut (success state).
            MotionSegment(f"close_release_{tag}", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=6,
                          compute="hold", ignore_objects=cab),
        ]

    # ---- runtime resolution of cabinet-specific compute tags -------------------------
    def resolve_compute(self, tag: str, seg: MotionSegment, ctx: TaskContext):
        P = self._prepare(ctx)
        arm = ctx.robot.default_arm
        ep, eq = ctx.robot.eef_links[arm].get_position_orientation()
        ep, eq = _np(ep), _np(eq)
        x = seg.extra
        if tag == "hold":
            return ep, eq
        if tag == "up":
            return ep + np.array([0.0, 0.0, float(x.get("dz", RETREAT_DZ))]), eq
        if tag == "grasp":
            return self._grasp_pose(ctx, P, x)
        if tag == "move_to":
            xy = np.asarray(x["xy"], float)
            z = float(x["z"]) if x.get("z") is not None else ep[2]
            return np.array([xy[0], xy[1], z]), eq
        if tag == "over_cavity":                          # D_up_dst — drive the held object to the rim-crossing /
            L = P["layout"]                               # drop XY = exposed-cavity centre along the slide, BIASED
            # toward the robot (+p near edge) by PLACE_NEAR_EDGE_BIAS so the HIGH rim-crossing stays inside the
            # arm's top-down reach. The FAR cavity centre is at the top-down orientation-reachability ceiling for
            # a tall object (IK-verified: centre z=1.15 knife-edge, near-edge REACHES with margin), so we pull the
            # carry/drop toward the robot. Clamped inside the drawer interior (off the near wall). SAME point the
            # place-grasp selection checked (``_predicted_cavity_xy``), so selection and execution stay consistent.
            j = float(P["cab"].get_joint_positions()[self._drawer_jidx(P["cab"], ctx)])   # live open dist
            xy = self._carry_target_xy(L, j, self._obj_width(ctx.target_key) / 2.0)
            print(f"[datagen.cab] over_cavity/across: object -> near-edge-biased drop xy={np.round(xy, 3)} "
                  f"(open={j:.3f}, p_center={L.p_center:.3f} bias+{PLACE_NEAR_EDGE_BIAS}, eef_z held {ep[2]:.3f})", flush=True)
            return np.array([xy[0], xy[1], ep[2]]), eq
        if tag == "extract":                              # slide the eef +slide (away from the cabinet)
            d3 = np.array([P["layout"].d[0], P["layout"].d[1], 0.0])   # at the current height
            return ep + d3 * float(x.get("dist", 0.10)), eq
        if tag == "drawer":
            d3 = np.array([P["layout"].d[0], P["layout"].d[1], 0.0])
            if x["to"] == "open":
                return ep + d3 * float(x["dist"]), eq
            j = float(P["cab"].get_joint_positions()[self._drawer_jidx(P["cab"], ctx)])
            target_j = float(x.get("joint", 0.0))        # slide the GRIPPED handle to this joint (a sliver
            return ep - d3 * (j - target_j), eq          # above 0 leaves it off the hard stop, still "closed")
        if tag == "lift_above_support":                   # relocate lift: raise the held bottom to
            bottom = self._held_bottom(ctx, x["held_name"])   # support_top + clearance (cabinet-independent;
            st = float(_np(ctx.support.aabb[1])[2]) if ctx.support is not None else 0.0   # the FREE transit
            dz = max(0.0, (st + float(x.get("clear", LIFT_CLEAR))) - bottom)              # avoids the cabinet)
            return np.array([ep[0], ep[1], ep[2] + dz]), eq
        if tag == "lift_over_rim":                        # XY fixed; raise the held object's MEASURED
            bottom = self._held_bottom(ctx, x["held_name"])   # bottom to (drawer rim + clearance)
            dz = max(0.0, (P["drawer_top_z"] + float(x["clearance"])) - bottom)
            print(f"[datagen.cab] lift_over_rim: bottom={bottom:.3f} rim={P['drawer_top_z']:.3f} "
                  f"-> dz={dz:.3f} eef_z {ep[2]:.3f}->{ep[2] + dz:.3f}", flush=True)
            return np.array([ep[0], ep[1], ep[2] + dz]), eq
        if tag == "uprightify":                           # reorient the eef in place so the HELD object
            o = ctx.env.scene.object_registry("name", x["held_name"])   # is square to vertical (the grasp
            yaw = float(Rot.from_quat(_np(o.get_position_orientation()[1])).as_euler("zyx")[0])  # is rigid:
            R_obj = Rot.from_euler("z", yaw).as_matrix()                # eef = upright-object × eef-in-object)
            R_eo = Rot.from_quat(self._grasp_record(x["key"], x["grasp_id"])["orientation_xyzw"]).as_matrix()
            R_eo = self._eo_rolled(R_eo, x.get("grasp_roll"))     # stay consistent with the rolled grip
            return ep, Rot.from_matrix(R_obj @ R_eo).as_quat()
        if tag == "rail_clear":                           # roll the eef about the vertical (approach) axis so
            d = P["layout"].d                             # the gripper finger-rail (eef local Y = the long flat
            R = Rot.from_quat(eq).as_matrix()             # slide housing) is ⊥ the opening direction (∥ the
            railY = R[:, 1]                               # upper drawer's handle) → it clears the handle on the
            yaw = float(np.arctan2(d[1], d[0]) + np.pi / 2 - np.arctan2(railY[1], railY[0]))   # descent. Roll
            yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2   # about world Z keeps the object upright; rail is a
            print(f"[datagen.cab] rail_clear: roll {np.degrees(yaw):+.1f}° -> finger-rail ⊥ opening", flush=True)
            return ep, Rot.from_matrix(Rot.from_euler("z", yaw).as_matrix() @ R).as_quat()  # line → |roll|≤90°
        if tag == "edge_yaw":                             # relocate: roll the held object so its LONG axis is
            d = P["layout"].d                             # ∥ the table edge (∥ opening d) → only its SHORT side
            q = self._yaw_align_quat(x["key"], x["grasp_id"], d, eq, rolled=x.get("grasp_roll"))   # corridor-∥
            print("[datagen.cab] edge_yaw: long axis -> ∥ table edge (clear of opening corridor)", flush=True)
            return ep, q                                  # elongated blocker clears the path + the re-pick
        if tag == "fit_yaw":                              # place: roll the held object so its LONG axis ∥ the
            L = P["layout"]                               # WIDER exposed cavity axis → an elongated object lies
            ci = ctx.diagnostics["cabinet_info"]          # across the full drawer width instead of jamming on
            ib = ci.get("interior_bbox") or [0.0, 0.0, 0.0]   # the short, only-partially-open slide depth
            perp_w = float(ib[1 if ci.get("slide_axis") == "x" else 0])
            exposed_d = float(P["cab"].get_joint_positions()[self._drawer_jidx(P["cab"], ctx)])
            target = L.d if exposed_d >= perp_w else L.p
            q = self._yaw_align_quat(x["key"], x["grasp_id"], target, eq, rolled=x.get("grasp_roll"))
            print(f"[datagen.cab] fit_yaw: exposed_slide={exposed_d:.3f} perp_width={perp_w:.3f} -> long axis ∥ "
                  f"{'opening' if exposed_d >= perp_w else 'width'}", flush=True)
            return ep, q
        if tag == "lower_to_floor":                       # descend so the held object's TOP ends just under the
            top = self._held_top(ctx, x["held_name"])     # drawer rim -> it rests on/near the true interior floor
            bottom = self._held_bottom(ctx, x["held_name"])  # WITHOUT overshooting into it. NB: P["layout"].
            rim = float(P["drawer_top_z"])                # drawer_floor_z is the drawer LINK's AABB underside
            #                                               (skirt/front-face bottom), BELOW the real interior
            #   floor panel -- so the old floor-relative descent aimed the bottom under the floor and the rigid
            #   SERVO (no compliance) jammed the object through it, destabilising it + dragging the sliding
            #   drawer. Rim-relative avoids that. (A short object keeps a small release gap; the end-of-rollout
            #   settle gate now catches it if the gap-drop tips it. Measuring the true floor is a deferred follow-up.)
            dz = min(max(0.0, top - (rim - PLACE_Z_MARGIN)), float(x.get("max_dz", LOWER_IN_MAX)))
            print(f"[datagen.cab] lower_to_floor: top={top:.3f}->{top - dz:.3f} bottom={bottom:.3f}->{bottom - dz:.3f} "
                  f"rim={rim:.3f} -> dz=-{dz:.3f}", flush=True)
            return np.array([ep[0], ep[1], ep[2] - dz]), eq
        if tag == "lower_to_support":                     # relocate set-down: lower the held bottom onto the
            bottom = self._held_bottom(ctx, x["held_name"])   # TABLE top (live measured), capped — NOT a fixed
            st = float(_np(ctx.support.aabb[1])[2]) if ctx.support is not None else 0.0   # eef-z (which drives
            dz = min(max(0.0, bottom - (st + PLACE_Z_MARGIN)), float(x.get("max_dz", 0.30)))   # the object INTO
            # RE-TARGET the computed place_xy (the on-table edge spot), not the LIVE eef xy: the carry / edge_yaw
            # reorient drifts the eef ~7cm toward the table edge, and lowering at the drifted xy drops the object
            # off the table (-> falls -> upright violation). The topdown grasp holds the object directly below
            # the eef, so eef xy == object centre == place_xy; driving there lands the object ON the table.
            xyt = np.asarray(x["xy"], float) if x.get("xy") is not None else ep[:2]
            return np.array([float(xyt[0]), float(xyt[1]), ep[2] - dz]), eq
        if tag == "over_handle":                          # translate HORIZONTAL (z + orientation held) a
            hp, _ = self._grasp_pose(ctx, P, {"obj": "handle",   # FRACTION of the way toward the close-handle
                                              "standoff": float(x.get("standoff", STANDOFF))})   # pre-grasp XY
            frac = float(x.get("frac", 1.0))              # frac<1 stops PARTWAY (the rest is the cuRobo close_pre)
            txy = np.asarray(ep[:2], float) + frac * (np.asarray(hp[:2], float) - np.asarray(ep[:2], float))
            print(f"[datagen.cab] over_handle(frac={frac:.2f}): -> xy={np.round(txy, 3)} "
                  f"(handle xy={hp[:2].round(3)}, eef_z held {ep[2]:.3f})", flush=True)
            return np.array([txy[0], txy[1], ep[2]]), eq
        raise ValueError(f"cabinet resolve_compute: unknown tag {tag!r}")

    def _held_bottom(self, ctx, name) -> float:
        return float(_np(ctx.env.scene.object_registry("name", name).aabb[0])[2])

    def _held_top(self, ctx, name) -> float:
        return float(_np(ctx.env.scene.object_registry("name", name).aabb[1])[2])

    def _grasp_pose(self, ctx, P, x):
        """World eef (pos, quat) for grasping ``x['obj']`` at its LIVE pose, offset back by
        ``x['standoff']`` along the approach axis (eef +Z)."""
        if x["obj"] == "handle":
            g = self._handle_grasp()                       # the chosen REACHABLE handle grasp
            T = self._handle_grasp_T(P["cab"], ctx,
                                     _pose_to_mat(g["position"], g["orientation_xyzw"]),
                                     joint=x.get("joint"))  # joint override: close-push -> 0 (closed)
        else:
            obj = ctx.target if x["obj"] == "target" else P["obstacle"]
            op, oq = obj.get_position_orientation()
            g = self._grasp_record(x["key"], x["grasp_id"])
            T = _pose_to_mat(_np(op), _np(oq)) @ _pose_to_mat(g["position"], g["orientation_xyzw"])
        pos, R = T[:3, 3], T[:3, :3]
        if x.get("grasp_roll"):                                # singularity-aware selection picked the
            R = R @ Rot.from_rotvec([0.0, 0.0, np.pi]).as_matrix()   # 180°-about-approach roll variant
        standoff = float(x.get("standoff", 0.0))
        pos = pos - standoff * R[:, 2]                         # eef +Z = approach (roll leaves it fixed)
        return pos, Rot.from_matrix(R).as_quat()

    # ---- small helpers ---------------------------------------------------------------
    def _handle_grasp(self) -> dict:
        """The chosen drawer-handle grasp record — the best REACHABLE id from select_grasps, else the
        first annotation (a blind gs[0] is often unreachable: the far end of the bar is out of range)."""
        gs = self._local_grasps(f"{self._geom['category']}/{self._geom['model']}")
        if not gs:
            raise ValueError("cabinet handle not annotated yet (annotate bottom_cabinet/bamfsz)")
        gids = (self._p or {}).get("handle_gids") or []
        gid = gids[0] if gids else gs[0]["id"]
        return next(g for g in gs if int(g["id"]) == int(gid))

    def _handle_grasp_T(self, cab, ctx, glocal, *, joint=None):
        """World 4x4 eef pose of a handle grasp (object-local 4x4 ``glocal``, no standoff). The handle
        rides the drawer link, so shift along slide by (joint - j_extract); ``joint`` defaults to the
        live drawer joint (the close-push overrides it to 0 = closed)."""
        cp, cq = cab.get_position_orientation()
        T = _pose_to_mat(_np(cp), _np(cq)) @ glocal
        if joint is None:
            joint = float(cab.get_joint_positions()[self._drawer_jidx(cab, ctx)])
        d = self._p["layout"].d
        T[:3, 3] += np.array([d[0], d[1], 0.0]) * (float(joint) - self._geom["j_extract"])
        return T

    def _grasp_record(self, key, gid):
        for g in self._local_grasps(key):
            if int(g["id"]) == int(gid):
                return g
        raise ValueError(f"{key} grasp id {gid} not annotated")

    # ---- object-geometry-aware orientation (elongated-object handling) ----------------
    def _object_footprint(self, key):
        """The object's UPRIGHT horizontal footprint from its annotated bbox: returns
        ``(long_axis_local_unit, long_len, short_len)``. ``long_axis_local`` is the object-frame
        axis that lies horizontal (small world-z component when upright) AND is longest — the axis
        we orient to clear the corridor / fit the cavity. Pure geometry, generalises to any object."""
        o = self._db["objects"][key]
        bbox = np.asarray(o["bbox_size"], float)
        R_up = Rot.from_quat(np.asarray(o["upright_orientation_xyzw"], float)).as_matrix()
        horiz = [(i, float(bbox[i])) for i in range(3) if abs(R_up[2, i]) < 0.5]   # ~horizontal axes
        horiz.sort(key=lambda t: t[1], reverse=True)
        long_i = horiz[0][0] if horiz else int(np.argmax(bbox))
        long_local = np.zeros(3, float)
        long_local[long_i] = 1.0
        long_len = horiz[0][1] if horiz else float(bbox.max())
        short_len = horiz[1][1] if len(horiz) > 1 else float(bbox.min())
        return long_local, long_len, short_len

    def _relocate_halves(self, key):
        """``(p_half, d_half)`` for relocate edge-positioning. An elongated blocker is parked long-axis
        ∥ the table edge (``edge_yaw``), so it faces the drawer corridor with its SHORT half and spans
        the edge with its LONG half; a compact blocker keeps its square footprint half on both axes."""
        half = self._obj_width(key) / 2.0
        if not self._is_elongated(key):
            return half, half
        _, long_len, short_len = self._object_footprint(key)
        return short_len / 2.0, long_len / 2.0

    def _is_elongated(self, key, thresh: float = 1.5) -> bool:
        """True if the object's upright horizontal footprint is elongated enough that its yaw matters
        (long ≥ ``thresh`` × short). Compact / near-symmetric footprints (a can, a shaker) return
        False — their relocate/place orientation is irrelevant, so the geometry-aware yaw steps are
        skipped (force-rotating a symmetric, often tippy object only risks a tip)."""
        _, long_len, short_len = self._object_footprint(key)
        return long_len >= thresh * max(short_len, 1e-6)

    def _eo_rolled(self, R_eo, rolled):
        """The eef-in-object rotation, post-rolled 180° about the eef approach axis (+Z) when the
        singularity-aware selection chose the rolled grasp variant. Every reorient that reconstructs the
        held object's pose from the annotated grasp must use this, else it twists the wrist back toward
        the singular config the roll avoided (the grip is 180° about approach from the annotation)."""
        if rolled:
            R_eo = R_eo @ Rot.from_rotvec([0.0, 0.0, np.pi]).as_matrix()
        return R_eo

    def _yaw_align_quat(self, key, gid, target_dir_xy, cur_eef_quat, *, rolled: bool = False):
        """eef quat that holds object ``key`` (grasp ``gid``) UPRIGHT with its long horizontal axis
        aligned to ``target_dir_xy`` (world). The roll is about world Z (an upright object stays
        upright). The long axis is a LINE, so the two alignments 180° apart are equivalent — pick the
        one nearest the current eef orientation (smallest wrist roll, no needless 180° flip)."""
        o = self._db["objects"][key]
        R_up = Rot.from_quat(np.asarray(o["upright_orientation_xyzw"], float)).as_matrix()
        long_local, _, _ = self._object_footprint(key)
        long0 = (R_up @ long_local)[:2]                         # long axis dir at the object's upright pose
        theta0 = float(np.arctan2(long0[1], long0[0]))
        theta_t = float(np.arctan2(float(target_dir_xy[1]), float(target_dir_xy[0])))
        R_eo = Rot.from_quat(np.asarray(self._grasp_record(key, gid)["orientation_xyzw"], float)).as_matrix()
        R_eo = self._eo_rolled(R_eo, rolled)                    # stay consistent with the rolled grip
        cur = Rot.from_quat(np.asarray(cur_eef_quat, float))
        best = None
        for k in (0, 1):                                        # long axis line → two equivalent yaws
            R_obj = Rot.from_euler("z", theta_t - theta0 + k * np.pi).as_matrix() @ R_up
            q = Rot.from_matrix(R_obj @ R_eo)
            d = float((q * cur.inv()).magnitude())
            if best is None or d < best[0]:
                best = (d, q.as_quat())
        return best[1]

    def _best_grasp_id(self, key) -> int:
        gs = self._local_grasps(key)
        if not gs:
            raise ValueError(f"{key} not annotated yet")
        gids = (self._p or {}).get("obstacle_gids") or []
        if not gids:
            return int(gs[0]["id"])
        # prefer REAL top-down (a straight vertical relocate lift stalls a side grasp's wrist); restrict
        # to the cached top-down subset first, fall back to all reachable if none is top-down.
        td = (self._p or {}).get("topdown_by_key_gid", {})
        td_gids = [g for g in gids if td.get((key, int(g)), False)]
        pool = td_gids or gids
        # within the pool, prefer a CENTRAL grasp: an end-grasp on an elongated object (e.g. a ladle by
        # its handle tip) slips, so the relocate pick comes up empty and the unmoved blocker gets knocked
        # when the drawer opens. Weight by aspect so compact objects (≈1) keep the cuRobo-score order.
        long_local, long_len, short_len = self._object_footprint(key)
        aspect = long_len / max(short_len, 1e-6)
        center = float(np.mean([float(np.dot(g["position"], long_local)) for g in gs]))

        def _bal(gid):
            g = self._grasp_record(key, gid)
            return abs(float(np.dot(g["position"], long_local)) - center) * aspect

        return int(min(pool, key=_bal))

    def _obj_width(self, key) -> float:
        bb = self._db["objects"].get(key, {}).get("bbox_size", [0.05, 0.05, 0.05])
        return max(bb[0], bb[1])

    @staticmethod
    def _drawer_jidx(cab, ctx) -> int:
        return list(cab.joints.keys()).index(ctx.diagnostics["cabinet_info"]["joint"])

    def debug_state(self, ctx: TaskContext) -> str:
        P = self._prepare(ctx)
        j = float(P["cab"].get_joint_positions()[self._drawer_jidx(P["cab"], ctx)])
        txy = _np(ctx.target.get_position_orientation()[0])[:2]
        return f"drawer_joint={j:.4f} target_xy={txy.round(3)}"

    def on_segment(self, seg: MotionSegment, ctx: TaskContext) -> None:
        """STIFFEN the drawer joint while the arm reaches over the OPEN drawer to place the target (a
        stray brush would otherwise shove the very-soft drawer shut), then SOFTEN it again for the
        deliberate close push. Keyed by segment name."""
        j = self._prepare(ctx)["cab"].joints[ctx.diagnostics["cabinet_info"]["joint"]]
        if seg.name in ("place_lift", "place_across", "place_upright", "place_rail_clear",
                        "place_lower", "place_lift_out", "place_toward_handle"):
            if float(j.friction) < DRAWER_HOLD_FRICTION:
                j.damping, j.friction = DRAWER_HOLD_DAMPING, DRAWER_HOLD_FRICTION
                print(f"[datagen.cab] drawer HELD (friction={j.friction}) for place", flush=True)
        elif seg.name.startswith("close_pre"):
            if float(j.friction) > DRAWER_FRICTION:
                j.damping, j.friction = DRAWER_DAMPING, DRAWER_FRICTION
                print(f"[datagen.cab] drawer softened (friction={j.friction}) for close", flush=True)
