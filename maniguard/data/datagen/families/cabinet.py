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

The 5-phase flow::

    1  close drawer (initial)   FREE → handle pre-grasp / SERVO push shut / replay back
    2  relocate blockers        FREE pick → SERVO descend → SERVO lift → FREE transit → SERVO place
    3  open drawer              FREE → handle pre-grasp / SERVO grasp / SERVO pull to MAX / open
    4  place target in drawer    FREE → SERVO deep grasp / SERVO inverted-U (↑lift →over upright ↓lower)
    5  close drawer (final)     FREE → handle pre-grasp / SERVO push shut / replay back

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
PLACE_Z_MARGIN = 0.01     # drop height above the table / drawer floor
RETREAT_DZ = 0.12         # straight-up retreat after releasing a relocated blocker
HANDLE_BACK_DIST = 0.10   # after opening, retreat the gripper this far along +open so fingers clear the handle
CLOSE_FRACTION = 0.12     # close to this fraction of the spawn opening (≈88% shut; << Open threshold)
LIFT_CLEAR = 0.05         # relocate: lift the held blocker this far above the table top before transit
RIM_CLEARANCE = 0.06      # place: lift the target's bottom this far above the drawer rim before going over
# Per-demo diversity bands (sampled from the variant seed in derive_segments; small so the long-horizon
# task still reliably completes). Dim 1: how far above the rim the target's bottom is carried (the lift
# height). Dim 2: how far the relocated TARGET slides along +opening on the near edge (where it lands).
RIM_CLEAR_BAND = (0.06, 0.12)
TARGET_D_SHIFT_BAND = (0.18, 0.30)
LOWER_IN_MAX = 0.35       # place: cap the final straight descent into the OPEN drawer. This chest's drawer
#                           is DEEP (rim 0.734, interior floor ~0.47), so the 0.21 m target needs a ~0.26 m
#                           descent to sit fully below the rim (else it jams the close) — NOT the ~10 cm
#                           first assumed. The descent stops physically on the interior floor; no free-fall.
REACH_TOL = 0.05          # eef-reached-target tolerance for the place lift/over stuck check
LIFT_OUT_ABOVE_RIM = 0.20   # after releasing the target, lift the empty gripper straight up to this height
#                             above the rim (≈ the carry height the target was moved over at) BEFORE the
#                             final close — clears the rim + the long finger-rail so the next cuRobo plan to
#                             the handle routes a clean arc instead of scraping a diagonal across the door
# Drawer prismatic-joint resistance. Default (damping 5.0 / friction 0.30) stalls a position-controlled
# push; soften it for the whole demo so open + close both move. STIFFEN it only while the arm reaches
# OVER the open drawer to place (Phase 4), so a stray brush can't shove the soft drawer shut.
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

    # ---- per-task preparation (live env reads; cabinet/table/robot are fixed) ---------
    def _prepare(self, ctx: TaskContext) -> dict:
        if self._p is not None:
            return self._p
        env, diag = ctx.env, ctx.diagnostics
        ci = diag["cabinet_info"]
        cab = env.scene.object_registry("name", ci["name"])
        cp, cq = cab.get_position_orientation()
        rp, _ = ctx.robot.get_position_orientation()
        layout = CG.build_layout(diag, self._geom, cab_pos=_np(cp), cab_quat=_np(cq),
                                 robot_xy=_np(rp)[:2])

        obstacle = env.scene.object_registry("name", diag["obstacle_info"]["name"])
        # Phase-2 relocation list, in a role-FIXED order: the OBSTACLE is cleared FIRST (it parks on
        # the cabinet's far-from-base side), THEN the target (it parks on the near-base side, where
        # Phase 4 re-picks it). Only objects actually in the drawer's opening sweep are moved — the
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
        # OmniGibson tags the drawer's interior with a "fillable" meta-link — the ground-truth volume
        # centre objects belong in. We drop the target onto its LIVE xy (the drawer is open by then,
        # so it tracks the slid-out interior centre, clear of the handle + the cabinet overhang) — far
        # more reliable than estimating the cavity centre from handle-inclusive AABB projections.
        fillable_link = f"meta__{ci['link']}_fillable_0_0_link"
        if fillable_link not in cab.links:
            fillable_link = None
        self._p = {
            "layout": layout, "cab": cab, "cab_name": ci["name"], "obstacle": obstacle,
            "obstacle_key": self._obj_key(env, diag["obstacle_info"]["name"]),
            "blockers": blockers, "drawer_top_z": drawer_top_z, "fillable_link": fillable_link,
        }
        print(f"[datagen.cab] prepared: blockers={[r for r, _ in blockers]} rim={drawer_top_z:.3f} "
              f"fillable={fillable_link}", flush=True)
        return self._p

    @staticmethod
    def _obj_key(env, name):
        o = env.scene.object_registry("name", name)
        return f"{o.category}/{o.model}"

    def _local_grasps(self, key: str) -> list[dict]:
        return self._db.get("objects", {}).get(key, {}).get("grasps", [])

    # ---- grasp candidates = the TARGET's annotated grasps (driver scores + varies; these are
    # ---- the grasps used to RELOCATE the target in Phase 2; the Phase-4 PLACE grasp is chosen
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
        obstacle, and the Phase-4 PLACE grasp on the target."""
        P = self._prepare(ctx)
        hkey = f"{self._geom['category']}/{self._geom['model']}"
        # A handle grasp must be COLLISION-FREE + reachable at the 3 binding poses, scored with the
        # in-path blocker PRESENT (only the cabinet is dropped) so a handle point whose contact pose
        # rams a wedged blocker is rejected here.
        #   pre     = close PRE-pose  (live joint, 0.10 standoff, +Y-most reach)
        #   contact = handle CONTACT  (live joint, no standoff)  <- rejects blocker-colliding points
        #   close   = push ENDPOINT   (target joint, no standoff, -Y-most reach)
        pre = self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=P["cab"],
                              on_handle=True, joint=None, standoff_m=0.10)
        contact = set(self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=P["cab"],
                                      on_handle=True, joint=None, standoff_m=0.0))
        close = set(self._score_aux(world, robot, ctx, obj=P["cab"], key=hkey, ignore=P["cab"],
                                    on_handle=True, joint=self.close_target_joint(), standoff_m=0.0))
        clear = [g for g in pre if g in contact and g in close]
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
                                                 on_handle=False)
            print(f"[datagen.cab] obstacle reachable gids (best-first): {P['obstacle_gids']}", flush=True)
        self._select_place_grasp(ctx, world, robot, P)

    def _select_place_grasp(self, ctx, world, robot, P) -> None:
        """Choose the Phase-4 PLACE grasp on the target. The target is a solid base + thin post: a
        top-down grasp on the post slips, so prefer the DEEPEST reachable TOP-DOWN grasp (lowest
        object-local z = closest to the solid base = a firmer hold + a lower eef to keep the over-rim
        lift within reach). Score reachability at the target's PREDICTED Phase-4 location (its
        relocated base-side spot if it is in the drawer path, else spawn) — scoring at spawn rejects
        the deep grasps that only become reachable once the target sits in the arm's sweet zone.
        Caches ``P['place_gid']``; the runtime pre-grasp cuRobo plan is the final reach check."""
        key = ctx.target_key
        grasps = self._local_grasps(key)
        if not grasps:
            return
        obj_pose = self._predicted_target_pose(ctx, P)
        reachable = set(self._score_aux(world, robot, ctx, obj=ctx.target, key=key,
                                        ignore=ctx.target, on_handle=False, obj_pose=obj_pose))
        if not reachable:                              # fall back to spawn if the predicted pose scores empty
            reachable = set(self._score_aux(world, robot, ctx, obj=ctx.target, key=key,
                                            ignore=ctx.target, on_handle=False))
        if not reachable:
            print("[datagen.cab] place grasp: none reachable -> fallback to target_grasp at runtime", flush=True)
            return

        def vertical(g):                               # +1 = straight-down (top-down), 0 = side
            appr = Rot.from_quat(g["orientation_xyzw"]).as_matrix()[:, 2]   # eef +Z in OBJECT frame
            return float(-appr[2])                     # object upright ⇒ object-z ≈ world-z

        cand = [g for g in grasps if int(g["id"]) in reachable]
        top_down = [g for g in cand if vertical(g) > 0.5] or cand        # tilt-safe pinch; fall back if none
        deepest = min(top_down, key=lambda g: float(g["position"][2]))   # lowest object-local z = deepest hold
        P["place_gid"] = int(deepest["id"])
        print(f"[datagen.cab] place grasp: id={deepest['id']} eef_z(local)={float(deepest['position'][2]):.3f} "
              f"vert={vertical(deepest):.2f} pool={'top_down' if top_down is not cand else 'all'} "
              f"reachable={sorted(reachable)}", flush=True)

    def _predicted_target_pose(self, ctx, P):
        """Where the target will sit when Phase 4 picks it: its Phase-2 relocated (base-side) xy if it
        is in the drawer's path, else its spawn pose. Orientation + z kept from spawn (the relocate
        sets it back upright on the table). Mirrors the Phase-2 ``blocker_placement`` call exactly
        (``open_distance`` is deterministic, target placement is rng-independent) so the prediction
        matches the actual relocation."""
        tp, tq = (_np(v) for v in ctx.target.get_position_orientation())
        if not any(role == "target" for role, _ in P["blockers"]):
            return tp, tq                              # not in path -> picked at spawn
        L = P["layout"]
        open_dist = CG.open_distance(self._obj_width(ctx.target_key), L.remaining_travel,
                                     np.random.default_rng(0))
        place_xy = CG.blocker_placement(L, tp[:2], self._obj_width(ctx.target_key) / 2, "target",
                                        open_dist)
        if place_xy is None:
            return tp, tq
        return np.array([place_xy[0], place_xy[1], tp[2]]), tq

    def _score_aux(self, world, robot, ctx, *, obj, key, ignore, on_handle: bool,
                   joint=None, standoff_m: float = 0.10, obj_pose=None) -> list[int]:
        """Build world-frame GraspCands for ``key``'s annotation grasps (handle ones ride the drawer
        link at ``joint``; other objects sit at ``obj_pose`` if given, else their live pose),
        cuRobo-score them at ``standoff_m``, and return the reachable ids best-first."""
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
            return []
        score_grasps(world, robot, ignore, cands, standoff_m=standoff_m)   # sets reachable, sorts best-first
        return [c.id for c in cands if c.reachable]

    # ---- the full 5-phase sequence ---------------------------------------------------
    def derive_segments(self, ctx: TaskContext, target_grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        P = self._prepare(ctx)
        L = P["layout"]
        rng = np.random.default_rng(params.seed)
        open_dist = CG.open_distance(self._obj_width(ctx.target_key), L.remaining_travel, rng)
        # per-demo diversity draws (deterministic per variant seed)
        d_shift = float(rng.uniform(*TARGET_D_SHIFT_BAND))     # dim 2: target's landing point along the edge
        rim_clear = float(rng.uniform(*RIM_CLEAR_BAND))        # dim 1: carry height above the rim
        print(f"[datagen.cab] diversity: d_shift={d_shift:.3f} rim_clear={rim_clear:.3f}", flush=True)

        segs: list[MotionSegment] = []
        # Phase 1 — close the drawer first. The bench spawns it 0.2-open; a blocker wedged at the
        # drawer front leaves no headroom for the top-down pick. Retracting it frees that space.
        segs += self._close_drawer(ctx, tag="initial")

        # Phase 2 — move each path-blocking object aside (clean pick-and-place).
        for role, obj in P["blockers"]:
            key = ctx.target_key if role == "target" else P["obstacle_key"]
            gid = target_grasp.id if role == "target" else self._best_grasp_id(key)
            place_xy = CG.blocker_placement(L, _np(obj.get_position_orientation()[0])[:2],
                                            self._obj_width(key) / 2, role, open_dist, d_shift=d_shift)
            if place_xy is None:                       # no room → a "third-type" task to report
                return []
            print(f"[datagen.cab] relocate[{role}] "
                  f"obj_xy={_np(obj.get_position_orientation()[0])[:2].round(3)} "
                  f"-> place_xy={np.asarray(place_xy).round(3)}", flush=True)
            segs += self._relocate_blocker(ctx, key, gid, role, place_xy, params)

        # Phase 3 — open the drawer to its widest.
        segs += self._open_drawer(ctx, dist=open_dist)

        # Phase 4 — pick the target (now at its moved spot) + place it INTO the open drawer.
        segs += self._place_in_drawer(ctx, target_grasp, params, rim_clear)

        # Phase 5 — close the drawer.
        segs += self._close_drawer(ctx, tag="final")
        return segs

    # ---- segment builders ------------------------------------------------------------
    def _relocate_blocker(self, ctx, key, gid, role, place_xy, params) -> list[MotionSegment]:
        """Phase 2: a clean pick-and-place (the boxy clutter pattern). FREE cuRobo to the grasp
        standoff, SERVO straight down + close, SERVO small lift off the table, FREE cuRobo transit to
        above the place spot (collision-aware — avoids the cabinet + the other blocker), SERVO
        straight down, open, lift off. The straight descents/lifts are SERVO (pure IK) — LINEAR's
        partial-pose query is broken on this fork and drifts the carried object off the place spot.
        The blockers relocate toward the robot / along the table front, so the FREE transit clears
        the cabinet with a short collision-aware path."""
        obj_name = ctx.target_name if role == "target" else ctx.diagnostics["obstacle_info"]["name"]
        cab = self._prepare(ctx)["cab_name"]
        e = {"obj": role, "key": key, "grasp_id": gid}
        held = {"held_name": obj_name}
        q0 = np.array([0.0, 0.0, 0.0, 1.0])            # placeholder; resolve_compute fills the real target
        xy = {"xy": [float(place_xy[0]), float(place_xy[1])]}
        return [
            MotionSegment("pick_pre", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**e, "standoff": params.standoff_m},
                          ignore_objects=(obj_name, cab)),
            MotionSegment("pick_descend", q0[:3], q0, mode=Mode.SERVO, grip=Grip.CLOSE, grip_steps=8,
                          compute="grasp", extra={**e, "standoff": 0.0},
                          ignore_objects=(obj_name,), ignore_clutter=True),
            MotionSegment("pick_lift", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lift_above_support", extra={**held, "clear": LIFT_CLEAR},
                          ignore_objects=(cab,)),
            MotionSegment("move_transit", q0[:3], q0, mode=Mode.FREE, attach=True, grip=Grip.HOLD,
                          compute="move_to", extra={**xy, **held}),     # avoid ALL (incl. cabinet)
            MotionSegment("move_place", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lower_to_support", extra={**held},   # descend at the current xy until the
                          ignore_objects=(cab,)),                       # held bottom rests ON the table (no jam)
            MotionSegment("move_release", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=6,
                          compute="hold", ignore_objects=(cab,)),       # open in place (gripper wraps the object)
            MotionSegment("move_retreat", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="up", extra={"dz": RETREAT_DZ}, ignore_objects=(cab,)),
        ]

    def _open_drawer(self, ctx, *, dist: float) -> list[MotionSegment]:
        """Phase 3: cuRobo-reach the handle pre-pose, CLOSE on the handle, a straight pure-IK SERVO
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
            MotionSegment("handle_back_open", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="extract", extra={"dist": HANDLE_BACK_DIST}, ignore_objects=cab),
        ]

    def _place_in_drawer(self, ctx, target_grasp, params, rim_clear: float = RIM_CLEARANCE) -> list[MotionSegment]:
        """Phase 4: pick the target with its DEEP grasp, then an inverted-"门"-frame placement so it
        lands upright in the open drawer without a tumbling free-fall. All-SERVO straight (pure IK,
        nearest-seed → the held object stays upright, only the eef translates):
          lift   ↑ straight UP to the top-left corner (target bottom above the rim, hard-verified)
          over   → straight HORIZONTAL to the top-right corner (directly over the cavity centre)
          upright  reorient the eef in place so the held object is square to vertical
          lower  ↓ straight DOWN ~10 cm to set the bottom near the drawer floor (no drop)
          release  open at the low set-down → the target settles upright inside."""
        P = self._prepare(ctx)
        cab, tname = P["cab_name"], ctx.target_name
        gid = P.get("place_gid") or target_grasp.id    # deep top-down grasp from select_grasps
        e = {"obj": "target", "key": ctx.target_key, "grasp_id": gid}
        held = {"held_name": tname}
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        rim = float(P["drawer_top_z"])                 # open-drawer wall top — the target bottom must clear THIS
        print(f"[datagen.cab] place_in_drawer: target grasp id={gid} rim={rim:.3f}", flush=True)
        return [
            # FREE to the deep-grasp standoff (target at its moved loc, live pose) → straight down + close.
            MotionSegment("place_pre_grasp", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**e, "standoff": params.standoff_m},
                          ignore_objects=(tname, cab)),
            MotionSegment("place_descend", q0[:3], q0, mode=Mode.SERVO, grip=Grip.CLOSE, grip_steps=8,
                          compute="grasp", extra={**e, "standoff": 0.0},
                          ignore_objects=(tname,), ignore_clutter=True),
            # ↑ UP to the top-left corner: lift the target bottom above the rim + clearance, HARD-VERIFY
            # it cleared before ANY lateral move (a lift that stays below the rim catches it + rams the drawer).
            MotionSegment("place_lift", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lift_over_rim", extra={**held, "clearance": rim_clear},
                          reach_tol_m=REACH_TOL, verify_held_above_z=rim, ignore_objects=(cab,)),
            # → HORIZONTAL to the top-right corner: eef vertical axis directly over the LIVE exposed-cavity
            # centre (computed from the actual drawer joint, not the commanded open), height held; no
            # diagonal carry-out-front; verify_held_above_z guards the rim throughout.
            MotionSegment("place_over", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="over_cavity", extra={**held},
                          reach_tol_m=REACH_TOL, verify_held_above_z=rim, ignore_objects=(cab,)),
            # reorient the eef in place so the held object is square to vertical before lowering.
            MotionSegment("place_upright", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="uprightify", extra={**e, **held}, ignore_objects=(cab,)),
            # roll the eef so the gripper finger-rail is ⊥ the opening direction (∥ the upper drawer's
            # handle) → the long flat rail clears the upper drawer's handle during the descent. Object
            # stays upright (roll is about the vertical approach axis).
            MotionSegment("place_rail_clear", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="rail_clear", extra={**held}, ignore_objects=(cab,)),
            # ↓ DOWN ~10 cm to set the bottom near the drawer floor, upright the WHOLE way (no free-fall —
            # a 0.21 m object dropped 0.3-0.5 m tumbles on impact).
            MotionSegment("place_lower", q0[:3], q0, mode=Mode.SERVO, attach=True, grip=Grip.HOLD,
                          compute="lower_to_floor", extra={**held, "max_dz": LOWER_IN_MAX},
                          ignore_objects=(cab,)),
            MotionSegment("place_release", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN, grip_steps=10,
                          compute="hold", ignore_objects=(cab,)),
            # lift the empty gripper STRAIGHT UP (pure IK, xy held) out of the open cavity to the carry
            # height above the rim. (Planning to the handle straight from inside the deep cavity made
            # cuRobo scrape a diagonal across the cabinet door.)
            MotionSegment("place_lift_out", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="lift_out", extra={"above_rim": LIFT_OUT_ABOVE_RIM}, ignore_objects=(cab,)),
            # then translate STRAIGHT HORIZONTAL (z held at the carry height) to directly above the close
            # handle pre-grasp XY, so the following cuRobo plan only has to descend straight DOWN onto the
            # handle from OUTSIDE the cavity — it never has to route around / scrape the cabinet door.
            MotionSegment("place_over_handle", q0[:3], q0, mode=Mode.SERVO, grip=Grip.OPEN,
                          compute="over_handle", extra={"standoff": STANDOFF}, ignore_objects=(cab,)),
        ]

    def close_target_joint(self) -> float:
        """Target drawer joint for a 'closed' demo: a small fraction of the spawn opening
        (``j_extract``), comfortably below OmniGibson's Open threshold (5% of stroke) so the
        ``closed`` predicate holds, while leaving a sliver so we never ram the drawer fully shut."""
        return float(self._geom["j_extract"]) * CLOSE_FRACTION

    def _close_drawer(self, ctx, *, tag: str) -> list[MotionSegment]:
        """Phase 1 & 5: close the drawer by an honest push (as in teleop). cuRobo-reach a
        collision-aware handle pre-pose closest to the robot base (gripper OPEN), a straight Cartesian
        IK SERVO push drives the OPEN gripper straight in the closing direction (a finger catches the
        drawer front → physics carries it shut to ``close_target_joint``). The pre-pose tracks the
        handle's LIVE position (the final close runs after the drawer was pulled wide open — a fixed
        spawn-joint pre-pose would miss the open handle and under-push).

        The INITIAL close then replays the push lane in REVERSE to retreat straight back out (freeing
        the arm to go pick the blocker). The FINAL close ENDS at the push: closing the drawer with the
        target inside IS the task's success state (``inside & closed``), so the demo terminates on the
        success moment — no trailing retreat recorded past it (the engine's end-of-run success check
        then confirms it)."""
        cab = (self._prepare(ctx)["cab_name"],)         # ignore only the cabinet (we push it)
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        h = {"obj": "handle"}
        tj = self.close_target_joint()                  # ~88% closed
        segs = [
            MotionSegment(f"close_pre_{tag}", q0[:3], q0, mode=Mode.FREE, grip=Grip.OPEN, grip_steps=6,
                          compute="grasp", extra={**h, "standoff": STANDOFF}, ignore_objects=cab),
            MotionSegment(f"close_push_{tag}", q0[:3], q0, mode=Mode.SERVO, grip=Grip.HOLD, carry_closed=False,
                          compute="grasp", extra={**h, "standoff": 0.0, "joint": tj}, ignore_objects=cab),
        ]
        if tag != "final":                              # final close = success state → stop at the push
            segs.append(MotionSegment(f"close_retreat_{tag}", q0[:3], q0, replay_reverse=True,
                                      grip=Grip.OPEN, grip_steps=6, carry_closed=False, ignore_objects=cab))
        return segs

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
        if tag == "over_cavity":                          # the OPEN drawer's INTERIOR centre = its LIVE
            fl = P.get("fillable_link")                   # fillable meta-link xy (OmniGibson ground truth:
            if fl is not None:                            # tracks the slid-out interior, clear of the handle
                fp = _np(P["cab"].links[fl].get_position_orientation()[0])   # + the cabinet overhang)
                print(f"[datagen.cab] over_cavity (fillable): xy={fp[:2].round(3)} "
                      f"(eef_z held {ep[2]:.3f})", flush=True)
                return np.array([fp[0], fp[1], ep[2]]), eq
            L = P["layout"]                               # fallback: estimate from the live joint (handle-biased)
            j = float(P["cab"].get_joint_positions()[self._drawer_jidx(P["cab"], ctx)])
            dc = (L.d_front - L.j_current) + 0.5 * j
            xy = L.to_world(dc, L.p_center)
            print(f"[datagen.cab] over_cavity (geom est): j={j:.3f} dc={dc:.3f} -> xy={np.round(xy, 3)}", flush=True)
            return np.array([xy[0], xy[1], ep[2]]), eq
        if tag == "extract":                              # slide the eef +slide (away from the cabinet)
            d3 = np.array([P["layout"].d[0], P["layout"].d[1], 0.0])   # at the current height
            return ep + d3 * float(x.get("dist", 0.10)), eq
        if tag == "drawer":
            d3 = np.array([P["layout"].d[0], P["layout"].d[1], 0.0])
            if x["to"] == "open":
                return ep + d3 * float(x["dist"]), eq
            j = float(P["cab"].get_joint_positions()[self._drawer_jidx(P["cab"], ctx)])
            return ep - d3 * j, eq                       # push the handle back to joint 0
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
            return ep, Rot.from_matrix(R_obj @ R_eo).as_quat()
        if tag == "rail_clear":                           # roll the eef about the vertical (approach) axis so
            d = P["layout"].d                             # the gripper finger-rail (eef local Y = the long flat
            R = Rot.from_quat(eq).as_matrix()             # slide housing) is ⊥ the opening direction (∥ the
            railY = R[:, 1]                               # upper drawer's handle) → it clears the handle on the
            yaw = float(np.arctan2(d[1], d[0]) + np.pi / 2 - np.arctan2(railY[1], railY[0]))   # descent. Roll
            yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2   # about world Z keeps the object upright; rail is a
            print(f"[datagen.cab] rail_clear: roll {np.degrees(yaw):+.1f}° -> finger-rail ⊥ opening", flush=True)
            return ep, Rot.from_matrix(Rot.from_euler("z", yaw).as_matrix() @ R).as_quat()  # line → |roll|≤90°
        if tag == "lower_to_floor":                       # descend straight so the held bottom nears the
            bottom = self._held_bottom(ctx, x["held_name"])   # drawer floor, capped (~10 cm, no free-fall)
            floor = P["layout"].drawer_floor_z + PLACE_Z_MARGIN
            dz = min(max(0.0, bottom - floor), float(x.get("max_dz", LOWER_IN_MAX)))
            print(f"[datagen.cab] lower_to_floor: bottom={bottom:.3f} floor={floor:.3f} -> dz=-{dz:.3f}", flush=True)
            return np.array([ep[0], ep[1], ep[2] - dz]), eq
        if tag == "lower_to_support":                     # relocate set-down: lower the held bottom onto the
            bottom = self._held_bottom(ctx, x["held_name"])   # TABLE top (live measured), capped — NOT a fixed
            st = float(_np(ctx.support.aabb[1])[2]) if ctx.support is not None else 0.0   # eef-z (which drives
            dz = min(max(0.0, bottom - (st + PLACE_Z_MARGIN)), float(x.get("max_dz", 0.30)))   # the object INTO
            return np.array([ep[0], ep[1], ep[2] - dz]), eq   # the table → jam + drift). XY held = the edge spot.
        if tag == "lift_out":                             # straight UP (xy held) to the carry height above
            z = P["drawer_top_z"] + float(x.get("above_rim", LIFT_OUT_ABOVE_RIM))   # the rim — clear of the
            print(f"[datagen.cab] lift_out: eef_z {ep[2]:.3f} -> {max(ep[2], z):.3f} (rim+{x.get('above_rim'):.2f})", flush=True)
            return np.array([ep[0], ep[1], max(ep[2], z)]), eq   # cavity before cuRobo routes to the handle
        if tag == "over_handle":                          # translate HORIZONTAL (z + orientation held) to
            hp, _ = self._grasp_pose(ctx, P, {"obj": "handle",   # directly above the close handle pre-grasp
                                              "standoff": float(x.get("standoff", STANDOFF))})   # XY (live open
            print(f"[datagen.cab] over_handle: -> xy={hp[:2].round(3)} (eef_z held {ep[2]:.3f})", flush=True)
            return np.array([hp[0], hp[1], ep[2]]), eq    # joint) so cuRobo then only descends straight down
        raise ValueError(f"cabinet resolve_compute: unknown tag {tag!r}")

    def _held_bottom(self, ctx, name) -> float:
        return float(_np(ctx.env.scene.object_registry("name", name).aabb[0])[2])

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
        pos = pos - float(x.get("standoff", 0.0)) * R[:, 2]    # eef +Z = approach
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

    def _best_grasp_id(self, key) -> int:
        gs = self._local_grasps(key)
        if not gs:
            raise ValueError(f"{key} not annotated yet")
        gids = (self._p or {}).get("obstacle_gids") or []
        return int(gids[0]) if gids else int(gs[0]["id"])

    def _obj_width(self, key) -> float:
        bb = self._db["objects"].get(key, {}).get("bbox_size", [0.05, 0.05, 0.05])
        return max(bb[0], bb[1])

    def _obj_half_h(self, key) -> float:
        bb = self._db["objects"].get(key, {}).get("bbox_size", [0.05, 0.05, 0.05])
        return 0.5 * bb[2]

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
        if seg.name in ("place_lift", "place_over", "place_upright", "place_rail_clear",
                        "place_lower", "place_lift_out", "place_over_handle"):
            if float(j.friction) < DRAWER_HOLD_FRICTION:
                j.damping, j.friction = DRAWER_HOLD_DAMPING, DRAWER_HOLD_FRICTION
                print(f"[datagen.cab] drawer HELD (friction={j.friction}) for place", flush=True)
        elif seg.name.startswith("close_pre"):
            if float(j.friction) > DRAWER_FRICTION:
                j.damping, j.friction = DRAWER_DAMPING, DRAWER_FRICTION
                print(f"[datagen.cab] drawer softened (friction={j.friction}) for close", flush=True)
