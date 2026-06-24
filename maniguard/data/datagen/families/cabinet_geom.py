"""Runtime geometry for the cabinet family — pure numpy, no OmniGibson / cuRobo.

Resolves a task's cabinet layout (slide direction, the drawer's opening corridor, the table
bounds, which perpendicular side faces the robot) and derives the placement decisions the
skeleton needs: where to move a path-blocking object, how far to open the drawer, and where
inside the drawer to place the target.

All inputs are explicit (poses / bboxes / bounds derivable from a task's ``diagnostics`` +
``scene_ep1``), so every function here is unit-testable offline with no sim. The skeleton/engine
feed live sim reads (the current drawer joint, object poses) into the same functions at runtime.

Coordinates: work in the world xy plane via an orthonormal basis ``(d, p)`` where ``d`` =
slide/opening direction and ``p`` = perpendicular oriented toward the robot (``+p`` = near side).
A world point at projections ``(dc, pc)`` is ``dc*d + pc*p``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

EDGE_MARGIN = 0.04        # keep a moved object this far inside the table edge
PLACE_Z_MARGIN = 0.01     # drop height above the drawer floor
TARGET_D_SHIFT = 0.22     # slide the relocated TARGET this far along +opening, off the robot base's
#                           straight-ahead line, so the top-down re-grasp isn't folded too tight to solve
#                           (≈0.3 m straight-ahead → ≈0.39 m diagonal; the obstacle is never re-grasped → 0)
OBSTACLE_D_NUDGE = 0.08   # nudge the relocated OBSTACLE this far off the base's perpendicular foot (the
#                           straight-ahead, most-folded pose) toward the cabinet face — a diagonal place
#                           solves more reliably (clamped so it never crosses behind the cabinet face)


def slide_axes(slide_dir, toward_xy=None, origin_xy=None):
    """Unit (d, p): d = opening direction; p = perpendicular, flipped so +p points toward
    ``toward_xy`` (relative to ``origin_xy``) when given — i.e. +p = the near-robot side."""
    d = np.asarray(slide_dir, float)[:2]
    d = d / (np.linalg.norm(d) + 1e-9)
    p = np.array([-d[1], d[0]])
    if toward_xy is not None and origin_xy is not None:
        if (np.asarray(toward_xy, float)[:2] - np.asarray(origin_xy, float)[:2]) @ p < 0:
            p = -p
    return d, p


def _aabb_corners(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def drawer_world_projection(cab_pos, cab_quat, drawer_lo, drawer_hi, d, p):
    """Project the drawer-link AABB (cabinet-root-local) into world (d, p, z). Returns
    ``(d_front, d_back, p_lo, p_hi, z_lo, z_hi)`` — ``d_front`` = the leading face along +d."""
    world = (Rot.from_quat(cab_quat).as_matrix() @ _aabb_corners(drawer_lo, drawer_hi).T).T \
        + np.asarray(cab_pos, float)
    dd, pp = world[:, :2] @ d, world[:, :2] @ p
    return float(dd.max()), float(dd.min()), float(pp.min()), float(pp.max()), \
        float(world[:, 2].min()), float(world[:, 2].max())


@dataclass
class CabinetLayout:
    """A task's resolved cabinet geometry (world frame, projected onto d/p)."""

    d: np.ndarray            # slide/opening unit (xy)
    p: np.ndarray            # perp unit (xy), +p = near-robot side
    cab_xy: np.ndarray       # cabinet root xy
    d_front: float           # drawer leading-face d-coord at the current joint
    d_back: float
    p_lo: float
    p_hi: float
    drawer_floor_z: float
    stroke: float
    j_extract: float
    j_current: float
    table_lo: np.ndarray     # table AABB min xy
    table_hi: np.ndarray
    robot_xy: np.ndarray

    @property
    def p_center(self) -> float:
        return 0.5 * (self.p_lo + self.p_hi)

    @property
    def remaining_travel(self) -> float:
        return max(0.0, self.stroke - self.j_current)

    def to_world(self, dc: float, pc: float) -> np.ndarray:
        return dc * self.d + pc * self.p


def build_layout(diag: dict, cab_geom: dict, *, cab_pos, cab_quat, robot_xy,
                 j_current=None) -> CabinetLayout:
    """Resolve a CabinetLayout from explicit poses (cabinet root + robot xy, live from the env at
    runtime or from a snapshot offline) + the task diagnostics + cached cabinet geom."""
    ci = diag["cabinet_info"]
    cab_pos = np.asarray(cab_pos, float)
    robot_xy = np.asarray(robot_xy, float)[:2]
    d, p = slide_axes(ci["slide_dir"], toward_xy=robot_xy, origin_xy=cab_pos[:2])
    g = cab_geom["drawer_aabb_root_local"]
    d_front, d_back, p_lo, p_hi, z_lo, z_hi = drawer_world_projection(
        cab_pos, cab_quat, g["lo"], g["hi"], d, p)
    j_cur = float(ci.get("open_fraction", 0.2) * ci["stroke_m"]) if j_current is None else float(j_current)
    tb = diag["surface_info"]["bounds_xy"]
    return CabinetLayout(
        d=d, p=p, cab_xy=cab_pos[:2], d_front=d_front, d_back=d_back, p_lo=p_lo, p_hi=p_hi,
        drawer_floor_z=z_lo, stroke=float(ci["stroke_m"]), j_extract=float(cab_geom["j_extract"]),
        j_current=j_cur, table_lo=np.asarray(tb[0], float), table_hi=np.asarray(tb[1], float),
        robot_xy=robot_xy)


def build_layout_from_scene(diag: dict, scene: dict, cab_geom: dict, *, j_current=None) -> CabinetLayout:
    """Offline convenience: extract the cabinet root + robot poses from a scene_ep1 snapshot, then
    ``build_layout``. (Runtime callers read live poses from the env and call ``build_layout``.)"""
    reg = scene["state"]["registry"]["object_registry"]

    def world_pose(name):
        for k, v in reg.items():
            if name in k:
                return np.asarray(v["root_link"]["pos"], float), np.asarray(v["root_link"]["ori"], float)
        return None, None

    cab_pos, cab_quat = world_pose(diag["cabinet_info"]["name"])
    robot_xy = world_pose("agent")[0]
    if robot_xy is None:
        robot_xy = cab_pos - np.array([0.5, 0.0, 0.0])
    return build_layout(diag, cab_geom, cab_pos=cab_pos, cab_quat=cab_quat,
                        robot_xy=robot_xy, j_current=j_current)


# --- decisions (small pure functions over a layout) -------------------------------------
def open_distance(target_width: float, remaining_travel: float, rng,
                  lo: float = 2.5, hi: float = 3.5) -> float:
    """Drawer opening = OPEN AS WIDE AS IT GOES (≈ full remaining travel, less a small margin off the
    hard stop). Blockers are always relocated to the cabinet SIDES (never the far opening end), so
    there is no reason to keep the drawer narrow — the widest opening gives the dropped target the
    largest, most central cavity to fall straight into, and minimises the diagonal eef travel to reach
    the cavity centre (``target_width``/``lo``/``hi``/``rng`` kept for signature compat, unused)."""
    return float(max(0.0, remaining_travel - 0.02))


def corridor(L: CabinetLayout, open_dist: float):
    """(d_lo, d_hi, p_lo, p_hi) the drawer front sweeps when opened by ``open_dist``."""
    return (L.d_front, L.d_front + open_dist, L.p_lo, L.p_hi)


def drawer_interior_center(L: CabinetLayout, open_dist: float, obj_half_h: float = 0.0):
    """World (xyz) to drop the target into the OPEN drawer. Not the cavity's geometric centre
    (that is deep inside the cabinet body — unreachable, and shoving the target there pushes the
    drawer shut); rather the centre of the EXPOSED span that slid out past the cabinet front
    (reachable from above, +open side). The drawer ends up open by ``open_dist`` from closed, and
    its closed leading face = ``d_front - j_current`` (the spawn leading face retracts to closed)."""
    leading_closed = L.d_front - L.j_current        # drawer leading face once (re-)closed ≈ cabinet front
    dc = leading_closed + 0.5 * float(open_dist)    # centre of the exposed open span
    xy = L.to_world(dc, L.p_center)
    z = L.drawer_floor_z + obj_half_h + PLACE_Z_MARGIN
    return np.array([xy[0], xy[1], z])


def blocker_placement(L: CabinetLayout, obj_xy, obj_half: float, role: str,
                      open_dist: float = 0.0, d_shift: float | None = None,
                      avoid_dc: float | None = None, avoid_half: float = 0.0,
                      p_half: float | None = None, d_half: float | None = None):
    """Where to put a path-blocking object so the drawer can open. Returns world xy. role in
    {'target','obstacle'}. ``d_shift`` (target only) = how far to slide along +opening off the base's
    straight-ahead line; ``None`` => the nominal ``TARGET_D_SHIFT`` (callers pass a per-demo sampled
    value for diversity, but the place-grasp pre-selection keeps the nominal so its prediction holds).
    ``avoid_dc``/``avoid_half`` (obstacle only): the target's already-resolved destination d-coord and
    half-width — the obstacle is kept ``obj_half + avoid_half + EDGE_MARGIN`` clear of it so the two
    parked bboxes never overlap.

    BOTH roles park on the SAME ``+p`` (near-robot) table edge — pushed FLUSH against it, all the way
    out of the drawer-opening corridor (the ``[p_lo, p_hi]`` sweep) so the drawer clears them — and
    BOTH stay IN FRONT of the cabinet face (never carried behind it across the opening plane: the far
    ``-p`` edge and the cabinet-back region are out of the mounted arm's reach AND invite a knock
    during the later pick/place). They are STAGGERED along that one edge so they never collide:
      * TARGET   → slid ``TARGET_D_SHIFT`` along +opening off the base's straight-ahead line. The bare
        perpendicular-foot spot sits ~0.3 m straight in front of the base where a top-down re-grasp
        folds the arm too tightly for cuRobo; the slide moves it to a comfortable diagonal reach. It is
        re-picked + dropped INTO the drawer, so it must stay re-graspable.
      * OBSTACLE → near the base's perpendicular foot on the edge (the nearest stable point), nudged
        ``OBSTACLE_D_NUDGE`` off the straight-ahead fold and clamped to stay in front of the cabinet
        face. A pure distractor, never re-grasped, so a reachable one-shot place is all it needs. It
        then resolves clear of the target's bbox: stagger further toward the face if there is room,
        else hop to the far (``+d``) side past the target.
    Everything is clamped so the bbox stays on the table (a narrow table → falls back to its widest).
    """
    obj_xy = np.asarray(obj_xy, float)[:2]
    # ``p_half`` / ``d_half`` = the object's half-extent toward the corridor (p) and along the edge (d)
    # AFTER its relocate orientation. An elongated object parked long-axis ∥ the edge faces the corridor
    # with only its SHORT half (p_half = short/2) → it sits flush to the edge, well clear of the sweep
    # (using the square obj_half over-pulled it inboard, leaving a long object hugging the corridor).
    ph = obj_half if p_half is None else float(p_half)
    dh = obj_half if d_half is None else float(d_half)
    tcorners = np.array([[x, y] for x in (L.table_lo[0], L.table_hi[0])
                         for y in (L.table_lo[1], L.table_hi[1])])
    p_proj, d_proj = tcorners @ L.p, tcorners @ L.d
    p_edge = float(p_proj.max())                                # +p = near-robot edge (BOTH roles)
    pc = p_edge - (ph + EDGE_MARGIN)                            # pull the bbox in from the edge
    if role == "target":
        dc = float(obj_xy @ L.d)                                # keep along-slide pos ...
        dc += TARGET_D_SHIFT if d_shift is None else float(d_shift)   # ... slid +d off the base line
    else:                                                       # obstacle: foot, nudged off the straight-ahead fold
        min_front = (L.d_front - L.j_current) + dh + EDGE_MARGIN   # never behind the (re-)closed cabinet face
        dc = max(float(L.robot_xy @ L.d) - OBSTACLE_D_NUDGE, min_front)
        if avoid_dc is not None:                                # keep the two parked bboxes from overlapping
            gap = dh + avoid_half + EDGE_MARGIN
            if abs(dc - avoid_dc) < gap:
                back = avoid_dc - gap                           # stagger further -d (toward the face) ...
                dc = back if back >= min_front else avoid_dc + gap   # ... or +d past the target if no front room
    dc = float(np.clip(dc, d_proj.min() + dh + EDGE_MARGIN,
                       d_proj.max() - dh - EDGE_MARGIN))         # clamped on-table (along-edge half)
    return L.to_world(dc, pc)
