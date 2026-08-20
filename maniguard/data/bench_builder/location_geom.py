"""Geometry for the `location` perturbation level of ManiGuard-Bench.

The `location` axis shifts the task objects to a different in-plane position on
the support surface while the **robot base and its init pose A stay fixed** — the
mis-alignment between the moved objects and the stationary arm is exactly the
out-of-distribution signal we want. So this module only ever computes object
displacements; it never touches the robot.

Three concerns, separated so the math is unit-testable without a simulator:

* ``resolve_move_units(env, diag, family)`` — group the live scene's task objects
  into the family's MOVE UNITS (per spec §4d): clutter = each object alone,
  cabinet = target & obstacle independently, jar/lid/stack = the whole pack as one
  rigid unit, dusty = (source+food) and (dest) as two units (sponge re-placed
  separately). Structural entities (robot, surface, goal marker, cabinet fixture)
  are excluded.
* ``sample_displacement(...)`` — deterministic in-plane displacement for one unit
  (seeded by task + unit index): random-plane / along-slide-dir / xy-independent.
* ``clamp_to_surface(...)`` — clamp a displacement so the unit's XY footprint stays
  inside ``surface_info.bounds_xy`` minus ``CLAMP_MARGIN_M`` (never falls off the
  table). A direction retry picks the most feasible move when the first is starved.

``plan_unit_move`` ties sampling + clamp + retry together and reports whether the
clamp starved the move (for QC).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# A fixed bench-wide seed so every location variant is reproducible; the per-unit
# seed mixes in the task id + unit index (see the worker's derive_seed call).
BENCH_LOCATION_SEED = 0x10C0A11  # arbitrary fixed constant for this level

CLAMP_MARGIN_M = 0.03          # keep the footprint >= 3 cm inside the table edge
CLUTTER_CLEARANCE_M = 0.05     # clutter jitter basis (pack min clearance)
_STARVED_FRAC = 0.5            # a move clamped below this fraction of intent -> retry / report
_RETRY_ANGLES = 12             # evenly-spaced fallback directions for random-plane / slide

# Per-family location rule (spec §4d). ``frac`` multiplies the magnitude basis:
# the unit's longest horizontal bbox edge (random_plane / slide_dir) or the fixed
# clutter clearance (xy_independent).
LOCATION_RULES: dict[str, dict[str, Any]] = {
    "clutter_pickup": {"grouping": "each",            "mode": "xy_independent", "frac": (0.3, 0.5)},
    "cabinet_pickup": {"grouping": "target_obstacle", "mode": "slide_dir",      "frac": (0.8, 1.5)},
    "jar_transport":  {"grouping": "all",             "mode": "random_plane",   "frac": (0.8, 1.5)},
    "lid_transport":  {"grouping": "all",             "mode": "random_plane",   "frac": (0.3, 0.5)},
    "dusty_transfer": {"grouping": "dusty_two",       "mode": "random_plane",   "frac": (0.3, 0.5)},
    "stack_retrieve": {"grouping": "all",             "mode": "random_plane",   "frac": (0.3, 0.5)},
}


# --------------------------------------------------------------------------- AABB helpers

def _aabb_lo_hi(obj):
    lo, hi = obj.aabb
    lo = lo.tolist() if hasattr(lo, "tolist") else list(lo)
    hi = hi.tolist() if hasattr(hi, "tolist") else list(hi)
    return [float(x) for x in lo], [float(x) for x in hi]


def union_footprint_xy(objs) -> tuple[list[float], list[float]]:
    """XY (lo, hi) of the union of the objects' world AABBs."""
    los, his = [], []
    for o in objs:
        lo, hi = _aabb_lo_hi(o)
        los.append(lo[:2])
        his.append(hi[:2])
    lo_xy = [min(p[0] for p in los), min(p[1] for p in los)]
    hi_xy = [max(p[0] for p in his), max(p[1] for p in his)]
    return lo_xy, hi_xy


def longest_horizontal_edge(objs) -> float:
    lo, hi = union_footprint_xy(objs)
    return max(hi[0] - lo[0], hi[1] - lo[1])


# --------------------------------------------------------------------------- sampling

def sample_displacement(rule: dict, longest_edge: float, slide_dir, rng) -> np.ndarray:
    """Desired (dx, dy) before clamping. Deterministic in ``rng``."""
    mode = rule["mode"]
    lo, hi = rule["frac"]
    if mode == "xy_independent":
        # clutter: each axis independently R*clearance, R in +/-(lo, hi)
        d = []
        for _ in range(2):
            r = float(rng.uniform(lo, hi)) * (1.0 if rng.random() < 0.5 else -1.0)
            d.append(r * CLUTTER_CLEARANCE_M)
        return np.asarray(d, dtype=np.float64)
    mag = float(rng.uniform(lo, hi)) * float(longest_edge)
    if mode == "slide_dir":
        v = np.asarray(slide_dir, dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-9 else np.asarray([1.0, 0.0])
        return mag * v
    # random_plane
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    return np.asarray([mag * math.cos(theta), mag * math.sin(theta)], dtype=np.float64)


def clamp_to_surface(foot_lo, foot_hi, d, bounds, margin: float = CLAMP_MARGIN_M):
    """Clamp displacement ``d`` so the footprint stays inside ``bounds`` minus
    ``margin`` on every axis. Returns (clamped_d, feasible) where feasible is False
    on an axis whose unit is wider than the inset surface (cannot fit -> 0 on it)."""
    (bx0, by0), (bx1, by1) = bounds
    lo_b = [bx0 + margin, by0 + margin]
    hi_b = [bx1 - margin, by1 - margin]
    out = np.asarray(d, dtype=np.float64).copy()
    feasible = True
    for ax in range(2):
        allow_min = lo_b[ax] - foot_lo[ax]   # most negative move keeping lo inside
        allow_max = hi_b[ax] - foot_hi[ax]   # most positive move keeping hi inside
        if allow_min > allow_max:            # unit bigger than inset surface on this axis
            out[ax] = 0.0
            feasible = False
        else:
            out[ax] = float(min(max(out[ax], allow_min), allow_max))
    return out, feasible


def _retry_directions(mode: str, mag: float, slide_dir) -> list:
    """Alternative displacement directions to try when the clamp starves the first
    pick — MODE-AWARE so the retry never violates the family's direction rule:
      * slide_dir (cabinet) — only along the drawer axis, both signs.
      * random_plane (jar/lid/stack/dusty) — evenly-spaced full-circle angles.
      * xy_independent (clutter) — no retry (moves are tiny, clamp rarely bites).
    """
    if mode == "slide_dir":
        v = np.asarray(slide_dir, dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-9 else np.asarray([1.0, 0.0])
        return [mag * v, -mag * v]
    if mode == "random_plane":
        return [np.asarray([mag * math.cos(2.0 * math.pi * k / _RETRY_ANGLES),
                            mag * math.sin(2.0 * math.pi * k / _RETRY_ANGLES)])
                for k in range(_RETRY_ANGLES)]
    return []


def plan_unit_move(foot_lo, foot_hi, longest_edge, rule, slide_dir, bounds, seed) -> dict:
    """Sample + clamp (+ mode-aware direction retry) one unit's move. Returns a
    record with the final displacement and whether the clamp starved it."""
    rng = np.random.default_rng(int(seed))
    desired = sample_displacement(rule, longest_edge, slide_dir, rng)
    desired_mag = float(np.linalg.norm(desired))
    best, best_feasible = clamp_to_surface(foot_lo, foot_hi, desired, bounds)
    best_clamped_mag = float(np.linalg.norm(best))

    # If the first direction was starved by the clamp, try mode-appropriate
    # alternatives at the same magnitude and keep the largest feasible move.
    if desired_mag > 1e-6 and best_clamped_mag < _STARVED_FRAC * desired_mag:
        for cand in _retry_directions(rule["mode"], desired_mag, slide_dir):
            cd, feas = clamp_to_surface(foot_lo, foot_hi, cand, bounds)
            m = float(np.linalg.norm(cd))
            if m > best_clamped_mag:
                best, best_clamped_mag, best_feasible = cd, m, feas

    starved = desired_mag > 1e-6 and best_clamped_mag < _STARVED_FRAC * desired_mag
    return {
        "displacement": [round(float(best[0]), 5), round(float(best[1]), 5)],
        "desired_mag": round(desired_mag, 5),
        "final_mag": round(best_clamped_mag, 5),
        "starved": bool(starved or not best_feasible),
    }


# --------------------------------------------------------------------------- move-unit resolution

def aabb_xy_overlap(box_a, box_b, margin: float = 0.0) -> bool:
    """True if two XY AABBs overlap (inflated by ``margin``). Each box = (lo, hi)."""
    (alx, aly), (ahx, ahy) = box_a
    (blx, bly), (bhx, bhy) = box_b
    return not (ahx + margin <= blx or bhx + margin <= alx
                or ahy + margin <= bly or bhy + margin <= aly)


def fallback_rule(family: str, slide_dir):
    """Phase-2 fallback rule, used only when ALL primary attempts fail. Moves
    PERPENDICULAR to the primary axis at a gentler 0.5-0.8x bbox magnitude:
      * cabinet — perpendicular to the slide axis (reuses 'slide_dir' mode with the
        perpendicular vector, so both signs are tried), giving the target a lateral
        escape when the drawer axis is blocked (edge on one side, out-of-reach on
        the other).
      * omnidirectional families — a fresh random-plane sweep at the same gentler
        band (their primary already covers all directions, so this just retries at
        a different, usually smaller magnitude).
    Returns (rule, slide_for_phase).
    """
    if family == "cabinet_pickup" and slide_dir is not None:
        sx, sy = float(slide_dir[0]), float(slide_dir[1])
        perp = [-sy, sx]
        return {"grouping": LOCATION_RULES[family]["grouping"], "mode": "slide_dir",
                "frac": (0.5, 0.8)}, perp
    return {"grouping": LOCATION_RULES[family]["grouping"], "mode": "random_plane",
            "frac": (0.5, 0.8)}, None


def surface_bounds_xy(diag: dict):
    b = ((diag.get("surface_info") or {}).get("bounds_xy"))
    if not b:
        return None
    return ((float(b[0][0]), float(b[0][1])), (float(b[1][0]), float(b[1][1])))


def _spawn_role_categories(diag: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in ((diag.get("selection") or {}).get("spawn_specs") or []):
        out.setdefault(s.get("role"), []).append(s.get("category"))
    return out


def _structural_names(env, diag: dict, family: str) -> set[str]:
    """Names that are NOT movable task objects: the robot, the support surface
    entity, the goal marker, and (cabinet) the cabinet fixture."""
    names: set[str] = {r.name for r in env.robots}
    gr = diag.get("goal_region") or {}
    for key in ("marker_name", "support_name"):
        if gr.get(key):
            names.add(gr[key])
    surf = diag.get("surface") or ""
    surf_cat = (diag.get("surface_info") or {}).get("category")
    surf_model = (diag.get("surface_info") or {}).get("model")
    for o in env.scene.objects:
        if o.name == surf or (surf_cat and getattr(o, "category", "") == surf_cat
                              and getattr(o, "model", None) == surf_model):
            names.add(o.name)
    if family == "cabinet_pickup":
        cab = (diag.get("cabinet_info") or {}).get("name")
        if cab:
            names.add(cab)
    return names


def _objs_by_category(env, cats, exclude: set[str]):
    cats = {c for c in (cats or []) if c}
    return [o for o in env.scene.objects
            if getattr(o, "category", "") in cats and o.name not in exclude]


def resolve_move_units(env, diag: dict, family: str) -> dict:
    """Return ``{"units": [[obj, ...], ...], "sponge": obj|None}`` — the family's
    move units (each a list of objects translated together by one vector) plus the
    dusty sponge (re-placed separately, not displaced)."""
    rule = LOCATION_RULES[family]
    structural = _structural_names(env, diag, family)
    task_objs = [o for o in env.scene.objects if o.name not in structural]
    grouping = rule["grouping"]
    sponge = None

    if grouping == "each":
        units = [[o] for o in task_objs]
    elif grouping == "all":
        units = [task_objs] if task_objs else []
    elif grouping == "target_obstacle":
        roles = _spawn_role_categories(diag)
        tgt_name = (diag.get("target_info") or {}).get("name")
        target = next((o for o in task_objs if o.name == tgt_name), None) \
            or next(iter(_objs_by_category(env, roles.get("target"), structural)), None)
        obstacle = next(iter(_objs_by_category(env, roles.get("obstacle"),
                                               structural | ({target.name} if target else set()))), None)
        units = [[o] for o in (target, obstacle) if o is not None]
    elif grouping == "dusty_two":
        roles = _spawn_role_categories(diag)
        sponge_name = diag.get("sponge_name")
        sponge = next((o for o in env.scene.objects if o.name == sponge_name), None)
        used = set(structural) | ({sponge.name} if sponge else set())
        src = _objs_by_category(env, roles.get("source"), used)
        food = _objs_by_category(env, roles.get("food"), used | {o.name for o in src})
        dest = _objs_by_category(env, roles.get("dest"), used | {o.name for o in src} | {o.name for o in food})
        units = [u for u in (src + food, dest) if u]
    else:
        units = [task_objs]

    return {"units": units, "sponge": sponge}
