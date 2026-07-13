"""Geometry + room-matching for the `env` perturbation level of ManiGuard-Bench.

The `env` axis takes a finalized `base` task — a canonical arm + one support
surface + the task objects, with NO surrounding room — and **injects it,
rigidly, into a real BEHAVIOR room** so the policy sees the same manipulation
geometry against a totally different visual background (walls, furniture,
clutter). That background mismatch is the out-of-distribution signal.

The injection is anchored on the **support table**: we find the SAME table model
already present in some room, then map the whole base layout onto it with one
rigid transform::

    T = T_room_table · T_base_table⁻¹           # SE(3), table -> table

Because the room table and the base table are the *same model* (identical mesh),
the table-top-relative offset is identical, so a single rigid ``T`` applied to
every base object (task objects + cabinet fixture + goal marker + arm + review
cameras) drops the layout onto the room table at exactly the right height and
orientation — no per-object height fix-ups.

This module is deliberately simulator-free so the math + the room matching are
unit-testable offline:

* :func:`build_table_scene_db` — scan the 51 room ``*_best.json`` files once and
  index every instance of the table models the bench actually uses
  (``{model: [{scene, name, pos, ori, category}]}``). Saved to
  :data:`TABLE_SCENE_DB_PATH`.
* :func:`select_room_instance` — pick which room + table instance a base task
  injects into: the ORIGINAL room when the base carries a ``scene_model`` that
  still holds the table (→ ``T`` is identity), else a seeded-random room among
  those that contain the model.
* :func:`compute_injection_transform` / :func:`apply_transform_to_pose` /
  :func:`apply_transform_to_point` — the rigid-transform primitives.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from maniguard.data.bench_builder.perturbation import derive_seed

# A fixed bench-wide seed so every env variant's room choice is reproducible; the
# per-task seed mixes in family + task id (see select_room_instance).
BENCH_ENV_SEED = 0xE0E0A11  # arbitrary fixed constant for this level

# Generated table -> rooms index lives next to this module (Step 0 output).
TABLE_SCENE_DB_PATH = Path(__file__).resolve().parent / "data" / "table_scene_db.json"

# Two instances of the same model are treated as the "same" table (→ identity T)
# when their world poses agree within this tolerance.
SAME_TABLE_TOL_M = 1e-3


# --------------------------------------------------------------------------- scene-json access

def _scene_registry(scene_info: dict) -> dict:
    """Per-object state registry, tolerant of both snapshot schemas."""
    st = scene_info.get("state", {})
    if "registry" in st:
        return st["registry"].get("object_registry", {})
    return st.get("object_registry", {})


def _best_json(scenes_root: Path, scene: str) -> Path:
    return scenes_root / scene / "json" / f"{scene}_best.json"


# --------------------------------------------------------------------------- rigid transforms
# Scene JSON quaternions are xyzw (w last): identity = [0, 0, 0, 1].

def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def mat_to_quat_xyzw(m: np.ndarray) -> list[float]:
    """Rotation matrix -> xyzw quaternion (Shepperd's method, numerically stable)."""
    m = np.asarray(m, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= (np.linalg.norm(q) or 1.0)
    return [float(v) for v in q]


def pose_to_mat4(pos, ori) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_xyzw_to_mat(ori)
    m[:3, 3] = [float(v) for v in pos]
    return m


def mat4_to_pose(m: np.ndarray) -> tuple[list[float], list[float]]:
    m = np.asarray(m, dtype=float)
    return [float(v) for v in m[:3, 3]], mat_to_quat_xyzw(m[:3, :3])


def compute_injection_transform(base_table_pose, room_table_pose) -> np.ndarray:
    """``T = T_room_table · T_base_table⁻¹`` — maps base-frame poses into the room.

    ``*_table_pose`` is ``(pos[3], ori_xyzw[4])``. Applied to every injected base
    object so the base table lands exactly on the room table.
    """
    t_base = pose_to_mat4(*base_table_pose)
    t_room = pose_to_mat4(*room_table_pose)
    return t_room @ np.linalg.inv(t_base)


def apply_transform_to_pose(T: np.ndarray, pos, ori) -> tuple[list[float], list[float]]:
    return mat4_to_pose(T @ pose_to_mat4(pos, ori))


def apply_transform_to_point(T: np.ndarray, p) -> list[float]:
    v = T @ np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
    return [float(v[0]), float(v[1]), float(v[2])]


# --------------------------------------------------------------------------- base surfaces

def collect_base_surfaces(bench_root: Path) -> dict[str, dict]:
    """``{surface_model: {"category": str, "tasks": [(family, task_id)]}}`` for the
    surface every finalized base task rests on (read from its ``surface_info``)."""
    out: dict[str, dict] = defaultdict(lambda: {"category": None, "tasks": []})
    for diag_path in sorted(bench_root.glob("*/task_*/base/diagnostics.jsonl")):
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        si = diag.get("surface_info") or {}
        model = si.get("model")
        if not model:
            continue
        family = diag_path.parents[2].name
        task = diag_path.parents[1].name
        out[model]["category"] = si.get("category")
        out[model]["tasks"].append((family, task))
    return dict(out)


# --------------------------------------------------------------------------- room index (db)

def build_table_scene_db(scenes_root: Path, models: set[str]) -> dict[str, list[dict]]:
    """Scan every room ``*_best.json`` and index each instance of ``models``.

    Returns ``{model: [{scene, name, pos, ori, category}]}``. Only the table
    models the bench actually uses are indexed, so the file stays small.
    """
    scenes_root = Path(scenes_root)
    index: dict[str, list[dict]] = {m: [] for m in models}
    for scene in sorted(p.name for p in scenes_root.iterdir() if p.is_dir()):
        bj = _best_json(scenes_root, scene)
        if not bj.exists():
            continue
        sc = json.loads(bj.read_text(encoding="utf-8"))
        init = sc.get("objects_info", {}).get("init_info", {})
        reg = _scene_registry(sc)
        for name, info in init.items():
            args = info.get("args", {})
            model = args.get("model")
            if model not in index:
                continue
            rl = reg.get(name, {}).get("root_link", {})
            pos, ori = rl.get("pos"), rl.get("ori")
            if pos is None or ori is None:
                continue
            index[model].append({
                "scene": scene,
                "name": name,
                "pos": [float(v) for v in pos],
                "ori": [float(v) for v in ori],
                "category": args.get("category"),
            })
    return index


def select_room_instance(
    candidates: list[dict],
    *,
    scene_model: str | None,
    base_table_pose: tuple[Any, Any] | None,
    seed: int,
) -> dict | None:
    """Choose which room + table instance a base task injects into.

    Decision order (spec §4a):
      1. If the base carries a ``scene_model`` that still holds the table, use
         the ORIGINAL room. Among that room's instances of the model, prefer the
         one whose world pose matches the base table pose (→ ``T`` identity).
      2. Otherwise pick a seeded-random instance among all rooms that hold the
         model.

    Returns the chosen ``{scene, name, pos, ori, ...}`` or ``None`` if the model
    is in no room (the task is then skipped + recorded for case-by-case review).
    """
    if not candidates:
        return None
    if scene_model:
        in_room = [c for c in candidates if c["scene"] == scene_model]
        if in_room:
            if base_table_pose is not None:
                bp = np.asarray(base_table_pose[0], dtype=float)
                in_room = sorted(
                    in_room,
                    key=lambda c: float(np.linalg.norm(np.asarray(c["pos"], dtype=float) - bp)),
                )
            return in_room[0]
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(len(candidates)))]


# --------------------------------------------------------------------------- table SUBSTITUTION
# When a base task's ONLY room fails to spawn (objects fall / arm collides / room asset crashes)
# even after decluttering, we re-spawn the base layout on a DIFFERENT, adequately-sized table in a
# different room (spec fallback). The substitute table just provides a surface + the room provides
# background. Selection is purely geometric: the substitute surface must fit the base task's object
# pack (width / length / area), with a small margin.

SUBSTITUTE_MARGIN_M = 0.02  # the substitute surface must exceed the pack footprint by >= this

_OBJECT_FOOTPRINTS = None


def _load_object_footprints() -> dict:
    """``{category: {model: {extent_xyz, footprint_m2}}}`` — per-object unscaled axis-aligned bbox,
    from the task-gen footprint table. Used to size the base task's object pack offline."""
    global _OBJECT_FOOTPRINTS
    if _OBJECT_FOOTPRINTS is None:
        p = Path(__file__).resolve().parents[2] / "task_generation" / "utils" / "object_footprints.json"
        _OBJECT_FOOTPRINTS = json.loads(p.read_text(encoding="utf-8"))
    return _OBJECT_FOOTPRINTS


def base_surface_name(base_si: dict, diag: dict) -> str | None:
    """Name of the base support-surface object: a direct ``diag['surface']`` name when it is one,
    else the object matching ``surface_info`` cat+model (cabinet/jar carry a 'category/model' string
    + a generic ``support_surface``)."""
    init = base_si.get("objects_info", {}).get("init_info", {})
    surf = diag.get("surface") or ""
    if surf in init:
        return surf
    si = diag.get("surface_info") or {}
    cat, model = si.get("category"), si.get("model")
    for name, info in init.items():
        a = info.get("args", {})
        if a.get("category") == cat and a.get("model") == model:
            return name
    return None


def _as_scale3(scale) -> list[float]:
    if scale is None:
        return [1.0, 1.0, 1.0]
    if isinstance(scale, (int, float)):
        return [float(scale)] * 3
    return [float(scale[0]), float(scale[1]), float(scale[2])]


def collect_model_surface_geom(bench_root: Path) -> dict[str, dict]:
    """``{model: {category, w, l, area, top_offset, center_offset_xy}}`` — each table model's surface
    geometry, derived from the FIRST finalized base task that uses it (surface_info + the table
    object's origin). ``top_offset`` = surface top z − table origin z; ``center_offset_xy`` = surface
    bounds centre − table origin xy (table local frame; base table ori is ~identity)."""
    bench_root = Path(bench_root)
    geom: dict[str, dict] = {}
    for diag_path in sorted(bench_root.glob("*/task_*/base/diagnostics.jsonl")):
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        si = diag.get("surface_info") or {}
        model = si.get("model")
        bounds = si.get("bounds_xy")
        if not model or model in geom or not bounds:
            continue
        (lox, loy), (hix, hiy) = bounds
        scene_file = diag_path.parent / "scene_ep1.json"
        if not scene_file.exists():
            continue
        base_si = json.loads(scene_file.read_text(encoding="utf-8"))
        surf_name = base_surface_name(base_si, diag)
        opos = (_scene_registry(base_si).get(surf_name, {}).get("root_link", {}) or {}).get("pos")
        if not opos:
            continue
        ox, oy, oz = float(opos[0]), float(opos[1]), float(opos[2])
        geom[model] = {
            "category": si.get("category"),
            "w": round(float(hix - lox), 4),
            "l": round(float(hiy - loy), 4),
            "area": float(si.get("area_m2") or (hix - lox) * (hiy - loy)),
            "top_offset": round(float(si.get("top_z", oz)) - oz, 4),
            "center_offset_xy": [round((lox + hix) / 2 - ox, 4), round((loy + hiy) / 2 - oy, 4)],
        }
    return geom


def compute_pack_footprint(base_si: dict, diag: dict) -> dict | None:
    """The base task's object-pack footprint (everything resting on the table — task objects +
    fixtures + goal marker, excluding the robot + the support surface). Returns ``{w, l, area,
    center_xy, top_z}``. Object XY extents come from ``object_footprints`` (DatasetObjects) or the
    primitive radius (the goal marker); object rotation is ignored (axis-aligned extent at the
    origin) — a slight under-estimate that the +margin + the in-sim feasibility check absorb."""
    fp = _load_object_footprints()
    init = base_si.get("objects_info", {}).get("init_info", {})
    reg = _scene_registry(base_si)
    surf = base_surface_name(base_si, diag)
    los_x, his_x, los_y, his_y = [], [], [], []
    for name, info in init.items():
        if name == surf:
            continue
        cls = info.get("class_name", "")
        if cls == "FrankaPanda" or "robot" in info.get("class_module", "").lower():
            continue
        pos = (reg.get(name, {}).get("root_link", {}) or {}).get("pos")
        if not pos:
            continue
        cx, cy = float(pos[0]), float(pos[1])
        a = info.get("args", {})
        if cls == "PrimitiveObject":
            hx = hy = float(a.get("radius", 0.03))
        else:
            ext = ((fp.get(a.get("category"), {}) or {}).get(a.get("model"), {}) or {}).get("extent_xyz")
            sc = _as_scale3(a.get("scale"))
            hx = float(ext[0]) * sc[0] / 2 if ext else 0.06
            hy = float(ext[1]) * sc[1] / 2 if ext else 0.06
        los_x.append(cx - hx); his_x.append(cx + hx); los_y.append(cy - hy); his_y.append(cy + hy)
    if not los_x:
        return None
    lox, hix, loy, hiy = min(los_x), max(his_x), min(los_y), max(his_y)
    return {"w": round(hix - lox, 4), "l": round(hiy - loy, 4),
            "area": round((hix - lox) * (hiy - loy), 4),
            "center_xy": [round((lox + hix) / 2, 4), round((loy + hiy) / 2, 4)],
            "top_z": (diag.get("surface_info") or {}).get("top_z")}


def substitute_transform(pack: dict, base_top: float, table_pos, table_ori, geom: dict) -> np.ndarray:
    """Pure-translation transform that re-anchors the base layout onto a SUBSTITUTE table: base pack
    XY centre → substitute surface XY centre, base table top → substitute table top. No rotation
    (the pack sits on the flat surface in its base orientation; the room background + declutter +
    the in-sim feasibility check handle the surroundings)."""
    R = quat_xyzw_to_mat(table_ori)
    co = geom["center_offset_xy"]
    off = R @ np.array([float(co[0]), float(co[1]), 0.0])
    sub_cx = float(table_pos[0]) + float(off[0])
    sub_cy = float(table_pos[1]) + float(off[1])
    sub_top = float(table_pos[2]) + float(geom["top_offset"])
    T = np.eye(4)
    T[0, 3] = sub_cx - float(pack["center_xy"][0])
    T[1, 3] = sub_cy - float(pack["center_xy"][1])
    T[2, 3] = sub_top - float(base_top)
    return T


def surface_world_box(table_pos, table_ori, geom: dict, pad: float = 0.0) -> tuple[float, float, float, float]:
    """World-axis-aligned XY box ``(cx, cy, hx, hy)`` of a table's placement surface, from the model
    ``geom`` (``w``/``l``/``center_offset_xy``) + the table's world pose. ``hx/hy`` are the surface
    rectangle's half-extents rotated into world axes (a slight over-cover for a rotated table), + pad.
    Used to drop the room's on-surface obstacles within the merged scene before it is ever built."""
    R = quat_xyzw_to_mat(table_ori)
    co = geom.get("center_offset_xy", [0.0, 0.0])
    off = R @ np.array([float(co[0]), float(co[1]), 0.0])
    cx = float(table_pos[0]) + float(off[0])
    cy = float(table_pos[1]) + float(off[1])
    w, l = float(geom["w"]), float(geom["l"])
    c, s = abs(float(R[0, 0])), abs(float(R[1, 0]))  # |cos yaw|, |sin yaw|
    hx = c * w / 2.0 + s * l / 2.0 + pad
    hy = s * w / 2.0 + c * l / 2.0 + pad
    return (cx, cy, hx, hy)


def find_substitute_candidates(pack: dict, model_geom: dict, index: dict, *,
                               exclude_rooms: set | None = None, prefer: list | None = None,
                               margin_m: float = SUBSTITUTE_MARGIN_M) -> list[dict]:
    """Ordered substitute (room, table) candidates whose surface fits the pack (orientation-agnostic:
    sorted dims + area, each ≥ pack + margin). ``prefer`` is a list of ``(scene, model)`` already
    proven feasible (sibling env tasks) — those come first; the rest are ordered smallest-adequate-
    area first (keeps the arm near the table edge). Excludes ``exclude_rooms``."""
    exclude_rooms = exclude_rooms or set()
    prefer_set = set(prefer or [])
    pmin, pmax = sorted((pack["w"], pack["l"]))
    fits = []
    for model, g in model_geom.items():
        tmin, tmax = sorted((g["w"], g["l"]))
        if tmin >= pmin + margin_m and tmax >= pmax + margin_m and g["area"] >= pack["area"] + margin_m:
            for inst in index.get(model, []):
                if inst["scene"] in exclude_rooms:
                    continue
                fits.append({**inst, "model": model, "dims": g,
                             "preferred": (inst["scene"], model) in prefer_set})
    # sibling-proven (room, model) first; within each group the smallest adequate surface first
    # (keeps the pack centred + the arm near the table edge rather than sitting on a large table).
    fits.sort(key=lambda c: (0 if c["preferred"] else 1, c["dims"]["area"]))
    return fits


# --------------------------------------------------------------------------- db builder CLI

def _default_scenes_root() -> Path:
    """Resolve the BEHAVIOR ``behavior-1k-assets/scenes`` dir: ``$OMNIGIBSON_DATA_PATH`` first
    (the standard override), else the repo-local ``datasets/`` (the default install layout)."""
    data = os.environ.get("OMNIGIBSON_DATA_PATH", "")
    roots = []
    if data:
        roots.append(Path(data) / "behavior-1k-assets" / "scenes")
    repo = Path(__file__).resolve().parents[3]
    roots.append(repo / "datasets" / "behavior-1k-assets" / "scenes")
    for r in roots:
        if r.exists():
            return r
    raise FileNotFoundError(
        f"no scenes root found (set OMNIGIBSON_DATA_PATH); tried {[str(r) for r in roots]}")


def _scenes_root_label(scenes_root: Path) -> str:
    """Dataset-relative tail of the scenes root (e.g. ``behavior-1k-assets/scenes``) for the db's
    ``_meta`` — keeps the committed provenance machine-independent (no absolute local path)."""
    parts = Path(scenes_root).parts
    if "behavior-1k-assets" in parts:
        return str(Path(*parts[parts.index("behavior-1k-assets"):]))
    return Path(scenes_root).name


def build_and_save(bench_root: Path, scenes_root: Path | None = None) -> dict:
    """Build the table->scene db from the finalized base tasks + room files, write
    it to :data:`TABLE_SCENE_DB_PATH`, and return a coverage summary."""
    scenes_root = Path(scenes_root) if scenes_root else _default_scenes_root()
    surfaces = collect_base_surfaces(Path(bench_root))
    index = build_table_scene_db(scenes_root, set(surfaces))

    covered_models = {m for m, hits in index.items() if hits}
    covered_tasks, missing_tasks = [], []
    for model, meta in surfaces.items():
        bucket = covered_tasks if model in covered_models else missing_tasks
        bucket.extend((*t, model, meta["category"]) for t in meta["tasks"])

    db = {
        "_meta": {
            "scenes_root": _scenes_root_label(scenes_root),
            "n_scenes": sum(1 for p in scenes_root.iterdir() if p.is_dir()),
            "n_surface_models": len(surfaces),
            "n_models_covered": len(covered_models),
            "n_tasks_covered": len(covered_tasks),
            "n_tasks_missing": len(missing_tasks),
            "missing_tasks": sorted(missing_tasks),
        },
        "index": index,
    }
    TABLE_SCENE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_SCENE_DB_PATH.write_text(json.dumps(db, indent=2) + "\n", encoding="utf-8")
    return db["_meta"]


def load_table_scene_db() -> dict[str, list[dict]]:
    db = json.loads(TABLE_SCENE_DB_PATH.read_text(encoding="utf-8"))
    return db["index"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the table->scene db for env injection.")
    ap.add_argument("--bench-root",
                    default="outputs/lerobot_datasets/maniguard-bench",
                    help="finalized maniguard-bench root (reads */task_*/base/diagnostics.jsonl)")
    ap.add_argument("--scenes-root", default=None,
                    help="behavior-1k-assets/scenes dir (default: auto-resolve)")
    args = ap.parse_args()

    meta = build_and_save(Path(args.bench_root), args.scenes_root)
    print(f"wrote {TABLE_SCENE_DB_PATH}")
    print(f"  scenes scanned        : {meta['n_scenes']}")
    print(f"  surface models (base) : {meta['n_surface_models']}")
    print(f"  models covered        : {meta['n_models_covered']}")
    print(f"  tasks covered         : {meta['n_tasks_covered']}")
    print(f"  tasks MISSING (skip)  : {meta['n_tasks_missing']}")
    for fam, task, model, cat in meta["missing_tasks"]:
        print(f"      - {fam}/{task}  {cat}/{model}")
