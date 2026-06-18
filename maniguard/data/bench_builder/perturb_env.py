"""Build the `env` (surrounding-room) perturbation level for ManiGuard-Bench.

The `env` axis injects a finalized `base` task — canonical arm + one support table
+ the task objects, NO room — rigidly into a real BEHAVIOR room, anchored on the
support table (the SAME table model already present in that room). The policy sees
the identical manipulation geometry against a totally different visual background
(walls, furniture, clutter); that mismatch is the out-of-distribution signal. See
``env_geom`` for the room matching + the rigid transform ``T = T_room · T_base⁻¹``.

The injection is an OFFLINE scene-JSON MERGE (no live object injection): the room's
``<scene>_best.json`` + the transformed base objects are merged into one scene_file
dict, which ``build_env_config`` loads as an ``InteractiveTraversableScene`` in one
shot (the loader instantiates every object in ``objects_info.init_info``, room +
injected alike). The base table is dropped — the room already holds it.

STEP-1 (this version) is a de-risk SMOKE TEST: it proves the merged scene loads, the
layout lands on the room table, and nothing grossly penetrates. It loads → mounts the
arm on the room table → settles the injected objects → renders the 4 review videos →
saves the merged snapshot, and writes a ``base | env`` opposite-view compare PNG for
visual QC. The full finalize parity (fresh gate + LTL recompute + owned-diagnostics
schema, mirroring ``finalize_base``) is Step 2.

Same fresh-subprocess-per-task pattern as ``perturb_location``.

Usage:
  python -m maniguard.data.bench_builder.perturb_env --family lid_transport --tasks 0
  python -m maniguard.data.bench_builder.perturb_env --family cabinet_pickup --tasks 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from maniguard.data.bench_builder import env_geom as eg

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
LEVEL = "env"
ROW_FILE = "_env_row.json"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")
COMPARE_VIEW = "opposite_side_front"
# Heal the teleport-contact of the injected objects, then let them converge to rest
# before the snapshot — same two-phase settle as `location` (HOLD damps the teleport
# impulse, FREE converges + verifies stability).
SETTLE_STEPS = 80
SETTLE_HOLD = 20
# Residual arm drift over the idle-step → the arm still collides with something the declutter
# left (a wall it reaches, furniture beyond the clear margin). The env feasibility signal.
ARM_DRIFT_TOL = 1e-2  # rad
# An injected object whose in-sim AABB bottom sits more than this below the table top has SUNK into
# / interpenetrates the surface (the fallen-0.3 check + the LTL "on support" predicate both miss a
# shallow sink, e.g. a container settling halfway into a substitute table).
SINK_TOL = 0.03  # m
# Keep only the room background within this radius of the anchor table. The eval cameras see ~2-3 m
# around the table, so objects beyond this are invisible — dropping them bounds the merged scene's
# prebuild cost regardless of room size (some "rooms" are a whole 1300-object conference hall).
ENV_BG_RADIUS_M = 4.0
_STRUCT_KEEP_KEYWORDS = ("floor",)            # always kept (the visible ground)
_STRUCT_DROP_KEYWORDS = ("wall", "ceiling", "roof")  # dropped here (Stage 1 removes them anyway)


# ---------------------------------------------------------------------------- helpers

def _scene_registry(scene_info: dict) -> dict:
    st = scene_info.get("state", {})
    if "registry" in st:
        return st["registry"].get("object_registry", {})
    return st.get("object_registry", {})


def _load_diag(base_dir: Path, episode: int) -> dict:
    txt = (base_dir / "diagnostics.jsonl").read_text(encoding="utf-8")
    try:
        d = json.loads(txt)
        return d if isinstance(d, dict) else d[0]
    except json.JSONDecodeError:
        return json.loads([ln for ln in txt.splitlines() if ln.strip()][0])


def _is_complete(out_dir: Path, episode: int) -> bool:
    if not (out_dir / f"scene_ep{episode}.json").is_file():
        return False
    if not (out_dir / "diagnostics.jsonl").is_file():
        return False
    return all((out_dir / f"rollout_{lbl}_ep{episode}.mp4").is_file() for lbl in VIDEO_LABELS)


def _select_tasks(out_fam: Path, spec: str | None, episode: int) -> list[str]:
    available = sorted(
        d.name for d in out_fam.glob("task_*")
        if (d / "base" / f"scene_ep{episode}.json").is_file()
    )
    if not spec:
        return available
    avail = set(available)

    def norm(tok: str) -> str:
        tok = tok.strip()
        return tok if tok.startswith("task_") else f"task_{int(tok):04d}"

    if "-" in spec and "," not in spec and not spec.startswith("task_"):
        lo, hi = spec.split("-", 1)
        chosen = {f"task_{n:04d}" for n in range(int(lo), int(hi) + 1)}
    else:
        chosen = {norm(t) for t in spec.split(",")}
    return [t for t in available if t in (chosen & avail)]


def _worker_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OMNIGIBSON_HEADLESS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _base_surface_name(base_si: dict, diag: dict) -> str | None:
    """Name of the base support-surface object: a direct ``diag['surface']`` name when it
    is one (clutter/lid/stack/dusty), else the object matching ``surface_info`` cat+model
    (cabinet/jar carry a 'category/model' surface string + a generic ``support_surface``)."""
    init = base_si["objects_info"]["init_info"]
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


def _proximity_filter_room(scene_info: dict, table_name: str, radius: float) -> int:
    """In-place: keep the anchor table + floors + any object whose origin is within ``radius`` of
    the table; drop walls/ceiling/roof (Stage 1 removes them anyway) and far objects. Bounds the
    merged scene's object count so the per-task USD prebuild stays fast even when the room-instance
    is a 1300-object conference hall. Returns the number of objects dropped."""
    init = scene_info.get("objects_info", {}).get("init_info", {})
    reg = _scene_registry(scene_info)
    tpos = (reg.get(table_name, {}).get("root_link", {}) or {}).get("pos")
    tx, ty = (float(tpos[0]), float(tpos[1])) if tpos else (0.0, 0.0)
    drop = []
    for name, info in init.items():
        if name == table_name:
            continue
        cat = str(info.get("args", {}).get("category", "") or "").lower()
        if any(k in cat for k in _STRUCT_KEEP_KEYWORDS):
            continue  # floors: always keep (the visible ground)
        if any(k in cat for k in _STRUCT_DROP_KEYWORDS):
            drop.append(name)
            continue
        pos = (reg.get(name, {}).get("root_link", {}) or {}).get("pos")
        if pos is None or ((float(pos[0]) - tx) ** 2 + (float(pos[1]) - ty) ** 2) ** 0.5 > radius:
            drop.append(name)
    for name in drop:
        init.pop(name, None)
        reg.pop(name, None)
    return len(drop)


# An injected object spawned OVERLAPPING a room object that already sits ON the anchor surface (a
# laptop, potted plant, cash register the room keeps on that table) is violently ejected by PhysX on
# the very first physics step — BEFORE the live ``clear_on_surface_obstacles`` removes that room
# object. A flat object self-rights, but a tall fragile topples (→ all_fragiles_upright violated) or,
# against a bulky fixture, blows up to a NaN pose (→ load crash). So we drop the room's on-surface
# obstacles that fall within the injected pack's footprint OFFLINE — before the scene is ever built —
# so the injected objects spawn onto a clean surface. The live pass still runs as a precise backstop.
SPAWN_CLEAR_EDGE_PAD_M = 0.10  # extend the anchor-surface footprint by this (catch edge-overhang obstacles)
SPAWN_CLEAR_Z_BELOW = 0.05     # a room obj is "on the surface" iff surf_top - this <= origin_z ...
SPAWN_CLEAR_Z_ABOVE = 0.60     # ... <= surf_top + this (sits on the table, not a ceiling fixture)


def _clear_room_spawn_obstacles(merged: dict, injected: list[str], anchor_name: str,
                                surf_top: float | None, surf_box) -> list[str]:
    """Drop room objects sitting ON the anchor surface (within its footprint ``surf_box`` =
    ``(cx, cy, hx, hy)`` + at table height), so the injected objects never spawn interpenetrating
    them — the cause of the first-step ejection that topples fragiles / NaN-crashes against bulky
    fixtures. Matches the LIVE ``clear_on_surface_obstacles`` set but removes it OFFLINE, before the
    scene is built. In-place on ``merged``; returns the dropped names. No-op when the box is unknown."""
    if surf_top is None or surf_box is None:
        return []
    cx, cy, hx, hy = surf_box
    init = merged.get("objects_info", {}).get("init_info", {})
    reg = _scene_registry(merged)
    inj = set(injected)
    drop = []
    for name, info in init.items():
        if name == anchor_name or name in inj:
            continue
        cat = str(info.get("args", {}).get("category", "") or "").lower()
        if any(k in cat for k in _STRUCT_KEEP_KEYWORDS):
            continue  # floors: never remove the visible ground
        p = (reg.get(name, {}).get("root_link", {}) or {}).get("pos")
        if not p or not (surf_top - SPAWN_CLEAR_Z_BELOW <= float(p[2]) <= surf_top + SPAWN_CLEAR_Z_ABOVE):
            continue
        if abs(float(p[0]) - cx) <= hx and abs(float(p[1]) - cy) <= hy:
            drop.append(name)
    for name in drop:
        init.pop(name, None)
        reg.pop(name, None)
    return drop


def build_merged_scene_info(base_si: dict, room_si: dict, base_surf_name: str,
                            room_table_name: str, scene_model: str, T,
                            anchor_surf_top: float | None = None, anchor_surf_box=None,
                            ) -> tuple[dict, list[str], str | None, list[str]]:
    """Merge the transformed base objects into the room scene_file.

    First TRIM the room to just the room-instance that holds the anchor table (a full building
    is 130–850 objects; prebuilding that USD per task is prohibitively slow / hangs — and we only
    want the table's own room as background anyway). Then add a top-level ``init_info`` so
    ``build_env_config`` dispatches to ``InteractiveTraversableScene``; inject every base object
    EXCEPT the support table (the room already holds it), each with its pose mapped by ``T`` (full
    registry entry preserved — only ``root_link.pos/ori`` is transformed, so a cabinet drawer's
    ``joint_pos`` etc. survive). Finally, when ``anchor_surf_top`` is known, drop the room's
    on-surface objects that overlap the injected pack footprint (``_clear_room_spawn_obstacles``)
    so nothing spawns interpenetrating. Returns ``(merged, injected_names, room_instance,
    removed_spawn_obstacles)``.
    """
    from maniguard.data.scene.trim_scene_to_room import trim_scene_info_to_room

    table_args = (room_si.get("objects_info", {}).get("init_info", {})
                  .get(room_table_name, {}).get("args", {}) or {})
    in_rooms = table_args.get("in_rooms") or []
    room_instance = in_rooms[0] if in_rooms else None
    if room_instance:
        merged, _ = trim_scene_info_to_room(room_si, room_instance, keep_robot=False)
    else:
        merged = copy.deepcopy(room_si)
    # Bound the prebuild cost: keep only the table + floors + background within ENV_BG_RADIUS_M
    # (a single room-instance can still be a 1269-object conference hall).
    _proximity_filter_room(merged, room_table_name, ENV_BG_RADIUS_M)
    merged["init_info"] = {"class_name": "InteractiveTraversableScene",
                           "args": {"scene_model": scene_model}}
    m_init = merged["objects_info"]["init_info"]
    m_reg = _scene_registry(merged)

    b_init = base_si["objects_info"]["init_info"]
    b_reg = _scene_registry(base_si)

    injected: list[str] = []
    collisions: list[str] = []
    for name, info in b_init.items():
        if name == base_surf_name:
            continue  # drop the base table; the room's same-model table anchors T
        if name in m_init:
            collisions.append(name)
        st = copy.deepcopy(b_reg.get(name, {}))
        rl = st.get("root_link", {})
        pos, ori = rl.get("pos"), rl.get("ori")
        if pos is not None and ori is not None:
            npos, nori = eg.apply_transform_to_pose(T, pos, ori)
            rl["pos"], rl["ori"] = npos, nori
            st["root_link"] = rl
        m_init[name] = copy.deepcopy(info)
        m_reg[name] = st
        injected.append(name)
    if collisions:
        raise RuntimeError(f"injected names collide with room objects: {collisions}")

    # Drop the room's on-surface obstacles within the anchor footprint so nothing spawns
    # interpenetrating (the first-step ejection that topples fragiles / NaN-crashes the can).
    removed_spawn = _clear_room_spawn_obstacles(merged, injected, room_table_name,
                                                anchor_surf_top, anchor_surf_box)

    # Carry any base particle systems (dusty's dust) — none for lid/cabinet/jar/stack/clutter.
    b_sys = (base_si.get("state", {}).get("registry", {}) or {}).get("system_registry", {})
    if b_sys:
        m_sys = merged.setdefault("state", {}).setdefault("registry", {}).setdefault("system_registry", {})
        m_sys.update(copy.deepcopy(b_sys))
    return merged, injected, room_instance, removed_spawn


def _aabb_top_z(obj) -> float:
    _lo, hi = obj.aabb
    z = hi[2]
    return float(z.item() if hasattr(z, "item") else z)


_WALL_KEYWORDS = ("wall", "ceiling", "roof")


def remove_walls_and_ceiling(env) -> list[str]:
    """Physically remove wall/ceiling/roof geometry (keep floors/doors/windows).

    Unlike ``hide_walls_and_ceiling`` (visibility-only), this DELETES the prims, so a
    physical wall behind the injected layout can no longer collide with the fixed-base
    arm (the env-injection robot pose is inherited from base and may face a room wall).
    Visually identical to hiding (the wall isn't seen either way), and the removal bakes
    into the saved scene — no reload replay needed. Floors stay so the ground is visible.
    """
    import omnigibson as og

    to_remove = [o for o in env.scene.objects
                 if any(t in str(getattr(o, "category", "") or "").lower() for t in _WALL_KEYWORDS)]
    if to_remove:
        og.sim.batch_remove_objects(to_remove)
    return [getattr(o, "name", "") for o in to_remove]


def clear_on_surface_obstacles(env, surf, support_top: float, keep_names: set) -> list[str]:
    """Remove ONLY the room objects sitting ON the table surface within its footprint — i.e. real
    obstacles in the manipulation region. Under-table / beside / behind furniture (the cabinets,
    shelves, display cases that give the room its character) is KEPT as background: that IS the env
    OOD signal. This replaces the base-gen ``clear_support_area`` (a whole-table + 0.6 m ring sweep),
    which strips every nearby furniture and leaves an almost-empty scene for tables that sit amid
    built-in furniture (a cafe bar's cabinets, an office desk's shelving)."""
    from maniguard.task_generation.pipeline_common import is_structural_object
    import omnigibson as og

    slo, shi = surf.aabb
    fx0, fy0, fx1, fy1 = float(slo[0]), float(slo[1]), float(shi[0]), float(shi[1])
    to_remove = []
    for obj in env.scene.objects:
        if getattr(obj, "name", "") in keep_names or is_structural_object(obj):
            continue  # keep the table, injected objects, robot, floors
        try:
            olo, ohi = obj.aabb
        except Exception:
            continue
        on_surface = float(olo[2]) >= support_top - 0.05   # bottom at/above the table top
        xy_overlap = not (float(ohi[0]) < fx0 or float(olo[0]) > fx1
                          or float(ohi[1]) < fy0 or float(olo[1]) > fy1)
        if on_surface and xy_overlap:
            to_remove.append(obj)
    if to_remove:
        og.sim.batch_remove_objects(to_remove)
    return [getattr(o, "name", "") for o in to_remove]


# ---------------------------------------------------------------------------- worker

def _run_worker(base_dir: Path, out_dir: Path, family: str, episode: int,
                sub_room: str | None = None, sub_table: str | None = None) -> None:
    task = base_dir.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        row = _make_env_variant(base_dir, out_dir, family, episode,
                                sub_room=sub_room, sub_table=sub_table)
    except Exception as e:  # noqa: BLE001
        import traceback
        row = {"task": task, "family": family, "status": "fail",
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    (out_dir / ROW_FILE).write_text(json.dumps(row, default=float), encoding="utf-8")


def _make_env_variant(base_dir: Path, out_dir: Path, family: str, episode: int,
                      sub_room: str | None = None, sub_table: str | None = None) -> dict:
    """ONE injection attempt (one Isaac build — OmniGibson does not support rebuilding the env in the
    same process; the external cameras go stale on the 2nd build). NORMAL mode (``sub_room`` None)
    anchors on the SAME table model in the base task's own room (``T = T_room·T_base⁻¹``). SUBSTITUTE
    mode re-anchors the base pack onto a different, adequately-sized table (``sub_room``/``sub_table``).
    The driver orchestrates the normal→substitute fallback by spawning fresh worker subprocesses."""
    import omnigibson as og
    import torch as th

    from maniguard.data.bench_builder.finalize_base import (
        _build_active_objects, _compute_gate, _fresh_surface_info, _patch_lid_ltl,
    )
    from maniguard.data.bench_builder.perturbation import derive_seed
    from maniguard.data.bench_builder.render import render_views
    from maniguard.envs.frozen_task_runtime import build_env_config
    from maniguard.task_generation.pipeline_common import (
        _yaw_from_quat, clear_robot_base_region, robot_half_extent_xy,
    )
    from maniguard.utils.camera_setup import EXTERNAL_CAMERA_NAMES
    from maniguard.utils.robot_pose import (
        BENCH_CONTROLLER_PRESET, BENCH_GRASPING_MODE, BENCH_INIT_QPOS, ROBOT_MOUNT_OFFSET,
    )
    from maniguard.utils.safety_monitor import TaskLTLMonitor

    task = base_dir.parent.name
    diag = _load_diag(base_dir, episode)
    base_si = json.loads((base_dir / f"scene_ep{episode}.json").read_text(encoding="utf-8"))
    model = (diag.get("surface_info") or {}).get("model")
    base_surf_name = _base_surface_name(base_si, diag)
    if base_surf_name is None:
        raise RuntimeError(f"cannot resolve base surface object (model={model})")
    bp = _scene_registry(base_si)[base_surf_name]["root_link"]
    base_table_pose = (bp["pos"], bp["ori"])
    base_top = (diag.get("surface_info") or {}).get("top_z")

    # Fluid/particle tasks only simulate under the GPU pipeline (the CPU pipeline NaN-segfaults).
    if (bool((diag.get("selection") or {}).get("system_name"))
            or bool((base_si.get("state", {}).get("registry", {}) or {}).get("system_registry"))):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    scenes_root = eg._default_scenes_root()

    # --- resolve the anchor table + transform ---
    if sub_room is None:
        index = eg.load_table_scene_db()
        sel = eg.select_room_instance(index.get(model, []), scene_model=diag.get("scene_model"),
                                      base_table_pose=base_table_pose,
                                      seed=derive_seed(eg.BENCH_ENV_SEED, family, task))
        if sel is None:
            return {"task": task, "family": family, "status": "skip", "room": None,
                    "reason": f"no room contains table model {model}"}
        room, anchor_name = sel["scene"], sel["name"]
        anchor_pos, anchor_ori, anchor_model = sel["pos"], sel["ori"], model
        T = eg.compute_injection_transform(base_table_pose, (anchor_pos, anchor_ori))
        mode = "normal"
    else:
        room_si0 = json.loads(eg._best_json(scenes_root, sub_room).read_text(encoding="utf-8"))
        tinfo = room_si0.get("objects_info", {}).get("init_info", {}).get(sub_table, {})
        rl0 = (_scene_registry(room_si0).get(sub_table, {}) or {}).get("root_link", {})
        anchor_pos, anchor_ori = rl0.get("pos"), rl0.get("ori")
        anchor_model = tinfo.get("args", {}).get("model")
        geom = eg.collect_model_surface_geom(base_dir.parents[2]).get(anchor_model)
        pack = eg.compute_pack_footprint(base_si, diag)
        if anchor_pos is None or geom is None or pack is None or base_top is None:
            raise RuntimeError(f"substitute spec unavailable (room={sub_room} table={sub_table})")
        T = eg.substitute_transform(pack, base_top, anchor_pos, anchor_ori, geom)
        room, anchor_name, mode = sub_room, sub_table, "substitute"

    # --- single injection: build merged ITS scene, mount, declutter, settle, verdict, render ---
    # Anchor-surface footprint + world top, used to drop the room's on-surface obstacles BEFORE the
    # scene is built (so injected objects spawn onto a clean surface). Top = base table top mapped
    # through T's z-shift; box = the anchor table's surface footprint in world axes.
    anchor_geom = (eg.collect_model_surface_geom(base_dir.parents[2]).get(anchor_model)
                   if sub_room is None else geom)
    anchor_surf_top = (float(base_top) + float(T[2, 3])) if base_top is not None else None
    anchor_surf_box = (eg.surface_world_box(anchor_pos, anchor_ori, anchor_geom, pad=SPAWN_CLEAR_EDGE_PAD_M)
                       if (anchor_geom and anchor_pos is not None and anchor_ori is not None) else None)
    room_si = json.loads(eg._best_json(scenes_root, room).read_text(encoding="utf-8"))
    merged, injected, room_instance, removed_spawn = build_merged_scene_info(
        base_si, room_si, base_surf_name, anchor_name, room, T,
        anchor_surf_top=anchor_surf_top, anchor_surf_box=anchor_surf_box)
    out_diag = copy.deepcopy(diag)
    out_diag["scene_model"] = room
    out_diag["surface"] = anchor_name

    og_cfg = build_env_config(
        merged, out_diag, camera_names=EXTERNAL_CAMERA_NAMES,
        external_camera_kwargs={"resolution": 256},
        controller_preset=BENCH_CONTROLLER_PRESET, grasping_mode=BENCH_GRASPING_MODE)
    if og.sim is not None:
        og.sim.stop()
        og.clear()
    env = og.Environment(configs=og_cfg)
    env.reset()
    robot = env.robots[0]

    surf = env.scene.object_registry("name", anchor_name)
    if surf is None:
        raise RuntimeError(f"anchor table {anchor_name!r} not found after load in {room}")
    support_top = _aabb_top_z(surf)
    rp, rq = robot.get_position_orientation()
    rp = rp.clone()
    rp[2] = support_top + ROBOT_MOUNT_OFFSET
    robot.set_position_orientation(position=rp, orientation=rq)
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()
    og.sim.step()

    # Stage-1 declutter: walls/ceiling + on-table obstacles + robot-keepout furniture; keep the rest.
    spawned = {n: o for n in injected if (o := env.scene.object_registry("name", n)) is not None}
    removed_walls = remove_walls_and_ceiling(env)
    keep_names = set(spawned) | {anchor_name, getattr(robot, "name", "agent_0")}
    removed_area = clear_on_surface_obstacles(env, surf, support_top, keep_names)
    rpos, rori = robot.get_position_orientation()
    removed_base = clear_robot_base_region(
        env, surf, (float(rpos[0]), float(rpos[1])), robot_half_extent_xy(robot),
        margin_m=0.05, base_yaw=_yaw_from_quat(rori), spawned_objects=spawned)
    robot.keep_still()
    og.sim.step()

    injected_objs = [o for o in (env.scene.object_registry("name", n) for n in injected)
                     if o is not None and not str(getattr(o, "name", "")).startswith("agent")]
    base_z = float(robot.get_position_orientation()[0][2])
    # Spawn (pre-settle) height of each injected object — it has landed on its resting contact (the
    # spawn-collision obstacles are gone, so this is the clean base rest pose mapped through T). The
    # sink verdict is measured RELATIVE to this, not to the surface AABB top: a counter with a raised
    # back panel (checkout_counter) has an AABB top tens of cm above its real placement surface, so a
    # surface-AABB sink check false-flags an object that rests perfectly on the actual counter top.
    spawn_z = {o.name: float(o.get_position_orientation()[0][2]) for o in injected_objs}

    # settle (physics only; verdict is on the FINAL scene)
    for i in range(SETTLE_STEPS):
        robot.keep_still()
        if i < SETTLE_HOLD:
            for o in injected_objs:
                o.keep_still()
        og.sim.step()

    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()
    env.scene.save(json_path=str(out_dir / f"scene_ep{episode}.json"))

    fallen = [o.name for o in injected_objs
              if float(o.get_position_orientation()[0][2]) < support_top - 0.3]
    # interpenetration / sink: an object that DROPPED more than SINK_TOL below its spawn rest height
    # has settled INTO the surface (the fallen-0.3 check + the LTL "on support" predicate both miss a
    # shallow sink). Measured spawn-relative, not surface-AABB-relative — see ``spawn_z`` above.
    sunk = [o.name for o in injected_objs
            if float(o.get_position_orientation()[0][2]) < spawn_z.get(o.name, float("-inf")) - SINK_TOL]
    spawn_specs = (diag.get("selection") or {}).get("spawn_specs") or []
    ltl_safety = _patch_lid_ltl(family, diag.get("ltl_safety") or {}, spawn_specs, anchor_name)
    monitor = None
    init_doomed = False
    if ltl_safety:
        active = _build_active_objects(env, ltl_safety, anchor_name, objects=injected_objs + [surf])
        monitor = TaskLTLMonitor(env, ltl_safety=ltl_safety, activity_name=diag.get("activity_name", ""),
                                 scene_model=None, active_objects_by_inst=active)
        monitor.reset()
        init_doomed = bool(monitor.step(0).get("doomed", False))
    gate_pass, gate_detail = _compute_gate(env, robot, diag, support_top, base_z,
                                           ROBOT_MOUNT_OFFSET, init_doomed)
    out_diag, stats = render_views(env, out_diag, out_dir, episode=episode, ltl_monitor=monitor)
    ltl_violated = bool(monitor.violated) if monitor is not None else False
    arm_ok = float(stats.get("arm_drift") or 0.0) < ARM_DRIFT_TOL
    feasible = bool(gate_pass and not ltl_violated and not fallen and not sunk and arm_ok)

    out_diag["surface_info"] = _fresh_surface_info(surf, support_top, merged["objects_info"]["init_info"])
    gr = out_diag.get("goal_region")
    if isinstance(gr, dict) and gr.get("center_world"):
        gr["center_world"] = eg.apply_transform_to_point(T, gr["center_world"])
    out_diag["gate_pass"] = bool(gate_pass)
    out_diag["ltl_violated"] = bool(ltl_violated)
    out_diag["steps_executed"] = int(stats.get("steps_executed", 0))
    if monitor is not None:
        s = monitor.summary()
        out_diag["ltl_summary"] = {k: s[k] for k in
                                   ("formula", "violated", "violation_step",
                                    "violation_count", "total_steps_monitored") if k in s}
    bench = out_diag.setdefault("bench", {})
    bench.update({"support_top": round(support_top, 4), "base_z": round(base_z, 4),
                  "gate": gate_detail, "arm_drift": stats.get("arm_drift"),
                  "obj_disp": stats.get("obj_disp"), "env_injected": True})
    out_diag["perturbation"] = {
        "kind": "env", "mode": mode, "room": room, "room_instance": room_instance,
        "table": {"model": anchor_model, "name": anchor_name, "pos": anchor_pos, "ori": anchor_ori},
        "base_table_model": model,
        "transform": {"translation": [float(x) for x in T[:3, 3]]},
        "scene_model_origin": diag.get("scene_model"),
        "injected": injected,
        "removed_walls": removed_walls, "removed_support_area": removed_area,
        "removed_robot_base": removed_base, "removed_spawn_obstacles": removed_spawn,
    }
    (out_dir / "diagnostics.jsonl").write_text(json.dumps(out_diag, default=float), encoding="utf-8")
    return {
        "task": task, "family": family, "status": "ok" if feasible else "infeasible",
        "feasible": feasible, "mode": mode, "room": room, "table": anchor_name,
        "table_model": anchor_model, "n_injected": len(injected),
        "gate_pass": bool(gate_pass), "ltl_violated": bool(ltl_violated), "fallen": fallen,
        "sunk": sunk, "arm_drift": stats.get("arm_drift"), "obj_disp": stats.get("obj_disp"),
    }


def _last_frame(video_path: Path):
    import av
    last = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            last = frame
    return None if last is None else last.to_ndarray(format="rgb24")


def _write_compare_snapshot(base_dir: Path, env_dir: Path, out_png: Path, episode: int) -> bool:
    """2x2 QC for one env task: each ROW is ``base | env`` for one view — the opposite
    (front) view AND the left_overview (≈ the eval external_cam, which faces the room
    furniture). A single opposite view can point at a bright window and read as empty even
    when the room is richly loaded, so the left view is shown alongside it."""
    import numpy as np
    import imageio.v2 as imageio

    def row(view):
        bv = base_dir / f"rollout_{view}_ep{episode}.mp4"
        ev = env_dir / f"rollout_{view}_ep{episode}.mp4"
        if not bv.exists() or not ev.exists():
            return None
        bf, ef = _last_frame(bv), _last_frame(ev)
        if bf is None or ef is None:
            return None
        h = max(bf.shape[0], ef.shape[0])
        pad = lambda im: im if im.shape[0] == h else np.vstack(
            [im, np.zeros((h - im.shape[0], im.shape[1], 3), im.dtype)])
        return np.hstack([pad(bf), np.full((h, 4, 3), 255, bf.dtype), pad(ef)])

    rows = [r for r in (row("opposite_side_front"), row("left_overview")) if r is not None]
    if not rows:
        return False
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), r.dtype)])
            for r in rows]
    sep = np.full((4, w, 3), 200, dtype=rows[0].dtype)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stacked = [x for r in rows for x in (r, sep)][:-1]
    imageio.imwrite(out_png, np.vstack(stacked))
    return True


# ---------------------------------------------------------------------------- driver

def _spawn_worker(base_dir: Path, out_dir: Path, family: str, episode: int, env: dict, timeout: int,
                  sub_room: str | None = None, sub_table: str | None = None):
    cmd = [
        sys.executable, "-m", "maniguard.data.bench_builder.perturb_env",
        "--worker", "--base-dir", str(base_dir), "--out-dir", str(out_dir),
        "--family", family, "--episode", str(episode),
    ]
    if sub_room and sub_table:
        cmd += ["--substitute-room", sub_room, "--substitute-table", sub_table]
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return "timeout"


def _read_row(out_dir: Path, task: str, family: str, episode: int) -> dict:
    row_path = out_dir / ROW_FILE
    if not row_path.exists():
        return {"task": task, "family": family, "status": "fail",
                "error": "worker produced no row (crashed before writing)"}
    row = json.loads(row_path.read_text(encoding="utf-8"))
    if row.get("status") in ("fail", "skip"):
        return row
    if not _is_complete(out_dir, episode):
        return {"task": task, "family": family, "status": "fail",
                "error": "incomplete output (missing snapshot/videos)", "worker": row}
    return row


def _row_for_task(task: str, out_fam: Path, family: str, episode: int) -> dict:
    return _read_row(out_fam / task / LEVEL, task, family, episode)


def _reusable(out_dir: Path, episode: int) -> bool:
    """Reuse (skip re-running) ONLY a previously-feasible NORMAL-mode task. Substitute-mode tasks are
    re-verified (the sink check was added after they were produced), and infeasible/needs_review/fail
    are always re-run. Normal-mode injection preserves the base resting geometry exactly, so those
    are trusted (spot-checked separately)."""
    rp = out_dir / ROW_FILE
    if not rp.exists() or not _is_complete(out_dir, episode):
        return False
    try:
        r = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Reuse a feasible task UNLESS it was produced by substitution (mode == "substitute"), which the
    # sink check post-dates. Older rows predating the mode field have no "mode" → they are normal-mode
    # (substitution did not exist then), so they reuse.
    return r.get("status") == "ok" and r.get("mode") != "substitute"


def _process_task(out_fam: Path, task: str, family: str, episode: int, env: dict, timeout: int,
                  geom: dict, index: dict, max_substitute: int, launch_gap: float = 0.0) -> dict:
    """Orchestrate a task: try the NORMAL same-model injection (one worker subprocess); if that is
    infeasible/skip and the task is NOT liquid, SUBSTITUTE adequately-sized tables (one subprocess
    each) until one is feasible, else needs_review. One build per subprocess — OmniGibson can't
    rebuild the env in-process. Liquid tasks that fail normal are deferred to the settle-fix pass."""
    base_dir, out_dir = out_fam / task / "base", out_fam / task / LEVEL
    diag = _load_diag(base_dir, episode)
    is_liquid = bool((diag.get("selection") or {}).get("system_name"))

    # Optional inter-launch gap: subprocess.run already blocks until the prior worker process is
    # fully dead (so a single-job run never overlaps GPU teardown with the next build), but a small
    # gap gives the driver headroom when running two concurrent sessions on one GPU.
    if launch_gap:
        time.sleep(launch_gap)
    _spawn_worker(base_dir, out_dir, family, episode, env, timeout)
    row = _read_row(out_dir, task, family, episode)
    if row.get("status") == "ok":
        return row
    # Defer to the settle-fix pass ONLY for the liquid-settle signature (a fragile/container tipped:
    # ltl_violated, nothing fallen, arm undisturbed). A liquid task that fails for a PLACEMENT reason
    # (arm collision, object fell, load NaN) can still be fixed by a different table → substitute it.
    settle_sig = (is_liquid and row.get("status") == "infeasible" and row.get("ltl_violated")
                  and not row.get("fallen") and float(row.get("arm_drift") or 0.0) < ARM_DRIFT_TOL)
    if settle_sig:
        return {**row, "note": "liquid settle: deferred to settle-fix pass"}

    base_si = json.loads((base_dir / f"scene_ep{episode}.json").read_text(encoding="utf-8"))
    pack = eg.compute_pack_footprint(base_si, diag)
    if pack is None:
        return {**row, "status": "needs_review", "reason": "pack footprint unavailable"}
    excl = {row.get("room")} if row.get("room") else set()
    cands = eg.find_substitute_candidates(pack, geom, index, exclude_rooms=excl)
    orig_model = (diag.get("surface_info") or {}).get("model")
    for ci, cand in enumerate(cands[:max_substitute], 1):
        _spawn_worker(base_dir, out_dir, family, episode, env, timeout,
                      sub_room=cand["scene"], sub_table=cand["name"])
        srow = _read_row(out_dir, task, family, episode)
        if srow.get("status") == "ok":
            return {**srow, "substituted_for": orig_model, "n_substitute_tried": ci}
    last = _read_row(out_dir, task, family, episode)
    return {**last, "status": "needs_review",
            "n_substitute_tried": min(len(cands), max_substitute), "substitute_for": orig_model}


def _driver(args: argparse.Namespace) -> int:
    out_fam = Path(args.bench_root) / args.family
    if not out_fam.is_dir():
        print(f"[env] ERROR: family dir not found: {out_fam}", flush=True)
        return 2
    tasks = _select_tasks(out_fam, args.tasks, args.episode)
    if not tasks:
        print(f"[env] no matching base tasks in {out_fam} (--tasks {args.tasks!r})", flush=True)
        return 1
    env = _worker_env()
    snap_dir = out_fam / "snapshots_env"
    manifest = out_fam / "env_manifest.jsonl"
    index = eg.load_table_scene_db()
    geom = eg.collect_model_surface_geom(Path(args.bench_root))  # model -> surface dims (for substitution)

    # Partition: reuse already-SUCCESSFUL env dirs (--skip-existing); run everything else. No more
    # offline pre-skip — the worker now SUBSTITUTES a different adequately-sized table when the base
    # task's own room is missing/infeasible, so a no-room task is no longer a guaranteed skip.
    reuse, to_run = [], []
    for t in tasks:
        out_dir = out_fam / t / LEVEL
        if args.skip_existing and _reusable(out_dir, args.episode):
            reuse.append(t)
        else:
            to_run.append(t)
    print(f"[env] {args.family}: {len(tasks)} tasks | reuse={len(reuse)} run={len(to_run)} "
          f"jobs={args.jobs} -> {out_fam}/*/env  manifest={manifest.name}", flush=True)

    # Manifest is rewritten in full (w) covering the WHOLE family — run the family WITHOUT --tasks to
    # keep it complete (--tasks subsets would only list those tasks).
    rows: list[dict] = []
    mf = manifest.open("w", encoding="utf-8")

    def _record(task: str, tag: str, i: int, n: int, row: dict | None = None) -> None:
        if row is None:
            row = _row_for_task(task, out_fam, args.family, args.episode)
        rows.append(row)
        mf.write(json.dumps(row, default=float) + "\n")
        mf.flush()
        st = row.get("status")
        if st in ("ok", "infeasible", "needs_review"):
            ok = _write_compare_snapshot(out_fam / task / "base", out_fam / task / LEVEL,
                                         snap_dir / f"{task}.png", args.episode)
            extra = (f"  mode={row.get('mode')} room={row.get('room')} table_model={row.get('table_model')} "
                     f"gate={row.get('gate_pass')} ltl_v={row.get('ltl_violated')} fallen={row.get('fallen')} "
                     f"drift={row.get('arm_drift')} sub_tried={row.get('n_substitute_tried')} "
                     f"snap={'ok' if ok else 'MISS'}")
            if row.get("substituted_for"):
                extra += f"  SUBSTITUTED(orig_model={row.get('substituted_for')})"
            if row.get("note"):
                extra += f"  [{row.get('note')}]"
        elif st == "skip":
            extra = f"  SKIP: {row.get('reason')}"
        else:
            extra = f"  FAIL: {row.get('error')}"
        print(f"[{tag} {i}/{n}] {task}: {st}{extra}", flush=True)

    for i, t in enumerate(reuse, 1):
        _record(t, "reuse", i, len(reuse))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_process_task, out_fam, t, args.family, args.episode, env, args.timeout,
                          geom, index, args.max_substitute, args.launch_gap): t for t in to_run}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            row = fut.result()
            done += 1
            _record(t, "run", done, len(to_run), row=row)

    mf.close()
    counts = Counter(r["status"] for r in rows)
    review = [r["task"] for r in rows if r.get("status") in ("needs_review", "infeasible")]
    fails = [r["task"] for r in rows if r.get("status") == "fail"]
    subbed = [r["task"] for r in rows if r.get("substituted_for")]
    print(f"=== {args.family} env: {dict(counts)} ({len(rows)} tasks) -> {manifest}", flush=True)
    if subbed:
        print(f"    SUBSTITUTED table: {subbed}", flush=True)
    if review:
        print(f"    NEEDS_REVIEW: {review}", flush=True)
    if fails:
        print(f"    FAILED: {fails}", flush=True)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the env (surrounding-room) perturbation level.")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--base-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--family", required=True)
    ap.add_argument("--bench-root", default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-substitute", type=int, default=8,
                    help="max substitute tables to try when the base task's own room is infeasible")
    ap.add_argument("--launch-gap", type=float, default=0.0,
                    help="seconds to wait before each worker launch (headroom for two concurrent "
                         "single-job sessions sharing one GPU)")
    ap.add_argument("--substitute-room", default=None, help="(worker) re-anchor onto this room")
    ap.add_argument("--substitute-table", default=None, help="(worker) re-anchor onto this table object")
    args = ap.parse_args()

    if args.worker:
        if not args.base_dir or not args.out_dir:
            ap.error("--worker requires --base-dir and --out-dir")
        _run_worker(Path(args.base_dir), Path(args.out_dir), args.family, args.episode,
                    sub_room=args.substitute_room, sub_table=args.substitute_table)
        return 0
    return _driver(args)


if __name__ == "__main__":
    sys.exit(main())
