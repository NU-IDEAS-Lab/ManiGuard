"""Finalize one generated task for ManiGuard-Bench.

Load a source snapshot and diagnostics, replace its robot with the benchmark
configuration, set the mount height and initial joint pose, and save the result
as scene_ep{episode}.json in a separate output directory.

Task identity, prompts, goals, safety specifications, and family metadata are
selected explicitly from the source diagnostics. Camera poses, surface geometry,
spawn-feasibility checks, and idle-rollout safety results are recomputed. The
output diagnostics also record the robot configuration and stability statistics.
Four review videos show the idle rollout.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from maniguard.data.bench_builder.render import DEFAULT_FPS, DEFAULT_N_FRAMES, DEFAULT_RESOLUTION
from maniguard.data.bench_builder.validate_base import _category_lemma  # shared bddl taxonomy bridge

ARM_DRIFT_TOL = 1e-2     # rad
BASE_Z_TOL = 1e-3        # m
OBJ_DISP_WARN = 0.02     # m
REACH_MIN, REACH_MAX = 0.20, 1.10   # target reachability band (mirrors the generation gate)

# --- diagnostics fields -------------------------------------------------------------------------
# Task-definition fields copied from the source diagnostics.
_CARRY_UNIVERSAL = ["episode", "activity_name", "scene_model", "surface", "prompt",
                    "selection", "ltl_safety", "goal_conditions", "goal_region", "pipeline"]
_CARRY_FAMILY = {
    "jar_transport": ["jar_info", "item_info"],
    "cabinet_pickup": ["cabinet_info", "obstacle_info", "target_info", "blocker_mode"],
    "stack_retrieve": ["stack_mode", "stack_height", "ontop_valid"],
    "dusty_transfer": ["dust_system", "dusted_dest", "sponge_category", "sponge_model",
                       "sponge_name", "source_food_transfer_task"],
    "clutter_pickup": [],
    "lid_transport": [],
}
# Fields omitted from direct copying: camera, surface, and rollout measurements
# are recomputed; recording paths and robot-base metadata are not carried over.
# Lid selection metadata is retained in selection and summarized in lid_info.
# Other unrecognized source fields are reported in the output diagnostics.
_DROP = {"cameras", "surface_info", "gate_pass", "ltl_violated", "steps_executed", "ltl_summary",
         "snapshots", "videos", "robot_base",
         "container_category", "container_model", "food_category", "food_model",
         "item_category", "item_model", "n_objects_active", "n_objects_requested", "system_name"}


def _load_diagnostics(base_dir: Path) -> dict:
    raw = (base_dir / "diagnostics.jsonl").read_text(encoding="utf-8").lstrip()
    return json.JSONDecoder().raw_decode(raw)[0]


def _source_scene_file(src_base_dir: Path, episode: int) -> Path:
    for name in (f"scene_ep{episode}.json", f"scene_ep{episode}_replay.json"):
        p = src_base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no scene_ep{episode}[_replay].json in {src_base_dir}")


def _resolve_surface(env, diag: dict):
    """The live support-surface object. Handles the object-name surface string (clutter/lid/stack/
    dusty, e.g. ``desk_qpuflh_2``), the ``category/model`` string (jar/cabinet, e.g. ``desk/obvwds``)
    via the goal_region's support_name or a category match."""
    reg = env.scene.object_registry
    surface = diag.get("surface") or ""
    for cand in (surface, (diag.get("goal_region") or {}).get("support_name")):
        if cand:
            o = reg("name", cand)
            if o is not None:
                return o
    if "/" in surface:
        cat = surface.split("/")[0]
        return next((o for o in env.scene.objects if getattr(o, "category", "") == cat), None)
    return None


def _aabb_lo_hi(obj):
    import numpy as np
    lo, hi = obj.aabb
    lo = np.asarray(lo.cpu().numpy() if hasattr(lo, "cpu") else lo, dtype=float)
    hi = np.asarray(hi.cpu().numpy() if hasattr(hi, "cpu") else hi, dtype=float)
    return lo, hi


# Round-top surfaces whose world AABB is a SQUARE but whose usable surface is the inscribed
# circle. The AABB proxy parks/jitters objects off the round rim (objects fall off the table);
# emit the inscribed usable square instead. Map model -> inscribed-square half-side (m), measured
# about the live AABB centre (so the same table at any world pose / scene gets the right region).
# Add a model here if a new round-top surface enters the bench.
_ROUND_SURFACE_HALF = {
    # coffee_table/semdkc: a 0.43 m half-side keeps square corners inside the
    # approximately 0.665 m radius of the circular tabletop.
    "semdkc": 0.43,
}


def _surface_bounds(lo, hi, model):
    """Return (bounds_xy, area_m2, frame) for a support surface. Default = the world AABB. For a
    known round-top surface (``model`` in ``_ROUND_SURFACE_HALF``) return the inscribed usable
    square centred on the AABB instead, so downstream placement never lands an object off the rim."""
    half = _ROUND_SURFACE_HALF.get(model)
    if half is not None:
        cx = 0.5 * (float(lo[0]) + float(hi[0]))
        cy = 0.5 * (float(lo[1]) + float(hi[1]))
        bounds_xy = [[round(cx - half, 4), round(cy - half, 4)],
                     [round(cx + half, 4), round(cy + half, 4)]]
        return bounds_xy, round((2 * half) ** 2, 4), "world_usable_rect"
    bounds_xy = [[round(float(lo[0]), 4), round(float(lo[1]), 4)],
                 [round(float(hi[0]), 4), round(float(hi[1]), 4)]]
    return bounds_xy, round(float((hi[0] - lo[0]) * (hi[1] - lo[1])), 4), "world_aabb"


def _fresh_surface_info(surf, support_top: float, init: dict) -> dict:
    """Uniform surface metadata, freshly computed from the resolved surface object's world AABB
    (inscribed usable rect for known round-top surfaces — see ``_surface_bounds``)."""
    lo, hi = _aabb_lo_hi(surf)
    args = (init.get(getattr(surf, "name", ""), {}) or {}).get("args", {})
    model = args.get("model")
    bounds_xy, area_m2, frame = _surface_bounds(lo, hi, model)
    return {
        "category": getattr(surf, "category", None) or args.get("category"),
        "model": model,
        "top_z": round(float(support_top), 4),
        "bounds_xy": bounds_xy,
        "height_m": round(float(hi[2] - lo[2]), 4),
        "area_m2": area_m2,
        "frame": frame,
    }


def _derive_family_info(family: str, diag: dict, n_task_objects: int | None = None) -> dict:
    """Build clutter or lid metadata from selection and spawn_specs.

    If n_task_objects is provided, use the actual scene inventory for the
    clutter count. Generation may place fewer objects than spawn_specs requests.
    Other families retain their existing metadata and return an empty mapping.
    """
    sel = diag.get("selection") or {}
    specs = sel.get("spawn_specs") or []
    if family == "clutter_pickup":
        target = next((s for s in specs if s.get("role") == "target"), None)
        if target is None:
            return {}
        if n_task_objects is not None:
            n_clutter = max(0, n_task_objects - 1)  # actual task objects minus the single target
        else:
            n_clutter = sum(int(s.get("count", 1)) for s in specs if s.get("role") != "target")
        return {"clutter_info": {
            "target": {
                "name": (diag.get("goal_region") or {}).get("target_name"),
                "category": target.get("category"),
                "model": target.get("model"),
            },
            "n_clutter_objects": n_clutter,
        }}
    if family == "lid_transport":
        return {"lid_info": {
            "container": {"category": sel.get("container_category"), "model": sel.get("container_model")},
            "lid": {"model": sel.get("lid_model")},
            "food": {"synset": sel.get("food_synset")},
            "mode": sel.get("lid_mode"),
        }}
    return {}


def _needs_gpu_dynamics(diag: dict) -> bool:
    """Return whether selection.system_name requests a physical particle system.

    These tasks require PhysX GPU dynamics, which must be enabled before
    constructing the environment. Tasks without this field use the default
    dynamics configuration.
    """
    return bool((diag.get("selection") or {}).get("system_name"))


def _patch_lid_ltl(family: str, ltl_safety: dict, spawn_specs: list, surface_name: str | None = None) -> dict:
    """Normalize lid-task proposition patterns to the spawned objects.

    Replace lid.n.02_* with the lid role's declared synset when needed, and
    replace breakfast_table.n.01_* with the support object's category pattern.
    Apply this before constructing the monitor and save the resulting specification
    with the task. Return the original mapping when no replacement is needed.
    """
    if family != "lid_transport" or not ltl_safety:
        return ltl_safety
    lid_spec = next((s for s in spawn_specs if s.get("role") == "lid"), None)
    lid_syn = (lid_spec or {}).get("synset")
    fix_lid = bool(lid_syn) and lid_syn != "lid.n.02"
    # surface object name is "<category>_<model>_<index>" (model = 6-char hash, index = digit)
    surf_cat = surface_name.rsplit("_", 2)[0] if surface_name and surface_name.count("_") >= 2 else None
    import copy
    patched = copy.deepcopy(ltl_safety)
    changed = False
    for pdef in (patched.get("propositions") or {}).values():
        over = pdef.get("over")
        if fix_lid and isinstance(over, list) and "lid.n.02_*" in over:
            pdef["over"] = [f"{lid_syn}_*" if o == "lid.n.02_*" else o for o in over]
            changed = True
        rel = pdef.get("relative_to")
        if surf_cat and isinstance(rel, list) and "breakfast_table.n.01_*" in rel:
            pdef["relative_to"] = [f"{surf_cat}_*" if r == "breakfast_table.n.01_*" else r for r in rel]
            changed = True
    return patched if changed else ltl_safety


def _build_active_objects(env, ltl_safety: dict, surface_name: str | None, objects=None) -> dict:
    """Resolve proposition patterns to live objects for safety monitoring.

    Match categories, taxonomy lemmas, and object-name patterns, with a support
    fallback for unresolved synset patterns. The optional objects argument
    limits candidate objects. Environment variants pass injected task objects
    and the anchor surface to avoid binding unrelated room furniture.
    """
    patterns: set[str] = set()
    for pdef in ((ltl_safety or {}).get("propositions") or {}).values():
        for key in ("over", "relative_to"):
            v = pdef.get(key)
            if isinstance(v, list):
                patterns.update(v)
            elif isinstance(v, str):
                patterns.add(v)
    robot = env.robots[0] if env.robots else None
    objs = list(objects) if objects is not None else list(env.scene.objects)
    cat2lemma = {}
    for o in objs:
        c = getattr(o, "category", "")
        if c and c not in cat2lemma:
            cat2lemma[c] = _category_lemma(c)
    surface_obj = env.scene.object_registry("name", surface_name) if surface_name else None
    active: dict = {}
    for pat in patterns:
        prefix = pat.removesuffix("_*")
        if prefix.startswith("agent"):
            if robot is not None:
                active[f"{prefix}_0"] = robot
            continue
        base = prefix.split(".n.")[0]
        matched = [o for o in objs if getattr(o, "category", "") == base
                   or cat2lemma.get(getattr(o, "category", "")) == base]
        matched += [o for o in objs if o not in matched
                    and fnmatch.fnmatch(getattr(o, "name", ""), pat)]
        if not matched and ".n." in prefix:
            role = [o for o in objs if getattr(o, "name", "").startswith(base + "_")]
            matched = role if role else ([surface_obj] if surface_obj is not None else [])
        for i, obj in enumerate(matched):
            active[f"{prefix}_{i}"] = obj
    return active


def _compute_gate(env, robot, diag: dict, support_top: float, base_z: float,
                  mount_offset: float, init_doomed: bool) -> tuple[bool, dict]:
    """Fresh spawn-feasibility gate on the FINALIZED scene (mirrors the generation gate, adapted to
    the bench's surface mount): finite poses + mount sanity + target reachability + init LTL not
    already doomed. Returns ``(gate_pass, detail)``."""
    import math
    rp = [float(v) for v in robot.get_position_orientation()[0][:3]]
    finite = all(math.isfinite(v) for v in rp)
    mount_ok = abs(base_z - (support_top + mount_offset)) <= 1e-3
    # the task's target object: goal_region.target_name (jar/clutter/...) or target_info.name
    # (cabinet, whose goal is inside-cabinet+closed with no goal_region). None -> skip reachability.
    target_name = ((diag.get("goal_region") or {}).get("target_name")
                   or (diag.get("target_info") or {}).get("name"))
    tdist = None
    reach_ok = True
    if target_name:
        tobj = env.scene.object_registry("name", target_name)
        if tobj is not None:
            tp = [float(v) for v in tobj.get_position_orientation()[0][:3]]
            finite = finite and all(math.isfinite(v) for v in tp)
            tdist = math.hypot(rp[0] - tp[0], rp[1] - tp[1])
            reach_ok = REACH_MIN <= tdist <= REACH_MAX
    gate = bool(finite and mount_ok and reach_ok and not init_doomed)
    detail = {
        "finite": finite, "mount_ok": mount_ok,
        "target_dist": round(tdist, 4) if tdist is not None else None,
        "reach_ok": reach_ok, "init_ltl_doomed": bool(init_doomed),
    }
    return gate, detail


def _robot_from_snapshot(header: dict):
    init = header.get("objects_info", {}).get("init_info", {})
    reg = header.get("state", {}).get("registry", {}).get("object_registry", {})
    for name, info in init.items():
        cm = info.get("class_module", "")
        cn = info.get("class_name", "")
        if cm.startswith("omnigibson.robots.") or cn.endswith(("Robot", "Mounted", "Panda")):
            r = reg.get(name, {})
            pos = r.get("root_link", {}).get("pos")
            return r.get("joint_pos"), (float(pos[2]) if pos else None)
    return None, None


def finalize_base_task(
    src_base_dir,
    out_base_dir,
    *,
    family: str,
    episode: int = 1,
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    resolution: int = DEFAULT_RESOLUTION,
) -> dict:
    """Finalize one base task into ``out_base_dir``; never writes to ``src_base_dir``."""
    import omnigibson as og
    import torch as th

    from maniguard.data.bench_builder.render import render_views
    from maniguard.envs.frozen_task_runtime import build_env_config
    from maniguard.utils.camera_setup import EXTERNAL_CAMERA_NAMES
    from maniguard.utils.robot_pose import (
        BENCH_CONTROLLER_PRESET,
        BENCH_GRASPING_MODE,
        BENCH_INIT_QPOS,
        ROBOT_MOUNT_OFFSET,
    )
    from maniguard.utils.safety_monitor import TaskLTLMonitor

    src_base_dir = Path(src_base_dir)
    out_base_dir = Path(out_base_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)
    task_name = src_base_dir.parent.name

    diag = _load_diagnostics(src_base_dir)
    scene_file = _source_scene_file(src_base_dir, episode)
    scene_info = json.loads(scene_file.read_text(encoding="utf-8"))

    # --- GPU dynamics: fluid/particle tasks (clutter-liquid carry selection.system_name) only
    # simulate under the PhysX GPU pipeline; the default CPU pipeline NaN-segfaults on the first
    # physics step. Must be set BEFORE the env is built (cannot toggle mid-session). Dry tasks are
    # left on the default CPU pipeline so their behaviour is unchanged. Subprocess-per-task isolates
    # this per task. Mirrors the source pipeline's GPU-dynamics gating. ---
    needs_gpu = _needs_gpu_dynamics(diag)
    if needs_gpu:
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    # --- build env: scene from snapshot, source robot stripped + replaced by ONE canonical robot ---
    og_cfg = build_env_config(
        scene_info, diag,
        camera_names=EXTERNAL_CAMERA_NAMES,
        external_camera_kwargs={"resolution": resolution},
        controller_preset=BENCH_CONTROLLER_PRESET,
        grasping_mode=BENCH_GRASPING_MODE,
    )
    if og.sim is not None:
        og.sim.stop()
        og.clear()
    env = og.Environment(configs=og_cfg)
    env.reset()
    robot = env.robots[0]

    # --- enforce mount: base_z = (resolved surface AABB top) + offset, keep XY + yaw ---
    surf = _resolve_surface(env, diag)
    if surf is None:
        raise ValueError(f"cannot resolve support surface for {task_name} (surface={diag.get('surface')!r})")
    support_top = float(_aabb_lo_hi(surf)[1][2])
    # Raised panels can make the support AABB top exceed the placement plane.
    # Use the lower of that top and the target AABB bottom as the mount reference.
    _tname = ((diag.get("goal_region") or {}).get("target_name")
              or (diag.get("target_info") or {}).get("name"))
    if _tname:
        _tobj = env.scene.object_registry("name", _tname)
        if _tobj is not None:
            support_top = min(support_top, float(_aabb_lo_hi(_tobj)[0][2]))
    rp_t, rq_t = robot.get_position_orientation()
    rp = [float(v) for v in rp_t[:3]]
    base_z_before = rp[2]
    base_z_after = float(support_top + ROBOT_MOUNT_OFFSET)
    robot.set_position_orientation(position=[rp[0], rp[1], base_z_after], orientation=rq_t)

    # --- bake canonical init pose (held by the stiff Isaac drive through the idle-step) ---
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()

    # --- save the clean init snapshot to OUTPUT (canonical filename), BEFORE idle-stepping ---
    out_scene = out_base_dir / f"scene_ep{episode}.json"
    env.scene.save(json_path=str(out_scene))

    # --- build a FRESH LTL monitor on the finalized scene; step(0) feeds the gate's init-doomed ---
    # patch the lid family's hardcoded lid synset to the actual spawned object (no-op otherwise), so
    # the monitor AND the carried out_diag below both use the corrected spec.
    ltl_safety = _patch_lid_ltl(family, diag.get("ltl_safety") or {},
                                (diag.get("selection") or {}).get("spawn_specs") or [],
                                diag.get("surface"))
    monitor = None
    init_doomed = False
    if ltl_safety:
        active = _build_active_objects(env, ltl_safety, getattr(surf, "name", None))
        monitor = TaskLTLMonitor(
            env, ltl_safety=ltl_safety, activity_name=diag.get("activity_name", ""),
            scene_model=None, active_objects_by_inst=active,
        )
        monitor.reset()
        init_doomed = bool(monitor.step(0).get("doomed", False))

    # --- fresh spawn-feasibility gate on the finalized scene ---
    gate_pass, gate_detail = _compute_gate(env, robot, diag, support_top, base_z_after,
                                           ROBOT_MOUNT_OFFSET, init_doomed)

    # --- render idle-step videos + step the LTL monitor over the SAME steps; collect stats ---
    rv_diag, stats = render_views(
        env, diag, out_base_dir,
        episode=episode, n_frames=n_frames, fps=fps, resolution=resolution,
        mode="idle_step", ltl_monitor=monitor,
    )
    cameras = rv_diag["cameras"]
    steps_executed = int(stats["steps_executed"])
    ltl_violated = bool(monitor.violated) if monitor is not None else False
    ltl_summary = None
    if monitor is not None:
        s = monitor.summary()
        ltl_summary = {k: s[k] for k in
                       ("formula", "violated", "violation_step", "violation_count",
                        "total_steps_monitored") if k in s}

    # --- fresh uniform surface_info from the resolved surface object ---
    src_init = scene_info.get("objects_info", {}).get("init_info", {})
    surface_info = _fresh_surface_info(surf, support_top, src_init)

    # --- actual object inventory: count what is ACTUALLY in the finalized snapshot, not what
    # spawn_specs intended (the source pipeline drops objects it cannot place at generation time, so
    # spawn_specs over-counts). Counted from the just-saved bench snapshot so it matches validate. ---
    bench_init = json.loads(out_scene.read_text(encoding="utf-8")).get("objects_info", {}).get("init_info", {})
    robot_names = {r.name for r in env.robots}
    scene_names = set(bench_init) - robot_names
    marker_name = (diag.get("goal_region") or {}).get("marker_name")
    n_structural = sum(1 for nm in (getattr(surf, "name", None), marker_name) if nm and nm in scene_names)
    n_task_objects = len(scene_names) - n_structural  # target + clutter/fragile, actually placed
    n_task_intended = sum(int(s.get("count", 1)) for s in (diag.get("selection") or {}).get("spawn_specs", []) or [])
    n_src_objects = len(src_init) - sum(  # source snapshot non-robot count (excludes the 1 source robot)
        1 for info in src_init.values() if "Franka" in (info.get("class_name") or ""))

    # --- assemble diagnostics from selected fields and recomputed measurements ---
    carried = set(_CARRY_UNIVERSAL) | set(_CARRY_FAMILY.get(family, []))
    out_diag: dict = {f: diag[f] for f in (_CARRY_UNIVERSAL + _CARRY_FAMILY.get(family, []))
                      if f in diag}
    unexpected = sorted(f for f in diag if f not in carried and f not in _DROP)
    if "ltl_safety" in diag:
        out_diag["ltl_safety"] = ltl_safety  # save the specification used by the monitor
    # derived convenience family-info for clutter/lid (source pipeline never wrote one)
    out_diag.update(_derive_family_info(family, diag, n_task_objects))
    out_diag["cameras"] = cameras
    out_diag["surface_info"] = surface_info
    out_diag["gate_pass"] = bool(gate_pass)
    out_diag["ltl_violated"] = ltl_violated
    out_diag["steps_executed"] = steps_executed
    if ltl_summary is not None:
        out_diag["ltl_summary"] = ltl_summary
    out_diag["bench"] = {
        "finalized": True,
        "controller_preset": BENCH_CONTROLLER_PRESET,
        "grasping_mode": BENCH_GRASPING_MODE,
        "gpu_dynamics": needs_gpu,
        "n_src_objects": n_src_objects,
        "n_task_objects": n_task_objects,
        "n_task_intended": n_task_intended,
        "mount_offset": ROBOT_MOUNT_OFFSET,
        "support_top": round(support_top, 4),
        "base_z": round(base_z_after, 4),
        "base_z_before": round(base_z_before, 4),
        "mount_shift": round(base_z_after - base_z_before, 4),
        "init_pose": list(BENCH_INIT_QPOS),
        "arm_drift": stats["arm_drift"],
        "obj_disp": stats["obj_disp"],
        "gate": gate_detail,
        "dropped_unexpected_src_fields": unexpected or None,
    }
    (out_base_dir / "diagnostics.jsonl").write_text(json.dumps(out_diag, default=float) + "\n",
                                                    encoding="utf-8")

    # --- readback self-check from the saved snapshot ---
    header = json.loads(out_scene.read_text(encoding="utf-8"))
    jp, base_z_rb = _robot_from_snapshot(header)
    pose_ok = jp is not None and len(jp) == len(BENCH_INIT_QPOS) and max(
        abs(float(a) - b) for a, b in zip(jp, BENCH_INIT_QPOS)) < 1e-2
    basez_ok = base_z_rb is not None and abs(base_z_rb - base_z_after) < BASE_Z_TOL
    n_mp4 = len(list(out_base_dir.glob(f"rollout_*_ep{episode}.mp4")))

    warnings: list[str] = []
    if stats["arm_drift"] >= ARM_DRIFT_TOL:
        warnings.append(f"arm_drift={stats['arm_drift']:.3g}>={ARM_DRIFT_TOL}")
    if stats["obj_disp"] >= OBJ_DISP_WARN:
        warnings.append(f"obj_disp={stats['obj_disp']:.3g}>={OBJ_DISP_WARN}")
    if not gate_pass:
        warnings.append(f"gate_pass=False ({gate_detail})")
    if ltl_violated:
        warnings.append("ltl_violated=True over the idle-step")
    if not pose_ok:
        warnings.append("pose readback != A")
    if not basez_ok:
        warnings.append(f"base_z readback {base_z_rb} != {base_z_after:.4f}")
    if n_mp4 != 4:
        warnings.append(f"n_mp4={n_mp4} != 4")
    if unexpected:
        warnings.append(f"dropped unexpected src diag fields: {unexpected}")

    hard_fail = (stats["arm_drift"] >= ARM_DRIFT_TOL) or (not gate_pass) or ltl_violated \
        or (not pose_ok) or (not basez_ok) or (n_mp4 != 4)
    status = "fail" if hard_fail else ("warn" if warnings else "ok")

    return {
        "task": task_name,
        "family": family,
        "surface": diag.get("surface"),
        "support_top": round(support_top, 4),
        "base_z_before": round(base_z_before, 4),
        "base_z_after": round(base_z_after, 4),
        "mount_shift": round(base_z_after - base_z_before, 4),
        "pose": "A" if pose_ok else "DIFF",
        "gate_pass": bool(gate_pass),
        "ltl_violated": ltl_violated,
        "steps_executed": steps_executed,
        "arm_drift": stats["arm_drift"],
        "obj_disp": stats["obj_disp"],
        "n_mp4": n_mp4,
        # placed vs designed task objects, shown only when the source dropped some at generation time
        # (e.g. "7 -> 5"); empty when fully placed — scan the manifest to spot under-placed tasks.
        "object_spawn_num": f"{n_task_intended} -> {n_task_objects}" if n_task_intended != n_task_objects else "",
        "status": status,
        "warnings": warnings,
    }
