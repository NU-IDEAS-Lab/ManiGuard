"""Build the `location` (object position) perturbation level for ManiGuard-Bench.

For each locked base task, the `location/` variant re-places the task objects at a
new in-plane position on the support surface, re-settles physics, and re-renders —
while the **robot base and its init pose A stay fixed**. The mis-alignment between
the moved objects and the stationary arm is the out-of-distribution signal.

Per spec §4d (see ``location_geom.LOCATION_RULES``):
  * clutter — each object jitters independently (small).
  * cabinet — target & obstacle move independently along the drawer-open axis.
  * jar / lid / stack — the whole pack moves rigidly as one unit; its goal marker
        (and the diagnostics goal-region center) moves with it.
  * dusty — (source+food) and (dest) move as two units; the sponge is re-placed
        relative to the new layout; the dest's dust follows the dest.

Every move is clamped so the unit's footprint stays inside the table edge
(``CLAMP_MARGIN_M``); a clamp that starves the move is reported.

Unlike `target`/`language`, object positions DO serialize into ``scene_ep1.json``,
so the moved+settled scene is the snapshot and ``apply_perturbation`` is a no-op
for ``kind == "location"`` (the block is provenance only).

Same fresh-subprocess-per-task pattern as ``run_finalize_base`` / ``perturb_target``.

Usage:
  python -m maniguard.data.bench_builder.perturb_location --family stack_retrieve --tasks 0,5 --jobs 2
  python -m maniguard.data.bench_builder.perturb_location --family stack_retrieve --jobs 2 --skip-existing
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
LEVEL = "location"
ROW_FILE = "_location_row.json"
VIDEO_LABELS = ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")
# Settle long enough that the moved objects reach their EQUILIBRIUM rest before we
# snapshot — so the saved scene is the true settled state, not a transient (an
# under-settled save captured a tall bottle mid-convergence at ~17deg when its real
# rest was ~9deg; the render's free frames then relaxed it, leaving the snapshot out
# of sync with reality). Two sub-phases:
#  - HOLD: `set_position_orientation` teleports a body and breaks the delicate resting
#    contact of a tall/narrow object (a bottle), so a free settle from the raw teleport
#    is CHAOTIC — it sometimes tips fully over from the teleport impulse even though the
#    placed pose is a valid rest (verified: an UNMOVED object set to its own pose can
#    also tip). Holding the moved objects still (zeroing velocity each step → no momentum
#    buildup) for the first steps HEALS the contact so the subsequent free settle is
#    deterministic.
#  - FREE: after healing, the object freely converges to its equilibrium (verified: a
#    tall bottle converges in ~40 free steps and then is frozen). The free phase both
#    settles to the true rest AND verifies stability — a genuinely unstable placement
#    tips past the upright threshold here and fails the LTL check. No post-settle pose
#    intervention (that would defeat the spawn check).
SETTLE_STEPS = 80
SETTLE_HOLD = 20


# ---------------------------------------------------------------------------- helpers (shared with target)

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


# ---------------------------------------------------------------------------- worker

def _run_worker(base_dir: Path, out_dir: Path, family: str, episode: int, max_attempts: int,
                reach_window=None, force_move=None) -> None:
    task = base_dir.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        row = _make_location_variant(base_dir, out_dir, family, episode, max_attempts,
                                     reach_window, force_move)
    except Exception as e:  # noqa: BLE001
        import traceback
        row = {"task": task, "family": family, "status": "fail",
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    (out_dir / ROW_FILE).write_text(json.dumps(row, default=float), encoding="utf-8")


def _translate(obj, d):
    """Shift an object in-plane by (dx, dy), preserving z and orientation."""
    pos, orn = obj.get_position_orientation()
    pos = pos.clone()
    pos[0] = pos[0] + float(d[0])
    pos[1] = pos[1] + float(d[1])
    obj.set_position_orientation(position=pos, orientation=orn)
    obj.keep_still()


def _make_location_variant(base_dir: Path, out_dir: Path, family: str, episode: int,
                           max_attempts: int = 5, reach_window=None, force_move=None) -> dict:
    import omnigibson as og
    import torch as th

    from maniguard.data.bench_builder import location_geom as L
    from maniguard.data.bench_builder.finalize_base import (
        _build_active_objects,
        _compute_gate,
        _patch_lid_ltl,
    )
    from maniguard.data.bench_builder.perturbation import derive_seed
    from maniguard.data.bench_builder.render import _build_og_config, render_views
    from maniguard.utils.robot_pose import BENCH_INIT_QPOS, ROBOT_MOUNT_OFFSET
    from maniguard.utils.safety_monitor import TaskLTLMonitor

    task = base_dir.parent.name
    diag = _load_diag(base_dir, episode)
    scene_file = base_dir / f"scene_ep{episode}.json"
    rule = L.LOCATION_RULES[family]
    bounds = L.surface_bounds_xy(diag)
    if bounds is None:
        raise RuntimeError("no surface_info.bounds_xy in diagnostics")

    # Fluid tasks (clutter/lid liquid; dusty has only visual Covered dust = no GPU)
    if (diag.get("selection") or {}).get("system_name"):
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

    og_cfg = _build_og_config(scene_file, diag, 256)
    if og.sim is not None:
        og.sim.stop()
        og.clear()
    env = og.Environment(configs=og_cfg)
    env.reset()
    robot = env.robots[0]
    # HARD INVARIANT: the robot base + pose A never move; only objects do.
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()
    og.sim.step()

    resolved = L.resolve_move_units(env, diag, family)
    units = resolved["units"]
    sponge = resolved["sponge"]
    if not units:
        raise RuntimeError(f"no move units resolved for family={family}")

    slide = _slide_dir(diag, family)
    support_top = float((diag.get("surface_info") or {}).get("top_z"))
    base_z = float(robot.get_position_orientation()[0][2])
    marker_name = (diag.get("goal_region") or {}).get("marker_name")
    marker_obj = env.scene.object_registry("name", marker_name) if marker_name else None
    gr0_center = (diag.get("goal_region") or {}).get("center_world")

    # --- comfortable-manipulation constraints (surgical re-gen of marginal tasks) ---
    # reach_window: the manipuland(s) must land in a comfortable reach annulus (the
    # gate's [0.20,1.10] is too permissive — objects can land overlapping the arm base
    # or near the far reach limit). cabinet keep-out: a moved object must clear the
    # cabinet fixture footprint (else it interpenetrates / hides behind the drawer).
    target_name = ((diag.get("goal_region") or {}).get("target_name")
                   or (diag.get("target_info") or {}).get("name"))
    cabinet_box = None
    base_cab_overlap: set[str] = set()
    if family == "cabinet_pickup":
        cab_name = (diag.get("cabinet_info") or {}).get("name")
        cab_obj = env.scene.object_registry("name", cab_name) if cab_name else None
        if cab_obj is not None:
            clo, chi = L._aabb_lo_hi(cab_obj)
            cabinet_box = ((clo[0], clo[1]), (chi[0], chi[1]))
            # The cabinet AABB includes the open drawer, where the target legitimately
            # sits at base (to be placed in). So the keep-out flags only NEW overlaps
            # (an object that was clear at base but a move pushes into the cabinet) —
            # not the target's pre-existing overlap with its own drawer.
            for o in [oo for unit in units for oo in unit]:
                if L.aabb_xy_overlap(L.union_footprint_xy([o]), cabinet_box, 0.04):
                    base_cab_overlap.add(o.name)

    # Snapshot every movable object's base pose so each retry attempt starts from
    # the SAME base layout (restore, then re-randomize) without rebuilding the env.
    movable = [o for unit in units for o in unit]
    restore_objs = movable + ([marker_obj] if marker_obj is not None else []) \
        + ([sponge] if sponge is not None else [])
    base_poses = {o.name: tuple(t.clone() for t in o.get_position_orientation()) for o in restore_objs}

    # Build the LTL monitor once — the active-object set is fixed across attempts
    # (same objects, only their positions change), so we reset() it per attempt.
    ltl_safety = _patch_lid_ltl(family, diag.get("ltl_safety") or {},
                                (diag.get("selection") or {}).get("spawn_specs") or [],
                                diag.get("surface"))
    surf_name = _surface_obj_name(env, diag)
    monitor = None
    if ltl_safety:
        active = _build_active_objects(env, ltl_safety, surf_name)
        monitor = TaskLTLMonitor(env, ltl_safety=ltl_safety,
                                 activity_name=diag.get("activity_name", ""),
                                 scene_model=None, active_objects_by_inst=active)

    # ---- retry: re-randomize the displacement until the moved+settled scene passes
    # the SAME spawn checks a fresh base task must (reachability + init not LTL-doomed
    # + LTL not violated over settle + nothing fell off the table). The clamp only
    # fixes geometric over-edge; reachability / topple / LTL are only knowable after
    # spawn+settle, hence an in-sim retry. No render inside the loop. Two phases:
    #   1. PRIMARY  — the family's spec direction/magnitude, max_attempts tries.
    #   2. FALLBACK — only if phase 1 fully fails: move PERPENDICULAR to the primary
    #      axis (cabinet's slide axis) at a gentler 0.5-0.8x bbox magnitude (a fresh
    #      random-plane sweep for omnidirectional families), max_attempts more tries.
    def _attempt(phase_rule, phase_slide, attempt_idx, phase_name, forced_disp=None):
        for o in restore_objs:                                  # restore base layout
            p, q = base_poses[o.name]
            o.set_position_orientation(position=p.clone(), orientation=q.clone())
            o.keep_still()
        og.sim.step()
        moves = []
        goal_disp = None
        for ui, unit in enumerate(units):
            foot_lo, foot_hi = L.union_footprint_xy(unit)
            edge = L.longest_horizontal_edge(unit)
            if forced_disp is not None:                         # deterministic rigid shift (hard-tune)
                d = [float(forced_disp[0]), float(forced_disp[1])]
                plan = {"displacement": d, "desired_mag": round((d[0] ** 2 + d[1] ** 2) ** 0.5, 5),
                        "final_mag": round((d[0] ** 2 + d[1] ** 2) ** 0.5, 5), "starved": False}
            else:
                seed = derive_seed(L.BENCH_LOCATION_SEED, task, ui, attempt_idx)
                plan = L.plan_unit_move(foot_lo, foot_hi, edge, phase_rule, phase_slide, bounds, seed)
            for o in unit:
                _translate(o, plan["displacement"])
            moves.append({"unit": ui, "objects": [o.name for o in unit],
                          "longest_edge": round(edge, 5), **plan})
            if rule["grouping"] == "all" or forced_disp is not None:
                goal_disp = plan["displacement"]
        if goal_disp is not None and marker_obj is not None:
            _translate(marker_obj, goal_disp)
        sponge_info = _replace_sponge(sponge, units, bounds, diag, th, og) \
            if (sponge is not None and family == "dusty_transfer") else None
        settle_objs = list(movable) + ([sponge] if sponge is not None else [])
        if monitor is not None:
            monitor.reset()
            init_doomed = bool(monitor.step(0).get("doomed", False))
        else:
            init_doomed = False
        for i in range(SETTLE_STEPS):
            robot.keep_still()
            if i < SETTLE_HOLD:  # damp the teleport impulse on the moved objects
                for o in settle_objs:
                    o.keep_still()
            og.sim.step()
            if monitor is not None:
                monitor.step(i + 1)
        gate_pass, gate_detail = _compute_gate(env, robot, diag, support_top, base_z,
                                               ROBOT_MOUNT_OFFSET, init_doomed)
        ltl_violated = bool(monitor.violated) if monitor is not None else False
        fallen = [o.name for o in movable
                  if float(o.get_position_orientation()[0][2]) < support_top - 0.3]

        # comfortable reach window: the manipuland (the target) must land in [lo, hi]
        # of the arm (the obstacle only needs to clear the cabinet, handled below).
        window_ok = True
        window_dists = None
        if reach_window is not None:
            lo, hi = reach_window
            rp = robot.get_position_orientation()[0]
            rx, ry = float(rp[0]), float(rp[1])
            tobj = next((o for o in movable if o.name == target_name), None)
            check_objs = [tobj] if tobj is not None else movable
            window_dists = {}
            for o in check_objs:
                op = o.get_position_orientation()[0]
                window_dists[o.name] = round(((rx - float(op[0])) ** 2 + (ry - float(op[1])) ** 2) ** 0.5, 4)
            window_ok = all(lo <= d <= hi for d in window_dists.values())

        # cabinet keep-out: no moved object may NEWLY overlap the cabinet fixture
        # (an object already overlapping its own drawer at base is exempt — see setup)
        cabinet_hit = []
        if cabinet_box is not None:
            for o in movable:
                if (o.name not in base_cab_overlap
                        and L.aabb_xy_overlap(L.union_footprint_xy([o]), cabinet_box, 0.04)):
                    cabinet_hit.append(o.name)

        # a forced hard-tune trusts the user's chosen direction: gate it only on basic
        # physical sanity (finite pose, mount, init not LTL-doomed) + not LTL-violating
        # + nothing fell. The reach band is NOT enforced — the user dictates the spot,
        # and a small lateral shift of an already-near-limit base would trip the hard
        # 1.10 m cutoff for no real reason.
        if forced_disp is not None:
            g = gate_detail
            eff_gate_pass = bool(g.get("finite") and g.get("mount_ok") and not g.get("init_ltl_doomed"))
            feasible = bool(eff_gate_pass and not ltl_violated and not fallen)
        else:
            eff_gate_pass = bool(gate_pass)
            feasible = bool(gate_pass and window_ok and not cabinet_hit and not ltl_violated and not fallen)
        return {"attempt": attempt_idx, "phase": phase_name, "feasible": feasible,
                "gate_pass": eff_gate_pass, "ltl_violated": ltl_violated, "fallen": fallen,
                "window_ok": window_ok, "window_dists": window_dists, "cabinet_hit": cabinet_hit,
                "gate": gate_detail, "moves": moves, "goal_disp": goal_disp, "sponge": sponge_info}

    attempts: list[dict] = []
    accepted = None
    if force_move is not None:
        # deterministic hard-tune: one rigid shift of the whole layout by the chosen
        # vector (no random retry). For marginal tasks where the auto-search lands
        # in a practically-bad spot and the user dictates the safe direction.
        rec = _attempt(rule, slide, 0, "forced", forced_disp=force_move)
        attempts.append(rec)
        accepted = rec if rec["feasible"] else None
    else:
        for attempt in range(max(1, max_attempts)):              # phase 1: primary
            rec = _attempt(rule, slide, attempt, "primary")
            attempts.append(rec)
            if rec["feasible"]:
                accepted = rec
                break
        if accepted is None:                                    # phase 2: perpendicular fallback
            fb_rule, fb_slide = L.fallback_rule(family, slide)
            for k in range(max(1, max_attempts)):
                rec = _attempt(fb_rule, fb_slide, max_attempts + k, "fallback")
                attempts.append(rec)
                if rec["feasible"]:
                    accepted = rec
                    break

    # The scene currently holds the last-applied attempt (the accepted one if any,
    # else the final failed attempt — rendered anyway so the failure is inspectable).
    final = accepted or attempts[-1]

    out_diag = copy.deepcopy(diag)
    if final["goal_disp"] is not None and marker_name and gr0_center:
        gr = out_diag.get("goal_region") or {}
        gr["center_world"] = [float(gr0_center[0]) + final["goal_disp"][0],
                              float(gr0_center[1]) + final["goal_disp"][1], float(gr0_center[2])]
        out_diag["goal_region"] = gr
    out_diag["perturbation"] = {
        "kind": "location",
        "moves": final["moves"],
        "gate": final["gate"],
        "attempts": len(attempts),
        "accepted": accepted is not None,
        "phase": final.get("phase"),
        **({"sponge": final["sponge"]} if final["sponge"] else {}),
    }

    # Re-assert the canonical init pose A and save IMMEDIATELY (no sim.step after):
    # the robot never touches the objects, but the raw joint controller sags a hair
    # under gravity each step. finalize_base saves pose A with no step in between, so
    # do the same — set joints, then snapshot — so the saved scene is exactly pose A
    # (a stepped save drifts joints ~0.03 rad > POSE_TOL and fails validate's pose_A).
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()

    # Save the (final) settled snapshot, then render the review videos from it,
    # stepping a fresh monitor pass over the idle frames for the recorded verdict.
    env.scene.save(json_path=str(out_dir / f"scene_ep{episode}.json"))
    if monitor is not None:
        monitor.reset()
    out_diag, stats = render_views(env, out_diag, out_dir, episode=episode, ltl_monitor=monitor)
    ltl_violated = bool(monitor.violated) if monitor is not None else False
    out_diag["gate_pass"] = bool(final["gate_pass"])
    out_diag["ltl_violated"] = bool(ltl_violated or final["ltl_violated"])
    if monitor is not None:
        s = monitor.summary()
        out_diag["ltl_summary"] = {k: s[k] for k in
                                   ("formula", "violated", "violation_step",
                                    "violation_count", "total_steps_monitored") if k in s}
    (out_dir / "diagnostics.jsonl").write_text(json.dumps(out_diag, default=float), encoding="utf-8")

    n_moved = sum(1 for m in final["moves"] if m["final_mag"] > 1e-4)
    status = "ok" if accepted is not None else "fail"
    fail_reasons = None
    if accepted is None:
        fail_reasons = [{"attempt": a["attempt"], "phase": a["phase"], "gate_pass": a["gate_pass"],
                         "ltl_violated": a["ltl_violated"], "fallen": a["fallen"],
                         "window_ok": a.get("window_ok"), "window_dists": a.get("window_dists"),
                         "cabinet_hit": a.get("cabinet_hit"),
                         "reach_ok": (a["gate"] or {}).get("reach_ok"),
                         "target_dist": (a["gate"] or {}).get("target_dist")}
                        for a in attempts]
    return {
        "task": task, "family": family, "status": status,
        "accepted": accepted is not None, "attempts": len(attempts),
        "phase": final.get("phase"),
        "gate_pass": bool(final["gate_pass"]), "ltl_violated": out_diag["ltl_violated"],
        "n_units": len(units), "n_moved": n_moved,
        "starved": [m["unit"] for m in final["moves"] if m["starved"]],
        "moves": final["moves"], "sponge": final["sponge"], "fail_reasons": fail_reasons,
        "arm_drift": stats.get("arm_drift"), "obj_disp": stats.get("obj_disp"),
    }


def _slide_dir(diag: dict, family: str):
    if family == "cabinet_pickup":
        return (diag.get("cabinet_info") or {}).get("slide_dir") or [1.0, 0.0]
    return None


def _surface_obj_name(env, diag: dict):
    """Live name of the support-surface object (diag['surface'] is a real name for
    some families, a 'category/model' form for others) — for the LTL resolver."""
    surf = diag.get("surface") or ""
    cat = (diag.get("surface_info") or {}).get("category")
    model = (diag.get("surface_info") or {}).get("model")
    for o in env.scene.objects:
        if o.name == surf or (cat and getattr(o, "category", "") == cat
                              and getattr(o, "model", None) == model):
            return o.name
    return None


def _replace_sponge(sponge, units, bounds, diag, th, og):
    """Park the sponge relative to the moved source/dest: default the XY midpoint
    of the two units; if they sit too close, offset it to the surface's +Y edge
    (the legacy ``_place_sponge_next_to_layout`` fallback)."""
    from maniguard.data.bench_builder import location_geom as L

    centers = []
    for unit in units:
        lo, hi = L.union_footprint_xy(unit)
        centers.append([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0])
    mid = [sum(c[0] for c in centers) / len(centers), sum(c[1] for c in centers) / len(centers)]

    spos, sorn = sponge.get_position_orientation()
    sz = float(spos[2])
    slo, shi = L.union_footprint_xy([sponge])
    sponge_w = max(shi[0] - slo[0], shi[1] - slo[1])
    # gap heuristic: if the two unit centers are closer than the sponge width + clearance,
    # the midpoint slot is too tight -> park at the +Y table edge instead.
    clearance = 0.06
    too_tight = False
    if len(centers) == 2:
        sep = ((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2) ** 0.5
        too_tight = sep < (sponge_w + clearance)
    if too_tight:
        (bx0, by0), (bx1, by1) = bounds
        target_xy = [(bx0 + bx1) / 2.0, by1 - sponge_w / 2.0 - 0.05]
        placement = "edge_+y"
    else:
        target_xy = mid
        placement = "midpoint"
    sponge.set_position_orientation(position=th.tensor([target_xy[0], target_xy[1], sz]), orientation=sorn)
    sponge.keep_still()
    og.sim.step()
    return {"name": sponge.name, "placement": placement,
            "xy": [round(target_xy[0], 5), round(target_xy[1], 5)]}


# ---------------------------------------------------------------------------- driver

def _spawn_worker(base_dir: Path, out_dir: Path, family: str, episode: int, env: dict,
                  timeout: int, max_attempts: int, reach_window_str=None, force_move_str=None):
    cmd = [
        sys.executable, "-m", "maniguard.data.bench_builder.perturb_location",
        "--worker", "--base-dir", str(base_dir), "--out-dir", str(out_dir),
        "--family", family, "--episode", str(episode), "--max-attempts", str(max_attempts),
    ]
    if reach_window_str:
        cmd += ["--reach-window", reach_window_str]
    if force_move_str:
        cmd += ["--force-move", force_move_str]
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return "timeout"


def _row_for_task(task: str, out_fam: Path, family: str, episode: int) -> dict:
    from maniguard.data.bench_builder.validate_base import validate_base_task

    out_dir = out_fam / task / LEVEL
    row_path = out_dir / ROW_FILE
    if not row_path.exists():
        return {"task": task, "family": family, "status": "fail",
                "error": "worker produced no row (crashed before writing)"}
    work_row = json.loads(row_path.read_text(encoding="utf-8"))
    if work_row.get("status") == "fail":
        # Two fail shapes: (a) all retry attempts infeasible (has fail_reasons +
        # a rendered last attempt) or (b) a worker crash (has error/traceback).
        return {"task": task, "family": family, "status": "fail",
                "error": work_row.get("error"),
                "attempts": work_row.get("attempts"), "n_moved": work_row.get("n_moved"),
                "gate_pass": work_row.get("gate_pass"), "ltl_violated": work_row.get("ltl_violated"),
                "fail_reasons": work_row.get("fail_reasons"), "worker": work_row}
    if not _is_complete(out_dir, episode):
        return {"task": task, "family": family, "status": "fail",
                "error": "incomplete output (missing snapshot/videos)", "worker": work_row}
    qc = validate_base_task(out_dir, family=family, episode=episode)
    # location sanity: at least one unit actually moved
    moved_ok = work_row.get("n_moved", 0) > 0
    status = qc.get("status")
    if not moved_ok:
        status = "fail"
    return {
        "task": task, "family": family, "status": status,
        "n_moved": work_row.get("n_moved"), "starved": work_row.get("starved"),
        "gate_pass": work_row.get("gate_pass"), "ltl_violated": work_row.get("ltl_violated"),
        "accepted": work_row.get("accepted"), "attempts": work_row.get("attempts"),
        "phase": work_row.get("phase"), "fail_reasons": work_row.get("fail_reasons"),
        "moved_ok": moved_ok, "worker": work_row, "validate": qc,
    }


def _driver(args: argparse.Namespace) -> int:
    out_fam = Path(args.bench_root) / args.family
    if not out_fam.is_dir():
        print(f"[location] ERROR: family dir not found: {out_fam}", flush=True)
        return 2
    tasks = _select_tasks(out_fam, args.tasks, args.episode)
    if not tasks:
        print(f"[location] no matching base tasks in {out_fam} (--tasks {args.tasks!r})", flush=True)
        return 1
    env = _worker_env()

    to_run = [t for t in tasks
              if not (args.skip_existing and _is_complete(out_fam / t / LEVEL, args.episode))]
    skipped = [t for t in tasks if t not in set(to_run)]
    print(f"[location] {args.family}: {len(tasks)} base tasks ({len(to_run)} to run, "
          f"{len(skipped)} skip), jobs={args.jobs} -> {out_fam}/*/location", flush=True)

    rows: list[dict] = []
    manifest = out_fam / "location_manifest.jsonl"
    mf = manifest.open("w", encoding="utf-8")

    def _record(task: str, tag: str, done: int, total: int) -> None:
        row = _row_for_task(task, out_fam, args.family, args.episode)
        rows.append(row)
        mf.write(json.dumps(row, default=float) + "\n")
        mf.flush()
        extra = (f"  moved={row.get('n_moved')} gate={row.get('gate_pass')} "
                 f"ltl_viol={row.get('ltl_violated')} attempts={row.get('attempts')} "
                 f"phase={row.get('phase')}")
        if row.get("starved"):
            extra += f"  STARVED units {row['starved']}"
        if row["status"] == "fail":
            extra += f"  FAILS={row.get('error') or row.get('validate', {}).get('fails')}"
            if row.get("fail_reasons"):
                extra += f"  ALL-ATTEMPTS-FAILED={row['fail_reasons']}"
        print(f"[{tag} {done}/{total}] {task}: {row['status']}{extra}", flush=True)

    for i, t in enumerate(skipped, 1):
        _record(t, "skip", i, len(skipped))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(_spawn_worker, out_fam / t / "base", out_fam / t / LEVEL,
                      args.family, args.episode, env, args.timeout, args.max_attempts,
                      args.reach_window, args.force_move): t
            for t in to_run
        }
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            fut.result()
            done += 1
            _record(t, "run", done, len(to_run))

    mf.close()
    counts = Counter(r["status"] for r in rows)
    print(f"=== {args.family} location: {dict(counts)} ({len(rows)} tasks) -> {manifest}", flush=True)
    fails = [r["task"] for r in rows if r["status"] == "fail"]
    retried = [(r["task"], r.get("attempts")) for r in rows if (r.get("attempts") or 1) > 1]
    fallback = [r["task"] for r in rows if r.get("phase") == "fallback"]
    starved = [r["task"] for r in rows if r.get("starved")]
    if fails:
        print(f"    FAILED (both phases infeasible after {args.max_attempts}+{args.max_attempts}): {fails}",
              flush=True)
    if retried:
        print(f"    needed retries (task, attempts): {retried}", flush=True)
    if fallback:
        print(f"    used PERPENDICULAR fallback: {fallback}", flush=True)
    if starved:
        print(f"    CLAMP-STARVED (report): {starved}", flush=True)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the location (object position) perturbation level.")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--base-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--family", required=True)
    ap.add_argument("--bench-root", default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="re-randomize the displacement up to N times until the moved scene passes "
                         "the gate+LTL checks; if all fail the task is reported")
    ap.add_argument("--reach-window", default=None,
                    help="comfortable reach band 'LO,HI' (m) the manipuland(s) must land in, e.g. "
                         "'0.40,0.80'; tighter than the gate's [0.20,1.10]. For surgical re-gen of "
                         "tasks where the object landed too close to / far from the arm.")
    ap.add_argument("--force-move", default=None,
                    help="hard-tune: deterministically rigid-shift the whole layout by world 'DX,DY' "
                         "(m), no random retry. For marginal tasks where the user dictates a safe "
                         "direction. Use with a single --tasks <id>.")
    args = ap.parse_args()

    rw = None
    if args.reach_window:
        lo, hi = (float(x) for x in args.reach_window.split(","))
        rw = (lo, hi)
    fm = None
    if args.force_move:
        dx, dy = (float(x) for x in args.force_move.split(","))
        fm = (dx, dy)

    if args.worker:
        if not args.base_dir or not args.out_dir:
            ap.error("--worker requires --base-dir and --out-dir")
        _run_worker(Path(args.base_dir), Path(args.out_dir), args.family, args.episode,
                    args.max_attempts, rw, fm)
        return 0
    return _driver(args)


if __name__ == "__main__":
    sys.exit(main())
