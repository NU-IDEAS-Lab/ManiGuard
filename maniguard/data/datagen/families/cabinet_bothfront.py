"""One-shot: respawn the 12 far-target cabinet base tasks as both-front.

The 12 ``blocker_mode="obstacle"`` cabinet tasks spawn the *target* on the
off-side of the drawer (perpendicular ``side_clearance_m`` away), ~1.05 m from
the robot base — unreachable. This script reloads each base task into
OmniGibson, calls the pipeline's own ``_layout_target_and_obstacle`` with
``blocker_mode="both"`` (both objects in front of the drawer's leading face,
staggered along the slide axis), settles, and rewrites ``scene_ep1.json`` +
``diagnostics.jsonl`` so both objects are reachable. Originals are backed up
to ``*.bak_bothfront`` (only on the first run — re-runs never clobber a backup).

OmniGibson's sim is a singleton, so this runs ONE task per process;
``--task all`` fans out to a subprocess per task (single GPU = single process).
Run headless via ``python -u``; the exit-139 teardown segfault is benign — all
file writes happen before any teardown.

  # dry-run one task (no writes): prints the new xy + dist-to-robot
  python -u -m maniguard.data.datagen.families.cabinet_bothfront --task task_0001 --dry-run
  # apply all 12
  python -u -m maniguard.data.datagen.families.cabinet_bothfront --task all
  # offline verify (no sim): 12-row table
  python -u -m maniguard.data.datagen.families.cabinet_bothfront --verify
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# The 12 far-target tasks (blocker_mode="obstacle", target on the off-side).
FAR_TASKS = [
    "task_0001", "task_0003", "task_0007", "task_0013", "task_0015",
    "task_0018", "task_0021", "task_0023", "task_0026", "task_0028",
    "task_0031", "task_0034",
]

BENCH_ROOT = Path("outputs/lerobot_datasets/maniguard-bench/cabinet_pickup")

# Gap from the drawer's leading face to the target (pipeline default object_gap_m).
GAP_M = 0.05


def _load_diag(task_dir: Path) -> dict:
    p = task_dir / "diagnostics.jsonl"
    with open(p) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{p}: expected 1 diagnostics row, got {len(rows)}")
    return rows[0]


def _horiz_half(obj) -> float:
    """Max horizontal half-extent (x or y) of an object's live AABB."""
    a, b = obj.aabb
    a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
    b = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
    return 0.5 * max(float(b[0] - a[0]), float(b[1] - a[1]))


def _layout_both_front_adaptive(og, drawer_link, target_obj, obstacle_obj,
                                slide_dir_xy, top_z, robot, surf_min, surf_max,
                                *, gap_m=GAP_M, margin_m=0.04, perp_margin_m=0.02,
                                obstacle_offside=False):
    """Both-front layout with ADAPTIVE, non-overlapping separation.

    The pipeline's ``_layout_target_and_obstacle(blocker_mode="both")`` staggers
    the two objects by a FIXED ``obstacle_extra_gap_m=0.10`` along the slide
    axis. For object pairs whose combined half-width exceeds 0.10 they spawn
    interpenetrating, and a settle launches them (the task_0003/0007/0031
    explosions). Here the obstacle is set back along +slide by
    ``target_hw + obstacle_hw + margin`` so the two never overlap — then a
    short velocity-zeroed settle, and the ACTUAL post-settle poses are returned
    (gravity stays disabled, matching how the native both-tasks were saved).

    Both objects sit at the drawer's leading-face x (= reachable column for a
    robot offset along x); the obstacle is the one set farther forward.
    """
    import torch as th
    from maniguard.task_generation.cabinet_pickup_pipeline import (
        _aabb_np,
        _place_obj_upright_on_surface,
    )

    d_min, d_max = _aabb_np(drawer_link)
    cx = float((d_min[0] + d_max[0]) * 0.5)
    cy = float((d_min[1] + d_max[1]) * 0.5)
    sx, sy = float(slide_dir_xy[0]), float(slide_dir_xy[1])
    extent_xy = np.array([float(d_max[0] - d_min[0]),
                          float(d_max[1] - d_min[1])], dtype=np.float64)
    half_along = 0.5 * float(np.dot(extent_xy, np.abs([sx, sy])))
    lead_x = cx + sx * half_along
    lead_y = cy + sy * half_along

    def along(extra):
        return (lead_x + sx * (gap_m + extra), lead_y + sy * (gap_m + extra))

    # Near-edge rule (§11b): if the robot is clearly offset along the cross-slide
    # axis p, shift an in-path object to the band edge NEAREST the robot (still
    # inside the band -> still blocks the drawer), clamped by the object's
    # p-half-width. No-op when the robot is ~aligned (|rp| <= band_half).
    p = np.array([-sy, sx], dtype=np.float64)               # unit perpendicular to slide
    drawer_c = np.array([cx, cy], dtype=np.float64)
    band_half = 0.5 * float(np.dot(
        np.array([d_max[0]-d_min[0], d_max[1]-d_min[1]], dtype=np.float64), np.abs(p)))
    rob_xy = robot.get_position_orientation()[0].cpu().numpy()[:2].astype(np.float64)
    rp = float(np.dot(rob_xy - drawer_c, p))

    def perp_half(obj):
        a, b = obj.aabb
        a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        b = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
        return 0.5 * float(np.dot(np.abs(b[:2] - a[:2]), np.abs(p)))

    def half_xy(obj):
        a, b = obj.aabb
        a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        b = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
        return np.array([0.5 * float(b[0] - a[0]), 0.5 * float(b[1] - a[1])])

    def near_edge_shift(ph):
        if abs(rp) <= band_half:
            return np.zeros(2)
        return np.sign(rp) * max(0.0, band_half - ph - perp_margin_m) * p

    si = 0 if abs(sx) >= abs(sy) else 1       # slide axis (slide is axis-aligned)
    pi = 1 - si                               # cross-slide (band) axis
    npi = 1.0 if rob_xy[pi] > drawer_c[pi] else -1.0   # robot-near band side along pi

    # Target: leading spot + near-edge (robot side).
    t0 = np.array(along(0.0), dtype=np.float64)
    _place_obj_upright_on_surface(og, target_obj, float(t0[0]), float(t0[1]), top_z)
    t_hw = _horiz_half(target_obj)
    t_half = half_xy(target_obj)
    t_xy = t0 + near_edge_shift(perp_half(target_obj))
    _place_obj_upright_on_surface(og, target_obj, float(t_xy[0]), float(t_xy[1]), top_z)

    # Obstacle: provisional far read for size.
    of = np.array(along(0.45), dtype=np.float64)
    _place_obj_upright_on_surface(og, obstacle_obj, float(of[0]), float(of[1]), top_z)
    o_hw = _horiz_half(obstacle_obj)
    o_half = half_xy(obstacle_obj)
    sep = t_hw + o_hw + margin_m

    def off_table(c, h):
        return (c[0] - h[0] < surf_min[0] + 0.01 or c[0] + h[0] > surf_max[0] - 0.01 or
                c[1] - h[1] < surf_min[1] + 0.01 or c[1] + h[1] > surf_max[1] - 0.01)

    def hits_target(c, h):
        return (abs(c[0] - t_xy[0]) < h[0] + t_half[0] + margin_m and
                abs(c[1] - t_xy[1]) < h[1] + t_half[1] + margin_m)

    if obstacle_offside:
        # target-mode: obstacle is decorative, parked OFF the drawer's swept band
        # on the side AWAY from the robot (out of path, on-table). It is never moved.
        o_xy = np.array(along(0.0), dtype=np.float64)        # same slide column as target
        o_xy[pi] = drawer_c[pi] - npi * (band_half + o_half[pi] + 0.06)
        o_xy[si] = float(np.clip(o_xy[si], surf_min[si] + o_half[si] + 0.01,
                                 surf_max[si] - o_half[si] - 0.01))
        o_xy[pi] = float(np.clip(o_xy[pi], surf_min[pi] + o_half[pi] + 0.01,
                                 surf_max[pi] - o_half[pi] - 0.01))
    else:
        # both-mode: obstacle in-path, forward-stagger + near-edge, with a MINIMAL
        # on-table nudge (pull in along slide, then push along band to clear target).
        o_xy = np.array(along(sep), dtype=np.float64) + near_edge_shift(perp_half(obstacle_obj))
        if off_table(o_xy, o_half):
            o_xy[si] = float(np.clip(o_xy[si], surf_min[si] + o_half[si] + 0.01,
                                     surf_max[si] - o_half[si] - 0.01))
            if hits_target(o_xy, o_half):
                need = t_half[pi] + o_half[pi] + margin_m
                o_xy[pi] = t_xy[pi] - npi * need
                o_xy[pi] = float(np.clip(o_xy[pi], surf_min[pi] + o_half[pi] + 0.01,
                                         surf_max[pi] - o_half[pi] - 0.01))
    _place_obj_upright_on_surface(og, obstacle_obj, float(o_xy[0]), float(o_xy[1]), top_z)

    # Settle with a drift guard: this cuRobo/PhysX fork occasionally LAUNCHES an
    # object during settle (observed task_0012: target flung 0.88 m off-table). If
    # any object drifts > 8 cm from its intended xy, re-place it and re-settle.
    intended = [
        ("target", target_obj, t_xy),
        ("obstacle", obstacle_obj, o_xy),
    ]
    for _retry in range(4):
        for _r, obj, _xy in intended:
            obj.root_link.set_linear_velocity(th.zeros(3, dtype=th.float32))
            obj.root_link.set_angular_velocity(th.zeros(3, dtype=th.float32))
        for _ in range(5):
            og.sim.step()
        drifted = False
        for _r, obj, xy_int in intended:
            cur = obj.get_position_orientation()[0].cpu().numpy()[:2]
            if float(np.linalg.norm(cur - xy_int)) > 0.08:
                drifted = True
                _place_obj_upright_on_surface(og, obj, float(xy_int[0]), float(xy_int[1]), top_z)
        if not drifted:
            break

    out = {"sep": sep, "t_hw": t_hw, "o_hw": o_hw}
    for role, obj in (("target", target_obj), ("obstacle", obstacle_obj)):
        p = obj.get_position_orientation()[0].cpu().numpy()
        b = obj.aabb[1]
        b = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
        out[role] = {
            "xy": (float(p[0]), float(p[1])),
            "z": float(p[2]),
            "top_z": float(b[2]),
            "in_path": not (role == "obstacle" and obstacle_offside),
        }
    return out


def _respawn_one(task_id: str, *, dry_run: bool, mode: str = "both") -> None:
    task_dir = BENCH_ROOT / task_id / "base"
    diag = _load_diag(task_dir)
    offside = (mode == "target")   # target-mode: obstacle parked off-side, not moved

    # Heavy imports are lazy so --verify stays sim-free.
    from maniguard.data.datagen.primitives.scene import (
        init_omnigibson,
        scene_from_task_dir,
    )
    from maniguard.task_generation.cabinet_pickup_pipeline import _aabb_np
    from maniguard.task_generation.pipeline_common import save_episode_scene

    og = init_omnigibson(headless=True)
    bundle = scene_from_task_dir(task_dir)
    env, robot = bundle.env, bundle.robot

    cab_info = diag["cabinet_info"]
    cab = env.scene.object_registry("name", cab_info["name"])
    drawer_link = cab.links[cab_info["link"]]
    slide_dir = np.array(cab_info["slide_dir"], dtype=np.float32)

    # Force the correct drawer state before laying out: open the SELECTED joint
    # (cabinet_info["joint"]) to open_fraction, close every other prismatic drawer.
    # Guards against the task_0034 drift where an upper drawer (link_3) slid open
    # while the selected big bottom drawer (link_1) sat nearly closed — which also
    # corrupts the leading-face the objects are placed against. No-op when already
    # correct.
    from omnigibson.utils.constants import JointType
    open_frac = float(cab_info.get("open_fraction", 0.2))
    for jname, j in cab.joints.items():
        if j.joint_type != JointType.JOINT_PRISMATIC:
            continue
        lo, hi = float(j.lower_limit), float(j.upper_limit)
        j.set_pos(lo + (open_frac * (hi - lo) if jname == cab_info["joint"] else 0.0))
    cab.keep_still()
    for _ in range(5):
        og.sim.step()
    target_obj = env.scene.object_registry("name", diag["target_info"]["name"])
    obstacle_obj = env.scene.object_registry("name", diag["obstacle_info"]["name"])

    surf_min, surf_max = _aabb_np(bundle.surface)
    top_z = float(surf_max[2])

    placements = _layout_both_front_adaptive(
        og, drawer_link, target_obj, obstacle_obj, slide_dir, top_z, robot,
        surf_min, surf_max, obstacle_offside=offside,
    )

    rob_xy = robot.get_position_orientation()[0].cpu().numpy()[:2]
    dist = {}
    for role in ("target", "obstacle"):
        p = placements[role]
        dist[role] = float(np.linalg.norm(np.array(p["xy"], dtype=np.float64) - rob_xy))
        print(f"[bothfront] {task_id} {role:8s}: "
              f"xy=({p['xy'][0]:+.3f},{p['xy'][1]:+.3f}) "
              f"z={p['z']:.3f} top_z={p['top_z']:.3f} in_path={p['in_path']} "
              f"dist_to_robot={dist[role]:.3f}", flush=True)
    print(f"[bothfront] {task_id} sep={placements['sep']:.3f} "
          f"t_hw={placements['t_hw']:.3f} o_hw={placements['o_hw']:.3f}", flush=True)
    def _off_table(p):
        x, y = p["xy"]
        return not (surf_min[0] - 0.02 <= x <= surf_max[0] + 0.02
                    and surf_min[1] - 0.02 <= y <= surf_max[1] + 0.02)

    # off-side obstacle is decorative (never grasped) -> don't gate on its reach
    obs_far = (not offside) and dist["obstacle"] >= 0.85
    if dist["target"] >= 0.80 or obs_far:
        print(f"[bothfront] {task_id}: WARN still unreachable "
              f"(tgt={dist['target']:.3f} obs={dist['obstacle']:.3f}) — ESCALATE",
              flush=True)
    if _off_table(placements["target"]) or _off_table(placements["obstacle"]):
        print(f"[bothfront] {task_id}: WARN object OFF-TABLE after settle "
              f"(tgt={placements['target']['xy']} obs={placements['obstacle']['xy']}) "
              f"— bad settle, RE-RUN", flush=True)

    if dry_run:
        print(f"[bothfront] {task_id}: DRY-RUN — no writes", flush=True)
        return

    scene_path = task_dir / "scene_ep1.json"
    diag_path = task_dir / "diagnostics.jsonl"
    for src in (scene_path, diag_path):
        bak = src.with_suffix(src.suffix + ".bak_bothfront")
        if not bak.exists():
            shutil.copy2(src, bak)

    save_episode_scene(og, env.scene, str(scene_path), exclude_names=set())

    diag["blocker_mode"] = mode
    for role, info_key in (("target", "target_info"), ("obstacle", "obstacle_info")):
        p = placements[role]
        diag[info_key]["placement"] = {
            "xy": [float(p["xy"][0]), float(p["xy"][1])],
            "in_path": bool(p["in_path"]),
            "z": float(p["z"]),
            "top_z": float(p["top_z"]),
        }
    with open(diag_path, "w") as f:
        f.write(json.dumps(diag) + "\n")
    print(f"[bothfront] {task_id}: WROTE scene + diagnostics "
          f"(backups *.bak_bothfront)", flush=True)


def _verify_all() -> None:
    """Offline: assert every far-task is now both-front + reachable."""
    hdr = (f"{'task':12} {'mode':9} {'tgt_in':7} {'obs_in':7} "
           f"{'tgt_dist':9} {'obs_dist':9} ok")
    print(hdr)
    all_ok = True
    for tid in FAR_TASKS:
        task_dir = BENCH_ROOT / tid / "base"
        diag = _load_diag(task_dir)
        scene = json.load(open(task_dir / "scene_ep1.json"))
        reg = scene["state"]["registry"]["object_registry"]

        def xy(name):
            st = reg[name]
            p = (st.get("root_link", {}) or {}).get("pos") or st.get("pos")
            return np.array(p[:2], dtype=np.float64)

        rob = xy("agent_0")
        td = float(np.linalg.norm(xy(diag["target_info"]["name"]) - rob))
        od = float(np.linalg.norm(xy(diag["obstacle_info"]["name"]) - rob))
        mode = diag.get("blocker_mode")
        ti = bool(diag["target_info"]["placement"].get("in_path"))
        oi = bool(diag["obstacle_info"]["placement"].get("in_path"))
        ok = (mode == "both" and ti and oi and td < 0.80 and od < 0.85)
        all_ok = all_ok and ok
        print(f"{tid:12} {str(mode):9} {str(ti):7} {str(oi):7} "
              f"{td:9.3f} {od:9.3f} {'OK' if ok else 'FAIL'}")
    print("ALL OK" if all_ok else "SOME FAILED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=None,
                    help="task_NNNN, or 'all' to fan out a subprocess per task")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print new poses, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="offline check of all 12 converted tasks (no sim)")
    ap.add_argument("--mode", choices=("both", "target"), default="both",
                    help="both=both objects in-path; target=obstacle parked off-side "
                         "(decorative, out of path), only the target blocks (task_0007)")
    args = ap.parse_args()

    if args.verify:
        _verify_all()
        return

    if args.task is None:
        ap.error("--task required (task_NNNN | all) unless --verify")

    if args.task == "all":
        for tid in FAR_TASKS:
            print(f"\n=== [bothfront] subprocess: {tid} ===", flush=True)
            cmd = [sys.executable, "-u", "-m",
                   "maniguard.data.datagen.families.cabinet_bothfront",
                   "--task", tid]
            if args.dry_run:
                cmd.append("--dry-run")
            r = subprocess.run(cmd)
            # exit 139 / -11 is the benign teardown segfault (writes already done)
            if r.returncode not in (0, -11, 139):
                print(f"[bothfront] {tid}: subprocess rc={r.returncode} — CHECK",
                      flush=True)
        return

    _respawn_one(args.task, dry_run=args.dry_run, mode=args.mode)


if __name__ == "__main__":
    main()
