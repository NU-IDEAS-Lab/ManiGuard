"""Single-task base finalizer for ManiGuard-Bench.

``finalize_base_task`` reads a 6fam-base task's ``base/`` snapshot (READ-ONLY), enforces the
canonical mount (``base_z = support_top + ROBOT_MOUNT_OFFSET``, keeping XY+yaw) + the canonical
init pose (``BENCH_INIT_QPOS``), saves the clean init snapshot to the NEW maniguard-bench output
dir, renders the 4 idle-step review videos via the shared ``render_views``, and self-checks the
arm-hold / object stability / round-trip. Returns a manifest row.

Data isolation (design doc §1): the 6fam-base source is never modified; output always uses the
canonical filename ``scene_ep{episode}.json`` (so a source ``scene_ep1_replay.json`` is read but
the output is ``scene_ep1.json``). The env is built with the SOURCE robot stripped and replaced
by ONE uniform canonical robot (FrankaPanda+longfinger + the bench's declared controller/grasping
defaults, design doc §2 layer ②) so every base snapshot carries an identical, correct robot config
— this is a source-design choice, not an eval-compat patch (eval reloads controllers after load).
"""
from __future__ import annotations

import json
from pathlib import Path

from maniguard.data.bench_builder.render import (
    DEFAULT_FPS,
    DEFAULT_N_FRAMES,
    DEFAULT_RESOLUTION,
)

ARM_DRIFT_TOL = 1e-2   # rad — arm must hold the baked pose under physics (else it was flung)
BASE_Z_TOL = 1e-3      # m — saved base z must match the enforced mount
OBJ_DISP_WARN = 0.02   # m — object settle beyond this is flagged (possible instability/penetration)


def _load_diagnostics(base_dir: Path) -> dict:
    raw = (base_dir / "diagnostics.jsonl").read_text(encoding="utf-8").lstrip()
    return json.JSONDecoder().raw_decode(raw)[0]


def _source_scene_file(src_base_dir: Path, episode: int) -> Path:
    """Canonical first, then the LidTransport/replay ``_replay`` variant (e.g. stack)."""
    for name in (f"scene_ep{episode}.json", f"scene_ep{episode}_replay.json"):
        p = src_base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no scene_ep{episode}[_replay].json in {src_base_dir}")


def _support_top(env, diag: dict) -> tuple[float, str]:
    """Top-of-surface z. jar/cabinet record ``surface_info.top_z`` (their ``surface`` field is a
    ``category/model`` string, not an object name); the BasePipeline families (clutter/lid/stack/
    dusty) name the surface object directly -> AABB max z.
    """
    si = diag.get("surface_info")
    if isinstance(si, dict) and si.get("top_z") is not None:
        return float(si["top_z"]), "surface_info.top_z"
    surf = env.scene.object_registry("name", diag.get("surface"))
    if surf is not None:
        return float(surf.aabb[1][2]), "surface.aabb_max_z"
    raise ValueError(
        f"cannot resolve support surface {diag.get('surface')!r} "
        f"(no surface_info.top_z and not in object_registry)"
    )


def _robot_from_snapshot(header: dict) -> tuple[list | None, float | None]:
    """(joint_pos, base_z) of the FrankaPanda robot read back out of a saved snapshot JSON."""
    init = header.get("objects_info", {}).get("init_info", {})
    reg = header.get("state", {}).get("registry", {}).get("object_registry", {})
    for name, info in init.items():
        cm = info.get("class_module", "")
        cn = info.get("class_name", "")
        if cm.startswith("omnigibson.robots.") or cn.endswith(("Robot", "Mounted", "Panda")):
            r = reg.get(name, {})
            pos = r.get("root_link", {}).get("pos")
            base_z = float(pos[2]) if pos else None
            return r.get("joint_pos"), base_z
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
    """Finalize one base task: enforce mount + pose, save to ``out_base_dir``, render, self-check.

    Returns a manifest row (dict). Never writes to ``src_base_dir``.
    """
    import numpy as np
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

    src_base_dir = Path(src_base_dir)
    out_base_dir = Path(out_base_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)
    task_name = src_base_dir.parent.name

    diag = _load_diagnostics(src_base_dir)
    scene_file = _source_scene_file(src_base_dir, episode)
    scene_info = json.loads(scene_file.read_text(encoding="utf-8"))

    # --- build env: scene from the snapshot, SOURCE robot stripped + replaced by ONE uniform
    #     canonical robot (FrankaPanda+longfinger + the bench's declared controller/grasping
    #     defaults, pulled from the single shared CONTROLLER_PRESETS). Bakes a uniform correct
    #     robot config into every base snapshot (design doc §2 layer ②) — NOT an eval-compat
    #     patch (eval reloads controllers after load). ---
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

    # --- enforce mount: base_z = support_top + offset, keep XY + yaw ---
    support_top, top_src = _support_top(env, diag)
    rp_t, rq_t = robot.get_position_orientation()
    rp = np.asarray(rp_t.cpu().numpy() if hasattr(rp_t, "cpu") else rp_t, dtype=np.float64)
    base_z_before = float(rp[2])
    base_z_after = float(support_top + ROBOT_MOUNT_OFFSET)
    robot.set_position_orientation(
        position=[float(rp[0]), float(rp[1]), base_z_after], orientation=rq_t
    )

    # --- bake canonical init pose (held by the stiff Isaac drive through the idle-step render) ---
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()

    # --- save the clean init snapshot to OUTPUT (canonical filename), BEFORE idle-stepping ---
    out_scene = out_base_dir / f"scene_ep{episode}.json"
    env.scene.save(json_path=str(out_scene))

    # --- render idle-step videos + stability stats; cameras re-stamped into the diag copy ---
    diag2, stats = render_views(
        env, diag, out_base_dir,
        episode=episode, n_frames=n_frames, fps=fps, resolution=resolution, mode="idle_step",
    )
    # provenance block: records HOW this base was finalized (self-documenting output + lets the
    # offline validator check mount without re-deriving support_top in sim).
    diag2["bench"] = {
        "finalized": True,
        "controller_preset": BENCH_CONTROLLER_PRESET,
        "grasping_mode": BENCH_GRASPING_MODE,
        "mount_offset": ROBOT_MOUNT_OFFSET,
        "support_top": round(support_top, 4),
        "support_top_src": top_src,
        "base_z": round(base_z_after, 4),
        "base_z_before": round(base_z_before, 4),
        "mount_shift": round(base_z_after - base_z_before, 4),
        "init_pose": list(BENCH_INIT_QPOS),
        "arm_drift": stats["arm_drift"],
        "obj_disp": stats["obj_disp"],
    }
    # write OUTPUT diagnostics: render_views replaced 'cameras'; every other field is preserved
    (out_base_dir / "diagnostics.jsonl").write_text(json.dumps(diag2) + "\n", encoding="utf-8")

    # --- readback self-check from the saved snapshot ---
    header = json.loads(out_scene.read_text(encoding="utf-8"))
    jp, base_z_rb = _robot_from_snapshot(header)
    pose_ok = jp is not None and len(jp) == len(BENCH_INIT_QPOS) and max(
        abs(float(a) - b) for a, b in zip(jp, BENCH_INIT_QPOS)
    ) < 1e-2
    basez_ok = base_z_rb is not None and abs(base_z_rb - base_z_after) < BASE_Z_TOL
    n_mp4 = len(list(out_base_dir.glob(f"rollout_*_ep{episode}.mp4")))

    warnings: list[str] = []
    if stats["arm_drift"] >= ARM_DRIFT_TOL:
        warnings.append(f"arm_drift={stats['arm_drift']:.3g}>={ARM_DRIFT_TOL}")
    if stats["obj_disp"] >= OBJ_DISP_WARN:
        warnings.append(f"obj_disp={stats['obj_disp']:.3g}>={OBJ_DISP_WARN}")
    if not pose_ok:
        warnings.append("pose readback != A")
    if not basez_ok:
        warnings.append(f"base_z readback {base_z_rb} != {base_z_after:.4f}")
    if n_mp4 != 4:
        warnings.append(f"n_mp4={n_mp4} != 4")

    # fail = something structurally wrong; warn = stable but flagged (e.g. obj settled a bit)
    hard_fail = (stats["arm_drift"] >= ARM_DRIFT_TOL) or (not pose_ok) or (not basez_ok) or (n_mp4 != 4)
    status = "fail" if hard_fail else ("warn" if warnings else "ok")

    return {
        "task": task_name,
        "family": family,
        "surface": diag.get("surface"),
        "support_top": round(support_top, 4),
        "support_top_src": top_src,
        "base_z_before": round(base_z_before, 4),
        "base_z_after": round(base_z_after, 4),
        "mount_shift": round(base_z_after - base_z_before, 4),
        "pose": "A" if pose_ok else "DIFF",
        "arm_drift": stats["arm_drift"],
        "obj_disp": stats["obj_disp"],
        "n_mp4": n_mp4,
        "status": status,
        "warnings": warnings,
    }
