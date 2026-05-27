"""Replay captured tasks in EMPTY scenes with only support + task objects.

Given a task folder from a benchmark dataset (containing ``scene_ep<N>.json``
and ``diagnostics.jsonl``), this script reconstructs each task in a bare
OmniGibson ``Scene`` (floor plane only) with:

  * the support surface (e.g. ``desk_qpuflh_2``), fixed_base, at its
    snapshot scale (per-axis, copied from ``init_info.args.scale``)
  * every object listed in ``selection.spawn_specs`` (target / fragile /
    clutter / stack / transfer), at its snapshot scale, posed at its
    final pose from the scene snapshot
  * the Franka robot at its dump pose + joint configuration
  * the green goal-region sphere from ``diagnostics.goal_region``
  * the four diagnostics cameras (opposite / left / right / left_shoulder)

All other scene objects (walls, ceilings, other furniture) are dropped.
Absolute world poses are preserved -- the goal region's ``center_world``
and the camera eye/lookat all stay consistent.

Outputs in ``<task-dir>/replay_empty/``:
  * ``diagnostics.jsonl``         -- copy of the source diagnostics row
  * ``scene_ep<N>_replay.json``   -- full snapshot of the rebuilt empty scene
  * ``rollout_<view>_ep<N>.mp4``  -- per-camera replays (if --save-video)

Usage:
    # Single task
    python -m tools.replay_empty_from_dataset \\
        --task-dir datasets/.../table/task_0000 --steps 60 --save-video

    # All tasks in a family folder
    python -m tools.replay_empty_from_dataset \\
        --root-dir datasets/.../table --steps 60 --save-video
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-dir", type=Path,
                   help="Single task folder (contains scene_ep*.json + diagnostics.jsonl)")
    g.add_argument("--root-dir", type=Path,
                   help="Parent folder containing task subfolders (each with scene_ep*.json + diagnostics.jsonl)")
    p.add_argument("--episode", type=int, default=1,
                   help="Episode number to replay (default: 1)")
    p.add_argument("--steps", type=int, default=60,
                   help="Number of idle physics steps to run (default: 60)")
    p.add_argument("--save-video", action="store_true",
                   help="Record per-camera mp4s like the original dataset")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tasks whose output dir already has a snapshot")
    p.add_argument("--gui", action="store_true",
                   help="Run with GUI (default: headless)")
    p.add_argument("--scene-subdir", type=str, default="",
                   help="Subdirectory under each task containing scene_ep*.json + "
                        "diagnostics.jsonl (default: task root). Use 'env' for the "
                        "post-merge dataset layout.")
    p.add_argument("--output-subdir", type=str, default="replay_empty",
                   help="Subdirectory under each task to write the replay into "
                        "(default: 'replay_empty'). Use 'base' for ep1 in the "
                        "post-merge dataset layout, or 'table' for ep>=2.")
    return p.parse_args()


def _load_diagnostics_row(task_dir: Path, episode: int) -> dict[str, Any]:
    path = task_dir / "diagnostics.jsonl"
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if int(row.get("episode", 0)) == episode:
            return row
    raise ValueError(f"No diagnostics entry for episode {episode} in {path}")


def _load_scene_info(task_dir: Path, episode: int) -> dict[str, Any]:
    primary = task_dir / f"scene_ep{episode}.json"
    if primary.is_file():
        return json.loads(primary.read_text())
    fallback = task_dir / f"scene_ep{episode}_replay.json"
    if fallback.is_file():
        return json.loads(fallback.read_text())
    raise FileNotFoundError(f"No scene_ep{episode}.json or scene_ep{episode}_replay.json in {task_dir}")


_TASK_OBJ_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_\d+$")
# Scheme B: role-prefixed names ending in "_ep<N>_<idx>" — used by some
# lid_transport tasks whose env/ scene_ep1.json keeps the full
# 200+ object scene tree and disambiguates task objects via role/episode.
_SCHEME_B_PATTERN = re.compile(r"_ep\d+_\d+$")


def _is_task_object_name(name: str, category: str) -> bool:
    if not _TASK_OBJ_PATTERN.match(name):
        return False
    prefix, _, tail = name.rpartition("_")
    if not tail.isdigit():
        return False
    # Scheme A: auto-named by category, e.g. ``lid_43`` (category="lid").
    # Prefix matches category exactly.
    if prefix == category:
        return True
    # Scheme B: ``<role>_<category>_ep<N>_<idx>``, e.g.
    # ``target_milk_carton_ep1_1`` (category="milk_carton"). The
    # ``_ep<N>_<idx>`` suffix marks task-object names that the dataset
    # generator inserts; background furniture never carries it.
    if _SCHEME_B_PATTERN.search(name):
        # Verify the category appears as a contiguous token sequence
        # in the name so we don't match unrelated objects that happen
        # to end in ``_ep1_1``.
        cat_toks = category.split("_")
        name_toks = name.split("_")
        for i in range(len(name_toks) - len(cat_toks) + 1):
            if name_toks[i:i + len(cat_toks)] == cat_toks:
                return True
    return False


def _identify_task_objects(
    scene_info: dict[str, Any],
    diagnostics: dict[str, Any],
) -> list[str]:
    """Return [support, task_obj_1, task_obj_2, ...].

    spawn_specs records the pipeline's *intent*, but a spawn can fail
    silently (placement / physics gate), so the snapshot may have fewer
    instances than requested. Trust the snapshot: pull every object that
    matches a spawn-spec category AND uses the task-object name pattern
    (``<category>_<digits>``, distinct from scene-furniture
    ``<category>_<model>_<id>``).
    """
    init_info = scene_info["objects_info"]["init_info"]
    surface = str(diagnostics["surface"])
    if surface not in init_info:
        raise ValueError(f"Surface {surface!r} not found in scene snapshot")

    spec_categories = {spec["category"] for spec in diagnostics["selection"]["spawn_specs"]}

    names: list[str] = [surface]
    used: set[str] = {surface}
    for n, info in init_info.items():
        if n in used:
            continue
        cat = info.get("args", {}).get("category")
        if cat in spec_categories and _is_task_object_name(n, cat):
            names.append(n)
            used.add(n)
    return names


def _build_object_cfg(name: str, scene_info: dict[str, Any], fixed_base: bool) -> dict[str, Any]:
    init_info = scene_info["objects_info"]["init_info"][name]
    reg = scene_info["state"]["registry"]["object_registry"][name]
    args = init_info.get("args", {})
    scale = args.get("scale")
    scale = [float(v) for v in scale] if scale is not None else [1.0, 1.0, 1.0]
    return {
        "type": "DatasetObject",
        "name": name,
        "category": args["category"],
        "model": args["model"],
        "scale": scale,
        "fixed_base": bool(fixed_base),
        "position": [float(v) for v in reg["root_link"]["pos"]],
        "orientation": [float(v) for v in reg["root_link"]["ori"]],
    }


def _build_robot_cfg(robot_setup: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "type": "FrankaPanda",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "fixed_base": True,
        "position": list(robot_setup["position"]) if robot_setup.get("position") else [0.0, 0.0, 0.0],
        "orientation": list(robot_setup["orientation"]) if robot_setup.get("orientation") else [0.0, 0.0, 0.0, 1.0],
        "controller_config": {
            "arm_0": {
                "name": "JointController",
                "motor_type": "position",
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
            },
        },
    }
    if robot_setup.get("reset_joint_pos") is not None:
        cfg["reset_joint_pos"] = list(robot_setup["reset_joint_pos"])
    return cfg


def _find_task_dirs(root: Path, episode: int, scene_subdir: str = "") -> list[Path]:
    out: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        scene_root = entry / scene_subdir if scene_subdir else entry
        if (scene_root / f"scene_ep{episode}.json").is_file() and (scene_root / "diagnostics.jsonl").is_file():
            out.append(entry)
    return out


def _reset_runtime(og) -> None:
    """Stop the sim and clear the scene so the next task starts fresh."""
    if og is None or getattr(og, "sim", None) is None:
        return
    try:
        viewer = getattr(og.sim, "viewer_camera", None)
        if viewer is not None:
            viewer.active_camera_path = "/OmniverseKit_Persp"
    except Exception:
        pass
    try:
        og.sim.stop()
    except Exception:
        pass
    try:
        og.clear()
    except Exception:
        pass


def _replay_one_task(task_dir: Path, og, *, args) -> bool:
    """Build the empty scene + replay loop for one task. Returns True on success."""
    from sentinel.envs.frozen_task_runtime import (
        ReviewVideoRecorder,
        configure_review_sensors,
        extract_scene_robot_setup,
        position_diagnostics_cameras,
        save_scene_snapshot,
        step_idle,
    )
    from sentinel.utils.camera_setup import build_external_camera_configs
    from sentinel.utils.goal_region import GoalRegionSpec, spawn_goal_region_marker

    scene_root = task_dir / args.scene_subdir if args.scene_subdir else task_dir
    out_dir = (task_dir / args.output_subdir).resolve()
    if args.skip_existing and (out_dir / f"scene_ep{args.episode}_replay.json").is_file():
        print(f"[Replay] SKIP {task_dir.name} (already done)", flush=True)
        return True
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _load_diagnostics_row(scene_root, args.episode)
    scene_info = _load_scene_info(scene_root, args.episode)
    task_names = _identify_task_objects(scene_info, diagnostics)
    print(f"[Replay] {task_dir.name}: {len(task_names)} task objects "
          f"(surface={task_names[0]})", flush=True)

    object_cfgs = [_build_object_cfg(task_names[0], scene_info, fixed_base=True)]
    object_cfgs += [_build_object_cfg(n, scene_info, fixed_base=False) for n in task_names[1:]]

    robot_setup = extract_scene_robot_setup(scene_info)
    if robot_setup is None:
        raise RuntimeError("No robot found in scene snapshot")
    robot_cfg = _build_robot_cfg(robot_setup)

    camera_names = [c["sensor_name"] for c in diagnostics.get("cameras", []) if c.get("sensor_name")]
    external_sensors = build_external_camera_configs(
        names=camera_names, resolution=(720, 1280), modalities=("rgb",)
    )

    env_cfg = {
        "scene": {"type": "Scene"},
        "robots": [robot_cfg],
        "objects": object_cfgs,
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": 20,
            "rendering_frequency": 20,
            "physics_frequency": 120,
            "external_sensors": external_sensors,
        },
    }

    _reset_runtime(og)
    env = og.Environment(configs=env_cfg)
    try:
        env.reset()

        # Re-apply object poses + keep_still (env.reset can perturb).
        # Fixed-base objects already have their pose baked in at construction;
        # re-applying it post-reset can detach the world fixed-joint on some
        # Isaac builds and let the surface fall.
        for cfg in object_cfgs:
            obj = env.scene.object_registry("name", cfg["name"])
            if obj is None:
                print(f"[Replay]   WARN: {cfg['name']} not in registry post-reset", flush=True)
                continue
            if not cfg.get("fixed_base"):
                obj.set_position_orientation(
                    position=cfg["position"], orientation=cfg["orientation"],
                )
            if hasattr(obj, "keep_still"):
                obj.keep_still()

        robot = env.robots[0]
        if robot_setup.get("position") is not None:
            robot.set_position_orientation(
                position=robot_setup["position"], orientation=robot_setup["orientation"],
            )
        if hasattr(robot, "keep_still"):
            robot.keep_still()
        og.sim.step()

        gr_payload = diagnostics.get("goal_region")
        if gr_payload is not None:
            spawn_goal_region_marker(env, GoalRegionSpec.from_json(gr_payload))
            og.sim.step()

        configure_review_sensors(env)
        position_diagnostics_cameras(env, og, diagnostics, set_viewer=True)

        if args.save_video:
            with ReviewVideoRecorder(
                path=out_dir, fps=args.video_fps, camera_names=camera_names,
            ) as recorder:
                recorder.record(env, og)
                step_idle(env, og, steps=args.steps, video_recorder=recorder)
        else:
            step_idle(env, og, steps=args.steps)

        save_scene_snapshot(env, out_dir / f"scene_ep{args.episode}_replay.json")

        # Mirror the source diagnostics row alongside the replay.
        shutil.copy2(scene_root / "diagnostics.jsonl", out_dir / "diagnostics.jsonl")
        return True
    finally:
        try:
            env.close()
        except Exception:
            pass


def _init_omnigibson(headless: bool):
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if headless:
        gm.HEADLESS = True
    import omnigibson as og
    return og


def _run_subprocess_per_task(task_dirs: list[Path], args: argparse.Namespace) -> None:
    """Spawn a fresh python process for each task.

    OmniGibson's USD/Kit state can't be cleanly reset within one process --
    creating a second Environment after closing the first leaves dangling
    USD camera prims and matrix math errors. Single-process-per-task is the
    only reliable path; we pay ~30s of Kit startup per task in exchange.
    """
    import os
    import subprocess

    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    script = Path(__file__).resolve()
    base_cmd = [
        sys.executable, str(script),
        "--episode", str(args.episode),
        "--steps", str(args.steps),
        "--video-fps", str(args.video_fps),
        "--scene-subdir", args.scene_subdir,
        "--output-subdir", args.output_subdir,
    ]
    if args.save_video:
        base_cmd.append("--save-video")
    if args.skip_existing:
        base_cmd.append("--skip-existing")
    if args.gui:
        base_cmd.append("--gui")

    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("OMNIGIBSON_HEADLESS", "0" if args.gui else "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    for i, td in enumerate(task_dirs, start=1):
        print(f"\n[Replay] === Task {i}/{len(task_dirs)}: {td.name} ===", flush=True)
        if args.skip_existing and (td / args.output_subdir / f"scene_ep{args.episode}_replay.json").is_file():
            print(f"[Replay] SKIP {td.name} (already done)", flush=True)
            successes.append(td.name)
            continue
        cmd = base_cmd + ["--task-dir", str(td)]
        proc = subprocess.run(cmd, env=env)
        # OmniGibson commonly segfaults during shutdown after a clean save;
        # treat output presence (snapshot) as the actual success signal.
        ok = (td / args.output_subdir / f"scene_ep{args.episode}_replay.json").is_file()
        if ok:
            successes.append(td.name)
        else:
            failures.append((td.name, f"exit_code={proc.returncode}, no snapshot produced"))
            print(f"[Replay] FAIL {td.name}: exit_code={proc.returncode}", flush=True)

    print(f"\n[Replay] Summary: {len(successes)} ok, {len(failures)} failed")
    for name, err in failures:
        print(f"  FAIL {name}: {err}")


def main() -> None:
    args = parse_args()

    # Batch mode: dispatch one subprocess per task. OmniGibson cannot
    # cleanly tear down + reload assets within a single process.
    if args.root_dir is not None:
        root = args.root_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"root-dir not found: {root}")
        task_dirs = _find_task_dirs(root, args.episode, args.scene_subdir)
        if not task_dirs:
            raise SystemExit(f"No task folders with scene_ep{args.episode}.json + diagnostics.jsonl under {root}")
        print(f"[Replay] Found {len(task_dirs)} tasks under {root}", flush=True)
        _run_subprocess_per_task(task_dirs, args)
        return

    # Single-task mode: run inline in this process.
    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"task-dir not found: {task_dir}")
    og = _init_omnigibson(headless=not args.gui)
    try:
        _replay_one_task(task_dir, og, args=args)
    except Exception as exc:
        print(f"[Replay] FAIL {task_dir.name}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
