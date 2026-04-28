#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sentinel.envs.frozen_task_runtime import (
    FrozenTaskRuntimeSession,
    build_env_config,
    save_scene_snapshot,
)
from sentinel.utils.goal_region import (
    build_goal_region_spec,
    build_task_prompt,
    family_uses_goal_region,
    infer_family_from_diagnostics,
    inject_goal_region_metadata,
    remove_goal_region_from_scene_info,
    resolve_goal_region_entities,
    restore_robot_entries,
    spawn_goal_region_marker,
)
from sentinel.utils.backfill_resume import has_task_snapshot, plan_family_resume


def _read_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl_record(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True) + "\n", encoding="utf-8")


def _iter_task_dirs(root: Path):
    for scene_file in sorted(root.rglob("scene_ep1.json")):
        task_dir = scene_file.parent
        if (task_dir / "diagnostics.jsonl").is_file():
            yield task_dir


def _selected_task_dirs(root: Path, families: set[str], task_ids: set[str]) -> list[Path]:
    selected = []
    for task_dir in _iter_task_dirs(root):
        family = task_dir.parent.name
        if families and family not in families:
            continue
        if task_ids and task_dir.name not in task_ids:
            continue
        selected.append(task_dir)
    return selected


def _summarize_outputs(
    *,
    input_root: Path,
    output_root: Path,
    selected_task_dirs: list[Path],
    resume_actions: list,
) -> dict:
    results = []
    completed_tasks = 0
    goal_region_tasks = 0
    for task_dir in selected_task_dirs:
        family = task_dir.parent.name
        output_task_dir = output_root / family / task_dir.name
        row = {
            "task_dir": str(task_dir),
            "family": family,
            "output_task_dir": str(output_task_dir),
            "completed": False,
            "goal_region_added": False,
            "prompt_updated": False,
        }
        if has_task_snapshot(output_task_dir):
            completed_tasks += 1
            row["completed"] = True
            diag = _read_first_jsonl(output_task_dir / "diagnostics.jsonl")
            row["goal_region_added"] = bool(diag.get("goal_region"))
            row["prompt_updated"] = bool(diag.get("prompt"))
            if row["goal_region_added"]:
                goal_region_tasks += 1
                goal_region = diag.get("goal_region") or {}
                if isinstance(goal_region, dict) and goal_region.get("marker_name"):
                    row["marker_name"] = str(goal_region["marker_name"])
        results.append(row)

    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total_tasks": len(selected_task_dirs),
        "completed_tasks": completed_tasks,
        "goal_region_tasks": goal_region_tasks,
        "resume_actions": [action.to_json() for action in resume_actions],
        "results": results,
    }


def _reset_runtime(og) -> None:
    if og is None or getattr(og, "sim", None) is None:
        return
    try:
        viewer_camera = getattr(og.sim, "viewer_camera", None)
        if viewer_camera is not None:
            viewer_camera.active_camera_path = "/OmniverseKit_Persp"
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


def _copy_task_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for video_path in dst.glob("rollout_*.mp4"):
        video_path.unlink()
    for log_path in dst.glob("render_worker_*.log"):
        log_path.unlink()


def _backfill_one(task_dir: Path, output_task_dir: Path, session: FrozenTaskRuntimeSession) -> dict:
    original_scene = json.loads((task_dir / "scene_ep1.json").read_text(encoding="utf-8"))
    original_diag = _read_first_jsonl(task_dir / "diagnostics.jsonl")
    scene_info, diagnostics = remove_goal_region_from_scene_info(original_scene, original_diag)

    family = infer_family_from_diagnostics(diagnostics)
    diagnostics["pipeline"] = family

    _copy_task_dir(task_dir, output_task_dir)

    if not family_uses_goal_region(family):
        diagnostics["prompt"] = build_task_prompt(scene_info, diagnostics, goal_region=None)
        _write_json(output_task_dir / "scene_ep1.json", scene_info)
        _write_jsonl_record(output_task_dir / "diagnostics.jsonl", diagnostics)
        return {
            "task_dir": str(task_dir),
            "family": family,
            "goal_region_added": False,
            "prompt_updated": True,
        }

    entities = resolve_goal_region_entities(scene_info, diagnostics)
    if entities is None:
        raise RuntimeError(f"Could not resolve goal-region entities for {task_dir}")

    og = session.og
    assert og is not None
    _reset_runtime(og)
    env = og.Environment(configs=build_env_config(scene_info, diagnostics, camera_names=None))
    env.reset()
    try:
        spec = build_goal_region_spec(
            env,
            diagnostics,
            family=entities.family,
            target_name=entities.target_name,
            support_name=entities.support_name,
            pack_object_names=entities.pack_object_names,
        )
        spawn_goal_region_marker(env, spec)
        og.sim.step()

        scene_path = output_task_dir / "scene_ep1.json"
        save_scene_snapshot(env, scene_path)
        saved_scene = json.loads(scene_path.read_text(encoding="utf-8"))
        prompt = build_task_prompt(saved_scene, diagnostics, goal_region=spec.to_json())
        saved_scene = restore_robot_entries(saved_scene, scene_info)
        saved_scene = inject_goal_region_metadata(saved_scene, spec, prompt)
        diagnostics["goal_region"] = spec.to_json()
        diagnostics["prompt"] = prompt
        _write_json(scene_path, saved_scene)
        _write_jsonl_record(output_task_dir / "diagnostics.jsonl", diagnostics)
        return {
            "task_dir": str(task_dir),
            "family": family,
            "goal_region_added": True,
            "prompt_updated": True,
            "marker_name": spec.marker_name,
        }
    finally:
        env.close()
        _reset_runtime(og)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill goal-region markers and prompts into frozen task snapshots.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--family", dest="families", action="append", default=None)
    parser.add_argument("--task-id", dest="task_ids", action="append", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    families = set(args.families or [])
    task_ids = set(args.task_ids or [])

    selected_task_dirs = _selected_task_dirs(input_root, families, task_ids)
    if not selected_task_dirs:
        raise RuntimeError(f"No tasks selected under {input_root}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        tasks_to_run = selected_task_dirs
        resume_actions = []
    else:
        tasks_to_run, resume_actions = plan_family_resume(selected_task_dirs, output_root)

    _write_json(
        output_root / "goal_region_backfill_manifest.json",
        _summarize_outputs(
            input_root=input_root,
            output_root=output_root,
            selected_task_dirs=selected_task_dirs,
            resume_actions=resume_actions,
        ),
    )

    with FrozenTaskRuntimeSession(headless=bool(args.headless)) as session:
        for task_dir in tasks_to_run:
            family = task_dir.parent.name
            dst = output_root / family / task_dir.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            _backfill_one(task_dir, dst, session)
            _write_json(
                output_root / "goal_region_backfill_manifest.json",
                _summarize_outputs(
                    input_root=input_root,
                    output_root=output_root,
                    selected_task_dirs=selected_task_dirs,
                    resume_actions=resume_actions,
                ),
            )

    summary = _summarize_outputs(
        input_root=input_root,
        output_root=output_root,
        selected_task_dirs=selected_task_dirs,
        resume_actions=resume_actions,
    )
    _write_json(output_root / "goal_region_backfill_manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
