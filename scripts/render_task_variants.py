#!/usr/bin/env python3
"""Load task snapshots, step simulation for a fixed horizon, and save multiview review videos in place."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

from maniguard.envs.frozen_task_runtime import (
    DEFAULT_REVIEW_CAMERA_NAMES,
    FrozenTaskRuntimeSession,
    ReviewVideoRecorder,
    build_env_config,
    configure_review_sensors,
    position_diagnostics_cameras,
    resolve_runtime_python,
    step_idle,
)
from maniguard.envs.perturbation_runtime import apply_runtime_perturbations


PERTURBATION_KINDS = ("object", "position", "semantic", "env")


def _read_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


def _has_task_files(task_dir: Path) -> bool:
    return (task_dir / "scene_ep1.json").is_file() and (task_dir / "diagnostics.jsonl").is_file()


def _base_dir(task_root: Path) -> Path | None:
    base_dir = task_root / "base"
    if base_dir.is_dir() and _has_task_files(base_dir):
        return base_dir
    if _has_task_files(task_root):
        return task_root
    return None


def _selected_camera_names(diagnostics: dict) -> list[str]:
    names = [
        str(entry.get("sensor_name"))
        for entry in (diagnostics.get("cameras") or [])
        if isinstance(entry.get("sensor_name"), str) and entry.get("sensor_name")
    ]
    return names or list(DEFAULT_REVIEW_CAMERA_NAMES)


def collect_render_items(
    root: Path,
    *,
    families: set[str] | None,
    task_ids: set[str] | None,
    kinds: set[str] | None,
    include_base: bool,
) -> list[dict]:
    items = []
    for family_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        family = family_dir.name
        if families and family not in families:
            continue
        for task_root in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            if task_ids and task_root.name not in task_ids:
                continue
            base_dir = _base_dir(task_root)
            if include_base and base_dir is not None:
                items.append({"family": family, "task_id": task_root.name, "kind": "base", "task_dir": base_dir})
            for kind in PERTURBATION_KINDS:
                if kinds and kind not in kinds:
                    continue
                kind_dir = task_root / kind
                if not kind_dir.is_dir():
                    continue
                for variant_dir in sorted(path for path in kind_dir.iterdir() if path.is_dir() and _has_task_files(path)):
                    items.append(
                        {
                            "family": family,
                            "task_id": task_root.name,
                            "kind": kind,
                            "variant_id": variant_dir.name,
                            "task_dir": variant_dir,
                        }
                    )
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--family", dest="families", action="append", default=None)
    parser.add_argument("--task-id", dest="task_ids", action="append", default=None)
    parser.add_argument("--kind", dest="kinds", action="append", default=None)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--skip-existing-logs",
        action="store_true",
        help="Skip task dirs that already contain render_worker_stdout.log.",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=None,
        help="Optional timeout for each legacy worker subprocess render.",
    )
    parser.add_argument(
        "--spawn-worker",
        action="store_true",
        help="Use the legacy one-worker-per-task subprocess path instead of reusing a single runtime session.",
    )
    parser.add_argument("--worker-task-dir", default=None)
    parser.add_argument("--worker-output-path", default=None)
    return parser.parse_args(argv)


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


def _render_one_task(
    task_dir: Path,
    *,
    steps: int,
    video_fps: int,
    headless: bool,
    session: FrozenTaskRuntimeSession | None = None,
) -> dict:
    task_dir = task_dir.resolve()
    results = []
    owns_session = session is None
    runtime_session = session or FrozenTaskRuntimeSession(headless=bool(headless))
    if owns_session:
        runtime_session = runtime_session.__enter__()
    og = runtime_session.og
    assert og is not None
    env = None
    try:
        _reset_runtime(og)
        scene_info = json.loads((task_dir / "scene_ep1.json").read_text(encoding="utf-8"))
        diagnostics = _read_first_jsonl(task_dir / "diagnostics.jsonl")
        camera_names = _selected_camera_names(diagnostics)
        env = og.Environment(configs=build_env_config(scene_info, diagnostics, camera_names=camera_names))
        env.reset()
        configure_review_sensors(env)
        position_diagnostics_cameras(env, og, diagnostics, set_viewer=False)
        with ReviewVideoRecorder(path=task_dir, fps=int(video_fps), camera_names=camera_names) as recorder:
            recorder.record(env, og)
            apply_runtime_perturbations(env, scene_root=task_dir, og=og, video_recorder=recorder)
            step_idle(env, og, steps=int(steps), video_recorder=recorder)
        results.append({"task_dir": str(task_dir), "ok": True})
    finally:
        if env is not None:
            env.close()
        _reset_runtime(og)
        if owns_session:
            runtime_session.__exit__(None, None, None)
    return {"total_items": len(results), "results": results}


def _run_worker(args: argparse.Namespace) -> int:
    if not args.worker_task_dir or not args.worker_output_path:
        raise ValueError("worker mode requires --worker-task-dir and --worker-output-path")
    payload = _render_one_task(
        Path(args.worker_task_dir),
        steps=int(args.steps),
        video_fps=int(args.video_fps),
        headless=bool(args.headless),
    )
    Path(args.worker_output_path).write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_task_dir:
        return _run_worker(args)

    if not args.input_root:
        raise ValueError("--input-root is required unless running in worker mode")
    root = Path(args.input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root not found: {root}")

    items = collect_render_items(
        root,
        families=set(args.families) if args.families else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        kinds=set(args.kinds) if args.kinds else None,
        include_base=bool(args.include_base),
    )
    if args.skip_existing_logs:
        items = [
            item
            for item in items
            if not (Path(item["task_dir"]).resolve() / "render_worker_stdout.log").is_file()
        ]
    if not items:
        raise RuntimeError(f"No render items found under {root}")

    results = []
    if args.spawn_worker:
        for item in items:
            task_dir = Path(item["task_dir"]).resolve()
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
                output_path = Path(handle.name)
            cmd = [
                str(resolve_runtime_python()),
                str(Path(__file__).resolve()),
                "--worker-task-dir",
                str(task_dir),
                "--worker-output-path",
                str(output_path),
                "--steps",
                str(int(args.steps)),
                "--video-fps",
                str(int(args.video_fps)),
            ]
            if args.headless:
                cmd.append("--headless")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=int(args.worker_timeout_seconds) if args.worker_timeout_seconds is not None else None,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_text = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr_text = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                (task_dir / "render_worker_stdout.log").write_text(stdout_text, encoding="utf-8")
                (task_dir / "render_worker_stderr.log").write_text(
                    (stderr_text or "") + f"\n[render] worker timed out after {int(args.worker_timeout_seconds)}s\n",
                    encoding="utf-8",
                )
                results.append(
                    {
                        "task_dir": str(task_dir),
                        "kind": item["kind"],
                        "ok": False,
                        "returncode": -9,
                        "error": "render_worker_timeout",
                        "timeout_seconds": int(args.worker_timeout_seconds),
                    }
                )
            else:
                try:
                    (task_dir / "render_worker_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
                    (task_dir / "render_worker_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
                    if output_path.exists():
                        raw = output_path.read_text(encoding="utf-8").strip()
                        if raw:
                            payload = json.loads(raw)
                            row = dict(payload["results"][0]) if payload.get("results") else {"task_dir": str(task_dir), "ok": False}
                            row["kind"] = item["kind"]
                            row["returncode"] = int(proc.returncode)
                            results.append(row)
                        else:
                            results.append(
                                {
                                    "task_dir": str(task_dir),
                                    "kind": item["kind"],
                                    "ok": False,
                                    "returncode": int(proc.returncode),
                                    "error": "render_worker_no_payload",
                                }
                            )
                    else:
                        results.append({"task_dir": str(task_dir), "kind": item["kind"], "ok": False, "returncode": int(proc.returncode)})
                finally:
                    output_path.unlink(missing_ok=True)
            finally:
                output_path.unlink(missing_ok=True)
    else:
        with FrozenTaskRuntimeSession(headless=bool(args.headless)) as session:
            for item in items:
                task_dir = Path(item["task_dir"]).resolve()
                stdout_log = task_dir / "render_worker_stdout.log"
                stderr_log = task_dir / "render_worker_stderr.log"
                try:
                    payload = _render_one_task(
                        task_dir,
                        steps=int(args.steps),
                        video_fps=int(args.video_fps),
                        headless=bool(args.headless),
                        session=session,
                    )
                    row = dict(payload["results"][0]) if payload.get("results") else {"task_dir": str(task_dir), "ok": False}
                    row["kind"] = item["kind"]
                    row["returncode"] = 0
                    results.append(row)
                    stdout_log.write_text(
                        json.dumps(
                            {
                                "task_dir": str(task_dir),
                                "kind": item["kind"],
                                "ok": True,
                                "mode": "in_process_session_reuse",
                                "steps": int(args.steps),
                                "video_fps": int(args.video_fps),
                            },
                            indent=2,
                            ensure_ascii=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    stderr_log.write_text("", encoding="utf-8")
                except Exception:
                    results.append(
                        {
                            "task_dir": str(task_dir),
                            "kind": item["kind"],
                            "ok": False,
                            "returncode": 1,
                            "error": "in_process_render_failed",
                        }
                    )
                    stderr_log.write_text(traceback.format_exc(), encoding="utf-8")
                    stdout_log.write_text(
                        json.dumps(
                            {
                                "task_dir": str(task_dir),
                                "kind": item["kind"],
                                "ok": False,
                                "mode": "in_process_session_reuse",
                            },
                            indent=2,
                            ensure_ascii=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

    print(json.dumps({"total_items": len(results), "results": results}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
