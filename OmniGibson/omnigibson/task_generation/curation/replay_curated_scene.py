#!/usr/bin/env python
"""Replay a curated clutter scene snapshot and record a deterministic rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omnigibson.task_generation.curation.curation_manifest import load_curation_manifest
from omnigibson.task_generation.pipeline_common import (
    append_jsonl,
    build_descriptors,
    build_task_config,
    build_task_object_sets,
    close_video_writer,
    discover_best_surface,
    get_scope_obj,
    init_video_writer,
    pipeline_exit,
    record_frame,
    run_ltl_rollout,
    setup_run_dir,
)


def make_parser():
    parser = argparse.ArgumentParser(description="Replay a curated clutter scene snapshot")
    parser.add_argument("--scene-model", required=True)
    parser.add_argument("--curation-manifest", required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter-scale", type=float, default=0.01)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--debug-jsonl", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-camera-eye", nargs=3, type=float, default=None)
    parser.add_argument("--video-camera-lookat", nargs=3, type=float, default=None)
    parser.add_argument("--print-manifest-camera-snippet", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--showcase-gui", action="store_true")
    return parser


def _expected_episode_video_path(base_path: str | None, episode: int) -> str | None:
    if not base_path:
        return None
    path = Path(base_path)
    stem = path.with_suffix("") if path.suffix == ".mp4" else path
    return str((stem.parent / f"{stem.name}_ep{episode + 1}.mp4").resolve())


def _artifact_state(path: str | None) -> tuple[bool, int | None]:
    if not path:
        return (False, None)
    p = Path(path)
    if not p.exists():
        return (False, None)
    return (True, p.stat().st_size)


def _print_artifact_result(label: str, path: str | None, before: tuple[bool, int | None]) -> None:
    if not path:
        return
    exists_after, size_after = _artifact_state(path)
    existed_before, size_before = before
    if not existed_before and exists_after:
        print(f"[Replay] Created {label}: {path} ({size_after} bytes)")
        return
    if existed_before and exists_after and size_after != size_before:
        print(f"[Replay] Updated {label}: {path} ({size_before} -> {size_after} bytes)")
        return
    if exists_after:
        print(f"[Replay] Reused existing {label}: {path} ({size_after} bytes)")
        return
    print(f"[Replay] No {label} produced at: {path}")


def resolve_effective_camera_config(entry, args) -> dict[str, tuple[float, float, float] | None]:
    eye = tuple(args.video_camera_eye) if args.video_camera_eye is not None else entry.video_camera_eye
    lookat = tuple(args.video_camera_lookat) if args.video_camera_lookat is not None else entry.video_camera_lookat
    return {"eye": eye, "lookat": lookat}


def expected_replay_artifact_paths(args, episode: int) -> tuple[str | None, str | None]:
    video_path = _expected_episode_video_path(args.save_video, episode) if args.save_video else None
    diagnostics_path = str(Path(args.debug_jsonl).resolve()) if args.debug_jsonl else None
    return video_path, diagnostics_path


def _build_scene_only_config(scene_model: str, snapshot_path: str) -> dict:
    return {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "scene_file": snapshot_path,
            "scene_instance": None,
        },
        "task": {
            "type": "DummyTask",
        },
        "robots": [],
    }


def _print_manifest_camera_snippet(scene_model: str, entry, camera_override) -> None:
    print("[Replay] Manifest camera snippet:")
    print("{")
    print(f'  "{scene_model}": {{')
    if camera_override["eye"] is not None:
        print(f'    "video_camera_eye": {list(camera_override["eye"])},')
    if camera_override["lookat"] is not None:
        print(f'    "video_camera_lookat": {list(camera_override["lookat"])},')
    print(f'    "video_viewer_only": {str(entry.video_viewer_only).lower()}')
    print("  }")
    print("}")


def _scene_only_camera_override(env, entry, camera_override):
    if camera_override["eye"] is not None and camera_override["lookat"] is not None:
        return camera_override

    forced_name = entry.surface_name if entry.surface_name else None
    forced_category = entry.support_category if entry.support_category else None
    _, support_obj = discover_best_surface(env, forced_name=forced_name, forced_category=forced_category)
    aabb_min, aabb_max = support_obj.aabb
    support_center = [
        0.5 * (float(aabb_min[0]) + float(aabb_max[0])),
        0.5 * (float(aabb_min[1]) + float(aabb_max[1])),
        float(aabb_max[2]),
    ]
    return {
        "eye": camera_override["eye"] or (
            support_center[0] - 1.0,
            support_center[1] - 1.1,
            support_center[2] + 0.6,
        ),
        "lookat": camera_override["lookat"] or (
            support_center[0],
            support_center[1],
            support_center[2] + 0.1,
        ),
    }


def _run_snapshot_only_viewer_rollout(env, args, scene_model, snapshot_path, entry, episode, camera_override):
    import math
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    import torch as th

    args.video_viewer_only = True
    if args.save_video:
        center = [float(v) for v in camera_override["lookat"]]
        cam_pos = [float(v) for v in camera_override["eye"]]
        d = np.asarray([center[i] - cam_pos[i] for i in range(3)], dtype=np.float32)
        d /= max(1e-6, np.linalg.norm(d))
        cam_quat = T.euler2quat(th.tensor(
            [math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))), 0.0, float(np.arctan2(-d[0], d[1]))],
            dtype=th.float32,
        ))
        og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat.tolist())
        for _ in range(3):
            og.sim.step()
        og.sim.render()
        og.sim.render()
        video_writer = init_video_writer(args.save_video, episode, args.video_fps, robot=None)
        if video_writer is None:
            raise RuntimeError("Failed to initialize snapshot-only viewer video writer.")
    else:
        video_writer = None

    executed = 0
    for _ in range(args.steps):
        og.sim.step()
        executed += 1
        if video_writer is not None:
            record_frame(video_writer)

    if video_writer is not None:
        close_video_writer(video_writer)

    summary = {"violated": None, "snapshot_only": True}
    append_jsonl(args.debug_jsonl, {
        "event": "replay_snapshot_only",
        "scene_model": scene_model,
        "activity_name": entry.activity_name,
        "source_snapshot": snapshot_path,
        "steps_executed": executed,
        "camera_override": {
            "eye": list(camera_override["eye"]) if camera_override["eye"] is not None else None,
            "lookat": list(camera_override["lookat"]) if camera_override["lookat"] is not None else None,
        },
        "issue_tags": list(entry.issue_tags),
    })
    return summary, executed


def main():
    parser = make_parser()
    args = parser.parse_args()
    setup_run_dir(args)

    manifest = load_curation_manifest(args.curation_manifest)
    entry = manifest.get_scene_entry(args.scene_model)
    episode = args.episode if args.episode is not None else entry.canonical_episode
    print(f"[Replay] Manifest: {manifest.source_path}")
    print(
        "[Replay] Scene selection: "
        f"scene={args.scene_model}, activity={entry.activity_name}, "
        f"status={entry.status}, repair_mode={entry.repair_mode}, canonical_episode={entry.canonical_episode}"
    )
    print(f"[Replay] Benchmark run dir: {entry.benchmark_run_dir}")
    print(f"[Replay] Scene dir: {entry.scene_dir}")
    if entry.issue_tags:
        print(f"[Replay] Issue tags: {', '.join(entry.issue_tags)}")
    if entry.review_note:
        print(f"[Replay] Review note: {entry.review_note}")

    checked_candidates = entry.snapshot_candidates(episode=episode)
    print(f"[Replay] Snapshot candidates for episode {episode!r}:")
    for candidate in checked_candidates:
        marker = "FOUND" if Path(candidate).is_file() else "missing"
        print(f"[Replay]   - {candidate} [{marker}]")

    snapshot_path = str(Path(args.snapshot_path).resolve()) if args.snapshot_path else entry.resolve_snapshot_path(episode=episode)
    if snapshot_path is None:
        available = entry.available_snapshot_paths()
        if available:
            print("[Replay] Available snapshots in scene directory:")
            for candidate in available:
                print(f"[Replay]   - {candidate}")
        else:
            print(f"[Replay] No snapshot files found under: {entry.scene_dir}")
        raise RuntimeError(
            f"No curated snapshot found for scene '{args.scene_model}' "
            f"(episode={episode!r}); repair/regenerate this scene first."
        )
    print(f"[Replay] Selected snapshot: {snapshot_path}")

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.manipulation_task_spec import build_manipulation_task_spec

    gm.ENABLE_OBJECT_STATES = True

    if args.snapshot_only:
        cfg = _build_scene_only_config(args.scene_model, snapshot_path)
        print("[Replay] Environment mode: snapshot_only (DummyTask, viewer-focused)")
    else:
        cfg = build_task_config(args.scene_model, entry.activity_name)
        cfg["scene"]["scene_file"] = snapshot_path
        cfg["scene"]["scene_instance"] = None
        cfg["task"]["online_object_sampling"] = False
        cfg["task"]["use_presampled_robot_pose"] = False
        print("[Replay] Environment overrides: online_object_sampling=False, use_presampled_robot_pose=False")

    video_path, diagnostics_path = expected_replay_artifact_paths(args, episode=0)
    video_before = _artifact_state(video_path)
    diagnostics_before = _artifact_state(diagnostics_path)
    if video_path:
        print(f"[Replay] Planned video output: {video_path}")
    if diagnostics_path:
        print(f"[Replay] Planned diagnostics output: {diagnostics_path}")
    if args.snapshot_only:
        print("[Replay] Video mode: viewer_only")
    else:
        print(
            "[Replay] Video mode: "
            f"{'viewer_only' if entry.video_viewer_only else 'viewer_plus_wrist'}"
        )
    camera_override = resolve_effective_camera_config(entry, args)
    if camera_override["eye"] is not None or camera_override["lookat"] is not None:
        print(
            f"[Replay] Camera override: eye={camera_override['eye']}, "
            f"lookat={camera_override['lookat']}"
        )
    if args.print_manifest_camera_snippet and not args.snapshot_only:
        _print_manifest_camera_snippet(args.scene_model, entry, camera_override)

    env = og.Environment(configs=cfg)
    try:
        env.reset()
        og.sim.step()

        if args.snapshot_only:
            camera_override = _scene_only_camera_override(env, entry, camera_override)
            print(
                f"[Replay] Camera override: eye={camera_override['eye']}, "
                f"lookat={camera_override['lookat']}"
            )
            if args.print_manifest_camera_snippet:
                _print_manifest_camera_snippet(args.scene_model, entry, camera_override)
            print(f"[Replay] Starting snapshot-only viewer rollout: steps={args.steps}")
            summary, executed = _run_snapshot_only_viewer_rollout(
                env=env,
                args=args,
                scene_model=args.scene_model,
                snapshot_path=snapshot_path,
                entry=entry,
                episode=0,
                camera_override=camera_override,
            )
        else:
            task_spec = build_manipulation_task_spec(entry.activity_name)
            obj_sets = build_task_object_sets(env, task_spec)
            if not obj_sets["target_ids"]:
                raise RuntimeError(f"No target objects found while replaying scene '{args.scene_model}'")

            _, active_objects = build_descriptors(env, obj_sets)
            target_obj = get_scope_obj(env, obj_sets["target_ids"][0])
            if target_obj is None:
                raise RuntimeError(f"Failed to resolve target object for scene '{args.scene_model}'")

            args.video_viewer_only = entry.video_viewer_only
            print(f"[Replay] Starting rollout: steps={args.steps}, seed={args.seed}, jitter_scale={args.jitter_scale}")

            summary, executed = run_ltl_rollout(
                env=env,
                activity_name=entry.activity_name,
                scene_model=args.scene_model,
                active_objects_by_inst=active_objects,
                robot=env.robots[0],
                target_obj=target_obj,
                args=args,
                episode=0,
                rng=np.random.default_rng(args.seed),
                camera_override=camera_override,
            )
        print(
            "[Replay] Rollout summary: "
            f"steps_executed={executed}, violated={summary['violated']}"
        )
        if not args.snapshot_only:
            append_jsonl(args.debug_jsonl, {
                "event": "replay",
                "scene_model": args.scene_model,
                "activity_name": entry.activity_name,
                "source_snapshot": snapshot_path,
                "ltl_violated": summary["violated"],
                "steps_executed": executed,
                "issue_tags": list(entry.issue_tags),
            })
        _print_artifact_result("video", video_path, video_before)
        _print_artifact_result("diagnostics", diagnostics_path, diagnostics_before)
    finally:
        print("[Replay] Shutdown simulator.")
        pipeline_exit()


if __name__ == "__main__":
    main()
