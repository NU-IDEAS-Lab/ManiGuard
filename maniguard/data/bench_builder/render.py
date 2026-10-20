"""Record review videos and camera metadata for benchmark construction.

High-fidelity rendering records all five views, preserving saved cameras
and robot pose and applying appearance overrides. Idle-step advances physics;
frozen mode only renders. CLI previews write to a separate staging directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from maniguard.visual.capture import prepare_source_environment as prepare_preview_environment

DEFAULT_FPS = 30
DEFAULT_N_FRAMES = 60        # 60 frames @ 30 fps = a ~2 s idle-step stability clip
DEFAULT_RESOLUTION = 640


def _build_og_config(scene_file: Path, diagnostics: dict, resolution: int) -> dict:
    """Minimal "load this snapshot + the 4 external cameras" config — no policy, no task
    logic. Mirrors ``maniguard.eval.benchmark.build_og_config``: pass ``scene_file`` as a
    path string with ``robots:[] / include_robots:True`` so OmniGibson reconstructs the
    robot + objects straight from the snapshot.
    """
    from maniguard.utils.camera_setup import (
        EXTERNAL_CAMERA_NAMES,
        build_external_camera_configs,
    )

    header = json.loads(Path(scene_file).read_text(encoding="utf-8"))
    scene_class = header.get("init_info", {}).get("class_name", "")
    scene_model = diagnostics.get("scene_model")

    if scene_class == "InteractiveTraversableScene" and scene_model:
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "scene_file": str(scene_file),
            "scene_instance": None,
            "include_robots": True,
        }
    else:  # empty-scene families (base) -> a bare Scene loaded from the snapshot
        scene_cfg = {"type": "Scene", "scene_file": str(scene_file)}

    return {
        "scene": scene_cfg,
        "robots": [],
        "objects": [],
        "task": {"type": "DummyTask"},
        "env": {
            "external_sensors": build_external_camera_configs(
                EXTERNAL_CAMERA_NAMES, resolution=resolution
            ),
        },
    }

def review_videos_complete(directory, episode=1, source_profile="recorded"):
    """Validate the requested (or recorded) visual layout without starting Kit."""
    from maniguard.visual.preview import preview_is_complete
    from maniguard.visual.profile import resolve_profile_selection
    directory = Path(directory)
    if source_profile == "recorded":
        try:
            diag = json.loads((directory / "diagnostics.jsonl").read_text().splitlines()[0])
            source_profile = diag.get("source_visual", {}).get("profile")
        except (OSError, ValueError, IndexError):
            return False
        if source_profile is None:
            return all((directory / f"rollout_{label}_ep{episode}.mp4").is_file() for label in
                       ("opposite_side_front", "left_overview", "right_overview", "left_shoulder"))
    profile = resolve_profile_selection(source_profile)
    return preview_is_complete(directory / f"source_preview_ep{episode}", profile)

def _object_positions(env, robot, names=None):
    """World-XYZ of non-robot scene objects, keyed by name (for obj-displacement stats). ``names``
    optionally restricts to a set of object names — the env perturbation passes the injected task
    objects so an unsupported ROOM background object falling into the void can't pollute the stat
    (it would otherwise dominate ``max`` displacement while the task objects sit perfectly still)."""
    import numpy as np

    out = {}
    for obj in env.scene.objects:
        if obj is robot or (names is not None and obj.name not in names):
            continue
        p, _ = obj.get_position_orientation()
        out[obj.name] = np.asarray(p.cpu().numpy() if hasattr(p, "cpu") else p, dtype=np.float32)
    return out


def render_views(
    env,
    diagnostics: dict,
    out_dir,
    *,
    episode: int = 1,
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    resolution: int | None = None,
    mode: str = "idle_step",
    ltl_monitor=None,
    track_object_names=None,
    source_profile="high_fidelity",
) -> tuple[dict, dict]:
    """Render the five source views from a LIVE env + return (diagnostics, stats).

    The caller is responsible for putting the env in its final state first (robot at the
    canonical pose, mount enforced, snapshot saved). This function only places the cameras,
    records, and reports stability — it is the single shared render entry point reused by
    ``render_task`` and the bench finalizer for base + every perturbation level.

    Args:
        env: a live ``og.Environment`` (already reset + robot posed).
        diagnostics: task diagnostics; a COPY is returned with ``cameras`` REPLACED by the
            saved or computed external-view poses. All other fields are left untouched.
        out_dir: parent directory of ``source_preview_ep{episode}``.
        n_frames: frames to record (60 @ 30fps ≈ 2 s).
        mode: ``"idle_step"`` advances physics each frame (``og.sim.step()``) while the arm is
            held at its set pose by the stiff Isaac position drive — so the clip shows physical
            stability. ``"frozen"`` only re-renders (no physics) — a static showcase.
        ltl_monitor: optional reset TaskLTLMonitor. It receives one observation
            per recorded frame, including repeated-state observations in frozen
            mode. The caller reads its results after rendering.
        track_object_names: optional set of object names to restrict the ``obj_disp`` stat to —
            the env perturbation passes the injected task objects so a falling ROOM background
            object can't pollute it. ``None`` (base / location / target / standalone) tracks every
            non-robot object, which there IS the task set (those scenes hold no room furniture).

    Returns (diagnostics, stats). arm_drift is the largest absolute joint change
    between the first and last frame; obj_disp is the largest object displacement
    between those frames. steps_executed is the number of physics steps (zero
    in frozen mode).
    """
    import numpy as np
    import omnigibson as og

    from maniguard.task_generation.utils.video import setup_cameras
    from maniguard.utils.camera_setup import compute_robot_frame_views
    from maniguard.visual.profile import resolve_profile_selection, resolve_source_resolution
    source_profile = resolve_profile_selection(source_profile)
    resolution = resolve_source_resolution(source_profile, resolution)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robot = env.robots[0]

    # --- place the 4 canonical robot-frame cameras; setup_cameras applies AND returns
    #     the posed specs (eye/orientation/sensor_name) we stamp into diagnostics ---
    diagnostics = dict(diagnostics)
    if diagnostics.get('cameras'):
        from maniguard.utils.camera_setup import place_recorded_task_cameras
        place_recorded_task_cameras(env, diagnostics, set_viewer=False)
    else:
        views = compute_robot_frame_views(env)
        posed = setup_cameras(env, views)
        diagnostics["cameras"] = [
            {"label": v["label"], "sensor_name": v["sensor_name"],
             "eye": [float(x) for x in v["position"]],
             "lookat": [float(x) for x in v["lookat"]],
             "orientation": [float(x) for x in v["orientation"]]}
            for v in posed
        ]

    # --- baseline for the stability/hold stats (frame-0 joint + object state) ---
    q0 = robot.get_joint_positions().cpu().numpy()
    obj0 = _object_positions(env, robot, track_object_names)

    from maniguard.visual.preview import render_preview
    report = render_preview(env, out_dir / f"source_preview_ep{episode}", source_profile,
                            frames=n_frames, fps=round(1 / og.sim.get_sim_step_dt()) if mode == "idle_step" else fps,
                            mode=mode, monitor=ltl_monitor, overwrite=True)
    q1 = robot.get_joint_positions().cpu().numpy()
    obj1 = _object_positions(env, robot, track_object_names)
    diagnostics["source_visual"] = {"profile": report["profile"], "cameras": report["cameras"]}
    return diagnostics, {"arm_drift": float(np.max(np.abs(q1-q0))) if len(q0) else 0.,
                         "obj_disp": max((float(np.linalg.norm(obj1[k]-v)) for k,v in obj0.items() if k in obj1), default=0.),
                         "steps_executed": n_frames if mode == "idle_step" else 0}



def render_task(
    scene_file,
    diagnostics: dict,
    out_dir,
    *,
    episode: int = 1,
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    resolution: int | None = None,
    mode: str = "idle_step",
    source_profile="high_fidelity",
) -> dict:
    """Load a snapshot, apply its appearance overrides, and render all five views.

    Thin wrapper around ``render_views`` for ad-hoc rendering of an existing snapshot (it does
    NOT enforce the mount — that is the finalizer's job). Returns the updated diagnostics dict
    (with refreshed ``cameras``); MP4s are written to ``out_dir``.
    """
    from maniguard.visual.profile import resolve_profile_selection, resolve_source_resolution
    source_profile = resolve_profile_selection(source_profile)
    resolution = resolve_source_resolution(source_profile, resolution)
    from maniguard.data.datagen.primitives.scene import init_omnigibson, task_needs_gpu_dynamics
    from maniguard.data.bench_builder.perturbation import apply_perturbation
    from maniguard.visual.capture import prepare_source_launch

    scene_file = Path(scene_file)
    out_dir = Path(out_dir)
    prepare_source_launch()
    og = init_omnigibson(needs_gpu_dynamics=task_needs_gpu_dynamics(scene_file.parent), source_profile=source_profile)
    og_cfg = _build_og_config(scene_file, diagnostics, resolution)
    og_cfg["env"].update(action_frequency=fps, rendering_frequency=60, physics_frequency=120)
    prepare_preview_environment(og_cfg, source_profile, clear=True)
    env = og.Environment(configs=og_cfg)
    env.reset()
    # Appearance is part of the task definition, independent of rendering settings.
    apply_perturbation(env, diagnostics)

    diagnostics, _stats = render_views(
        env, diagnostics, out_dir,
        episode=episode, n_frames=n_frames, fps=fps, resolution=resolution, mode=mode,
        source_profile=source_profile,
    )
    return diagnostics


def main():
    """Render a source-quality staging preview without rewriting benchmark GT."""
    import argparse
    import hashlib

    from maniguard.data.datagen.primitives.task_io import load_diagnostics_row
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--task-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-profile', default='high_fidelity')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--mode', choices=['frozen', 'idle_step'], default='frozen')
    args = parser.parse_args()
    task = args.task_dir.resolve()
    output = args.output.resolve()
    if output == task or task in output.parents:
        parser.error('Write previews to a separate staging directory')
    files = [task / 'scene_ep1.json', task / 'diagnostics.jsonl']
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    render_task(files[0], load_diagnostics_row(task, 1), output, n_frames=args.frames, fps=args.fps,
                mode=args.mode, source_profile=args.source_profile)
    if before != [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]:
        raise RuntimeError('Source benchmark files changed during preview')
    print('Source scene and diagnostics are unchanged.', flush=True)
    import omnigibson as og
    og.shutdown()


if __name__=='__main__': main()
