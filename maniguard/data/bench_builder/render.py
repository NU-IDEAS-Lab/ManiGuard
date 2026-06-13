"""Shared render step for the ManiGuard-Bench builder.

Loads a finalized task's `scene_ep*.json` into OmniGibson (the snapshot already holds
FrankaPanda+longfinger + the task objects), positions the 4 canonical robot-frame
external cameras via the shared ``camera_setup``, RE-STAMPS ``diagnostics['cameras']``
with the live-computed poses, idles the sim, and writes the 4 review MP4s
(opposite_side_front / left_overview / right_overview / left_shoulder).

This is the single place every base + perturbation task gets its videos + valid camera
metadata, so the viewpoints are identical end-to-end (task-def -> collection -> SFT -> eval).
Standalone: reuses only camera_setup + task_generation.utils.video; no legacy perturbation imports.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_FPS = 30
DEFAULT_N_FRAMES = 60        # 60 frames @ 30 fps = a ~2 s static scene-showcase clip
DEFAULT_RESOLUTION = 256


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


def render_task(
    scene_file,
    diagnostics: dict,
    out_dir,
    *,
    episode: int = 1,
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    resolution: int = DEFAULT_RESOLUTION,
) -> dict:
    """Render a task's 4 canonical review videos + return diagnostics with refreshed cameras.

    The robot is put at the canonical natural init pose and the scene is rendered STATICALLY
    (no physics step) — a short scene-showcase clip, not a rollout.

    Args:
        scene_file: path to the task's ``scene_ep{episode}.json`` snapshot.
        diagnostics: the task's diagnostics dict (read for ``scene_model``; its
            ``cameras`` field is REPLACED with the live-computed poses).
        out_dir: directory to write ``rollout_<label>_ep{episode}.mp4`` into.
        episode: 1-indexed episode number (matches scene_ep{episode}.json / _ep{episode}.mp4).
        n_frames: number of (identical, static) frames to write (60 @ 30fps = ~2 s).

    Returns the updated diagnostics dict (caller persists it). MP4s are written to out_dir.
    """
    import av
    import numpy as np
    import omnigibson as og
    import torch as th

    from maniguard.utils.camera_setup import compute_robot_frame_views
    from maniguard.utils.robot_pose import BENCH_INIT_QPOS
    from maniguard.task_generation.utils.video import (
        close_video_writer,
        init_video_writer,
        setup_cameras,
    )

    scene_file = Path(scene_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load the snapshot into a fresh env ---
    og_cfg = _build_og_config(scene_file, diagnostics, resolution)
    if og.sim is not None:
        og.sim.stop()
        og.clear()
    env = og.Environment(configs=og_cfg)
    env.reset()

    # --- canonical natural init pose (folded "ready"; wrist cam looks down at the pack).
    #     Static showcase => no physics step below, so the arm stays exactly here. ---
    robot = env.robots[0]
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()

    # --- place the 4 canonical robot-frame cameras; setup_cameras applies AND returns
    #     the posed specs (eye/orientation/sensor_name) we stamp into diagnostics ---
    views = compute_robot_frame_views(env)
    posed = setup_cameras(env, views)
    diagnostics = dict(diagnostics)
    diagnostics["cameras"] = [
        {
            "label": v["label"],
            "sensor_name": v["sensor_name"],
            "eye": [float(x) for x in v["position"]],
            "lookat": [float(x) for x in v["lookat"]],
            "orientation": [float(x) for x in v["orientation"]],
        }
        for v in posed
    ]

    # --- probe one frame to size the writers from the ACTUAL rgb (sensor-kwarg resolution
    #     is unreliable in OmniGibson; sizing from the real frame avoids a swscale mismatch).
    #     No physics step: arm + scene stay frozen at the init pose (static showcase). ---
    og.sim.render()
    raw_obs, _ = env.get_obs()
    external = raw_obs.get("external", {})

    writers = []  # (writer_dict, sensor_name)
    for v in posed:
        rgb = external.get(v["sensor_name"], {}).get("rgb")
        h, w = (int(rgb.shape[0]), int(rgb.shape[1])) if rgb is not None else (resolution, resolution)
        wr = init_video_writer(
            str(out_dir / f"rollout_{v['label']}"), episode - 1, fps,
            robot=None, frame_hw=(h, w),
        )
        writers.append((wr, v["sensor_name"]))

    # --- static render loop (frozen scene, no physics step) ---
    for _ in range(n_frames):
        og.sim.render()
        raw_obs, _ = env.get_obs()
        external = raw_obs.get("external", {})
        for wr, sensor_name in writers:
            rgb = external.get(sensor_name, {}).get("rgb")
            if rgb is None:
                continue
            frame = rgb[..., :3].cpu().numpy().astype(np.uint8)
            vframe = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in wr["stream"].encode(vframe):
                wr["container"].mux(packet)

    for wr, _sensor_name in writers:
        close_video_writer(wr)

    return diagnostics
