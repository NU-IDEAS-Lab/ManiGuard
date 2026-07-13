"""Shared render step for the ManiGuard-Bench builder.

``render_views`` is the SINGLE shared render entry point: given a live env (robot already
at the canonical pose), it positions the 4 canonical robot-frame external cameras via the
shared ``camera_setup``, RE-STAMPS ``diagnostics['cameras']`` with the live-computed poses,
records the 4 review MP4s (opposite_side_front / left_overview / right_overview /
left_shoulder), and returns stability stats. Recording is **idle-step**: each frame advances
physics (``og.sim.step()``) while the arm is held at the init pose by the stiff Isaac drive,
so the clip shows whether the scene is physically stable after init (objects settle, nothing
falls). ``render_task`` is a thin wrapper that loads a snapshot, sets the canonical pose, and
calls ``render_views``.

This is the single place every base + perturbation task gets its videos + valid camera
metadata, so the viewpoints are identical end-to-end (task-def -> collection -> SFT -> eval).
Standalone: reuses only camera_setup + task_generation.utils.video; no legacy perturbation imports.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_FPS = 30
DEFAULT_N_FRAMES = 60        # 60 frames @ 30 fps = a ~2 s idle-step stability clip
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
    resolution: int = DEFAULT_RESOLUTION,
    mode: str = "idle_step",
    ltl_monitor=None,
    track_object_names=None,
) -> tuple[dict, dict]:
    """Render the 4 canonical review videos from a LIVE env + return (diagnostics, stats).

    The caller is responsible for putting the env in its final state first (robot at the
    canonical pose, mount enforced, snapshot saved). This function only places the cameras,
    records, and reports stability — it is the single shared render entry point reused by
    ``render_task`` and the bench finalizer for base + every perturbation level.

    Args:
        env: a live ``og.Environment`` (already reset + robot posed).
        diagnostics: task diagnostics; a COPY is returned with ``cameras`` REPLACED by the
            live-computed 4-view poses. All other fields are left untouched.
        out_dir: directory to write ``rollout_<label>_ep{episode}.mp4`` into.
        n_frames: frames to record (60 @ 30fps ≈ 2 s).
        mode: ``"idle_step"`` advances physics each frame (``og.sim.step()``) while the arm is
            held at its set pose by the stiff Isaac position drive — so the clip shows physical
            stability. ``"frozen"`` only re-renders (no physics) — a static showcase.
        ltl_monitor: optional already-``reset()``+``step(0)``'d ``TaskLTLMonitor``. When given,
            it is stepped once per recorded frame so the bench can read its OWN fresh
            ``ltl_violated`` over the same idle-step the video shows (the analog of the
            generation-time jitter rollout). The caller reads the monitor afterwards.
        track_object_names: optional set of object names to restrict the ``obj_disp`` stat to —
            the env perturbation passes the injected task objects so a falling ROOM background
            object can't pollute it. ``None`` (base / location / target / standalone) tracks every
            non-robot object, which there IS the task set (those scenes hold no room furniture).

    Returns ``(diagnostics, stats)`` where ``stats`` = ``{"arm_drift", "obj_disp",
    "steps_executed"}``: the max joint drift, the max tracked-object displacement, and the
    number of in-sim steps recorded (for the finalizer's fresh stability/LTL self-check).
    """
    import av
    import numpy as np
    import omnigibson as og

    from maniguard.utils.camera_setup import compute_robot_frame_views
    from maniguard.task_generation.utils.video import (
        close_video_writer,
        init_video_writer,
        setup_cameras,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robot = env.robots[0]

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

    # --- baseline for the stability/hold stats (frame-0 joint + object state) ---
    q0 = robot.get_joint_positions().cpu().numpy()
    obj0 = _object_positions(env, robot, track_object_names)

    # --- probe one frame to size the writers from the ACTUAL rgb (sensor-kwarg resolution
    #     is unreliable in OmniGibson; sizing from the real frame avoids a swscale mismatch). ---
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

    # --- record loop. idle_step: og.sim.step() advances physics while the arm is held at its
    #     set pose by the Isaac drive (NOT env.step(zero_action) — for an absolute JointController
    #     a zero action commands all joints to 0 and flings the arm). frozen: render only. ---
    for i in range(n_frames):
        if mode == "idle_step":
            og.sim.step()
        else:
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
        # step the LTL monitor over the SAME idle-step the video shows (caller reads it after)
        if ltl_monitor is not None:
            ltl_monitor.step(i + 1)

    for wr, _sensor_name in writers:
        close_video_writer(wr)

    # --- stability/hold stats: max joint drift + max non-robot object displacement ---
    q1 = robot.get_joint_positions().cpu().numpy()
    arm_drift = float(np.abs(q1 - q0).max()) if len(q0) else 0.0
    obj1 = _object_positions(env, robot, track_object_names)
    obj_disp = 0.0
    for name, p0 in obj0.items():
        p1 = obj1.get(name)
        if p1 is not None:
            obj_disp = max(obj_disp, float(np.linalg.norm(p1 - p0)))
    return diagnostics, {"arm_drift": arm_drift, "obj_disp": obj_disp, "steps_executed": int(n_frames)}


def render_task(
    scene_file,
    diagnostics: dict,
    out_dir,
    *,
    episode: int = 1,
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    resolution: int = DEFAULT_RESOLUTION,
    mode: str = "idle_step",
) -> dict:
    """Load a snapshot, set the canonical init pose, and render its 4 review videos.

    Thin wrapper around ``render_views`` for ad-hoc rendering of an existing snapshot (it does
    NOT enforce the mount — that is the finalizer's job). Returns the updated diagnostics dict
    (with refreshed ``cameras``); MP4s are written to ``out_dir``.
    """
    import omnigibson as og
    import torch as th

    from maniguard.utils.robot_pose import BENCH_INIT_QPOS

    scene_file = Path(scene_file)
    out_dir = Path(out_dir)

    og_cfg = _build_og_config(scene_file, diagnostics, resolution)
    if og.sim is not None:
        og.sim.stop()
        og.clear()
    env = og.Environment(configs=og_cfg)
    env.reset()

    robot = env.robots[0]
    robot.set_joint_positions(th.tensor(BENCH_INIT_QPOS, dtype=th.float32))
    robot.keep_still()

    diagnostics, _stats = render_views(
        env, diagnostics, out_dir,
        episode=episode, n_frames=n_frames, fps=fps, resolution=resolution, mode=mode,
    )
    return diagnostics
