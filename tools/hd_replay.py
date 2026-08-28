#!/usr/bin/env python
"""Re-render a datagen RAW trajectory at high resolution from its per-step sim-state dumps.

Every RAW datagen trajectory ships a per-step serialized ``og.sim.dump_state`` (the
``states`` dataset in ``traj.hdf5`` — see ``maniguard/data/datagen/data_format.md``).
This tool rebuilds the task scene exactly as datagen did, then per frame restores that
state and renders the chosen camera at ``--resolution`` — a native re-render (no physics
stepping, no policy): pixels come straight from the renderer, at any resolution, e.g. to
produce high-resolution videos of released demonstrations.

Run in the ``behavior`` conda env, headless:

    OMNIGIBSON_HEADLESS=1 python tools/hd_replay.py \
        --traj <datagen_raw_root>/stack_retrieve/task_0013/traj_000 \
        --task-dir <bench_root>/stack_retrieve/task_0013/base \
        --out out.mp4 --resolution 1024 --camera cam_left

Isaac Sim may segfault at teardown AFTER the video is fully written — judge success by
the output file, not the exit code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="datagen RAW traj_XXX dir (traj.hdf5 + meta.json)")
    ap.add_argument("--task-dir", required=True,
                    help="bench task LEVEL dir holding diagnostics.jsonl (e.g. <bench>/<family>/task_NNNN/base)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--camera", default="cam_left",
                    help="external sensor name to record (cam_left/cam_right/cam_opposite/cam_left_shoulder) or 'wrist'")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    traj_dir = Path(args.traj).expanduser()
    with h5py.File(traj_dir / "traj.hdf5", "r") as f:
        states = f["states"][:]
    n = states.shape[0]
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"[hd_replay] {n} frames @ {args.resolution}px from {traj_dir}", flush=True)

    from maniguard.data.datagen.primitives.cameras import (
        external_camera_configs,
        find_wrist_sensor,
        install_wrist_camera,
        place_and_resize_cameras,
    )
    from maniguard.data.datagen.primitives.scene import (
        init_omnigibson,
        scene_from_task_dir,
        task_needs_gpu_dynamics,
    )

    task_dir = Path(args.task_dir).expanduser()
    og = init_omnigibson(headless=True, needs_gpu_dynamics=task_needs_gpu_dynamics(task_dir))
    bundle = scene_from_task_dir(
        task_dir,
        external_sensors=external_camera_configs(args.resolution),
        pre_build_hooks=[install_wrist_camera],
    )
    env, robot = bundle.env, bundle.robot
    n_cams = place_and_resize_cameras(env, robot, og, bundle.diagnostics, resolution=args.resolution)
    print(f"[hd_replay] scene up, {n_cams} cameras at {args.resolution}", flush=True)

    # resolve the sensor to record
    if args.camera == "wrist":
        sensor = find_wrist_sensor(robot)
    else:
        sensor = None
        for name, s in (env.external_sensors or {}).items():
            if args.camera in name:
                sensor = s
                break
    assert sensor is not None, f"camera {args.camera} not found"

    import av
    import torch as th

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), mode="w")
    stream = container.add_stream("h264", rate=30)
    stream.width = stream.height = args.resolution
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "17"}

    for i in range(0, n, args.stride):
        og.sim.load_state(th.as_tensor(states[i], dtype=th.float32), serialized=True)
        og.sim.render()
        obs = sensor.get_obs()
        rgb = obs[0].get("rgb") if isinstance(obs, tuple) else obs.get("rgb")
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        frame = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8))
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
        if i and i % 300 == 0:
            print(f"[hd_replay] {i}/{n}", flush=True)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"[hd_replay] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
