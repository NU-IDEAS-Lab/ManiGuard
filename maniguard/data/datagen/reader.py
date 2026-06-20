"""Reader for the datagen demo dataset.

On-disk layout (written by ``driver.run_task``)::

    outputs/datagen/<dataset>/<family>/<task>/<traj>/
        image_opposite.mp4  image_left.mp4  image_right.mp4  image_left_shoulder.mp4  wrist_image.mp4
        traj.hdf5     # state (N,8), actions (N,8), actions_commanded (N,8), states (N,*) sim-dump
        meta.json     # family / source_task / task / traj / target_key / grasp_id / approach / seed / success...

This is the SINGLE entry point the LeRobot converter + review tooling use to enumerate and
load collected demos, so the on-disk layout and its consumers stay in sync. Pure h5py / PyAV
/ json — no OmniGibson.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/datagen")
VIDEO_KEYS = ("image_opposite", "image_left", "image_right", "image_left_shoulder", "wrist_image")


def iter_traj_dirs(dataset: str = "demos", family: str | None = None, *, root=ROOT):
    """Yield every traj dir (with a traj.hdf5) under ``<root>/<dataset>[/<family>]``, sorted."""
    base = Path(root) / dataset
    if not base.exists():
        return
    fams = [family] if family else [p.name for p in sorted(base.iterdir()) if p.is_dir()]
    for fam in fams:
        for task_dir in sorted((base / fam).glob("task_*")):
            for traj_dir in sorted(task_dir.glob("traj_*")):
                if (traj_dir / "traj.hdf5").exists():
                    yield traj_dir


def load_meta(traj_dir) -> dict:
    p = Path(traj_dir) / "meta.json"
    return json.load(open(p)) if p.exists() else {}


def load_traj(traj_dir, *, load_sim_states: bool = False) -> dict:
    """Load one demo: meta + joint arrays + the video-file paths. ``state``/``actions``/
    ``actions_commanded`` are (N,8); ``sim_states`` (N,*) only if ``load_sim_states``."""
    import h5py

    traj_dir = Path(traj_dir)
    out = {
        "path": str(traj_dir),
        "meta": load_meta(traj_dir),
        "videos": {k: str(traj_dir / f"{k}.mp4")
                   for k in VIDEO_KEYS if (traj_dir / f"{k}.mp4").exists()},
    }
    with h5py.File(traj_dir / "traj.hdf5", "r") as h:
        out["state"] = h["state"][:]
        out["actions"] = h["actions"][:]
        out["actions_commanded"] = h["actions_commanded"][:]
        if load_sim_states and "states" in h:
            out["sim_states"] = h["states"][:]
    return out


def read_frames(video_path) -> np.ndarray:
    """Decode an mp4 stream to (N, H, W, 3) uint8 (the per-frame images for LeRobot)."""
    import av

    frames = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames) if frames else np.empty((0, 0, 0, 3), dtype=np.uint8)


def summarize(dataset: str = "demos", family: str | None = None, *, root=ROOT) -> list:
    dirs = list(iter_traj_dirs(dataset, family, root=root))
    base = Path(root) / dataset
    print(f"[reader] {len(dirs)} trajs under {base}" + (f"/{family}" if family else ""))
    for d in dirs:
        m = load_meta(d)
        print(f"  {d.relative_to(base)}  target={m.get('target_key')} grasp={m.get('grasp_id')} "
              f"seed={m.get('seed')} n_steps={m.get('n_steps')} held={m.get('held_in_goal')}")
    return dirs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="demos")
    ap.add_argument("--family", default=None)
    summarize(**vars(ap.parse_args()))
