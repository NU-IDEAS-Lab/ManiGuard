"""Datagen RAW -> LeRobot v2.1 converter (family-agnostic; runs in the lerobot uv env).

Repackages outputs/datagen/<dataset>/<family>/task_*/traj_*/ (traj.hdf5 + 5 mp4 + meta.json) into ONE
LeRobot v2.1 dataset per family: numeric (state + actions + actions_commanded) in parquet, the 5 camera
mp4s passthrough-placed (no re-encode), prompt per episode from meta. Reuses datagen.reader +
data_format; imports NONE of maniguard.data.lerobot.* (technique referenced, not imported). Heavy imports
(lerobot / h5py / reader) are lazy so the pure helpers test in any env.
"""
from __future__ import annotations

import numpy as np


def build_prompt_table(metas: list[dict]) -> tuple[list[str], list[int]]:
    """Unique prompts (first-seen order) + each traj's task_index into that list."""
    unique: list[str] = []
    index_of: dict[str, int] = {}
    task_indices: list[int] = []
    for m in metas:
        p = m.get("prompt", "") or ""
        if p not in index_of:
            index_of[p] = len(unique)
            unique.append(p)
        task_indices.append(index_of[p])
    return unique, task_indices


def frame_rows(traj: dict) -> list[dict]:
    """N per-timestep numeric frame dicts from a reader.load_traj output (state/actions/
    actions_commanded, each (N,8)). float32. Video keys are added later (passthrough)."""
    state = np.asarray(traj["state"], dtype=np.float32)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    commanded = np.asarray(traj["actions_commanded"], dtype=np.float32)
    n = len(state)
    return [{"state": state[t], "actions": actions[t], "actions_commanded": commanded[t]}
            for t in range(n)]


# --- passthrough (ported + VERIFIED against lerobot 0.3.3; NOT imported from lerobot_writer) ---

def _png_to_mp4(png_path):
    """Map a would-be PNG path <root>/images/<key>/episode_NNNNNN/frame_MMMMMM.png to the pre-placed
    mp4 <root>/videos/chunk-XXX/<key>/episode_NNNNNN.mp4 (stats sampling is the only PNG readback)."""
    from pathlib import Path
    p = Path(png_path)
    if p.parent.parent.parent.name != "images":
        return None
    root = p.parent.parent.parent.parent
    key = p.parent.parent.name
    ep = p.parent.name
    m = list((root / "videos").glob(f"*/{key}/{ep}.mp4"))
    return m[0] if m else None


def _decode_mp4(mp4_path, indices):
    """Decode frames at 0-based indices -> (len(indices), C, H, W) uint8."""
    import av
    want = {int(i) for i in indices}; mx = max(want); out = {}
    with av.open(str(mp4_path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24").transpose(2, 0, 1)
                if len(out) == len(want):
                    break
            if i > mx:
                break
    return np.stack([out[int(i)] for i in indices], axis=0)


def _frame_count(mp4_path) -> int:
    import av
    with av.open(str(mp4_path)) as c:
        return sum(1 for _ in c.decode(c.streams.video[0]))


def _passthrough_images():
    """Context manager: patch lerobot 0.3.3 so add_frame writes no PNGs and save_episode does NOT
    re-encode a camera whose mp4 is already at its target path. Three patches (verified Task 0):
    _save_image no-op; get_safe_version local (offline repo_id); compute_stats.sample_images mp4-aware
    (the only PNG readback during commit). Restored on exit."""
    import contextlib
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.datasets.utils as _lu
    import lerobot.datasets.lerobot_dataset as _lds
    import lerobot.datasets.compute_stats as _cs

    @contextlib.contextmanager
    def _cm():
        saved = (LeRobotDataset._save_image, _lu.get_safe_version, _lds.get_safe_version, _cs.sample_images)
        orig_sample = _cs.sample_images
        from pathlib import Path

        def _mp4_aware(image_paths):
            if image_paths and Path(image_paths[0]).is_file():
                return orig_sample(image_paths)
            mp4 = _png_to_mp4(image_paths[0]) if image_paths else None
            if mp4 is None or not Path(mp4).is_file():
                raise FileNotFoundError(f"no PNG nor MP4 for stats: {image_paths[:1]}")
            idx = _cs.sample_indices(len(image_paths))
            return np.stack([_cs.auto_downsample_height_width(im) for im in _decode_mp4(mp4, idx)], axis=0)

        LeRobotDataset._save_image = lambda self, image, fpath: None
        _lu.get_safe_version = lambda repo_id, version: str(version)
        _lds.get_safe_version = _lu.get_safe_version
        _cs.sample_images = _mp4_aware
        try:
            yield
        finally:
            (LeRobotDataset._save_image, _lu.get_safe_version,
             _lds.get_safe_version, _cs.sample_images) = saved

    return _cm()


def place_video(traj_dir, dataset, ep_index: int, image_keys) -> int:
    """Copy each raw <key>.mp4 to Path(dataset.root)/get_video_file_path(ep,key) (root-relative).
    Returns the shared frame count (raises ValueError if the 5 cams disagree)."""
    import shutil
    from pathlib import Path
    counts = []
    for key in image_keys:
        src = Path(traj_dir) / f"{key}.mp4"
        dst = Path(dataset.root) / dataset.meta.get_video_file_path(ep_index, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        counts.append(_frame_count(dst))
    if len(set(counts)) != 1:
        raise ValueError(f"{traj_dir}: mp4 frame counts differ: {dict(zip(image_keys, counts))}")
    return counts[0]


def convert(dataset: str, family: str, *, out_root: str, repo_id: str, limit=None) -> dict:
    """Convert every traj of <family> in <dataset> into ONE LeRobot v2.1 dataset at
    <out_root>/<family>/ (all 5 cams passthrough; state + actions(b) + actions_commanded(a))."""
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from maniguard.data.datagen import reader, data_format

    traj_dirs = list(reader.iter_traj_dirs(dataset, family))
    if limit:
        traj_dirs = traj_dirs[:limit]
    metas = [reader.load_meta(d) for d in traj_dirs]
    prompts, task_indices = build_prompt_table(metas)
    print(f"[to_lerobot] {family}: {len(traj_dirs)} trajs, {len(prompts)} unique prompts", flush=True)

    root = Path(out_root) / family
    features = data_format.lerobot_features(data_format.RESOLUTION)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=data_format.FPS, features=features,
                               root=root, robot_type=data_format.ROBOT_TYPE, use_videos=True)
    keys = list(data_format.IMAGE_KEYS)
    dummy = {k: np.zeros((data_format.RESOLUTION, data_format.RESOLUTION, 3), dtype=np.uint8) for k in keys}

    kept, skipped = 0, []
    with _passthrough_images():
        for i, tdir in enumerate(traj_dirs):
            traj = reader.load_traj(tdir)
            rows = frame_rows(traj)
            n = len(rows)
            try:
                vframes = place_video(tdir, ds, ds.meta.total_episodes, keys)
            except (FileNotFoundError, ValueError) as e:
                skipped.append((str(tdir), f"video:{e}")); continue
            if vframes != n:
                skipped.append((str(tdir), f"len mismatch mp4={vframes} hdf5={n}")); continue
            task = prompts[task_indices[i]]
            for row in rows:
                ds.add_frame({**dummy, **row}, task=task)
            ds.save_episode()
            kept += 1
            if kept % 100 == 0:
                print(f"[to_lerobot] {kept}/{len(traj_dirs)} episodes", flush=True)

    summary = {"family": family, "episodes": kept, "skipped": len(skipped),
               "unique_prompts": len(prompts), "root": str(root)}
    print(f"[to_lerobot] DONE {summary}", flush=True)
    if skipped:
        print(f"[to_lerobot] SKIPPED {len(skipped)}: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. v1")
    ap.add_argument("--family", required=True, help="bench family dir, e.g. clutter_pickup")
    ap.add_argument("--out-root", default=None, help="default: outputs/datagen/<dataset>_lerobot_format")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out_root = a.out_root or f"outputs/datagen/{a.dataset}_lerobot_format"
    convert(a.dataset, a.family, out_root=out_root, repo_id=a.repo_id, limit=a.limit)

