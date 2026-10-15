"""Convert RAW demonstrations into LeRobot v2.1 datasets.

Read joint arrays and prompts from RAW trajectory folders, copy the five MP4
streams without re-encoding, and write numeric features as Parquet. Image
statistics are sampled from the copied videos through temporary LeRobot 0.3.3
adapters. Conversion runs separately from simulation collection.
"""
from __future__ import annotations

import numpy as np


def video_contract(metas):
    """Validate a homogeneous export before creating a LeRobot dataset."""
    from .data_format import FPS, IMAGE_KEYS, RESOLUTION
    contracts = set()
    for meta in metas:
        resolution, fps = meta.get('resolution', RESOLUTION), meta.get('fps', FPS)
        if not isinstance(resolution, int) or resolution <= 0 or not isinstance(fps, int) or fps <= 0:
            raise ValueError('LeRobot export requires positive square resolution and integer FPS')
        if set(meta.get('video_keys', IMAGE_KEYS)) != set(IMAGE_KEYS):
            raise ValueError('LeRobot export requires all five source views')
        action_source = meta.get('action_source', 'next_achieved')
        if action_source != 'next_achieved' and meta.get('action_space') != 'joint':
            raise ValueError('This LeRobot schema requires declared joint commands')
        if action_source != 'next_achieved' and meta.get('state_space') != 'joint':
            raise ValueError('Source export requires declared joint state')
        contracts.add((resolution, fps, action_source))
    if len(contracts) != 1:
        raise ValueError('Mixed resolution, FPS or action semantics; export separate datasets')
    return contracts.pop()


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
    if state.shape != (n, 8) or actions.shape != (n, 8) or commanded.shape != (n, 8):
        raise ValueError('Joint LeRobot features require aligned (T, 8) state and command arrays')
    extra = {key: np.asarray(traj[key]) for key in ("command_issued", "source_observation_id", "source_boundary_id", "target_observation_id", "sim_time_s") if key in traj}
    if any(len(value) != n for value in extra.values()):
        raise ValueError('Source interval metadata lengths differ')
    return [{"state": state[t], "actions": actions[t], "actions_commanded": commanded[t],
             **{key: value[t:t+1] for key, value in extra.items()}}
            for t in range(n)]


# --- MP4 passthrough adapters for LeRobot 0.3.3 ---

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
    """Temporarily adapt LeRobot 0.3.3 for MP4 passthrough: suppress image writes, resolve versions locally, and sample image statistics from existing videos. Restore the patched functions on exit."""
    import contextlib

    import lerobot.datasets.compute_stats as _cs
    import lerobot.datasets.lerobot_dataset as _lds
    import lerobot.datasets.utils as _lu
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
    from importlib.metadata import PackageNotFoundError, version

    try:
        runtime_version = version("lerobot")
    except PackageNotFoundError:
        runtime_version = None
    if runtime_version != "0.3.3":
        raise RuntimeError(
            f"RAW conversion to LeRobot v2.1 requires lerobot==0.3.3; found {runtime_version!r}. "
            "Use the separate conversion environment with pip install -e '.[conversion]'."
        )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from maniguard.data.datagen import data_format, reader

    traj_dirs = list(reader.iter_traj_dirs(dataset, family))
    if limit:
        traj_dirs = traj_dirs[:limit]
    if not traj_dirs:
        raise ValueError(f"No RAW trajectories found for {family} in {dataset}")
    metas = [reader.load_meta(d) for d in traj_dirs]
    resolution, fps, action_source = video_contract(metas)
    prompts, task_indices = build_prompt_table(metas)
    print(f"[to_lerobot] {family}: {len(traj_dirs)} trajs, {len(prompts)} unique prompts", flush=True)

    root = Path(out_root) / family
    features = data_format.lerobot_features(resolution)
    if action_source != 'next_achieved':
        features['actions']['names'] = data_format.ACTION_COMMANDED_NAMES
        for key, dtype in (("command_issued", "bool"), ("source_observation_id", "int64"), ("source_boundary_id", "int64"),
                           ("target_observation_id", "int64"), ("sim_time_s", "float64")):
            features[key] = {"dtype": dtype, "shape": (1,), "names": [key]}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features,
                               root=root, robot_type=data_format.ROBOT_TYPE, use_videos=True)
    keys = list(data_format.IMAGE_KEYS)
    dummy = {k: np.zeros((resolution, resolution, 3), dtype=np.uint8) for k in keys}

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
