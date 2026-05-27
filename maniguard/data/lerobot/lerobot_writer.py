"""LeRobot v2.1 direct writer — uses pre-rendered MP4s, skips encode roundtrip.

Standard LeRobotDataset flow during collection is:

    add_frame(frame)   writes one PNG per image feature per frame to
                       ``<root>/images/<key>/.../frame_NNNNNN.png``
    save_episode()     ffmpeg-encodes those PNGs into
                       ``<root>/videos/.../episode_NNNNNN.mp4``,
                       writes ``<root>/data/.../episode_NNNNNN.parquet``,
                       updates meta/*.

For sim collection that already streams h264 MP4s via imageio (e.g.
``maniguard/data/curobo/_sft_recorder.SFTRecorder``), the PNG-write + re-encode is pure
overhead. This module makes ``LeRobotDataset`` accept pre-rendered MP4s
instead:

  1. ``LeRobotEpisodeWriter(dataset)`` reserves the next ``episode_index`` and
     exposes ``target_mp4_paths`` — the locations under
     ``<root>/videos/chunk-NNN/<key>/episode_NNNNNN.mp4`` where caller-side
     imageio writers should stream frames directly.
  2. Per-step state/action is buffered via ``add_step``.
  3. ``commit(prompt)`` patches ``LeRobotDataset._save_image`` to no-op (so
     ``add_frame`` doesn't write PNGs we don't need), feeds dummy uint8 zeros
     for the image features to satisfy ``validate_frame``, and calls
     ``save_episode``. ``encode_episode_videos`` sees the MP4 already at the
     target path and skips encoding (line 977 of lerobot_dataset.py).
  4. On ``abort()``, any pre-placed MP4s are deleted so they don't leak into
     the dataset on next episode write.

Only ``_save_image`` is monkey-patched; the rest of LeRobot's parquet/meta
machinery is untouched.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import torch

_TRAIL_INSTANCE_RE = re.compile(r"_\d+$")
_PATCHED = False


def clean_target(name: str) -> str:
    """``teacup_178`` -> ``teacup``; idempotent if no trailing instance id."""
    return _TRAIL_INSTANCE_RE.sub("", name).replace("_", " ").strip()


def episode_prompt(target_name: str, template: str, override: str | None = None) -> str:
    if override is not None:
        return override
    return template.format(target=target_name, target_clean=clean_target(target_name))


def lerobot_features(resolution: int) -> dict:
    return {
        "image_left":  {"dtype": "video", "shape": (resolution, resolution, 3),
                        "names": ["height", "width", "channel"]},
        "image_right": {"dtype": "video", "shape": (resolution, resolution, 3),
                        "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "video", "shape": (resolution, resolution, 3),
                        "names": ["height", "width", "channel"]},
        "state": {
            "dtype": "float32", "shape": (8,),
            "names": ["eef_x", "eef_y", "eef_z",
                      "axisangle_x", "axisangle_y", "axisangle_z",
                      "gripper_l", "gripper_r"],
        },
        "actions": {
            "dtype": "float32", "shape": (7,),
            "names": ["dpos_x", "dpos_y", "dpos_z",
                      "drot_x", "drot_y", "drot_z", "gripper"],
        },
    }


def _import_dataset_cls():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    return LeRobotDataset


def _png_path_to_mp4_path(png_path) -> Path | None:
    """Given a PNG path like ``<root>/images/<key>/episode_NNNNNN/frame_MMMMMM.png``,
    return the matching MP4 ``<root>/videos/chunk-XXX/<key>/episode_NNNNNN.mp4``
    by globbing under videos/."""
    p = Path(png_path)
    if p.parent.parent.parent.name != "images":
        return None
    root = p.parent.parent.parent.parent
    key = p.parent.parent.name
    ep_name = p.parent.name  # episode_NNNNNN
    matches = list((root / "videos").glob(f"*/{key}/{ep_name}.mp4"))
    return matches[0] if matches else None


def _decode_mp4_frames(mp4_path, indices: list[int]):
    """Read frames at the given 0-based indices from an MP4 file. Returns
    (N, C, H, W) uint8 array."""
    import av
    import numpy as np
    indices_set = set(int(i) for i in indices)
    max_idx = max(indices_set)
    out: dict[int, np.ndarray] = {}
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in indices_set:
                arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                out[i] = arr.transpose(2, 0, 1)  # (C, H, W)
                if len(out) == len(indices_set):
                    break
            if i > max_idx:
                break
    return np.stack([out[int(i)] for i in indices], axis=0)


def apply_no_png_patch() -> None:
    """Patch LeRobot's PNG-writing + PNG-reading paths to use the pre-placed
    MP4 instead. Idempotent.

    Two patches:
    1. ``LeRobotDataset._save_image`` becomes a no-op so ``add_frame`` writes
       no PNGs.
    2. ``compute_stats.sample_images`` is replaced with an MP4-aware version
       that derives the target MP4 from the PNG paths and decodes the sampled
       frames directly. This is the only place where LeRobot reads the PNGs
       back during episode commit (for stats.json).

    Parquet writing and ``encode_episode_videos`` were already safe: parquet
    skips video features entirely (see ``get_hf_features_from_features`` line
    366), and ``encode_episode_videos`` skips encoding if the MP4 already
    exists at the target path (line 977 of lerobot_dataset.py).
    """
    global _PATCHED
    if _PATCHED:
        return
    LeRobotDataset = _import_dataset_cls()

    def _noop(self, image, fpath):  # noqa: ARG001
        return None

    LeRobotDataset._save_image = _noop

    # Neutralize LeRobot's HF version-check on open. By default, opening an
    # existing dataset calls `get_safe_version(repo_id)` which hits
    # `api.list_repo_refs` — fails offline or when repo_id is a local-only
    # label like "maniguard/validate". For our use the repo_id is just a tag;
    # the dataset version comes from the on-disk meta/info.json.
    try:
        import lerobot.common.datasets.utils as _lu
        import lerobot.common.datasets.lerobot_dataset as _lds
    except ModuleNotFoundError:
        import lerobot.datasets.utils as _lu  # type: ignore
        import lerobot.datasets.lerobot_dataset as _lds  # type: ignore

    def _local_get_safe_version(repo_id, version):  # noqa: ARG001
        return str(version) if not isinstance(version, str) else version

    _lu.get_safe_version = _local_get_safe_version
    _lds.get_safe_version = _local_get_safe_version

    try:
        import lerobot.common.datasets.compute_stats as _cs
    except ModuleNotFoundError:
        import lerobot.datasets.compute_stats as _cs  # type: ignore

    _orig_sample = _cs.sample_images

    def _mp4_aware_sample_images(image_paths):
        # If PNGs exist (legacy fat-write), fall back to the original impl.
        if image_paths and Path(image_paths[0]).is_file():
            return _orig_sample(image_paths)
        mp4 = _png_path_to_mp4_path(image_paths[0]) if image_paths else None
        if mp4 is None or not mp4.is_file():
            raise FileNotFoundError(
                f"Neither PNG ({image_paths[0] if image_paths else '?'}) nor "
                f"matching MP4 found for stats computation"
            )
        sampled_indices = _cs.sample_indices(len(image_paths))
        imgs = _decode_mp4_frames(mp4, sampled_indices)
        # Apply the same downsample the original does post-load.
        return np.stack(
            [_cs.auto_downsample_height_width(img) for img in imgs], axis=0
        )

    _cs.sample_images = _mp4_aware_sample_images
    _PATCHED = True


def lerobot_features_joint(resolution: int) -> dict:
    """Schema variant with absolute-joint state + actions (DROID-style).

    state   = [arm_q(7), gripper_pos(1)]            (current joint config)
    actions = [arm_q_target(7), gripper_cmd(1)]     (absolute next-step target)
    """
    f = lerobot_features(resolution)
    f["state"] = {
        "dtype": "float32", "shape": (8,),
        "names": [f"joint_{i}" for i in range(7)] + ["gripper_pos"],
    }
    f["actions"] = {
        "dtype": "float32", "shape": (8,),
        "names": [f"joint_{i}_target" for i in range(7)] + ["gripper_cmd"],
    }
    return f


def create_or_open_dataset(repo_id: str, root: str | Path | None,
                           fps: int, resolution: int,
                           apply_passthrough: bool = True,
                           features: dict | None = None):
    """Open an existing LeRobot v2.1 dataset at ``root``, or create one.

    ``apply_passthrough=True`` (default) installs the no-PNG / MP4-aware-stats
    patches needed when MP4s are pre-placed at the target paths (the live
    SFTRecorder path). Set False for the legacy batch exporter that actually
    needs LeRobot to encode from in-memory frames.

    ``features`` overrides the default eef-delta schema (e.g.
    ``lerobot_features_joint`` for absolute-joint state/actions).
    """
    if apply_passthrough:
        apply_no_png_patch()
    LeRobotDataset = _import_dataset_cls()
    feats = features if features is not None else lerobot_features(resolution)
    root_p = Path(root) if root else None
    # Reopen-existing requires both meta/info.json AND at least one
    # committed parquet under data/. LeRobotDataset cannot load an
    # initialised-but-empty dataset (HF datasets layer demands at least
    # one data file). An "empty shell" (info.json but no parquets — left
    # by an all-abort run or fresh create with no commits) is wiped and
    # recreated so the next process starts clean.
    if root_p is not None and (root_p / "meta" / "info.json").is_file():
        has_data = (root_p / "data").is_dir() and any(
            (root_p / "data").rglob("*.parquet")
        )
        if has_data:
            return LeRobotDataset(repo_id=repo_id, root=root_p)
        # LeRobotDataset.create() uses ``mkdir(exist_ok=False)`` on the
        # root, so wipe the whole shell rather than just the subdirs.
        import shutil as _sh
        _sh.rmtree(root_p)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps,
        features=feats,
        root=root_p, use_videos=True, robot_type="FrankaPanda",
    )
    return ds


class LeRobotEpisodeWriter:
    """Per-episode writer that uses pre-rendered MP4s.

    Lifecycle::

        writer = LeRobotEpisodeWriter(dataset)
        # caller writes MP4s directly to writer.target_mp4_paths during step
        for state, action in steps:
            writer.add_step(state, action)
        if success:
            writer.commit(prompt)   # writes parquet+meta, MP4s already in place
        else:
            writer.abort()          # deletes any pre-placed MP4s
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.episode_index = dataset.meta.total_episodes
        self.target_mp4_paths: dict[str, Path] = {
            key: dataset.root / dataset.meta.get_video_file_path(self.episode_index, key)
            for key in dataset.meta.video_keys
        }
        for p in self.target_mp4_paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    def add_step(self, state, action) -> None:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._states)

    def commit(self, prompt: str, extra_episode_meta: dict | None = None) -> int:
        """Verify MP4s exist, push frames through dataset, save episode.

        ``extra_episode_meta`` — optional dict of keys to add to the just-
        written line in ``meta/episodes.jsonl``. Useful for per-episode tags
        the trainer should filter on (e.g. ``{"ltl_violated": True}``).
        LeRobot ignores unknown keys when loading; the trainer reads them.
        """
        if not self._states:
            raise RuntimeError("commit() called with no buffered steps")
        for key, path in self.target_mp4_paths.items():
            if not path.is_file():
                raise RuntimeError(
                    f"target MP4 missing for {key!r}: {path}. "
                    f"Did the SFTRecorder write to writer.target_mp4_paths?"
                )

        shape = self.dataset.features["image_left"]["shape"]
        dummy = np.zeros(shape, dtype=np.uint8)
        for s, a in zip(self._states, self._actions, strict=True):
            self.dataset.add_frame({
                "image_left": dummy,
                "image_right": dummy,
                "wrist_image": dummy,
                "state": s,
                "actions": a,
                "task": prompt,
            })
        n = len(self._states)
        self.dataset.save_episode()
        if extra_episode_meta:
            self._patch_last_episode_entry(extra_episode_meta)
        self._states.clear()
        self._actions.clear()
        return n

    def _patch_last_episode_entry(self, extra: dict) -> None:
        """Merge ``extra`` into the last line of ``meta/episodes.jsonl``."""
        import json
        ep_file = self.dataset.root / "meta" / "episodes.jsonl"
        lines = ep_file.read_text().splitlines()
        if not lines:
            return
        last = json.loads(lines[-1])
        last.update(extra)
        lines[-1] = json.dumps(last)
        ep_file.write_text("\n".join(lines) + "\n")

    @staticmethod
    def safe_episode_indices(dataset_root) -> list[int]:
        """Read ``meta/episodes.jsonl`` and return episode_index values that
        are NOT flagged ``ltl_violated``. Use this at trainer load time to
        filter unsafe trajectories.
        """
        import json
        from pathlib import Path
        ep_file = Path(dataset_root) / "meta" / "episodes.jsonl"
        if not ep_file.is_file():
            return []
        kept = []
        for line in ep_file.read_text().splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            if not ep.get("ltl_violated", False):
                kept.append(ep["episode_index"])
        return kept

    def abort(self) -> None:
        """Drop buffered state/action and remove any pre-placed MP4s."""
        for p in self.target_mp4_paths.values():
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        # Also remove any orphaned images dir the dataset may have started
        # creating before the no-op patch landed (defensive).
        img_dir = Path(self.dataset.root) / "images"
        if img_dir.is_dir() and not any(img_dir.iterdir()):
            shutil.rmtree(img_dir, ignore_errors=True)
        self._states.clear()
        self._actions.clear()
