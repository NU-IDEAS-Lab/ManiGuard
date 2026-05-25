#!/usr/bin/env python3
"""One-off: convert a fat-HDF5+sibling-MP4 PnP collection to LeRobot v2.1.

Target layout produced by ``tools.pick_and_place_from_dataset --record-sft``
before the live-LeRobot path landed:

    outputs/variants_n10x2_first25/
      task_NNNN/
        seed_MM/
          variant_PP/
            rollout.hdf5             (state, action, image arrays, sim states)
            rollout_image_left.mp4   (h264 from imageio)
            rollout_image_right.mp4
            rollout_wrist.mp4

This script:

  1. Discovers ``variant_*/rollout.hdf5`` files recursively under --input-dir.
  2. For each variant:
       * hardlinks the three MP4s into the LeRobot dataset's expected paths
         (``videos/chunk-NNN/<feature_key>/episode_NNNNNN.mp4``). Same
         filesystem → zero data copied, just an inode reference.
       * reads ``state`` + ``action`` from the HDF5 (image arrays ignored —
         LeRobot's video is now the pixel carrier).
       * commits via :class:`sentinel.data.lerobot_writer.LeRobotEpisodeWriter`
         which calls ``save_episode`` under the no-PNG / MP4-aware-stats
         patches, so the only disk work is the parquet + meta updates.
  3. Rewrites the source HDF5 in place to drop ``obs/image_*`` datasets.
     The thin HDF5 still carries ``states`` / ``action`` / ``datagen_info`` /
     ``env_args`` / ``model_file`` — everything sentinel.datagen reads.
     The source MP4s are then replaced with symlinks pointing to their
     LeRobot home (so the variant dir still has reviewable artifacts).

Per-variant cost ~ HDF5 read (50ms) + 3× hardlink (<1ms) + parquet write
(100ms) + HDF5 strip (500ms-1s). 448 variants converts in ~5-10 minutes.

Re-runnability: refuses to run if the target dataset is non-empty, unless
``--append`` is passed. Default is to start fresh — ``rm -rf
outputs/lerobot_pnp_clutter_first25/`` before invoking.

Usage::

    conda run -n behavior --no-capture-output python tools/lerobot_from_mp4s.py \\
        --input-dir outputs/variants_n10x2_first25 \\
        --repo-id sentinel/clutter_pickup_n10x2_first25 \\
        --root outputs/lerobot_pnp_clutter_first25
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from sentinel.data.lerobot_writer import (
    LeRobotEpisodeWriter,
    create_or_open_dataset,
    episode_prompt,
)

# rollout_<filename>.mp4 (in the variant dir) -> LeRobot feature key
_MP4_MAP = {
    "rollout_image_left.mp4":  "image_left",
    "rollout_image_right.mp4": "image_right",
    "rollout_wrist.mp4":       "wrist_image",
}

_IMAGE_LEAVES = {"image_left", "image_right", "wrist_image"}


def _decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def _read_state_action(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(str(path), "r") as f:
        target_name = _decode(f.attrs.get("target_name", "object"))
        demo_keys = sorted(f["data"].keys())
        if len(demo_keys) != 1:
            raise RuntimeError(f"{path}: expected one demo, got {demo_keys}")
        demo = f["data"][demo_keys[0]]
        state = np.asarray(demo["obs/state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)
    return state, action, str(target_name)


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink src to dst when same-fs; else copy. Returns the mode used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def _strip_images_in_place(path: Path) -> int:
    """Rewrite the HDF5 minus ``data/<demo>/obs/{image_left,image_right,
    wrist_image}``. Returns bytes saved.

    h5py does not reclaim space on dataset delete, so we copy to a temp
    file (filtering the visit) then atomic-rename onto the original.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    old_size = path.stat().st_size

    with h5py.File(str(path), "r") as src, h5py.File(str(tmp), "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                grp = dst.require_group(name)
                for k, v in obj.attrs.items():
                    grp.attrs[k] = v
            elif isinstance(obj, h5py.Dataset):
                leaf = name.rsplit("/", 1)[-1]
                if leaf in _IMAGE_LEAVES and "/obs/" in name:
                    return None
                parent = name.rsplit("/", 1)[0] if "/" in name else ""
                parent_grp = dst.require_group(parent) if parent else dst
                ds = parent_grp.create_dataset(
                    leaf, data=obj[()],
                    compression=obj.compression,
                    compression_opts=obj.compression_opts,
                )
                for k, v in obj.attrs.items():
                    ds.attrs[k] = v
            return None

        src.visititems(visit)

    new_size = tmp.stat().st_size
    os.replace(tmp, path)
    return old_size - new_size


def _is_already_converted(variant_dir: Path) -> bool:
    """A variant whose source MP4s are symlinks has been processed before."""
    return all((variant_dir / fname).is_symlink() for fname in _MP4_MAP)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Root containing variant_*/rollout.hdf5 (recursive).")
    p.add_argument("--repo-id", required=True,
                   help="LeRobot repo id, e.g. sentinel/clutter_pickup_n10x2_first25.")
    p.add_argument("--root", required=True, type=Path,
                   help="LeRobot dataset root on disk.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--prompt-template",
                   default="pick up the {target_clean} in the middle of "
                           "the table and place it at the green goal")
    p.add_argument("--no-strip", action="store_true",
                   help="Skip the HDF5 image-array strip pass.")
    p.add_argument("--no-symlink", action="store_true",
                   help="Don't replace source MP4s with symlinks after "
                        "hardlinking. Source dir keeps the original MP4 files.")
    p.add_argument("--append", action="store_true",
                   help="Allow appending to an existing non-empty dataset. "
                        "Default: refuse and exit if dataset has episodes.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input-dir not found: {args.input_dir}")

    dataset = create_or_open_dataset(
        repo_id=args.repo_id, root=args.root,
        fps=args.fps, resolution=args.resolution,
    )
    n_existing = dataset.meta.total_episodes
    if n_existing > 0 and not args.append:
        raise SystemExit(
            f"Target dataset already has {n_existing} episodes. "
            f"Pass --append to add to it, or `rm -rf {args.root}` to start fresh."
        )

    variants = sorted(args.input_dir.rglob("variant_*/rollout.hdf5"))
    if not variants:
        # Some collections use a flat layout — fall back to any rollout.hdf5
        variants = sorted(args.input_dir.rglob("rollout.hdf5"))
    if args.limit is not None:
        variants = variants[: args.limit]
    if not variants:
        raise SystemExit(f"no rollout.hdf5 found under {args.input_dir}")

    print(f"[convert] {len(variants)} variants -> {args.root} "
          f"(starting from episode {n_existing})", flush=True)

    t0 = time.time()
    n_skipped = 0
    n_committed = 0
    bytes_saved = 0
    for i, hdf5_path in enumerate(variants):
        vdir = hdf5_path.parent
        rel = vdir.relative_to(args.input_dir)
        if _is_already_converted(vdir):
            n_skipped += 1
            continue

        missing = [fname for fname in _MP4_MAP if not (vdir / fname).is_file()]
        if missing:
            print(f"[convert] SKIP {rel}: missing {missing}", flush=True)
            n_skipped += 1
            continue

        state, action, target_name = _read_state_action(hdf5_path)
        n = len(action)
        if len(state) != n:
            print(f"[convert] SKIP {rel}: len(state)={len(state)} != "
                  f"len(action)={n}", flush=True)
            n_skipped += 1
            continue

        writer = LeRobotEpisodeWriter(dataset)
        for fname, feature_key in _MP4_MAP.items():
            _link_or_copy(vdir / fname, writer.target_mp4_paths[feature_key])
        for t in range(n):
            writer.add_step(state[t], action[t])
        prompt = episode_prompt(target_name, args.prompt_template)
        ep_idx = writer.episode_index
        writer.commit(prompt)
        n_committed += 1

        if not args.no_strip:
            try:
                bytes_saved += _strip_images_in_place(hdf5_path)
            except Exception as e:  # noqa: BLE001
                print(f"[convert] WARN strip failed for {rel}: {e}", flush=True)

        if not args.no_symlink:
            for fname, feature_key in _MP4_MAP.items():
                src = vdir / fname
                tgt = writer.target_mp4_paths[feature_key].resolve()
                try:
                    src.unlink()
                    src.symlink_to(tgt)
                except OSError as e:
                    print(f"[convert] WARN symlink failed for {src}: {e}",
                          flush=True)

        if (i + 1) % 10 == 0 or i + 1 == len(variants):
            elapsed = time.time() - t0
            rate = n_committed / max(elapsed, 1e-6) * 60
            print(f"[convert] [{i+1}/{len(variants)}] ep={ep_idx:06d} "
                  f"n={n} target={target_name} "
                  f"elapsed={elapsed:.1f}s rate={rate:.1f}ep/min "
                  f"saved={bytes_saved/1e9:.2f}GB", flush=True)

    elapsed = time.time() - t0
    print(f"\n[convert] DONE  committed={n_committed} skipped={n_skipped} "
          f"elapsed={elapsed:.1f}s  "
          f"bytes_saved={bytes_saved/1e9:.2f}GB  "
          f"final_total_episodes={dataset.meta.total_episodes} "
          f"final_total_frames={dataset.meta.total_frames}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
