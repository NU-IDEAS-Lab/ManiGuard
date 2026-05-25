#!/usr/bin/env python3
"""Stage 2: rendered SFT HDF5s -> LeRobot v2.1 dataset.

Consumes the success/balanced split produced upstream of this stage:

    <input-root>/
      task_<NNNN>__seed_<MM>/
        rollout.hdf5
        rollout_image_left.mp4
        rollout_image_right.mp4
        rollout_wrist.mp4

and writes a LeRobot dataset at ``${HF_LEROBOT_HOME:-~/.cache/huggingface/lerobot}/<repo_id>``.

Feature layout (matches OpenPI's 3-camera input slots — base_0 / left_wrist /
right_wrist for pi0/pi0.5, or base_0 / base_1 / wrist_0 for pi0-FAST):

    image_left    (video HxWx3 uint8) -> observation/image_left
    image_right   (video HxWx3 uint8) -> observation/image_right
    wrist_image   (video HxWx3 uint8) -> observation/wrist_image
    state         (float32, 8)        -> observation/state
    actions       (float32, 7)        -> actions
    task          (string)            -> prompt (via LeRobot's task index)

Per-episode prompts are derived from the HDF5 root attr ``target_name`` using
``--prompt-template`` (default: "pick up the {target_clean} and place it at the
goal"). ``{target_clean}`` strips the trailing ``_NNN`` suffix that BDDL adds
to instance names (``teacup_178`` -> ``teacup``). A flat ``--prompt`` is also
accepted to override per-episode templating with a single string.

The SFT recorder writes one (state, action) pair per env.step, so action and
state arrays both have length N — no trimming needed.

Environment requirements:
    lerobot == 0.1.0 (git rev pinned by openpi/pyproject.toml — CODEBASE_VERSION
        "v2.1"; later lerobot releases write v3.0 datasets that openpi's pinned
        reader will refuse).
    torch, torchvision, numpy, h5py, pyarrow

Use ``/data/Projects/SENTINEL-Lite/.venv-lerobot-v2`` (already provisioned with
the right lerobot rev) or openpi's own ``openpi/.venv``. Do *not* use the
older ``.venv-lerobot`` (lerobot 0.4.4 — writes v3.0).

    /data/Projects/SENTINEL-Lite/.venv-lerobot-v2/bin/python \
        -m sentinel.data.lerobot_export \
        --input-dir outputs/sft_dataset_2026-05-16/success_balanced \
        --repo-id sentinel/pnp_multitask \
        --fps 30
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch


_TRAIL_INSTANCE_RE = re.compile(r"_\d+$")


def _clean_target(name: str) -> str:
    """``teacup_178`` -> ``teacup``; idempotent if no trailing instance id."""
    return _TRAIL_INSTANCE_RE.sub("", name).replace("_", " ").strip()


def _load_episode_from_hdf5(path: str) -> tuple[list[dict], dict]:
    """Read one rendered HDF5 -> (frames, attrs).

    Expects the SFTRecorder layout:
        data/demo_0/obs/image_left  (N, H, W, 3) uint8
        data/demo_0/obs/image_right (N, H, W, 3) uint8
        data/demo_0/obs/wrist_image (N, H, W, 3) uint8
        data/demo_0/obs/state       (N, 8) float32
        data/demo_0/action          (N, 7) float32
        data/demo_0/done            (N,)   bool
    Root attrs include ``target_name``, ``task_dir``, ``seed``.
    """
    with h5py.File(path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        demo_keys = sorted(f["data"].keys())
        if len(demo_keys) != 1:
            raise RuntimeError(f"{path}: expected exactly one demo, got {demo_keys}")
        demo = f["data"][demo_keys[0]]
        image_left = np.asarray(demo["obs/image_left"])
        image_right = np.asarray(demo["obs/image_right"])
        wrist_image = np.asarray(demo["obs/wrist_image"])
        state = np.asarray(demo["obs/state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)

    n = len(action)
    if not (len(state) == len(image_left) == len(image_right) == len(wrist_image) == n):
        raise RuntimeError(
            f"{path}: length mismatch — action={n}, state={len(state)}, "
            f"image_left={len(image_left)}, image_right={len(image_right)}, "
            f"wrist={len(wrist_image)}"
        )

    frames = []
    for t in range(n):
        frames.append({
            "image_left": torch.from_numpy(image_left[t]),
            "image_right": torch.from_numpy(image_right[t]),
            "wrist_image": torch.from_numpy(wrist_image[t]),
            "state": torch.from_numpy(state[t]),
            "actions": torch.from_numpy(action[t]),
        })
    return frames, attrs


def _decode_attr(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def _episode_prompt(attrs: dict, template: str, override: str | None) -> str:
    if override is not None:
        return override
    target = _decode_attr(attrs.get("target_name", "object"))
    target_clean = _clean_target(target)
    return template.format(target=target, target_clean=target_clean)


def _find_episodes(input_dir: str) -> list[str]:
    # Recursive discovery — handles both the flat
    # <input-dir>/task_*__seed_*/rollout.hdf5 layout AND the nested
    # transport-variants layout
    # <input-dir>/task_*/seed_*/variant_*/rollout.hdf5.
    import pathlib
    nested = sorted(str(p) for p in pathlib.Path(input_dir).rglob("rollout.hdf5"))
    if nested:
        return nested
    # Fallback: flat directory of *.hdf5 (legacy teleop_rendered layout)
    flat = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    return flat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="outputs/sft_dataset_2026-05-16/success_balanced",
                   help="Directory containing per-episode subdirs <task>__<seed>/rollout.hdf5, "
                        "or a flat dir of *.hdf5 (legacy).")
    p.add_argument("--repo-id", required=True,
                   help="LeRobot dataset identifier, e.g. sentinel/pnp_multitask")
    p.add_argument("--prompt", default=None,
                   help="If set, applied to every episode (overrides --prompt-template).")
    p.add_argument("--prompt-template", default="pick up the {target_clean} and place it at the goal",
                   help="Per-episode prompt template. Substitutions: {target} (raw BDDL "
                        "instance name like 'teacup_178'), {target_clean} (instance suffix "
                        "stripped, underscores -> spaces).")
    p.add_argument("--root", default=None,
                   help="Root dir for the dataset folder. Defaults to $HF_LEROBOT_HOME "
                        "or ~/.cache/huggingface/lerobot.")
    p.add_argument("--fps", type=int, default=30,
                   help="Playback FPS recorded in dataset metadata.")
    p.add_argument("--resolution", type=int, default=256,
                   help="Image side length (H = W). Must match SFTRecorder output.")
    p.add_argument("--single", type=str, default=None,
                   help="If set, only ingest this one HDF5 file (smoke test).")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only ingest the first N episodes (smoke test).")
    p.add_argument("--push-to-hub", default=None,
                   help="If set (e.g. IDEAS-Lab-Northwestern/sentinel-pnp-multitask), "
                        "push the dataset via LeRobot's push_to_hub() after building. "
                        "This auto-creates the codebase_version git tag that openpi "
                        "requires (plain huggingface_hub.upload_folder does not).")
    p.add_argument("--hub-private", action="store_true",
                   help="Push the HF dataset as a private repo.")
    args = p.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"lerobot not importable: {e}\n"
                f"Install in an isolated venv (see this script's docstring)."
            )

    # Feature spec. Flat names line up with a RepackTransform that maps:
    #   observation/image_left  -> image_left
    #   observation/image_right -> image_right
    #   observation/wrist_image -> wrist_image
    #   observation/state       -> state
    #   actions                 -> actions
    features = {
        "image_left": {
            "dtype": "video",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "image_right": {
            "dtype": "video",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "eef_x", "eef_y", "eef_z",
                "axisangle_x", "axisangle_y", "axisangle_z",
                "gripper_l", "gripper_r",
            ],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dpos_x", "dpos_y", "dpos_z", "drot_x", "drot_y", "drot_z", "gripper"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
        robot_type="FrankaPanda",
    )

    if args.single is not None:
        input_files = [args.single]
    else:
        input_files = _find_episodes(args.input_dir)
    if not input_files:
        raise SystemExit(
            f"No episodes found under {args.input_dir} "
            f"(looked for */rollout.hdf5 and *.hdf5)."
        )
    if args.limit is not None:
        input_files = input_files[: args.limit]

    print(f"[Stage2] Writing {len(input_files)} episode(s) to {dataset.root}")
    input_root = Path(args.input_dir).resolve()
    for i, path in enumerate(input_files):
        frames, attrs = _load_episode_from_hdf5(path)
        prompt = _episode_prompt(attrs, args.prompt_template, args.prompt)
        if Path(path).name == "rollout.hdf5":
            try:
                rel = Path(path).resolve().parent.relative_to(input_root)
                ep_tag = str(rel).replace(os.sep, "__") or Path(path).parent.name
            except ValueError:
                ep_tag = Path(path).parent.name
        else:
            ep_tag = Path(path).stem
        print(f"  [{i+1}/{len(input_files)}] {ep_tag}  (n={len(frames)})  prompt={prompt!r}")
        for frame in frames:
            frame["task"] = prompt
            dataset.add_frame(frame)
        dataset.save_episode()

    print(f"\n[Stage2] Done. Dataset root: {dataset.root}")
    print(f"[Stage2] repo_id: {args.repo_id}")
    print(f"[Stage2] Episodes: {len(input_files)}")

    if args.push_to_hub:
        print(f"\n[Stage2] Pushing to HF: {args.push_to_hub}")
        dataset.repo_id = args.push_to_hub
        dataset.push_to_hub(
            tags=["panda", "omnigibson", "sim", "sentinel", "pnp"],
            license="apache-2.0",
            private=args.hub_private,
            push_videos=True,
            tag_version=True,
        )
        print("[Stage2] Pushed + codebase_version tag auto-created.")


if __name__ == "__main__":
    main()
