#!/usr/bin/env python3
"""Multi-task Stage 2: rendered teleop HDF5s -> one LeRobot v2.1 dataset.

Discovers rendered episodes under --input-root (both layouts auto-detected):
    flat:   <input_root>/task_NNNN_traj_MMM.hdf5      (maniguard.data.playback)
    nested: <input_root>/task_NNNN/<subdir>/scene_ep*.hdf5

and looks up each task's language prompt at
    <diag_root>/<task_id>/diagnostics.jsonl

All episodes merge into ONE dataset whose meta/tasks.jsonl enumerates the unique
prompts; each frame's task_index resolves to the right prompt at train time.

The schema is auto-selected from the playback fingerprint stamped on each HDF5
(controller_mode + n_cams) -- no per-run schema flags:
    controller=joint -> state [joint_0..6, gripper_pos] (8),
                        actions [joint_*_target, gripper_cmd] (8, passthrough)
    controller=eef   -> state eef_8d, actions [dpos, drot, gripper] (7, delta)
    n_cams=3         -> image_left + image_right + wrist_image
    n_cams=2         -> image + wrist_image

Usage:
    .venv-lerobot/bin/python -m maniguard.data.lerobot.multitask_lerobot_export \\
        --input-root outputs/teleop_rendered_maniguard-demo/dusty_transfer \\
        --diag-root outputs/benchmark_base_task_sets_reviewed/05_HF_6fam-base/dusty_transfer \\
        --repo-id IDEAS-Lab-Northwestern/sim-dusty-transfer-joint \\
        --root outputs/lerobot_datasets/sim-dusty-transfer-joint \\
        --push-to-hub IDEAS-Lab-Northwestern/sim-dusty-transfer-joint \\
        --hub-private
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

# Flat rendered-file naming: ``task_NNNN_traj_MMM.hdf5`` (one episode per file,
# as produced by maniguard.data.playback). The captured group is the task id
# used to look up the prompt in the diagnostics tree.
_FLAT_RE = re.compile(r"^(task_\d+)_traj_\d+$")


def _load_prompt(diag_path: Path) -> str:
    """Extract ``prompt`` from a diagnostics file.

    Handles two on-disk shapes: a pretty-printed single JSON object (the
    6fam-base benchmark tasks, multi-line) and JSONL (one compact record per
    line). Tries whole-file JSON first, then falls back to the first line.
    """
    text = diag_path.read_text()
    if not text.strip():
        raise RuntimeError(f"{diag_path} is empty")
    try:
        rec = json.loads(text)
    except json.JSONDecodeError:
        rec = json.loads(text.splitlines()[0])
    if "prompt" not in rec:
        raise KeyError(f"{diag_path} lacks 'prompt'; keys={list(rec.keys())}")
    return rec["prompt"]


def _discover_episodes(input_root: Path, subdir: str) -> list[tuple[str, Path]]:
    """Find rendered episode HDF5s and map each to its task id.

    Supports two layouts (auto-detected):
      flat:   ``<input_root>/task_NNNN_traj_MMM.hdf5``   (one ep per file)
      nested: ``<input_root>/task_NNNN/<subdir>/scene_ep*.hdf5``

    Returns a list of ``(task_id, episode_path)`` pairs.
    """
    flat = sorted(input_root.glob("task_*_traj_*.hdf5"))
    if flat:
        pairs = []
        for p in flat:
            m = _FLAT_RE.match(p.stem)
            if m:
                pairs.append((m.group(1), p))
        return pairs

    pairs = []
    for d in sorted(input_root.iterdir()):
        if not (d.is_dir() and d.name.startswith("task_")):
            continue
        for p in sorted((d / subdir).glob("scene_ep*.hdf5")):
            pairs.append((d.name, p))
    return pairs


def _axisangle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix (3, 3) via Rodrigues' formula."""
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3, dtype=aa.dtype)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=aa.dtype)
    return np.eye(3, dtype=aa.dtype) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> axis-angle (3,)."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=R.dtype)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]], dtype=R.dtype) / (2.0 * np.sin(theta))
    return axis * theta


def _compute_eef_delta_actions(
    states: np.ndarray, gripper_cmds: np.ndarray
) -> np.ndarray:
    """Compute 7D EEF-delta actions from N+1 states and N gripper commands.

    states:       (N+1, 8) — eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
    gripper_cmds: (N,) — binary gripper commands from original action dim 7

    Returns: (N, 7) — delta_pos(3) + delta_rot_axisangle(3) + gripper(1)
    """
    n = len(gripper_cmds)
    actions = np.zeros((n, 7), dtype=np.float32)
    for t in range(n):
        actions[t, :3] = states[t + 1, :3] - states[t, :3]
        R_t = _axisangle_to_rotmat(states[t, 3:6])
        R_next = _axisangle_to_rotmat(states[t + 1, 3:6])
        R_delta = R_next @ R_t.T
        actions[t, 3:6] = _rotmat_to_axisangle(R_delta)
        actions[t, 6] = gripper_cmds[t]
    return actions


# Schema tables, indexed by the playback fingerprint. These ARE the
# "hardcoded per-config" mappings -- the export reads the stamp and looks the
# schema up here, no per-run flags. Image obs keys are in dataset column order
# (rendered HDF5 stores them under obs/<key>); state is always 8D.
_IMAGE_KEYS = {
    2: ["image", "wrist_image"],
    3: ["image_left", "image_right", "wrist_image"],
}
_STATE_NAMES = {
    "joint": [f"joint_{i}" for i in range(7)] + ["gripper_pos"],
    "eef": ["eef_x", "eef_y", "eef_z",
            "axisangle_x", "axisangle_y", "axisangle_z",
            "gripper_l", "gripper_r"],
}
_ACTION_NAMES = {
    "joint": [f"joint_{i}_target" for i in range(7)] + ["gripper_cmd"],
    "eef": ["dpos_x", "dpos_y", "dpos_z", "drot_x", "drot_y", "drot_z", "gripper"],
}


def _read_stamp(path: Path) -> tuple[str, int]:
    """Read the playback fingerprint (controller_mode, n_cams) from a rendered
    HDF5's ``data`` group. Raises if absent -- the schema can't be inferred from
    the arrays alone (eef and joint state are both 8D float)."""
    with h5py.File(path, "r") as f:
        attrs = f["data"].attrs
        if "controller_mode" not in attrs or "n_cams" not in attrs:
            raise SystemExit(
                f"{path}: missing controller_mode/n_cams stamp. Re-render with the "
                f"current maniguard.data.playback (it fingerprints the output)."
            )
        return str(attrs["controller_mode"]), int(attrs["n_cams"])


def _build_features(controller: str, n_cams: int, resolution: int) -> dict:
    """LeRobot feature schema for a (controller, n_cams) combination."""
    feats = {
        k: {"dtype": "video", "shape": (resolution, resolution, 3),
            "names": ["height", "width", "channel"]}
        for k in _IMAGE_KEYS[n_cams]
    }
    feats["state"] = {"dtype": "float32", "shape": (8,),
                      "names": _STATE_NAMES[controller]}
    feats["actions"] = {"dtype": "float32", "shape": (len(_ACTION_NAMES[controller]),),
                        "names": _ACTION_NAMES[controller]}
    return feats


def _load_episode_from_hdf5(
    path: Path, controller: str, n_cams: int
) -> tuple[list[dict], int]:
    """Read one rendered HDF5 -> list of per-frame dicts (task set later).

    N+1 obs frames pair with N actions (last obs dropped). For ``eef`` the raw
    joint-target action is converted to a 7D EEF-delta from consecutive eef
    states; for ``joint`` the raw 8D joint-target action passes through.
    """
    img_keys = _IMAGE_KEYS[n_cams]
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if len(demo_keys) != 1:
            raise RuntimeError(f"{path}: expected 1 demo, got {demo_keys}")
        demo = f["data"][demo_keys[0]]
        images = {k: np.asarray(demo[f"obs/{k}"]) for k in img_keys}
        state = np.asarray(demo["obs/state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)

    n_action = len(action)
    if controller == "eef":
        action = _compute_eef_delta_actions(state[: n_action + 1], action[:, -1])

    frames = []
    for t in range(n_action):
        frame = {k: torch.from_numpy(images[k][t]) for k in img_keys}
        frame["state"] = torch.from_numpy(state[t])
        frame["actions"] = torch.from_numpy(action[t])
        frames.append(frame)
    return frames, n_action


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-root", type=Path, required=True,
                   help="e.g. outputs/gello_teleop_rendered/table")
    p.add_argument("--diag-root", type=Path, required=True,
                   help="parallel tree to --input-root with diagnostics.jsonl per task")
    p.add_argument("--subdir", default="base",
                   help="Subdirectory within each task_id holding scene_ep*.hdf5 + diagnostics.jsonl")
    p.add_argument("--repo-id", required=True, help="Logical LeRobot repo id (metadata only)")
    p.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--push-to-hub", default=None,
                   help="HF repo id to push to (auto-creates v2.1 codebase tag).")
    p.add_argument("--hub-private", action="store_true")
    p.add_argument("--limit-per-task", type=int, default=None,
                   help="(Debug) cap episodes ingested per task_id.")
    args = p.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as e:
            raise SystemExit(f"lerobot not importable: {e}")

    # ----- discover episodes (flat or nested layout) -----
    episodes = _discover_episodes(args.input_root, args.subdir)
    if not episodes:
        raise SystemExit(f"No rendered episodes found under {args.input_root}")
    by_task: dict[str, list[Path]] = defaultdict(list)
    for tid, ep_path in episodes:
        by_task[tid].append(ep_path)

    # Auto-detect the schema from the playback fingerprint (one run = one family,
    # homogeneous controller/cams -- read it off the first episode).
    controller, n_cams = _read_stamp(episodes[0][1])
    print(f"[Multitask] {len(episodes)} episodes across {len(by_task)} tasks "
          f"under {args.input_root}  (controller={controller}, n_cams={n_cams})")

    # ----- prepare LeRobot dataset -----
    features = _build_features(controller, n_cams, args.resolution)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
        robot_type="FrankaPanda",
    )

    # ----- iterate tasks -----
    total_eps = 0
    total_frames = 0
    prompts_seen: set[str] = set()

    for tid in sorted(by_task):
        # Prompt lookup honours --subdir: flat-rendered families (task_*_traj_*.hdf5)
        # still keep their per-task diagnostics under <tid>/<subdir>/ in the base-task
        # tree (e.g. 6fam-base/<family>/task_NNNN/base/diagnostics.jsonl). Fall back
        # to <tid>/diagnostics.jsonl for trees that store it flat at the task level.
        diag_path = args.diag_root / tid / args.subdir / "diagnostics.jsonl"
        if not diag_path.exists():
            diag_path = args.diag_root / tid / "diagnostics.jsonl"
        if not diag_path.exists():
            print(f"[Multitask] SKIP {tid}: no diagnostics at {diag_path}")
            continue

        try:
            prompt = _load_prompt(diag_path)
        except Exception as e:
            print(f"[Multitask] SKIP {tid}: prompt parse failed -- {e}")
            continue

        ep_files = by_task[tid]
        if args.limit_per_task is not None:
            ep_files = ep_files[: args.limit_per_task]

        prompts_seen.add(prompt)
        print(f"[{tid}] {len(ep_files)} eps  prompt='{prompt[:60]}...'")

        for ep_path in ep_files:
            frames, n = _load_episode_from_hdf5(ep_path, controller, n_cams)
            for frame in frames:
                # lerobot >=0.3 takes the task as a separate arg, not a frame key.
                dataset.add_frame(frame, task=prompt)
            dataset.save_episode()
            total_eps += 1
            total_frames += n

    print(f"\n[Multitask] Done.")
    print(f"  unique prompts: {len(prompts_seen)}")
    print(f"  episodes      : {total_eps}")
    print(f"  total frames  : {total_frames}")
    print(f"  dataset root  : {dataset.root}")

    if args.push_to_hub:
        print(f"\n[Multitask] Pushing to HF: {args.push_to_hub}")
        dataset.repo_id = args.push_to_hub
        dataset.push_to_hub(
            tags=["panda", "omnigibson", "sim", "maniguard", "multitask"],
            license="apache-2.0",
            private=args.hub_private,
            push_videos=True,
            tag_version=True,
        )
        print(f"[Multitask] Pushed + v2.1 tag auto-created.")


if __name__ == "__main__":
    main()
