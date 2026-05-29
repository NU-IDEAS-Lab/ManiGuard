#!/usr/bin/env python3
"""Multi-task Stage 2: per-task rendered HDF5 -> single LeRobot v2.1 dataset.

Walks a directory tree of the form

    <input_root>/<category>/<task_id>/<base>/scene_ep*.hdf5

(e.g. ``outputs/gello_teleop_rendered/table/task_0000/base/scene_ep1.hdf5``)
and looks up the per-task language prompt from a parallel diagnostics tree

    <diag_root>/<category>/<task_id>/<base>/diagnostics.jsonl

(each line has ``"prompt": "..."`` among other fields).

All episodes from all tasks are merged into ONE LeRobot dataset whose
``meta/tasks.jsonl`` enumerates the unique prompt strings, and each
frame's ``task_index`` resolves to the right prompt at training time.

Schema written matches ``maniguard/data/lerobot/lerobot_export.py`` (LIBERO-compatible
columns), so openpi's ``LeRobotLiberoDataConfig`` can repack it directly
IF action is 7D. For 8D action (e.g. gello joint-target + gripper), see
the action_dim flag below.

Usage:
    .venv-lerobot/bin/python -m maniguard.data.lerobot.multitask_lerobot_export \\
        --input-root outputs/gello_teleop_rendered/table \\
        --diag-root datasets/final_unique_accepted-goal_region_sphere-full-perturbed_with_base-20260426/table \\
        --repo-id maniguard/clutter_pickup_libero \\
        --root outputs/lerobot_datasets/maniguard/clutter_pickup_libero \\
        --action-dim 8 \\
        --push-to-hub IDEAS-Lab-Northwestern/sim-clutter-pickup \\
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


def _load_episode_from_hdf5(
    path: Path, eef_delta: bool = False
) -> tuple[list[dict], int]:
    """Read one rendered HDF5 -> list of per-frame dicts (frame['task'] is set later).

    Returns (frames, n_action) where N+1 frames of state are paired with
    N actions (last state dropped -- no action follows it).

    If eef_delta=True, replaces the raw joint-target actions with 7D
    EEF-delta actions computed from consecutive states.
    """
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if len(demo_keys) != 1:
            raise RuntimeError(f"{path}: expected 1 demo, got {demo_keys}")
        demo = f["data"][demo_keys[0]]
        image = np.asarray(demo["obs/image"])
        wrist_image = np.asarray(demo["obs/wrist_image"])
        state = np.asarray(demo["obs/state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)

    n_action = len(action)

    if eef_delta:
        action = _compute_eef_delta_actions(
            state[: n_action + 1], action[:, -1]
        )

    image = image[:n_action]
    wrist_image = wrist_image[:n_action]
    state = state[:n_action]

    frames = []
    for t in range(n_action):
        frames.append({
            "image": torch.from_numpy(image[t]),
            "wrist_image": torch.from_numpy(wrist_image[t]),
            "state": torch.from_numpy(state[t]),
            "actions": torch.from_numpy(action[t]),
        })
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
    p.add_argument("--state-dim", type=int, default=8)
    p.add_argument("--action-dim", type=int, default=7,
                   help="7 for EEF-delta (matches LIBERO out-of-the-box); 8 for joint-target style.")
    p.add_argument("--eef-delta-actions", action="store_true",
                   help="Convert joint-target actions to 7D EEF-delta (dpos+drot+gripper) "
                        "computed from consecutive EEF states. Forces action-dim=7.")
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

    if args.eef_delta_actions:
        args.action_dim = 7

    # ----- discover episodes (flat or nested layout) -----
    episodes = _discover_episodes(args.input_root, args.subdir)
    if not episodes:
        raise SystemExit(f"No rendered episodes found under {args.input_root}")
    by_task: dict[str, list[Path]] = defaultdict(list)
    for tid, ep_path in episodes:
        by_task[tid].append(ep_path)
    print(f"[Multitask] {len(episodes)} episodes across {len(by_task)} tasks "
          f"under {args.input_root}")

    # ----- prepare LeRobot dataset -----
    state_names = [f"state_{i}" for i in range(args.state_dim)]
    if args.state_dim == 8:
        state_names = ["eef_x", "eef_y", "eef_z",
                       "axisangle_x", "axisangle_y", "axisangle_z",
                       "gripper_l", "gripper_r"]
    if args.eef_delta_actions:
        action_names = ["dpos_x", "dpos_y", "dpos_z",
                        "drot_x", "drot_y", "drot_z", "gripper"]
    else:
        action_names = [f"action_{i}" for i in range(args.action_dim)]

    features = {
        "image": {
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
            "shape": (args.state_dim,),
            "names": state_names,
        },
        "actions": {
            "dtype": "float32",
            "shape": (args.action_dim,),
            "names": action_names,
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

    # ----- iterate tasks -----
    total_eps = 0
    total_frames = 0
    prompts_seen: set[str] = set()

    for tid in sorted(by_task):
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
            frames, n = _load_episode_from_hdf5(ep_path, eef_delta=args.eef_delta_actions)
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
