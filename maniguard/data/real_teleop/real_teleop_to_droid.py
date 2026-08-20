#!/usr/bin/env python3
"""Real-teleop npz -> LeRobot dataset in openpi DROID schema.

Matches openpi's `LeRobotDROIDDataConfig` expected feature set so the
resulting dataset can be loaded directly by `pi05_droid`-family TrainConfigs.

Reference: openpi examples/droid/convert_droid_data_to_lerobot.py

LeRobot columns written:
    exterior_image_1_left  (video, 180x320x3)   <- cam0 (main), center-cropped to 16:9
    exterior_image_2_left  (video, 180x320x3)   <- ALL-ZERO frames (we have no 2nd exterior)
    wrist_image_left       (video, 180x320x3)   <- cam1 (wrist), center-cropped to 16:9
    joint_position         (float32, (7,))      <- observation/joint_position
    gripper_position       (float32, (1,))      <- observation/gripper_position (normalized 0-1)
    actions                (float32, (8,))      <- [joint_velocity(7), gripper_position[t+1](1)]
    task                   (string)             <- prompt (via LeRobot's task index)

Real-npz schema (input):
    observation/image/cam0           (N+1,) object   JPEG bytes (640x480)
    observation/image/cam1           (N+1,) object   JPEG bytes (640x480)
    observation/joint_position       (N+1, 7) f64
    observation/joint_velocity       (N+1, 7) f64
    observation/gripper_position     (N+1,)  f64    normalized 0-1 (0=open, 1=closed)
    timestamp, meta                  (unused)

Action convention (matches openpi DROID pretrained pi0.5):
    actions[t, 0:7] = joint_velocity[t]        (rad/s)
    actions[t, 7]   = gripper_position[t+1]    (target gripper, normalized)
    -> last frame dropped (no valid action).

Notes:
    - Aspect ratio handled by center-cropping 640x480 -> 640x360 (16:9) then
      resizing to 320x180. Preserves horizontal FOV, crops top/bottom equally.
    - Stored as `dtype: "video"` (not "image") for HF transfer efficiency.
      openpi's data loader reads videos identically.

Usage:
    python -m maniguard.data.real_teleop.real_teleop_to_droid \\
        --input-dir outputs/real_teleop \\
        --repo-id maniguard/real_mug_into_bowl_droid \\
        --prompt "pick up the smallest mug and place it in the bowl" \\
        --root outputs/lerobot_datasets/maniguard/real_mug_into_bowl_droid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DROID_H, DROID_W = 180, 320   # target resolution (matches openpi DROID example)
DROID_FPS = 15                # DROID standard fps


def _decode_crop_resize(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG -> BGR ndarray -> center-crop to 16:9 -> resize to 320x180 -> RGB."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None (corrupt JPEG?)")
    h, w = img.shape[:2]
    # Center-crop to 16:9 aspect ratio (height = 9/16 * width)
    target_h = int(round(w * 9.0 / 16.0))  # 640 -> 360
    if target_h < h:
        off = (h - target_h) // 2
        img = img[off : off + target_h, :, :]
    img = cv2.resize(img, (DROID_W, DROID_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _build_frames(npz_path: Path, prompt: str):
    """Yield one dict per retained frame (N-1 frames; last obs dropped)."""
    d = np.load(npz_path, allow_pickle=True)

    joint_pos = d["observation/joint_position"].astype(np.float32)     # (N+1, 7)
    joint_vel = d["observation/joint_velocity"].astype(np.float32)     # (N+1, 7)
    grip = d["observation/gripper_position"].astype(np.float32)        # (N+1,)
    cam0 = d["observation/image/cam0"]                                  # JPEG bytes
    cam1 = d["observation/image/cam1"]

    n_total = len(joint_pos)
    n_frames = n_total - 1   # drop last (no next gripper target available)

    for t in range(n_frames):
        cam0_b = cam0[t].tobytes() if hasattr(cam0[t], "tobytes") else cam0[t]
        cam1_b = cam1[t].tobytes() if hasattr(cam1[t], "tobytes") else cam1[t]

        ext1 = _decode_crop_resize(cam0_b)
        wrist = _decode_crop_resize(cam1_b)
        ext2 = np.zeros_like(ext1)                          # per user: all zeros

        action = np.concatenate([
            joint_vel[t],                                   # joint_velocity[t]   -> 7D
            np.asarray([grip[t + 1]], dtype=np.float32),    # gripper_pos[t+1]    -> 1D
        ]).astype(np.float32)

        yield {
            "exterior_image_1_left": ext1,
            "exterior_image_2_left": ext2,
            "wrist_image_left": wrist,
            "joint_position": joint_pos[t],
            "gripper_position": np.asarray([grip[t]], dtype=np.float32),
            "actions": action,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Dir with <id>.npz real teleop files")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo_id (logical name; used in meta)")
    parser.add_argument("--prompt", required=True, help="Language instruction stored as task")
    parser.add_argument("--root", type=Path, required=True, help="Output dir for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=DROID_FPS)
    parser.add_argument("--push-to-hub", default=None,
                        help="If set (e.g. IDEAS-Lab-Northwestern/real-mug-into-bowl-droid), "
                             "push the dataset via LeRobot's push_to_hub after building; "
                             "this also creates the codebase_version git tag automatically.")
    parser.add_argument("--hub-private", action="store_true", help="Push as private repo")
    args = parser.parse_args()

    # lerobot 0.3.x (v2.1 codebase, matches openpi's pinned rev) puts LeRobotDataset
    # under lerobot.datasets; 0.4.x keeps same path but writes v3.0 (incompatible).
    # We install lerobot<0.4 so openpi can read our dataset.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if args.root.exists():
        import shutil
        shutil.rmtree(args.root)
    # LeRobotDataset.create() mkdir's root itself and errors if it pre-exists.

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        robot_type="panda",
        fps=args.fps,
        features={
            "exterior_image_1_left": {
                "dtype": "video", "shape": (DROID_H, DROID_W, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "video", "shape": (DROID_H, DROID_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "video", "shape": (DROID_H, DROID_W, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        use_videos=True,
    )

    npz_files = sorted(args.input_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files in {args.input_dir}")

    print(f"[DROID-export] Converting {len(npz_files)} episodes -> {args.root}")
    for i, npz_path in enumerate(npz_files):
        n = 0
        for frame in _build_frames(npz_path, args.prompt):
            dataset.add_frame(frame, task=args.prompt)
            n += 1
        dataset.save_episode()
        print(f"  [{i+1:3d}/{len(npz_files)}] {npz_path.name}  {n} frames")

    print(f"[DROID-export] Done. Dataset root: {args.root}")
    print(f"[DROID-export] repo_id: {args.repo_id}")

    if args.push_to_hub:
        print(f"[DROID-export] Pushing to HF: {args.push_to_hub}")
        # LeRobot's push_to_hub uses dataset.repo_id. Override it to the HF path.
        dataset.repo_id = args.push_to_hub
        dataset.push_to_hub(
            tags=["droid", "panda", "real"],
            license="apache-2.0",
            private=args.hub_private,
            push_videos=True,
            tag_version=True,
        )
        print("[DROID-export] Pushed, codebase version tag auto-created.")


if __name__ == "__main__":
    main()
