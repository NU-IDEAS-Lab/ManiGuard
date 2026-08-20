#!/usr/bin/env python3
"""Real-teleop npz -> sim-compatible HDF5 (Stage-2 input schema).

Consumes the npz files written by the real-franka teleop capture
(outputs/real_teleop/<id>.npz) and emits one HDF5 per episode that
matches the schema `maniguard.data.lerobot.multitask_lerobot_export` expects:

    data/demo_0/obs/image        (N+1, H, W, 3) uint8   <- cam0, resized
    data/demo_0/obs/wrist_image  (N+1, H, W, 3) uint8   <- cam1, resized
    data/demo_0/obs/state        (N+1, 8)       f32     <- eef_pos(3) + axisangle(3) + gripper/2 (x2)
    data/demo_0/action           (N,   7)       f32     <- dpos(3) + drot_axisangle(3) + gripper_sign(1)

Real-npz schema (reference):
    observation/image/cam0       (N+1,) object     JPEG bytes
    observation/image/cam1       (N+1,) object     JPEG bytes
    observation/cartesian_position (N+1, 7) f64   [x, y, z, qw, qx, qy, qz]   (WXYZ!)
    observation/gripper_position   (N+1,)  f64    NORMALIZED 0-1 (0 = fully open,
                                                   1 = fully closed), bimodal in
                                                   practice

Conventions matched with sim:
    - State 8D mirrors sim HDF5; gripper_{L,R} both map to Franka
      half-aperture in meters: 0.04 * (1 - gripper_position), so fully
      open -> 0.04 like sim, fully closed -> 0.
    - Action is delta EEF with axisangle rotation delta; gripper
      binarized via midpoint 0.5 with sim convention: +1 = open, -1 = close.
    - Quat WXYZ; consecutive quats flipped onto the same hemisphere for
      axisangle continuity.

Usage:
    python -m maniguard.data.real_teleop.real_teleop_to_hdf5 \
        --input-dir outputs/real_teleop \
        --output-dir outputs/real_teleop_hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

GRIPPER_MIDPOINT_NORM = 0.5   # bimodal around 0.05 (open) / 0.99 (closed)
GRIPPER_FULL_OPEN_M = 0.04    # matches sim grip_L/grip_R max (Franka half-aperture, m)


def _decode_resize(jpeg_bytes: bytes, size: int) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None (corrupt JPEG?)")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _quat_continuous_wxyz(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive wxyz quats sit on the same hemisphere."""
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t - 1], q[t]) < 0.0:
            q[t] = -q[t]
    return q


def _quat_to_axisangle_wxyz(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3) axis * angle."""
    w = np.clip(q[:, 0], -1.0, 1.0)
    xyz = q[:, 1:4]
    sin_half = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    angle = 2.0 * np.arccos(w)
    small = sin_half < 1e-8
    axis = np.where(small[:, None], np.zeros_like(xyz), xyz / np.where(small[:, None], 1.0, sin_half[:, None]))
    return (axis * angle[:, None]).astype(np.float32)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batched wxyz quat multiplication: q1 ⊗ q2."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def _delta_axisangle_wxyz(q_prev: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """Axisangle representation of q_next ⊗ conj(q_prev)."""
    return _quat_to_axisangle_wxyz(_quat_mul_wxyz(q_next, _quat_conj_wxyz(q_prev)))


def convert_episode(npz_path: Path, hdf5_out: Path, img_size: int) -> None:
    d = np.load(npz_path, allow_pickle=True)

    cart = d["observation/cartesian_position"].astype(np.float32)  # (N+1, 7) wxyz at [3:7]
    pos = cart[:, :3]
    quat_wxyz = _quat_continuous_wxyz(cart[:, 3:7])
    grip = d["observation/gripper_position"].astype(np.float32)    # (N+1,)

    # ---- state (N+1, 8): eef_pos(3) + axisangle(3) + gripper_{L,R}_m (x2) ----
    # gripper_position is normalized [0, 1] (0 = open, 1 = closed). Map to
    # Franka half-aperture in meters so state[:6] and state[6:8] live at
    # similar scales and match sim's physical units.
    axisangle = _quat_to_axisangle_wxyz(quat_wxyz)
    grip_half_m = (GRIPPER_FULL_OPEN_M * (1.0 - np.clip(grip, 0.0, 1.0)))[:, None]
    state = np.concatenate([pos, axisangle, grip_half_m, grip_half_m], axis=1).astype(np.float32)

    # ---- action (N, 7): dpos(3) + drot_axisangle(3) + gripper_sign(1) ----
    # Sim convention: +1 = open, -1 = close. Binarize next-step aperture
    # around the bimodal midpoint (0.5 in normalized space).
    dpos = (pos[1:] - pos[:-1]).astype(np.float32)
    drot = _delta_axisangle_wxyz(quat_wxyz[:-1], quat_wxyz[1:])
    grip_next_sign = np.where(grip[1:] < GRIPPER_MIDPOINT_NORM, 1.0, -1.0).astype(np.float32)[:, None]
    action = np.concatenate([dpos, drot, grip_next_sign], axis=1).astype(np.float32)

    # ---- images (N+1, S, S, 3) uint8 ----
    base_jpegs = d["observation/image/cam0"]
    wrist_jpegs = d["observation/image/cam1"]
    n = len(base_jpegs)
    image = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    wrist_image = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    for t in range(n):
        image[t] = _decode_resize(base_jpegs[t].tobytes() if hasattr(base_jpegs[t], "tobytes") else base_jpegs[t], img_size)
        wrist_image[t] = _decode_resize(wrist_jpegs[t].tobytes() if hasattr(wrist_jpegs[t], "tobytes") else wrist_jpegs[t], img_size)

    hdf5_out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_out, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.create_dataset("obs/image", data=image, compression="gzip", compression_opts=4)
        demo.create_dataset("obs/wrist_image", data=wrist_image, compression="gzip", compression_opts=4)
        demo.create_dataset("obs/state", data=state)
        demo.create_dataset("action", data=action)

    print(f"  [{npz_path.name}] N+1={n}  state={state.shape}  action={action.shape}  -> {hdf5_out.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with <id>.npz real teleop files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for traj_*.hdf5")
    parser.add_argument("--img-size", type=int, default=256, help="Square image resize target (default 256)")
    args = parser.parse_args()

    npz_files = sorted(args.input_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files found in {args.input_dir}")

    print(f"[Real->HDF5] Converting {len(npz_files)} episodes from {args.input_dir}")
    for i, npz_path in enumerate(npz_files):
        hdf5_out = args.output_dir / f"traj_{i}.hdf5"
        convert_episode(npz_path, hdf5_out, args.img_size)
    print(f"[Real->HDF5] Done. {len(npz_files)} HDF5 files in {args.output_dir}")


if __name__ == "__main__":
    main()
