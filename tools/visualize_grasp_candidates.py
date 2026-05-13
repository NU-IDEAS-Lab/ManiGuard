#!/usr/bin/env python3
"""Render top-K OBB-sampler grasp candidates overlaid on the target object.

Spawns the target in an empty scene, runs the OBB sampler against its mesh,
then for each top-K candidate teleports a single set of 5 OBB rectangle
markers (left/right finger, palm, swept volume) to the candidate's pose and
saves a PNG. Useful for confirming visually that the sampler is placing the
gripper rectangles ON the object — when AG fails to engage, the rectangle
overlay tells us whether it's a sampler geometry bug or a downstream issue.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
      VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
      conda run -n behavior python tools/visualize_grasp_candidates.py \
        --target soda_cup:fsfsas --topk 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import sentinel  # noqa: F401 — applies OG patches
from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


# Match sentinel/rl/grasps/obb_sampler.py
from sentinel.rl.grasps.obb_sampler import (  # noqa: E402
    _FRANKA_MAX_OPENING,
    _FRANKA_FINGER_LEN,
    _FRANKA_FINGER_BREAD,
    _FRANKA_FINGER_THICK,
    _FRANKA_EEF_TO_TIP,
    _FRANKA_PALM_HALF_LEN,
    _FRANKA_PALM_HALF_WIDTH,
    _FRANKA_PALM_HALF_BREAD,
    _FRANKA_AG_Z_FROM_EEF_LOW,
    _FRANKA_AG_Z_FROM_EEF_HIGH,
)

_S = 0.002

# Colors per box (RGB), alpha applied at material time.
_BOX_COLORS = {
    "left_finger":  (0.95, 0.30, 0.30),
    "right_finger": (0.95, 0.30, 0.30),
    "palm":         (0.95, 0.55, 0.20),
    "swept":        (0.30, 0.85, 0.40),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="soda_cup:fsfsas",
                   help="category:model identifier from BEHAVIOR catalog.")
    p.add_argument("--topk", type=int, default=8,
                   help="Number of top-scoring candidates to render.")
    p.add_argument("--n-candidates", type=int, default=400,
                   help="Number of OBB candidates to sample before topk slice.")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "grasp_candidate_viz"))
    p.add_argument("--image-size", type=int, default=900)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_grasp_frame_boxes(open_half: float):
    """5 boxes in grasp frame: (name, center_offset (3,), full_extents (3,))."""
    fl = _FRANKA_FINGER_LEN
    fb = _FRANKA_FINGER_BREAD
    ft = _FRANKA_FINGER_THICK
    et = _FRANKA_EEF_TO_TIP

    fz = et - fl / 2                          # finger-body z-center
    pz = et - fl - _FRANKA_PALM_HALF_LEN      # palm z-center
    sz_center  = 0.5 * (_FRANKA_AG_Z_FROM_EEF_LOW + _FRANKA_AG_Z_FROM_EEF_HIGH)
    sz_halfraw = 0.5 * (_FRANKA_AG_Z_FROM_EEF_HIGH - _FRANKA_AG_Z_FROM_EEF_LOW)

    boxes = [
        ("left_finger",  np.array([0.0, -(open_half + ft / 2), fz]),
                         2.0 * np.array([fb / 2 - _S, ft / 2 - _S, fl / 2 - _S])),
        ("right_finger", np.array([0.0, +(open_half + ft / 2), fz]),
                         2.0 * np.array([fb / 2 - _S, ft / 2 - _S, fl / 2 - _S])),
        ("palm",         np.array([0.0, 0.0, pz]),
                         2.0 * np.array([_FRANKA_PALM_HALF_BREAD - _S,
                                         _FRANKA_PALM_HALF_WIDTH - _S,
                                         _FRANKA_PALM_HALF_LEN - _S])),
        ("swept",        np.array([0.0, 0.0, sz_center]),
                         2.0 * np.array([fb / 2 - _S, open_half - _S, sz_halfraw - _S])),
    ]
    return boxes


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → (x, y, z, w) quaternion."""
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> list[float]:
    import omnigibson.utils.transform_utils as T  # noqa: N814
    direction = target - eye
    direction /= max(np.linalg.norm(direction), 1e-6)
    horizontal = math.atan2(-direction[0], direction[1])
    vertical = math.pi / 2 + math.asin(float(np.clip(direction[2], -1.0, 1.0)))
    quat = T.euler2quat(torch.tensor([vertical, 0.0, horizontal], dtype=torch.float32))
    return quat.tolist()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category, model = args.target.split(":")

    gm.HEADLESS = True
    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from omnigibson.objects.primitive_object import PrimitiveObject

    from sentinel.rl.grasps.mesh import mesh_from_og_object
    from sentinel.rl.grasps.obb_sampler import OBBConfig, sample_obb_assisted_grasps

    cfg = {
        "scene": {"type": "Scene"},
        "env":   {"action_frequency": 20, "rendering_frequency": 20, "physics_frequency": 120},
    }
    env = og.Environment(configs=cfg)
    env.reset()

    # Spawn target at a fixed world pose; place its bottom at z=0 so the
    # camera framing math below works out independently of the model.
    name = f"viz_target_{category}_{model}"
    obj = DatasetObject(name=name, category=category, model=model)
    env.scene.add_object(obj)
    obj.set_position_orientation(
        position=torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    )
    obj.root_link.disable_gravity()
    for _ in range(10):
        og.sim.step()
    # Drop to z = -aabb_min so the bottom sits at z=0.
    aabb_min, aabb_max = obj.aabb
    obj.set_position_orientation(
        position=torch.tensor([0.0, 0.0, 0.5 - float(aabb_min[2])], dtype=torch.float32),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    )
    for _ in range(5):
        og.sim.step()
    aabb_min, aabb_max = obj.aabb
    extents = (aabb_max - aabb_min).cpu().numpy().astype(np.float64)
    obj_world_pos, obj_world_quat = obj.get_position_orientation()
    obj_world_pos = obj_world_pos.detach().cpu().numpy().astype(np.float64)
    obj_world_quat = obj_world_quat.detach().cpu().numpy().astype(np.float64)
    print(f"[target] {category}/{model}  extents={extents}  "
          f"world_pos={obj_world_pos.tolist()}", flush=True)

    # Extract mesh + sample.
    mesh = mesh_from_og_object(obj, use_visual=True)
    print(f"[mesh] bounds={mesh.bounds.tolist()}", flush=True)
    rng = np.random.default_rng(args.seed)
    sampler_cfg = OBBConfig(max_candidates=args.n_candidates)
    poses, scores = sample_obb_assisted_grasps(mesh, sampler_cfg, rng)
    if len(poses) == 0:
        print("[err] sampler returned 0 grasps", flush=True)
        os._exit(1)
    n = min(args.topk, len(poses))
    print(f"[sampler] {len(poses)} candidates, top {n} scores={scores[:n]}",
          flush=True)

    # Spawn the 5 markers once, FAR from the action. We'll teleport them
    # per candidate. Visual-only + fixed_base so physics doesn't fight us.
    open_half = 0.5 * _FRANKA_MAX_OPENING
    boxes = _build_grasp_frame_boxes(open_half)
    markers = []
    for i, (bname, center_g, full_size) in enumerate(boxes):
        marker = PrimitiveObject(
            relative_prim_path=f"/obb_cand_{bname}",
            name=f"obb_cand_{bname}",
            category="obb_cand_marker",
            primitive_type="Cube",
            size=1.0,
            scale=list(map(float, full_size)),
            fixed_base=True,
            visual_only=True,
            rgba=[*_BOX_COLORS[bname], args.alpha],
        )
        env.scene.add_object(marker)
        marker.set_position_orientation(
            position=[10.0 + i * 0.5, 10.0, -2.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
        )
        # Force the diffuse color (rgba kwarg alone often doesn't paint).
        rgb = _BOX_COLORS[bname]
        for visual_mesh in marker.root_link.visual_meshes.values():
            material = visual_mesh.material
            if material is None:
                continue
            try:
                material.diffuse_color_constant = torch.tensor(rgb, dtype=torch.float32)
            except Exception:  # noqa: BLE001
                pass
            try:
                material.opacity_constant = float(args.alpha)
            except Exception:  # noqa: BLE001
                pass
        markers.append((bname, center_g, marker))

    for _ in range(5):
        og.sim.render()

    # Per-candidate render. The sampler returned poses in the mesh-local
    # (= object-local) frame; convert to world via the target's pose.
    R_obj = _quat_xyzw_to_rotmat(obj_world_quat)
    t_obj = obj_world_pos

    # Camera framing — three angles so we can see the geometry from
    # multiple sides; place the focus on the object center.
    radius = max(0.45, 1.4 * float(np.max(extents)))
    obj_center_world = (t_obj + np.array([0.0, 0.0, 0.5 * float(extents[2])])).astype(np.float64)
    views = {
        "iso":  obj_center_world + np.array([ radius,  0.7 * radius,  0.55 * radius]),
        "side": obj_center_world + np.array([ 0.0,     radius,        0.10 * radius]),
        "top":  obj_center_world + np.array([ 0.001,   0.001,         1.20 * radius]),
    }
    og.sim.viewer_height = args.image_size
    og.sim.viewer_width = args.image_size
    cam = og.sim.viewer_camera

    for k in range(n):
        T_local = np.asarray(poses[k], dtype=np.float64)
        R_local = T_local[:3, :3]
        t_local = T_local[:3, 3]
        R_world = R_obj @ R_local
        t_world = R_obj @ t_local + t_obj
        cand_quat = _rotmat_to_quat_xyzw(R_world)

        for bname, center_g, marker in markers:
            box_world = t_world + R_world @ center_g
            marker.set_position_orientation(
                position=box_world.astype(np.float64).tolist(),
                orientation=cand_quat.tolist(),
            )
        for _ in range(6):
            og.sim.render()

        for vname, eye_world in views.items():
            quat_cam = _look_at_quat(eye_world, obj_center_world)
            cam.set_position_orientation(position=eye_world.tolist(), orientation=quat_cam)
            for _ in range(4):
                og.sim.render()
            obs = cam.get_obs()[0]
            rgb_t = obs["rgb"][..., :3]
            rgb = rgb_t.detach().cpu().numpy().astype(np.uint8)
            path = output_dir / f"cand_{k:02d}_score{scores[k]:.3f}_{vname}.png"
            imageio.imwrite(str(path), rgb)
        print(f"  wrote cand_{k:02d}_score{scores[k]:.3f}_*  "
              f"pose_origin={t_world.tolist()}", flush=True)

    print(f"[done] {n} candidates in {output_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)


if __name__ == "__main__":
    main()
