#!/usr/bin/env python3
"""Overlay the cuRobo collision-sphere envelope on the active Franka in an empty scene.

Boots OmniGibson with an empty Scene and a single FrankaPanda (long-finger
bundle when the sentinel patch is active), drives the fingers fully open, then
spawns one translucent visual sphere per entry in ``collision_spheres`` of the
robot's active cuRobo YAML. Saves PNGs from four canonical viewpoints.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
      VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
      conda run -n behavior python tools/visualize_collision_spheres.py
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import sentinel  # noqa: F401  -- installs longfinger / Dropped / Upright patches
from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


# Distinct base colors per link, alpha applied separately
_LINK_COLORS: dict[str, tuple[float, float, float]] = {
    "panda_link0": (0.70, 0.70, 0.70),
    "panda_link1": (0.85, 0.45, 0.30),
    "panda_link2": (0.90, 0.75, 0.20),
    "panda_link3": (0.40, 0.80, 0.40),
    "panda_link4": (0.30, 0.75, 0.85),
    "panda_link5": (0.45, 0.55, 0.95),
    "panda_link6": (0.80, 0.40, 0.85),
    "panda_link7": (0.95, 0.55, 0.55),
    "panda_hand": (1.00, 0.85, 0.20),
    "panda_leftfinger": (0.20, 0.95, 0.55),
    "panda_rightfinger": (0.20, 0.55, 0.95),
}
_DEFAULT_COLOR = (0.85, 0.85, 0.85)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "collision_sphere_viz"))
    p.add_argument(
        "--gripper-open",
        type=float,
        default=0.04,
        help="Per-finger position in metres (upper limit ≈ 0.04 → 80 mm tip-to-tip).",
    )
    p.add_argument("--image-size", type=int, default=1024, help="Square render size in pixels.")
    p.add_argument("--alpha", type=float, default=0.35, help="Sphere marker translucency [0,1].")
    p.add_argument("--obb-overlay", action="store_true", default=True,
                   help="Also render OBB-sampler rectangle overlays at 80mm and 70mm openings.")
    p.add_argument("--obb-finger-len", type=float, default=0.054,
                   help="Sampler-modelled finger length along approach (m). Sampler default is 0.054.")
    return p.parse_args()


# Shared margins (kept in sync with sentinel/rl/grasps/obb_sampler.py).
_OBB_S  = 0.002   # shrink margin for empty boxes
_OBB_E  = 0.002   # expand margin for swept boxes
_HAND_TO_EEF_OFFSET  = 0.1034   # panda_hand origin → eef_link along +z

# Configurable per-render (the script renders multiple variants so these are
# overridden via the configs list).
_OBB_FB_DEFAULT = 0.022   # finger breadth (along perp) — short-Panda value
_OBB_FT_DEFAULT = 0.015   # finger thickness (along closing) — short-Panda value
_OBB_PALM_HALF_LEN_DEFAULT   = 0.030
_OBB_PALM_HALF_WIDTH_DEFAULT = 0.040
_OBB_PALM_HALF_BREAD_DEFAULT = 0.025

_OBB_COLORS = {
    "left_finger":  (0.95, 0.30, 0.30),  # red — must be empty
    "right_finger": (0.95, 0.30, 0.30),
    "palm":         (0.95, 0.55, 0.20),  # orange — must be empty
    "swept":        (0.30, 0.85, 0.40),  # green — must be non-empty
}


def _build_obb_specs(
    max_opening: float,
    finger_len: float,
    eef_to_tip: float = 0.0,
    finger_thick: float = _OBB_FT_DEFAULT,
    finger_bread: float = _OBB_FB_DEFAULT,
    palm_half_len: float = _OBB_PALM_HALF_LEN_DEFAULT,
    palm_half_width: float = _OBB_PALM_HALF_WIDTH_DEFAULT,
    palm_half_bread: float = _OBB_PALM_HALF_BREAD_DEFAULT,
):
    """Return dict of {name: (panda_hand_center, full_extents)} for the 5 OBBs.

    Boxes live in the grasp frame (perp=X, closing=Y, approach=Z) which is
    co-aligned with panda_hand. The grasp-frame origin = eef_link, which sits
    at panda_hand_z = +0.1034. So a box at grasp-frame offset (ox, oy, oz)
    is at panda_hand (ox, oy, 0.1034 + oz).

    The sampler's original placement puts finger/swept boxes at grasp-frame
    z = -fl/2 with half-extent fl/2 — so they span [-fl, 0] (behind eef_link
    by fl). With ``eef_to_tip > 0`` the boxes shift forward so they span
    [eef_to_tip - fl, eef_to_tip] — i.e. ending at the fingertip rather than
    at eef_link. Use this when eef_link does NOT sit at the fingertip
    (the long-finger Panda has eef_to_tip ≈ 0.098 m).
    """
    open_half = 0.5 * max_opening
    fl = finger_len
    fb = finger_bread
    ft = finger_thick
    s = _OBB_S
    e = _OBB_E

    fz = eef_to_tip - fl / 2  # grasp-frame z-center of finger / swept boxes
    pz = eef_to_tip - fl - palm_half_len  # palm sits behind the finger root

    boxes_grasp = {
        "left_finger":  (np.array([0.0, -(open_half + ft / 2), fz]),
                         np.array([fb / 2 - s, ft / 2 - s, fl / 2 - s])),
        "right_finger": (np.array([0.0, +(open_half + ft / 2), fz]),
                         np.array([fb / 2 - s, ft / 2 - s, fl / 2 - s])),
        "palm":         (np.array([0.0, 0.0, pz]),
                         np.array([palm_half_bread - s, palm_half_width - s,
                                   palm_half_len - s])),
        "swept":        (np.array([0.0, 0.0, fz]),
                         np.array([fb / 2 + e, open_half + e, fl / 2 + e])),
    }
    out = {}
    for name, (center_g, half_g) in boxes_grasp.items():
        center_hand = np.array([center_g[0], center_g[1], _HAND_TO_EEF_OFFSET + center_g[2]])
        full_size = (2.0 * half_g).astype(np.float64)
        out[name] = (center_hand, full_size)
    return out


def spawn_obb_markers(
    env, robot, max_opening: float, finger_len: float, alpha: float,
    eef_to_tip: float = 0.0,
    finger_thick: float = _OBB_FT_DEFAULT,
    finger_bread: float = _OBB_FB_DEFAULT,
    palm_half_width: float = _OBB_PALM_HALF_WIDTH_DEFAULT,
    palm_half_bread: float = _OBB_PALM_HALF_BREAD_DEFAULT,
):
    from omnigibson.objects.primitive_object import PrimitiveObject
    import omnigibson.utils.transform_utils as T  # noqa: N814

    hand = robot.links["panda_hand"]
    hand_pos, hand_quat = hand.get_position_orientation()
    hand_pos = hand_pos.detach().cpu()
    hand_quat = hand_quat.detach().cpu()

    specs = _build_obb_specs(
        max_opening, finger_len, eef_to_tip,
        finger_thick=finger_thick, finger_bread=finger_bread,
        palm_half_width=palm_half_width, palm_half_bread=palm_half_bread,
    )
    spawned = []
    for name, (center_hand, full_size) in specs.items():
        # Box pose in world = (hand_pos + R_hand · center_hand, hand_quat).
        offset_local = torch.tensor(center_hand, dtype=torch.float32)
        offset_world = T.quat_apply(hand_quat, offset_local).squeeze()
        world_pos = (hand_pos + offset_world).tolist()
        rgb = _OBB_COLORS[name]
        tag = (f"obb_{int(round(max_opening*1000))}mm_fl{int(round(finger_len*1000))}"
               f"_t{int(round(eef_to_tip*1000))}"
               f"_ft{int(round(finger_thick*1000))}_{name}")
        marker = PrimitiveObject(
            relative_prim_path=f"/{tag}",
            name=tag,
            category="obb_overlay_marker",
            primitive_type="Cube",
            size=1.0,
            scale=list(map(float, full_size)),
            fixed_base=True,
            visual_only=True,
            rgba=[rgb[0], rgb[1], rgb[2], alpha],
        )
        env.scene.add_object(marker)
        marker.set_position_orientation(position=world_pos, orientation=hand_quat.tolist())
        for visual_mesh in marker.root_link.visual_meshes.values():
            material = visual_mesh.material
            if material is None:
                continue
            try:
                material.diffuse_color_constant = torch.tensor(rgb, dtype=torch.float32)
            except Exception:  # noqa: BLE001
                pass
            try:
                material.opacity_constant = float(alpha)
            except Exception:  # noqa: BLE001
                pass
        spawned.append(marker)
    return spawned


def hide_markers(markers, hide_z: float = -50.0):
    """Translate markers far below the scene so they don't occlude the robot.

    Visual-only PrimitiveObjects can't be removed cleanly from a running OG
    scene (state-dumping assertion fires), so we just park them out of frame.
    """
    for i, m in enumerate(markers):
        try:
            m.set_position_orientation(
                position=[100.0 + 0.5 * i, 100.0, hide_z],
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
        except Exception:  # noqa: BLE001
            pass


def build_env_cfg() -> dict:
    gm.HEADLESS = True
    robot_cfg = {
        "type": "FrankaPanda",
        "name": "franka",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "physical",
        "position": [0.0, 0.0, 0.0],
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {"image_height": 64, "image_width": 64},
            },
        },
    }
    env_cfg = {
        "action_frequency": 20,
        "rendering_frequency": 20,
        "physics_frequency": 120,
    }
    return dict(scene={"type": "Scene"}, robots=[robot_cfg], env=env_cfg)


def look_at_quat(eye: np.ndarray, target: np.ndarray) -> list[float]:
    """USD-style quat (x,y,z,w) for a camera at ``eye`` looking at ``target``."""
    import omnigibson.utils.transform_utils as T  # noqa: N814
    direction = target - eye
    direction /= max(np.linalg.norm(direction), 1e-6)
    horizontal = math.atan2(-direction[0], direction[1])
    vertical = math.pi / 2 + math.asin(float(np.clip(direction[2], -1.0, 1.0)))
    quat = T.euler2quat(torch.tensor([vertical, 0.0, horizontal], dtype=torch.float32))
    return quat.tolist()


def spawn_sphere_markers(
    env,
    robot,
    spheres_by_link: dict,
    buffer_m: float,
    alpha: float,
):
    from omnigibson.objects.primitive_object import PrimitiveObject
    import omnigibson.utils.transform_utils as T  # noqa: N814

    spawned = []
    for link_name, spheres in spheres_by_link.items():
        if link_name not in robot.links:
            print(f"  [skip] link {link_name!r} not on robot")
            continue
        link = robot.links[link_name]
        link_pos, link_quat = link.get_position_orientation()
        link_pos = link_pos.detach().cpu()
        link_quat = link_quat.detach().cpu()
        base_rgb = _LINK_COLORS.get(link_name, _DEFAULT_COLOR)
        for i, sp in enumerate(spheres):
            local = torch.tensor(sp["center"], dtype=torch.float32)
            world = link_pos + T.quat_apply(link_quat, local).squeeze()
            radius = float(sp["radius"]) + buffer_m
            marker_name = f"cs_{link_name}_{i}"
            marker = PrimitiveObject(
                relative_prim_path=f"/{marker_name}",
                name=marker_name,
                category="collision_sphere_marker",
                primitive_type="Sphere",
                radius=radius,
                fixed_base=True,
                visual_only=True,
                rgba=[base_rgb[0], base_rgb[1], base_rgb[2], alpha],
            )
            env.scene.add_object(marker)
            marker.set_position_orientation(
                position=world.tolist(),
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
            # rgba on PrimitiveObject doesn't always paint the material —
            # force the diffuse colour explicitly (matches spawn_goal_region_marker).
            for visual_mesh in marker.root_link.visual_meshes.values():
                material = visual_mesh.material
                if material is None:
                    continue
                try:
                    material.diffuse_color_constant = torch.tensor(
                        base_rgb, dtype=torch.float32
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    material.opacity_constant = float(alpha)
                except Exception:  # noqa: BLE001
                    pass
            spawned.append(marker)
    return spawned


def render_view(env, name: str, eye: tuple[float, float, float], target: tuple[float, float, float],
                output_dir: Path, image_size: int) -> None:
    import omnigibson as og

    quat = look_at_quat(np.asarray(eye, dtype=np.float32), np.asarray(target, dtype=np.float32))
    og.sim.viewer_height = image_size
    og.sim.viewer_width = image_size
    cam = og.sim.viewer_camera
    cam.set_position_orientation(position=list(eye), orientation=quat)

    # A few render ticks so the camera flush and sphere transforms settle.
    for _ in range(8):
        og.sim.render()

    obs = cam.get_obs()[0]
    rgb_t = obs["rgb"][..., :3]
    rgb = rgb_t.detach().cpu().numpy().astype(np.uint8)
    out_path = output_dir / f"{name}.png"
    imageio.imwrite(str(out_path), rgb)
    print(f"  wrote {out_path}  shape={rgb.shape}  cam.image_height={cam.image_height}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import omnigibson as og

    cfg = build_env_cfg()
    env = og.Environment(configs=cfg)
    env.reset()
    robot = env.robots[0]

    yaml_path = Path(robot.curobo_path)
    print(f"[curobo] active config: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        curobo_cfg = yaml.safe_load(f)
    kin = curobo_cfg["robot_cfg"]["kinematics"]
    spheres_by_link = kin["collision_spheres"]
    buffer_m = float(kin.get("collision_sphere_buffer", 0.0))
    print(
        f"[curobo] {sum(len(v) for v in spheres_by_link.values())} spheres "
        f"across {len(spheres_by_link)} links, sphere_buffer={buffer_m*1000:.1f} mm"
    )

    # Open fingers to the requested width (clamped to the upper limit).
    arm = robot.default_arm
    gripper_idx = robot.gripper_control_idx[arm]
    upper = robot.joint_upper_limits[gripper_idx]
    target = torch.full_like(upper, float(args.gripper_open))
    open_q = torch.minimum(target, upper)
    robot.set_joint_positions(open_q, gripper_idx)

    # Let physics settle while holding the joints in place.
    for _ in range(20):
        robot.keep_still()
        og.sim.step()
    for _ in range(4):
        og.sim.render()

    arm = robot.default_arm
    hand_pos, _ = robot.links["panda_hand"].get_position_orientation()
    eef_link = robot.eef_links[arm]
    eef_pos, _ = eef_link.get_position_orientation()
    offset = (eef_pos - hand_pos).norm().item()
    print(f"[geom] panda_hand pos: {hand_pos.tolist()}")
    print(f"[geom] eef_link ({eef_link.body_name}) pos: {eef_pos.tolist()}")
    print(f"[geom] hand_to_eef distance: {offset*1000:.1f} mm")

    sphere_markers = spawn_sphere_markers(env, robot, spheres_by_link, buffer_m, args.alpha)
    print(f"[viz] spawned {len(sphere_markers)} sphere markers")

    for _ in range(6):
        og.sim.render()

    target = (0.10, 0.0, 0.45)
    views = {
        "perspective":  ( 0.85,  0.65, 0.95),
        "front":        ( 0.90,  0.00, 0.55),
        "side":         ( 0.10,  0.90, 0.55),
        "top":          ( 0.10,  0.05, 1.30),
        "gripper":      ( 0.55,  0.25, 0.95),
    }
    for name, eye in views.items():
        render_view(env, name, eye, target, output_dir, args.image_size)

    if args.obb_overlay:
        # Hide the sphere markers — comparing OBB rectangles against actual
        # robot geometry is easier without the sphere envelope on top.
        hide_markers(sphere_markers)
        for _ in range(4):
            og.sim.render()

        obb_views = {
            "perspective":  ( 0.85,  0.65, 0.95),
            "gripper":      ( 0.55,  0.25, 0.95),
            "side":         ( 0.10,  0.90, 0.55),
        }
        # (max_opening, finger_len, eef_to_tip, finger_thick, finger_bread,
        #  palm_half_width, palm_half_bread, tag-suffix)
        # 1) original sampler defaults (short-Panda values).
        # 2) opening corrected for cuRobo's 71 mm clear corridor.
        # 3) opening AND finger length corrected (still placed behind eef_link).
        # 4) aligned: also shift forward so boxes end at the fingertip.
        # 5) FINAL: also fix finger thickness/breadth + palm width/bread to the
        #    actual long-finger mesh extents.
        SHORT = (_OBB_FT_DEFAULT, _OBB_FB_DEFAULT,
                 _OBB_PALM_HALF_WIDTH_DEFAULT, _OBB_PALM_HALF_BREAD_DEFAULT)
        LONG  = (0.040, 0.038, 0.046, 0.032)
        configs = [
            (0.080, 0.054, 0.000, *SHORT, "obb80mm"),
            (0.070, 0.054, 0.000, *SHORT, "obb70mm"),
            (0.070, 0.145, 0.000, *SHORT, "obb70mm_long"),
            (0.070, 0.145, 0.098, *SHORT, "obb70mm_long_aligned"),
            (0.070, 0.145, 0.098, *LONG,  "obb70mm_long_final"),
        ]
        for opening, fl, eef_to_tip, ft, fb, phw, phb, tag in configs:
            print(f"[obb] {tag}: opening={opening:.3f}  fl={fl:.3f}  "
                  f"eef_to_tip={eef_to_tip:.3f}  ft={ft:.3f}  fb={fb:.3f}  "
                  f"palm_w={phw:.3f}  palm_b={phb:.3f}")
            obb_markers = spawn_obb_markers(
                env, robot, opening, fl, alpha=0.55,
                eef_to_tip=eef_to_tip,
                finger_thick=ft, finger_bread=fb,
                palm_half_width=phw, palm_half_bread=phb,
            )
            for _ in range(6):
                og.sim.render()
            for name, eye in obb_views.items():
                render_view(env, f"{tag}_{name}", eye, target, output_dir, args.image_size)
            hide_markers(obb_markers)
            for _ in range(4):
                og.sim.render()

    print(f"[done] outputs in {output_dir}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
