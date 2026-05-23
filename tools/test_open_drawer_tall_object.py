"""Test scene: bottom_cabinet (bamfsz) with one drawer cracked partway
open and a graspable object planted on the floor inside the drawer's
opening trajectory, so the drawer cannot open further without collision.

Scene layout:
  - Empty Scene (floor only)
  - bottom_cabinet-bamfsz at origin (fixed_base, facing +x)
  - One prismatic joint (drawer) opened to ``--open-fraction`` of its
    stroke (default 20 %).
  - A graspable object (default: wine_bottle) placed on the floor
    along the drawer's slide direction, between the drawer's current
    leading face and the position that face would reach if fully open
  - Franka mounted past everything (drawer + blocking object) along the
    slide direction, looking back at the cabinet

Diagnostics printed:
  - drawer slide direction + total stroke
  - drawer current opening (m and % of stroke)
  - object footprint position + remaining gap to drawer's leading face
  - whether the object intercepts the drawer's full-open trajectory

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m tools.test_open_drawer_tall_object
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Trigger sentinel's OmniGibson patches (long-finger Franka, etc.) before
# omnigibson is imported. Required for any script that builds a Franka.
import sentinel  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cabinet-model", default="bamfsz",
                   help="bottom_cabinet model id.")
    p.add_argument("--object-category", default="wine_bottle",
                   help="Graspable object category. Default wine_bottle "
                        "is ~25 cm tall.")
    p.add_argument("--object-model", default="hlzfxw",
                   help="DatasetObject model id for the tall object.")
    p.add_argument("--robot-gap-m", type=float, default=0.20,
                   help="Extra gap between the fully-opened drawer's "
                        "far face and the robot base.")
    p.add_argument("--open-fraction", type=float, default=0.20,
                   help="Fraction of the prismatic joint's stroke to "
                        "leave the drawer open at (0.0 closed, 1.0 fully "
                        "open). Default 0.20.")
    p.add_argument("--object-gap-m", type=float, default=0.05,
                   help="Gap (m) between the drawer's leading face at "
                        "the current open position and the object — "
                        "i.e. how far the drawer can still slide before "
                        "hitting the object.")
    p.add_argument("--settle-steps", type=int, default=30)
    p.add_argument("--headless", action="store_true",
                   help="Run with no viewer window (off by default so "
                        "you can visually inspect the placement).")
    p.add_argument("--hold-seconds", type=float, default=20.0,
                   help="How long to keep the sim alive after rendering "
                        "the snapshots (gives you time to inspect the "
                        "viewer window).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/test_drawer_views"),
                   help="Where to save per-camera PNG snapshots.")
    return p.parse_args()


_CAM_NAMES = ("cam_back", "cam_right", "cam_top")
_CAM_HW = (720, 1280)


def _init_og(headless: bool):
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if headless:
        gm.HEADLESS = True
    import omnigibson as og
    return og


def _build_env(og, cabinet_model: str, obj_category: str, obj_model: str):
    """Empty scene + fixed cabinet at origin + tall object parked far
    away (we'll move it into the drawer after the drawer is opened) +
    Franka parked far away (we'll move it in front of the drawer).
    """
    env_cfg = {
        "scene": {"type": "Scene"},
        "robots": [{
            "type": "FrankaPanda",
            "name": "agent_0",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": False,
            "fixed_base": True,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": "assisted",
            "self_collisions": True,
            "controller_config": {
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        }],
        "objects": [
            {
                "type": "DatasetObject", "name": "cabinet",
                "category": "bottom_cabinet", "model": cabinet_model,
                "scale": [1.0, 1.0, 1.0], "fixed_base": True,
                "position": [1.5, 0.0, 0.30],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "type": "DatasetObject", "name": "tall_obj",
                "category": obj_category, "model": obj_model,
                "scale": [1.0, 1.0, 1.0], "fixed_base": False,
                "position": [3.0, 3.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        ],
        "task": {"type": "DummyTask"},
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "name": name,
                    "relative_prim_path": f"/{name}",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {
                        "image_height": _CAM_HW[0],
                        "image_width": _CAM_HW[1],
                    },
                }
                for name in _CAM_NAMES
            ],
        },
    }
    env = og.Environment(configs=env_cfg)
    env.reset()
    og.sim.step()
    return env


def _settle_cabinet_on_floor(og, cabinet):
    """Drop the cabinet so its bottom face sits on z=0."""
    import torch as th
    aabb_min, aabb_max = cabinet.aabb
    pos, quat = cabinet.get_position_orientation()
    bottom_z = float(aabb_min[2])
    # Shift in z so bottom_z → 0.
    new_z = float(pos[2]) - bottom_z
    cabinet.set_position_orientation(
        position=th.tensor([float(pos[0]), float(pos[1]), new_z],
                           dtype=th.float32),
        orientation=quat,
    )
    for _ in range(3):
        og.sim.step()


def _pick_drawer_joint(cabinet):
    """Return (joint_name, joint) for the first prismatic joint on the
    cabinet (drawers are prismatic; doors are revolute). Raises if none.
    """
    from omnigibson.utils.constants import JointType
    prismatic = [
        (name, j) for name, j in cabinet.joints.items()
        if j.joint_type == JointType.JOINT_PRISMATIC
    ]
    if not prismatic:
        raise RuntimeError(
            f"Cabinet {cabinet.model!r} has no prismatic joints — "
            f"all joints are: {list(cabinet.joints)}"
        )
    return prismatic[0]


def _drawer_link_for_joint(cabinet, joint):
    """OmniGibson convention: joint ``j_link_N`` drives child ``link_N``.
    Parse the link name from the joint's body1 prim path and look it up
    in cabinet.links.
    """
    body1_path = joint.body1  # absolute prim path
    link_name = body1_path.rsplit("/", 1)[-1]
    link = cabinet.links.get(link_name)
    if link is None:
        raise RuntimeError(
            f"Drawer joint {joint.name!r} points to link "
            f"{link_name!r} which is not in cabinet.links "
            f"({list(cabinet.links)})"
        )
    return link_name, link


def _open_drawer(og, cabinet, joint_name, joint, fraction: float,
                 settle_steps: int):
    """Drive the drawer joint to ``fraction`` of its stroke and settle."""
    upper = float(joint.upper_limit)
    lower = float(joint.lower_limit)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    target = lower + fraction * (upper - lower)
    print(f"[drawer] joint={joint_name!r}  "
          f"limits=[{lower:.3f}, {upper:.3f}] (m)  "
          f"target={target:.3f} ({fraction*100:.0f}% open)")
    joint.set_pos(target)
    cabinet.keep_still()
    for _ in range(settle_steps):
        og.sim.step()
    return target


def _measure_full_open_drawer_aabb(og, cabinet, joint, drawer_link):
    """Return (aabb_min, aabb_max) of the drawer link with the joint
    driven to its upper limit. Restores the joint's current position.
    """
    current = float(joint.get_state()[0][0]) if hasattr(joint, "get_state") \
        else float(joint.upper_limit)
    joint.set_pos(float(joint.upper_limit)); cabinet.keep_still()
    for _ in range(3):
        og.sim.step()
    f_min, f_max = drawer_link.aabb
    f_min = np.asarray(f_min.cpu() if hasattr(f_min, "cpu") else f_min,
                       dtype=np.float32)
    f_max = np.asarray(f_max.cpu() if hasattr(f_max, "cpu") else f_max,
                       dtype=np.float32)
    joint.set_pos(current); cabinet.keep_still()
    for _ in range(3):
        og.sim.step()
    return f_min, f_max


def _place_object_in_drawer_path(og, drawer_link, tall_obj, slide_dir_xy,
                                 *, gap_m: float):
    """Plant the object upright on the floor, in the drawer's opening
    trajectory.

    The object is placed along ``slide_dir_xy`` (CLOSED→OPEN direction),
    at ``gap_m`` past the drawer's current leading face (the face on the
    +slide_dir side of the drawer's AABB). With gap_m < remaining stroke,
    the drawer will collide with the object before reaching its upper
    limit, so it cannot open further.

    Returns (object_xy, gap_m, drawer_leading_along_slide).
    """
    import torch as th
    d_min, d_max = drawer_link.aabb
    d_min = np.asarray(d_min.cpu() if hasattr(d_min, "cpu") else d_min,
                       dtype=np.float32)
    d_max = np.asarray(d_max.cpu() if hasattr(d_max, "cpu") else d_max,
                       dtype=np.float32)
    cx = float((d_min[0] + d_max[0]) * 0.5)
    cy = float((d_min[1] + d_max[1]) * 0.5)
    sx, sy = float(slide_dir_xy[0]), float(slide_dir_xy[1])

    # Drawer leading-face offset along slide_dir from drawer center.
    # (Project the AABB extents onto |slide_dir|; works for axis-aligned
    # slide directions.)
    extent_xy = np.array([float(d_max[0] - d_min[0]),
                          float(d_max[1] - d_min[1])], dtype=np.float32)
    half_along = 0.5 * float(np.dot(extent_xy, np.abs([sx, sy])))
    leading_x = cx + sx * half_along
    leading_y = cy + sy * half_along

    obj_x = leading_x + sx * gap_m
    obj_y = leading_y + sy * gap_m

    # Pre-place above the floor at identity orientation so the AABB is
    # gravity-aligned for the standing-height measurement.
    tall_obj.set_position_orientation(
        position=th.tensor([obj_x, obj_y, 0.5], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    tall_obj.root_link.disable_gravity()
    for _ in range(3):
        og.sim.step()
    t_min, _ = tall_obj.aabb
    t_pos, _ = tall_obj.get_position_orientation()
    center_to_bottom = float(t_pos[2] - t_min[2])

    new_z = 0.0 + center_to_bottom + 0.002
    tall_obj.set_position_orientation(
        position=th.tensor([obj_x, obj_y, new_z], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    tall_obj.root_link.set_linear_velocity(th.zeros(3, dtype=th.float32))
    tall_obj.root_link.set_angular_velocity(th.zeros(3, dtype=th.float32))
    for _ in range(10):
        og.sim.step()

    return (obj_x, obj_y, new_z), gap_m, (leading_x, leading_y)


def _measure_drawer_slide(cabinet, joint, drawer_link):
    """Open/close the drawer once to measure the slide direction (xy
    unit vector in world frame) and the slide stroke (m). Restores the
    drawer's current position when done.

    We need this to:
      1. place the robot in front of the OPEN drawer's far face
      2. confirm enough back-clearance for the robot
    """
    import torch as th
    current = float(joint.get_state()[0][0]) if hasattr(joint, "get_state") \
        else float(joint.upper_limit)
    upper = float(joint.upper_limit)
    lower = float(joint.lower_limit)

    joint.set_pos(lower); cabinet.keep_still()
    p_closed = drawer_link.get_position_orientation()[0].cpu().numpy()
    joint.set_pos(upper); cabinet.keep_still()
    p_open = drawer_link.get_position_orientation()[0].cpu().numpy()

    # Restore.
    joint.set_pos(current); cabinet.keep_still()

    dxy = (p_open - p_closed)[:2]
    stroke = float(np.linalg.norm(dxy))
    if stroke < 1e-4:
        # Degenerate — assume +x.
        return np.array([1.0, 0.0]), 0.0
    return dxy / stroke, stroke


def _eye_lookat_to_quat(eye, lookat):
    """Same convention as sentinel.task_generation.utils.video."""
    import torch as th
    import omnigibson.utils.transform_utils as T
    import math
    d = np.asarray(lookat, dtype=np.float32) - np.asarray(eye, dtype=np.float32)
    d = d / max(1e-6, float(np.linalg.norm(d)))
    return T.euler2quat(th.tensor([
        math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))),
        0.0,
        float(np.arctan2(-d[0], d[1])),
    ], dtype=th.float32))


def _place_cameras_around_drawer(og, env, drawer_link, slide_dir_xy,
                                 object_xy):
    """Position the 3 external cameras to frame the partially-open
    drawer *and* the blocking object together.

    Lookat is the midpoint (in xy) between the drawer center and the
    blocking object, at drawer mid-height. Cameras orbit that midpoint:
      - cam_back  : 3/4 angle from in front of the object, looking back
                    toward the drawer (so the gap between drawer and
                    object is visible).
      - cam_right : pure-perpendicular side view along slide_dir, so the
                    drawer→object→robot line is shown in profile.
      - cam_top   : directly above the midpoint, looking straight down.
    """
    d_min, d_max = drawer_link.aabb
    dcx = float((d_min[0] + d_max[0]) * 0.5)
    dcy = float((d_min[1] + d_max[1]) * 0.5)
    cz = float((d_min[2] + d_max[2]) * 0.5)
    rim_z = float(d_max[2])
    ox, oy = float(object_xy[0]), float(object_xy[1])

    # Aim midway between drawer center and object so both are in frame.
    cx = 0.5 * (dcx + ox)
    cy = 0.5 * (dcy + oy)
    lookat = (cx, cy, cz)

    sx, sy = float(slide_dir_xy[0]), float(slide_dir_xy[1])
    # 90° right of slide_dir (rotated CW looking down -z).
    rx, ry = sy, -sx

    # Distance to spread cameras: drawer→object plus a margin for the
    # whole scene (drawer body is ~0.5 m wide).
    drawer_obj_dist = float(np.hypot(ox - dcx, oy - dcy))
    radius = max(0.85, drawer_obj_dist + 0.6)

    # cam_back: 3/4 from past the object side, looking back at drawer.
    # Mix of +slide_dir (so we're on the object side, not the cabinet
    # side) and +right (clears the robot, which is even farther along
    # +slide_dir).
    cam_back_eye = (cx + (sx - rx) * radius * 0.55,
                    cy + (sy - ry) * radius * 0.55,
                    rim_z + 0.40)
    cam_right_eye = (cx + rx * radius * 0.85,
                     cy + ry * radius * 0.85,
                     rim_z + 0.20)
    cam_top_eye   = (cx, cy, rim_z + max(0.60, drawer_obj_dist * 1.5))

    eyes = {
        "cam_back": cam_back_eye,
        "cam_right": cam_right_eye,
        "cam_top": cam_top_eye,
    }
    for name, eye in eyes.items():
        quat = _eye_lookat_to_quat(eye, lookat).tolist()
        sensor = env.external_sensors.get(name)
        if sensor is None:
            raise RuntimeError(f"External sensor {name!r} not found.")
        sensor.set_position_orientation(
            position=list(eye), orientation=quat, frame="world",
        )
        print(f"[camera] {name}: eye={tuple(round(v,3) for v in eye)} "
              f"→ lookat={tuple(round(v,3) for v in lookat)}")

    # Pin the GUI viewer to the back camera so the user sees the same
    # angle as the rendered PNG.
    back_quat = _eye_lookat_to_quat(cam_back_eye, lookat).tolist()
    og.sim.viewer_camera.set_position_orientation(
        position=list(cam_back_eye), orientation=back_quat,
    )
    return eyes


def _save_camera_snapshots(og, env, out_dir: Path, eyes):
    """Render once and dump each external camera's RGB to a PNG."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    # Multiple renders so the rendertargets stabilize (DLSS / sample
    # accumulation in the RTX pipeline produces noisy first-frame
    # output otherwise).
    for _ in range(8):
        og.sim.render()
    raw_obs, _ = env.get_obs()
    external = raw_obs.get("external", {})
    saved = []
    for name in eyes:
        bundle = external.get(name)
        if bundle is None:
            print(f"[snapshot] WARN: no obs for {name!r}")
            continue
        rgb = bundle.get("rgb")
        if rgb is None:
            print(f"[snapshot] WARN: no rgb modality for {name!r}")
            continue
        # OG returns RGBA as a torch.uint8 tensor (H,W,4).
        arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        path = out_dir / f"{name}.png"
        Image.fromarray(arr.astype(np.uint8)).save(path)
        saved.append(path)
        print(f"[snapshot] wrote {path}")
    return saved


def _place_robot_in_front_of_drawer(og, robot, full_open_aabb,
                                    slide_dir_xy, object_xy,
                                    *, gap_m: float):
    """Position the Franka base on the floor past the farthest forward
    point along slide_dir among {fully-open drawer leading face, the
    blocking object}, looking back toward the cabinet.

    Using the FULL-open drawer AABB (rather than the current partial
    state) keeps the robot from being inside the drawer's swept volume
    if the joint later opens further; using the object's xy ensures the
    robot doesn't sit on top of the blocker.
    """
    import torch as th
    f_min, f_max = full_open_aabb
    cx = float((f_min[0] + f_max[0]) * 0.5)
    cy = float((f_min[1] + f_max[1]) * 0.5)
    extent_xy = np.array([float(f_max[0] - f_min[0]),
                          float(f_max[1] - f_min[1])], dtype=np.float32)
    sx, sy = float(slide_dir_xy[0]), float(slide_dir_xy[1])
    half_along = 0.5 * float(np.dot(extent_xy, np.abs([sx, sy])))
    # Object's offset from drawer center, projected onto slide_dir.
    obj_dx = float(object_xy[0]) - cx
    obj_dy = float(object_xy[1]) - cy
    obj_along_from_drawer = obj_dx * sx + obj_dy * sy
    forward_offset = max(half_along, obj_along_from_drawer) + gap_m
    base_x = cx + sx * forward_offset
    base_y = cy + sy * forward_offset

    # Robot faces -slide_dir (i.e. looking back at the cabinet). Yaw
    # angle of -slide_dir in world frame:
    yaw = float(np.arctan2(-slide_dir_xy[1], -slide_dir_xy[0]))
    qz = float(np.sin(0.5 * yaw))
    qw = float(np.cos(0.5 * yaw))

    robot.set_position_orientation(
        position=th.tensor([base_x, base_y, 0.0], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, qz, qw], dtype=th.float32),
    )
    for _ in range(5):
        og.sim.step()
    return (base_x, base_y), yaw


def main():
    args = parse_args()
    og = _init_og(headless=args.headless)
    env = _build_env(og, args.cabinet_model,
                     args.object_category, args.object_model)
    try:
        cabinet = env.scene.object_registry("name", "cabinet")
        tall_obj = env.scene.object_registry("name", "tall_obj")
        robot = env.robots[0]

        _settle_cabinet_on_floor(og, cabinet)
        cab_min, cab_max = cabinet.aabb
        cabinet_top_z = float(cab_max[2])
        print(f"[cabinet] AABB xy=[{float(cab_min[0]):.3f},"
              f"{float(cab_min[1]):.3f}]→"
              f"[{float(cab_max[0]):.3f},{float(cab_max[1]):.3f}]  "
              f"top_z={cabinet_top_z:.3f}")

        joint_name, joint = _pick_drawer_joint(cabinet)
        link_name, drawer_link = _drawer_link_for_joint(cabinet, joint)
        print(f"[drawer] driving link={link_name!r}")

        # Measure slide direction BEFORE we leave the drawer open.
        slide_dir, stroke = _measure_drawer_slide(cabinet, joint, drawer_link)
        print(f"[drawer] slide_dir_xy=({slide_dir[0]:+.2f},"
              f"{slide_dir[1]:+.2f})  stroke={stroke:.3f} m")

        # Measure the full-open drawer AABB before partial-opening, for
        # robot placement (we want the robot past where the drawer COULD
        # reach, not just past its current 20% position).
        full_open_aabb = _measure_full_open_drawer_aabb(
            og, cabinet, joint, drawer_link,
        )
        f_min, f_max = full_open_aabb
        print(f"[drawer] full-open AABB xy=[{f_min[0]:.3f},{f_min[1]:.3f}]"
              f"→[{f_max[0]:.3f},{f_max[1]:.3f}]")

        target_pos = _open_drawer(
            og, cabinet, joint_name, joint, args.open_fraction,
            args.settle_steps,
        )

        (obj_xy_z, gap, leading_xy) = _place_object_in_drawer_path(
            og, drawer_link, tall_obj, slide_dir, gap_m=args.object_gap_m,
        )
        obj_x, obj_y, obj_z = obj_xy_z
        lead_x, lead_y = leading_xy
        t_min, t_max = tall_obj.aabb
        obj_height = float(t_max[2] - t_min[2])
        # Remaining slide stroke from current open position before
        # the drawer would hit its upper limit (assuming free path).
        remaining_stroke = max(0.0, stroke * (1.0 - args.open_fraction))
        # The drawer travels in +slide_dir until something blocks it.
        # With object_gap_m < remaining_stroke, the drawer hits the
        # object first.
        blocks_open = gap < remaining_stroke
        print(f"[drawer] current_pos={target_pos:.3f}  "
              f"remaining_stroke={remaining_stroke:.3f} m")
        print(f"[drawer] leading_face xy=({lead_x:+.3f},{lead_y:+.3f})")
        print(f"[object] xy=({obj_x:+.3f},{obj_y:+.3f})  "
              f"height={obj_height:.3f} m")
        print(f"[result] gap drawer→object = {gap*100:.1f} cm  "
              f"(drawer can open up to {gap*100:.1f} cm more before "
              f"collision, but has {remaining_stroke*100:.1f} cm of "
              f"stroke left → blocks_further_open={blocks_open})")

        base_xy, yaw = _place_robot_in_front_of_drawer(
            og, robot, full_open_aabb, slide_dir, (obj_x, obj_y),
            gap_m=args.robot_gap_m,
        )
        print(f"[robot] base xy=({base_xy[0]:+.3f},{base_xy[1]:+.3f})  "
              f"yaw={np.degrees(yaw):+.1f}°")

        if not blocks_open:
            print(f"[WARN] Object→drawer gap ({gap*100:.1f} cm) >= "
                  f"remaining stroke ({remaining_stroke*100:.1f} cm) — "
                  f"drawer could open fully. Reduce --object-gap-m or "
                  f"--open-fraction.")
        else:
            print(f"[OK] Drawer can only open another {gap*100:.1f} cm "
                  f"of its remaining {remaining_stroke*100:.1f} cm "
                  f"stroke before hitting the object.")

        eyes = _place_cameras_around_drawer(
            og, env, drawer_link, slide_dir, (obj_x, obj_y),
        )
        _save_camera_snapshots(og, env, args.out_dir, eyes)

        # Hold the scene so a non-headless viewer has time to display
        # (and any external watcher can inspect the PNGs).
        import time as _time
        if args.hold_seconds > 0:
            print(f"[hold] keeping sim alive for {args.hold_seconds:.1f}s")
            t_end = _time.time() + float(args.hold_seconds)
            while _time.time() < t_end:
                og.sim.step()
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
