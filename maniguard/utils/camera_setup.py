"""Canonical camera setup shared across task-generation, teleop, training, and eval.

Three external cameras (opposite / left / right of the robot) are the
canonical layout. The pipeline computes their positions per episode; every
downstream consumer (teleop, finetuning, evaluation) just loads the same
sensor configs and optionally picks which view to feed the policy.

Policy input composition: if `policy_external_cameras` selects one camera,
its RGB is returned unchanged. If it selects multiple, they are horizontally
concatenated along the width axis -- simplest scheme that keeps a single
(H, W, 3) tensor for downstream models.
"""

from __future__ import annotations

from typing import Iterable, Sequence


CAMERA_RESOLUTION = 256
EXTERNAL_CAMERA_NAMES = ("cam_opposite", "cam_left", "cam_right", "cam_left_shoulder")
POLICY_EXTERNAL_CAMERAS_DEFAULT = ("cam_opposite",)


def left_shoulder_eye(rp, forward, left, cam_z, back=0.43, side=0.55) -> tuple:
    """Canonical over-the-left-shoulder camera eye, computed identically in both camera modes.

    Behind-and-to-the-left of the robot at height ``cam_z``, looking forward at the workspace
    -> the robot's left shoulder/arm sits in the foreground with the pack beyond. ``rp`` = robot
    base position, ``forward`` = ground-plane unit vector toward the pack, ``left`` = up x forward
    (3-vectors; index access only). Defined DIRECTLY from the robot frame (NOT as a blend of the
    opposite+left eyes), so it stays a true over-the-shoulder view regardless of where the
    opposite cam sits.
    """
    return (
        float(rp[0] - forward[0] * back + left[0] * side),
        float(rp[1] - forward[1] * back + left[1] * side),
        float(cam_z),
    )


def build_external_camera_configs(
    names: Sequence[str] = EXTERNAL_CAMERA_NAMES,
    resolution: int | tuple[int, int] | None = None,
    modalities: Sequence[str] = ("rgb",),
) -> list[dict]:
    """Return a list of VisionSensor config dicts ready for env_config['external_sensors'].

    Args:
        names: sensor names; also used for the /{name} relative_prim_path.
        resolution: int for square, or (height, width) tuple, or None to
            leave image_height/image_width unset -- OmniGibson's VisionSensor
            will fall back to Kit's viewport default (typically 1280x720).
            Only set this when you need a specific size for policy input
            (training/eval); pair it with the setter+load_observation_space
            workaround at the call site, since sensor_kwargs alone is
            unreliable (see StanfordVL/OmniGibson#266, #1875).
        modalities: sensor modalities; defaults to ["rgb"].
    """
    sensor_kwargs: dict = {}
    if resolution is not None:
        if isinstance(resolution, int):
            image_height = image_width = int(resolution)
        else:
            image_height, image_width = int(resolution[0]), int(resolution[1])
        sensor_kwargs = {"image_height": image_height, "image_width": image_width}
    return [
        {
            "sensor_type": "VisionSensor",
            "name": name,
            "relative_prim_path": f"/{name}",
            "modalities": list(modalities),
            "sensor_kwargs": sensor_kwargs,
        }
        for name in names
    ]


def compose_main_image(rgb_by_camera: dict, camera_names: Iterable[str]):
    """Concatenate selected camera RGB frames along width to form a single 'main' image.

    A single camera is returned unchanged. Multiple cameras are concatenated in
    the given order along the width axis (axis=-2 for (H, W, 3) tensors).
    Works for numpy arrays and torch tensors.
    """
    names = list(camera_names)
    if not names:
        raise ValueError("compose_main_image requires at least one camera name")
    if len(names) == 1:
        return rgb_by_camera[names[0]]
    imgs = [rgb_by_camera[n] for n in names]
    # Prefer the tensor's own concat to stay on the same device / backend.
    if hasattr(imgs[0], "__module__") and "torch" in type(imgs[0]).__module__:
        import torch
        return torch.cat(imgs, dim=-2)
    import numpy as np
    return np.concatenate(imgs, axis=-2)


def normalize_policy_cameras(value) -> list[str]:
    """Coerce config values (None / str / list) into a list of camera names."""
    if value is None:
        return list(POLICY_EXTERNAL_CAMERAS_DEFAULT)
    if isinstance(value, str):
        return [value]
    return list(value)


def compute_robot_frame_views(env) -> list:
    """Compute the 4 canonical robot-frame external-camera views from the robot base.

    Returns view dicts (label / eye / lookat) for opposite / left / right / left_shoulder,
    placed relative to the robot's base frame (forward = base +X projected to ground).
    Pass the result to ``task_generation.utils.video.setup_cameras(env, views)`` to apply
    the poses to the env AND get back the posed specs (eye/orientation/sensor_name) to
    stamp into ``diagnostics['cameras']``. This is the canonical bench camera placement —
    robot-frame, independent of any ``support_surface`` (so it is identical across families).
    """
    import numpy as np
    import omnigibson.utils.transform_utils as _T

    robot = env.robots[0]
    rp_t, rq_t = robot.get_position_orientation()
    rp = np.asarray(rp_t.cpu().numpy() if hasattr(rp_t, "cpu") else rp_t, dtype=np.float32)
    rmat_t = _T.quat2mat(rq_t)
    rmat = np.asarray(rmat_t.cpu().numpy() if hasattr(rmat_t, "cpu") else rmat_t, dtype=np.float32)
    forward = rmat[:, 0].copy()
    forward[2] = 0.0
    n = float(np.linalg.norm(forward))
    forward = forward / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)

    cam_height_off, back_off, side_off, side_forward_off = 0.9, 1.2, 1.0, 0.2
    workspace_off = 0.45
    workspace = rp + forward * workspace_off + np.array([0, 0, 0.05], dtype=np.float32)
    # opposite_side_front: across the workspace from the robot (beyond the pack), looking
    # back at it -> depth far->near = robot -> pack -> cam (NOT behind the robot).
    opp_eye = rp + forward * (workspace_off + back_off) + np.array([0, 0, cam_height_off], dtype=np.float32)
    left_eye = rp + left * side_off + forward * side_forward_off \
        + np.array([0, 0, cam_height_off], dtype=np.float32)
    right_eye = rp - left * side_off + forward * side_forward_off \
        + np.array([0, 0, cam_height_off], dtype=np.float32)
    return [
        {"label": "opposite_side_front", "eye": opp_eye.tolist(),  "lookat": workspace.tolist()},
        {"label": "left_overview",       "eye": left_eye.tolist(), "lookat": workspace.tolist()},
        {"label": "right_overview",      "eye": right_eye.tolist(),"lookat": workspace.tolist()},
        {"label": "left_shoulder",       "eye": list(left_shoulder_eye(rp, forward, left, rp[2] + cam_height_off)), "lookat": workspace.tolist()},
    ]


def setup_external_cameras_robot_frame(env) -> None:
    """Position the external cameras (cam_opposite / cam_left / cam_right) at the
    canonical poses used during teleop collection + playback re-render, so every
    downstream consumer sees the SAME third-person views.

    This is the single source of truth for external-camera placement, shared by
    teleop (so101 / gello), playback re-render, and eval — so an eval rollout's
    image_left / image_right match the training data's views exactly (no OOD from
    a different camera pose).

    Two modes, in priority order (mirrors the original teleop/playback logic):
      1. support_surface present  -> build_video_view_specs from robot + a target
         object + the support surface (the canonical "opposite-side overview").
      2. no support_surface (e.g. 6fam-base HF furnished scenes) -> robot-frame
         fallback: place opp/left/right relative to the robot's base frame.

    setup_cameras positions whichever of cam_opposite / cam_left / cam_right
    actually exist in the env, so this works unchanged for both 2-cam and 3-cam
    external-sensor sets. All heavy imports are in-function (lazy) to avoid a
    circular import with task_generation.utils.video (which imports this module).
    """
    try:
        from maniguard.task_generation.utils.video import (
            build_video_view_specs,
            setup_cameras,
        )
    except ImportError as e:
        print(f"[camera_setup] WARNING: camera setup helpers not importable ({e}); "
              f"external cameras will stay at their default pose.")
        return

    scene = env.scene
    robot = env.robots[0]
    support_obj = scene.object_registry("name", "support_surface")

    if support_obj is not None:
        # Pick any non-robot, non-support object as the "target" for the lookat.
        target_obj = next(
            (obj for obj in scene.objects if obj is not robot and obj is not support_obj),
            support_obj,
        )
        views = build_video_view_specs(
            None, robot, target_obj, support_obj=support_obj,
        )
        setup_cameras(env, views)
        print(f"[camera_setup] mode = support_surface "
              f"(target={target_obj.name}, support={support_obj.name}).")
        return

    # Robot-frame fallback for scenes with no 'support_surface' (e.g. the 6fam-base
    # benchmark scenes). The canonical robot-frame placement now lives in
    # compute_robot_frame_views (shared with the bench render step).
    views = compute_robot_frame_views(env)
    setup_cameras(env, views)
    print(f"[camera_setup] mode = robot-frame ({len(views)} views)")
