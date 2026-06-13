"""Camera + video helpers for task-generation rollouts.

Builds support-relative camera views (opposite / left / right of the robot),
positions them in the scene, and drives a PyAV video writer that stitches
viewer + wrist camera frames into per-episode MP4 files.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch as th
import logging

log = logging.getLogger(__name__)


def init_video_writer(base_path, episode, fps, robot=None, frame_hw=None):
    """Initialize a PyAV video writer.

    Args:
        frame_hw: (height, width) of the frames that will be pushed into the
            stream. If None, falls back to viewer_camera's resolution -- which
            is only correct if you're actually rendering viewer frames. For
            external_sensor recordings, pass the sensor's image_height/width
            so PyAV doesn't silently upscale via swscale.
    """
    try:
        import av
    except ImportError:
        print("[Pipeline] WARNING: PyAV not available — video recording disabled.")
        return None
    import omnigibson as og

    if frame_hw is not None:
        vh, vw = int(frame_hw[0]), int(frame_hw[1])
    else:
        try:
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            vh, vw = int(rgb.shape[0]), int(rgb.shape[1])
        except Exception:
            vh, vw = 720, 1280

    # Find wrist camera for picture-in-picture overlay.
    wrist = None
    wrist_name = None
    sensor_names = []
    if robot:
        from omnigibson.sensors import VisionSensor
        for name, sensor in robot.sensors.items():
            sensor_names.append(name)
            if isinstance(sensor, VisionSensor) and "hand" in name.lower():
                wrist = sensor
                wrist_name = name
                break
        if wrist is None:
            for name, sensor in robot.sensors.items():
                if isinstance(sensor, VisionSensor):
                    wrist = sensor
                    wrist_name = name
                    break
    wh, ww = 0, 0
    if wrist:
        try:
            wrist_rgb = wrist.get_obs()[0].get("rgb")
            if wrist_rgb is not None:
                wh, ww = int(wrist_rgb.shape[0]), int(wrist_rgb.shape[1])
        except Exception as exc:
            log.warning("init_video_writer: wrist rgb probe failed: %s", exc)
            pass

    if wh > 0 and ww > 0:
        scale = vh / wh
        scaled_ww = int(ww * scale)
        total_w = vw + scaled_ww
    else:
        total_w = vw

    stem = base_path[:-4] if base_path.endswith(".mp4") else base_path
    fpath = f"{stem}_ep{episode + 1}.mp4"
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)

    container = av.open(fpath, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = total_w
    stream.height = vh
    stream.pix_fmt = "yuv420p"

    if robot:
        print(
            "[Pipeline] Robot sensors: "
            + (", ".join(sensor_names) if sensor_names else "<none>")
        )
    if wrist_name:
        print(f"[Pipeline] Wrist sensor selected: {wrist_name} ({ww}x{wh})")
    elif robot:
        print("[Pipeline] Wrist sensor selected: <none>; falling back to viewer-only frames")

    return {"container": container, "stream": stream, "wrist": wrist,
            "wrist_name": wrist_name, "viewer_hw": (vh, vw), "wrist_hw": (wh, ww)}


def expected_video_path(base_path, episode):
    stem = base_path[:-4] if base_path.endswith(".mp4") else base_path
    return f"{stem}_ep{episode + 1}.mp4"


def _object_world_position(obj):
    return np.asarray([float(v) for v in obj.get_position_orientation()[0][:3]], dtype=np.float32)


def _support_relative_video_views(robot, target_obj, support_obj=None, active_objects_by_inst=None):
    rp = _object_world_position(robot)
    tp = _object_world_position(target_obj)

    if support_obj is not None:
        try:
            aabb_min, aabb_max = support_obj.aabb
            x0, y0, z0 = [float(v) for v in aabb_min[:3]]
            x1, y1, z1 = [float(v) for v in aabb_max[:3]]
            support_center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z1], dtype=np.float32)
            hx = max(0.10, 0.5 * abs(x1 - x0))
            hy = max(0.10, 0.5 * abs(y1 - y0))
            table_top_z = z1
        except Exception:
            support_center = np.asarray([(rp[0] + tp[0]) * 0.5, (rp[1] + tp[1]) * 0.5, max(rp[2], tp[2])], dtype=np.float32)
            hx, hy = 0.35, 0.35
            table_top_z = float(max(rp[2], tp[2]))
    else:
        support_center = np.asarray([(rp[0] + tp[0]) * 0.5, (rp[1] + tp[1]) * 0.5, max(rp[2], tp[2])], dtype=np.float32)
        hx, hy = 0.35, 0.35
        table_top_z = float(max(rp[2], tp[2]))

    cluster_positions = []
    if active_objects_by_inst:
        for obj in active_objects_by_inst.values():
            try:
                cluster_positions.append(_object_world_position(obj))
            except Exception as exc:
                log.warning("video view: cluster position lookup failed: %s", exc)
                continue
    if cluster_positions:
        cluster_center = np.mean(np.stack(cluster_positions, axis=0), axis=0)
    else:
        cluster_center = tp

    lookat = np.asarray([
        float(0.40 * support_center[0] + 0.60 * cluster_center[0]),
        float(0.40 * support_center[1] + 0.60 * cluster_center[1]),
        float(max(table_top_z + 0.15, cluster_center[2], tp[2])),
    ], dtype=np.float32)

    # Camera height: robot highest point + 0.4m
    try:
        robot_aabb_max = float(robot.aabb[1][2])
    except Exception:
        robot_aabb_max = rp[2] + 1.0  # fallback: base + ~1m
    cam_z = robot_aabb_max + 0.4

    # Surface AABB bounds
    if support_obj is not None:
        try:
            aabb_min, aabb_max = support_obj.aabb
            sx0, sy0 = float(aabb_min[0]), float(aabb_min[1])
            sx1, sy1 = float(aabb_max[0]), float(aabb_max[1])
        except Exception:
            sx0, sy0 = float(support_center[0] - hx), float(support_center[1] - hy)
            sx1, sy1 = float(support_center[0] + hx), float(support_center[1] + hy)
    else:
        sx0, sy0 = float(support_center[0] - hx), float(support_center[1] - hy)
        sx1, sy1 = float(support_center[0] + hx), float(support_center[1] + hy)

    sx_center = (sx0 + sx1) * 0.5
    sy_center = (sy0 + sy1) * 0.5
    margin = 0.3  # camera offset past surface edge

    # Determine which edge of the surface the robot is closest to
    dist_to_edges = {
        "x_min": abs(rp[0] - sx0),
        "x_max": abs(rp[0] - sx1),
        "y_min": abs(rp[1] - sy0),
        "y_max": abs(rp[1] - sy1),
    }
    robot_edge = min(dist_to_edges, key=dist_to_edges.get)
    print(f"[Camera] robot_pos=({rp[0]:.2f}, {rp[1]:.2f}), surface=x[{sx0:.2f},{sx1:.2f}] y[{sy0:.2f},{sy1:.2f}]")
    print("[Camera] dist_to_edges:", {k: round(v, 2) for k, v in dist_to_edges.items()})
    print(f"[Camera] robot_edge={robot_edge}, cam_z={cam_z:.2f}")

    # Opposite camera: on the far side of the surface from robot
    # Left/right cameras: on the two remaining sides perpendicular to robot-opposite axis
    if robot_edge == "x_min":
        opp_eye = (sx1 + margin, sy_center, cam_z)
        left_eye = (sx_center, sy0 - margin, cam_z)
        right_eye = (sx_center, sy1 + margin, cam_z)
    elif robot_edge == "x_max":
        opp_eye = (sx0 - margin, sy_center, cam_z)
        left_eye = (sx_center, sy1 + margin, cam_z)
        right_eye = (sx_center, sy0 - margin, cam_z)
    elif robot_edge == "y_min":
        opp_eye = (sx_center, sy1 + margin, cam_z)
        left_eye = (sx0 - margin, sy_center, cam_z)
        right_eye = (sx1 + margin, sy_center, cam_z)
    else:  # y_max
        opp_eye = (sx_center, sy0 - margin, cam_z)
        left_eye = (sx1 + margin, sy_center, cam_z)
        right_eye = (sx0 - margin, sy_center, cam_z)

    print(f"[Camera] opp_eye={opp_eye}, left_eye={left_eye}, right_eye={right_eye}")
    print(f"[Camera] lookat=({lookat[0]:.2f}, {lookat[1]:.2f}, {lookat[2]:.2f})")
    from maniguard.utils.camera_setup import left_shoulder_eye  # lazy: avoid circular import
    return [
        {
            "label": "opposite_side_front",
            "eye": tuple(float(v) for v in opp_eye),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": True,
        },
        {
            "label": "left_overview",
            "eye": tuple(float(v) for v in left_eye),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": False,
        },
        {
            "label": "right_overview",
            "eye": tuple(float(v) for v in right_eye),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": False,
        },
        {
            "label": "left_shoulder",
            "eye": left_shoulder_eye(opp_eye, left_eye),
            "lookat": tuple(float(v) for v in lookat),
            "canonical": False,
        },
    ]


def build_video_view_specs(args, robot, target_obj, support_obj=None,
                           active_objects_by_inst=None, camera_override=None):
    """Build three camera views: opposite, left, right relative to robot and support surface."""
    return _support_relative_video_views(
        robot=robot,
        target_obj=target_obj,
        support_obj=support_obj,
        active_objects_by_inst=active_objects_by_inst,
    )


def eye_lookat_to_quat(eye, lookat):
    """Compute camera orientation quaternion from eye position and lookat point."""
    import omnigibson.utils.transform_utils as T
    d = np.asarray(lookat, dtype=np.float32) - np.asarray(eye, dtype=np.float32)
    d = d / max(1e-6, np.linalg.norm(d))
    return T.euler2quat(th.tensor([
        math.pi / 2 + float(np.arcsin(np.clip(d[2], -1, 1))),
        0.0,
        float(np.arctan2(-d[0], d[1])),
    ], dtype=th.float32))


def setup_cameras(env, video_views):
    """Position the external cameras (one per EXTERNAL_CAMERA_NAMES, by order) and
    set the viewer to the opposite side.

    Returns the list of view dicts with position/orientation/sensor_name added — i.e.
    this both APPLIES the poses to the env and RETURNS the computed specs, so callers
    (e.g. the bench render step) can stamp them into diagnostics['cameras'].
    """
    import omnigibson as og
    from maniguard.utils.camera_setup import EXTERNAL_CAMERA_NAMES

    cam_names = list(EXTERNAL_CAMERA_NAMES)
    for view, cam_name in zip(video_views, cam_names):
        eye = [float(v) for v in view["eye"]]
        lookat = [float(v) for v in view["lookat"]]
        quat = eye_lookat_to_quat(eye, lookat).tolist()
        view["position"] = eye
        view["orientation"] = quat
        view["sensor_name"] = cam_name

        sensor = env.external_sensors.get(cam_name)
        if sensor is not None:
            sensor.set_position_orientation(position=eye, orientation=quat, frame="world")

    # Viewer camera = opposite side
    opp = video_views[0]
    og.sim.viewer_camera.set_position_orientation(
        position=opp["position"], orientation=opp["orientation"],
    )
    return video_views


def close_video_writer(vw):
    try:
        for packet in vw["stream"].encode():
            vw["container"].mux(packet)
        vw["container"].close()
    except Exception as exc:
        log.warning("close_video_writer: final packet flush failed: %s", exc)
        pass
