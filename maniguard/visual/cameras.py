"""Five source views, without policy selection or image resizing."""

from __future__ import annotations

import numpy as np

from maniguard.utils.camera_setup import SOURCE_EXTERNAL_SENSORS
from maniguard.visual.profile import load_profile


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def disable_headless_previews(viewports, sensors: dict, *, headless: bool) -> list[str]:
    """Stop GUI-only render products; leave sensor annotators rendering.

    OmniGibson creates a viewport in addition to each sensor's render product.
    Rendering both is redundant in headless collection. Protect shared render
    products and leave interactive sessions unchanged.
    """
    if not headless:
        return []
    source_paths = {str(sensor.render_product.path) for sensor in sensors.values()}
    disabled = []
    for viewport in viewports:
        api = viewport.viewport_api
        if str(api.get_render_product_path()) not in source_paths:
            api.updates_enabled = False
            disabled.append(viewport.name)
    return disabled


def configure_source_sensors(env, profile: dict) -> dict:
    """Set actual render-product sizes, retaining recorded poses and optics.

    Require a unique hand-mounted camera rather than silently substituting a
    different robot view. Camera placement remains the scene loader's job.
    """
    p = load_profile(profile)
    sensors = {}
    for key, name in SOURCE_EXTERNAL_SENSORS.items():
        sensor = (env.external_sensors or {}).get(name)
        if sensor is None:
            raise ValueError(f"Missing source camera: {name}")
        sensors[key] = sensor
    robot = env.robots[0]
    eef = robot.eef_links[robot.default_arm].prim_path
    wrist = [
        sensor
        for sensor in robot.sensors.values()
        if getattr(sensor, "prim_path", "").rsplit("/", 1)[0] == eef and hasattr(sensor, "image_height")
    ]
    if len(wrist) != 1:
        raise ValueError(f"Expected one hand-mounted wrist camera, found {len(wrist)}")
    sensors["wrist_image"] = wrist[0]
    height, width = p["resolution"]
    for key, sensor in sensors.items():
        if sensor.image_height != height:
            sensor.image_height = height
        if sensor.image_width != width:
            sensor.image_width = width
        if (sensor.image_height, sensor.image_width) != (height, width):
            raise RuntimeError(f"Source camera {key} rejected resolution {height}x{width}")
    env.load_observation_space()
    return {key: sensors[key] for key in p["camera_keys"]}


def read_rgb(sensors: dict, profile: dict) -> dict:
    """Copy RGB buffers after the caller's render barrier; missing data fails."""
    p = load_profile(profile)
    frames = {}
    for key in p["camera_keys"]:
        observation, _ = sensors[key].get_obs()
        if "rgb" not in observation:
            raise ValueError(f"No RGB from source camera {key}")
        rgb = as_numpy(observation["rgb"])
        if rgb.ndim != 3 or rgb.shape[:2] != tuple(p["resolution"]) or rgb.shape[2] not in (3, 4):
            raise ValueError(f"Source camera {key} returned unexpected shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"Source camera {key} must return uint8, got {rgb.dtype}")
        frames[key] = rgb[..., :3].copy()
    return frames


def camera_metadata(sensors: dict) -> dict:
    """Per-observation poses; intrinsics use pixel coordinates and xyzw quaternions."""
    result = {}
    for key, sensor in sensors.items():
        position, orientation = sensor.get_position_orientation()
        result[key] = {
            "position_world_m": as_numpy(position).tolist(),
            "orientation_world_xyzw": as_numpy(orientation).tolist(),
            "intrinsic_matrix": as_numpy(sensor.intrinsic_matrix).tolist(),
            "resolution_hw": [sensor.image_height, sensor.image_width],
            "focal_length_mm": float(sensor.focal_length),
            "horizontal_aperture_mm": float(sensor.horizontal_aperture),
            "horizontal_fov_rad": float(2 * np.arctan(sensor.horizontal_aperture / (2 * sensor.focal_length))),
            "camera_axes": "USD: +X right, +Y up, -Z forward",
        }
    return result


def playback_observation(obs, robot_name, capture):
    """Refresh flattened playback RGB after state restoration, without stepping physics."""
    import torch

    observation = capture.capture(capture.observation_id + 1)
    frames = observation["camera_frames"]
    result = dict(obs)
    for key, sensor in SOURCE_EXTERNAL_SENSORS.items():
        flat_key = f"external::{sensor}::rgb"
        if flat_key in result:
            result[flat_key] = torch.from_numpy(frames[key])
    # Match the hand-mounted camera, never a base/head camera.
    candidates = [key for key in result if key.startswith(f"{robot_name}::") and key.endswith("::rgb")]
    wrist = [key for key in candidates if any(token in key.lower() for token in ("eef_link", "wrist", "hand"))]
    if len(wrist) != 1:
        raise ValueError(f"Expected one wrist RGB observation in playback, found {wrist}")
    result[wrist[0]] = torch.from_numpy(frames["wrist_image"])
    return result
