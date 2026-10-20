"""DLAA source defaults and explicit renderer overrides, without simulator imports."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from maniguard.utils.camera_setup import SOURCE_CAMERA_KEYS

# Built in so installed packages do not depend on a repository-relative config.
_COMMON = {
    "resolution": [640, 640],
    "camera_keys": list(SOURCE_CAMERA_KEYS),
    "tonemap": "aces",
    "motion_blur": False,
    "depth_of_field": False,
    "frame_generation": False,
    "rgb_encoding": {"codec": "jpeg", "quality": 95, "subsampling": "444", "color_order": "RGB"},
}
_RENDERERS = {
    "RaytracedLighting": {
        "name": "high_fidelity",
        "denoiser": "ray_reconstruction",
        "render_deliveries": 4,
        "glass_shadows": "exclude_omniglass",
    },
}


def load_profile(source=None) -> dict:
    """Load DLAA defaults with explicit overrides from a mapping or YAML file.

    Unknown settings fail rather than silently retaining an unintended default.
    Renderer-specific options cannot leak into the other route.
    """
    if isinstance(source, (str, Path)):
        import yaml

        with Path(source).open(encoding="utf-8") as stream:
            source = yaml.safe_load(stream)
    if source is not None and not isinstance(source, dict):
        raise ValueError("Source profile must be a mapping")
    source = source or {}
    renderer = source.get("renderer", "RaytracedLighting")
    if renderer not in _RENDERERS:
        raise ValueError(f"Unsupported renderer: {renderer!r}")
    defaults = {**_COMMON, **_RENDERERS[renderer], "renderer": renderer}
    unknown = set(source) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown source profile options: {sorted(unknown)}")
    profile = deepcopy(defaults)
    profile.update(deepcopy(source))
    keys = profile["camera_keys"]
    if not isinstance(keys, (list, tuple)) or len(keys) != 5 or set(keys) != set(SOURCE_CAMERA_KEYS):
        raise ValueError(f"camera_keys must contain the five unique source cameras: {SOURCE_CAMERA_KEYS}")
    resolution = profile["resolution"]
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or any(type(x) is not int or x <= 0 for x in resolution)
        or resolution[0] != resolution[1]
    ):
        raise ValueError("Source resolution must be two equal positive integers")
    if type(profile["render_deliveries"]) is not int or profile["render_deliveries"] <= 0:
        raise ValueError("render_deliveries must be a positive integer")
    for key in ("renderer", "denoiser", "tonemap", "motion_blur", "depth_of_field", "frame_generation"):
        if profile[key] != defaults[key] or type(profile[key]) is not type(defaults[key]):
            raise ValueError(f"Unsupported {key}: {profile[key]!r}")
    if renderer == "RaytracedLighting" and profile["glass_shadows"] not in ("native", "exclude_omniglass"):
        raise ValueError("glass_shadows must be native or exclude_omniglass")
    encoding = profile["rgb_encoding"]
    if not isinstance(encoding, dict) or set(encoding) != set(_COMMON["rgb_encoding"]):
        raise ValueError("rgb_encoding requires codec, quality, subsampling and color_order")
    if encoding["codec"] != "jpeg" or encoding["subsampling"] != "444" or encoding["color_order"] != "RGB":
        raise ValueError("Source RGB encoding must be jpeg, 444, RGB")
    if type(encoding["quality"]) is not int or not 1 <= encoding["quality"] <= 100:
        raise ValueError("JPEG quality must be an integer in [1, 100]")
    return profile


def resolve_profile_selection(selection="high_fidelity") -> dict:
    """Resolve new rendering runs to high fidelity, including omitted/null settings.

    Historical recordings can still be read without invoking this runtime selector.
    """
    if selection == "legacy":
        raise ValueError("The legacy profile has been removed; use high_fidelity for new rendering runs")
    profile = load_profile(None if selection == "high_fidelity" else selection)
    if profile["resolution"][0] < 640:
        raise ValueError("High-fidelity source resolution must be at least 640 x 640; resize policy inputs downstream")
    return profile


def resolve_source_resolution(profile: dict | None, configured_resolution: int | None) -> int:
    """Keep source size independent of policy preprocessing; reject conflicts."""
    resolution = resolve_profile_selection(profile)["resolution"][0]
    if configured_resolution is not None and configured_resolution != resolution:
        raise ValueError(f"camera_resolution={configured_resolution} conflicts with source resolution={resolution}")
    return resolution


def renderer_settings(profile: dict) -> dict:
    p = load_profile(profile)
    settings = {
        "/rtx/rendermode": p["renderer"],
        "/rtx/post/tonemap/op": 6,
        "/rtx/post/motionblur/enabled": False,
        "/rtx/post/dof/overrideEnabled": True,
        "/rtx/post/dof/enabled": False,
        "/rtx-transient/dlssg/enabled": False,
    }
    # IsaacLab 4.5 quality preset used by RoboDojo, with native-resolution DLAA.
    # Replicator tiled-render warning limits are unrelated to these five cameras.
    settings.update(
        {
            "/rtx/post/aa/op": 4,
            "/rtx/post/dlss/execMode": 2,
            "/rtx/translucency/enabled": True,
            "/rtx/translucency/maxRefractionBounces": 6,
            "/rtx/reflections/enabled": True,
            "/rtx/reflections/denoiser/enabled": True,
            "/rtx/directLighting/sampledLighting/enabled": True,
            "/rtx/sceneDb/ambientLightIntensity": 1.0,
            "/rtx/shadows/enabled": True,
            "/rtx/indirectDiffuse/enabled": True,
            "/rtx/indirectDiffuse/denoiser/enabled": True,
            "/rtx/ambientOcclusion/enabled": True,
            "/rtx/ambientOcclusion/denoiserMode": 0,
            "/rtx/raytracing/subpixel/mode": 1,
            "/rtx/raytracing/cached/enabled": True,
            "/rtx-transient/dldenoiser/enabled": True,
        }
    )
    return settings


def apply_profile(settings, profile: dict) -> dict:
    """Apply and read back Kit settings; never step or render the simulation."""
    requested = renderer_settings(profile)
    for key, value in requested.items():
        settings.set(key, value)
    actual = {key: settings.get(key) for key in requested}
    mismatches = {key: (requested[key], actual[key]) for key in requested if requested[key] != actual[key]}
    if mismatches:
        raise RuntimeError(f"Renderer rejected source settings (requested, actual): {mismatches}")
    return {"requested": requested, "actual": actual}
