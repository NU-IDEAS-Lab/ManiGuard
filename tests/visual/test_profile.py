from copy import deepcopy
from pathlib import Path

import pytest


def test_default_and_canonical_profile_select_validated_dlaa():
    from maniguard.visual.profile import load_profile, renderer_settings

    root = Path(__file__).resolve().parents[2]
    profile = load_profile()
    assert profile["renderer"] == "RaytracedLighting"
    assert profile["render_deliveries"] == 4
    assert profile["glass_shadows"] == "exclude_omniglass"
    assert profile == load_profile(root / "configs/render/high_fidelity.yaml")
    assert not any("pathtracing" in key for key in renderer_settings(profile))


def test_profile_keeps_source_resolution_independent_of_policy():
    from maniguard.visual.profile import load_profile, resolve_source_resolution

    profile = load_profile()
    assert resolve_source_resolution(profile, None) == 640
    with pytest.raises(ValueError, match="resolution"):
        resolve_source_resolution(profile, 256)
    assert resolve_source_resolution(None, 256) == 256


def test_duplicate_cameras_and_unknown_options_fail():
    from maniguard.visual.profile import load_profile

    with pytest.raises(ValueError, match="camera"):
        load_profile({"camera_keys": ["image_left"] * 5})
    with pytest.raises(ValueError, match="Unknown"):
        load_profile({"total_sp": 128})


def test_renderer_setting_rejection_is_not_silently_accepted():
    from maniguard.visual.profile import apply_profile, load_profile

    class Settings:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            self.values[key] = deepcopy(value)

        def get(self, key):
            return self.values.get(key)

    s = Settings()
    report = apply_profile(s, load_profile())
    assert report["actual"]["/rtx/rendermode"] == "RaytracedLighting"
    assert report["requested"] == report["actual"]
    original_set = s.set

    def ignore_renderer(key, value):
        if key != "/rtx/rendermode":
            original_set(key, value)

    s.values.clear()
    s.set = ignore_renderer
    with pytest.raises(RuntimeError, match="rendermode"):
        apply_profile(s, load_profile())


def test_dlaa_profile_has_distinct_settings_without_pt_sampling():
    from maniguard.visual.profile import load_profile, renderer_settings

    p = load_profile({"renderer": "RaytracedLighting"})
    assert p["denoiser"] == "ray_reconstruction"
    assert p["glass_shadows"] == "exclude_omniglass"
    assert "total_spp" not in p
    settings = renderer_settings(p)
    assert settings["/rtx/post/aa/op"] == 4
    assert settings["/rtx/shadows/enabled"] is True
    assert settings["/rtx/translucency/enabled"] is True
    assert not any("pathtracing" in k for k in settings)
    for bad in ({"total_spp": 16}, {"render_deliveries": 0}, {"glass_shadows": "all"}):
        with pytest.raises(ValueError):
            load_profile({"renderer": "RaytracedLighting", **bad})
    with pytest.raises(ValueError):
        load_profile({"renderer": "PathTracing", "glass_shadows": "exclude_omniglass"})


def test_omniglass_detection_uses_shader_identity_not_material_name():
    from maniguard.visual.materials import is_omniglass_shader

    assert is_omniglass_shader("/NVIDIA/Materials/Base/Glass/OmniGlass.mdl", "OmniGlass")
    assert not is_omniglass_shader("OmniPBR.mdl", "OmniGlass")
    assert not is_omniglass_shader("OmniGlass.mdl", "water")


@pytest.mark.parametrize("selection,size", [("high_fidelity", 640), ("legacy", 256), (None, 256), ({}, 640)])
def test_visual_selection(selection, size, monkeypatch, tmp_path):
    from maniguard.visual.profile import resolve_profile_selection, resolve_source_resolution

    monkeypatch.chdir(tmp_path)
    profile = resolve_profile_selection(selection)
    assert resolve_source_resolution(profile, None) == size
