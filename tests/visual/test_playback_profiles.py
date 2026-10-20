import pytest


def test_source_playback_keeps_recorded_profile_unless_explicitly_overridden():
    from maniguard.data.recording.replay import resolve_playback_profile
    from maniguard.visual.profile import load_profile

    metadata = {"source_profile": load_profile({"resolution": [768, 768]})}
    assert resolve_playback_profile(metadata)["resolution"] == [768, 768]
    assert resolve_playback_profile(metadata, "high_fidelity")["resolution"] == [640, 640]
    with pytest.raises(ValueError, match="legacy"):
        resolve_playback_profile(metadata, "legacy")


def test_historical_pt_can_be_read_but_not_silently_rerendered():
    from maniguard.data.recording.replay import resolve_playback_profile
    from maniguard.visual.profile import load_profile

    metadata = {"source_profile": {"renderer": "PathTracing"}}
    with pytest.raises(ValueError, match="Unsupported renderer"):
        resolve_playback_profile(metadata)
    with pytest.raises(ValueError, match="Unsupported renderer"):
        load_profile({"renderer": "PathTracing"})
    assert resolve_playback_profile(metadata, "high_fidelity")["renderer"] == "RaytracedLighting"


def test_historical_low_resolution_rerender_requires_current_profile():
    from maniguard.data.recording.replay import resolve_playback_profile

    metadata = {"source_profile": {"resolution": [256, 256]}}
    with pytest.raises(ValueError, match="640"):
        resolve_playback_profile(metadata)
    assert resolve_playback_profile(metadata, "high_fidelity")["resolution"] == [640, 640]
