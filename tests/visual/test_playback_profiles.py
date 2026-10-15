import pytest


def test_source_playback_keeps_recorded_profile_unless_explicitly_overridden():
    from maniguard.data.recording.replay import resolve_playback_profile
    from maniguard.visual.profile import load_profile

    metadata = {"source_profile": load_profile({"resolution": [512, 512]})}
    assert resolve_playback_profile(metadata)["resolution"] == [512, 512]
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
