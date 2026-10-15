import pytest


def test_eval_source_configuration_is_explicit_and_legacy_stays_unchanged():
    from maniguard.eval.eval_config import EvalConfig, resolve_source_config

    cfg = EvalConfig(source_profile="legacy")
    resolve_source_config(cfg)
    assert cfg.camera_resolution == 256 and cfg.rendering_frequency == 20
    cfg = EvalConfig(source_profile={"resolution": [640, 640]}, camera_resolution=640, save_video=False)
    resolve_source_config(cfg)
    assert cfg.rendering_frequency == 60 and cfg.action_frequency == 20
    cfg = EvalConfig(source_profile={}, camera_resolution=256, save_video=False)
    with pytest.raises(ValueError, match="resolution"):
        resolve_source_config(cfg)


def test_source_mode_builds_all_views_and_selects_policy_view_without_resize():
    from maniguard.eval.benchmark import _build_eval_external_sensors
    from maniguard.eval.eval_config import EvalConfig, resolve_source_config

    cfg = EvalConfig(source_profile={}, camera_resolution=640, save_video=False)
    resolve_source_config(cfg)
    configs = _build_eval_external_sensors(cfg)
    assert {c["name"] for c in configs} == {"cam_left", "cam_right", "cam_left_shoulder", "cam_opposite"}
    assert all(c["sensor_kwargs"]["image_height"] == 640 for c in configs)


def test_default_resolves_without_repository_cwd(monkeypatch, tmp_path):
    from maniguard.eval.eval_config import EvalConfig, resolve_source_config

    monkeypatch.chdir(tmp_path)
    cfg = EvalConfig(save_video=False)
    resolve_source_config(cfg)
    resolve_source_config(cfg)
    assert cfg.camera_resolution == 640
    assert cfg.source_profile["renderer"] == "RaytracedLighting"


@pytest.mark.parametrize("selection", ["legacy", None])
def test_legacy_round_trip(tmp_path, selection):
    from maniguard.eval.eval_config import EvalConfig, load_eval_config, resolve_source_config

    cfg = EvalConfig(source_profile=selection)
    resolve_source_config(cfg)
    path = tmp_path / "eval.yaml"
    cfg.save(path)
    restored = load_eval_config(path)
    resolve_source_config(restored)
    assert restored.source_profile is None
    assert restored.camera_resolution == 256
    assert restored.rendering_frequency == 20
    assert restored.to_dict()["source_profile"] == "legacy"


@pytest.mark.parametrize("flag,expected", [(None, False), ("--save-video", True), ("--no-save-video", False)])
def test_cli_preserves_or_overrides_yaml_video(tmp_path, monkeypatch, flag, expected):
    from maniguard.eval.eval_config import config_from_cli, resolve_source_config

    path = tmp_path / "old.yaml"
    path.write_text("camera_resolution: 256\nsave_video: false\n")
    argv = ["benchmark", "--config", str(path), "--source-profile", "legacy"]
    monkeypatch.setattr("sys.argv", argv + ([flag] if flag else []))
    cfg = config_from_cli()
    resolve_source_config(cfg)
    assert cfg.source_profile is None
    assert cfg.camera_resolution == 256
    assert cfg.save_video is expected


@pytest.mark.parametrize("video", [True, False])
@pytest.mark.parametrize("record", [True, False])
def test_hq_rendering_and_output_selection_are_independent(video, record):
    from maniguard.eval.eval_config import EvalConfig, resolve_source_config

    cfg = EvalConfig(save_video=video, recording_output_dir="raw" if record else None)
    resolve_source_config(cfg)
    assert cfg.camera_resolution == 640
    assert cfg.save_video is video
    assert bool(cfg.recording_factory) is record
