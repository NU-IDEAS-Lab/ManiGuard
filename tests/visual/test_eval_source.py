import pytest


def test_eval_source_configuration_is_explicit_and_legacy_stays_unchanged():
    from maniguard.eval.eval_config import EvalConfig, resolve_source_config

    cfg = EvalConfig()
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
