import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    mod_path = (
        Path(__file__).resolve().parents[2] / "maniguard" / "task_generation"
        / "run_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_benchmark", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_scene_artifacts_detects_missing_files(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "gates_bedroom"
    run_dir.mkdir()

    missing = mod._validate_scene_artifacts(str(run_dir), episodes=2)

    assert "diagnostics.jsonl" in missing
    assert "scene_ep1.json" in missing and "scene_ep2.json" in missing
    assert "rollout_left_overview_ep1.mp4" in missing
    assert "rollout_left_overview_ep2.mp4" in missing


def test_validate_scene_artifacts_accepts_complete_outputs(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "gates_bedroom"
    run_dir.mkdir()
    (run_dir / "diagnostics.jsonl").write_text("".join(json.dumps({
        "episode": ep, "gate_pass": True, "ltl_violated": False}) + "\n" for ep in (1, 2)))
    for ep in (1, 2):
        (run_dir / f"scene_ep{ep}.json").write_text("{}")
    (run_dir / "rollout_ep1.mp4").write_bytes(b"fake")
    (run_dir / "rollout_ep2.mp4").write_bytes(b"fake")

    missing = mod._validate_scene_artifacts(str(run_dir), episodes=2)

    assert missing == []


def test_validate_scene_artifacts_accepts_four_camera_outputs(tmp_path):
    mod = _load_module()
    (tmp_path / 'diagnostics.jsonl').write_text('{"episode": 1, "gate_pass": true, "ltl_violated": false}\n')
    (tmp_path / 'scene_ep1.json').write_text('{}')
    for view in ['opposite_side_front', 'left_overview', 'right_overview', 'left_shoulder']:
        (tmp_path / f'rollout_{view}_ep1.mp4').write_bytes(b'video')
    assert mod._validate_scene_artifacts(str(tmp_path), 1) == []


def test_project_root_is_maniguard():
    mod = _load_module()
    assert Path(mod._PROJECT_ROOT) == Path(__file__).resolve().parents[2]


def test_standalone_pipeline_gets_supported_arguments(tmp_path, monkeypatch):
    from types import SimpleNamespace
    mod = _load_module()
    captured = []
    monkeypatch.setattr(mod.subprocess, 'run', lambda cmd, **kw: captured.append(cmd) or SimpleNamespace(returncode=1))
    for pipeline in ['cabinet_pickup', 'jar_transport']:
        monkeypatch.setattr(sys, 'argv', ['run_benchmark', '--pipeline', pipeline])
        args = mod.parse_args()
        mod._run_scene(None, args, str(tmp_path / pipeline))
        assert '--mount-gap-m' not in captured[-1]
        assert '--strict-gate' not in captured[-1]
        assert '--no-strict-gate' not in captured[-1]


def test_standalone_pipeline_rejects_room_selection(monkeypatch):
    import pytest
    mod = _load_module()
    monkeypatch.setattr(sys, 'argv', ['run_benchmark', '--pipeline', 'cabinet_pickup', '--scenes', 'Rs_int'])
    with pytest.raises(SystemExit):
        mod.parse_args()
