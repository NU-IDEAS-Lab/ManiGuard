import importlib.util
import sys
from pathlib import Path


def _load_module():
    mod_path = (
        Path(__file__).resolve().parents[1] / "sentinel" / "task_generation"
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

    assert missing == ["diagnostics.jsonl", "rollout_ep1.mp4", "rollout_ep2.mp4"]


def test_validate_scene_artifacts_accepts_complete_outputs(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "gates_bedroom"
    run_dir.mkdir()
    (run_dir / "diagnostics.jsonl").write_text("{}", encoding="utf-8")
    (run_dir / "rollout_ep1.mp4").write_bytes(b"fake")
    (run_dir / "rollout_ep2.mp4").write_bytes(b"fake")

    missing = mod._validate_scene_artifacts(str(run_dir), episodes=2)

    assert missing == []
