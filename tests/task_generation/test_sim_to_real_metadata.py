"""Metadata export paths and recoverable malformed model metadata."""
import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[2] / "maniguard/task_generation/utils/build_sim_to_real.py"
    spec = importlib.util.spec_from_file_location("build_sim_to_real_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_model_metadata_is_skipped(tmp_path, monkeypatch, caplog):
    module = load_module()
    monkeypatch.setattr(module, "OBJECTS_ROOT", str(tmp_path))
    path = tmp_path / "mug/example/misc/metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text("{invalid")
    assert module.read_model_metadata("mug", "example") is None
    assert "read_model_metadata" in caplog.text


def test_paths_follow_repository_layout(monkeypatch):
    monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
    module = load_module()
    root = Path(__file__).resolve().parents[2]
    assert Path(module.AVG_SPECS_PATH) == root / "behavior-1k/OmniGibson/omnigibson/configs/avg_category_specs.json"
    assert Path(module.OBJECTS_ROOT) == root / "behavior-1k/datasets/behavior-1k-assets/objects"


def test_custom_asset_root_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", str(tmp_path))
    module = load_module()
    assert Path(module.OBJECTS_ROOT) == tmp_path / "behavior-1k-assets/objects"


def test_export_keeps_valid_models_after_bad_metadata(tmp_path, monkeypatch, caplog):
    import json
    import sys
    assets = tmp_path / 'datasets' / 'behavior-1k-assets' / 'objects'
    for model, text in [('good', '{"bbox_size": [0.1, 0.2, 0.3]}'), ('bad', '{invalid')]:
        path = assets / 'mug' / model / 'misc/metadata.json'
        path.parent.mkdir(parents=True)
        path.write_text(text)
    monkeypatch.setenv('OMNIGIBSON_DATA_PATH', str(tmp_path / 'datasets'))
    module = load_module()
    specs = tmp_path / 'specs.json'
    specs.write_text(json.dumps({'mug': {'mass': 0.25, 'volume': 0.001, 'density': 250.}}))
    monkeypatch.setattr(module, 'AVG_SPECS_PATH', specs)
    output = tmp_path / 'export.json'
    monkeypatch.setattr(sys, 'argv', ['build_sim_to_real', '--categories', 'mug', '--output', str(output)])
    module.main()
    entry = json.loads(output.read_text())['categories']['mug']
    assert entry['mass_kg'] == 0.25 and entry['density_kg_m3'] == 250.
    assert entry['n_models'] == 1
    assert entry['models'] == [{'model': 'good', 'bbox_size_m': [0.1, 0.2, 0.3], 'bbox_volume_m3': 0.006}]
    assert 'bad' in caplog.text and 'read_model_metadata' in caplog.text


def test_missing_metadata_is_skipped_without_logger_exception(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'OBJECTS_ROOT', tmp_path)
    assert module.read_model_metadata('mug', 'missing') is None
