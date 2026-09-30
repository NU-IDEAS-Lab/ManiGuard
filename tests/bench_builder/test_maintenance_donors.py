"""Maintenance tools use explicit donor sources without touching real benchmark data."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_tool(kind):
    path = ROOT / 'tools/bench_surgery' / kind / ('swap_content.py' if kind == 'jar' else 'swap_object.py')
    spec = importlib.util.spec_from_file_location('swap_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def donor_scene(root):
    path = root / 'task_0000/base/scene_ep1.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'objects_info': {'init_info': {'donor': {'args': {
        'category': 'apple', 'model': 'new', 'scale': [1,2,3], 'expected_file_hash': None}}}}}))


@pytest.mark.parametrize('kind', ['jar', 'stack'])
def test_explicit_donor_root_supplies_asset_parameters(tmp_path, kind):
    tool = load_tool(kind)
    donor_scene(tmp_path)
    assert tool._donor_args('apple', 'new', [tmp_path]) == {'scale': [1,2,3], 'expected_file_hash': None}


def task_fixture(path):
    path.mkdir()
    scene = {'objects_info': {'init_info': {'old_item': {'args': {
        'name': 'old_item', 'category': 'pear', 'model': 'old', 'expected_file_hash': 'old-hash'}}}},
        'state': {'registry': {'object_registry': {
            'old_item': {'root_link': {'pos': [.1,.2,.3], 'ori': [0,0,0,1]}},
            'jar': {'root_link': {'pos': [0,0,0]}}}}}}
    diag = {'selection': {'item_category': 'pear', 'item_model': 'old', 'spawn_specs': []},
            'item_info': {'name': 'old_item'}, 'jar_info': {'name': 'jar'}, 'prompt': 'move the pear'}
    (path / 'scene_ep1.json').write_text(json.dumps(scene))
    (path / 'diagnostics.jsonl').write_text(json.dumps(diag, indent=2))


def test_jar_swap_reads_multiline_diagnostics_and_clears_old_asset_hash(tmp_path):
    tool = load_tool('jar')
    donors = tmp_path / 'donors'
    donor_scene(donors)
    task = tmp_path / 'task'
    task_fixture(task)
    tool.swap(str(task), 'apple', 'new', donor_roots=[donors])
    scene = json.loads((task / 'scene_ep1.json').read_text())
    args = scene['objects_info']['init_info']['food_apple_ep1_1']['args']
    assert args['category'] == 'apple' and args['model'] == 'new'
    assert args['scale'] == [1,2,3] and 'expected_file_hash' not in args
    assert scene['state']['registry']['object_registry']['food_apple_ep1_1']['root_link']['pos'] == [.1,.2,.3]
    assert json.loads((task / 'diagnostics.jsonl').read_text())['prompt'] == 'move the apple'


def test_missing_donor_does_not_modify_task_files(tmp_path):
    tool = load_tool('jar')
    task = tmp_path / 'task'
    task_fixture(task)
    before = {p.name: p.read_bytes() for p in task.iterdir()}
    with pytest.raises(SystemExit, match='donor'):
        tool.swap(str(task), 'apple', 'missing', donor_roots=[tmp_path / 'absent'])
    assert {p.name: p.read_bytes() for p in task.iterdir()} == before


def test_stack_thickness_uses_configured_asset_root(tmp_path, monkeypatch):
    monkeypatch.setenv('OMNIGIBSON_DATA_PATH', str(tmp_path))
    path = tmp_path / 'behavior-1k-assets/objects/apple/new/misc/metadata.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"bbox_size": [0.1,0.2,0.3]}')
    assert load_tool('stack')._donor_thickness('apple', 'new') == .3
