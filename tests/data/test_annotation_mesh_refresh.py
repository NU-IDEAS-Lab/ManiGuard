"""Refresh generated meshes without losing existing object metadata."""
import json
import sys
from types import SimpleNamespace

import pytest
import trimesh

from maniguard.data.datagen.annotation import extract_meshes as extraction


def write_task(root, family, *, multiline=False):
    task = root / family / 'task_0000/base'
    task.mkdir(parents=True)
    diag = {'goal_region': {'target_name': 'cup_0'}}
    scene = {'objects_info': {'init_info': {'cup_0': {'args': {'category': 'mug', 'model': 'model'}}}}}
    (task / 'diagnostics.jsonl').write_text(json.dumps(diag, indent=2 if multiline else None)+'\n')
    (task / 'scene_ep1.json').write_text(json.dumps(scene))


def test_shared_model_keeps_all_requested_families(tmp_path, monkeypatch):
    for family in ['clutter_pickup', 'jar_transport']:
        write_task(tmp_path, family)
    monkeypatch.setattr(extraction, 'BENCH', tmp_path)
    obj = extraction.enumerate_targets(['clutter_pickup', 'jar_transport'])['mug/model']
    assert obj['source_task'] == 'clutter_pickup/task_0000'
    assert obj['families'] == ['clutter_pickup', 'jar_transport']


def test_multiline_diagnostics_are_not_silently_omitted(tmp_path, monkeypatch):
    write_task(tmp_path, 'clutter_pickup', multiline=True)
    monkeypatch.setattr(extraction, 'BENCH', tmp_path)
    assert 'mug/model' in extraction.enumerate_targets(['clutter_pickup'])


@pytest.mark.parametrize('broken', [False, True])
def test_mesh_refresh_preserves_annotations_and_reports_extraction_failure(tmp_path, monkeypatch, broken):
    from maniguard.data.datagen.primitives import scene, grasp_obb
    import omnigibson as og
    output = tmp_path / 'annotation'
    output.mkdir()
    existing = {'category': 'mug', 'model': 'model', 'source_task': 'cabinet_pickup/task_0001',
                'families': ['cabinet_pickup'], 'bbox_size': [9,9,9], 'mesh': 'meshes/old.glb',
                'grasps': [{'pose': [1,2,3], 'note': 'hand-authored'}],
                'mesh_bare': 'meshes/bare.glb', 'custom_annotation': {'keep': True}}
    other = {'category': 'bowl', 'model': 'other', 'custom_annotation': 'unchanged'}
    (output / 'mesh_db.json').write_text(json.dumps({'objects': {'mug/model': existing, 'bowl/other': other}}))
    annotations = output / 'grasp_annotations.json'
    annotations.write_text('{"manual": "do not edit"}')
    original = annotations.read_bytes()
    target = {'category': 'mug', 'model': 'model', 'source_task': 'clutter_pickup/task_0000',
              'families': ['clutter_pickup', 'jar_transport'], 'upright_orientation_xyzw': [0,0,0,1]}
    monkeypatch.setattr(extraction, 'OUT', output)
    monkeypatch.setattr(extraction, 'enumerate_targets', lambda families: {'mug/model': target})
    monkeypatch.setattr(scene, 'init_omnigibson', lambda **kw: og)
    monkeypatch.setattr(og, 'sim', SimpleNamespace(step=lambda: None, stop=lambda: None))
    monkeypatch.setattr(og, 'Environment', lambda **kw: SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *a: object())))

    def mesh(*a, **kw):
        if broken:
            raise RuntimeError('mesh unavailable')
        return trimesh.creation.box(extents=[.1,.2,.3])

    monkeypatch.setattr(grasp_obb, 'mesh_from_og_object', mesh)
    monkeypatch.setattr(sys, 'argv', ['extract_meshes'])
    result = extraction.main()
    assert result == (1 if broken else 0)
    db = json.loads((output / 'mesh_db.json').read_text())
    assert annotations.read_bytes() == original
    assert db['objects']['bowl/other'] == other
    if broken:
        assert db['objects']['mug/model'] == existing
    else:
        updated = db['objects']['mug/model']
        assert updated['grasps'] == existing['grasps']
        assert updated['mesh_bare'] == 'meshes/bare.glb'
        assert updated['custom_annotation'] == {'keep': True}
        assert updated['source_task'] == 'cabinet_pickup/task_0001'
        assert updated['families'] == ['cabinet_pickup', 'clutter_pickup', 'jar_transport']
        assert updated['bbox_size'] == [.1,.2,.3]
        assert (output / updated['mesh']).is_file()
