"""Offline planning must not require runtime-only placement diagnostics."""
import json
from types import SimpleNamespace

import pytest

from maniguard.task_generation import pipeline_common
from maniguard.task_generation.clutter_scene_pipeline import ClutterPipeline
from maniguard.task_generation.liquid_transport_pipeline import LiquidTransportPipeline
from maniguard.task_generation.wet_transport_pipeline import WetTransportPipeline


@pytest.mark.parametrize('pipeline_cls,pipeline_name', [
    (ClutterPipeline, 'table'), (LiquidTransportPipeline, 'liquid_transport'),
    (WetTransportPipeline, 'wet_transport'),
])
def test_offline_run_writes_plan_without_claiming_runtime_measurements(tmp_path, monkeypatch, pipeline_cls, pipeline_name):
    pipeline = pipeline_cls()
    picked = dict(scene_model='fixture', category='desk', model='fixture',
                  room_instance='kitchen_0', area_m2=1.)
    monkeypatch.setattr(pipeline_common, 'pick_scene_from_placeable', lambda *a, **kw: picked)
    monkeypatch.setattr(pipeline, 'select_objects', lambda *a: {'required_area_m2': .1})
    selection = {'spawn_specs': [], 'system_name': 'water', 'overhead_margin_m': .02}
    monkeypatch.setattr(pipeline, 'generate_activity', lambda *a: ({'combined_ltl': 'true'}, selection))

    def pack(activities, *args):
        activities[0][1]['planned_layout'] = {'target': [0.1, 0.2]}

    monkeypatch.setattr(pipeline, 'offline_pack', pack)
    output = tmp_path / 'diagnostics.jsonl'
    args = SimpleNamespace(seed=0, scene_model=None, activity_name=None, dry_run=True,
        clutter_density='medium', difficulty='medium', debug_jsonl=str(output))
    pipeline._run_dry_run(args)
    row = json.loads(output.read_text())
    assert row['event'] == 'dry_run' and row['pipeline'] == pipeline_name
    assert row['selection']['planned_layout'] == {'target': [0.1, 0.2]}
    assert row['density'] == 'medium'
    if pipeline_name != 'table':
        assert row['system_name'] == 'water' and row['difficulty'] == 'medium'
    if pipeline_name == 'wet_transport':
        assert row['overhead_margin_m'] == .02
    for key in ['active_object_summary', 'removed_area_objects', 'removed_robot_base_objects',
                'resolved_video_views', 'gate_pass', 'ltl_violated']:
        assert key not in row


def test_live_diagnostics_still_require_and_include_placement_measurements():
    ctx = pipeline_common.EpisodeContext(selection={}, args=SimpleNamespace(dry_run=False, clutter_density='medium'))
    pipeline = ClutterPipeline()
    with pytest.raises(AttributeError, match='_active_object_summary'):
        pipeline.diagnostics_extra(ctx)
    ctx._active_object_summary = [{'name': 'cup_0'}]
    ctx.removed_area_objects = ['obstacle_0']
    ctx.removed_robot_base_objects = ['obstacle_1']
    ctx.resolved_video_views = [{'label': 'left_overview'}]
    extra = pipeline.diagnostics_extra(ctx)
    assert extra['active_object_summary'] == [{'name': 'cup_0'}]
    assert extra['removed_area_objects'] == ['obstacle_0']
    assert extra['removed_robot_base_objects'] == ['obstacle_1']
    assert extra['resolved_video_views'] == [{'label': 'left_overview'}]
