"""Run the generator's real rollout/diagnostics boundary with the actual monitor."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maniguard.task_generation.pipeline_common import append_jsonl, run_ltl_rollout
import omnigibson as og
from omnigibson.object_states import Dropped


SOURCE = Path(os.environ.get('MANIGUARD_TEST_EMPTY_SOURCE',
    Path(__file__).resolve().parents[2] / 'maniguard/task_generation/empty_scene_pipeline.py'))


def _episode_tail():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_episode_inner')
    # Execute the production rollout and JSONL writer, omitting physical layout
    # construction and final object parking, which require the simulator.
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'rng' for t in n.targets))
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
               and n.value.func.id == 'append_jsonl')
    return compile(ast.Module(body=fn.body[start:end + 1], type_ignores=[]), str(SOURCE), 'exec')


def run_episode(tmp_path, monkeypatch, *, falls=False, missing_binding=False):
    state = SimpleNamespace(value=False)
    state.get_value = lambda: state.value
    obj = SimpleNamespace(name='cup_0', category='cup', states={Dropped: state})
    env = SimpleNamespace(task=SimpleNamespace(object_scope={}), scene=SimpleNamespace(objects=[obj]),
                          _pre_step=lambda action: None)
    monkeypatch.setattr(og, 'sim', SimpleNamespace(step=lambda: setattr(state, 'value', falls)))
    spec = {'combined_ltl': 'G !dropped', 'propositions': {
        'dropped': {'state': 'dropped', 'over': ['absent_0' if missing_binding else 'cup_0']}}}
    path = tmp_path / 'diagnostics.jsonl'
    ns = dict(np=np, run_ltl_rollout=run_ltl_rollout, append_jsonl=append_jsonl,
        ep=0, ep_seed=7, env=env, activity_name='fixture', objects_by_inst={'cup_0': obj},
        robot=SimpleNamespace(action_space=SimpleNamespace(shape=(8,), low=np.full(8, -1.), high=np.ones(8))),
        target_obj=obj, support_obj=None, surface_cat='desk', surface_model='fixture',
        gate_pass=True, selection={}, goal_region_payload=None, ltl_safety=spec,
        args=SimpleNamespace(scene_model=None, save_video=False, steps=2, jitter_scale=0.,
                             setup='clutter', debug_jsonl=str(path)))
    exec(_episode_tail(), ns)
    return json.loads(path.read_text()), ns['summary']


@pytest.mark.parametrize('falls', [False, True])
def test_written_spec_is_used_for_monitoring(tmp_path, monkeypatch, falls):
    row, summary = run_episode(tmp_path, monkeypatch, falls=falls)
    assert row['ltl_violated'] is falls
    assert row['steps_executed'] == 2
    assert summary['formula'] == row['ltl_safety']['combined_ltl']
    assert summary['total_steps_monitored'] == 3
    assert summary['violation_step'] == (1 if falls else None)
    assert [step['ap']['dropped'] for step in summary['log']] == [False, falls, falls]


def test_invalid_task_spec_cannot_emit_a_safe_diagnostic(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        run_episode(tmp_path, monkeypatch, missing_binding=True)
    assert not (tmp_path / 'diagnostics.jsonl').exists()
