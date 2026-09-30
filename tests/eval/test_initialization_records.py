"""Exercise real per-scene preparation/error persistence without Isaac Sim."""
import ast
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_rollout_diagnostics import SOURCE


def _preparation_code():
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple))
    # Keep real preparation and its handlers; omit the policy loop and video I/O.
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Try)
               and any(isinstance(x, ast.While) for x in n.body))
    loop.body = loop.body[:end] + ast.parse('prepared.append(scene_info["name"])').body
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == '_initialization_failure']
    return compile(ast.fix_missing_locations(ast.Module(body=helpers + [loop], type_ignores=[])),
                   str(SOURCE), 'exec')


def run_preparation(tmp_path, monkeypatch, stage):
    current = {'name': None}

    def fail(where):
        if stage == where and current['name'] == 'task_0000/base':
            raise RuntimeError(f'{where} unavailable')

    def override(scene, *_):
        current['name'] = scene['name']
        fail('scene_configuration')
        return scene

    action_space = SimpleNamespace(shape=(8,), low=np.full(8, -1.), high=np.full(8, 1.))
    robot = SimpleNamespace(sensors={}, grasping_mode='physical', action_space=action_space,
                            get_joint_positions=lambda: torch.full((8,), float('nan') if stage == 'nonfinite_warmup' else 0.),
                            arm_control_idx={'0': list(range(7))}, default_arm='0', links={})
    env = SimpleNamespace(robots=[robot], external_sensors={}, reset=lambda: None,
                          step=lambda _: fail('warmup'),
                          scene=SimpleNamespace(objects=[], object_registry=lambda *args: fail('engagement_initialization')))

    def environment(**kw):
        fail('scene_load')
        return env

    def checker(*args):
        fail('goal_checker')
        return None

    def observe(*args):
        fail('initial_observation')
        return {'states': np.full(8, np.inf if stage == 'nonfinite_initial_observation' else 0.)}

    def monitor(*args, **kw):
        fail('monitor_initialization')
        return SimpleNamespace(reset=lambda: None)

    def initial_safety(*args):
        fail('initial_safety')

    def active(*args):
        fail('object_binding')
        return {}

    for module, members in {
        'maniguard.eval.horizon_override': {'apply_horizon_override': override},
        'omnigibson.sensors': {'VisionSensor': type('VisionSensor', (), {})},
        'maniguard.utils.camera_setup': {'place_recorded_task_cameras': lambda *a, **kw: None},
        'maniguard.data.bench_builder.perturbation': {'apply_perturbation': lambda *a: {}},
        'maniguard.eval.goal_checker': {'build_goal_checker': checker},
        'maniguard.utils.safety_monitor': {'TaskLTLMonitor': monitor},
        'omnigibson.object_states': {'ContactBodies': object},
    }.items():
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**members))
    scenes = [dict(name=f'task_000{i}/base', prompt='move', target_name='cup', target_rooms=[],
                   scene_file=str(tmp_path / 'scene_ep1.json'), ltl_safety={'combined_ltl': 'true'})
              for i in range(2)]
    cfg = SimpleNamespace(horizon_override='map', prompt_template=None, prompt_condition=None,
                          seed=0, action_dim=8, camera_resolution=224, override_controller_config=None,
                          controller_preset=None, grasping_mode='physical', ik_eef_to_joint=False,
                          save_video=False, save_wrist_video=False)
    if stage == 'wrong_controller_width':
        cfg.action_dim = 9
    elif stage == 'wrong_eef_width':
        cfg.ik_eef_to_joint = True
    ns = dict(scenes=scenes, cfg=cfg, resolved_root=tmp_path, Path=Path, json=json,
              np=np, torch=torch, og=SimpleNamespace(sim=SimpleNamespace(stop=lambda: None,
                  render=lambda: None), clear=lambda: None, Environment=environment),
              build_og_config=lambda *a: {}, extract_obs=observe,
              _build_active_objects_for_ltl=active, _validate_initial_safety=initial_safety,
              run_success=True, run_safety=True, metrics=['success', 'safety'],
              all_results=[], results_path=tmp_path / 'results.jsonl', prepared=[])
    exec(_preparation_code(), ns)
    return ns


@pytest.mark.parametrize('stage', ['scene_configuration', 'scene_load', 'goal_checker', 'warmup',
                                  'initial_observation', 'object_binding', 'monitor_initialization',
                                  'initial_safety', 'engagement_initialization'])
def test_failed_preparation_is_saved_once_and_next_scene_can_prepare(tmp_path, monkeypatch, stage):
    ns = run_preparation(tmp_path, monkeypatch, stage)
    rows = [json.loads(s) for s in ns['results_path'].read_text().splitlines()]
    assert len(rows) == 1 and rows == ns['all_results']
    row = rows[0]
    assert row['scene_name'] == 'task_0000/base' and row['seed'] == 0
    assert row['status'] == ('load_failed' if stage == 'scene_load' else 'initialization_failed')
    assert row['success'] is row['ltl_violated'] is row['counted_violation'] is None
    assert row['safety_evaluated'] is False and row['steps'] == 0
    diag = row['rollout_diagnostics']
    assert diag['exception']['type'] == 'RuntimeError'
    assert f'{stage} unavailable' in diag['exception']['traceback']
    assert diag['phase'] == ('monitor_initialization' if stage == 'object_binding' else stage)
    assert ns['prepared'] == ['task_0001/base']


def test_successful_preparation_does_not_write_failure_row(tmp_path, monkeypatch):
    ns = run_preparation(tmp_path, monkeypatch, None)
    assert ns['prepared'] == ['task_0000/base', 'task_0001/base']
    assert ns['all_results'] == [] and not ns['results_path'].exists()


@pytest.mark.parametrize('status', ['load_failed', 'initialization_failed', 'crashed', 'numerical_failed'])
@pytest.mark.parametrize('has_completed', [False, True])
def test_saved_summary_rejects_partial_run(tmp_path, status, has_completed):
    rows = [{'status': status}]
    if has_completed:
        rows.insert(0, {'status': 'completed', 'success': True, 'ever_contacted': True})
    results = tmp_path / 'results.jsonl'
    results.write_text('\n'.join(json.dumps(r) for r in rows))
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_rows' for t in n.targets))
    exits = []
    exec(compile(ast.Module(body=main.body[start:], type_ignores=[]), str(SOURCE), 'exec'),
         dict(json=json, results_path=results, output_dir=tmp_path, metrics=['success', 'safety'],
              run_success=True, sys=SimpleNamespace(stdout=io.StringIO()),
              os=SimpleNamespace(_exit=exits.append)))
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['success_rate'] is None and summary['contact_gated_violation_rate'] is None
    assert summary['n_numerical_failed'] == int(status == 'numerical_failed')
    assert summary['n_failed'] == 1 and summary['n_attempted'] == (2 if has_completed else 1)
    assert exits == [1]


@pytest.mark.parametrize('stage,phase', [('nonfinite_warmup', 'warmup'),
                                       ('nonfinite_initial_observation', 'initial_observation')])
def test_nonfinite_initial_state_cannot_reach_policy(tmp_path, monkeypatch, stage, phase):
    ns = run_preparation(tmp_path, monkeypatch, stage)
    assert ns['prepared'] == []
    rows = [json.loads(s) for s in ns['results_path'].read_text().splitlines()]
    assert len(rows) == 2
    assert all(r['status'] == 'initialization_failed' for r in rows)
    assert all(r['rollout_diagnostics']['phase'] == phase for r in rows)
    assert all(r['rollout_diagnostics']['exception']['type'] == 'FloatingPointError' for r in rows)


@pytest.mark.parametrize('stage', ['wrong_controller_width', 'wrong_eef_width'])
def test_controller_and_policy_conventions_must_agree_before_warmup(tmp_path, monkeypatch, stage):
    ns = run_preparation(tmp_path, monkeypatch, stage)
    assert ns['prepared'] == []
    assert len(ns['all_results']) == 2
    for row in ns['all_results']:
        assert row['status'] == 'initialization_failed'
        assert 'action' in row['error'].lower()
        assert row['rollout_diagnostics']['phase'] == 'action_space'
        assert row['rollout_diagnostics']['exception']['type'] == 'ValueError'
