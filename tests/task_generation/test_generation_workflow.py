"""Task generation must preserve attempt identity and validate actual outputs."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path):
    spec = importlib.util.spec_from_file_location('workflow_under_test', ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def driver():
    return module('maniguard/task_generation/run_benchmark.py')


def artifacts(root, episodes=1, nested=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'diagnostics.jsonl').write_text(''.join(json.dumps({
        'episode': ep, 'gate_pass': True, 'ltl_violated': False})+'\n' for ep in range(1, episodes+1)))
    for ep in range(1, episodes+1):
        dest = root / 'snapshots' / f'ep{ep:03d}' if nested else root
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'scene_ep{ep}.json').write_text('{}')
        for view in ['opposite_side_front', 'left_overview', 'right_overview', 'left_shoulder']:
            (dest / f'rollout_{view}_ep{ep}.mp4').write_bytes(b'video')


@pytest.mark.parametrize('pipeline', ['jar_transport', 'cabinet_pickup'])
def test_standalone_outputs_in_episode_subdirectories_are_complete(tmp_path, pipeline):
    mod = driver()
    artifacts(tmp_path, episodes=2, nested=True)
    assert mod._validate_scene_artifacts(str(tmp_path), 2) == []
    (tmp_path / 'snapshots/ep002/rollout_left_overview_ep2.mp4').unlink()
    assert mod._validate_scene_artifacts(str(tmp_path), 2)


@pytest.mark.parametrize('bad', ['snapshot', 'empty_video', 'bad_json', 'missing_episode', 'bad_verdict'])
def test_incomplete_or_invalid_artifacts_are_rejected(tmp_path, bad):
    mod = driver()
    artifacts(tmp_path, episodes=2)
    if bad == 'snapshot':
        (tmp_path / 'scene_ep2.json').unlink()
    elif bad == 'empty_video':
        (tmp_path / 'rollout_left_overview_ep2.mp4').write_bytes(b'')
    elif bad == 'bad_json':
        (tmp_path / 'diagnostics.jsonl').write_text('{')
    elif bad == 'missing_episode':
        (tmp_path / 'diagnostics.jsonl').write_text('{"episode": 1, "gate_pass": true, "ltl_violated": false}\n')
    else:
        (tmp_path / 'diagnostics.jsonl').write_text('{"episode": 1, "gate_pass": null, "ltl_violated": false}\n')
    assert mod._validate_scene_artifacts(str(tmp_path), 2)


def test_resume_does_not_trust_stale_summary_csv(tmp_path):
    mod = driver()
    (tmp_path / 'summary.csv').write_text('scene,status\nscene_a,success\n')
    assert mod._find_completed_scenes(str(tmp_path), 1) == set()


@pytest.mark.parametrize('scenes', [True, False])
def test_resume_keeps_original_indices_and_complete_summary(tmp_path, monkeypatch, scenes):
    import csv
    mod = driver()
    argv = ['run', '--resume', str(tmp_path), '--num-trials', '3']
    if scenes:
        argv += ['--scenes', 'a', 'b', 'c']
    labels = ['a', 'b', 'c'] if scenes else ['trial_0', 'trial_1', 'trial_2']
    artifacts(tmp_path / labels[0])
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(mod, '_spot_preflight_or_exit', lambda: None)
    calls = []

    def run(scene, args, output_dir, scene_index=0):
        calls.append(scene_index)
        return dict(scene=labels[scene_index], status='success', duration_s=0,
                    gate_pass=True, ltl_violated=False, error='', run_dir=str(tmp_path / labels[scene_index]))

    monkeypatch.setattr(mod, '_run_scene', run)
    assert mod.main() == 0
    assert calls == [1, 2]
    rows = list(csv.DictReader((tmp_path / 'summary.csv').open()))
    assert {r['scene'] for r in rows} == set(labels)


@pytest.mark.parametrize('returncode', [0, -11])
def test_old_outputs_cannot_make_a_new_crashed_attempt_successful(tmp_path, monkeypatch, returncode):
    mod = driver()
    artifacts(tmp_path / 'trial_0')
    monkeypatch.setattr(sys, 'argv', ['run', '--pipeline', 'table'])
    args = mod.parse_args()
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=returncode))
    assert mod._run_scene(None, args, str(tmp_path))['status'] != 'success'


@pytest.mark.parametrize('returncode', [0, -11])
def test_fresh_standalone_outputs_can_survive_shutdown_error(tmp_path, monkeypatch, returncode):
    mod = driver()
    monkeypatch.setattr(sys, 'argv', ['run', '--pipeline', 'jar_transport'])
    args = mod.parse_args()

    def run(*a, **kw):
        artifacts(tmp_path / 'trial_0', nested=True)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(mod.subprocess, 'run', run)
    row = mod._run_scene(None, args, str(tmp_path))
    assert row['status'] == 'success' and row['gate_pass'] is True
    assert row['ltl_violated'] is False


def test_failed_generation_returns_nonzero(tmp_path, monkeypatch):
    mod = driver()
    monkeypatch.setattr(sys, 'argv', ['run', '--num-trials', '1', '--output-dir', str(tmp_path)])
    monkeypatch.setattr(mod, '_spot_preflight_or_exit', lambda: None)
    monkeypatch.setattr(mod, '_run_scene', lambda *a, **kw: dict(scene='trial_0', status='failed',
        duration_s=0, gate_pass=False, ltl_violated=None, error='failed', run_dir=str(tmp_path)))
    assert mod.main() == 1


def test_jar_selector_reads_configured_assets(tmp_path, monkeypatch):
    monkeypatch.setenv('OMNIGIBSON_DATA_PATH', str(tmp_path))
    path = tmp_path / 'behavior-1k-assets/objects/hinged_jar/gqtsam/misc/metadata.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'link_bounding_boxes': {'glass': {
        'collision': {'axis_aligned': {'extent': [0.14, 0.20, 0.30]}}}}}))
    jar = module('maniguard/task_generation/utils/jar_transport_pipeline/select.py')
    assert jar.jar_opening_min_dim('gqtsam') == pytest.approx(.12)


def test_snapshot_validator_root_cli_accepts_its_own_options(tmp_path, capsys):
    from maniguard.eval import snapshot_validator
    assert snapshot_validator.main(['--root', str(tmp_path), '--family', 'table', '--video-output-exact-dir']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['total_tasks'] == 0


def test_resume_refuses_changed_protocol(tmp_path, monkeypatch):
    mod = driver()
    cfg = {'pipeline': 'table', 'episodes': 1, 'seed': 7}
    config_path = tmp_path / 'benchmark_config.json'
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['run', '--resume', str(tmp_path), '--seed', '9'])
    monkeypatch.setattr(mod, '_spot_preflight_or_exit', lambda: None)
    with pytest.raises(ValueError, match='seed'):
        mod.main()
    assert json.loads(config_path.read_text()) == cfg


def test_multi_episode_verdicts_do_not_use_only_the_last_row(tmp_path, monkeypatch):
    mod = driver()
    monkeypatch.setattr(sys, 'argv', ['run', '--pipeline', 'table', '--episodes', '2'])
    args = mod.parse_args()

    def run(*a, **kw):
        out = tmp_path / 'trial_0'
        artifacts(out, episodes=2)
        (out / 'diagnostics.jsonl').write_text(
            '{"episode": 1, "gate_pass": false, "ltl_violated": true}\n'
            '{"episode": 2, "gate_pass": true, "ltl_violated": false}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, 'run', run)
    row = mod._run_scene(None, args, str(tmp_path))
    assert row['status'] == 'success'
    assert row['gate_pass'] is False and row['ltl_violated'] is True


def test_batch_exact_video_directories_are_distinct_per_task(tmp_path, monkeypatch):
    from maniguard.eval import snapshot_validator as validator
    root = tmp_path / 'tasks'
    for task in ['task_0000', 'task_0001']:
        path = root / task / 'base'
        path.mkdir(parents=True)
        (path / 'scene_ep1.json').write_text('{}')
        (path / 'diagnostics.jsonl').write_text('{}')
    destinations = []

    def validate_task(**kw):
        destinations.append(Path(kw['save_video']))
        assert kw['video_output_exact_dir'] is True
        return SimpleNamespace(to_dict=lambda: {'overall_ok': True})

    monkeypatch.setattr(validator, 'validate_task', validate_task)
    validator.validate_root(root=root, family='table', save_video=tmp_path / 'videos', video_output_exact_dir=True)
    assert destinations == [tmp_path / 'videos/task_0000/base', tmp_path / 'videos/task_0001/base']


def test_snapshot_ltl_imports_reach_documented_bddl_precondition(tmp_path):
    from maniguard.eval import snapshot_validator as validator
    checks = validator._ltl_checks(None, None, SimpleNamespace(problem_file=None), {}, horizon_steps=0)
    assert len(checks) == 1 and checks[0].ok is False
    assert checks[0].name == 'ltl_problem_available'


def test_resume_retries_failed_attempt_even_when_old_files_are_complete(tmp_path):
    mod = driver()
    artifacts(tmp_path / 'scene_a')
    (tmp_path / 'summary.csv').write_text('scene,status\nscene_a,failed\n')
    assert mod._find_completed_scenes(str(tmp_path), 1) == set()
