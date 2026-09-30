"""A retried task must use the current worker's result, including failed workers."""
import subprocess
from types import SimpleNamespace

import pytest

from maniguard.data.bench_builder import run_finalize_base as driver


@pytest.mark.parametrize("failure", ["crash", "timeout"])
def test_failed_retry_cannot_reuse_previous_row(tmp_path, monkeypatch, failure):
    out = tmp_path / "out"
    out.mkdir()
    row = out / driver.ROW_FILE
    row.write_text('{"status": "ok"}')

    def run(cmd, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(driver.subprocess, "run", run)
    result = driver._spawn_worker(tmp_path / "src", out, "lid_transport", 1, {}, 10)
    assert result == ("timeout" if failure == "timeout" else 1)
    assert not row.exists()


def test_fresh_result_survives_teardown_failure(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    row = out / driver.ROW_FILE
    row.write_text('{"status": "fail"}')

    def run(cmd, **kwargs):
        assert not row.exists()
        row.write_text('{"status": "ok"}')
        return SimpleNamespace(returncode=-11)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver._spawn_worker(tmp_path / "src", out, "lid_transport", 1, {}, 10) == -11
    assert row.read_text() == '{"status": "ok"}'


def _output(tmp_path, *, row=None):
    import json
    out_fam = tmp_path / 'out' / 'lid_transport'
    out = out_fam / 'task_0000' / 'base'
    out.mkdir(parents=True)
    for name in ['scene_ep1.json', 'diagnostics.jsonl'] + [
        f'rollout_{label}_ep1.mp4' for label in driver.VIDEO_LABELS
    ]:
        (out / name).write_text('fixture')
    valid = {'task': 'task_0000', 'family': 'lid_transport', 'status': 'ok'}
    (out / driver.ROW_FILE).write_text(json.dumps(valid if row is None else row))
    return out_fam, out


@pytest.mark.parametrize('row', [[], {}, {'status': 'unknown'}, {'status': []},
    {'task': 'task_9999', 'family': 'lid_transport', 'status': 'ok'},
    {'task': 'task_0000', 'family': 'jar_transport', 'status': 'ok'}])
def test_invalid_worker_row_cannot_pass_offline_qc(tmp_path, monkeypatch, row):
    from maniguard.data.bench_builder import validate_base
    out_fam, _ = _output(tmp_path, row=row)
    monkeypatch.setattr(validate_base, 'validate_base_task', lambda *a, **kw: {'status': 'ok'})
    result = driver._row_for_task('task_0000', out_fam, 'lid_transport', 1)
    assert result['status'] == 'fail' and result.get('error')


def test_truncated_worker_row_is_a_task_failure_not_driver_crash(tmp_path):
    out_fam, out = _output(tmp_path)
    (out / driver.ROW_FILE).write_text('{"status":')
    result = driver._row_for_task('task_0000', out_fam, 'lid_transport', 1)
    assert result['status'] == 'fail' and 'JSON' in result['error']


def test_validation_exception_is_reported_in_manifest_row(tmp_path, monkeypatch):
    from maniguard.data.bench_builder import validate_base
    out_fam, _ = _output(tmp_path)

    def invalid(*a, **kw):
        raise ValueError('broken snapshot')

    monkeypatch.setattr(validate_base, 'validate_base_task', invalid)
    result = driver._row_for_task('task_0000', out_fam, 'lid_transport', 1)
    assert result['status'] == 'fail' and 'broken snapshot' in result['error']


@pytest.mark.parametrize('mode', ['fresh_teardown', 'failed_skip', 'corrupt_skip', 'launch_error', 'stale_crash', 'valid_skip'])
def test_driver_manifest_accounts_for_retry_and_worker_exit(tmp_path, monkeypatch, mode):
    import json
    from maniguard.data.bench_builder import validate_base
    out_fam, out = _output(tmp_path)
    src_root = tmp_path / 'src'
    src = src_root / 'lid_transport' / 'task_0000' / 'base'
    src.mkdir(parents=True)
    (src / 'scene_ep1.json').write_text('{}')
    if mode == 'failed_skip':
        (out / driver.ROW_FILE).write_text(json.dumps({'task': 'task_0000', 'family': 'lid_transport', 'status': 'fail'}))
    if mode == 'corrupt_skip':
        (out / driver.ROW_FILE).write_text('{')
    monkeypatch.setattr(validate_base, 'validate_base_task', lambda *a, **kw: {'status': 'ok'})
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        assert not (out / driver.ROW_FILE).exists()
        if mode == 'launch_error':
            raise OSError('cannot launch worker')
        if mode == 'stale_crash':
            return SimpleNamespace(returncode=1)
        (out / driver.ROW_FILE).write_text(json.dumps({'task': 'task_0000', 'family': 'lid_transport', 'status': 'ok'}))
        return SimpleNamespace(returncode=-11 if mode == 'fresh_teardown' else 0)

    monkeypatch.setattr(driver.subprocess, 'run', run)
    args = SimpleNamespace(src_root=str(src_root), out_root=str(tmp_path / 'out'),
        family='lid_transport', tasks=None, src_subdir='base', episode=1,
        skip_existing=mode in {'failed_skip', 'corrupt_skip', 'valid_skip'}, jobs=1, timeout=10)
    code = driver._driver(args)
    rows = [json.loads(s) for s in (out_fam / 'base_manifest.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    if mode in {'launch_error', 'stale_crash'}:
        assert code == 1 and row['status'] == 'fail'
        if mode == 'launch_error':
            assert 'cannot launch worker' in row['error']
        else:
            assert row['worker_exit'] == 1
    else:
        assert code == 0 and row['status'] == 'ok'
        if mode != 'valid_skip':
            assert row['worker_exit'] == (-11 if mode == 'fresh_teardown' else 0)
    assert len(calls) == (0 if mode == 'valid_skip' else 1)


@pytest.mark.parametrize('qc', [{}, {'status': []}, None])
def test_validator_without_valid_verdict_cannot_accept_task(tmp_path, monkeypatch, qc):
    from maniguard.data.bench_builder import validate_base
    out_fam, _ = _output(tmp_path)
    monkeypatch.setattr(validate_base, 'validate_base_task', lambda *a, **kw: qc)
    assert driver._row_for_task('task_0000', out_fam, 'lid_transport', 1)['status'] == 'fail'
