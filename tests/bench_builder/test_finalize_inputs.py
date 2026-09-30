"""Input validation for benchmark construction."""
import sys

import pytest

from maniguard.data.bench_builder.finalize_base import finalize_base_task
from maniguard.data.bench_builder import run_finalize_base


def test_finalize_refuses_to_overwrite_source(tmp_path):
    with pytest.raises(ValueError, match='must be different'):
        finalize_base_task(tmp_path, tmp_path, family='clutter_pickup')


def test_cli_requires_explicit_source(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_finalize_base', '--family', 'clutter_pickup'])
    with pytest.raises(SystemExit) as result:
        run_finalize_base.main()
    assert result.value.code == 2


def test_cli_rejects_same_source_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_finalize_base', '--family', 'clutter_pickup',
                                    '--src-root', str(tmp_path), '--out-root', str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        run_finalize_base.main()
    assert result.value.code == 2


@pytest.mark.parametrize('entry', ['spawn', 'worker'])
def test_worker_never_modifies_source_even_through_symlink(tmp_path, entry):
    source = tmp_path / 'source'
    source.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(source, target_is_directory=True)
    row = source / run_finalize_base.ROW_FILE
    row.write_text('original marker')
    with pytest.raises(ValueError, match='must be different'):
        if entry == 'spawn':
            run_finalize_base._spawn_worker(source, alias, 'lid_transport', 1, {}, 10)
        else:
            run_finalize_base._run_worker(source, alias, 'lid_transport', 1)
    assert row.read_text() == 'original marker'
