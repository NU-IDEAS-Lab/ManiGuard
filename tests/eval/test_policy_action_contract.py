"""Reject malformed policy responses before they can stall or drive physics."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from maniguard.eval.eval_config import config_from_cli
from test_rollout_diagnostics import SOURCE, run_rollout


def call_query(actions, action_dim=8):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'query_policy')
    ns = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['query_policy'](SimpleNamespace(act=lambda _: actions), {}, 'random',
                              SimpleNamespace(action_dim=action_dim))


@pytest.mark.parametrize('actions', [np.empty((0, 8)), np.empty((2, 0)), np.zeros((2, 7)),
                                     np.zeros((2, 9)), np.zeros((1, 2, 8)), np.array(0.)])
def test_malformed_policy_response_is_rejected(actions):
    with pytest.raises(ValueError, match='action'):
        call_query(actions)


@pytest.mark.parametrize('shape', [(8,), (1, 8), (16, 8)])
def test_valid_single_and_chunked_actions_preserve_values(shape):
    actions = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    out = call_query(actions)
    np.testing.assert_array_equal(out, actions.reshape(-1, 8))


def test_empty_response_stops_after_first_request_without_physics():
    result, calls, _ = run_rollout(actions=np.empty((0, 8)))
    assert result['status'] == 'crashed' and result['success'] is None
    assert calls['queries'] == 1 and calls['actions'] == []
    assert result['rollout_diagnostics']['phase'] == 'query_policy'
    assert 'action' in result['rollout_diagnostics']['exception']['message'].lower()


@pytest.mark.parametrize('dim', [1, 7, 9])
def test_wrong_action_width_cannot_broadcast_or_truncate(dim):
    result, calls, _ = run_rollout(actions=np.zeros((2, dim)))
    assert result['status'] == 'crashed' and calls['actions'] == []
    assert result['counted_violation'] is None


@pytest.mark.parametrize('field', ['execute_horizon', 'max_steps', 'action_dim', 'success_hold_steps'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_execution_config_rejected_before_simulator(tmp_path, monkeypatch, field, value):
    import yaml
    config = tmp_path / 'eval.yaml'
    config.write_text(yaml.safe_dump({field: value}))
    monkeypatch.setattr(sys, 'argv', ['eval', '--config', str(config)])
    with pytest.raises(ValueError, match=field):
        config_from_cli()


def test_all_shipped_eval_configs_keep_valid_execution_contract(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    for path in (root / 'configs/eval').rglob('*.yaml'):
        monkeypatch.setattr(sys, 'argv', ['eval', '--config', str(path)])
        cfg = config_from_cli()
        assert cfg.execute_horizon > 0 and cfg.action_dim > 0


@pytest.mark.parametrize('dim', [1, 7, 9])
def test_wrong_conversion_output_cannot_broadcast_into_controller(dim):
    result, calls, _ = run_rollout(convert=True, conversion_value=0., conversion_dim=dim)
    assert result['status'] == 'crashed' and calls['actions'] == []
    assert result['rollout_diagnostics']['phase'] == 'action_conversion'
    assert 'shape' in result['rollout_diagnostics']['exception']['message']


def test_short_chunk_still_requeries_until_horizon():
    result, calls, _ = run_rollout(actions=np.zeros((1, 8)))
    assert result['status'] == 'completed' and result['steps'] == 2
    assert calls['queries'] == 2 and len(calls['actions']) == 2
