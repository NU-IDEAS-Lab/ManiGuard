"""Exercise the ray guard against the pinned upstream implementation and a query recorder."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


class QueryRecorder:
    def __init__(self):
        self.calls = []
        self.hits = [
            SimpleNamespace(rigid_body='/far', collision='/far/mesh', distance=3., position=[3., 0., 0.], normal=[1., 0., 0.]),
            SimpleNamespace(rigid_body='/near', collision='/near/mesh', distance=1., position=[1., 0., 0.], normal=[1., 0., 0.]),
        ]

    def raycast_closest(self, **kwargs):
        self.calls.append(('closest', kwargs))
        hits = [h for h in self.hits if h.distance <= kwargs['distance']]
        if not hits:
            return {'hit': False}
        hit = min(hits, key=lambda h: h.distance)
        return {'hit': True, 'rigidBody': hit.rigid_body, 'collision': hit.collision,
                'position': hit.position, 'normal': hit.normal, 'distance': hit.distance}

    def raycast_all(self, **kwargs):
        self.calls.append(('all', {k: v for k, v in kwargs.items() if k != 'reportFn'}))
        for hit in self.hits:
            if hit.distance <= kwargs['distance'] and not kwargs['reportFn'](hit):
                break


@pytest.fixture
def runtime(monkeypatch):
    source = ROOT/'behavior-1k/OmniGibson/omnigibson/utils/sampling_utils.py'
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'raytest', 'raytest_batch'}]
    query = QueryRecorder()
    sampling = ModuleType('omnigibson.utils.sampling_utils')
    sampling.__dict__.update(th=torch, og=SimpleNamespace(sim=SimpleNamespace(psqi=query)))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), sampling.__dict__)  # noqa: S102
    utils = ModuleType('omnigibson.utils')
    utils.sampling_utils = sampling
    monkeypatch.setitem(sys.modules, 'omnigibson.utils', utils)
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.sampling_utils', sampling)
    spec = importlib.util.spec_from_file_location('ray_patch_under_test', ROOT/'maniguard/_omnigibson_patches.py')
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)
    original = sampling.raytest
    early_batch_alias = sampling.raytest_batch
    patches._patch_sampling_utils()
    return SimpleNamespace(su=sampling, query=query, original=original, batch=early_batch_alias, patches=patches)


@pytest.mark.parametrize('closest', [True, False])
@pytest.mark.parametrize('point', [[0., 0., 0.], [8.071981430053711, 10.578189849853516, .8856865167617798]])
def test_coincident_endpoints_never_reach_physx(runtime, closest, point):
    callback_calls = []
    result = runtime.su.raytest(point, point, only_closest=closest,
                                callback=lambda hit: callback_calls.append(hit) or True)
    assert result == ({'hit': False} if closest else [])
    assert runtime.query.calls == []
    assert callback_calls == []


@pytest.mark.parametrize('which', ['start', 'end'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_endpoints_raise_before_query(runtime, which, value):
    start, end = [0., 0., 0.], [1., 0., 0.]
    (start if which == 'start' else end)[0] = value
    with pytest.raises(ValueError, match='finite'):
        runtime.su.raytest(start, end)
    assert runtime.query.calls == []


@pytest.mark.parametrize('closest', [True, False])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_short_nonzero_rays_are_queried(runtime, closest, dtype):
    runtime.su.raytest(torch.zeros(3, dtype=dtype), torch.tensor([1e-8, 0., 0.], dtype=dtype), only_closest=closest)
    assert len(runtime.query.calls) == 1
    kwargs = runtime.query.calls[0][1]
    assert kwargs['distance'] > 0
    assert kwargs['dir'] == [1., 0., 0.]


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


@pytest.mark.parametrize('kwargs', [
    {}, {'only_closest': False}, {'ignore_bodies': ['/near']},
    {'ignore_collisions': ['/far/mesh']}, {'only_closest': False, 'ignore_bodies': ['/near']},
])
def test_normal_query_and_filter_semantics_match_upstream(runtime, kwargs):
    endpoints = ([0., 0., 0.], [4., 0., 0.])
    expected = runtime.original(*endpoints, **kwargs)
    calls = list(runtime.query.calls)
    runtime.query.calls.clear()
    actual = runtime.su.raytest(*endpoints, **kwargs)
    assert plain(actual) == plain(expected)
    assert runtime.query.calls == calls


def test_callback_can_stop_all_hits_as_before(runtime):
    calls = []
    result = runtime.su.raytest([0., 0., 0.], [4., 0., 0.], only_closest=False,
                               callback=lambda hit: calls.append(hit.rigid_body) or False)
    assert calls == ['/far']
    assert [hit['rigidBody'] for hit in result] == ['/far']


def test_existing_batch_alias_keeps_result_slots_and_uses_guard(runtime):
    result = runtime.batch([[8., 10., .9], [0., 0., 0.]], [[8., 10., .9], [1., 0., 0.]])
    assert len(result) == 2 and result[0] == {'hit': False}
    assert result[1]['hit'] and result[1]['rigidBody'] == '/near'
    assert len(runtime.query.calls) == 1


def test_patch_installation_is_idempotent(runtime):
    installed = runtime.su.raytest
    runtime.patches._patch_sampling_utils()
    assert runtime.su.raytest is installed
