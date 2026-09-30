"""Regression coverage for per-task demonstration scaling."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_patch():
    path = Path(__file__).resolve().parents[2] / 'maniguard/openpi_sft/_episode_subset_patch.py'
    spec = importlib.util.spec_from_file_location('episode_subset_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(n=120):
    return {i: {'episode_index': i, 'tasks': [f'task_{i // 40}'], 'length': 100} for i in range(n)}


@pytest.mark.parametrize('fraction,keep', [(0.2, 8), (0.5, 20), (0.8, 32)])
def test_subset_preserves_every_task(fraction, keep):
    patch = load_patch()
    selected = patch.select_episode_subset(metadata(), fraction)
    assert selected == [i for start in [0, 40, 80] for i in range(start, start + keep)]


def test_incomplete_task_block_rejected():
    with pytest.raises(ValueError, match='multiple'):
        load_patch().select_episode_subset(metadata(79), 0.5)


def test_mixed_task_block_rejected():
    data = metadata()
    data[15]['tasks'] = ['different_task']
    with pytest.raises(ValueError, match='mixes'):
        load_patch().select_episode_subset(data, 0.2)


def test_filtered_queries_use_subset_position(monkeypatch):
    patch = load_patch()
    leaf = types.ModuleType('lerobot.common.datasets.lerobot_dataset')
    class Dataset:
        def _get_query_indices(self, idx, ep_idx):
            return idx, ep_idx
    leaf.LeRobotDataset = Dataset
    for name in ['lerobot', 'lerobot.common', 'lerobot.common.datasets']:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    patch._fix_lerobot_filtered_query_indices()
    first = Dataset._get_query_indices
    patch._fix_lerobot_filtered_query_indices()
    assert Dataset._get_query_indices is first
    data = Dataset()
    data.episodes = [0, 1, 40, 41, 80, 81]
    assert data._get_query_indices(123, 80) == (123, 4)
    data.episodes = None
    assert data._get_query_indices(123, 80) == (123, 80)


@pytest.mark.parametrize('fraction', [0., -0.2, 1., 1.2, float('nan'), float('inf'), True, '0.2'])
def test_invalid_fraction_cannot_silently_select_wrong_scale(fraction):
    with pytest.raises(ValueError, match='fraction'):
        load_patch().select_episode_subset(metadata(), fraction)


@pytest.mark.parametrize('problem', ['empty', 'wrong_episode_id', 'shifted_keys', 'empty_tasks', 'zero_length'])
def test_invalid_metadata_cannot_change_task_block_boundaries(problem):
    data = metadata()
    if problem == 'empty':
        data = {}
    elif problem == 'wrong_episode_id':
        data[40]['episode_index'] = 41
    elif problem == 'shifted_keys':
        data = {i+1: r for i,r in data.items()}
    elif problem == 'empty_tasks':
        data[1]['tasks'] = []
    else:
        data[1]['length'] = 0
    with pytest.raises(ValueError):
        load_patch().select_episode_subset(data, .2)


def test_registration_and_full_dataset_do_not_require_legacy_lerobot(monkeypatch):
    patch = load_patch()
    sentinel = object()
    dl = types.ModuleType('openpi.training.data_loader')
    dl.create_torch_dataset = lambda *args: sentinel
    root = types.ModuleType('openpi')
    training = types.ModuleType('openpi.training')
    root.training = training
    training.data_loader = dl
    for name, mod in [('openpi', root), ('openpi.training', training), ('openpi.training.data_loader', dl)]:
        monkeypatch.setitem(sys.modules, name, mod)

    def forbidden():
        raise AssertionError('index patch is only needed for subset loading')

    monkeypatch.setattr(patch, '_fix_lerobot_filtered_query_indices', forbidden)
    patch.apply()
    assert dl.create_torch_dataset(types.SimpleNamespace(episode_fraction=None), 4, None) is sentinel
    installed = dl.create_torch_dataset
    patch.apply()
    assert dl.create_torch_dataset is installed
