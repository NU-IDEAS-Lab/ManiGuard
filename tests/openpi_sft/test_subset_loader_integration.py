"""Run in the pinned openpi environment; no weights, network, GPU or training."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_real_subset_action_windows_and_norm_stats(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('HF_DATASETS_OFFLINE', '1')
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    pytest.importorskip('lerobot.common.datasets.lerobot_dataset', reason='requires the pinned OpenPI/LeRobot v2.1 environment')
    dl = pytest.importorskip('openpi.training.data_loader')
    import datasets
    import lerobot.common.datasets.lerobot_dataset as lrd
    monkeypatch.setattr(lrd, 'HF_LEROBOT_HOME', tmp_path / 'lerobot')
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', tmp_path / 'dataset_cache')
    original_query = lrd.LeRobotDataset._get_query_indices

    import maniguard.openpi_sft
    from maniguard.openpi_sft import _episode_subset_patch as patch
    from maniguard.openpi_sft.data_configs import SubsetDataConfig
    from openpi.training.config import DataConfig

    installed = dl.create_torch_dataset
    patch.apply()
    assert dl.create_torch_dataset is installed
    assert getattr(installed, '_maniguard_subset_patch', False)

    repo = 'local/subset_fixture'
    data = lrd.LeRobotDataset.create(repo_id=repo, fps=10, root=lrd.HF_LEROBOT_HOME / repo,
        robot_type='test', use_videos=False,
        features={k: {'dtype': 'float32', 'shape': (8,), 'names': [str(i) for i in range(8)]}
                  for k in ['state', 'actions']})
    lengths = [3 + ep % 2 for ep in range(80)]
    for ep, length in enumerate(lengths):
        for frame in range(length):
            value = np.full(8, ep * 100 + frame, np.float32)
            data.add_frame({'state': value, 'actions': value.copy(), 'task': f'task {ep // 40}'})
        data.save_episode()

    # Both launchers use this real openpi normalization script and the same loader module.
    norm_path = Path(dl.__file__).resolve().parents[3] / 'scripts/compute_norm_stats.py'
    spec = importlib.util.spec_from_file_location('real_norm_stats', norm_path)
    norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm)

    for fraction, keep in [(.2, 8), (.5, 20), (.8, 32)]:
        cfg = SubsetDataConfig(repo_id=repo, episode_fraction=fraction, prompt_from_task=True)
        train = dl.create_torch_dataset(cfg, 4, None)
        chosen = list(range(keep)) + list(range(40, 40 + keep))
        raw = train._dataset
        assert raw.episodes == chosen
        assert len(train) == sum(lengths[ep] for ep in chosen)
        if fraction == .2 and not getattr(original_query, '_maniguard_subset_patch', False):
            with pytest.raises(IndexError):
                original_query(raw, sum(lengths[:keep]), 40)
        expected_states = []
        local = 0
        for ep in chosen:
            for frame in range(lengths[ep]):
                item = train[local]
                expected = [ep * 100 + min(frame + offset, lengths[ep]-1) for offset in range(4)]
                np.testing.assert_array_equal(item['actions'][:, 0], expected)
                np.testing.assert_array_equal(item['actions_is_pad'],
                    [frame + offset >= lengths[ep] for offset in range(4)])
                assert item['prompt'] == f'task {ep // 40}'
                expected_states.append(ep * 100 + frame)
                local += 1
        loader, batches = norm.create_torch_dataloader(cfg, 4, 4, None, 0)
        observed = np.concatenate([np.asarray(batch['state'])[:, 0] for batch in loader])
        assert batches * 4 == len(train)
        np.testing.assert_array_equal(observed, expected_states)

    full = dl.create_torch_dataset(DataConfig(repo_id=repo, prompt_from_task=True), 4, None)
    assert full._dataset.episodes is None and len(full) == sum(lengths)
    assert int(full[sum(lengths[:40])]['episode_index']) == 40


@pytest.mark.parametrize('wrapper', ['train.py', 'compute_norm_stats.py'])
def test_launchers_install_subset_hook_before_delegating(wrapper, tmp_path):
    import os
    import subprocess
    import sys
    pytest.importorskip('lerobot.common.datasets.lerobot_dataset', reason='requires the pinned OpenPI/LeRobot v2.1 environment')
    dl = pytest.importorskip('openpi.training.data_loader')
    root = Path(__file__).resolve().parents[2]
    path = root / 'tools/openpi_sft' / wrapper
    code = '''
import sys,runpy,pathlib
path=pathlib.Path(sys.argv[1])
sys.argv=[str(path)]
def delegated(script,run_name):
    from openpi.training import data_loader
    assert getattr(data_loader.create_torch_dataset, '_maniguard_subset_patch', False)
    assert pathlib.Path(script).name == path.name
    assert run_name == '__main__'
    print('SUBSET_READY')
runpy.run_path=delegated
exec(compile(path.read_text(),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
'''
    result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=root,
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'OPENPI_ROOT': str(Path(dl.__file__).resolve().parents[3]),
             'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'JAX_PLATFORMS': 'cpu',
             'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    assert 'SUBSET_READY' in result.stdout
