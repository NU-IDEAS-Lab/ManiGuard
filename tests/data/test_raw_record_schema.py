"""Actual RAW writing and optional LeRobot 0.3.3 conversion using synthetic observations."""
import hashlib
import importlib.metadata
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from maniguard.data.datagen import data_format, reader, to_lerobot
from maniguard.data.datagen.primitives import record


def write_raw(path, monkeypatch, states_mode='ragged', prompt='move the cup'):
    step = {'i': 0}
    sensors = [SimpleNamespace(get_obs=lambda c=c: {'rgb': np.full((256,256,3), c*30 + step['i'], np.uint8)}) for c in range(5)]
    env = SimpleNamespace(external_sensors=dict(zip(data_format.THIRD_PERSON_CAMS.values(), sensors[:4])))
    robot = SimpleNamespace(default_arm='0', arm_control_idx={'0': np.arange(7)},
        gripper_control_idx={'0': np.array([7,8])},
        get_joint_positions=lambda: np.r_[np.arange(7) + step['i']*.1, .02, .04])
    monkeypatch.setattr(record, 'find_wrist_sensor', lambda r: sensors[4])
    monkeypatch.setattr(record, '_dump_sim_state', lambda: np.arange(
        3 + step['i'] if states_mode == 'ragged' else 3, dtype=np.float32))
    recorder = record.Recorder(record_sim_states=states_mode != 'none')
    recorder.attach(env, robot, path, prompt)
    for i in range(4):
        step['i'] = i
        recorder.record_step(np.arange(7) + 10 + i, -1. if i % 2 else 1.)
    assert recorder.finalize(True) == path
    return path


@pytest.mark.parametrize('mode', ['none', 'fixed', 'ragged'])
def test_raw_writer_matches_documented_numeric_and_optional_state_schema(tmp_path, monkeypatch, mode):
    path = write_raw(tmp_path / 'traj', monkeypatch, mode)
    with h5py.File(path / 'traj.hdf5') as h:
        assert h['state'].shape == h['actions'].shape == h['actions_commanded'].shape == (4,8)
        np.testing.assert_allclose(h['state'][:,7], .03)
        np.testing.assert_allclose(h['actions'][:,0], [.1,.2,.3,.3])
        np.testing.assert_allclose(h['actions_commanded'][:,0], [10,11,12,13])
        np.testing.assert_array_equal(h['datagen_info/gripper_action'], [1,-1,1,-1])
        assert ('states' in h) == (mode != 'none')
        assert ('states_len' in h) == (mode == 'ragged')
        if mode == 'ragged':
            np.testing.assert_array_equal(h['states_len'], [3,4,5,6])
            assert h['states'].shape == (4,6)
        assert 'datagen_info/eef_pose' not in h
        assert 'datagen_info/object_poses' not in h
    meta = json.loads((path / 'meta.json').read_text())
    assert meta['video_keys'] == list(data_format.IMAGE_KEYS)
    assert meta['n_steps'] == 4 and meta['success'] is True
    assert set(reader.load_traj(path)['videos']) == set(data_format.IMAGE_KEYS)
    for key in data_format.IMAGE_KEYS:
        assert reader.read_frames(path / f'{key}.mp4').shape == (4,256,256,3)


def test_converter_rejects_unsupported_library_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.5.1')
    with pytest.raises(RuntimeError, match='0.3.3'):
        to_lerobot.convert(str(tmp_path / 'missing'), 'clutter_pickup', out_root=str(tmp_path / 'out'), repo_id='local/test')
    assert not (tmp_path / 'out').exists()


def test_raw_to_lerobot_preserves_all_five_videos_and_joint_columns(tmp_path, monkeypatch):
    try:
        version = importlib.metadata.version('lerobot')
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != '0.3.3':
        pytest.skip('requires the LeRobot 0.3.3 conversion environment')
    import pyarrow.parquet as pq
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    try:
        to_lerobot.convert(str(tmp_path / 'missing'), 'clutter_pickup',
                           out_root=str(tmp_path / 'empty_out'), repo_id='local/empty')
    except ValueError as exc:
        assert 'No RAW trajectories' in str(exc)
    else:
        raise AssertionError('empty input must not produce an empty successful dataset')
    assert not (tmp_path / 'empty_out').exists()
    raw = tmp_path / 'raw'
    paths = [write_raw(raw / 'clutter_pickup' / f'task_{i:04d}' / 'traj_000', monkeypatch,
                       prompt=f'move object {i}') for i in range(2)]
    summary = to_lerobot.convert(str(raw), 'clutter_pickup', out_root=str(tmp_path / 'out'), repo_id='local/test')
    assert summary['episodes'] == 2 and summary['skipped'] == 0
    out = tmp_path / 'out/clutter_pickup'
    info = json.loads((out / 'meta/info.json').read_text())
    assert info['codebase_version'] == 'v2.1'
    assert info['total_episodes'] == 2 and info['total_frames'] == 8
    for ep, path in enumerate(paths):
        rows = pq.read_table(out / f'data/chunk-000/episode_{ep:06d}.parquet').to_pydict()
        with h5py.File(path / 'traj.hdf5') as h:
            for key in ['state','actions','actions_commanded']:
                np.testing.assert_array_equal(rows[key], h[key][:])
        for key in data_format.IMAGE_KEYS:
            video = out / f'videos/chunk-000/{key}/episode_{ep:06d}.mp4'
            assert hashlib.sha256(video.read_bytes()).digest() == hashlib.sha256((path / f'{key}.mp4').read_bytes()).digest()
