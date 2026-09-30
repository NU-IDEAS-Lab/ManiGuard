"""Exercise the real capture -> HDF5 -> exporter boundary without simulation."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = REPO_ROOT / "maniguard/data/real_teleop/real_teleop_to_hdf5.py"


def _load_exporter():
    # Import the actual exporter without starting OmniGibson package hooks.
    path = REPO_ROOT / "maniguard/data/lerobot/multitask_lerobot_export.py"
    spec = importlib.util.spec_from_file_location("real_teleop_test_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def captures(tmp_path):
    input_dir = tmp_path / "captures"
    input_dir.mkdir()
    positions = np.array([
        [0.20, 0.10, 0.40], [0.21, 0.08, 0.43],
        [0.23, 0.10, 0.42], [0.23, 0.09, 0.41],
    ])
    # Cross pi in absolute orientation, and flip one quaternion's sign.
    angles = np.array([3.0, 3.2, 3.4, 3.5])
    quats = np.column_stack([
        np.cos(angles / 2), np.zeros((4, 2)), np.sin(angles / 2),
    ])
    quats[1] *= -1
    images = {}
    for cam, bgr in [("cam0", (10, 30, 200)), ("cam1", (200, 30, 10))]:
        jpegs = np.empty(4, dtype=object)
        for t in range(4):
            pixels = np.full((12, 16, 3), bgr, dtype=np.uint8)
            pixels[:, :, 1] += 10 * t
            ok, encoded = cv2.imencode(".jpg", pixels)
            assert ok
            jpegs[t] = encoded.tobytes()
        images[f"observation/image/{cam}"] = jpegs
    np.savez(
        input_dir / "capture.npz",
        **images,
        **{
            "observation/cartesian_position": np.column_stack([positions, quats]),
            "observation/gripper_position": np.array([0.0, 0.25, 0.5, 1.0]),
        },
    )
    return input_dir


def _run_converter(input_dir, output_dir, *extra):
    return subprocess.run(
        [sys.executable, str(CONVERTER), "--input-dir", str(input_dir),
         "--output-dir", str(output_dir), "--img-size", "8", *extra],
        capture_output=True, text=True, check=False,
    )


def test_default_filenames_are_preserved_and_stamp_is_exportable(captures, tmp_path):
    output_dir = tmp_path / "hdf5"
    result = _run_converter(captures, output_dir)
    assert result.returncode == 0, result.stderr
    assert [p.name for p in output_dir.iterdir()] == ["traj_0.hdf5"]
    assert _load_exporter()._read_stamp(output_dir / "traj_0.hdf5") == ("eef", 2)


def test_task_cli_output_loads_with_matching_actions_and_camera_order(captures, tmp_path):
    output_dir = tmp_path / "hdf5"
    result = _run_converter(captures, output_dir, "--task-id", "task_0042")
    assert result.returncode == 0, result.stderr
    exporter = _load_exporter()
    path = output_dir / "task_0042_traj_000.hdf5"
    assert exporter._discover_episodes(output_dir, "base") == [("task_0042", path)]
    controller, n_cams = exporter._read_stamp(path)
    assert (controller, n_cams) == ("eef", 2)

    diag_path = tmp_path / "diagnostics/task_0042/diagnostics.jsonl"
    diag_path.parent.mkdir(parents=True)
    diag_path.write_text('{"prompt": "Put the cup on the tray."}\n')
    assert exporter._load_prompt(diag_path) == "Put the cup on the tray."

    with h5py.File(path) as f:
        demo = f["data/demo_0"]
        assert demo["obs/image"].shape == (4, 8, 8, 3)
        assert demo["obs/wrist_image"].shape == (4, 8, 8, 3)
        assert demo["obs/state"].shape == (4, 8)
        assert demo["action"].shape == (3, 7)
        original_actions = demo["action"][:]
        np.testing.assert_allclose(
            demo["obs/state"][:, 6:8],
            [[0.04, 0.04], [0.03, 0.03], [0.02, 0.02], [0.0, 0.0]],
            atol=1e-7,
        )

    frames, n_actions = exporter._load_episode_from_hdf5(path, controller, n_cams)
    assert n_actions == len(frames) == 3
    loaded_actions = np.stack([frame["actions"].numpy() for frame in frames])
    expected = np.array([
        [0.01, -0.02, 0.03, 0, 0, 0.2, 1],
        [0.02, 0.02, -0.01, 0, 0, 0.2, -1],
        [0, -0.01, -0.01, 0, 0, 0.1, -1],
    ])
    np.testing.assert_allclose(original_actions, expected, atol=2e-6)
    np.testing.assert_allclose(loaded_actions, original_actions, atol=2e-6)
    for t, frame in enumerate(frames):
        assert list(frame) == ["image", "wrist_image", "state", "actions"]
        np.testing.assert_allclose(frame["image"][0, 0].numpy(), [200, 30 + 10 * t, 10], atol=2)
        np.testing.assert_allclose(frame["wrist_image"][0, 0].numpy(), [10, 30 + 10 * t, 200], atol=2)


@pytest.mark.parametrize("task_id", ["cup", "42", "../task_0042"])
def test_task_id_must_match_exporter_discovery(captures, tmp_path, task_id):
    output_dir = tmp_path / "hdf5"
    result = _run_converter(captures, output_dir, "--task-id", task_id)
    assert result.returncode != 0
    assert "task_<digits>" in result.stderr
    assert not output_dir.exists()


@pytest.mark.parametrize('problem', ['short_camera', 'extra_camera', 'one_observation',
    'pose_shape', 'gripper_shape', 'nan_pose', 'inf_gripper'])
def test_invalid_capture_is_rejected_before_writing_hdf5(captures, tmp_path, problem):
    path = captures / 'capture.npz'
    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    if problem == 'short_camera':
        data['observation/image/cam0'] = data['observation/image/cam0'][:2]
    elif problem == 'extra_camera':
        data['observation/image/cam1'] = np.append(data['observation/image/cam1'], data['observation/image/cam1'][-1:])
    elif problem == 'one_observation':
        data = {key: value[:1] for key, value in data.items()}
    elif problem == 'pose_shape':
        data['observation/cartesian_position'] = data['observation/cartesian_position'][:, :6]
    elif problem == 'gripper_shape':
        data['observation/gripper_position'] = data['observation/gripper_position'][:, None]
    elif problem == 'nan_pose':
        data['observation/cartesian_position'][1, 0] = np.nan
    else:
        data['observation/gripper_position'][1] = np.inf
    np.savez(path, **data)
    output_dir = tmp_path / 'hdf5'
    result = _run_converter(captures, output_dir, '--task-id', 'task_0000')
    assert result.returncode != 0
    assert 'capture.npz' in result.stderr
    assert not output_dir.exists()


@pytest.mark.parametrize('size', ['0', '-1'])
def test_invalid_image_size_rejected_before_conversion(captures, tmp_path, size):
    result = _run_converter(captures, tmp_path / 'hdf5', '--img-size', size)
    assert result.returncode != 0 and '--img-size must be positive' in result.stderr
