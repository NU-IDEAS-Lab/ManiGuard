"""Rendered image dimensions must survive teleop export without a resize."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def exporter():
    path = Path(__file__).resolve().parents[2] / "maniguard/data/lerobot/multitask_lerobot_export.py"
    spec = importlib.util.spec_from_file_location("teleop_image_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rendered(path, size, *, controller="joint", n_cams=3):
    keys = ["image_left", "image_right", "wrist_image"] if n_cams == 3 else ["image", "wrist_image"]
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs.update(controller_mode=controller, n_cams=n_cams)
        demo = data.create_group("demo_0")
        demo.create_dataset("obs/state", data=np.zeros((3, 8), dtype=np.float32))
        demo.create_dataset("action", data=np.zeros((2, 8), dtype=np.float32))
        for key in keys:
            demo.create_dataset(f"obs/{key}", shape=(3, size, size, 3), dtype="u1")
    return path


@pytest.mark.parametrize("size,n_cams", [(640, 3), (256, 2)])
def test_feature_shapes_match_actual_frames_without_resolution_flag(exporter, tmp_path, size, n_cams):
    path = rendered(tmp_path / "episode.hdf5", size, n_cams=n_cams)
    controller, cams, features = exporter._infer_schema([path])
    frames, count = exporter._load_episode_from_hdf5(path, controller, cams)
    assert count == 2
    for key in ("image_left", "image_right", "wrist_image") if n_cams == 3 else ("image", "wrist_image"):
        assert features[key]["shape"] == (size, size, 3)
        assert tuple(frames[0][key].shape) == features[key]["shape"]


def test_resolution_override_cannot_mislabel_high_resolution_frames(exporter, tmp_path):
    path = rendered(tmp_path / "episode.hdf5", 640)
    with pytest.raises(ValueError, match="resolution"):
        exporter._infer_schema([path], resolution=256)
    assert exporter._infer_schema([path], resolution=640)[2]["wrist_image"]["shape"] == (640, 640, 3)


@pytest.mark.parametrize("change", ["resolution", "controller", "camera_count", "wrist_shape"])
def test_mixed_input_is_rejected_before_dataset_creation(exporter, tmp_path, change):
    first = rendered(tmp_path / "first.hdf5", 640)
    second = rendered(tmp_path / "second.hdf5", 256 if change == "resolution" else 640,
                      controller="eef" if change == "controller" else "joint",
                      n_cams=2 if change == "camera_count" else 3)
    if change == "wrist_shape":
        with h5py.File(second, "a") as f:
            del f["data/demo_0/obs/wrist_image"]
            f.create_dataset("data/demo_0/obs/wrist_image", shape=(3, 256, 256, 3), dtype="u1")
    with pytest.raises(ValueError, match="second.hdf5"):
        exporter._infer_schema([first, second])
