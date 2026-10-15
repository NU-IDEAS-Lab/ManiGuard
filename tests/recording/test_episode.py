import json
from io import BytesIO

import h5py
import numpy as np
import pytest
from PIL import Image


def metadata():
    return {
        "episode_id": "example",
        "camera_keys": ["image_left", "wrist_image"],
        "resolution": [16, 16],
        "rgb_encoding": {"codec": "jpeg", "quality": 95, "subsampling": "444", "color_order": "RGB"},
        "physics_hz": 120,
    }


def observation(i):
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[:, :8, 0] = 50 + 40 * i
    rgb[:, 8:, 2] = 250
    return {
        "observation_id": i,
        "sim_time_s": 0.5 + 0.05 * i,
        "physics_tick": 60 + 6 * i,
        "camera_frames": {"image_left": rgb, "wrist_image": rgb},
        "robot": {"q": np.arange(9, dtype=np.float32) + i},
        "safety": {"ap_values": np.array([i == 1, False]), "ap_valid": np.ones(2, bool)},
        "camera_metadata": {"note": "example"},
    }


def transition(i):
    return {
        "source_observation_id": i,
        "target_observation_id": i + 1,
        "raw": np.ones(7, np.float32) * i,
        "applied": np.ones(8, np.float32) * (i + 1),
        "proposal_id": 0,
        "proposal_offset": i,
    }


def write_episode(root, n=3):
    from maniguard.data.recording.writer import EpisodeWriter

    with EpisodeWriter(root, metadata(), max_output_bytes=10_000_000) as w:
        w.append_observation(observation(0))
        w.append_event({"type": "proposal", "proposal_id": 0, "actions": [[0.0] * 7] * 8})
        for i in range(n):
            w.append_transition(transition(i))
            w.append_observation(observation(i + 1))
        w.finish("complete", {"success": False})


def test_instantaneous_boundary_preserves_previous_observation_and_its_gt(tmp_path):
    from maniguard.data.recording.reader import EpisodeReader
    from maniguard.data.recording.writer import EpisodeWriter

    root = tmp_path / "episode"
    with EpisodeWriter(root, metadata(), max_output_bytes=10_000_000) as w:
        w.append_observation(observation(0))
        changed = observation(0)
        changed["robot"]["q"] += 10
        changed["safety"] = {"monitor_valid": False}
        boundary = w.append_boundary(changed, {"scene": {"x": 1}}, "scripted mutation")
        w.append_transition({**transition(0), "source_boundary_id": boundary})
        w.append_observation(observation(1))
        w.finish("complete", {})
    with EpisodeReader(root) as r:
        r.validate(decode_images=True)
        np.testing.assert_array_equal(r.read_observation(0)["robot"]["q"], np.arange(9))
        np.testing.assert_array_equal(r.read_boundary(0)["observation"]["robot"]["q"], np.arange(9) + 10)
        assert r.read_boundary(0)["checkpoint"]["scene"]["x"] == 1


def test_roundtrip_is_single_jpeg_encode_and_correct_time_alignment(tmp_path):
    from maniguard.data.recording.reader import EpisodeReader

    root = tmp_path / "episode"
    write_episode(root)
    with EpisodeReader(root) as r:
        assert r.validate(decode_images=True)["observations"] == 4
        assert r.metadata["status"] == "complete"
        for i in range(4):
            expected = BytesIO()
            Image.fromarray(observation(i)["camera_frames"]["image_left"]).save(
                expected, format="JPEG", quality=95, subsampling=0
            )
            np.testing.assert_array_equal(r.read_rgb("image_left", i), np.asarray(Image.open(expected)))
            np.testing.assert_array_equal(r.read_observation(i)["robot"]["q"], observation(i)["robot"]["q"])
        assert r.read_transition(2)["source_observation_id"] == 2
        assert next(r.iter_events())["type"] == "proposal"


def test_missing_view_or_wrong_alignment_is_rejected_without_complete(tmp_path):
    from maniguard.data.recording.writer import EpisodeWriter

    with EpisodeWriter(tmp_path / "episode", metadata(), max_output_bytes=10_000_000) as w:
        bad = observation(0)
        bad["camera_frames"].pop("wrist_image")
        with pytest.raises(ValueError, match="camera"):
            w.append_observation(bad)
        w.append_observation(observation(0))
        with pytest.raises(ValueError, match="source"):
            w.append_transition(transition(1))
    assert json.loads((tmp_path / "episode/episode.json").read_text())["status"] == "partial"


def test_truncated_transition_cannot_finalize_complete(tmp_path):
    from maniguard.data.recording.writer import EpisodeWriter

    with EpisodeWriter(tmp_path / "episode", metadata(), max_output_bytes=10_000_000) as w:
        w.append_observation(observation(0))
        w.append_transition(transition(0))
        with pytest.raises(ValueError, match="observation"):
            w.finish("complete", {})


def test_offsets_corruption_is_detected(tmp_path):
    from maniguard.data.recording.reader import EpisodeReader

    root = tmp_path / "episode"
    write_episode(root)
    with h5py.File(root / "trajectory.hdf5", "r+") as h:
        h["images/image_left/offsets"][-1] = 10**9
    with EpisodeReader(root) as r, pytest.raises(ValueError, match="offset"):
        r.validate()


def test_budget_and_existing_episode_are_never_silently_overwritten(tmp_path):
    from maniguard.data.recording.writer import EpisodeWriter

    root = tmp_path / "episode"
    with EpisodeWriter(root, metadata(), max_output_bytes=1024) as w, pytest.raises(OSError, match="budget"):
        w.append_observation(observation(0))
    with pytest.raises(FileExistsError):
        EpisodeWriter(root, metadata(), max_output_bytes=1024)


def test_snapshot_arrays_and_ragged_particles_roundtrip_without_pickle(tmp_path):
    from maniguard.data.recording.reader import EpisodeReader
    from maniguard.data.recording.writer import EpisodeWriter

    root = tmp_path / "episode"
    with EpisodeWriter(root, metadata(), max_output_bytes=10_000_000) as w:
        w.append_observation(observation(0))
        w.append_snapshot(0, {"particles": np.arange(9, dtype=np.float32).reshape(3, 3)}, {"rng": (1, 2, 3)})
        w.append_transition(transition(0))
        w.append_observation(observation(1))
        w.append_snapshot(1, {"particles": np.zeros((0, 3), np.float32)})
        w.finish("complete", {})
    with EpisodeReader(root) as r:
        state, checkpoint = r.read_snapshot(0)
        np.testing.assert_array_equal(state["particles"], np.arange(9).reshape(3, 3))
        assert checkpoint["rng"] == (1, 2, 3)
        assert r.read_snapshot(1)[0]["particles"].shape == (0, 3)


def test_interrupted_writer_never_publishes_complete(tmp_path):
    import subprocess
    import sys

    root = tmp_path / "interrupted"
    script = """import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/"tests/recording"))
from test_episode import metadata, observation
from maniguard.data.recording.writer import EpisodeWriter
w=EpisodeWriter(sys.argv[1],metadata(),10000000)
w.append_observation(observation(0))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(root)], check=True)
    assert json.loads((root / "episode.json").read_text())["status"] == "partial"


def test_truncated_image_is_detected_by_full_validation(tmp_path):
    from maniguard.data.recording.reader import EpisodeReader

    root = tmp_path / "episode"
    write_episode(root)
    with h5py.File(root / "trajectory.hdf5", "r+") as h:
        group = h["images/image_left"]
        group["bytes"][int(group["offsets"][1]) - 2 :] = 0
    with EpisodeReader(root) as reader, pytest.raises(ValueError):
        reader.validate(decode_images=True)


def test_write_exception_cannot_be_relabelled_complete(tmp_path, monkeypatch):
    from maniguard.data.recording.writer import EpisodeWriter

    root = tmp_path / "episode"
    with EpisodeWriter(root, metadata(), 10_000_000) as writer:

        def fail(*args):
            raise OSError("simulated device failure")

        monkeypatch.setattr(writer, "_append_columns", fail)
        with pytest.raises(OSError):
            writer.append_observation(observation(0))
        with pytest.raises(RuntimeError):
            writer.finish("complete", {})
    assert json.loads((root / "episode.json").read_text())["status"] == "partial"


def test_encoded_rgb_is_available_without_another_lossy_encode(tmp_path):
    from maniguard.data.recording.reader import EpisodeReader

    root = tmp_path / 'episode'
    write_episode(root)
    with EpisodeReader(root) as reader:
        blob = reader.read_rgb_bytes('image_left', 0)
        assert blob[:2] == b'\xff\xd8'
        np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(blob))), reader.read_rgb('image_left', 0))
