import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest
from test_episode import write_episode


def test_unicode_episode_finalizes_and_exports_under_ascii_locale(tmp_path):
    # Isaac can leave LC_CTYPE=C. Locale must not change the UTF-8 file contract.
    script = textwrap.dedent("""
        import json, runpy, sys
        from pathlib import Path
        from maniguard.data.recording.writer import EpisodeWriter
        from maniguard.data.recording.reader import EpisodeReader
        from maniguard.data.recording.export import export_episode
        fixture = runpy.run_path(sys.argv[1])
        root = Path(sys.argv[2])
        prompt = 'Close drawer ' + chr(8212) + ' safely'
        meta = fixture['metadata']()
        meta['scene'] = {'prompt': prompt}
        with EpisodeWriter(root / 'source', meta, max_output_bytes=10000000) as w:
            w.append_observation(fixture['observation'](0))
            w.append_transition(fixture['transition'](0))
            w.append_observation(fixture['observation'](1))
            w.append_event({'type': 'note', 'text': prompt})
            w.finish('complete', {'success': True})
        with EpisodeReader(root / 'source') as r:
            assert r.metadata['status'] == 'complete'
            assert r.metadata['scene']['prompt'] == prompt
            assert list(r.iter_events())[0]['text'] == prompt
        export_episode(root / 'source', root / 'derived', state_field='q', fps=20)
        exported = json.loads((root / 'derived/meta.json').read_text(encoding='utf-8'))
        assert exported['prompt'] == prompt
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).with_name("test_episode.py")), str(tmp_path)],
        env={**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_export_retains_pre_action_frame_ids_and_issued_commands(tmp_path):
    from maniguard.data.recording.export import export_episode
    from maniguard.data.recording.reader import EpisodeReader

    root = tmp_path / "source"
    write_episode(root)
    output = tmp_path / "derived"
    export_episode(root, output, cameras=["image_left"], state_field="q", fps=20)
    with h5py.File(output / "traj.hdf5") as h:
        np.testing.assert_array_equal(h["source_observation_id"][:], [0, 1, 2])
        np.testing.assert_array_equal(h["source_boundary_id"][:], [-1, -1, -1])
        np.testing.assert_array_equal(h["actions"][:, 0], [1, 2, 3])
        assert h["state"].shape == (3, 9)
    meta = json.loads((output / "meta.json").read_text())
    assert meta["resolution"] == 16 and meta["fps"] == 20
    import av

    with av.open(str(output / "image_left.mp4")) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 3 and frames[0].height == 16
    assert not (output / "wrist_image.mp4").exists()
    with EpisodeReader(root) as reader:
        assert reader.metadata["observations"] == 4


def test_export_keeps_boundary_state_and_its_identity(tmp_path):
    from test_episode import metadata, observation, transition

    from maniguard.data.datagen.reader import load_traj
    from maniguard.data.recording.export import export_episode
    from maniguard.data.recording.writer import EpisodeWriter

    root, output = tmp_path / "source", tmp_path / "derived"
    with EpisodeWriter(root, metadata(), max_output_bytes=10_000_000) as writer:
        writer.append_observation(observation(0))
        changed = observation(0)
        changed["robot"]["q"] += 10
        changed["safety"] = {"monitor_valid": False}
        boundary = writer.append_boundary(changed, {"scene": {}}, "attachment")
        writer.append_transition({**transition(0), "source_boundary_id": boundary})
        writer.append_observation(observation(1))
        writer.finish("complete", {})
    export_episode(root, output, cameras=["image_left"], state_field="q", fps=20)
    traj = load_traj(output)
    np.testing.assert_array_equal(traj["state"][0], np.arange(9) + 10)
    np.testing.assert_array_equal(traj["source_boundary_id"], [0])


def test_export_datagen_selects_only_accepted_scripted_attempts(tmp_path):
    from maniguard.data.datagen.reader import iter_traj_dirs, load_traj
    from maniguard.data.recording.export import export_datagen

    root = tmp_path / "raw/jar_transport/task_0000"
    for name, status, success in [("attempt_000000_g0", "complete", True),
                                  ("attempt_000001_g0", "complete", False),
                                  ("attempt_000002_g0", "partial", True)]:
        path = root / name
        write_episode(path)
        metadata_path = path / "episode.json"
        meta = json.loads(metadata_path.read_text())
        meta.update(status=status, result={"success": success},
                    configuration={"policy": "scripted joint trajectory", "state_mode": "joint", "action_space": "joint"})
        metadata_path.write_text(json.dumps(meta))
    out = tmp_path / "export"
    result = export_datagen(tmp_path / "raw", out, state_field="q")
    assert len(result) == 1
    trajs = list(iter_traj_dirs("export", root=tmp_path))
    assert len(trajs) == 1
    assert load_traj(trajs[0])["state"].shape == (3, 9)
    assert export_datagen(tmp_path / "raw", out, state_field="q") == result
    with pytest.raises(ValueError, match="conflict"):
        export_datagen(tmp_path / "raw", out, state_field="q", cameras=["image_left"])


def test_historical_pt_episode_remains_readable_and_exportable(tmp_path):
    from maniguard.data.recording import EpisodeReader
    from maniguard.data.recording.export import export_episode
    root = tmp_path / "source"
    write_episode(root)
    path = root / "episode.json"
    meta = json.loads(path.read_text())
    meta["source_profile"] = {"renderer": "PathTracing", "total_spp": 128}
    path.write_text(json.dumps(meta))
    with EpisodeReader(root) as reader:
        assert reader.validate(decode_images=True)["observations"] == 4
    export_episode(root, tmp_path / "derived", state_field="q", fps=20)
    assert json.loads((tmp_path / "derived/meta.json").read_text())["source_profile"] == meta["source_profile"]
