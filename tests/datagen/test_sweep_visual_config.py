import json

import pytest


@pytest.mark.parametrize("selection", [None, "legacy", "custom.yaml"])
def test_sweep_forwards_profile_and_budget(tmp_path, monkeypatch, selection):
    from maniguard.data.datagen import sweep

    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom.yaml").write_text("resolution: [512, 512]\n")
    monkeypatch.setattr(sweep, "OUT_ROOT", tmp_path / "outputs")
    argv = ["sweep", "--dataset", "sample", "--family", "jar", "--tasks", "task_0000"]
    if selection is not None:
        argv += ["--source-profile", selection]
    monkeypatch.setattr("sys.argv", argv)
    commands = []
    monkeypatch.setattr(sweep.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    assert sweep.main() == 0
    cmd = commands[0]
    expected = str(tmp_path / selection) if selection == "custom.yaml" else selection or "high_fidelity"
    assert cmd[cmd.index("--source-profile") + 1] == expected
    assert cmd[cmd.index("--recording-max-bytes") + 1] == str(5 * 1024**3)


def test_resume_rejects_cross_profile_and_checks_source_attempts(tmp_path):
    from maniguard.data.datagen.executor.resume import validate_visual_resume
    from maniguard.visual.profile import load_profile

    (tmp_path / "traj_000").mkdir()
    (tmp_path / "traj_000/traj.hdf5").touch()
    with pytest.raises(ValueError, match="output directory"):
        validate_visual_resume(tmp_path, load_profile())
    validate_visual_resume(tmp_path, None)
    (tmp_path / "traj_000/traj.hdf5").unlink()
    (tmp_path / "attempt_000000_g0").mkdir()
    meta = tmp_path / "attempt_000000_g0/episode.json"
    meta.write_text(json.dumps({"source_profile": load_profile()}))
    validate_visual_resume(tmp_path, load_profile())
    with pytest.raises(ValueError, match="output directory"):
        validate_visual_resume(tmp_path, load_profile({"resolution": [512, 512]}))
    with pytest.raises(ValueError, match="output directory"):
        validate_visual_resume(tmp_path, None)


def test_source_raw_reader_requires_explicit_export(tmp_path):
    from maniguard.data.datagen.reader import iter_traj_dirs

    attempt = tmp_path / "sample/jar_transport/task_0000/attempt_000000_g0"
    attempt.mkdir(parents=True)
    (attempt / "episode.json").write_text("{}")
    with pytest.raises(ValueError, match="accepted-datagen"):
        list(iter_traj_dirs("sample", root=tmp_path))
