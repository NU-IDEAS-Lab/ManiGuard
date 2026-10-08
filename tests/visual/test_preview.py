from types import SimpleNamespace

import numpy as np


def test_preview_preserves_frozen_clock_and_keeps_all_five_views(tmp_path, monkeypatch):
    from maniguard.visual import preview
    from maniguard.visual.profile import load_profile

    calls = []
    profile = load_profile({"resolution": [16, 16]})

    class Capture:
        def __init__(self, *args):
            self.profile = profile
            self.render_report = {}
            self.sim = SimpleNamespace(get_sim_step_dt=lambda: 1 / 30)

        def capture(self, i):
            calls.append(i)
            return {
                "camera_frames": {k: np.zeros((16, 16, 3), np.uint8) for k in profile["camera_keys"]},
                "observation_id": i,
                "sim_time_s": 0.5,
                "physics_tick": 60,
                "render_metadata": {},
                "camera_metadata": {},
            }

    monkeypatch.setattr(preview, "SourceCapture", Capture)
    result = preview.render_preview(None, tmp_path / "preview", profile, frames=2, mode="frozen")
    assert result["complete"] and calls == [0, 1]
    assert len(list((tmp_path / "preview").glob("*.mp4"))) == 5
    assert {x["sim_time_s"] for x in result["frames"]} == {0.5}


def test_benchmark_preview_cli_reads_episode_one_without_modifying_gt(tmp_path, monkeypatch):
    import sys

    from maniguard.data.bench_builder import render

    task, output = tmp_path / "base", tmp_path / "preview"
    task.mkdir()
    (task / "scene_ep1.json").write_text("{}")
    (task / "diagnostics.jsonl").write_text('{"episode":1,"prompt":"example"}\n')
    calls = []
    monkeypatch.setattr(render, "render_task", lambda scene, diag, out, **kw: calls.append((diag, out)))
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(
        sys, "argv", ["render", "--task-dir", str(task), "--output", str(output), "--source-profile", "high_fidelity"]
    )
    render.main()
    assert calls == [({"episode": 1, "prompt": "example"}, output)]


def test_playback_controller_json_restores_indexable_limits_without_float_dof_indices():
    from maniguard.data.recording.replay import controller_config_from_json

    config = {
        "gripper": {
            "dof_idx": [1, 2],
            "control_limits": {"position": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "has_limit": [True, True, False]},
            "subsume_controllers": ["other"],
            "command_input_limits": None,
        }
    }
    import torch

    defaults = {
        "gripper": {
            "dof_idx": torch.tensor([1, 2]),
            "control_limits": {
                "position": (torch.zeros(3), torch.ones(3)),
                "has_limit": torch.ones(3, dtype=torch.bool),
            },
        }
    }
    restored = controller_config_from_json(config, defaults)["gripper"]
    np.testing.assert_array_equal(restored["control_limits"]["position"][0][restored["dof_idx"]], [0.0, 0.0])
    assert restored["dof_idx"].dtype == torch.int64
    assert restored["control_limits"]["has_limit"].dtype == torch.bool
    assert isinstance(restored["control_limits"]["position"], tuple)
    assert restored["subsume_controllers"] == ["other"]
    assert restored["command_input_limits"] is None
    assert isinstance(config["gripper"]["dof_idx"], list)
