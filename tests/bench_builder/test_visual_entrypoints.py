from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("module", ["jar_transport_pipeline", "cabinet_pickup_pipeline"])
def test_standalone_parsers_accept_high_fidelity(module, monkeypatch):
    import importlib
    pipeline = importlib.import_module(f"maniguard.task_generation.{module}")
    monkeypatch.setattr("sys.argv", [module, "--source-profile", "high_fidelity", "--dry-run"])
    assert pipeline.parse_args().source_profile == "high_fidelity"


def test_preview_environment_preserves_control_clock(monkeypatch):
    from maniguard.data.bench_builder.render import prepare_preview_environment
    from maniguard.visual import capture

    calls = []
    monkeypatch.setattr(capture, "initialize_source_simulator", lambda *args, **kw: calls.append((args, kw)))
    config = {"env": {"action_frequency": 20, "rendering_frequency": 20, "physics_frequency": 120}}
    prepare_preview_environment(config, "high_fidelity")
    assert config["env"]["action_frequency"] == 20
    assert config["env"]["rendering_frequency"] == 60
    assert calls == [((20, 60, 120), {"clear": False})]
    assert len(config["env"]["external_sensors"]) == 4
    assert all(c["sensor_kwargs"]["image_height"] == 640 for c in config["env"]["external_sensors"])


@pytest.mark.parametrize("module,extra", [("run_finalize_base", []), ("perturb_target", []),
                                           ("perturb_location", [5]), ("perturb_env", [])])
def test_bench_workers_forward_visual_selection(tmp_path, monkeypatch, module, extra):
    import importlib
    worker = importlib.import_module(f"maniguard.data.bench_builder.{module}")
    source, output = tmp_path / "base", tmp_path / "out"
    output.mkdir()
    commands = []
    monkeypatch.setattr(worker.subprocess, "run", lambda cmd, **kw: commands.append(cmd) or SimpleNamespace(returncode=0))
    worker._spawn_worker(source, output, "jar_transport", 1, {}, 60, *extra, source_profile="high_fidelity")
    cmd = commands[0]
    assert cmd[cmd.index("--source-profile") + 1] == "high_fidelity"
