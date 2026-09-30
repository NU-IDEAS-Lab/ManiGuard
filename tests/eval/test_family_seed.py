"""Run the family shell entrypoint with the simulator process replaced by a recorder."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(os.environ.get(
    "MANIGUARD_TEST_FAMILY_SCRIPT",
    Path(__file__).resolve().parents[2] / "scripts/eval_family.sh",
))


@pytest.fixture
def sweep(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(SCRIPT, scripts / "eval_family.sh")
    configs = tmp_path / "configs/eval"
    configs.mkdir(parents=True)
    (configs / "jar_transport_joint.yaml").write_text("{}")
    bench = tmp_path / "bench"
    for condition in ["base", "env"]:
        task = bench / "jar_transport/task_0000" / condition
        task.mkdir(parents=True)
        (task / "diagnostics.jsonl").write_text(json.dumps({"pipeline": "jar_transport", "prompt": "Move jar"}))
        (task / "scene_ep1.json").write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
if sys.argv[1] == "-":
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
if sys.argv[1:3] == ["-m", "maniguard.eval.benchmark"]:
    with open(os.environ["CAPTURE_ARGS"], "a") as out:
        out.write(json.dumps(sys.argv[3:]) + "\\n")
    sys.exit(0)
if sys.argv[1] == "tools/eval_summary.py":
    sys.exit(0)
raise SystemExit("Unexpected command")
''')
    python.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    capture = tmp_path / "args.jsonl"

    def run(seed, repeat=1):
        env = dict(os.environ)
        env.pop("SEED", None)
        env.update(PYTHON_CMD=str(python), BENCH_ROOT=str(bench), LEVELS="base env",
                   REPEAT=str(repeat), FORCE="0", CAPTURE_ARGS=str(capture),
                   PATH=str(bin_dir) + os.pathsep + env["PATH"])
        if seed is not None:
            env["SEED"] = str(seed)
        capture.write_text("")
        result = subprocess.run(
            ["bash", str(scripts / "eval_family.sh"), "jar_transport", f"jar/seed{seed}"],
            env=env, text=True, capture_output=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return [json.loads(line) for line in capture.read_text().splitlines()]
    return run


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seed_reaches_every_scenario(sweep, seed):
    calls = sweep(seed)
    assert len(calls) == 2
    for args in calls:
        assert "--seed" in args
        assert args[args.index("--seed") + 1] == str(seed)
        assert args[args.index("--output-dir") + 1].startswith(f"outputs/eval_logs/jar/seed{seed}")
    assert {args[args.index("--scenes") + 1] for args in calls} == {"task_0000/base", "task_0000/env"}


def test_omitting_seed_preserves_config_default(sweep):
    assert all("--seed" not in args for args in sweep(None))


def test_repeat_does_not_silently_change_explicit_seed(sweep):
    calls = sweep(2, repeat=2)
    assert len(calls) == 4
    for args in calls:
        assert "--seed" in args
        assert args[args.index("--seed") + 1] == "2"
