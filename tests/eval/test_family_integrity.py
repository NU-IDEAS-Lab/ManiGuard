"""Run the family launcher and real summarizer with only simulation replaced."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def campaign(tmp_path):
    for path in ["scripts/eval_family.sh", "tools/eval_summary.py"]:
        dst = tmp_path / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dst)
    config = tmp_path / "configs/eval/jar_transport_joint.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("seed: 7\n")
    for condition in ["base", "env"]:
        p = tmp_path / "bench/jar_transport/task_0000" / condition
        p.mkdir(parents=True)
        (p / "diagnostics.jsonl").write_text(json.dumps({"pipeline":"jar_transport", "prompt":"Move jar"}))
        (p / "scene_ep1.json").write_text("{}")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys, yaml
if sys.argv[1] == "-" or sys.argv[1] == "tools/eval_summary.py":
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
assert sys.argv[1:3] == ["-m", "maniguard.eval.benchmark"]
args=sys.argv[3:]
get=lambda key: args[args.index(key)+1]
out=pathlib.Path(get("--output-dir"))/get("--run-name")
out.mkdir(parents=True, exist_ok=True)
scene=get("--scenes")
assert (pathlib.Path(get("--benchmark-root"))/scene/"diagnostics.jsonl").is_file()
planned=any((p/"expected_rollouts.json").exists() for p in out.parents)
with open("calls.jsonl","a") as f:f.write(json.dumps({"scene":scene,"planned":planned})+"\\n")
mode=os.environ.get("TEST_MODE", "normal")
if scene.endswith("/env") and mode in ["crash", "empty"]:sys.exit(1 if mode=="crash" else 0)
config=yaml.safe_load(pathlib.Path(get("--config")).read_text()) or {}
seed=int(get("--seed")) if "--seed" in args else config.get("seed")
row={"scene_name":scene,"seed":seed,"status":"completed","success":True,
     "ever_contacted":True,"counted_violation":False,"ltl_monitored":True}
with (out/"results.jsonl").open("a") as f:
    f.write(json.dumps(row)+"\\n")
    if mode=="duplicate":f.write(json.dumps(row)+"\\n")
sys.exit(1 if mode=="nonzero_with_row" else 0)
''')
    python.chmod(0o755)
    sleep = bindir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)

    def run(mode="normal", repeat=1, levels="base env", family="jar_transport"):
        env = dict(os.environ)
        env.pop("SEED", None)
        env.update(PYTHON_CMD=str(python), BENCH_ROOT=str(tmp_path / "bench"), LEVELS=levels,
                   REPEAT=str(repeat), FORCE="0", TEST_MODE=mode,
                   PATH=str(bindir)+os.pathsep+env["PATH"])
        return subprocess.run(["bash", str(tmp_path / "scripts/eval_family.sh"),
                               family, "test_run"], env=env,
                              capture_output=True, text=True, timeout=20)
    return tmp_path, run


def test_plan_precedes_execution_and_preserves_config_seed(campaign):
    root, run = campaign
    result = run()
    assert result.returncode == 0, result.stdout+result.stderr
    calls = [json.loads(l) for l in (root/"calls.jsonl").read_text().splitlines()]
    assert all(c["planned"] for c in calls)
    plan = json.loads((root/"outputs/eval_logs/test_run/expected_rollouts.json").read_text())
    assert {(r["scene_name"],r["seed"],r["results_file"]) for r in plan["rollouts"]} == {
        ("task_0000/base",7,"ID/results.jsonl"), ("task_0000/env",7,"OOD/env/results.jsonl")}


@pytest.mark.parametrize("mode", ["crash", "empty", "duplicate", "nonzero_with_row"])
def test_incomplete_or_failed_campaign_exits_nonzero(campaign, mode):
    _, run = campaign
    result = run(mode)
    assert result.returncode != 0, result.stdout+result.stderr


def test_intentional_repetitions_match_the_saved_plan(campaign):
    root, run = campaign
    result = run(repeat=2)
    assert result.returncode == 0, result.stdout+result.stderr
    plan = json.loads((root/"outputs/eval_logs/test_run/expected_rollouts.json").read_text())
    assert len(plan["rollouts"]) == 4


def test_missing_requested_variant_is_not_removed_from_plan(campaign):
    root, run = campaign
    (root/"bench/jar_transport/task_0000/env/diagnostics.jsonl").unlink()
    result = run()
    assert result.returncode != 0
    assert not (root/"calls.jsonl").exists()


@pytest.mark.parametrize("repeat", [0, -1, "abc"])
def test_invalid_repeat_is_rejected_before_execution(campaign, repeat):
    root, run = campaign
    assert run(repeat=repeat).returncode != 0
    assert not (root/"calls.jsonl").exists()


def test_lid_uses_existing_food_config_and_requested_benchmark_root(campaign):
    root, run = campaign
    (root/"bench/jar_transport").rename(root/"bench/lid_transport")
    (root/"configs/eval/jar_transport_joint.yaml").rename(
        root/"configs/eval/lid_transport_food_joint.yaml")
    result = run(family="lid_transport")
    assert result.returncode == 0, result.stdout+result.stderr
