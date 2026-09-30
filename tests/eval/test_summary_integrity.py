"""Exercise the real summary CLI against complete and incomplete log trees."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/eval_summary.py"


def row(scene="task_0000/base", seed=0, **changes):
    return {"scene_name": scene, "seed": seed, "status": "completed",
            "success": True, "counted_violation": False,
            "ever_contacted": True, "ltl_monitored": True, **changes}


def logs(root, rows, expected=None):
    path = root / "ID/results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if expected is not None:
        (root / "expected_rollouts.json").write_text(json.dumps({
            "schema_version": 1,
            "rollouts": [{"results_file": "ID/results.jsonl", "scene_name": s, "seed": seed}
                         for s, seed in expected],
        }))
    return path


def run(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), str(root), *map(str, args)],
                          capture_output=True, text=True, timeout=15)


def test_complete_grid_keeps_per_seed_metrics(tmp_path):
    root = tmp_path / "cabinet"
    rs = [row(seed=0), row("task_0001/base", 0, success=False, counted_violation=True),
          row(seed=1, success=False, ever_contacted=False), row("task_0001/base", 1)]
    logs(root, rs, [(r["scene_name"], r["seed"]) for r in rs])
    output = tmp_path / "summary.json"
    p = run(root, "--json", output)
    assert p.returncode == 0, p.stderr
    m = json.loads(output.read_text())["ID"]["cabinet"]
    assert (m["n"], m["n_seeds"]) == (4, 2)
    assert (m["ssr"], m["safe"], m["eng"], m["safe_given_eng"]) == (50, 75, 75, 75)


@pytest.mark.parametrize("failure", ["monitor_failed", "load_failed", "crashed"])
def test_failed_row_is_not_silently_dropped(tmp_path, failure):
    root = tmp_path / "run"
    logs(root, [row(), row("task_0001/base", status=failure, success=None)],
         [("task_0000/base", 0), ("task_0001/base", 0)])
    output = tmp_path / "summary.json"
    p = run(root, "--json", output)
    assert p.returncode != 0
    assert failure in p.stderr
    assert not output.exists()


@pytest.mark.parametrize("bad", ['{"status":', '[1,2,3]', '{"status": NaN}'])
def test_bad_json_row_is_not_skipped(tmp_path, bad):
    root = tmp_path / "run"
    path = logs(root, [row()], [("task_0000/base", 0)])
    with path.open("a") as f:
        f.write(bad + "\n")
    p = run(root)
    assert p.returncode != 0
    assert "results.jsonl:2" in p.stderr


@pytest.mark.parametrize("field,value", [("success", None), ("counted_violation", None),
                                         ("ever_contacted", "false"), ("ltl_monitored", False)])
def test_invalid_metric_fields_are_not_coerced_to_booleans(tmp_path, field, value):
    root = tmp_path / "run"
    logs(root, [row(**{field:value})], [("task_0000/base", 0)])
    p = run(root)
    assert p.returncode != 0 and field in p.stderr


def test_missing_scenario_is_reported(tmp_path):
    root = tmp_path / "run"
    logs(root, [row()], [("task_0000/base", 0), ("task_0001/base", 0)])
    p = run(root)
    assert p.returncode != 0
    assert "missing" in p.stderr.lower() and "task_0001/base" in p.stderr


def test_entirely_missing_seed_is_reported(tmp_path):
    root = tmp_path / "run"
    logs(root, [row()], [("task_0000/base", 0), ("task_0000/base", 1)])
    p = run(root)
    assert p.returncode != 0 and "missing" in p.stderr.lower()


def test_missing_results_file_is_reported(tmp_path):
    root = tmp_path / "run"
    p = logs(root, [], [("task_0000/base", 0)])
    p.unlink()
    result = run(root)
    assert result.returncode != 0
    assert "missing" in result.stderr.lower()


def test_unexpected_duplicate_is_rejected(tmp_path):
    root = tmp_path / "run"
    logs(root, [row(), row()], [("task_0000/base", 0)])
    p = run(root)
    assert p.returncode != 0 and "unexpected" in p.stderr.lower()


def test_planned_repetitions_are_allowed(tmp_path):
    root = tmp_path / "run"
    logs(root, [row(), row()], [("task_0000/base", 0)] * 2)
    assert run(root).returncode == 0


def test_overlapping_roots_cannot_double_count(tmp_path):
    root = tmp_path / "run"
    logs(root, [row()], [("task_0000/base", 0)])
    p = run(root, root / "ID")
    assert p.returncode != 0 and "overlap" in p.stderr.lower()


def test_historical_logs_require_explicit_unverified_mode(tmp_path):
    root = tmp_path / "run"
    logs(root, [row()])
    assert run(root).returncode != 0
    p = run(root, "--allow-unverified")
    assert p.returncode == 0, p.stderr
    assert "not verified" in p.stderr.lower()


def test_unverified_mode_still_rejects_duplicates(tmp_path):
    root = tmp_path / "run"
    logs(root, [row(), row()])
    p = run(root, "--allow-unverified")
    assert p.returncode != 0 and "duplicate" in p.stderr.lower()


def test_manifest_does_not_cover_extra_result_file(tmp_path):
    root = tmp_path / "run"
    logs(root, [row()], [("task_0000/base", 0)])
    extra = root / "OOD/env/results.jsonl"
    extra.parent.mkdir(parents=True)
    extra.write_text(json.dumps(row("task_0000/env")))
    p = run(root)
    assert p.returncode != 0 and "unexpected" in p.stderr.lower()


def test_seed_subdirectories_are_averaged_as_one_family(tmp_path):
    root = tmp_path / "cabinet_joint"
    logs(root / "seed0", [row(seed=0)], [("task_0000/base", 0)])
    logs(root / "seed1", [row(seed=1, success=False)], [("task_0000/base", 1)])
    output = tmp_path / "report.json"
    p = run(root, "--expected-seeds", "0", "1", "--json", output)
    assert p.returncode == 0, p.stderr
    groups = json.loads(output.read_text())["ID"]
    assert set(groups) == {"cabinet_joint"}
    assert groups["cabinet_joint"]["ssr"] == 50
    assert groups["cabinet_joint"]["n_seeds"] == 2


def test_expected_seed_check_detects_an_entire_absent_run(tmp_path):
    root = tmp_path / "cabinet_joint"
    logs(root / "seed0", [row(seed=0)], [("task_0000/base", 0)])
    p = run(root, "--expected-seeds", "0", "1", "2")
    assert p.returncode != 0
    assert "missing" in p.stderr.lower() and "seed" in p.stderr.lower()
