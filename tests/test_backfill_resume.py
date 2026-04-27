from pathlib import Path

from sentinel.utils.backfill_resume import has_task_snapshot, plan_family_resume


def _make_task_dir(root: Path, family: str, task_id: str) -> Path:
    task_dir = root / family / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "scene_ep1.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "diagnostics.jsonl").write_text("{}\n", encoding="utf-8")
    return task_dir


def test_has_task_snapshot_requires_both_files(tmp_path: Path):
    task_dir = tmp_path / "family" / "task_0000"
    task_dir.mkdir(parents=True)
    assert has_task_snapshot(task_dir) is False
    (task_dir / "scene_ep1.json").write_text("{}\n", encoding="utf-8")
    assert has_task_snapshot(task_dir) is False
    (task_dir / "diagnostics.jsonl").write_text("{}\n", encoding="utf-8")
    assert has_task_snapshot(task_dir) is True


def test_plan_family_resume_deletes_last_completed_and_restarts_from_it(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    selected = [
        _make_task_dir(input_root, "table", "task_0000"),
        _make_task_dir(input_root, "table", "task_0001"),
        _make_task_dir(input_root, "table", "task_0002"),
    ]

    _make_task_dir(output_root, "table", "task_0000")
    _make_task_dir(output_root, "table", "task_0001")

    tasks_to_run, actions = plan_family_resume(selected, output_root)

    assert [task.name for task in tasks_to_run] == ["task_0001", "task_0002"]
    assert actions[0].family == "table"
    assert actions[0].status == "resume"
    assert actions[0].deleted_task_id == "task_0001"
    assert (output_root / "table" / "task_0001").exists() is False


def test_plan_family_resume_skips_completed_family(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    selected = [
        _make_task_dir(input_root, "transfer", "task_0000"),
        _make_task_dir(input_root, "transfer", "task_0001"),
    ]

    _make_task_dir(output_root, "transfer", "task_0000")
    _make_task_dir(output_root, "transfer", "task_0001")

    tasks_to_run, actions = plan_family_resume(selected, output_root)

    assert tasks_to_run == []
    assert actions[0].status == "complete"
    assert actions[0].completed_prefix == 2


def test_plan_family_resume_starts_fresh_when_no_outputs_exist(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    selected = [
        _make_task_dir(input_root, "stack_flat", "task_0000"),
        _make_task_dir(input_root, "stack_flat", "task_0001"),
    ]

    tasks_to_run, actions = plan_family_resume(selected, output_root)

    assert [task.name for task in tasks_to_run] == ["task_0000", "task_0001"]
    assert actions[0].status == "fresh"
    assert actions[0].restart_task_id == "task_0000"
