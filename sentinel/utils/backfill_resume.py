from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ResumeAction:
    family: str
    status: str
    restart_task_id: str | None
    deleted_task_id: str | None
    completed_prefix: int
    total_tasks: int

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "status": self.status,
            "restart_task_id": self.restart_task_id,
            "deleted_task_id": self.deleted_task_id,
            "completed_prefix": int(self.completed_prefix),
            "total_tasks": int(self.total_tasks),
        }


def has_task_snapshot(task_dir: Path) -> bool:
    return (task_dir / "scene_ep1.json").is_file() and (task_dir / "diagnostics.jsonl").is_file()


def plan_family_resume(selected_task_dirs: Iterable[Path], output_root: Path) -> tuple[list[Path], list[ResumeAction]]:
    grouped: dict[str, list[Path]] = {}
    for task_dir in selected_task_dirs:
        grouped.setdefault(task_dir.parent.name, []).append(task_dir)

    tasks_to_run: list[Path] = []
    actions: list[ResumeAction] = []

    for family, ordered_tasks in grouped.items():
        prefix_len = 0
        for task_dir in ordered_tasks:
            if has_task_snapshot(output_root / family / task_dir.name):
                prefix_len += 1
            else:
                break

        total_tasks = len(ordered_tasks)
        if prefix_len == total_tasks:
            actions.append(
                ResumeAction(
                    family=family,
                    status="complete",
                    restart_task_id=None,
                    deleted_task_id=None,
                    completed_prefix=prefix_len,
                    total_tasks=total_tasks,
                )
            )
            continue

        if prefix_len > 0:
            restart_idx = prefix_len - 1
            deleted_task_id = ordered_tasks[restart_idx].name
            shutil.rmtree(output_root / family / deleted_task_id, ignore_errors=True)
            actions.append(
                ResumeAction(
                    family=family,
                    status="resume",
                    restart_task_id=ordered_tasks[restart_idx].name,
                    deleted_task_id=deleted_task_id,
                    completed_prefix=prefix_len,
                    total_tasks=total_tasks,
                )
            )
            tasks_to_run.extend(ordered_tasks[restart_idx:])
        else:
            actions.append(
                ResumeAction(
                    family=family,
                    status="fresh",
                    restart_task_id=ordered_tasks[0].name if ordered_tasks else None,
                    deleted_task_id=None,
                    completed_prefix=0,
                    total_tasks=total_tasks,
                )
            )
            tasks_to_run.extend(ordered_tasks)

    return tasks_to_run, actions
