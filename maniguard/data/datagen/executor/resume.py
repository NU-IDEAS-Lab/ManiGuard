"""Resume-cursor policy for stop/resume datagen collection (pure, family-agnostic).

The per-demo master seed is deterministic in the draw index ``k`` (see variation.py). To guarantee a
resumed / topped-up run yields only UNSEEN seeds, the driver persists a task-level cursor ``next_draw``
in ``_summary.json`` and restarts the sampler stream from it. These two helpers are the whole policy,
kept pure so they unit-test without a sim."""
from __future__ import annotations


def resolve_start_k(existing_summary: dict | None, start_draw_override: int | None) -> int:
    """The draw index to start (or resume) a run's variant stream from.

    Precedence: explicit ``--start-draw`` override > the prior run's ``next_draw`` (top-up) > 0 (a fresh
    task, or a pre-fix summary written before this field existed)."""
    if start_draw_override is not None:
        return int(start_draw_override)
    if existing_summary and existing_summary.get("next_draw") is not None:
        return int(existing_summary["next_draw"])
    return 0


def compute_next_draw(last_run_draw: int | None, start_k: int) -> int:
    """The cursor to persist after a run. ``last_run_draw`` = the highest draw index actually attempted
    this run (incl. failures — failed k are deterministic, so the cursor must skip past them, not just
    past successes). If nothing ran, the cursor is unchanged (``start_k``)."""
    return int(start_k) if last_run_draw is None else int(last_run_draw) + 1
