"""Resume-cursor policy for stop/resume datagen collection (pure, family-agnostic).

The per-demo master seed is deterministic in the draw index ``k`` (see variation.py). To guarantee a
resumed / topped-up run yields only UNSEEN seeds, the driver persists a task-level cursor ``next_draw``
in ``_summary.json`` and restarts the sampler stream from it. These two helpers are the whole policy,
kept pure so they unit-test without a sim."""
from __future__ import annotations


def resolve_start_k(existing_summary: dict | None, start_draw_override: int | None,
                    ondisk_max_draw: int = -1) -> int:
    """The draw index to start (or resume) a run's variant stream from.

    Requested start precedence: explicit ``--start-draw`` override > the prior run's ``next_draw``
    (top-up) > 0 (a fresh task, or a pre-fix summary without the field). That request is then floored by
    ``ondisk_max_draw + 1`` — the highest draw index any trajectory ALREADY on disk used — so a resume can
    NEVER re-draw a k it already produced a demo for, even when the summary's ``next_draw`` was lost (a
    dedup pass wiped it, a crash skipped the summary write, a partial run reset it). The summary can still
    push the start HIGHER than the on-disk floor (it also skips FAILED k values, which leave no traj).
    ``ondisk_max_draw`` defaults to -1 (no on-disk trajs => no floor)."""
    if start_draw_override is not None:
        base = int(start_draw_override)
    elif existing_summary and existing_summary.get("next_draw") is not None:
        base = int(existing_summary["next_draw"])
    else:
        base = 0
    return max(base, int(ondisk_max_draw) + 1)


def compute_next_draw(last_run_draw: int | None, start_k: int) -> int:
    """The cursor to persist after a run. ``last_run_draw`` = the highest draw index actually attempted
    this run (incl. failures — failed k are deterministic, so the cursor must skip past them, not just
    past successes). If nothing ran, the cursor is unchanged (``start_k``)."""
    return int(start_k) if last_run_draw is None else int(last_run_draw) + 1
