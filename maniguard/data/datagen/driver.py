"""Layer-3 driver / orchestration (family-agnostic).

Single-task flow:
  scene_from_task_dir → family skeleton derives waypoints → per-subtask cuRobo plan
  (grasp / move / contact primitives) → JointController execute → Recorder → commit
  on success / abort on failure.

Also: batch sweep over a family's tasks (n variants each) + quality-audit artifacts
(per-family base|demo grids + success-rate stats), 2 tmux concurrent like the bench.

Filled in Step 2 (clutter end-to-end template) — see doc §6, §9.
"""
from __future__ import annotations

# TODO(Step 2): implement run_task(task_dir, family, ...) + sweep(...). Stub for now
# so the package scaffolds + imports cleanly.
