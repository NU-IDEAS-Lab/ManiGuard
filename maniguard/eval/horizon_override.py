"""Task-horizon variants: substitute a task's goal + instruction at load time.

A *horizon variant* stops a benchmark task short of its shipped goal. The motivating
case is the cabinet sim2real comparison: the real-robot setup has both a full-horizon
policy and a "firsthalf" one that ends once the blocking object is moved aside and the
drawer is pulled open, so the sim side needs a matching pair.

Two things must change together for that to be a well-posed task, and forgetting either
is a silent error rather than a crash:

  * ``goal_conditions`` -- the shipped cabinet goal is ``inside & closed``, which a
    firsthalf rollout can never satisfy (it ends with the drawer OPEN). Left alone, every
    demonstration would be discarded by the datagen gate and every eval rollout scored a
    failure.
  * ``prompt`` -- the shipped instruction asks the policy to put the jar inside and close
    the drawer. Left alone, the policy is told to do the whole task while being judged on
    half of it, in BOTH training and eval.

Both live in the same dict: datagen reads ``diagnostics`` (``executor.gate.build_gate``
for the goal, ``executor.engine`` for the prompt) and eval reads ``scene_info``. So one
substitution on that dict reaches success checking and language in both pipelines, and
``eval.goal_checker.build_goal_checker`` -- shared by both -- keeps "success" meaning
exactly the same thing in collection and evaluation, as it does for the shipped tasks.

The finalized bench on disk is never modified; variants exist only in these JSON tables
(``configs/firsthalf/*.json``) and are applied per run.

⚠️ LTL safety is deliberately NOT overridable here. The cabinet constraints are pure
safety formulas (``G (...)``), which stay meaningful on any prefix of a trajectory, so a
horizon variant inherits them unchanged. A liveness constraint (``F (...)``) would be
unsatisfiable on a truncated horizon and would need explicit thought rather than a
mechanism that quietly rewrites it.
"""
from __future__ import annotations

import json

_CACHE: dict[str, dict] = {}

# Only these keys may be substituted. Anything else in the table is a mistake we want to
# hear about immediately -- an override silently introducing, say, a different target
# object or camera pose would produce a variant that is no longer the same scene.
#
# ``goal_region`` is deliberately NOT allowed, even though eval would honour it: datagen's
# ``scene_from_task_dir`` parses the region into ``SceneBundle.goal_spec`` and spawns its
# marker while building the scene, before any override could run. Permitting it would give
# a mechanism that works in eval and silently half-works in collection. A goal_region
# family needing a horizon variant should get the override applied inside scene loading.
_ALLOWED = frozenset({"prompt", "goal_conditions"})


def load_table(map_path: str) -> dict:
    """Load (and cache) a horizon-variant table, validating its shape."""
    if map_path not in _CACHE:
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        tasks = raw.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"{map_path}: expected a non-empty 'tasks' object")
        for key, patch in tasks.items():
            if not isinstance(patch, dict) or not patch:
                raise ValueError(f"{map_path}: task {key!r} has an empty patch")
            extra = set(patch) - _ALLOWED
            if extra:
                raise ValueError(
                    f"{map_path}: task {key!r} overrides {sorted(extra)}, "
                    f"but only {sorted(_ALLOWED)} may be substituted"
                )
        _CACHE[map_path] = tasks
    return _CACHE[map_path]


def apply_horizon_override(source: dict, map_path: str, task_key: str) -> dict:
    """Return a COPY of ``source`` with ``task_key``'s variant fields substituted.

    ``source`` is a datagen ``diagnostics`` row or an eval ``scene_info`` dict;
    ``task_key`` is the bench-relative task id, e.g. ``cabinet_pickup/task_0019``.

    A miss RAISES rather than returning the input unchanged. A silent fallback is the one
    failure mode this cannot absorb: the run would collect or evaluate the full-horizon
    task while every artifact around it claims to be the variant.
    """
    tasks = load_table(map_path)
    if task_key not in tasks:
        raise KeyError(
            f"{task_key!r} has no horizon variant in {map_path} "
            f"(defined: {sorted(tasks)}). Refusing to fall back to the full-horizon task."
        )
    patched = dict(source)
    patched.update(tasks[task_key])
    return patched
