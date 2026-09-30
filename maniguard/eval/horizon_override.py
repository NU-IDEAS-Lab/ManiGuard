"""Apply per-task prompt and goal overrides from a JSON table.

Each task entry may replace prompt and goal_conditions in a copied mapping.
The source benchmark files are not changed. The same table can be supplied
to collection and evaluation to define a shortened task consistently.

Only those two fields are permitted. Safety specifications are inherited,
and goal-region changes are rejected because marker creation occurs during
scene loading. Missing task keys raise an error. LTL safety-monitor verdicts
should not be interpreted as proving liveness on a shortened finite trace."""
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
