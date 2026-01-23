# SENTINEL-Lite

### LTL Safety Checking (Local Add-On)
This fork adds optional LTL safety checking and monitoring for BehaviorTask scenes:

- Atomic propositions are generated from the BDDL object scope and predicates in `omnigibson/utils/ltl_utils.py` (`AtomicPropositionGenerator`).
- Safety constraints are loaded at task init in `omnigibson/tasks/behavior_task.py`, validated with Spot if available, and combined with `&`.
- An `LTLMonitor` (in `omnigibson/utils/ltl_utils.py`) converts LTL to an LDBA and tracks automaton state.
- Per-step LTL info is exposed via `info["ltl"]` in `omnigibson/envs/env_base.py` (also on `reset()`).
- Tests live in `tests/test_ltl_propositions.py`.

Where to add / edit constraints:

- Task-level: `bddl3/bddl/activity_definitions/<activity_name>/ltl_safety.json`
- Scene-level: `datasets/behavior-1k-assets/scenes/<scene_name>/safety/ltl_safety.json` (scene_dir resolved via `get_scene_path(scene_model)`)

Note: Spot is optional. If Spot is unavailable, safety validation and monitor init are skipped with a warning.
