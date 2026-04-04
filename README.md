# SENTINEL-Lite

## LTL Safety Checking (Local Add-On)

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


## Task Generation and Benchmark Pipeline

The current tabletop task-generation and benchmark pipeline lives in:

- [OmniGibson/omnigibson/task_generation/README.md](OmniGibson/omnigibson/task_generation/README.md)

This is the mainline path for benchmark-scale scene generation, subprocess-based scene execution, artifact validation, and reproducible output folders.

## Scene Curation Workflow

The post-run dataset repair and curation workflow lives in:

- [OmniGibson/omnigibson/task_generation/curation/README.md](OmniGibson/omnigibson/task_generation/curation/README.md)

This workflow is used when a benchmark run already exists but some scenes still need scene-level repair, replay, rerendering, or manual review before release.

## Legacy Kitchen-Bar MVP

The earlier kitchen-bar MVP pipeline is still preserved as a legacy reference:

- Legacy design / pipeline doc: [docs/omnigibson/cluttered_env_scene_generation_pipeline.md](docs/omnigibson/cluttered_env_scene_generation_pipeline.md)
- Runner: `OmniGibson/omnigibson/examples/environments/franka_mounted_mvp_runner_kitchen_bar.py`
- Config: `OmniGibson/omnigibson/configs/franka_mounted_behavior_cached_kitchen_bar.yaml`

This MVP line remains useful as historical context and for early cup-first manipulation experiments, but it is no longer the primary entrypoint for the current benchmark-scale task-generation workflow.

## Manipulation Safety-Critical BDDL Activity

Possible manipulation safety-critical predicates:

- `on_fire(?obj)` — object is on fire
- `hot(?obj)` — object is hot
- `touching(?obj1, ?obj2)` — object is touching another object
- `grasped(?obj1, ?obj2)` — agent is grasping object
- `covered(?obj1, ?obj2)` — object is covered by another object
- `broken(?obj)` — object is broken
- `ontop(?obj1, ?obj2)`, `nextto(?obj1, ?obj2)`, `inside(?obj1, ?obj2)` — spatial relationship
- `filled(?obj1, ?obj2)` — container is filled with liquid
- `toggled_on(?obj)` — device is turned on

Synset properties can be found in `bddl3/bddl/generated_data/syn_prop_annots_canonical.json`, or refer to [BEHAVIOR Synsets KnowledgeBase](https://behavior.stanford.edu/knowledgebase/synsets/index.html)

## BEHAVIOR Server Configuration

Notes have been moved to [docs/behavior_server_config_debug_notes.md](docs/behavior_server_config_debug_notes.md). 

> Path in repo: `SENTINEL-Lite/docs/behavior_server_config_debug_notes.md`

## RLinf Server Configuration

Notes have been moved to [RLinf/docs/rlinf_server_config_notes.md](RLinf/docs/rlinf_server_config_notes.md).

> Path in repo: `SENTINEL-Lite/RLinf/docs/rlinf_server_config_notes.md`
