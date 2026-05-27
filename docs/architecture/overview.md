# Architecture overview

ManiGuard is a thin, **maniguard-owned** layer on top of an unmodified
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) / OmniGibson install. It
adds LTL safety monitoring, task-generation pipelines, teleop + scripted data
collection, SFT data export, policy evaluation, and RL training.

## The pipeline lifecycle

Everything in the package falls into one of five lifecycle stages plus a shared
foundation layer. The docs are organized the same way.

```
                 ┌─────────────────────────── Foundations ───────────────────────────┐
                 │  env layer · LTL safety · object states · OmniGibson patches        │
                 └─────────────────────────────────────────────────────────────────────┘
                                              ▲ (used by every stage)
  Task generation ──► Data collection ──► SFT ──► Evaluation
   (task_generation)   (teleop, cuRobo)   (data/ → openpi)   (eval/, serve/)
                                  └────────────► RL (rl/, RLinf)
```

| Stage | Package | What it produces |
|---|---|---|
| **Task generation** | `maniguard/task_generation/` | Frozen scene snapshots + BDDL + `ltl_safety.json` |
| **Data collection** | `maniguard/data/teleop/`, `maniguard/rl/grasps/` | Teleop / scripted demo HDF5s |
| **SFT** | `maniguard/data/` → vendored `openpi/` | LeRobot v2.1 datasets + norm stats |
| **Evaluation** | `maniguard/eval/`, `maniguard/serve/` | Benchmark results, success metrics |
| **RL** | `maniguard/rl/`, submodule `RLinf/` | Trained grasp policies |

## Repo layout

```
.
├── maniguard/            # ManiGuard package (all maniguard-owned code)
│   ├── _omnigibson_patches.py   # runtime OmniGibson patches (applied on import)
│   ├── object_states/   #   Dropped, Upright
│   ├── utils/           #   LTL (ltl_utils, safety_monitor), task_spec, geometry
│   ├── task_generation/ #   clutter / stack / transfer / lid / liquid / … pipelines
│   ├── envs/            #   scene registry + frozen-snapshot runtime (no live env class)
│   ├── teleop/          #   SO-101 / GELLO → Franka teleop
│   ├── data/            #   HDF5 → LeRobot export, playback render, norm stats, scene utils
│   ├── eval/            #   benchmark runner, goal checker, scene discovery
│   ├── serve/           #   websocket VLA policy servers
│   ├── rl/              #   PPO grasp training + GraspGen/cuRobo grasp pipeline
│   └── configs/         #   franka_mounted_maniguard.yaml + config_path() helper
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K @ v3.7.2 (upstream)
├── RLinf/               # submodule → RLinf/RLinf (upstream, separate venv)
├── tests/               # maniguard-side pytest suites
├── configs/             # eval / RL / SFT YAML configs
├── scripts/ · tools/    # shell entrypoints / one-off utilities
└── docs/                # this site
```

!!! note "Stale-doc cleanup"
    Earlier revisions of this page listed `maniguard/tasks/`, `maniguard/rlinf/`,
    and `maniguard/openpi/` subpackages. **Those no longer exist.** The RLinf
    integration was excised; the active RL stack uses
    `maniguard.rl.tasks.pick_and_lift.PickAndLiftTask` directly, and OpenPI is
    consumed from the vendored top-level `openpi/` checkout.

## Upstream boundary

Anything under `behavior-1k/` or `RLinf/` is **upstream** — never edit those
trees. ManiGuard stays decoupled by:

- **Runtime patching** OmniGibson via `maniguard._omnigibson_patches` (see
  [OmniGibson patches](../foundations/omnigibson_patches.md)). Two upstream files
  still carry local edits on this branch (`utils/bddl_utils.py`,
  `tasks/grasp_task.py`); extracting them is tracked follow-up work.
- Building env configs from **frozen scene snapshots** rather than subclassing
  the env (see [Environment layer](../foundations/env_layer.md)).

## Data flow

```
BDDL activity + scene  ──►  task_generation pipeline
                                  │  spawns objects, runs LTL-monitored rollout
                                  ▼
              frozen snapshot (scene_ep1.json + diagnostics.jsonl)
                                  │
        ┌─────────────────────────┼─────────────────────────────┐
        ▼                         ▼                              ▼
   teleop / scripted        eval rollout                   RL training
   demo  → playback         (load snapshot,                (rl.tasks +
   render → HDF5            run VLA policy)                 PPO / RLinf)
        │                         ▲
        ▼                         │
   data/ LeRobot export ──► SFT (openpi) ──► policy checkpoint ──┘
```

LTL safety monitoring runs *alongside* the rollout at every stage: a
[`TaskLTLMonitor`](../foundations/ltl_safety.md) is attached to the env and
steps an automaton derived from the task/scene `ltl_safety.json`.

## Foundation layer

| Component | Page |
|---|---|
| Scene registry + frozen-snapshot env builder + controller presets | [Environment layer](../foundations/env_layer.md) |
| Atomic propositions, LTL → automaton monitoring, `Dropped`/`Upright` states | [LTL safety system](../foundations/ltl_safety.md) |
| Runtime OmniGibson patches + config helpers | [OmniGibson patches & configs](../foundations/omnigibson_patches.md) |
