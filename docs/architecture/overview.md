# Architecture overview

ManiGuard is a thin, **maniguard-owned** layer on top of an unmodified
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) / OmniGibson install. It
adds LTL safety monitoring, task-generation pipelines, teleop + scripted data
collection, SFT data export, and policy evaluation. RL training is under development.

## The pipeline lifecycle

Everything in the package falls into one of five lifecycle stages plus a shared
foundation layer. The docs are organized the same way.

```
                 ┌─────────────────────────── Foundations ───────────────────────────┐
                 │  env layer · LTL safety · object states · OmniGibson patches        │
                 └─────────────────────────────────────────────────────────────────────┘
                                              ▲ (used by every stage)
  Task generation ──► Data collection ──► SFT ──► Evaluation
   (task_generation)   (teleop · datagen)  (data/ → SFT)      (eval/, serve/)

  RL training (grasp policies) is under development.
```

| Stage | Package | What it produces |
|---|---|---|
| **Task generation** | `maniguard/task_generation/` | Frozen scene snapshots + BDDL + `ltl_safety.json` |
| **Data collection** | `maniguard/data/teleop/`, `maniguard/data/datagen/` | Teleop / scripted demo HDF5s + videos |
| **SFT** | `maniguard/data/` → per-model SFT (openpi / GR00T / SmolVLA) | LeRobot v2.1 datasets + trained checkpoints |
| **Evaluation** | `maniguard/eval/`, `maniguard/serve/` | Benchmark results, success metrics |
| **RL** *(under development)* | — | Grasp policies (planned) |

## Repo layout

```
.
├── maniguard/            # ManiGuard Python package (all maniguard-owned code)
│   ├── _omnigibson_patches.py   # runtime OmniGibson patches (applied on import)
│   ├── object_states/   #   Dropped, Upright
│   ├── utils/           #   LTL (ltl_utils, safety_monitor), task_spec, geometry
│   ├── task_generation/ #   clutter / cabinet / stack / jar / lid / dusty / transfer / liquid pipelines
│   ├── envs/            #   scene registry + frozen-snapshot runtime (no live env class)
│   ├── data/            #   datagen (scripted SFT demos), bench_builder, teleop, lerobot, real_teleop, scene + playback
│   ├── eval/            #   benchmark runner, goal checker, scene discovery
│   ├── {openpi,gr00t,smolvla}_sft/  # per-model SFT configs / embodiment
│   └── serve/           #   websocket VLA policy server (openpi_native)
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K (upstream)
├── docs/                # this documentation site (mkdocs sources)
├── configs/             # eval / SFT training configs
├── tools/               # SFT drivers + per-family bench-surgery utilities
├── scripts/             # shell entrypoints
├── tests/               # maniguard-side pytest suites
├── teleop_bridge/       # ZMQ bridge for SO-101 teleop
└── vla_models/          # VLA checkpoints (user-downloaded, gitignored)
```

## Upstream boundary

Anything under `behavior-1k/` is **upstream** — never edit that tree.
ManiGuard stays decoupled by:

- **Runtime patching** OmniGibson via `maniguard._omnigibson_patches` (see
  [OmniGibson patches](../foundations/omnigibson_patches.md)) — applied automatically
  on `import maniguard`, so the `behavior-1k/` tree is never edited.
- Building env configs from **frozen scene snapshots** rather than subclassing
  the env (see [Environment layer](../foundations/env_layer.md)).

## Data flow

```
BDDL activity + scene  ──►  task_generation pipeline
                                  │  spawns objects, runs LTL-monitored rollout
                                  ▼
              frozen snapshot (scene_ep1.json + diagnostics.jsonl)
                                  │
        ┌─────────────────────────┴─────────────────────────────┐
        ▼                                                        ▼
   data collection                                          eval rollout
   teleop → playback → HDF5                                 (load snapshot,
   datagen → RAW → LeRobot                                  run VLA policy)
        │                                                        ▲
        ▼                                                        │
   LeRobot v2.1 dataset ──► per-model SFT ──► policy checkpoint ──┘
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
