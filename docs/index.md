# ManiGuard

ManiGuard is a Python package built on top of [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
that adds **LTL (Linear Temporal Logic) safety checking**, **task-generation pipelines**,
and **VLA policy evaluation** for robotic manipulation in simulated household environments.

It integrates:

- **Physics simulation** — OmniGibson / NVIDIA Omniverse
- **Formal task specification** — BDDL
- **Safety verification** — LTL / Spot
- **Distributed RL training** — RLinf with Ray
- **VLA policy serving** — pi0.5, GR00T, OpenPI

## Where to start

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](getting-started/installation.md)**
  Set up the `behavior` conda env, RLinf venv, and BEHAVIOR datasets.

- :material-sitemap: **[Architecture overview](architecture/overview.md)**
  Repo layout, data flow, upstream-boundary rules, and the LTL safety system.

- :material-factory: **[Task-generation pipelines](pipelines/index.md)**
  Clutter, stack, transfer, cabinet, and empty-scene pipelines.

- :material-robot-industrial: **[GraspGen pipeline](graspgen_pipeline.md)**
  Per-object grasp eval feeding the RL reset dataset.

- :material-school: **[RL training](rl_training.md)**
  PPO grasp training on top of `ManiGuardEnv`.

- :material-broadcast: **[Evaluation](one_machine_pro6000_eval.md)**
  Single-machine and two-machine eval setups.

</div>

## Repository layout

```
.
├── maniguard/            # ManiGuard package (LTL, task-gen, envs, rl, serve, teleop)
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K @ v3.7.2
├── RLinf/               # submodule → RLinf/RLinf
├── tests/               # maniguard-side pytest suites
├── configs/             # RL / SFT training configs
├── scripts/             # shell entrypoints
├── tools/               # one-off utilities
└── docs/                # this site
```

**Upstream boundary:** anything under `behavior-1k/` or `RLinf/` is upstream.
Do not modify those trees — patch behaviors via `maniguard._omnigibson_patches`,
subclass via `maniguard.tasks.*`, or extend via `maniguard.rlinf.patches`.

## Building these docs locally

```bash
pip install mkdocs-material
mkdocs serve            # preview at http://127.0.0.1:8000
mkdocs build --strict   # build to ./site, fail on warnings
```
