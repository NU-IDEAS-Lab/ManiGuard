# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. 

## Approach
- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.
- Apply first principles thinking. Don't assume I fully understand the goal. Stay prudent, start from the original needs and problems. If the goal is unclear, please pause and discuss with me. If the goal is clear but the path is not optimal, please directly suggest a shorter, lower-cost approach.
- User instructions always override this file.


## Project Overview

ManiGuard is a Python package built on top of [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) that adds LTL (Linear Temporal Logic) safety checking, task-generation pipelines, and VLA policy eval for robotic manipulation in simulated household environments. It integrates physics simulation (OmniGibson/NVIDIA Omniverse), formal task specification (BDDL), safety verification (LTL/Spot), and distributed RL training (RLinf with Ray).

## Architecture

```
.
├── maniguard/            # ManiGuard Python package (all maniguard-owned code)
│   ├── object_states/   #   Dropped, Upright
│   ├── utils/           #   ltl_utils, safety_monitor, bddl_generator, …
│   ├── task_generation/ #   clutter / stack / transfer / cabinet / … pipelines
│   ├── tasks/           #   ManiGuardGraspTask
│   ├── envs/            #   ManiGuardEnv
│   ├── eval/            #   benchmark runner, websocket eval client
│   ├── serve/           #   pi0.5 / GR00T / websocket policy servers
│   ├── rl/              #   SB3 PPO grasp training
│   ├── rlinf/           #   RLinf enum extension + env dispatch patches
│   ├── openpi/          #   OpenPI dataconfig + policy adapters
│   ├── teleop/          #   SO-101 → Franka teleop
│   ├── configs/         #   franka_mounted_maniguard.yaml + helpers
│   └── _omnigibson_patches.py   # runtime OmniGibson patches
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K @ v3.7.2
│                        #   contains OmniGibson/, bddl3/, joylo/, docs/,
│                        #   asset_pipeline/, knowledgebase/, eval-jobqueue/
├── vla_models/          # VLA checkpoints (user-downloaded, gitignored)
├── tests/               # maniguard-side pytest suites
├── configs/             # RL / SFT training configs
├── scripts/             # shell entrypoints
├── tools/               # one-off utilities
├── teleop_bridge/       # ZMQ bridge for SO-101 teleop
└── datasets/            # BEHAVIOR dataset (user-downloaded, gitignored)
```

**Upstream boundary**: anything under `behavior-1k/` or `RLinf/` is upstream. Do not modify those trees — patch behaviors via `maniguard._omnigibson_patches`, subclass via `maniguard.tasks.*`, or extend via `maniguard.rlinf.patches`.

**Key data flow**: BDDL task definition → OmniGibson scene sampling → Environment reset/step → LTL safety monitoring → Agent observation → Policy action → Physics simulation → Reward signal → RL training (RLinf)

### LTL Safety System

- Atomic propositions generated from BDDL scope: `maniguard/utils/ltl_utils.py` (`AtomicPropositionGenerator`)
- `LTLMonitor` (same module) converts LTL formulas to LDBA form and tracks automaton state per step
- Per-step LTL info exposed via `info["ltl"]` from `maniguard/envs/maniguard_env.py`
- High-level wrapper that loads task + scene `ltl_safety.json` files: `maniguard/utils/safety_monitor.py`
- Spot library is optional — if unavailable, safety monitoring is skipped with a warning
- Task-level constraints: `behavior-1k/bddl3/bddl/activity_definitions/<activity>/ltl_safety.json`
- Scene-level constraints: `datasets/behavior-1k-assets/scenes/<scene>/safety/ltl_safety.json`

## Git & Collaboration Workflow

- **Commit frequently**: when a logical unit of work is complete (new feature, bug fix, refactor), propose a commit. Do not batch unrelated changes.
- **Always get approval first**: before committing, show the user the proposed commit message and list of files. Do not commit without explicit approval.
- **Commit message style**: `type(scope): short description` (e.g. `feat(isaac):`, `fix(clutter):`, `docs:`, `chore:`). Body explains "why", not "what".
- **After each commit**: add a one-line summary to the relevant `DEV_LOG.md` recording what was accomplished.
- **Separate concerns**: reference code, documentation, and feature code go in separate commits.
- **Never force-push** to shared branches without asking.

## Common Commands

### Installation

```bash
# Clone with submodules (or: git submodule update --init --recursive)
git clone --recursive <repo-url> && cd ManiGuard

# Install BEHAVIOR-1K + dataset (runs upstream's setup.sh from inside the submodule;
# --dataset downloads encrypted assets into behavior-1k/datasets/, which matches
# OmniGibson's default resolver — no env var needed afterwards).
cd behavior-1k
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
# Flags: --omnigibson, --bddl, --joylo, --dataset, --eval, --asset-pipeline, --primitives, --dev
# --omnigibson requires --bddl; --primitives requires --omnigibson
cd ..

# Install maniguard (editable)
conda activate behavior
pip install -e .                     # base
pip install -e ".[rl,serve]"        # with RL + policy-server extras

# (Override the dataset path only if you keep BEHAVIOR assets elsewhere,
# e.g. HPC shared storage. Default resolves to behavior-1k/datasets/.)
# export OMNIGIBSON_DATA_PATH=/abs/path/to/datasets
```

### Testing

```bash
# ManiGuard tests (our refactor + LTL + task-gen unit tests)
pytest tests/ -v

# Upstream tests (inside submodules, rarely needed)
pytest behavior-1k/OmniGibson/tests/ -v
pytest behavior-1k/bddl3/tests/
```

### Task Generation Pipelines

All pipelines live in `maniguard/task_generation/` and share a `BasePipeline` class from `pipeline_common.py`. Each randomly selects the objects, auto-discovers a surface in the given scene, spawns objects, places the robot, and runs LTL-monitored rollouts. See `task_generation` folder for details.


### Benchmark (multi-scene)

```bash
# Run a pipeline across all eligible scenes with per-scene timeout
python -m maniguard.task_generation.run_benchmark \
  --pipeline table --steps 300 --episodes 1 --timeout 300 --save-video

# Pipeline choices: table, cabinet, transfer, stack
# Restrict to specific scenes:
python -m maniguard.task_generation.run_benchmark \
  --pipeline transfer --scenes Benevolence_1_int Rs_int --steps 300
```

### Linting

```bash
ruff check . --fix
ruff format .
# Or via pre-commit:
pre-commit run --all-files
```

Root `ruff.toml` excludes `behavior-1k/`, `vla_models/`, `Omnireset/`, and runtime output dirs — lint only touches maniguard-owned code. The `behavior-1k/` submodule has its own ruff config (120-char line length for OmniGibson)

### Documentation (upstream BEHAVIOR-1K docs)

```bash
cd behavior-1k
mkdocs serve    # Preview at localhost:8000
mkdocs build    # Build static site
```

### GraspGen pipeline (per-object grasp eval + RL reset dataset)

End-to-end install + server + run instructions for the
`maniguard.rl.grasps.render_grasps` pipeline (NVlabs/GraspGen ZMQ server
→ cuRobo motion plan → physics validation → `.pt` for
`GraspDatasetResetter` + diagnostic PNGs + success MP4):
[`docs/graspgen_pipeline.md`](docs/graspgen_pipeline.md).

## Environment Setup

Two separate Python environments exist:
1. **`behavior` conda env** — for OmniGibson simulation, BDDL, teleoperation
2. **RLinf `.venv`** — uv-managed venv for distributed RL training (separate due to dependency conflicts)

Key environment variables (for RLinf/headless deployment):
- `ISAAC_PATH` — path to Isaac Sim package
- `OMNIGIBSON_DATA_PATH` — path to BEHAVIOR datasets
- `BEHAVIOR_PATH` — path to ManiGuard root
- `OMNIGIBSON_HEADLESS=1` — required for server/headless rendering
- `VK_ICD_FILENAMES` — Vulkan ICD config for headless GPU rendering
- `CUDA_VISIBLE_DEVICES` — GPU selection (use when transitioning between multi-GPU tasks)

## Common Debug Issues
- **Vulkan `ERROR_INCOMPATIBLE_DRIVER`**: Fix `VK_ICD_FILENAMES` to point to valid local ICD JSON