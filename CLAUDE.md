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

SENTINEL-Lite is a fork of the BEHAVIOR-1K benchmark that adds LTL (Linear Temporal Logic) safety checking and monitoring for robotic manipulation tasks in simulated household environments. It integrates physics simulation (OmniGibson/NVIDIA Omniverse), formal task specification (BDDL), safety verification (LTL/Spot), and distributed RL training (RLinf with Ray).

## Architecture

The repository is a monorepo with several major subsystems:

- **OmniGibson/** — Physics simulation engine built on NVIDIA Omniverse/Isaac Sim. Provides Gym-compatible environments (`envs/env_base.py`), robot controllers, object states, sensors, and scene management. The core environment class extends `gym.Env`.
- **bddl3/** — Behavior Domain Definition Language v3. Contains 1000+ activity definitions in `bddl/activity_definitions/`. Each activity has BDDL problem files defining objects, initial conditions, and goal states. Safety tasks also have `ltl_safety.json` files.
- **RLinf/** — Distributed RL framework using Ray. Supports PPO, GRPO, SAC algorithms with models like OpenPI (Pi0.5), OpenVLA, GR00T. Has its own venv managed by `uv`, separate from the main conda env.
- **joylo/** — Teleoperation interface (GELLO) for robot control and data collection.
- **knowledgebase/** — Flask web app for browsing BEHAVIOR-1K task/object/scene data.
- **scene_generation/** — Scene generation scripts and configs.
- **asset_pipeline/** — DVC-managed 3D asset processing pipeline.

**Key data flow**: BDDL task definition → OmniGibson scene sampling → Environment reset/step → LTL safety monitoring → Agent observation → Policy action → Physics simulation → Reward signal → RL training (RLinf)

### LTL Safety System

- Atomic propositions generated from BDDL scope: `sentinel/utils/ltl_utils.py` (`AtomicPropositionGenerator`)
- Safety constraints loaded at task init: `behavior-1k/OmniGibson/omnigibson/tasks/behavior_task.py`
- `LTLMonitor` converts LTL to LDBA and tracks automaton state per step
- Per-step LTL info exposed via `info["ltl"]` in env
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
# Full install (conda env "behavior")
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
# Flags: --omnigibson, --bddl, --joylo, --dataset, --eval, --asset-pipeline, --primitives, --dev
# --omnigibson requires --bddl; --primitives requires --omnigibson
```

### Testing

```bash
pytest behavior-1k/OmniGibson/tests/ -v          # OmniGibson tests (includes LTL tests)
pytest behavior-1k/bddl3/tests/                  # BDDL tests
pytest RLinf/tests/unit_tests/       # RLinf unit tests
pytest RLinf/tests/e2e_tests/        # RLinf E2E tests (requires GPU)
```

### Task Generation Pipelines

All pipelines live in `sentinel/task_generation/` and share a `BasePipeline` class from `pipeline_common.py`. Each auto-discovers a surface in the given scene, generates BDDL + `ltl_safety.json`, spawns objects, places the robot, and runs LTL-monitored rollouts.

```bash
conda activate behavior

# Tabletop clutter (retrieve target from fragile clutter)
python -m sentinel.task_generation.clutter_scene_pipeline \
  --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video --strict-gate

# Stack retrieval (retrieve target from under a stack)
python -m sentinel.task_generation.stack_scene_pipeline \
  --scene-model Benevolence_1_int --stack-height medium --episodes 1 --steps 300 --save-video

# Food transfer (move food between containers without touching)
python -m sentinel.task_generation.transfer_scene_pipeline \
  --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video

# Pinch-point (fragile near target handle)
python -m sentinel.task_generation.pinch_point_pipeline \
  --scene-model Benevolence_1_int --episodes 1 --steps 300 --save-video

# Cabinet clutter (retrieve from inside a cabinet)
python -m sentinel.task_generation.cabinet_clutter_pipeline \
  --scene-model Rs_int --episodes 1 --steps 300 --save-video

# Empty scene (no pre-existing furniture; setup: clutter/stack/transfer)
python -m sentinel.task_generation.empty_scene_pipeline \
  --setup stack --episodes 1 --steps 300 --save-video

# Dry-run any pipeline (generates BDDL + LTL only, no simulator)
python -m sentinel.task_generation.clutter_scene_pipeline \
  --scene-model Benevolence_1_int --dry-run
```

### Benchmark (multi-scene)

```bash
# Run a pipeline across all eligible scenes with per-scene timeout
python -m sentinel.task_generation.run_benchmark \
  --pipeline table --steps 300 --episodes 1 --timeout 300 --save-video

# Pipeline choices: table, cabinet, transfer, stack
# Restrict to specific scenes:
python -m sentinel.task_generation.run_benchmark \
  --pipeline transfer --scenes Benevolence_1_int Rs_int --steps 300
```

### Linting

```bash
ruff check . --fix
ruff format .
# Or via pre-commit:
pre-commit run --all-files
```

Root `ruff.toml` excludes `joylo/`. OmniGibson uses 120-char line length. RLinf uses 88-char line length with stricter rules (Google docstrings, type hints).

### RLinf Training

```bash
cd RLinf
source .venv/bin/activate    # Separate uv-managed venv, not the conda env
bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi
```

### Documentation

```bash
mkdocs serve    # Preview at localhost:8000
mkdocs build    # Build static site
```

## Environment Setup

Two separate Python environments exist:
1. **`behavior` conda env** — for OmniGibson simulation, BDDL, teleoperation
2. **RLinf `.venv`** — uv-managed venv for distributed RL training (separate due to dependency conflicts)

Key environment variables (for RLinf/headless deployment):
- `ISAAC_PATH` — path to Isaac Sim package
- `OMNIGIBSON_DATA_PATH` — path to BEHAVIOR datasets
- `BEHAVIOR_PATH` — path to SENTINEL-Lite root
- `OMNIGIBSON_HEADLESS=1` — required for server/headless rendering
- `VK_ICD_FILENAMES` — Vulkan ICD config for headless GPU rendering
- `CUDA_VISIBLE_DEVICES` — GPU selection (use when transitioning between multi-GPU tasks)

## Common Debug Issues

- **PhysX CUDA error 700**: Set `CUDA_VISIBLE_DEVICES=0` to pin a single GPU
- **`typing_extensions` errors with torch 2.6.0**: Remove outdated `typing_extensions` from Isaac Sim so conda's version is used
- **Vulkan `ERROR_INCOMPATIBLE_DRIVER`**: Fix `VK_ICD_FILENAMES` to point to valid local ICD JSON
- **CUDA OOM**: Reduce `total_num_envs` in YAML config; ensure `component_placement` GPUs don't overlap