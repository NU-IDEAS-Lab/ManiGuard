# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

- Atomic propositions generated from BDDL scope: `OmniGibson/omnigibson/utils/ltl_utils.py` (`AtomicPropositionGenerator`)
- Safety constraints loaded at task init: `OmniGibson/omnigibson/tasks/behavior_task.py`
- `LTLMonitor` converts LTL to LDBA and tracks automaton state per step
- Per-step LTL info exposed via `info["ltl"]` in env
- Spot library is optional — if unavailable, safety monitoring is skipped with a warning
- Task-level constraints: `bddl3/bddl/activity_definitions/<activity>/ltl_safety.json`
- Scene-level constraints: `datasets/behavior-1k-assets/scenes/<scene>/safety/ltl_safety.json`

### MVP Scene Entrypoints

Two fixed scene configurations for manipulation tasks:

1. **Coffee-table baseline** (regression): `franka_mounted_mvp_runner_coffee_table.py` + `franka_mounted_behavior_cached_coffee_table.yaml`
2. **Kitchen-bar mainline** (active development): `franka_mounted_mvp_runner_kitchen_bar.py` + `franka_mounted_behavior_cached_kitchen_bar.yaml`

## Common Commands

### Installation

```bash
# Full install (conda env "behavior")
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
# Flags: --omnigibson, --bddl, --joylo, --dataset, --eval, --asset-pipeline, --primitives, --dev
# --omnigibson requires --bddl; --primitives requires --omnigibson
```

### Running Simulations

```bash
conda activate behavior

# Kitchen-bar mainline (primary entrypoint)
python OmniGibson/omnigibson/examples/environments/franka_mounted_mvp_runner_kitchen_bar.py \
  --config OmniGibson/omnigibson/configs/franka_mounted_behavior_cached_kitchen_bar.yaml \
  --activity-name retrieve_filled_cup_from_clutter_safely \
  --episodes 1 --steps 300 --showcase-gui --strict-gate

# Coffee-table baseline
python OmniGibson/omnigibson/examples/environments/franka_mounted_mvp_runner_coffee_table.py \
  --config OmniGibson/omnigibson/configs/franka_mounted_behavior_cached_coffee_table.yaml \
  --episodes 1 --steps 300 --showcase-gui

# Basic installation check
python -m OmniGibson.examples.environments.behavior_env_demo
```

### Testing

```bash
pytest OmniGibson/tests/ -v          # OmniGibson tests (includes LTL tests)
pytest bddl3/tests/                  # BDDL tests
pytest RLinf/tests/unit_tests/       # RLinf unit tests
pytest RLinf/tests/e2e_tests/        # RLinf E2E tests (requires GPU)
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
- **`No module named 'spot'`**: Spot is optional; safety features degrade gracefully
- **Vulkan `ERROR_INCOMPATIBLE_DRIVER`**: Fix `VK_ICD_FILENAMES` to point to valid local ICD JSON
- **CUDA OOM**: Reduce `total_num_envs` in YAML config; ensure `component_placement` GPUs don't overlap

## Clutter Density Controls

When working with cluttered environment tasks:
- **Object count**: Edit BDDL file at `bddl3/bddl/activity_definitions/<activity>/problem0.bddl` (add objects in `:objects` and placement predicates in `:init`)
- **Packing tightness**: Runner flag `--clutter-density {low,medium,high,ultra}` with optional fine-grain overrides (`--pack-jitter-xy`, `--pack-min-clearance`, `--zone-utilization-cap`, `--pack-min-scale`)
