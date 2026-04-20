# SENTINEL-Lite

SENTINEL-Lite is a Python package built on top of
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) that adds
LTL (Linear Temporal Logic) safety checking, task-generation pipelines, and
VLA-policy eval plumbing for robotic manipulation in simulated household scenes.

## Repository layout

```
.
├── sentinel/            # Sentinel Python package (all sentinel-owned code)
│   ├── object_states/   #   Dropped, Upright
│   ├── utils/           #   ltl_utils, safety_monitor, bddl_generator, …
│   ├── task_generation/ #   clutter / stack / transfer / cabinet / … pipelines
│   ├── tasks/           #   SentinelGraspTask
│   ├── envs/            #   SentinelEnv
│   ├── eval/            #   benchmark runner, websocket eval client
│   ├── serve/           #   pi0.5 / GR00T / websocket policy servers
│   ├── rl/              #   SB3 PPO grasp training
│   ├── rlinf/           #   RLinf patches (enum extension, env dispatch)
│   ├── openpi/          #   OpenPI dataconfig + policy adapters
│   ├── teleop/          #   SO-101 → Franka teleop
│   ├── configs/         #   franka_mounted_sentinel.yaml + helpers
│   └── _omnigibson_patches.py   # runtime patches on vanilla OmniGibson
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K @ v3.7.2
├── RLinf/               # submodule → RLinf/RLinf @ 5714022
├── vla_models/          # VLA checkpoints (user-downloaded, .gitignore)
├── tests/               # sentinel-side tests
├── configs/             # RL / SFT training configs
├── scripts/             # shell entry points (run_rl.sh, prepare_sft_data.sh, …)
├── tools/               # one-off utilities
└── teleop_bridge/       # ZMQ bridge for SO-101 teleop
```

## Installation

1. **Clone with submodules**:

   ```bash
   git clone --recursive git@github.com:NU-IDEAS-Lab/SENTINEL-Lite.git
   cd SENTINEL-Lite
   # or, if already cloned:
   git submodule update --init --recursive
   ```

2. **Install BEHAVIOR-1K + download its dataset.** `setup.sh` both
   creates the `behavior` conda env (with OmniGibson, bddl3, JoyLo,
   primitives) and — with `--dataset` — downloads the encrypted
   BEHAVIOR-1K asset bundle into `behavior-1k/datasets/` (matching
   upstream's expected layout, which is also OmniGibson's default
   resolver path — no env var needed afterwards):

   ```bash
   cd behavior-1k
   ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
   cd ..
   ```

   If you keep the dataset somewhere else (shared HPC storage, etc.)
   set `OMNIGIBSON_DATA_PATH=/abs/path/to/datasets` to override.

3. **Install RLinf** (required for RL training + SFT adapters in
   `sentinel.rlinf` / `sentinel.openpi`). The RLinf submodule manages
   its own uv-based virtualenv:

   ```bash
   cd RLinf
   # Follow RLinf's own install guide — typically something like:
   uv sync
   cd ..
   ```

4. **Install sentinel** in the conda env (editable, from repo root):

   ```bash
   conda activate behavior
   pip install -e .                 # base install
   pip install -e ".[rl,serve]"    # with RL + websocket policy server extras
   ```

5. **Sentinel-generated benchmark scenes** (our own `scene_ep*.json`
   + `diagnostics.jsonl` bundles) are distributed separately from the
   BEHAVIOR asset bundle — see
   [`sentinel/task_generation/README.md`](sentinel/task_generation/README.md)
   for how pipelines write them, and the HuggingFace-hosted benchmark
   release for the frozen versions. (They live under a different root
   than `behavior-1k/datasets/` on purpose: BEHAVIOR assets are
   Stanford-licensed and not redistributable, whereas SENTINEL
   benchmark scenes are ours.)

Importing `sentinel` applies a small set of runtime patches that extend
OmniGibson with Sentinel-specific object states, BDDL predicates, grasp-goal
hold-step counting, a link-fallback-aware grasp reward, and a tensor-safe
`draw_debug_markers`. Set `SENTINEL_SKIP_OMNIGIBSON_PATCH=1` to skip the
patches entirely (useful for pure-Python consumers that don't need
OmniGibson at runtime).

## LTL safety monitoring

- Atomic propositions are generated from the BDDL object scope in
  [`sentinel/utils/ltl_utils.py`](sentinel/utils/ltl_utils.py)
  (`AtomicPropositionGenerator`).
- `LTLMonitor` converts LTL formulas to LDBA form and tracks automaton
  state per `env.step()`.
- Per-step monitor output is exposed via `info["ltl"]` inside
  [`sentinel/envs/sentinel_env.py`](sentinel/envs/sentinel_env.py).
- [`sentinel/utils/safety_monitor.py`](sentinel/utils/safety_monitor.py)
  wraps activity-level + scene-level `ltl_safety.json` files into a
  ready-to-use monitor.

Safety-constraint JSON locations:

- Task-level: `behavior-1k/bddl3/bddl/activity_definitions/<activity>/ltl_safety.json`
- Scene-level: `datasets/behavior-1k-assets/scenes/<scene>/safety/ltl_safety.json`

Spot is an optional dependency; if it's unavailable, LTL validation is
skipped with a warning.

## Task generation + benchmark

Pipelines live in [`sentinel/task_generation/`](sentinel/task_generation/).
Each pipeline auto-discovers a surface in the target scene, generates a
BDDL problem and `ltl_safety.json`, spawns objects, places the mounted
Franka, and runs an LTL-monitored rollout. See
[`sentinel/task_generation/README.md`](sentinel/task_generation/README.md)
for flags + the full taxonomy of available pipelines.

```bash
conda activate behavior

# Single scene, BDDL + LTL dry-run (no simulator)
python -m sentinel.task_generation.clutter_scene_pipeline \
  --scene-model Benevolence_1_int --dry-run

# Full-sim rollout
python -m sentinel.task_generation.clutter_scene_pipeline \
  --scene-model Benevolence_1_int --episodes 1 --steps 300 \
  --save-video --strict-gate

# Multi-scene benchmark
python -m sentinel.task_generation.run_benchmark \
  --pipeline table --steps 300 --episodes 1 --timeout 300 --save-video
```

Run `OMNIGIBSON_HEADLESS=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`
on headless nodes.

## VLA policy evaluation

Sentinel evaluates vision-language-action policies over websocket: the
policy runs in its own process (Pi0.5 / GR00T / GR00T-N1.6), the
environment loop runs in another, and
[`sentinel/eval/benchmark.py`](sentinel/eval/benchmark.py) drives one
scene at a time off a local dir or HuggingFace dataset repo. A per-scene
wrapper is in [`scripts/run_benchmark_all_scenes.sh`](scripts/run_benchmark_all_scenes.sh)
because OmniGibson can't reliably reload scenes in the same process —
the wrapper spawns one Python per scene and aggregates the
`results.jsonl` files afterwards.

**Terminal 1 — serve the policy**:

```bash
conda activate behavior

# Pi0.5 (SFT checkpoint served via native openpi):
python -m sentinel.serve.pi05_franka --port 8000 \
    --checkpoint-dir vla_models/RLinf-pi05-SFT-Stack-cube

# GR00T-N1.5 trained by RLinf:
python -m sentinel.serve.gr00t_server --port 8000 \
    --checkpoint-dir vla_models/RLinf-Gr00t-SFT-Stack-cube

# GR00T-N1.6 (upstream Isaac-GR00T API, run from vla_models/Isaac-GR00T/.venv):
python -m sentinel.serve.gr00t_n16_server --port 8000 \
    --checkpoint-dir vla_models/GR00T-N1.6-DROID
```

Eval profiles describing each policy's observation / action schema live
under [`sentinel/eval/profiles/`](sentinel/eval/profiles/).

**Terminal 2 — drive the eval loop**:

```bash
# Single scene (local dir)
python -m sentinel.eval.benchmark \
    --benchmark-root datasets/safety-benchmark \
    --host 127.0.0.1 --port 8000 --max-steps 300 --save-video

# All scenes in a HuggingFace-hosted benchmark (per-scene subprocess)
bash scripts/run_benchmark_all_scenes.sh \
    --benchmark-root IDEAS-Lab-Northwestern/sentinel-lite-taskgen-staging-20260415-v1 \
    --host <policy_host> --port 8000 --max-steps 300 --save-video
```

Per-episode artefacts land in `outputs/benchmark_eval/<scene>/`. Goal
checking uses real OmniGibson states via
[`sentinel/eval/goal_checker.py`](sentinel/eval/goal_checker.py) against
the pipeline-generated `diagnostics.jsonl`.

## RL fine-tuning (SFT + PPO via RLinf)

Sentinel wraps RLinf (submodule) with Pi0.5 / GR00T adapters. Training
runs inside RLinf's uv venv; Sentinel is imported first so its
TrainConfigs are in RLinf's registry before the entry point dispatches.
See [`sentinel/rlinf/patches.py`](sentinel/rlinf/patches.py) for how
the hooks attach.

**SFT** (supervised fine-tune on teleop demos):

```bash
# One-time: convert teleop HDF5 demos -> LeRobot dataset + norm_stats.json
bash scripts/prepare_sft_data.sh

# Train
export SENTINEL_PI05_BASE=/abs/path/to/pi05_base_or_sft_ckpt
export SENTINEL_LEROBOT_ROOT=$PWD/outputs/lerobot_datasets
export SENTINEL_LEROBOT_REPO_ID=sentinel/clutter_pickup_v1
bash scripts/run_sft.sh sentinel_clutter_sft_openpi
# Hydra overrides after the config name, e.g.
bash scripts/run_sft.sh sentinel_goblet_sft_openpi runner.max_steps=1
```

**PPO post-train** from an SFT checkpoint:

```bash
export SENTINEL_PI05_SFT_CKPT=/abs/path/to/sft/ckpt
export OMNIGIBSON_DATA_PATH=$PWD/behavior-1k/datasets
export ISAAC_PATH=/abs/path/to/isaac-sim
# Optional overrides — defaults point at the checked-in seed corpus:
#   SENTINEL_BENCHMARK_ROOT=$PWD/datasets/safety-benchmark
#   SENTINEL_ACTIVITY_ROOT=$PWD/behavior-1k/bddl3/bddl/activity_definitions
bash scripts/run_rl.sh sentinel_clutter_ppo_openpi_pi05
```

Configs under [`configs/rl/`](configs/rl/) and
[`configs/sft/`](configs/sft/); the SB3 PPO path (no RLinf) is in
[`sentinel/rl/training/ppo.py`](sentinel/rl/training/ppo.py).

## Teleoperation (SO-101 → Franka)

Two-process setup (LeRobot's 3.12 venv for the hardware side, Sentinel's
`behavior` conda env for OmniGibson), bridged by ZMQ at ~60 Hz. Detailed
hardware setup + ZMQ schema:
[`teleop_bridge/README.md`](teleop_bridge/README.md).

**Terminal 1** — SO-101 server (LeRobot venv):

```bash
conda activate lerobot
# Real hardware:
python teleop_bridge/so101_server.py --port /dev/ttyACM0
# Mock mode (no physical arm needed):
python teleop_bridge/so101_server.py --mock
```

**Terminal 2** — Franka teleop in OmniGibson (`behavior` env):

```bash
conda activate behavior
python -m sentinel.teleop.so101_franka_teleop \
    --snapshot outputs/pipeline_runs/<run>/scene_ep1.json \
    --record --output outputs/teleop/demo.hdf5
```

Recordings are HDF5 trajectories compatible with the SFT data prep
script above.

**Playback** a recorded trajectory (with optional observation dump for
dataset curation):

```bash
python -m sentinel.teleop.so101_franka_playback \
    --input outputs/teleop/demo.hdf5 \
    --output outputs/teleop/demo_obs.hdf5 --record
```

## Common manipulation-safety predicates

Useful BDDL predicates for expressing safety-critical constraints (see
[BEHAVIOR Synsets knowledge base](https://behavior.stanford.edu/knowledgebase/synsets/index.html)
for object-level properties):

`on_fire(?obj)`, `hot(?obj)`, `touching(?a, ?b)`, `grasped(?a, ?b)`,
`covered(?a, ?b)`, `broken(?obj)`, `ontop` / `inside` / `nextto`,
`filled(?container, ?liquid)`, `toggled_on(?obj)`, plus Sentinel's own
`upright(?obj)`, `dropped(?obj)`, `stashed(?obj)`.
