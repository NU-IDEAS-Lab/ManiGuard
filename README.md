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

## Common manipulation-safety predicates

Useful BDDL predicates for expressing safety-critical constraints (see
[BEHAVIOR Synsets knowledge base](https://behavior.stanford.edu/knowledgebase/synsets/index.html)
for object-level properties):

`on_fire(?obj)`, `hot(?obj)`, `touching(?a, ?b)`, `grasped(?a, ?b)`,
`covered(?a, ?b)`, `broken(?obj)`, `ontop` / `inside` / `nextto`,
`filled(?container, ?liquid)`, `toggled_on(?obj)`, plus Sentinel's own
`upright(?obj)`, `dropped(?obj)`, `stashed(?obj)`.
