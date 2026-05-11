# Architecture overview

## Repo layout

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
│   ├── rlinf/           #   RLinf enum extension + env dispatch patches
│   ├── openpi/          #   OpenPI dataconfig + policy adapters
│   ├── teleop/          #   SO-101 → Franka teleop
│   ├── configs/         #   franka_mounted_sentinel.yaml + helpers
│   └── _omnigibson_patches.py   # runtime OmniGibson patches
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K @ v3.7.2
├── RLinf/               # submodule → RLinf/RLinf
├── vla_models/          # VLA checkpoints (gitignored)
├── tests/               # sentinel-side pytest suites
├── configs/             # RL / SFT training configs
├── scripts/             # shell entrypoints
├── tools/               # one-off utilities
├── teleop_bridge/       # ZMQ bridge for SO-101 teleop
└── datasets/            # BEHAVIOR dataset (gitignored)
```

## Upstream boundary

Anything under `behavior-1k/` or `RLinf/` is **upstream**. Never modify those trees.
Instead:

- Patch OmniGibson behaviors via `sentinel._omnigibson_patches`.
- Subclass tasks via `sentinel.tasks.*`.
- Extend RLinf via `sentinel.rlinf.patches`.

## Data flow

```
BDDL task definition
   ↓
OmniGibson scene sampling
   ↓
Environment reset / step  ──► LTL safety monitoring
   ↓                              ↓
Agent observation         info["ltl"]
   ↓
Policy action
   ↓
Physics simulation
   ↓
Reward signal
   ↓
RL training (RLinf)
```

## LTL safety system

| Component | Location |
|---|---|
| Atomic-proposition generator (`AtomicPropositionGenerator`) | `sentinel/utils/ltl_utils.py` |
| LTL → LDBA monitor (`LTLMonitor`) | `sentinel/utils/ltl_utils.py` |
| Per-step LTL info (`info["ltl"]`) | `sentinel/envs/sentinel_env.py` |
| High-level wrapper that loads task + scene `ltl_safety.json` | `sentinel/utils/safety_monitor.py` |
| Task-level constraints | `behavior-1k/bddl3/bddl/activity_definitions/<activity>/ltl_safety.json` |
| Scene-level constraints | `datasets/behavior-1k-assets/scenes/<scene>/safety/ltl_safety.json` |

The Spot library is **optional** — if unavailable, safety monitoring is skipped with a warning.
