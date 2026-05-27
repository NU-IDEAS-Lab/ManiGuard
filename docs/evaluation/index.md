# Evaluation

ManiGuard evaluates a VLA policy on **frozen benchmark scenes** via a websocket
**client/server split**: the policy runs in its own process (any framework / GPU
/ Python version), and the OmniGibson eval loop runs in another, talking
msgpack over a socket.

```
policy server                         eval client
┌────────────────────────┐  ws:8000  ┌──────────────────────────────┐
│ maniguard.serve.        │◄────────► │ maniguard.eval.benchmark      │
│   openpi_native         │           │  load snapshot → step policy  │
│  (or openpi serve_policy)│          │  → goal check → results.jsonl │
│  JAX or PyTorch ckpt     │          │  (OmniGibson + Isaac Sim)     │
└────────────────────────┘           └──────────────────────────────┘
```

The split exists because OmniGibson can't reliably reload scenes in one
long-lived process, and because the policy and simulator often need different
Python/CUDA stacks. For per-scene isolation, `scripts/run_benchmark_all_scenes.sh`
spawns one eval-client process per scene and aggregates the results.

## The eval loop (`eval/benchmark.py`)

1. **Resolve the source** — local dir or HF dataset repo via
   `maniguard.data.scene.hf_benchmark.resolve_benchmark_root`.
2. **Discover scenes** — `eval/scene_discovery.py:discover_scenes` walks the
   benchmark root for `scene_ep1.json` + `diagnostics.jsonl` pairs and resolves
   each scene's target, prompt, cameras, and goal conditions.
3. **Connect the policy** — OpenPI websocket client, the OmniGibson native
   client, or a `--random-policy` smoke stub.
4. **Per scene:** build the OmniGibson env from the frozen snapshot, **reload
   the controller after `env.reset()`** (apply `controller_preset` /
   `override_controller_config`), warm up, then step the policy in chunks of
   `execute_horizon`, optionally converting EEF deltas to joint targets
   (`ik_eef_to_joint`), checking success each step.
5. **Record** — write `results.jsonl` (one line/scene), `summary.json`
   (`success_rate`), `eval_config.json`, and an optional per-scene MP4.

## Configuring a run (`EvalConfig`)

One YAML per experiment, with CLI overrides (`eval/eval_config.py:config_from_cli`).
Key knobs:

| Group | Fields |
|---|---|
| Source | `benchmark_root`, `benchmark_revision`, `scene_filter`, `scenes`, `max_scenes` |
| Policy link | `host`, `port`, `use_openpi_client`, `random_policy` |
| Obs / action | `state_mode` (`eef_8d_axisangle`), `obs_layout` (`single_plus_wrist` / `three_cam`), `policy_cameras`, `action_dim`, `execute_horizon`, `gripper_binarize` |
| Controller | `controller_preset`, `override_controller_config`, `ik_eef_to_joint`, `joint_pos_kp` |
| Sim | `action_frequency`, `rendering_frequency`, `physics_frequency`, `headless`, `longfinger` |
| Eval | `max_steps`, `camera_resolution`, `save_video`, `output_dir` |

!!! warning "Match training"
    `state_mode`, `obs_layout`, `action_dim`, and especially the **controller**
    must match how the SFT data was produced, or the realized motion diverges
    from training. See [Controller, data, action & eval](../sft/end_to_end.md) —
    e.g. a delta-EEF policy trained on joint-tracked data should eval with
    `controller_preset: joint_position_impedance` + `ik_eef_to_joint: true`.

## Serving the policy (`serve/`)

`maniguard.serve.openpi_native` serves an openpi checkpoint over websocket and
**auto-detects JAX vs PyTorch** (orbax `params/` ⇒ JAX). `serve/_msgpack_numpy.py`
is the array (de)serialization helper. Alternatively run openpi's own
`scripts/serve_policy.py`. The observation/action schema is **driven by
`EvalConfig`** (`state_mode` / `obs_layout`) plus the policy's training config —
there is no separate profiles directory.

```bash
openpi/.venv/bin/python -m maniguard.serve.openpi_native \
  --config pi05_clutter_libero_lora --checkpoint <local_path>/<step>
```

## Success checking (`eval/goal_checker.py`)

- **`GoalChecker`** — evaluates the scene's `goal_conditions`, either a flat list
  (implicit AND) or a compound `and`/`or`/`not` tree, over OmniGibson predicates
  (`inside`, `ontop`, `touching`, `grasping`, `open`, `closed`, `covered`).
- **`GoalRegionChecker`** — spatial success: the target is held *and* intersects
  the goal-region sphere (via `maniguard.utils.goal_region`).

## Snapshot validation (`eval/snapshot_validator.py`)

QA for frozen snapshots before they enter a benchmark: offline checks (exactly
one target, target/support present in the snapshot, goal references the target,
manifest prompt matches) plus optional runtime materialization with a review
video. Used both in task-gen acceptance and to vet a benchmark set.

## Preparing benchmark scenes (`data/scene/`)

| Tool | Purpose |
|---|---|
| `hf_benchmark.py` | resolve a benchmark source (local path **or** HF repo id) to a local dir |
| `benchmark_repair.py` | rebuild `diagnostics.jsonl` to the current scene-registry format |
| `rewrite_scene_robot.py` | copy a scene dir and swap its robot to match the teleop/training distribution |
| `trim_scene_to_room.py` | trim a snapshot to the target room only (faster load, fewer distractors) |

## Perturbation sets (`data/perturbation_scaling.py`)

Generates self-contained **single-level perturbation** task sets from frozen base
tasks — `object` / `position` / `semantic` / `env` kinds — for measuring policy
robustness. The perturbations are materialized at load by
[`maniguard.envs.perturbation_runtime`](../foundations/env_layer.md#runtime-perturbations-perturbation_runtimepy)
(visual overrides + local-reconstruct instructions).

## Running it

| Setup | Page |
|---|---|
| Single machine (RTX Pro 6000) | [Single-machine eval](../one_machine_pro6000_eval.md) |
| Policy + sim on separate GPUs | [Two-machine eval](../two_machine_eval.md) |

## Code map

| Concern | Code |
|---|---|
| Eval loop, obs packing, IK step | `maniguard/eval/benchmark.py` |
| Run config (YAML + CLI) | `maniguard/eval/eval_config.py` |
| Scene enumeration | `maniguard/eval/scene_discovery.py` |
| Success predicates / regions | `maniguard/eval/goal_checker.py` |
| Snapshot QA | `maniguard/eval/snapshot_validator.py` |
| Policy server | `maniguard/serve/openpi_native.py` |
| Benchmark prep / perturbation | `maniguard/data/scene/`, `maniguard/data/perturbation_scaling.py` |
