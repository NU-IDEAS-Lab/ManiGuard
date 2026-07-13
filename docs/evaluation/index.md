# Evaluation

ManiGuard evaluates a VLA policy on **frozen benchmark scenes** and scores it on
**two independent axes — success *and* safety**. A capable-but-reckless policy that
finishes the task while knocking a glass off the table is **not** a pass; the
benchmark reports safe-success vs. unsafe-success vs. failure.

ManiGuard's contribution to evaluation is the **benchmark + the checkers**, not any
one model's serving stack:

- the **eval client** (`maniguard.eval.benchmark`) — loads a frozen scene, steps
  whatever policy is connected, and drives the two checkers;
- the **success checker** (`eval/goal_checker.py`) and the **LTL safety monitor**
  (`utils/safety_monitor.py`) — the model-agnostic core;
- a small, well-defined **policy-integration contract** so any VLA can plug in.

How a given checkpoint is *served / loaded* is the model family's own concern
(openpi, GR00T, SmolVLA each have their own serving docs) — see
[Plugging in a policy](#plugging-in-a-policy).

```
policy (any framework/GPU/Python)      eval client (OmniGibson + Isaac Sim)
┌────────────────────────────┐  obs   ┌──────────────────────────────────┐
│  a policy exposed to the    │ ─────► │  maniguard.eval.benchmark         │
│  eval contract              │        │  load snapshot → step policy      │
│  (websocket, or in-process) │ ◄───── │  → success check + LTL safety     │
└────────────────────────────┘ action │  → results.jsonl                  │
                                       └──────────────────────────────────┘
```

The policy is decoupled from the simulator because OmniGibson can't reliably reload
scenes in one long-lived process, and because the policy and simulator often need
different Python/CUDA stacks. For per-scene isolation,
`scripts/run_benchmark_all_scenes.sh` spawns one eval-client process per scene and
aggregates the results.

## The eval loop (`eval/benchmark.py`)

1. **Resolve the source** — local dir or HF dataset repo via
   `maniguard.data.scene.hf_benchmark.resolve_benchmark_root`.
2. **Discover scenes** — `eval/scene_discovery.py:discover_scenes` walks the
   benchmark root for `scene_ep1.json` + `diagnostics.jsonl` pairs and resolves each
   scene's target, prompt, cameras, goal conditions, and LTL safety spec.
3. **Connect the policy** — through the integration contract below (or a
   `--random-policy` smoke stub).
4. **Per scene:** build the OmniGibson env from the frozen snapshot, **reload the
   controller after `env.reset()`**, warm up, then step the policy in chunks of
   `execute_horizon` — checking **success every step** and running the **LTL safety
   monitor every step**.
5. **Record** — write `results.jsonl` (one line/scene, success + safety outcome),
   `summary.json`, `eval_config.json`, and an optional per-scene MP4. Each run lands
   in its own `output_dir/<run_name>` subfolder (auto `YYYYmmdd_HHMMSS`, or
   `..._<TAG>` with `--tag`), so runs never overwrite.

## Plugging in a policy

The eval client and the policy meet at one **model-agnostic contract**:

- **Client → policy:** an observation — one external overview image + a wrist image,
  the robot state, and the language prompt (exact keys/shapes driven by `EvalConfig`
  `state_mode` / `external_cam`, to match the checkpoint's training config).
- **Policy → client:** an **action chunk** (`(horizon, action_dim)`), consumed
  `execute_horizon` steps at a time.

The default transport is an **openpi-client-compatible websocket** (msgpack-numpy):
any server implementing that interface connects with no glue. But **VLA families do
not share one serving convention** — openpi serves over websocket, GR00T ships its
own inference server, LeRobot/SmolVLA policies typically run in-process — so a
non-openpi model plugs in through a small **model-specific adapter** that bridges its
serving to this contract. The benchmark fixes the interface once; each model family
we evaluate gets its own thin adapter.

In code the boundary is already explicit — `connect_policy` returns
`(policy, client_type)` and `query_policy` dispatches per type — so adding a family
is a new adapter, not a change to the contract or the checkers.

| Policy family | How it connects | Status |
|---|---|---|
| openpi / pi0.5 | native websocket; reference adapter `maniguard.serve.openpi_native` wraps openpi's own loader | supported |
| GR00T, SmolVLA, others | a thin adapter bridging the family's own serving (its inference server, or in-process) to the contract | per-family adapter |

> **Serving the checkpoint itself is upstream.** To actually launch a pi0.5 / GR00T /
> SmolVLA policy, follow that family's serving docs; ManiGuard only defines the
> contract and ships `serve/openpi_native.py` as a reference.

## Configuring a run (`EvalConfig`)

One YAML per experiment, with CLI overrides (`eval/eval_config.py:config_from_cli`).
Key knobs:

| Group | Fields |
|---|---|
| Source | `benchmark_root`, `benchmark_revision`, `scene_filter`, `scenes`, `max_scenes` |
| Policy link | `host`, `port`, `use_openpi_client`, `random_policy` |
| Obs / action | `state_mode`, `external_cam` (`left` / `right` — which third-person overview feeds the fixed 1-overview + wrist layout), `action_dim`, `execute_horizon`, `gripper_binarize` |
| Controller | `controller_preset`, `override_controller_config`, `ik_eef_to_joint`, `joint_pos_kp` |
| Sim | `action_frequency`, `rendering_frequency`, `physics_frequency`, `headless`, `longfinger` |
| Eval | `max_steps`, `camera_resolution`, `save_video`, `output_dir`, `run_name` / `tag` |

!!! warning "Match training"
    `state_mode`, `external_cam`, `action_dim`, and especially the **controller** must
    match how the SFT data was produced, or the realized motion diverges from
    training. See [Controller, data, action & eval](../sft/end_to_end.md).

## Success checking (`eval/goal_checker.py`)

- **`GoalChecker`** — evaluates the scene's `goal_conditions`, either a flat list
  (implicit AND) or a compound `and`/`or`/`not` tree, over OmniGibson predicates
  (`inside`, `ontop`, `touching`, `grasping`, `open`, `closed`, `covered`).
- **`GoalRegionChecker`** — spatial success: the target is held *and* intersects the
  goal-region sphere (via `maniguard.utils.goal_region`).

## Safety checking (`utils/safety_monitor.py`)

This is what sets ManiGuard apart from a plain success benchmark. Each task ships an
**LTL safety spec** (`diagnostics.ltl_safety`: a Spot-compatible formula over atomic
propositions such as *dropped* / *upright* / *touching* / premature-lift). The eval
runs a **`TaskLTLMonitor`** every step over the rollout; a single violation marks the
episode unsafe. Success and safety are reported **together** (safe-success /
unsafe-success / failure), so a policy is credited only for finishing the task
*without* tipping, dropping, spilling, or lifting-before-closing. The same monitor
and active-object resolution (`build_active_objects_for_ltl`) are shared with datagen
and bench-finalize — one source of truth for what "safe" means.

## Snapshot validation (`eval/snapshot_validator.py`)

QA for frozen snapshots before they enter a benchmark: offline checks (exactly one
target; target/support present; goal references the target; manifest prompt matches)
plus optional runtime materialization with a review video.

## Preparing benchmark scenes (`data/scene/`)

| Tool | Purpose |
|---|---|
| `hf_benchmark.py` | resolve a benchmark source (local path **or** HF repo id) to a local dir |
| `benchmark_repair.py` | rebuild `diagnostics.jsonl` to the current scene-registry format |
| `rewrite_scene_robot.py` | copy a scene dir and swap its robot to match the training distribution |
| `trim_scene_to_room.py` | trim a snapshot to the target room only (faster load, fewer distractors) |

## Perturbation sets (`data/perturbation_scaling.py`)

Generates self-contained **single-level perturbation** task sets from frozen base
tasks — `object` / `position` / `semantic` / `env` kinds — for measuring policy
robustness. Perturbations are materialized at load by
[`maniguard.envs.perturbation_runtime`](../foundations/env_layer.md#runtime-perturbations-perturbation_runtimepy).

## Running it

Start any policy exposed to the contract (for openpi, the reference adapter), then
point the eval client at a benchmark — a local dir or an HF repo id:

```bash
# 1. serve a policy (openpi reference adapter; other families: their own serving)
openpi/.venv/bin/python -m maniguard.serve.openpi_native \
  --config <train-config-name> --checkpoint <local_path>/<step>

# 2. run the eval client against the benchmark (HF repo id or local dir)
python -m maniguard.eval.benchmark \
  --benchmark-root <org>/ManiGuard-Bench --config <eval_yaml> --tag smoke

# all scenes, one process per scene:
bash scripts/run_benchmark_all_scenes.sh ...
```

## Code map

| Concern | Code |
|---|---|
| Eval loop, obs packing, policy dispatch | `maniguard/eval/benchmark.py` |
| Run config (YAML + CLI) | `maniguard/eval/eval_config.py` |
| Scene enumeration | `maniguard/eval/scene_discovery.py` |
| Success predicates / regions | `maniguard/eval/goal_checker.py` |
| LTL safety monitor | `maniguard/utils/safety_monitor.py` |
| Snapshot QA | `maniguard/eval/snapshot_validator.py` |
| openpi reference adapter | `maniguard/serve/openpi_native.py` |
| Benchmark prep / perturbation | `maniguard/data/scene/`, `maniguard/data/perturbation_scaling.py` |
