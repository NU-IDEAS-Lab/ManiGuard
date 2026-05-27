# Task-generation pipelines

All pipelines live in `maniguard/task_generation/` and share a `BasePipeline`
runtime contract from `pipeline_common.py`. Each pipeline auto-discovers a
support surface in the given scene, generates BDDL + `ltl_safety.json` at
runtime, spawns objects, places the robot, runs gate checks, and executes an
LTL-monitored rollout.

The cross-cutting [Data flow](data_flow.md) page describes the four-stage
architecture every pipeline shares: offline JSON pool generation (raycast
scans + admission filters) → object selection → scene + surface selection →
placement.

## Pipelines

| Pipeline | Module | One-liner |
|---|---|---|
| [Tabletop clutter](clutter.md) | `clutter_scene_pipeline` | Retrieve target from fragile clutter |
| [Stack retrieval](stack.md) | `stack_scene_pipeline` | Retrieve target from under a stack |
| [Food transfer](transfer.md) | `transfer_scene_pipeline` | Move food between containers without touching |
| [Empty scene](empty_scene.md) | `empty_scene_pipeline` | No pre-existing furniture; setup: clutter / stack / transfer |
| [Empty invert](empty_invert.md) | `empty_invert_pipeline` | Inverted-orientation variant of empty scene |
| [Lid transport](lid_transport.md) | `lid_transport_pipeline` | Transport a lid between target containers |
| [Liquid transport](liquid_transport.md) | `liquid_transport_pipeline` | Transport an open liquid container without spilling |
| [Wet transport](wet_transport.md) | `wet_transport_pipeline` | Transport a wet object without dripping |

Each page documents what the pipeline does, its CLI flags, run examples,
expected output artifacts, gate checks, and LTL safety constraints.

## Multi-scene benchmark

`run_benchmark.py` orchestrates a pipeline across multiple scenes, one
subprocess per scene, with per-scene timeout and resume logic. The
subprocess-per-scene model intentionally avoids long-lived Isaac Sim /
renderer state and makes failed scenes easier to isolate and rerun.

```bash
conda activate behavior

python -m maniguard.task_generation.run_benchmark \
  --pipeline table \
  --scenes hall_conference_large \
  --episodes 1 --steps 300 --density medium --timeout 1800
```

Pipeline choices for `--pipeline`: `table`, `transfer`, `stack`.

## Dry-run (BDDL + LTL only, no simulator)

```bash
python -m maniguard.task_generation.clutter_scene_pipeline \
  --scene-model Benevolence_1_int --dry-run
```

## Artifact contract

A successful scene run is expected to produce, at minimum:

| File | Purpose |
|---|---|
| `diagnostics.jsonl` | Outcome signals: gate results, LTL status, density, active-object summary |
| `scene_ep1.json` | Frozen scene snapshot for replay, rerendering, later evaluation |
| `stdout.log` | Runtime trace for debugging |
| `rollout_ep1.mp4` | Canonical human-reviewable video |

## Gate vs LTL — what's the difference?

Two validation layers run sequentially:

- **Gate checks** (pre-rollout, structural): "Is this scene structurally valid
  enough to start?" Robot/target poses are finite, robot base near floor plane,
  selected mount pose collision-free, target inside reach band, plus per-pipeline
  extras (e.g. pack integrity for clutter). With `--strict-gate`, a gate
  failure aborts the episode.
- **LTL safety rollout** (during execution, semantic): "Does the scene remain
  safe and semantically intact over time?" `combined_ltl` from
  `ltl_safety.json` is evaluated step by step by `TaskLTLMonitor`.
