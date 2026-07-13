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

## ManiGuard-Bench families

The released [ManiGuard-Bench](../datagen/pipeline.md) is these **6 families**
(200 base tasks). Each page states the task goal + safety and how the family's
scenes are generated:

| Family | Tasks | One-liner |
|---|---:|---|
| [Clutter pickup](clutter_pickup.md) | 55 | Pick a named target out of a cluttered pack into the goal region (+ liquid subset) |
| [Cabinet pickup](cabinet_pickup.md) | 35 | Open a drawer, place the target inside, close it |
| [Lid transport](lid_transport.md) | 30 | Put the lid on before lifting the container to the goal |
| [Stack retrieve](stack_retrieve.md) | 28 | Pull the bottom object out from under a stack without toppling it |
| [Jar transport](jar_transport.md) | 26 | Close a hinged jar before carrying it to the goal |
| [Dusty transfer](dusty_transfer.md) | 26 | Wipe a dusty pot clean, then transfer food into it with the tool |

## Generation infrastructure

Shared machinery every family builds on:

- [Data flow](data_flow.md) — the four-stage architecture (offline pools → selection → surface → placement).
- [Add a custom pipeline](custom_pipeline.md) — author a new `BasePipeline` subclass.
- [Empty-scene runner](empty_scene.md) — synthesize a surface on a bare floor (used by the empty-scene families).
- [Food transfer (base)](transfer.md) — the transfer base that dusty transfer extends.

## Additional families

[Other families](other_families.md) the pipeline can generate but that are **not**
in the shipped bench (`wet_transport`, `empty_invert`).

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

Pipeline choices for `--pipeline` (keys of `_PIPELINE_SCRIPTS`): `table`
(clutter), `transfer`, `dusty_transfer`, `stack` (+ `stack_same` / `stack_flat`
/ `stack_receptacle`), `lid_transport`, `liquid_transport`, `wet_transport`,
`jar_transport`, `cabinet_pickup`. The empty-scene families (`cabinet_pickup`,
`jar_transport`) are usually run via their own CLI with `--task-id` rather than
the per-scene benchmark loop.

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
