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

Two families share the runtime contract.

**Tabletop (in-scene)** — auto-discover a real support surface in a named
`--scene-model`:

| Pipeline | Module | One-liner |
|---|---|---|
| [Tabletop clutter](clutter.md) | `clutter_scene_pipeline` | Retrieve target from fragile clutter |
| ↳ [Liquid transport](liquid_transport.md) — *variant* | `liquid_transport_pipeline` | Clutter with a liquid-filled target; no spill or tip |
| [Stack retrieval](stack.md) | `stack_scene_pipeline` | Retrieve target from under a stack |
| [Food transfer](transfer.md) | `transfer_scene_pipeline` | Move food between containers without touching it |
| ↳ [Dusty transfer](dusty_transfer.md) — *variant* | `dusty_transfer_pipeline` | Transfer, but wipe the dusty destination clean first |
| [Lid transport](lid_transport.md) | `lid_transport_pipeline` | Cover a container before lifting it (close-before-lift) |
| [Wet transport](wet_transport.md) | `wet_transport_pipeline` | Carry a container without passing it over water-sensitive objects |

**Empty-scene** — start from a bare floor and synthesize a surface (no `--scene-model`):

| Pipeline | Module | One-liner |
|---|---|---|
| [Empty scene](empty_scene.md) | `empty_scene_pipeline` | Random surface + a clutter / stack / transfer setup |
| [Empty invert](empty_invert.md) | `empty_invert_pipeline` | Empty a liquid container before inverting it; keep the table dry |
| [Cabinet pickup](cabinet_pickup.md) | `cabinet_pickup_pipeline` | Open a drawer and retrieve a target that fits its cavity |
| [Jar transport](jar_transport.md) | `jar_transport_pipeline` | Close a hinged jar before lifting it to a goal region |

Each page documents what the pipeline does plus its gate checks and LTL safety
constraints — with example renders (where available) from the `6fam-base`
dataset.

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
