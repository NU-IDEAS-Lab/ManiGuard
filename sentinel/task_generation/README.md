# Task Generation and Benchmark Pipeline

This directory contains the current task-generation mainline used for **tabletop cluttered environment** scene dataset production.

The primary entrypoint is `run_benchmark.py`. It launches one subprocess per scene, runs the selected task-generation pipeline, validates the expected artifacts, and writes a benchmark run directory containing reproducible outputs such as `summary.csv`, `diagnostics.jsonl`, `scene_ep*.json`, and `rollout_ep*.mp4`.

> [!NOTE]
> Current fixed group of dataset scenes: [Google Drive Link for 0319 benchmark_runs Zip](https://drive.google.com/file/d/1CgS0vGp36GdYgh3Ai19_zieFHJh_D-Ji/view?usp=sharing)

## Main Components

- `run_benchmark.py`: benchmark runner, subprocess orchestration, resume logic, artifact validation.
- `pipeline_common.py`: shared runtime contract for run directories, video capture, diagnostics, gate checks, and LTL rollout.
- `clutter_scene_pipeline.py`: tabletop clutter pipeline implementation.
- `empty_scene_pipeline.py`, `pinch_point_pipeline.py`, `cabinet_clutter_pipeline.py`: alternative task-generation pipelines.
- `curation/`: post-run scene repair and dataset curation workflow. See [curation/README.md](./curation/README.md).

## Execution Model

At a high level, a benchmark run proceeds as:

1. `run_benchmark.py` selects the requested scenes and starts one subprocess per scene.
2. The scene subprocess loads the pipeline implementation and enters `pipeline_common.py`.
3. The pipeline resolves the support surface, builds the task object sets, places the clutter objects, mounts the robot, runs gate checks and LTL rollout, and saves the artifacts.
4. `run_benchmark.py` validates that the expected artifacts were produced before marking the scene successful.

This subprocess-per-scene model is intentional. It avoids long-lived Isaac Sim / renderer state accumulation and makes failed scenes easier to isolate and rerun.

## BDDL and LTL Generation Flow

The tabletop clutter pipeline does not rely on one fixed, generic BDDL problem shared by all scenes.

Instead, each run generates a task-specific BDDL problem and a matching `ltl_safety.json` at runtime:

1. `clutter_scene_pipeline.py` calls `generate_clutter_activity()` in `omnigibson/utils/bddl_generator.py`.
2. The generator selects the target / fragile / clutter synsets for that run, builds the BDDL object list, writes the initial placement predicates, and constructs the task goal.
3. The same generator also creates a matching `ltl_safety.json` describing the generated safety propositions and the combined LTL formula.
4. Both files are written into the BEHAVIOR `activity_definitions/<activity_name>/` directory for the current generated activity name. (Currently the activity name for `tabletop clutter` is `auto_clutter_on_<scene_model>`)
5. `pipeline_common.py` refreshes the activity cache and launches `BehaviorTask` with that `activity_name`.

For the tabletop clutter task, the generated BDDL currently has:

- an object list derived from the selected target / fragile / clutter synsets,
- `ontop`-style initial predicates on the chosen support furniture,
- a `grasped(agent, target)` task goal for the selected target object.

The BDDL goal and the LTL safety logic are separate:

- the BDDL goal defines task completion,
- `ltl_safety.json` defines safety constraints monitored during rollout.

## Artifact Contract

A successful scene run is expected to produce, at minimum:

- `diagnostics.jsonl`: records outcome signals such as gate results, LTL status, density, and active-object summaries.
- `scene_ep1.json`: frozen scene snapshot used for replay, rerendering, and later evaluation.
- `stdout.log`: preserves the runtime trace for debugging.
- `rollout_ep1.mp4`: canonical human-reviewable video.

## Typical Commands

Minimal single-scene benchmark run:

```bash
conda activate behavior

python -m sentinel.task_generation.run_benchmark \
  --pipeline table \
  --scenes hall_conference_large \
  --episodes 1 \
  --steps 300 \
  --density medium \
  --timeout 1800
```

Run the same scene with scene-level runtime overrides (typically used for scene-level repair and review):

```bash
conda activate behavior

python -m sentinel.task_generation.run_benchmark \
  --pipeline table \
  --scenes hall_conference_large \
  --episodes 1 \
  --steps 300 \
  --density medium \
  --timeout 1800 \
  --curation-manifest <MANIFEST_PATH.json>
```

### Density Control

`--density` (runtime arg `--clutter-density` inside the pipeline) is an object-count preset, not a direct surface-coverage metric.

For the tabletop clutter pipeline, the current presets are defined in `omnigibson/utils/bddl_generator.py`:


| level    | target | fragile | clutter | nominal total |
| -------- | ------ | ------- | ------- | ------------- |
| `low`    | 1      | 2       | 1       | 4             |
| `medium` | 1      | 4       | 2       | 7             |
| `high`   | 1      | 6       | 4       | 11            |
| `ultra`  | 1      | 8       | 6       | 15            |


This preset is only the requested starting budget. The actual number of objects that survive into the final scene is determined in multiple stages:

1. `generate_clutter_activity()` selects the nominal target / fragile / clutter counts from the preset.
2. If an estimated support area is available, the generator greedily trims the object pool before writing BDDL.
3. During scene execution, the pack solver may still cull objects if the scene cannot be packed or validated stably.

As a result, the final saved scene should be interpreted using both:

- the requested density preset, and
- the realized active objects that remain after packing and validation

In practice, the most useful signals to inspect are:

- `selection` in `diagnostics.jsonl`
- `active_object_summary` in `diagnostics.jsonl`
- the final active object count visible in the saved scene snapshot and rollout video

### Gate and LTL Checks

The pipeline uses two different validation layers:

#### Gate checks

The gate is a pre-rollout structural sanity check. A scene passes the shared gate only if:

- the robot and target poses are finite,
- the robot base remains close to the floor plane,
- the selected robot mount pose is collision-free,
- the target lies inside a valid reach band,
- the pipeline-specific extra gate checks also pass.

For the tabletop clutter pipeline, the extra gate checks currently include:

- pack integrity after placement,
- and, when enabled by curation settings, resident support-object stability checks.

If `--strict-gate` is enabled, a gate failure aborts the episode before the normal rollout path continues.

#### LTL safety rollout

After the gate, the pipeline runs a rollout monitored by `TaskLTLMonitor`.

For the tabletop clutter generator, the default generated LTL currently monitors high-level object safety properties such as:

- whether any fragile object has been dropped,
- whether all fragile objects remain upright,
- whether the target has been dropped,
- whether the target remains upright.

The generated `combined_ltl` is written into `ltl_safety.json`, then loaded by `BehaviorTask` / `TaskLTLMonitor` and evaluated step by step during rollout.

At a high level:

- the gate answers "Is this scene structurally valid enough to start?",
- the LTL rollout answers "Does the scene remain safe and semantically intact over time?"

### Curation Manifest

Scene curation can override the requested `clutter_density` at runtime through a curation manifest. This is useful when a specific scene needs a different density target, support surface, robot placement, or camera setup in order to become stable and reviewable.

The manifest does not bypass the normal packing and validation logic. Even if a scene requests `high`, the final realized scene can still end up with fewer surviving objects if area limits, packing retries, or validation checks force additional culling.

For the full curation workflow and a real reference manifest, see:

- [curation/README.md](./curation/README.md)
- `sentinel/task_generation/curation/tabletop-cluttered_env-curation_manifest.json`

## Extending the Pipeline

The current codebase is organized so that:

- `task_generation/` remains the generic benchmark and scene-generation mainline.
- `task_generation/curation/` contains the direct scene-repair workflow.
- `utils/` contains reusable runtime helpers such as workspace geometry, pack placement, robot edge alignment, retry logic, and LTL utilities.

If you want to add more scenes, new tabletop task variants, or new benchmark repair recipes, the recommended path is:

1. keep the generator core in this directory generic,
2. add scene- or benchmark-specific overrides through a curation manifest when needed,
3. use the curation workflow only for post-generation repair and review.

## Related Documentation

- Current curation workflow: [curation/README.md](./curation/README.md)
