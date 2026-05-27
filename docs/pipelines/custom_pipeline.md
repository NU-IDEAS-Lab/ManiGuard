# Add a custom pipeline

A new task-generation pipeline is a subclass of `BasePipeline`
(`maniguard/task_generation/pipeline_common.py`). The base class runs the whole
machine — scene + surface selection, robot mounting, the gate loop, the
LTL-monitored rollout, snapshot freezing, diagnostics, and video — and calls
*your* hooks at the points where a task differs. You implement a handful of
methods; everything in the [Data flow](data_flow.md) four-stage architecture is
handled for you.

## Hooks, mapped to the four stages

Each stage from the [Data flow](data_flow.md) page corresponds to specific
override points:

| Stage (data flow) | Hook | Essential? | What you return / do |
|---|---|---|---|
| **1 — offline pools** | *(no hook)* | only if asset-fit-constrained | Build a committed JSON pool + a `select.py`, as in `utils/stack_pipeline/` etc. Skip it if your roles have no fit constraint (clutter/empty just hand-pick). |
| **2 — selection** | `activity_prefix()` | ✅ | Default activity-name prefix, e.g. `"auto_clutter_on"`. |
| | `select_objects(args, rng)` | ✅ | A dict with `required_area_m2` (drives Stage 3) plus your chosen synsets/models. |
| | `generate_activity(name, support_synset, support_room, args, rng)` | ✅ | The activity spec — `spawn_specs` + the `ltl_safety` dict — usually by delegating to a `generate_*_activity()` builder in `maniguard.utils.task_spec`. |
| | `configure_env(selection)` | optional | Flip macros before env creation (e.g. enable GPU dynamics for liquids). |
| **3 — scene + surface** | *(none)* | — | `BasePipeline.pick_scene_from_placeable` uses your `required_area_m2`; you don't override anything. |
| **4 — placement & rollout** | `identify_objects(ctx)` | ✅ | Group the spawned objects by role on `ctx` (set `ctx.target_obj`, etc.). |
| | `place_objects(ctx)` | ✅ | Arrange objects on the surface; set `ctx.active_objects` + `ctx._active_object_summary`. |
| | `make_edge_objects(ctx)` | ✅ | A tuple of `EdgeAlignObject` so the robot edge-picker knows what to avoid. |
| | `extra_gate_checks(ctx)` | optional | Extra structural gate beyond the shared one. Default: `True`. |
| | `goal_conditions(ctx)` | optional | Success predicate(s) recorded in diagnostics. |
| | `diagnostics_extra(ctx)` | optional | Extra fields for `diagnostics.jsonl`. |

The base also offers `offline_pack(...)` (pre-compute placements once per session,
as clutter/liquid do) and `scene_family(ctx)` (prompt / goal-region routing).

## Minimal skeleton

```python
from maniguard.task_generation.pipeline_common import BasePipeline, get_spawned_obj
from maniguard.utils.task_spec import estimate_object_set_footprint


class MyPipeline(BasePipeline):

    @classmethod
    def add_args(cls, parser):                 # optional: pipeline-specific CLI flags
        parser.add_argument("--my-flag", type=float, default=0.05)

    def activity_prefix(self):
        return "auto_mytask_on"

    # --- Stage 2: selection --------------------------------------------
    def select_objects(self, args, rng):
        # pick concrete (category, count, model) tuples, then size the surface
        counts = [...]
        return {
            "required_area_m2": estimate_object_set_footprint(counts),
            "target_synset": ...,            # read back in generate_activity
            # ... any other picks your generate_activity needs
        }

    def generate_activity(self, activity_name, support_synset, support_room, args, rng):
        # build spawn_specs + an ltl_safety spec (see the LTL safety page)
        return generate_mytask_activity(
            activity_name, support_synset, support_room,
            rng=rng, pre_selection=args._pre_selection,
        )

    # --- Stage 4: placement --------------------------------------------
    def identify_objects(self, ctx):
        ctx.target_obj = get_spawned_obj(ctx.spawned_objects, ctx.obj_sets["target"][0])

    def place_objects(self, ctx):
        # teleport objects onto ctx.support_obj within ctx.surface_bounds_xy,
        # settle physics, then record:
        ctx.active_objects = {...}            # inst_id -> DatasetObject
        ctx._active_object_summary = [...]    # per-object {inst_id, name, category, role}

    def make_edge_objects(self, ctx):
        from maniguard.utils.franka_edge_align import EdgeAlignObject
        return tuple(
            EdgeAlignObject(name=inst, role=role, position_xy=(x, y))
            for inst, (x, y, role) in ctx._world_positions.items()
        )

    def goal_conditions(self, ctx):           # optional
        return [{"predicate": "grasping", "subject": "robot",
                 "reference": ctx.target_obj.name}]


def main():
    MyPipeline().run()


if __name__ == "__main__":
    main()
```

`ClutterPipeline` (`maniguard/task_generation/clutter_scene_pipeline.py`) is the
canonical worked example — copy it and trim.

## The `EpisodeContext` (`ctx`)

`ctx` is a mutable bag the base and your hooks share across a single episode.
Read from it in your Stage-4 hooks; write the fields the base expects back.
Commonly used:

| Field | Set by | Meaning |
|---|---|---|
| `ctx.args` / `ctx.rng` | base | parsed CLI args / per-episode RNG |
| `ctx.selection` | base | the dict your `select_objects` / `generate_activity` produced |
| `ctx.spawned_objects` | base | `inst_id → DatasetObject` for every pre-spawned task object |
| `ctx.obj_sets` | base | task objects grouped by BDDL role (`target`, `fragile`, …) |
| `ctx.support_obj` / `ctx.surface_bounds_xy` / `ctx.table_top_z` | base | the chosen surface and its world geometry |
| `ctx.og` | base | the `omnigibson` module handle |
| `ctx.target_obj`, `ctx.active_objects`, `ctx._active_object_summary` | **you** | filled in `identify_objects` / `place_objects` |

## CLI, LTL, and the benchmark

- **CLI** — shared flags (`--scene-model`, `--episodes`, `--steps`, `--seed`,
  `--surface-category/-model`, `--clutter-density`, `--strict-gate`,
  `--save-video`, `--dry-run`, `--run-dir`, …) come from
  `make_base_arg_parser`; add your own in `add_args`. The entry point is always
  `MyPipeline().run()`.
- **LTL safety** — the spec you attach in `generate_activity` is run step-by-step
  by a `TaskLTLMonitor` during the rollout, and its outcome is written to
  `diagnostics.jsonl`. Build the spec with a `generate_*_ltl_safety_json()`
  helper in `task_spec`; see [LTL safety system](../foundations/ltl_safety.md).
- **Multi-scene benchmark** — to run your pipeline across scenes via
  `run_benchmark.py`, add it to `_PIPELINE_SCRIPTS` (and, if some scenes lack a
  suitable surface, `_EXCLUDED_SCENES`).

## Checklist

1. Subclass `BasePipeline`; implement `activity_prefix`, `select_objects`,
   `generate_activity`, `identify_objects`, `place_objects`, `make_edge_objects`.
2. Return a realistic `required_area_m2` from `select_objects` (Stage 3 depends on it).
3. Attach an `ltl_safety` spec in `generate_activity` (a `task_spec` builder).
4. *Optional:* `configure_env` (GPU dynamics), an offline pool + `select.py`
   (asset-fit filtering), `add_args` (custom flags), `run_benchmark` registration.
5. Validate with `--dry-run` (BDDL + plan, no simulator), then a short
   `--steps 300 --save-video` run.

## See also

- [Data flow](data_flow.md) — the four stages these hooks plug into.
- [LTL safety system](../foundations/ltl_safety.md) — building the `ltl_safety` spec.
- [Environment layer](../foundations/env_layer.md) — controller presets and the frozen-snapshot runtime.
