# Empty scene

## What it does

Starts from a bare Scene (floor plane only), spawns a randomized support surface from `placeable.json` plus task objects via the env config `objects` list (grasp_task_demo pattern), then runs one of the standard table tasks (clutter / stack / transfer) on top. Every per-episode random choice — surface model, target, fragile, clutter — is drawn from the same pools as the in-scene pipelines, so a single run produces broad domain randomization without needing a curated room.

This pipeline does not subclass `BasePipeline`; it has its own argparse + run loop.

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--setup` | (required) | One of `clutter`, `stack`, `transfer` — selects which table task to run on the spawned surface. |
| `--surface-category` | None | Restrict the surface to one category (e.g. `desk`); random from pool if omitted. |
| `--surface-model` | None | Pin a specific surface asset model id. |
| `--activity-name` | auto | Override generated activity name. |
| `--episodes` | 1 | Episodes per run. |
| `--steps` | 5000 | Max LTL rollout steps. |
| `--seed` | 0 | RNG seed. |
| `--mount-gap-m` | 0.10 | Robot-mount gap from table edge. |
| `--jitter-scale` | 0.01 | Pose jitter applied to spawned objects. |
| `--strict-gate` / `--no-strict-gate` | strict | Abort on gate failure. |
| `--save-video` | off | Emit `rollout_ep*.mp4`. |
| `--video-fps` | 30 | Recording frame rate. |
| `--dry-run` | off | Generate BDDL + LTL only. |
| `--batch-size` | all | Episodes per env load (None = all episodes in one batch). |
| `--clutter-density` | `medium` | Clutter setup only: `low` / `medium` / `high` / `ultra`. |
| `--pack-jitter-xy` / `--pack-min-clearance` | None | Clutter setup pack-retry knobs. |
| `--stack-mode` | `same` | Stack setup: `same` / `flat` / `receptacle`. |
| `--stack-height` | `medium` | Stack setup: number of items above the target. |
| `--target-model` / `--stack-model` | None | Stack setup model overrides. |
| `--food-model` / `--source-model` / `--dest-model` | None | Transfer setup model overrides. |
| `--goal-predicate` | None | Transfer setup: `inside` or `ontop`. |
| `--showcase-gui` | off | Open the OmniGibson viewer. |
| `--debug-jsonl` | None | Append diagnostics to a chosen JSONL. |
| `--run-dir` | auto | Override output directory. |

## Run

Clutter on a random surface:

```bash
conda activate behavior

python -m maniguard.task_generation.empty_scene_pipeline \
  --setup clutter --episodes 1 --steps 300 --save-video
```

Stack on a desk specifically:

```bash
python -m maniguard.task_generation.empty_scene_pipeline \
  --setup stack --surface-category desk --stack-height medium \
  --episodes 1 --steps 300 --save-video
```

Food transfer:

```bash
python -m maniguard.task_generation.empty_scene_pipeline \
  --setup transfer --episodes 1 --steps 300 --save-video
```

## Outputs

- `diagnostics.jsonl` — episode, setup, scene/surface, activity name, gate result, LTL outcome, selection, resolved camera views.
- `scene_ep1.json` — frozen Omniverse snapshot, replayable by `og.sim.load()` and the SO-101/Franka teleop bridge's `_build_from_snapshot`.
- `stdout.log` — runtime trace.
- `rollout_ep*.mp4` — rollout video (if `--save-video`).

## Gate checks

The pipeline runs its own inline gate, equivalent to the shared base:

- robot and target poses finite,
- robot base within 3 cm of the floor plane,
- robot mount edge-alignment had no collision hits,
- target inside the reach band (0.20 ≤ planar distance ≤ 1.10 m).

There are no setup-specific extra gate checks (unlike the in-scene clutter / transfer pipelines).

## LTL constraints

Delegated to the same generator as the chosen setup (`generate_clutter_activity` / `generate_stack_activity` / `generate_transfer_activity` in `maniguard.utils.task_spec`). The resulting `ltl_safety.json` is identical in shape to the matching in-scene pipeline; see `clutter.md`, `stack.md`, or `transfer.md` for the constraint set.

## Source

`maniguard/task_generation/empty_scene_pipeline.py`
