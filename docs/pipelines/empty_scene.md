# Empty scene

## What it does

Starts from a bare Scene (floor plane only), spawns a randomized support surface from `placeable_surfaces_v1.json` plus task objects via the env config `objects` list (grasp_task_demo pattern), then runs one of the standard table tasks (clutter / stack / transfer) on top. Every per-episode random choice — surface model, target, fragile, clutter — is drawn from the same pools as the in-scene pipelines, so a single run produces broad domain randomization without needing a curated room.

This pipeline does not subclass `BasePipeline`; it has its own argparse + run loop.

## Gate checks

The pipeline runs its own inline gate, equivalent to the shared base:

- robot and target poses finite,
- robot base within 3 cm of the floor plane,
- robot mount edge-alignment had no collision hits,
- target inside the reach band (0.20 ≤ planar distance ≤ 1.10 m).

There are no setup-specific extra gate checks (unlike the in-scene clutter / transfer pipelines).

## LTL constraints

Delegated to the same generator as the chosen setup (`generate_clutter_activity` / `generate_stack_activity` / `generate_transfer_activity` in `maniguard.utils.task_spec`). The resulting `ltl_safety.json` is identical in shape to the matching in-scene pipeline; see `clutter_pickup.md`, `stack_retrieve.md`, or `transfer.md` for the constraint set.

## Source

`maniguard/task_generation/empty_scene_pipeline.py`
