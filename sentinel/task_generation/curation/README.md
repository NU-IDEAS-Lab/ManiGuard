# Scene Curation Workflow

This directory contains the post-generation curation workflow used to repair, inspect, replay, and freeze tabletop clutter benchmark scenes.

It is complementary to the main task-generation pipeline in [`../README.md`](../README.md):

- `task_generation/` generates scenes and benchmark artifacts.
- `task_generation/curation/` repairs problematic scenes after a benchmark run already exists.

## File Structure

- `curation_manifest.py`: pure-Python manifest loader and scene-level runtime override layer.
- `replay_curated_scene.py`: replay and rerender tool for frozen scene snapshots.
- `inspect_curated_snapshot.py`: snapshot QA / inspection tool.
- `tabletop-cluttered_env-curation_manifest.json`: full reference manifest used for the `benchmark_20260319_170914` tabletop clutter curation workflow.

## When to Use Which Path

Use `run_benchmark.py` with a curation manifest when the issue is about:

- support surface choice,
- clutter density,
- object packing stability,
- Franka mount position,
- perimeter clearing,
- gate or LTL failures.

Use `replay_curated_scene.py` when the scene geometry is already correct and frozen, but the output video still needs repair, for example:

- bad simulation-view framing,
- missing review-friendly camera angle,
- rerendering a canonical viewer video from an existing `scene_ep*.json`.

Use `inspect_curated_snapshot.py` when you need snapshot-level QA without regenerating the scene.

## Curation Workflow

The standard scene curation workflow is:

1. Inspect the existing artifacts:
   - `diagnostics.jsonl`
   - `stdout.log`
   - `rollout_ep*.mp4`
   - `scene_ep*.json`
2. Decide whether the failure is a generation/runtime issue or a video/reviewability issue.
3. Add or update scene-level overrides in a <curation_manifest.json>.
4. Either rerun the scene through `run_benchmark.py` or replay the frozen snapshot.
5. Review the outputs and keep only the final successful scene folder.

## Typical Commands

Single-scene rerun with runtime overrides:

```bash
conda activate behavior

python OmniGibson/omnigibson/task_generation/run_benchmark.py \
  --pipeline table \
  --scenes grocery_store_half_stocked \
  --episodes 1 \
  --steps 300 \
  --density medium \
  --timeout 1800 \
  --curation-manifest <CURATION_MANIFEST_PATH.json>
```

Snapshot-only replay for video repair:

```bash
conda activate behavior

python OmniGibson/omnigibson/task_generation/curation/replay_curated_scene.py \
  --scene-model <SCENE_NAME> \
  --curation-manifest <CURATION_MANIFEST_PATH.json> \
  --episode 1 \
  --snapshot-only \
  --steps 300 \
  --run-dir <RUN_DIR> \
  --debug-jsonl <RUN_DIR>/diagnostics.jsonl \
  --save-video
```

Inspect a frozen snapshot:

```bash
conda activate behavior

python OmniGibson/omnigibson/task_generation/curation/inspect_curated_snapshot.py \
  --scene-model <SCENE_NAME> \
  --curation-manifest <CURATION_MANIFEST_PATH.json> \
  --episode 1
```

## Manifest Semantics

The curation manifest is a runtime override layer. It is meant to express scene-specific repair recipes without turning the pipeline code itself into hardcoded per-scene logic.

Typical fields include:

- support selection: `support_category`, `support_room`, `surface_name`
- geometry overrides: `surface_bounds_override_xy`, `obstacle_bounds_override_xy`
- robot placement: `preferred_edge`, `mount_base_pose_xyyaw`, `mount_anchor_offset_m`
- packing controls: `clutter_density`, `pack_jitter_xy`, `pack_min_clearance`
- stabilization / clearing: `pin_support_base`, `support_clear_mode`, `perimeter_clear_mode`
- camera controls: `video_viewer_only`, `video_candidate_views`, `video_final_view`
- lifecycle flags: `status`, `repair_mode`, `defer_reason`

## Reference Manifest

`tabletop-cluttered_env-curation_manifest.json` is included as a reference example, not as a starter template.

It documents the full scene-level override recipes used during the `benchmark_20260319_170914` tabletop clutter curation effort. In practice, future curation runs will usually still need a manifest of this kind, because scene repair often requires runtime overrides for support choice, robot placement, packing stability, and review-camera selection.

Use this file to understand:

- what kinds of runtime overrides are supported,
- how scene-specific repair decisions were recorded,
- how a real benchmark curation job was structured end to end.

Do not treat it as the minimal canonical template for new projects.

## Final Scene Folder Contract

For a curated scene to be considered ready for staging or release, the final folder should contain at least:

- `diagnostics.jsonl`
- `scene_ep1.json`
- `stdout.log`
- `rollout_ep1.mp4`

These files together provide:

- provenance and repair evidence,
- a frozen scene snapshot,
- runtime traceability,
- human-reviewable output.

## Related Documentation

- Main task-generation manual: [../README.md](../README.md)
- Legacy kitchen-bar MVP reference: [docs/omnigibson/cluttered_env_scene_generation_pipeline.md](../../../../docs/omnigibson/cluttered_env_scene_generation_pipeline.md)
