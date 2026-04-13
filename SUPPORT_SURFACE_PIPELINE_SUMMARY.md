# Support Surface Profile Pipeline

## Scope

This document freezes the current support-surface profiling pipeline used by the BEHAVIOR tabletop task-generation stack.

Current frozen assets:

- Canonical profile database:
  - `OmniGibson/omnigibson/task_generation/support_surface_profiles_v1.json`
- Canonical reviewed artifact root:
  - `outputs/support_surface_profiles/catalog_batch_full_20260412_v1`
- Reviewed full-batch artifact archive:
  - `outputs/support_surface_profiles/catalog_batch_full_20260412_v1.zip`

Current reviewed batch status:

- `187/187` assets profiled from `surface_catalog.json`
- `185/187` assets currently marked candidate-for-generation
- manually rejected:
  - `flat_bench/iisaia`
  - `pool_table/atjfhn`

`surface_catalog.json` is batch coverage metadata. It is not the placement geometry source of truth. The placement geometry source of truth is `support_surface_profiles_v1.json`.

`batch_progress.json` is an execution/progress ledger, not the canonical profile database.

Legacy `profile_*` and `batch_smoke_*` output folders under `outputs/support_surface_profiles/` are smoke or spot-check artifacts and are not part of the frozen reviewed batch.

## Pipeline Logic

### 1. Empty-scene asset profiling

Each support asset is profiled in isolation in a minimal empty scene.

The profiler:

1. Spawns exactly one support object.
2. Uses the object world AABB only to define the XY scan envelope.
3. Lays down a regular XY grid over that envelope.
4. Casts one vertical ray per grid cell center from `z = aabb_max_z + 0.15 m` to `z = aabb_min_z - 0.20 m`.
5. Keeps only hits that:
  - belong to the target support object
  - have upward-facing normals with `normal_z >= 0.9`
6. Converts valid hit points into the support object's local root frame.

Frozen profiler defaults:

- `grid_step_m = 0.03`
- `top_band_size_m = 0.02`
- `top_band_min_count_ratio = 0.2`
- `min_component_cells = 4`
- `min_region_area_m2 = 0.01`

### 2. Dominant top-plane selection

The raw ray hits can contain multiple horizontal bands. The pipeline does not trust the highest single hit. It selects a dominant support plane explicitly.

Algorithm:

1. Bucket all local hit heights by `round(z / top_band_size_m)`.
2. Find the densest bucket count.
3. Mark as eligible any bucket with count at least `ceil(max_bucket_count * top_band_min_count_ratio)`.
4. Choose the highest eligible bucket.
5. Use the mean height of that bucket as `top_plane_z_local`.
6. Build a binary `top_mask` by keeping all raw hits within `+- top_band_size_m` of `top_plane_z_local`.

This is why the pipeline is robust to shelves, rails, decorative lips, and multi-level geometry. It chooses the highest sufficiently supported horizontal band, not simply the highest point in the mesh or the AABB center.

### 3. Sub-rectangle decomposition

The selected top-plane occupancy mask is converted into usable placement rectangles in the support local XY frame.

Algorithm:

1. Remove small connected components with fewer than `4` cells.
2. Compute connected-component labels for bookkeeping.
3. Repeatedly extract the largest axis-aligned all-true rectangle from the remaining mask.
4. Zero out that rectangle and continue until no valid rectangle remains.
5. Drop rectangles with area below `0.01 m^2`.

This produces one or more usable rectangles:

- a rounded rectangular table yields a large inner rectangle
- an L-shaped table can yield multiple rectangles
- narrow strips remain narrow strips instead of being collapsed into a fake centered box

The output shape model is intentionally conservative:

- rectangle-only
- axis-aligned in the support local frame
- geometry-first, not visually prettified

### 4. Reachability annotation

Each local usable region is also annotated with a lightweight Franka-mounted reachability diagnostic.

Current frozen reachability parameters:

- reach distance range: `[0.45, 1.10] m`
- mount gap: `0.03 m`
- robot radius: `0.24 m`
- robot half extent: `(0.15, 0.15) m`
- edge margin: `0.08 m`

This annotation is currently diagnostic metadata. It is not yet the only hard gate for candidate inclusion.

### 5. Review artifact generation

Each profiled asset saves:

- binary raw-hit mask
- binary top-plane mask
- region overlay mask
- top-down simulation render
- oblique simulation render
- top-down render with highlighted usable regions
- oblique render with highlighted usable regions

These artifacts support manual review and rejection overrides without rerunning profiling logic.

## JSON Database Structure

The frozen profile database is:

- `OmniGibson/omnigibson/task_generation/support_surface_profiles_v1.json`

Top-level structure:

```json
{
  "version": "support_surface_profiles_v1",
  "frame_convention": {...},
  "generator_defaults": {...},
  "profiles": {
    "<category>": {
      "<model>": {
        "...": "profile entry"
      }
    }
  }
}
```

### Frame convention

All usable regions are expressed in the support object's local root frame.

- XY coordinates are local support-frame coordinates in meters.
- `top_plane_z_local` is in the same local frame.
- Quaternion convention is `xyzw`.

This means region geometry is asset-level and pose-independent. At task time the regions can be transformed into the current scene instance.

### Per-asset entry

Each `profiles[category][model]` entry stores:

- identity:
  - `category`
  - `model`
- plane geometry:
  - `top_plane_z_local`
- local AABB summary:
  - `aabb_xy_local.xy_min`
  - `aabb_xy_local.xy_max`
  - `aabb_area_m2`
- occupancy summary:
  - `occupancy_area_m2`
  - `effective_area_m2`
- usable region list:
  - `usable_regions`
- generation gate:
  - `candidate_for_generation`
  - `exclusion_reasons`
  - `review_status`
- reachability diagnostics:
  - `reachability.franka_mounted`
- review evidence:
  - `review_artifacts`
  - `review_checks`
- profiling diagnostics:
  - `diagnostics`
- provenance:
  - `provenance`

### Usable region entry

Each entry in `usable_regions` stores:

- `region_id`
- `shape`
- `xy_min`
- `xy_max`
- `span_xy_m`
- `area_m2`
- `coverage_ratio`
- `cell_count`
- `component_ids`
- `confidence`
- `source`
- `reachable`
- `reachable_edge_labels`

Meaning of the main fields:

- `region_id`:
  - unique within one asset entry
  - formatted as `region_00`, `region_01`, ...
  - deterministic for a frozen profile entry
  - not a global cross-asset ID
- `xy_min`, `xy_max`:
  - local-frame rectangle bounds in meters
- `span_xy_m`:
  - local-frame rectangle width and depth
- `area_m2`:
  - rectangle area in square meters
- `component_ids`:
  - which connected top-plane mask component(s) this rectangle belongs to

### Region indexing stability

Within a frozen JSON file, each region can be referenced stably by:

- `(category, model, region_id)`

That triple is the intended stable lookup key for downstream generation code.

Important nuance:

- region IDs are deterministic for the current frozen algorithm, parameters, and asset geometry
- if the profiling algorithm or its thresholds change and the asset is re-profiled, the rectangle decomposition may change and IDs may be renumbered

So:

- stable for this frozen database version: yes
- guaranteed stable across future profiler revisions: no

### What is and is not explicitly stored

Explicitly stored:

- local rectangle bounds
- local rectangle spans
- local rectangle areas
- local top-plane height
- local XY AABB bounds
- asset-level diagnostic ratios and counts

Implicitly derivable:

- local AABB XY span from `aabb_xy_local`
- largest usable region from `usable_regions`

Not explicitly stored in `v1`:

- full local AABB height extent of the asset
- arbitrary polygonal support shapes
- scene-level occupancy subtraction from sinks, stoves, or clutter already placed on the surface

## Task-Time Integration

The main task-generation pipeline already consumes this database.

Current integration flow:

1. Look up the profile entry for the chosen support asset.
2. Estimate usable support area from the profiled regions before sim scene selection.
3. Prefer support assets whose profiled regions satisfy the required object-set footprint.
4. After the support is instantiated in scene, transform local usable regions into world coordinates.
5. Choose a working placement region from the profiled set.
6. Fall back to asset AABB bounds only if:
  - no profile exists, or
  - no profiled region survives downstream constraints

Current region selection policy:

- filter by required area and minimum XY span
- prefer the largest eligible region
- optional RNG can randomize among eligible regions

The key split is:

- asset profiling answers: "where is the real support geometry?"
- task generation answers: "which recorded region should this task use?"

## Reproduction Commands

Use the `behavior` Python environment and run the module commands from `OmniGibson/`.

### 1. Single-asset profiling smoke or debug

Example:

```bash
cd /home/yiyanpeng/project/SENTINEL-Lite/OmniGibson
/home/yiyanpeng/miniconda3/envs/behavior/bin/python -m omnigibson.task_generation.support_surface_profiler \
  --category coffee_table \
  --model fqluyq \
  --grid-step-m 0.03 \
  --run-dir /home/yiyanpeng/project/SENTINEL-Lite/outputs/support_surface_profiles/profile_coffee_table_20260412_010334 \
  --output-json /home/yiyanpeng/project/SENTINEL-Lite/OmniGibson/omnigibson/task_generation/support_surface_profiles_v1.json \
  --overwrite
```

Useful variants:

- add `--showcase-gui` for interactive visual debug
- swap `--category` and `--model` to inspect a different asset
- point `--run-dir` to a fresh folder if the run should be kept separately

### 2. Single-category batch profiling

Example:

```bash
cd /home/yiyanpeng/project/SENTINEL-Lite/OmniGibson
/home/yiyanpeng/miniconda3/envs/behavior/bin/python -m omnigibson.task_generation.support_surface_batch_runner \
  --categories coffee_table \
  --order alpha \
  --grid-step-m 0.03 \
  --batch-root /home/yiyanpeng/project/SENTINEL-Lite/outputs/support_surface_profiles/catalog_batch_full_20260412_v1 \
  --progress-json /home/yiyanpeng/project/SENTINEL-Lite/outputs/support_surface_profiles/catalog_batch_full_20260412_v1/batch_progress.json \
  --output-json /home/yiyanpeng/project/SENTINEL-Lite/OmniGibson/omnigibson/task_generation/support_surface_profiles_v1.json \
  --overwrite
```

This writes:

- per-asset artifacts under the selected category folder
- category `summary.jsonl`
- batch-level `batch_progress.json`
- milestone refreshes in `SURFACE_PROFILE_MILESTONES.md`

### 3. Full reviewed batch generation

Frozen full-batch command pattern:

```bash
cd /home/yiyanpeng/project/SENTINEL-Lite/OmniGibson
/home/yiyanpeng/miniconda3/envs/behavior/bin/python -m omnigibson.task_generation.support_surface_batch_runner \
  --order size_asc \
  --grid-step-m 0.03 \
  --batch-root /home/yiyanpeng/project/SENTINEL-Lite/outputs/support_surface_profiles/catalog_batch_full_20260412_v1 \
  --progress-json /home/yiyanpeng/project/SENTINEL-Lite/outputs/support_surface_profiles/catalog_batch_full_20260412_v1/batch_progress.json \
  --output-json /home/yiyanpeng/project/SENTINEL-Lite/OmniGibson/omnigibson/task_generation/support_surface_profiles_v1.json \
  --overwrite
```

This is the command family that produced the canonical reviewed artifact tree under:

- `outputs/support_surface_profiles/catalog_batch_full_20260412_v1`

Manual review overrides were applied afterward for:

- `flat_bench/iisaia`
- `pool_table/atjfhn`

## Code Changes In This Freeze

Main new or modified files:

- `OmniGibson/omnigibson/task_generation/support_surface_profiles.py`
  - added the profile document schema
  - added geometry helpers, rectangle decomposition, reachability checks, and task-time region selection helpers
- `OmniGibson/omnigibson/task_generation/support_surface_profiler.py`
  - added empty-scene profiling with raycast-based surface discovery
  - added review-plot generation and simulation highlight renders
  - writes `profile_summary.json` per asset and updates the main JSON database
- `OmniGibson/omnigibson/task_generation/support_surface_batch_runner.py`
  - added category-batch orchestration
  - added `batch_progress.json`
  - updates `SURFACE_PROFILE_MILESTONES.md`
- `OmniGibson/omnigibson/task_generation/pipeline_common.py`
  - integrated profile-aware surface-area estimation
  - integrated profile lookup, world-region transforms, and profile-region selection at task time
  - records support-profile diagnostics in pipeline logs
- `OmniGibson/tests/test_support_surface_profiles.py`
  - added unit tests for schema IO, dominant-plane selection, mask decomposition, reachability, and region selection

Manual reviewed overrides currently frozen into the database:

- `flat_bench/iisaia` -> rejected as non-tabletop
- `pool_table/atjfhn` -> rejected as recessed non-tabletop for mainline reproducibility

## Notes For Future Revisions

The next layer should not replace this asset-level database. It should build on top of it.

The natural next step is scene-level filtering:

- subtract occupied or blocked zones on top of the support
- subtract sink / stove / fixture cutouts
- apply embodiment-specific reach and collision constraints in-scene

That is a second-stage scene-instance filter, not a replacement for the current asset-local source of truth.