# Batch Teleop Collection — `collect_teleop_batch.sh`

Wrapper around `sentinel.teleop.so101_franka_teleop` that walks a pipeline run
directory and records one HDF5 per snapshot (ground-truth scene first, then
`scene_ep1.json` .. `scene_epN.json` in numeric order).

## Prerequisites

- `conda activate behavior` (the env with OmniGibson + sentinel).
- SO-101 leader arm powered and reachable on the configured ZMQ port (5557
  by default in `so101_franka_teleop`).
- A scene directory produced by `mug_bowl_scene_pipeline.py` containing
  `scene_ground_truth.json` and/or `scene_ep*.json` (plus matching
  `.meta.json` sidecars — the teleop module reads them for the arm-base
  height).

## Running

```bash
# Use the committed 51-scene set + default output dir
bash scripts/collect_teleop_batch.sh

# Override scene dir and/or output dir
bash scripts/collect_teleop_batch.sh \
    outputs/pipeline_runs/mug_into_bowl_empty_20260419_202321 \
    outputs/my_teleop_run
```

Defaults (see top of the script):

- `SCENE_DIR=outputs/pipeline_runs/mug_into_bowl_empty_20260419_202321`
- `OUT_DIR=outputs/jixing_teleop_hdf5`

The script sorts snapshots numerically so `scene_ep2` comes before
`scene_ep10`. `ground_truth` always runs first. Only files matching
`scene_ep[0-9]+\.json` are picked up — `.meta.json` sidecars are excluded.

## Per-episode workflow

For each snapshot the script:

1. Prints a banner with the episode index (`i/total`), the snapshot path, and
   the target HDF5 path (`<OUT_DIR>/<snapshot_basename>.hdf5`).
2. Launches `python -m sentinel.teleop.so101_franka_teleop --snapshot ...
   --output-hdf5 ... --only-successes`.
3. Inside teleop, drive the SO-101 leader. Hotkeys:
   - `S` — toggle the success flag for this episode (required; `--only-successes`
     means the HDF5 is written only if the flag is set).
   - `C` — checkpoint the current state.
   - `R` — rollback to the last checkpoint.
   - `Q` — save HDF5 (if success flagged) and exit back to the batch script.
4. After teleop exits, the script inspects the HDF5 size. A threshold of
   `8192 B` distinguishes a real trajectory from an empty
   `DataCollectionWrapper` header (~2–4 KB).
5. Prompts:
   ```
   [Enter]=next / r=retry / s=skip / q=quit  >
   ```
   - **Enter / n** — advance to the next snapshot. If the HDF5 is below the
     threshold, a secondary confirmation (`y/N`) is required and the empty
     file is deleted if you decline.
   - **r** — delete the partial HDF5 and re-run the same snapshot. Use this
     if you dropped the mug, bumped a distractor, or forgot to press `S`.
   - **s** — delete the HDF5 and move to the next snapshot without recording.
   - **q** — abort the batch immediately.

## Output

HDF5s are named after their snapshot:

```
<OUT_DIR>/scene_ground_truth.hdf5
<OUT_DIR>/scene_ep1.hdf5
...
<OUT_DIR>/scene_ep50.hdf5
```

When the batch finishes (or you quit), the script prints a listing of every
HDF5 in `OUT_DIR` with its size in bytes.

## Common issues

- **"No snapshots found"** — the scene dir is wrong or doesn't contain
  `scene_ep*.json` matching the strict `scene_ep[0-9]+\.json$` pattern.
- **Mug slips out of the gripper mid-trajectory** — the default robot config
  uses `grasping_mode: "physical"`, which relies on friction alone. Switch to
  `"assisted"` (virtual attach on closed gripper) in the robot config used by
  `so101_franka_teleop` if this is chronic.
- **Empty HDF5 every time** — you probably didn't press `S` before `Q`.
  `--only-successes` drops the file silently if the success flag is unset.
- **Episode count surprises** — make sure the scene dir has no stray
  `scene_ep*.json` from older runs; the filter excludes `.meta.json` but
  does not dedup across different experiments.
