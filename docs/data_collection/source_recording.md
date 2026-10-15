# High-fidelity source recording

ManiGuard can record source observations and simulator state independently of a
policy's image processor. The source profile uses five 640 × 640 RGB views and
stores frames once as JPEG quality 95 with 4:4:4 chroma sampling.
`configs/render/high_fidelity.yaml` is the standard configuration: RTX realtime
lighting, native-resolution DLAA and ray reconstruction. The built-in defaults
select the same configuration. Policy servers receive
the selected source views and apply their checkpoint's native image transforms.

High-fidelity rendering is the default for new simulation runs. Use
`--source-profile legacy` (or `source_profile: legacy` in YAML) to select the
previous rendering path. Explicit YAML `null` also selects legacy; an omitted
setting selects high fidelity. Old evaluation results and released demonstration
files keep their original pixels and provenance. New rendering is a different
experimental visual condition.

Rendering and storage are separate choices. Ordinary evaluation can render for
the policy with no video, write streaming MP4 previews, record full source
episodes, or do both. Full source storage is enabled only by
`recording_output_dir`; choosing high fidelity alone does not create it.

## Rendering configuration

The built-in selector `high_fidelity` works from any working directory.
`configs/render/high_fidelity.yaml` provides the equivalent editable profile.
Pass a custom YAML path through `source_profile` / `--source-profile` for
evaluation, datagen, benchmark generation, simulated teleop and playback.
Camera calibration, source storage, action/state/GT records and physics/render
synchronization use shared modules; each entry point supplies its own actions.

The realtime settings adapt the IsaacLab 4.5 quality preset used by
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/266130a3ec41ba9b20b0e5648d2f3542f219c06c/env/environment/base_env.py),
with global illumination, reflections, translucency, ambient occlusion and shadows
retained. DLAA does not downsample the 640-square source. Motion blur, depth of
field and generated frames are disabled.

`glass_shadows: exclude_omniglass` excludes meshes bound entirely to the
`OmniGlass.mdl` / `OmniGlass` shader from casting shadows. This addresses opaque
shadows obscuring objects inside glass in RTX realtime. The override is authored
only on the glass primvars in the existing runtime session layer; asset files
and physical properties are unchanged. Original property opinions are restored
when capture is closed or replaced. Mixed material-subset meshes are left
unchanged. The recorded render report lists affected meshes, materials and
original values; playback applies its recorded profile again.

This is a visibility/throughput tradeoff: excluded glass does not cast physically
accurate glass shadows or caustics. The RTX flag can also affect segmentation
annotations; the source recorder provides RGB, not validated segmentation GT.
Use `glass_shadows: native` to keep original shadows. No family or task ID
controls the override.

DLAA uses `render_deliveries: 4` actual SDG deliveries per observation, allowing
propagation and temporal reconstruction while physics remains frozen. It does not
use PT's SPP accumulation or claim pixel-identical replay: DLAA and ray
reconstruction maintain rendering history, and disocclusions/motion may differ
from a long-converged static render. The per-frame record retains the actual render
pass count; assess motion quality when changing this setting.

Path Tracing is no longer a supported rendering route. Existing source episodes
remain readable and exportable, including episodes recorded with PT. Re-rendering
such an episode requires an explicit supported `--source-profile`; it never
silently substitutes a different renderer.

## Evaluation

Ordinary video-only evaluation uses the default quality:

```bash
python -m maniguard.eval.benchmark --config configs/eval/jar_transport_joint.yaml --save-video
# Previous rendering, with its original evaluation source resolution:
python -m maniguard.eval.benchmark --config configs/eval/jar_transport_joint.yaml --source-profile legacy
```

Use `--no-save-video --no-save-wrist-video` to disable both video outputs. Video
encoding streams frames at the control rate rather than buffering an episode.
Policy adapters still apply each checkpoint's own resizing/normalization.

To also save full source episodes, set these evaluation YAML fields:

```yaml
source_profile: high_fidelity
save_video: false
recording_output_dir: outputs/recordings/evaluation
recording_max_bytes: 5368709120  # Per-attempt limit, not a campaign budget.
```

Use the existing `python -m maniguard.eval.benchmark --config ...` entry point.
The source profile configures 60 Hz internal rendering with the configured
20 or 30 Hz control rate. A command is submitted once per control interval;
waiting for rendering does not advance simulated time. Image resolution and
profile conflicts are errors. Missing source cameras are errors.

Source capture owns render delivery; Replicator's automatic capture is paused
independently of physics. In headless sessions, GUI preview render products are disabled;
the five sensor render products remain active. Interactive previews are unchanged.

`recording_provenance` optionally names a JSON file describing the benchmark and
checkpoint revisions. Episodes also record the Python source digest, package
versions, configuration, camera calibration, controllers, source scene and
original diagnostics. Keep the simulator assets at the recorded version.

## Scripted demonstration collection

Both `python -m maniguard.data.datagen.driver` and
`python -m maniguard.data.datagen.sweep` default to high fidelity; pass
`--source-profile legacy` for the original `traj_*` RAW writer. Use a new dataset
name/output directory when changing profiles: resume rejects mixed rendering or
recording formats. The sweep forwards the selection to each worker.
Use `--recording-max-bytes` to
set the per-attempt limit. The new mode retains completed failed attempts as well
as accepted demonstrations, in `attempt_<draw index>_g<grasp id>` directories. Episode result
fields retain the demonstration acceptance decision and `termination_reason`
(success, safety stop, timeout, planning failure, or another acceptance failure).
Early failures also retain `failure_detail`. Completeness is a separate property.

Source capture includes final settling. Lid attachment settling that submits no
new command is marked `issued=false`; its command values describe the previously
active command, not a new policy proposal. Hooks may change attachment or physical
parameters instantaneously. Changed states are retained as `boundaries` records
with source images, physical state and a checkpoint at the same simulation time.
`commands/source_boundary_id` references the actual pre-command state, or -1 when
the regular source observation applies. The preceding observation and its GT stay
unchanged. Boundary GT is explicitly unavailable: no extra monitor transition is
invented. Consumers constructing causal examples must account for these updates.

A failed monitor reading is recorded as invalid. A failed recorder, missing frame,
or unmatched final observation cannot become a complete source episode. Files
left after interruption retain `partial` status; HDF5 recovery after a hard kill
is not guaranteed. Collection stops when its configured output limit is exceeded.

Accepted high-fidelity demonstrations are exported explicitly before using the
existing review or LeRobot conversion tools:

```bash
# Input layout: COLLECTION/family/task_*/attempt_*/episode.json
python -m maniguard.data.recording.export \
  outputs/datagen/COLLECTION outputs/datagen/COLLECTION_export --accepted-datagen
```

Only complete, accepted scripted demonstrations enter the `family/task/traj_*`
export; completed failed attempts remain in the source collection. Repeating an
export checks source identity and export options. It does not overwrite conflicting
outputs. The RAW reader reports an actionable export instruction if given source
attempts directly. Install the project's `conversion` extra in the conversion
environment to use the pinned LeRobot converter.

## Source episode layout

```text
episode.json         # Schema, completion status, counts, provenance, result
trajectory.hdf5      # Time, five RGB streams, robot/objects, actions, safety GT
events.jsonl         # Proposals, attempted actions, scripted boundaries
snapshots.hdf5       # Per-observation scene/controller/particle state and checkpoint
task_scene.json      # Initial scene description, references external simulator assets
diagnostics.jsonl    # Original task spec, cameras and appearance perturbations
```

For T control intervals there are T+1 observations. Row i connects observation i
to observation i+1. `commands/issued` distinguishes newly issued commands from
passive holds. `raw`, `transformed` and `applied` retain the executor's distinct
representations; no next-achieved state is substituted for a command. Proposal
events retain the complete returned chunk and the length selected for execution.
A proposal tail is not labelled as executed. Proposal events also retain controller
action bounds (`"inf"` / `"-inf"` encode unbounded dimensions in standard JSON).
`joint_groups` maps arm and gripper state indices independently of controller-action
indices. `proposition_bindings` records the objects actually resolved by the monitor.
Per-observation `constraint_records` retain each safety clause's bad-prefix verdict,
checked against the combined task monitor. They support offline attribution without
re-running simulation; they do not convert every false AP into a violation.

RGB is stored in `images/<camera>/bytes` with N+1 byte offsets, allowing direct
frame reads without seeking through an MP4. Cameras are `image_left`,
`image_right`, `image_left_shoulder`, `image_opposite` and `wrist_image`.
The numeric groups are `time`, `robot`, `objects`, `commands` and `safety`.
Ragged metadata, particles and checkpoints use compressed typed MessagePack
records with offsets; they do not use pickle. AP names, object names and joint
names are stored in the manifest. Episode UUIDs are independent of per-task attempt directory names. AP truth values have validity masks. AP false
is not universally equivalent to a violation.

```python
from maniguard.data.recording import EpisodeReader

with EpisodeReader("outputs/recordings/evaluation/<episode>") as episode:
    episode.validate(decode_images=True)
    rgb = episode.read_rgb("wrist_image", 1)
    observation = episode.read_observation(1)
    transition = episode.read_transition(0)
    scene_state, checkpoint = episode.read_snapshot(1)
```

The reader does not need a running simulator or policy. Monitor values are
supervision/diagnostic records; ManiGuard does not construct model questions,
training splits or model-specific labels.

## Derived exports and playback

An offline export streams selected source frames into MP4 and numeric HDF5:

```bash
python -m maniguard.data.recording.export \
  outputs/recordings/evaluation/EPISODE outputs/exports/task_0000/traj_000
```

The export declares both state and action semantics. Joint-only LeRobot conversion
rejects EEF states even when they also have eight components.

This export contains T pre-action frames and T commands. The terminal observation
remains in the source. MP4 uses H.264 CRF 14 and is a derived representation, not
the JPEG source. The output records FPS, resolution and action semantics.
`--cameras` selects views and `--action-source` selects the command representation.
No resizing is implicit. The existing joint LeRobot converter consumes the exported
trajectory layout, derives dimensions/FPS from metadata, and rejects mixed image
or action contracts. It requires all five views and an 8-D joint schema. Historical
RAW files retain their original next-achieved action convention.
Exports select boundary images/state when present and retain `command_issued`,
source/target observation IDs, source boundary ID and source time through LeRobot conversion.

Run conversion with the pinned conversion dependencies:

```bash
python3.12 -m venv .venv-conversion
.venv-conversion/bin/python -m pip install -e '.[conversion]'
.venv-conversion/bin/python -m maniguard.data.datagen.to_lerobot \
  --dataset /absolute/path/to/exports --family jar_transport \
  --out-root /absolute/path/to/lerobot --repo-id local/jar
```

Arrange exports as `exports/<family>/task_XXXX/traj_XXX/`. The output is
`lerobot/<family>/`, with Parquet numeric data and five MP4 video streams.
The conversion dependency is LeRobot 0.3.3; a different installed version is
rejected before conversion starts.

Recorded-state rendering requires the simulator, matching assets, scene objects
and controllers:

```bash
python -m maniguard.data.recording.replay EPISODE outputs/previews/playback \
  --observations 0 1 2
```

This restores physical snapshots and renders without advancing physics. It is
not a claim of bitwise policy replay: external policy RNG is not captured, absolute
simulator time is not rewound, and renderer history is not restored. Camera
poses and source timestamps remain identified separately from playback time.
Checkpoints also retain monitor memory and simulator-process RNG for controlled
command replay in a matching environment.

Benchmark previews go to a separate staging directory:

```bash
python -m maniguard.data.bench_builder.render \
  --task-dir outputs/lerobot_datasets/maniguard-bench/jar_transport/task_0000/base \
  --source-profile configs/render/high_fidelity.yaml \
  --output outputs/previews/jar --frames 2 --mode frozen
```

This preserves the source scene and diagnostics files. `idle_step` is a stability
preview, whereas `frozen` only renders the saved setup. Neither is a policy score.
Task-generation pipelines accept the same `--source-profile` for five-view previews.

## Simulated teleop and HDF5 playback

GELLO/SO101 simulation entrypoints accept `--source-profile high_fidelity|legacy`
or a YAML path. Input-device controls and state/action recording are unchanged;
interactive viewports remain active. Real-camera capture is independent.

`python -m maniguard.data.playback --input TELEOP.hdf5 --output RENDERED.hdf5`
uses high-fidelity rendering, retaining `--cams 2|3` export keys and the selected
controller convention. `--source-profile legacy` retains the previous render
path; `--resolution` must agree with an explicitly selected high-fidelity profile.
The input recording cannot be used as the output. Upstream contact-enabled
playback retains its 1000 Hz restoration clock; render-only captures do not add
physical steps. Source-snapshot playback instead defaults to its recorded profile
and records both original and replacement profiles when explicitly overridden.

## Runtime and migration

The rendering route is tested on the repository's pinned OmniGibson / Isaac Sim
4.5 stack. PyAV is a runtime dependency for streaming video output. On headless
machines set `OMNIGIBSON_HEADLESS=1` and a valid `VK_ICD_FILENAMES`. On hosts with
many CPU cores, bounded `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`
(for example 4) avoid excessive startup threads; tune these to the host.

Historical evaluation YAMLs with `camera_resolution: 256` must either select
`source_profile: legacy` or remove that source-size override to adopt the new
default. High fidelity fails explicitly on conflicting resolution, missing cameras
or unavailable render delivery; it does not silently fall back. New source records
include the resolved profile. Benchmark task JSON/specifications, old HF previews,
released demonstrations and checkpoints are not rewritten by installing this code.
