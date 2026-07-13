# SFT dataset & data-source configs

The SFT dataset is **model-agnostic**: one LeRobot v2.1 dataset in the
JointController convention feeds any VLA. This page defines that schema and the
three ways to produce it.

## The shared schema (absolute joint)

Every SFT dataset — regardless of source — stores **absolute joint** state and
actions:

```
state    (f32, 8)  [joint_0..6, gripper]          absolute joint config
actions  (f32, 8)  [joint_0..6_target, gripper]   absolute joint target + binary gripper
image_*  (video 256×256×3, 30 fps)                third-person overview(s) + wrist camera
```

- **Absolute**, not delta. `actions` are the next-step absolute joint targets an
  eval-time `JointController` can consume directly — no end-effector / IK.
- A model *may* re-encode these for training (e.g. openpi converts the arm joints
  to per-step deltas internally, then reconstructs to absolute at inference — see
  [openpi SFT](openpi.md)). That is a per-model config detail, **not** a property
  of the dataset.

### Cameras: dataset ships many, policy uses two

The dataset carries several third-person overviews plus the wrist view. A typical
2-camera VLA (pi0.5 / Franka) consumes **one overview + the wrist**; extra
overviews are dropped and any unused image slot is zeroed/masked. Which overview
is used is a per-run choice that **eval must read back from the checkpoint's train
config** to stay in distribution.

| source | overviews shipped | wrist |
|---|---|---|
| scripted datagen | 4 (`image_opposite/left/right/left_shoulder`) | `wrist_image` |
| sim teleop | 2 (`image_left/right`) | `wrist_image` |
| real teleop | 1 external (`exterior_image_1_left`) | `wrist_image_left` |

## LeRobot v2.1 conventions (all sources)

- Datasets are **LeRobot v2.1** (codebase_version `v2.1`). openpi pins a lerobot
  rev that expects v2.1; lerobot ≥ 0.4 writes v3.0 (different parquet layout).
- Conversion runs in a dedicated **`.venv-lerobot`** uv venv, pinned `lerobot<0.4`,
  separate from the `behavior` conda env:
  ```bash
  uv venv --python 3.11 .venv-lerobot
  uv pip install --python .venv-lerobot/bin/python 'lerobot<0.4' h5py pyarrow opencv-python
  ```
- **The `v2.1` git tag is mandatory** on the HF repo — openpi's data loader pulls
  the dataset at that tag. Always push via LeRobot's `push_to_hub(..., tag_version=True)`
  (or `--push-to-hub`); plain `huggingface_hub.upload_folder` does **not** create
  the tag and will make openpi fail with `RevisionNotFoundError`.

---

## Source 1 — Scripted datagen (primary)

The mature 6-family pipeline generates success+safe demos and converts them to
LeRobot v2.1. This is the current main SFT data source.

```
outputs/datagen/<dataset>/<family>/task_*/traj_*   →   to_lerobot   →   datagen-<fam>-v1-joint-5cam
```

Full recipe: **[Sim datagen pipeline](../datagen/pipeline.md)** and
**[RAW → LeRobot conversion](../datagen/lerobot_conversion.md)**. The datagen
dataset ships all 4 bench overviews + wrist (5 streams); pick the overview per
family via the policy config's `external_cam`.

## Source 2 — Sim teleop

GELLO / SO-101 teleop demos, re-rendered joint + 3-cam and exported to a
multitask dataset. Driven by the template `scripts/render_teleop_to_lerobot.sh`
(edit only its CONFIG block per family):

```bash
conda activate behavior
bash scripts/render_teleop_to_lerobot.sh            # both stages
bash scripts/render_teleop_to_lerobot.sh --stage1   # re-render raw teleop → joint+3cam HDF5
bash scripts/render_teleop_to_lerobot.sh --stage2   # rendered HDF5 → LeRobot v2.1 (local build)
```

- **Stage 1** — `maniguard.data.playback --input <raw> --output <rendered>`
  (defaults `--controller joint --cams 3`). Records 8-D joint state/action +
  `image_left`/`image_right`/`wrist_image` at 256×256. Resume-safe; the
  `og.clear()` teardown segfault *after* a complete write is expected and
  harmless (success = non-empty `action` dataset, not exit code).
- **Stage 2** — `maniguard.data.lerobot.multitask_lerobot_export` discovers
  `task_*_traj_*.hdf5`, looks up each task's prompt from
  `<diag-root>/<task>/base/diagnostics.jsonl`, and writes one multitask dataset
  with per-frame `task_index`. Schema is auto-detected from the playback
  fingerprint (no schema flags). The template builds locally (no push).
- **Push** (separate, explicit) — do **not** re-run the exporter with
  `--push-to-hub` on an already-built dataset (`LeRobotDataset.create()` aborts
  with `FileExistsError`). Push the local dataset directly with
  `LeRobotDataset(...).push_to_hub(tag_version=True, push_videos=True, private=True)`.

Naming: `<org>/sim-<fam>-30-joint-3cam` (e.g. `IDEAS-Lab-Northwestern/sim-dusty-transfer-30-joint-3cam`).

## Source 3 — Real teleop

Real Franka teleop capture (`.npz`) → LeRobot v2.1 in the DROID **joint**
convention (openpi's DROID pretrained convention is joint-space):

```bash
.venv-lerobot/bin/python -m maniguard.data.real_teleop.real_teleop_to_droid \
  --input-dir outputs/real_teleop \
  --repo-id maniguard/<task> --prompt "<instruction>" \
  --root outputs/lerobot_datasets/maniguard/<task> \
  --push-to-hub <org>/<task> --hub-private
```

`real_teleop_to_droid` assembles 8-D state `[joint_position(7), gripper]` + 8-D
action `[joint_velocity(7), gripper[t+1]]`, decodes/crops/resizes the cameras,
and (via `--push-to-hub`) creates the required v2.1 tag. This is joint-space
throughout — consistent with the sim tracks. (fps 15; the DROID schema keeps
state columns separate rather than a single `state` column.)
