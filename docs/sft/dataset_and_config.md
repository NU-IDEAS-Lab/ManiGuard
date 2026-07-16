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

Full recipe: **[Scripted datagen](../data_collection/index.md#scripted-datagen)** (collection
through RAW → LeRobot conversion). The datagen
dataset ships all 4 bench overviews + wrist (5 streams); pick the overview per
family via the policy config's `external_cam`.

## Source 2 — Sim teleop

GELLO / SO-101 teleop demos, re-rendered to joint + 3-cam and exported to a
multitask LeRobot dataset. Full recipe (Stage 1 render → Stage 2 export, the
`render_teleop_to_lerobot.sh` template): **[Sim teleop → LeRobot](../data_collection/index.md#sim-teleop-lerobot)**.

Naming: `<org>/sim-<fam>-30-joint-3cam` (e.g. `<org>/sim-dusty-transfer-30-joint-3cam`).

## Source 3 — Real teleop

Real Franka teleop capture (`.npz`) → LeRobot v2.1 in the DROID **joint**
convention: 8-D state `[joint_position(7), gripper]` + 8-D action, joint-space
throughout (consistent with the sim tracks), fps 15. Full recipe:
**[Real-robot teleop → LeRobot](../data_collection/index.md#real-robot-teleop-lerobot)**.
