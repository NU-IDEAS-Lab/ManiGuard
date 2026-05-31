# Fine-tuning pi0.5 on OmniGibson sim teleop via openpi

End-to-end recipe for SFT of a pi0.5 VLA on **OmniGibson teleop data**
using openpi's native JAX trainer. Companion to
`docs/openpi_real_teleop_sft.md`; the two share most plumbing but
choose **different LeRobot schemas** because of action/state convention
mismatches.

**Key difference from the real-teleop path**: sim data uses LIBERO schema
(EEF-delta actions), not DROID schema (joint-velocity actions). We don't
have joint-position observations recorded in our sim teleop, and re-playing
to extract them is more pain than benefit when the LIBERO path already
matches our data 1:1.

!!! tip "Pick controller / action / eval first"
    This page is the **delta-EEF (LIBERO)** track. For how that choice ties back
    to data collection and the matching **eval controller** (`osc` vs
    `joint_position_impedance` + `ik_eef_to_joint`), see
    [Controller, data, action & eval](sft/end_to_end.md).

---

## 1. Overview

**Input**: rendered HDF5 from `maniguard.data.playback` over OmniGibson
teleop, with the standard OmniGibson schema:

```
data/demo_0/obs/image        (N+1, 256, 256, 3) uint8   third-person view
data/demo_0/obs/wrist_image  (N+1, 256, 256, 3) uint8   wrist camera
data/demo_0/obs/state        (N+1, 8)            f32    [eef_pos(3), axisangle(3), grip_L, grip_R]
data/demo_0/action           (N,   7)            f32    [Δpos(3), Δrot(3), gripper_sign(1)]
```

**Output**: LoRA-finetuned pi0.5 ckpt evaluable in sim via
`maniguard/serve/openpi_native.py` + the [eval benchmark](one_machine_pro6000_eval.md).

**Pipeline**:
```
GELLO teleop -> Stage1 playback (joint+3cam .hdf5) -> Stage2 multitask_lerobot_export -> HF private repo
   (§4, template scripts/render_teleop_to_lerobot.sh)                                          |
cloud: pull dataset + pi05_base ckpt from GCS  ->  openpi train.py  ->  ckpt
```

(The §5+ sections below describe the older single-prompt 2-cam EEF/LIBERO path
for `mug-into-bowl`. The current joint + 3-cam multitask flow is §4 above; SFT
on it is driven by `tools/openpi_sft/` — see
[openpi_sft_maniguard_jointctr_pipeline.md](openpi_sft_maniguard_jointctr_pipeline.md).)

---

## 2. Why LIBERO schema (not DROID)

`LeRobotLiberoDataConfig`'s repack expects exactly the LeRobot column
names our `maniguard/data/lerobot/lerobot_export.py` already writes:

| LIBERO key | Our column | Match? |
|---|---|---|
| `observation/image` ← `image` | `image` (video, 256²×3) | ✅ |
| `observation/wrist_image` ← `wrist_image` | `wrist_image` (video, 256²×3) | ✅ |
| `observation/state` ← `state` | `state` (8D) | ✅ |
| `actions` ← `actions` | `actions` (7D EEF delta) | ✅ |
| `prompt` ← `prompt` | task index + tasks.jsonl | ✅ |

State semantics differ slightly (LIBERO uses Euler angles; we use
axisangle) but both are 3D continuous orientation reps that pi0.5 learns
from data via norm_stats. From `pi05_base` warm start the difference is
moot — `pi05_base` has not seen any robot-specific state, so any
3D rotation parameterization is fine.

**DROID schema would require joint_position(7) + joint_velocity(7)** as
new LeRobot columns. We don't record these in sim playback, and adding
them needs a `maniguard/data/playback.py` modification + 39-episode
re-playback (~3-4 h on a 4080). Not worth it unless you want to do
sim+real DROID-format mixed training.

---

## 3. Hardware requirements

Same as `docs/openpi_real_teleop_sft.md` §2 — pi0.5 LoRA finetune from
`pi05_base` fits on **A100 40GB** (frees the optimizer-state cliff that
forces full SFT to A100 80GB). Sim datasets tend to be larger than real
(our mug-into-bowl: 39 eps × ~1300 steps ≈ 51k frames vs real's 6k
frames), so plan for **longer wall clock** at the same step count.

---

## 4. Local data conversion

Two stages — re-render raw teleop into joint + 3-cam HDF5, then export to a
LeRobot v2.1 **multitask** dataset (one prompt per task, per-frame
`task_index`). Both are driven by the reusable template script
**`scripts/render_teleop_to_lerobot.sh`** — to run a new family you edit only its
CONFIG block (paths + repo id), the body is family-agnostic. Live example below
is `jar_transport`.

### 4.1 One-time env setup

Stage 2 runs in a dedicated lerobot uv venv (pinned to the v2.1 codebase),
separate from the `behavior` conda env that Stage 1 (OmniGibson) needs:

```bash
uv venv --python 3.11 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python 'lerobot<0.4' h5py pyarrow opencv-python
```

This is 1:1 with SENTINEL-Lite's `.venv-lerobot` (core packages — lerobot 0.3.3,
av, numpy, pyarrow, h5py — match). `.venv-lerobot/` is gitignored.

### 4.2 The template script

```bash
# scripts/render_teleop_to_lerobot.sh — edit the CONFIG block per family:
FAMILY=jar_transport
IN_DIR="outputs/teleop_collected/${FAMILY}"                  # raw GELLO teleop HDF5s
RENDER_DIR="outputs/teleop_rendered_joint_3cam/${FAMILY}"    # Stage 1 output (flat)
DIAG_ROOT="outputs/lerobot_datasets/6fam-base/${FAMILY}"     # <task>/base/diagnostics.jsonl (per-task prompt)
REPO_ID="IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam"
LEROBOT_ROOT="outputs/lerobot_datasets/sim-jar-transport-30-joint-3cam"
LEROBOT_PY=".venv-lerobot/bin/python"
GPU=0
```

Run both stages, or one at a time:

```bash
conda activate behavior                              # Stage 1 needs it for `conda run`
bash scripts/render_teleop_to_lerobot.sh             # both stages
bash scripts/render_teleop_to_lerobot.sh --stage1    # render only
bash scripts/render_teleop_to_lerobot.sh --stage2    # convert only (local build, no push)
```

### 4.3 Stage 1: re-render raw teleop → joint + 3-cam HDF5

Per trajectory, one process: `maniguard.data.playback --input <raw> --output
<rendered>` (defaults `--controller joint --cams 3`, no flags needed). Records
8D joint state/action + `image_left`/`image_right`/`wrist_image` at 256×256.

- **Resume**: a trajectory whose rendered HDF5 already exists is skipped.
- **Teardown segfault is expected & harmless** — OmniGibson segfaults at
  `og.clear()` *after* the HDF5 is fully written (Isaac syntheticdata USD-node
  bug). Success is judged by a non-empty `action` dataset, not the exit code.
  Occasionally a render segfaults *before* the write completes (no output file)
  — just re-run that one trajectory; it's transient.

### 4.4 Stage 2: rendered HDF5 → LeRobot v2.1 multitask dataset

`multitask_lerobot_export` discovers `task_*_traj_*.hdf5` under `--input-root`,
looks up each task's language prompt at
`<diag-root>/<task>/<subdir>/diagnostics.jsonl` (`--subdir base` default; falls
back to `<diag-root>/<task>/diagnostics.jsonl` for flat trees), and writes one
dataset with per-frame `task_index`. Schema (controller × cam-count) is
auto-detected from the playback fingerprint stamped into each HDF5 — no schema
flags. The template runs this **without** `--push-to-hub` (local build only).

### 4.5 Push to HF (separate, explicit step)

> **Do NOT re-run the exporter with `--push-to-hub` once Stage 2 has already
> built the dataset locally.** The exporter's `--push-to-hub` builds *then*
> pushes in one shot, but `LeRobotDataset.create()` does `mkdir(exist_ok=False)`
> and aborts with `FileExistsError` on the already-built `--root`. Instead, push
> the existing local dataset directly:

```bash
.venv-lerobot/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
rid = "IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam"
ds = LeRobotDataset(rid, root="outputs/lerobot_datasets/sim-jar-transport-30-joint-3cam")
ds.push_to_hub(
    tags=["panda", "omnigibson", "sim", "maniguard", "multitask"],
    license="apache-2.0",
    private=True,
    push_videos=True,   # upload the already-encoded mp4s — no re-encode
    tag_version=True,   # auto-create the v2.1 codebase-version git tag openpi requires
)
PY
```

`tag_version=True` is what creates the `v2.1` git tag that openpi needs when it
pulls the dataset from HF. Plain `huggingface_hub.upload_folder` does NOT create
this tag — never use it for openpi-bound datasets. `push_videos=True` uploads the
existing mp4s rather than re-encoding. (This is the same `push_to_hub` call the
exporter's `--push-to-hub` makes — we just invoke it on the already-built
dataset.)

The auto-generated dataset card is a bare stub; replace `README.md` in the repo
with a hand-written card (task table, schema, provenance) modelled on
[`sim-dusty-transfer-30-joint-3cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/sim-dusty-transfer-30-joint-3cam)
— keep the YAML frontmatter's `configs:` block so the HF viewer finds the
parquet. Uploading the README is a new commit on `main`; the `v2.1` tag stays on
its own commit.

### 4.6 Sanity check

```bash
.venv-lerobot/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
rid = "IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam"
ds = LeRobotDataset(rid, root="outputs/lerobot_datasets/sim-jar-transport-30-joint-3cam")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames)
print("sample task_index 0 & N-1:", int(ds[0]["task_index"]), int(ds[ds.num_frames-1]["task_index"]))
PY
```

Expect:
- `codebase_version == v2.1` (in `meta/info.json`)
- features: `image_left, image_right, wrist_image` (256×256×3 video) + `state`,
  `actions` (8D float32) + the standard LeRobot index columns
- per-frame `task_index` spanning all prompts (multitask)

Live reference: `IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam` (private,
30 eps / 12967 frames / 30 fps / 2 prompts) and the sibling
`sim-dusty-transfer-30-joint-3cam` (30 eps / 20265 frames / 3 prompts) were both
built and pushed this way. Their features are byte-for-byte identical, so one
JointController SFT config consumes either.

---

## 5. Cloud setup

Same as real path §4. SSH in, `git clone openpi`, `uv sync`,
`huggingface-cli login`. Skip if you already have an instance set up
from a prior run.

---

## 6. Register training config

Edit `src/openpi/training/config.py` and add this TrainConfig (a sibling
to `pi0_libero_low_mem_finetune` but for pi0.5 + our sim dataset):

```python
TrainConfig(
    name="pi05_<task>_libero_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="IDEAS-Lab-Northwestern/<hf_repo_name>",
        base_config=DataConfig(prompt_from_task=True),
        # Our actions are already EEF delta; do NOT set extra_delta_transform
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    batch_size=4,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
```

Two warm-start choices to register if you want A/B:

| Config name | Weight loader | Expected behavior |
|---|---|---|
| `pi05_<task>_libero_lora` (recommended) | `gs://openpi-assets/checkpoints/pi05_base/params` | Cleanest fit; pi05_base has no robot prior to fight |
| `pi05_droid_<task>_libero_lora` (ablation) | `gs://openpi-assets/checkpoints/pi05_droid/params` | DROID's joint-vel action prior fights LIBERO's EEF-delta target — usually worse, useful as comparison only |

Norm_stats: don't set `assets=AssetsConfig(...)` for pi05_base — openpi
will compute them on your dataset on first run (or via
`scripts/compute_norm_stats.py`).

---

## 7. Train

### 7.1 Smoke test

```bash
tmux new -s sft
uv run scripts/train.py pi05_<task>_libero_lora \
  --exp-name=<task>_smoke \
  --num_train_steps=100 \
  --overwrite
```

Watch for:
- LeRobot dataset pulled from HF (no `RevisionNotFoundError` — v2.1 tag works)
- `tokenized_prompt: (N, 200)@int32` printed → prompt connected
- Weights downloaded from `gs://openpi-assets/checkpoints/pi05_base/params` (~8GB)
- First step loss prints
- VRAM steady (not climbing past 85%)

### 7.2 Full run

```bash
uv run scripts/train.py pi05_<task>_libero_lora \
  --exp-name=<task>_20k \
  --overwrite
# Ctrl+b d to detach
```

Default `num_train_steps=20_000` at `batch_size=4` on A100 80GB ≈
**12-18 h** for our sim mug-into-bowl (51k frames per epoch ≈ 1.5
epochs at this step count).

### 7.3 Save schedule

`save_interval=5000` + `keep_period=5000` (openpi defaults) keeps steps
**5000 / 10000 / 15000 / 20000**. All four are useful for ablation —
some sim tasks overfit by 15k.

---

## 8. Post-training (upload + serve)

### 8.1 Upload kept ckpts to HF

```bash
cd ~/openpi
CKPT_DIR=checkpoints/pi05_<task>_libero_lora/<exp-name>
HF_REPO=IDEAS-Lab-Northwestern/pi05-<task>-libero-lora

HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import create_repo, upload_folder
create_repo('$HF_REPO', repo_type='model', private=True, exist_ok=True)
upload_folder(
    folder_path='$CKPT_DIR',
    repo_id='$HF_REPO',
    repo_type='model',
    commit_message='LoRA ckpts kept steps (pi05_base warm start, sim <task>)',
    ignore_patterns=['*/train_state/**'],
)
"
```

To upload only specific steps add e.g. `allow_patterns=['5000/**', '10000/**']`.

See `docs/openpi_real_teleop_sft.md` §7 for verification commands and
model-card template.

### 8.2 Serve in sim eval

Our `maniguard/serve/openpi_native.py` auto-detects JAX vs PyTorch
ckpts; openpi-trained ckpts are JAX (orbax `params/` subdir, no
`model.safetensors`):

```bash
sudo openpi/.venv/bin/python maniguard/serve/openpi_native.py \
  --config pi05_<task>_libero_lora \
  --checkpoint <local_path>/<step>
```

Then point the eval benchmark (`maniguard.eval.benchmark`) at `localhost:8000`.
The same serve path is used for real-teleop ckpts — backend selection is
automatic.

---

## 9. Gotchas (sim-specific)

1. **Don't use DROID schema for sim**. Our sim teleop never recorded
   joint state/velocity, so the DROID converter (`real_teleop_to_droid.py`)
   would need fake joint values, which break the model's proprio prior.
   Use LIBERO schema instead — it matches our existing data 1:1.

2. **Don't warm-start sim from `pi05_droid`** unless you're explicitly
   doing an ablation. pi05_droid's vision encoder is tuned to real
   camera statistics (lens distortion, sensor noise) that OmniGibson
   raytracing doesn't reproduce; its action expert is in joint-velocity
   space, which conflicts with our EEF-delta target. Both priors fight
   the LoRA. `pi05_base` is the right choice.

3. **`extra_delta_transform=False`** for our data. Our actions are
   already EEF deltas (per-step `pos[t+1] - pos[t]`), so the additional
   delta transform would double-difference them and produce nonsense.
   LIBERO's docstring also notes their data is delta-native.

4. **Sim data is much larger than real**. 39 eps × ~1300 frames is 8×
   the real dataset. Push/pull from HF is mostly mp4 video so it's
   fine, but training wall clock per epoch is correspondingly longer
   — budget 12-18 h for 20k steps on A100 80GB.

5. **Sim images are out-of-distribution for any vision prior**. pi05_base
   has no robot prior; pi05_droid has a real-camera prior. Neither has
   seen OmniGibson raytracing. Expect the vision encoder to need
   meaningful adaptation — LoRA on `paligemma_variant="gemma_2b_lora"`
   covers this, but tasks that depend on subtle texture cues may
   struggle.

6. **`LeRobotDataset` only loads from HF if v2.1 codebase tag exists**.
   Our older sim datasets (`IDEAS-Lab-Northwestern/SFT`,
   `IDEAS-Lab-Northwestern/real-mug-into-bowl`) were pushed with
   lerobot 0.4.4 and got v3.0 format, so openpi can't load them. Either
   re-export with `lerobot<0.4` and `--push-to-hub`, or accept they're
   only useful for the legacy PyTorch path.

---

## 10. Reference experiments and datasets

| Dataset | Source | Schema | Training config name | Status |
|---|---|---|---|---|
| `IDEAS-Lab-Northwestern/sim2real-mug-into-bowl` | sim teleop, 39 eps / 51k frames | LIBERO v2.1 | `pi05_<task>_libero_lora` | Ready ✅ |
| `IDEAS-Lab-Northwestern/SFT` (goblet) | sim teleop, 21 eps | OmniGibson v3.0 | — | Needs re-push as v2.1 if used |
| `IDEAS-Lab-Northwestern/real-mug-into-bowl-droid` | real teleop, 21 eps / 6.6k frames | DROID v2.1 | `pi05_droid_finetune_lora` | Ready ✅ (real-teleop doc) |

## 11. File pointers

- `scripts/render_teleop_to_lerobot.sh` — §4 template: both stages for one family; edit the CONFIG block per family
- `maniguard/data/playback.py` — Stage 1: re-renders OmniGibson env from raw teleop, emits joint+3cam image_left/image_right/wrist_image/state HDF5 (default `--controller joint --cams 3`)
- `maniguard/data/lerobot/multitask_lerobot_export.py` — Stage 2 (current): flat `task_*_traj_*.hdf5` → LeRobot v2.1 multitask, per-task prompt from `<diag>/<task>/<subdir>/diagnostics.jsonl`, schema auto-detected from playback fingerprint
- `maniguard/data/lerobot/lerobot_export.py` — Stage 2 (legacy single-prompt path, §5+ mug-into-bowl), with `--push-to-hub`
- `maniguard/data/real_teleop/real_teleop_to_droid.py` — sister script for real data → DROID schema (different path)
- `maniguard/serve/openpi_native.py` — JAX/PyTorch-auto-detect serve, used for both sim and real evals
- `docs/openpi_real_teleop_sft.md` — companion doc for the real-data DROID path
