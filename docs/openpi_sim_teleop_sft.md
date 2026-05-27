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
OmniGibson teleop -> Stage1 playback (.hdf5) -> Stage2 lerobot_export -> HF private repo
                                                                              |
cloud: pull dataset + pi05_base ckpt from GCS  ->  openpi train.py  ->  ckpt
```

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

### 4.1 One-time env setup

Same as the real path — pin lerobot to v2.1 codebase:

```bash
uv venv --python 3.11 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python 'lerobot<0.4' h5py pyarrow opencv-python
```

### 4.2 Stage 1: render obs from raw teleop (skip if already done)

Raw OmniGibson teleop HDF5s have only the 184-dim env-state blob and
actions. To get clean obs (image, wrist_image, 8D state), run:

```bash
for f in outputs/jixing_teleop_hdf5/scene_ep*.hdf5; do
  conda run -n behavior python -m maniguard.data.playback \
    --input "$f" \
    --output "outputs/teleop_rendered_<task>/$(basename $f)" \
    --record
done
```

Output is in the schema shown in §1. Already done for our `mug-into-bowl`
task → `outputs/teleop_rendered_mug_into_bowl/` (39 eps).

### 4.3 Stage 2: HDF5 → LeRobot v2.1 + push to HF

```bash
.venv-lerobot/bin/python -m maniguard.data.lerobot.lerobot_export \
  --input-dir outputs/teleop_rendered_<task> \
  --repo-id maniguard/<task>_libero \
  --prompt "<natural-language instruction>" \
  --fps 30 \
  --root outputs/lerobot_datasets/maniguard/<task>_libero \
  --push-to-hub IDEAS-Lab-Northwestern/<hf_repo_name> \
  --hub-private
```

`--push-to-hub` calls `LeRobotDataset.push_to_hub()` internally, which
**auto-creates the v2.1 codebase-version git tag** required by openpi.
Plain `huggingface_hub.upload_folder` does NOT create this tag — never
use it for openpi-bound datasets.

Reference: `IDEAS-Lab-Northwestern/sim2real-mug-into-bowl` was pushed
this way (39 eps / 51437 frames / 30 fps / 256×256, two av1 video
streams).

### 4.4 Sanity check

```bash
.venv-lerobot/bin/python -c "
import json, pyarrow.parquet as pq
from pathlib import Path
root = Path('outputs/lerobot_datasets/maniguard/<task>_libero')
info = json.loads((root/'meta/info.json').read_text())
assert info['codebase_version'] == 'v2.1', info['codebase_version']
print(f'eps={info[\"total_episodes\"]}  frames={info[\"total_frames\"]}  fps={info[\"fps\"]}')
t = pq.read_table(next((root/'data').rglob('*.parquet')))
print('parquet cols:', t.column_names)
"
```

Expect:
- `codebase_version == v2.1`
- parquet cols: `image, wrist_image, state, actions, timestamp, frame_index, episode_index, index, task_index`
  (single combined `state` column — distinct from DROID which splits into `joint_position` + `gripper_position`)

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

- `maniguard/data/playback.py` — Stage 1: re-renders OmniGibson env from raw teleop, emits image/wrist_image/state HDF5
- `maniguard/data/lerobot/lerobot_export.py` — Stage 2: HDF5 → LeRobot v2.1 (LIBERO-compatible columns), with `--push-to-hub`
- `maniguard/data/real_teleop/real_teleop_to_droid.py` — sister script for real data → DROID schema (different path)
- `maniguard/serve/openpi_native.py` — JAX/PyTorch-auto-detect serve, used for both sim and real evals
- `docs/openpi_real_teleop_sft.md` — companion doc for the real-data DROID path
