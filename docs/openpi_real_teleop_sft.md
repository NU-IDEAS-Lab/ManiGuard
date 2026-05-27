# Fine-tuning pi0.5 on real-teleop data via openpi

End-to-end recipe for SFT of a pi0.5 VLA on a real-robot teleop dataset,
using openpi's native JAX trainer. Documents what we learned getting the
`mug-into-bowl` task running; pattern generalizes to any new real
dataset.

The **openpi path** (this doc) is the canonical community-tested pipeline.
An alternative **RLinf PyTorch path** (documented separately, uses
`maniguard/openpi/omnigibson_dataconfig.py`) makes sense only if you need
to chain sim RL on top of the SFT ckpt — otherwise prefer this doc.

---

## 1. Overview

**Input**: per-episode `.npz` files from the real-franka teleop capture, containing:
- `observation/image/cam0`, `observation/image/cam1` (JPEG bytes, 640×480)
- `observation/joint_position`, `observation/joint_velocity` (7D)
- `observation/cartesian_position` (7D: xyz + wxyz quat)
- `observation/gripper_position` (scalar, normalized 0–1)

**Output**: LoRA-finetuned pi0.5 checkpoint deployable via openpi's
`serve_policy.py`.

**Pipeline**:
```
local .npz  ->  LeRobot v2.1 (DROID schema)  ->  HF private repo
                                                      |
cloud: pull dataset + pi0.5 base ckpt from GCS  ->  openpi train.py  ->  ckpt
```

---

## 2. Hardware requirements

| Config | VRAM needed | Where this runs |
|---|---|---|
| `pi05_droid_finetune` (full SFT, bs=32) | ~80GB | A100 80GB, H100 80GB |
| `pi05_droid_finetune` with `--batch_size=1` | still ~50GB static | **Won't fit on A100 40GB** |
| `pi05_droid_finetune_lora` (custom, below) | ~15-20GB | A100 40GB, maybe 4090 |
| Inference | ~7GB BF16 | 4080 / 4090 |

**Static cost math for full SFT**: pi0.5 has ~3B params → 6.6GB BF16 weights + 6.6GB BF16 grads + 36GB AdamW FP32 states ≈ **50GB**, independent of batch size. Anything under 50GB HBM requires LoRA.

**System RAM**: 200GB+ recommended. openpi's JAX trainer is light on RAM compared to RLinf (no Ray workers), but activation staging can spike.

---

## 3. Local data conversion

Done once per task, locally (needs `~/miniconda3/envs/behavior` for HF push and `.venv-lerobot` for LeRobot writing).

### 3.1 One-time env setup

```bash
# LeRobot venv pinned to 0.3.x (v2.1 codebase, openpi-compatible)
uv venv --python 3.11 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python 'lerobot<0.4' h5py pyarrow opencv-python
```

> **Critical**: openpi pins lerobot to a git rev that expects **codebase v2.1**. Lerobot ≥ 0.4 writes v3.0 (different parquet layout). Always pin `lerobot<0.4` for openpi-bound datasets.

### 3.2 Convert npz → LeRobot (DROID schema)

```bash
.venv-lerobot/bin/python -m maniguard.data.real_teleop.real_teleop_to_droid \
  --input-dir outputs/real_teleop \
  --repo-id maniguard/<task_name> \
  --prompt "<natural-language instruction>" \
  --root outputs/lerobot_datasets/maniguard/<task_name> \
  --push-to-hub IDEAS-Lab-Northwestern/<task_name> \
  --hub-private
```

`--push-to-hub` uses LeRobot's own `push_to_hub()` which **automatically creates the v2.1 codebase-version git tag** on the HF repo. (Plain `huggingface_hub.upload_folder` does not create this tag; the tag is required by openpi's data loader on the cloud side.)

**What the converter does** (`maniguard/data/real_teleop/real_teleop_to_droid.py`):
- Assembles DROID's 8D state: `[joint_position(7), gripper_position(1)]`
- Assembles 8D action: `[joint_velocity(7), gripper_position[t+1](1)]` — matches openpi DROID pretrained convention
- Decodes JPEG → center-crops 640×480 to 16:9 → resizes to 320×180
- Fills `exterior_image_2_left` with zero frames (we only have one external cam; openpi's `DroidInputs` uses only `exterior_image_1_left` + `wrist_image_left`, so zeros are ignored at inference anyway)
- Quat convention: **wxyz** (confirmed from our data — first component near ±1)
- Drops the last frame per episode (no valid action for it)

### 3.3 Sanity check

```bash
.venv-lerobot/bin/python -c "
import json, pyarrow.parquet as pq
from pathlib import Path
root = Path('outputs/lerobot_datasets/maniguard/<task_name>')
info = json.loads((root/'meta/info.json').read_text())
assert info['codebase_version'] == 'v2.1', f'Wrong codebase: {info[\"codebase_version\"]}'
print(f'eps={info[\"total_episodes\"]}  frames={info[\"total_frames\"]}  fps={info[\"fps\"]}')
t = pq.read_table(next((root/'data').rglob('*.parquet')))
print('parquet cols:', t.column_names)
"
```

Expect:
- `codebase_version == v2.1`
- parquet cols: `joint_position, gripper_position, actions, timestamp, frame_index, episode_index, index, task_index`
- fps=15, action shape (8,), state columns separate (not a single `state` column — that's the DROID distinction vs OmniGibson schema)

---

## 4. Cloud instance setup

### 4.1 Pick the right GPU

Recommended: **A100 80GB** on Lambda Labs or OCI (~$1.5-2.5/hr). Don't cheap out on A100 40GB for full SFT — it won't fit even at batch=1.

### 4.2 SSH + openpi install

```bash
# On the cloud box
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
source .venv/bin/activate

# Verify
uv run scripts/train.py --help | head
```

### 4.3 HF auth for private dataset

```bash
huggingface-cli login   # paste token with read access to IDEAS-Lab-Northwestern org
```

LeRobot pulls via the `huggingface_hub` cache automatically on first data-loader read — no manual `huggingface-cli download` step needed.

---

## 5. Register training configs

Edit `src/openpi/training/config.py`. Find `name="pi05_droid_finetune"` and **add two new configs after it** (LoRA variants for constrained GPUs):

### 5.1 LoRA from pi05_droid warm start (recommended)

```python
TrainConfig(
    name="pi05_droid_finetune_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotDROIDDataConfig(
        repo_id="IDEAS-Lab-Northwestern/<task_name>",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
            asset_id="droid",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_droid/params"
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

### 5.2 LoRA from pi05_base (baseline for A/B)

Same as above except:
- `name="pi05_base_finetune_lora"`
- `weight_loader` → `gs://openpi-assets/checkpoints/pi05_base/params`
- `assets_dir` → `gs://openpi-assets/checkpoints/pi05_base/assets/`

Expect this to perform worse — pi05_base hasn't seen any robot data, so 21 demos via LoRA is asking a lot.

---

## 6. Run training

### 6.1 Smoke test first (always)

```bash
tmux new -s sft
uv run scripts/train.py pi05_droid_finetune_lora \
  --exp-name=<task>_droid_smoke \
  --num_train_steps=100 \
  --overwrite
```

Watch for:
- Dataset pulled from HF (no `RevisionNotFoundError` — means v2.1 tag is present)
- `Loaded norm stats from gs://openpi-assets/...` (GCS anon read works)
- Weights downloaded (~11GB for pi05_droid, ~8GB for pi05_base)
- **First step loss** prints → pipeline is good
- `nvidia-smi` shows steady VRAM

If VRAM creeps above 85%, stop and reduce `--batch_size`. If loss is NaN from step 1, norm_stats likely mismatch dataset (shouldn't happen here since we use openpi's own DROID norm_stats).

### 6.2 Full run

```bash
tmux new -s sft
uv run scripts/train.py pi05_droid_finetune_lora \
  --exp-name=<task>_droid \
  --overwrite
# Ctrl+b d to detach
```

Default `num_train_steps=20_000` at `batch_size=4` on A100 80GB ≈ **8-12 hours**.

Checkpoints land in `checkpoints/pi05_droid_finetune_lora/<exp-name>/`.

### 6.3 Chaining experiments

Use distinct `--exp-name` so ckpts don't overwrite:

```bash
# After the droid-LoRA finishes:
uv run scripts/train.py pi05_base_finetune_lora \
  --exp-name=<task>_base \
  --overwrite
```

Checkpoints:
- `checkpoints/pi05_droid_finetune_lora/<task>_droid/`
- `checkpoints/pi05_base_finetune_lora/<task>_base/`

---

## 7. Post-training

### 7.1 Checkpoint layout

Each run writes one directory per saved step into
`checkpoints/<config_name>/<exp-name>/`. Orbax sub-structure per step:

```
<step>/
├── params/         ~12GB   model weights (orbax pytree)     — needed for inference
├── train_state/    ~36GB   optimizer states + RNG state     — needed only to resume
├── assets/         <10MB   norm_stats + callbacks snapshot  — needed for serve
└── metrics/        <1MB    training metrics (json)          — optional
```

**Default `save_interval=5000` + `keep_period=5000`**, so a full 20k run keeps
steps **5000 / 10000 / 15000 / 20000**. For A/B comparison you almost always
want all of them on HF.

### 7.2 Push checkpoints to HuggingFace

One command uploads every kept step and skips the heavy `train_state/`
(the part you don't need for inference or LoRA-on-top resumption):

```bash
cd ~/openpi
CKPT_DIR=checkpoints/pi05_droid_finetune_lora/<exp-name>   # e.g. mug_into_bowl_droid
HF_REPO=IDEAS-Lab-Northwestern/pi05-<task>-droid-lora      # e.g. pi05-mug-into-bowl-droid-lora

# Sanity-check which steps exist
ls $CKPT_DIR/

# Create private model repo (idempotent)
HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import create_repo
create_repo('$HF_REPO', repo_type='model', private=True, exist_ok=True)
print('repo ready')
"

# Upload every step, ignore optimizer states
HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import upload_folder
commit = upload_folder(
    folder_path='$CKPT_DIR',
    repo_id='$HF_REPO',
    repo_type='model',
    commit_message='LoRA ckpts, all kept steps (pi05-droid warm start, <task>)',
    ignore_patterns=['*/train_state/**'],
)
print(commit)
"
```

Size & time: 4 steps × ~12GB = **~48GB**. Cloud egress at 50–200MB/s gives
**10-30 min**.

Resulting repo layout:
```
IDEAS-Lab-Northwestern/pi05-<task>-droid-lora/
├── 5000/  {params/, assets/, metrics/}
├── 10000/ {params/, assets/, metrics/}
├── 15000/ {params/, assets/, metrics/}
└── 20000/ {params/, assets/, metrics/}
```

### 7.3 Verify the upload

```bash
HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('$HF_REPO', repo_type='model')
print(f'total files: {len(files)}')
steps = sorted({f.split('/')[0] for f in files if f[0].isdigit()}, key=int)
for s in steps:
    subs = sorted({f.split('/')[1] for f in files if f.startswith(s + '/')})
    print(f'  {s}: {subs}')
"
```

Expected: every step lists `['assets', 'metrics', 'params']`. If you see
`train_state` anywhere, the `ignore_patterns` didn't match — re-check the
glob (`*/train_state/**`, note the leading `*/` which is critical so it
matches `<step>/train_state/...`).

### 7.4 Add a model card

Easiest: after upload, open
`https://huggingface.co/<HF_REPO>` → Edit model card → paste this
template (swap placeholders):

```markdown
# pi0.5 LoRA, <task> (warm-started from pi05-droid)

**Base weights**: `gs://openpi-assets/checkpoints/pi05_droid/params`
**Training config**: `pi05_droid_finetune_lora` (custom, see repo /docs)
**Dataset**: `IDEAS-Lab-Northwestern/<task>-droid` (<N> eps / <M> frames / 15Hz, DROID LeRobot schema)
**Prompt**: "<exact prompt string used during training>"
**Hardware / duration**: A100 80GB, <hours> h, batch_size=4
**Kept steps**: 5000, 10000, 15000, 20000 — backbone frozen, LoRA adapters only
**Action space**: 8D `[joint_velocity(7), gripper_position(1)]` (DROID convention)

## Serve
\`\`\`
uv run scripts/serve_policy.py \
  --policy.config=pi05_droid_finetune_lora \
  --policy.dir=<local_dir>/20000
\`\`\`
```

### 7.5 Pull a specific step back locally (for eval)

On a 4080/4090 (inference needs only ~7GB VRAM):

```bash
# Download one step
mkdir -p vla_models
HF_HUB_DISABLE_XET=1 huggingface-cli download \
  IDEAS-Lab-Northwestern/pi05-<task>-droid-lora \
  --include "20000/**" \
  --local-dir vla_models/pi05-<task>-droid-lora

# Alternative: scp the single step folder directly off the cloud box
# scp -r <cloud-host>:openpi/checkpoints/pi05_droid_finetune_lora/<exp>/20000 \
#     ./vla_models/pi05-<task>-droid-lora/
```

### 7.6 Serve + eval

```bash
uv run scripts/serve_policy.py \
  --policy.config=pi05_droid_finetune_lora \
  --policy.dir=vla_models/pi05-<task>-droid-lora/20000
```

Then run your real-franka eval client against the WebSocket.

---

## 8. Gotchas we hit (save yourself time)

1. **`lerobot<0.4` is mandatory**. Newer lerobot writes v3.0 format; openpi can only read v2.1. Trying to run with v3.0 gives `RevisionNotFoundError: Your dataset must be tagged with a codebase version.`

2. **`upload_folder` doesn't create the codebase-version git tag**. Use `LeRobotDataset.push_to_hub()` instead (our converter does this when `--push-to-hub` is passed). If you uploaded via `upload_folder` and now openpi fails to load, you can retro-fit:
   ```python
   from huggingface_hub import HfApi
   HfApi().create_tag("<repo>", tag="v2.1", repo_type="dataset")
   ```
   but only if the data already on HF is in v2.1 layout.

3. **Quat convention is wxyz** (not scipy default xyzw). Check `observation/cartesian_position[0, 3]` — if |value| ≈ 1 it's wxyz; if |value| ≈ 0 it's xyzw. Verified on our data, the converter already handles it.

4. **Gripper is normalized 0–1**, not meters. Distribution is bimodal (~0.05 open / ~0.99 closed). Threshold at 0.5.

5. **`pi05_droid_finetune` (full SFT) needs 80GB HBM**. openpi's docstring says 8×H100 for the canonical recipe; single A100 80GB works too but at smaller batch. A100 40GB is a no-go for full SFT — use the LoRA configs above.

6. **openpi warns DROID + LoRA is weak**: README says *"haven't found the policies to perform well so far."* Our LoRA configs above work mechanically but expect modest task performance. If budget allows, rent 80GB and run full `pi05_droid_finetune` instead.

7. **Cloud RAM pressure**: VSCode on the cloud dev host + training can collide. Train inside `tmux` outside any IDE — avoids terminal death taking down the run.

---

## 9. Reference file pointers

- `maniguard/data/real_teleop/real_teleop_to_droid.py` — npz → LeRobot (DROID schema) converter
- `maniguard/data/real_teleop/real_teleop_to_hdf5.py` — npz → sim-compat HDF5 (for the RLinf path, not used here)
- `maniguard/data/lerobot/lerobot_export.py` — HDF5 → LeRobot (OmniGibson schema, RLinf path)
- `maniguard/data/lerobot/norm_stats.py` — computes openpi-format norm_stats from a LeRobot dataset (not needed for DROID path since we reuse openpi's official DROID norm_stats)

## 10. Current datasets on HF

| Task | HF repo | Schema | Used with |
|---|---|---|---|
| goblet-pick-place (sim) | `IDEAS-Lab-Northwestern/SFT` | OmniGibson v3.0 | RLinf path |
| mug-into-bowl (real) | `IDEAS-Lab-Northwestern/real-mug-into-bowl` | OmniGibson v3.0 | RLinf path |
| mug-into-bowl (real) | `IDEAS-Lab-Northwestern/real-mug-into-bowl-droid` | **DROID v2.1** | **openpi path (this doc)** |
