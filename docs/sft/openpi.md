# openpi (pi0.5 / pi0) SFT

LoRA SFT of a pi0.5 or pi0 VLA on a ManiGuard **JointController** dataset, using
openpi's native JAX trainer through ManiGuard's own `maniguard/openpi_sft`
task-config module against an **unmodified, parallel openpi clone**. Collection →
dataset → SFT → eval are all joint-space (no end-effector / IK anywhere).

!!! note "The pi0 track"
    Everything below is written for pi0.5; the module also registers a **pi0**
    config per family (`pi0-base_datagen_v1_<fam>_joint_2cam_lora`) with the same
    data pipeline and launcher. The diffs are exactly the model generation:
    warm-start `pi0_base` instead of `pi05_base`, continuous state input, and
    `action_horizon=50` (pi0's native chunk; the pi0.5 configs use 16). Norm
    stats are computed **fresh** under each pi0 config name — the stats pass
    chunks actions by horizon, so the pi0.5 stats are not reusable.

!!! note "Why a separate module instead of editing openpi"
    openpi is consumed as a pinned, **pristine** clone next to ManiGuard. All
    ManiGuard-specific task configs (data config, policy transforms, train configs)
    live in `maniguard/openpi_sft`; importing that package registers them into
    openpi's `_CONFIGS_DICT` at runtime. A compute box clones `ManiGuard` + `openpi`
    side by side, and nothing in openpi is patched (mirrors how
    `maniguard/_omnigibson_patches.py` keeps ManiGuard code out of OmniGibson).

## 1. Input & output

**Input**: any ManiGuard joint LeRobot v2.1 dataset (see
[Dataset & data-source configs](dataset_and_config.md)) — primarily the scripted
**datagen** datasets (`datagen-<fam>-v1-joint-5cam`), or a sim-teleop
(`sim-<fam>-30-joint-3cam`) dataset. The schema is absolute joint:

```
state    (f32, 8)  [joint_0..6, gripper]          absolute joint config
actions  (f32, 8)  [joint_0..6_target, gripper]   absolute joint target + binary gripper
image_*  (video 256×256×3)                         overviews + wrist
```

**Output**: a LoRA-finetuned pi0.5 checkpoint ladder pushed to an HF model repo,
ready for joint-controller eval.

**Pipeline**:
```
HF joint dataset + pi05_base (GCS)
  → [compute_norm_stats]   (openpi script, via our launcher)
  → [smoke 100 steps]
  → openpi train.py        (via our launcher; configs registered by maniguard.openpi_sft)
  → checkpoints/<cfg>/<exp>/<step>/   (orbax, async)
        ↳ hf_push_watcher.py streams each finalized ckpt to HF during the run
```

## 2. What the data config does (absolute → delta → absolute)

The dataset is **absolute joint**. `Sim2CamLiberoDataConfig`
(`maniguard/openpi_sft/data_configs.py`) drives the openpi pipeline with
`use_delta_joint_actions=True` (its default, and required for this pipeline):

- **State** `[joint_0..6, gripper]` (8-D) is fed as-is.
- **Action** `[joint_0..6_target, gripper]` (8-D): the 7 arm joints are converted
  to **per-step deltas** for the model (gripper kept absolute), via openpi's
  `DeltaActions`/`AbsoluteActions` with `make_bool_mask(7, -1)`. At inference the
  model's delta output is reconstructed to an **absolute joint target**, which an
  eval-time `JointController` consumes directly — no eef→joint IK. This mirrors
  openpi's `RLDSDroidDataConfig` JOINT_POSITION handling.

`use_delta_joint_actions` stays `True`. The `False` (EEF) branch exists only to
keep the class general — do not use it here (it would mis-read the absolute joint
actions as 7-D EEF deltas).

### Cameras: overviews shipped, 2-cam policy (LIBERO)

pi0.5 / Franka use one wrist + one third-person view (LIBERO 2-cam), so exactly
one overview is consumed and the third pi0.5 image slot is zeroed/masked:

```
base_0_rgb        ← image_<external_cam>   (chosen third-person overview)
left_wrist_0_rgb  ← wrist_image            (wrist)
right_wrist_0_rgb ← zeros, masked off
```

`external_cam` (config field, default `left`) picks WHICH dataset overview feeds
the fixed policy key `observation/image_left`. Per family one view may be higher
quality; **eval must read the same `external_cam` back from the checkpoint's
train config** to stay in distribution. See
`maniguard/openpi_sft/policies/sim_2cam_policy.py`.

## 3. One-time setup on a compute box

Clone ManiGuard and openpi **side by side** and export the env once in your rc:

```
<root>/
  ManiGuard/     # this repo
  openpi/        # pristine upstream clone, openpi's own venv installed
```

```bash
export OPENPI_ROOT=/abs/path/to/openpi   # default fallback is ../openpi
export HF_TOKEN=hf_...                    # dataset pull + checkpoint push
export WANDB_API_KEY=...                  # training logs (required)
```

`maniguard.openpi_sft` only needs to be importable on `PYTHONPATH` (the launchers
add the repo root automatically).

## 4. Register & inspect the config

Importing `maniguard.openpi_sft` registers the TrainConfigs into openpi — no
registration command. Each SFT run should get its **own uniquely-named config**
(fully inline in `maniguard/openpi_sft/train_configs.py`). A config carries the
run identity so a fresh box / another person / an agent can launch from just
`--config`:

- `project_name="maniguard-sft"` — the wandb project for all ManiGuard SFT.
- `policy_metadata` (openpi never interprets it; carries publish metadata):
  - `default_exp` — experiment / wandb run name + `outputs/sft_runs/<exp>/` folder.
  - `hf_repo` — the model repo checkpoints push to.
  - `hf_private` — push visibility (`False` = public model repo; datasets stay private).

`run_sft.sh` reads these via `tools/openpi_sft/_config_meta.py`, so launching
needs only `--config`; any CLI flag still overrides.

!!! tip "Training scale is a placeholder"
    Configs ship a placeholder scale. Retune `num_train_steps` per compute box;
    keep `decay_steps == num_train_steps`, `keep_period = steps // 5`, and
    sqrt-scale `peak_lr` when you raise the batch size.

## 5. Run

`tools/openpi_sft/run_sft.sh` orchestrates the whole run: computes norm stats
(sim configs warm-start from `pi05_base`, which ships none), runs a 100-step
smoke and deletes it, launches the HF-push watcher in the background, then runs
the full training and waits for the final upload. Exp name, HF repo, and
visibility default from the config, so the minimal launch is just `--config`:

```bash
cd ManiGuard
tools/openpi_sft/run_sft.sh \
  --config pi05-base_datagen_v1_dusty_joint_2cam_lora \
  --norm-stats
# → exp=pi05-base_datagen_v1_dusty_joint_2cam_lora,
#   push → <org>/pi05-base-datagen-v1-dusty-joint-2cam-lora (from the config's policy_metadata),
#   artifacts under outputs/sft_runs/pi05-base_datagen_v1_dusty_joint_2cam_lora/
```

Run it inside tmux (training is long); detach with `Ctrl+b d`. Override anything:
`--exp`, `--steps`, `--batch`, `--keep-period`, `--push-repo`, `--push-private`,
`--no-push`, `--no-smoke`, `--smoke-only`, `--resume`, `--overwrite`,
`--poll-interval`. A missing `WANDB_API_KEY` (or `HF_TOKEN` when pushing) is a
hard error.

### Run layout (everything under `outputs/`, gitignored)

```
ManiGuard/outputs/
  sft_runs/<exp>/
    checkpoints/<config_name>/<exp>/<step>/   # orbax ckpts (params/ assets/ train_state/)
    assets/<config_name>/...                  # computed norm stats
    logs/{normstats,smoke,train,watcher}.log
  openpi_cache/                               # pi05_base warm-start (shared)
  hf/{lerobot,home,datasets}/                 # HF caches incl. training dataset (shared)
```

`run_sft.sh` force-sets `--checkpoint-base-dir` / `--assets-base-dir`,
`OPENPI_DATA_HOME`, and the HF caches into `outputs/` (process-local, so a host rc
pointing these at an unwritable path can't break the run, and other projects are
unaffected).

### Manual building blocks

Each launcher imports `maniguard.openpi_sft` first, then delegates to the pristine
openpi script (these bypass `run_sft.sh`, so pass `--checkpoint-base-dir` /
`--assets-base-dir` yourself):

```bash
# norm stats (openpi's script is tyro.cli → config name is the --config-name flag)
python tools/openpi_sft/compute_norm_stats.py --config-name <config>

# training (all openpi train.py flags pass through)
python tools/openpi_sft/train.py <config> --exp-name <exp> [--overwrite]
```

## 6. Checkpoints → HF

Two uploaders share one de-dup rule (never double-push, resume-safe). "Already
pushed" = the local `<step>/params/` filename set is a subset of the live HF
repo's `<remote_label>/params/`, checked against HF itself. Both upload `params/`
+ `assets/` (norm stats), skip `train_state/`. The 0-indexed final step
(`num_train_steps - 1`) is relabeled to the round `num_train_steps` on HF.

- **`hf_push_watcher.py`** — launched in the background by `run_sft.sh`; uploads
  each checkpoint the moment it finalizes, so checkpoints land on HF *during* the
  run without blocking the GPU. Exits once the final step is confirmed on HF.
- **`hf_push.py`** — one-shot backfill; skips whatever is already complete on HF.

```bash
python tools/openpi_sft/hf_push.py \
  --ckpt-dir outputs/sft_runs/<exp>/checkpoints/<config>/<exp> \
  --repo <org>/<model-repo> \
  --num-train-steps <steps> [--readme <path/to/model_card.md>] [--private]
```

## 7. Gotchas

- **openpi stays pristine.** Never edit the openpi clone; add task configs in
  `maniguard/openpi_sft` and let `register()` inject them.
- **Norm stats are per-dataset.** `pi05_base` ships none, so `--norm-stats` is
  required the first time for each new dataset.
- **Final checkpoint is 0-indexed** (`steps-1`); the uploaders relabel it to the
  round number. Keep `--num-train-steps` consistent between train and `hf_push.py`.
- **`external_cam` must match at eval** — read it back from the checkpoint's train
  config, or the policy sees an out-of-distribution viewpoint.

## 8. File pointers

| Purpose | Path |
|---|---|
| Data config (joint, 2-cam LIBERO, `external_cam`) | `maniguard/openpi_sft/data_configs.py` |
| Policy transforms (2-cam mapping) | `maniguard/openpi_sft/policies/sim_2cam_policy.py` |
| Train configs + `register()` | `maniguard/openpi_sft/train_configs.py` |
| Train / norm-stats launchers | `tools/openpi_sft/{train,compute_norm_stats}.py` |
| Run orchestration | `tools/openpi_sft/run_sft.sh` |
| Config-metadata reader | `tools/openpi_sft/_config_meta.py` |
| HF uploaders + shared de-dup | `tools/openpi_sft/{hf_push_watcher,hf_push,_hf_push_common}.py` |
| Dataset production (upstream of SFT) | [Dataset & data-source configs](dataset_and_config.md) |
