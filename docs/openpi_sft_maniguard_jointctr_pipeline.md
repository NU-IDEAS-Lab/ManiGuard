# pi0.5 SFT on ManiGuard joint-controller sim teleop (parallel-openpi module)

End-to-end recipe for LoRA SFT of a pi0.5 VLA on **ManiGuard JointController
sim teleop data**, using openpi's native JAX trainer through ManiGuard's own
`maniguard/openpi_sft` task-config module against an **unmodified, parallel
openpi clone**.

This is the **JointController** track: collection → render → SFT → eval are all
joint-space (no end-effector / IK anywhere). It differs from the EEF/LIBERO
track in `docs/openpi_sim_teleop_sft.md` only in the action/state convention;
the camera layout is the same LIBERO 2-cam.

!!! note "Why a separate module instead of editing openpi"
    openpi is consumed as a pinned, **pristine** clone sitting next to ManiGuard.
    All ManiGuard-specific task configs (data config, policy transforms, train
    configs) live in `maniguard/openpi_sft`; importing that package registers
    them into openpi's `_CONFIGS_DICT` at runtime. So a compute box just clones
    `ManiGuard` + `openpi` side by side, and nothing in openpi is patched.

---

## 1. Overview

**Input**: a LeRobot v2.1 dataset on HF, rendered joint + 3-cam by the
ManiGuard pipeline (`maniguard.data.playback --controller joint --cams 3` →
`maniguard.data.lerobot.multitask_lerobot_export`). Reference dataset:
`IDEAS-Lab-Northwestern/sim-dusty-transfer-30-joint-3cam`.

```
image_left    (video 256x256x3)  third-person overview (cam_left)
image_right   (video 256x256x3)  third-person overview (cam_right)   <- NOT used at SFT
wrist_image   (video 256x256x3)  wrist camera
state         (f32, 8)  [joint_0..6, gripper_pos]          absolute joint config
actions       (f32, 8)  [joint_0..6_target, gripper_cmd]   absolute joint target + binary gripper
```

**Output**: a LoRA-finetuned pi0.5 checkpoint ladder pushed to an HF model repo,
ready for joint-controller eval (eval is a separate effort, not covered here).

**Pipeline**:
```
HF joint dataset + pi05_base (GCS)
  -> [compute_norm_stats]  (openpi script, via our launcher)
  -> [smoke 100 steps]
  -> openpi train.py  (via our launcher; configs registered by maniguard.openpi_sft)
  -> checkpoints/<cfg>/<exp>/<step>/   (orbax, async)
        \-> hf_push_watcher.py streams each finalized ckpt to HF during the run
```

---

## 2. Joint vs EEF: what the data config does

The dataset stores **absolute joint** state + actions. `Sim2CamLiberoDataConfig`
(in `maniguard/openpi_sft/data_configs.py`) drives the JointController pipeline
with `use_delta_joint_actions=True` (its default):

- **State** `[joint_0..6, gripper_pos]` (8-D) is fed as-is.
- **Action** `[joint_0..6_target, gripper_cmd]` (8-D): the 7 arm joints are
  converted to **per-step deltas** for the model (gripper kept absolute), via
  openpi's `DeltaActions`/`AbsoluteActions` with `make_bool_mask(7, -1)`. At
  inference the model's delta output is reconstructed to an **absolute joint
  target**, which an eval-time JointController consumes directly — no
  eef→joint IK. This mirrors openpi's `RLDSDroidDataConfig` JOINT_POSITION
  handling.

`use_delta_joint_actions` always stays `True` for this pipeline. (The `False`
EEF branch exists only to keep the class general; don't use it here.)

### Cameras: 3-cam dataset, 2-cam policy (LIBERO)

Although the dataset has three image streams, SFT (and later eval) follow the
**LIBERO 2-cam convention** — pi0.5 / Franka have one wrist + one third-person
view, so the third image slot is always blacked out:

```
base_0_rgb        <- image_left    (third-person overview)
left_wrist_0_rgb  <- wrist_image   (wrist)
right_wrist_0_rgb <- zeros, masked off
```

`image_right` is simply not mapped (dropped). See
`maniguard/openpi_sft/policies/sim_2cam_policy.py`.

---

## 3. Layout & one-time setup on a compute box

Clone ManiGuard and openpi **side by side**:

```
<root>/
  ManiGuard/     # this repo
  openpi/        # pristine upstream clone, openpi's own venv installed
```

Export the env vars once in your shell rc (`~/.bashrc` / `~/.zshrc`) so you never
have to prefix them on the command line:

```bash
export OPENPI_ROOT=/abs/path/to/openpi        # default fallback is ../openpi
export HF_TOKEN=hf_...                         # for dataset pull + checkpoint push
export WANDB_API_KEY=...                       # training logs (required)
```

Install openpi per its own instructions (its venv / `uv`). ManiGuard's
`maniguard.openpi_sft` only needs to be importable on `PYTHONPATH` (the launchers
add the repo root automatically).

---

## 4. Register & inspect the config

No registration command is needed — importing `maniguard.openpi_sft` registers
the TrainConfigs into openpi. The reference config is
`pi05_base_dusty_transfer_joint_2cam_lora` (fully inline in
`maniguard/openpi_sft/train_configs.py`):

- model: pi0.5, `gemma_2b_lora` + `gemma_300m_lora`, `action_dim=32`,
  `action_horizon=16`
- data: `Sim2CamLiberoDataConfig(repo_id=sim-dusty-transfer-30-joint-3cam,
  use_delta_joint_actions=True, prompt_from_task=True)`
- warm-start: `pi05_base`
- **scale is a placeholder** (10k steps @ batch 4, ~2 epochs over 20,265 frames,
  `keep_period=2000` → 5 evenly-spaced checkpoints). Retune per compute box;
  keep `decay_steps == num_train_steps`, `keep_period = steps // 5`, and
  sqrt-scale `peak_lr` if you raise the batch size.

---

## 5. Run

`tools/openpi_sft/run_sft.sh` orchestrates the whole run. It computes norm stats
(sim configs warm-start from `pi05_base`, which ships none), runs a 100-step
smoke and deletes it, launches the HF-push watcher in the background, then runs
the full training and waits for the watcher to finish uploading the final ckpt.

```bash
cd ManiGuard
tools/openpi_sft/run_sft.sh \
  --config pi05_base_dusty_transfer_joint_2cam_lora \
  --exp    dusty_joint_2cam \
  --norm-stats \
  --push-repo IDEAS-Lab-Northwestern/<model-repo> --push-private \
  [--steps 10000] [--batch 4] [--keep-period 2000]
```

Run it inside a tmux session (training is long); detach with `Ctrl+b d`.

Options: `--no-smoke`, `--smoke-only`, `--resume`, `--overwrite`,
`--poll-interval N` (watcher scan seconds, default 30). `OPENPI_ROOT`,
`HF_TOKEN`, `WANDB_API_KEY` come from the environment; a missing `WANDB_API_KEY`
(or `HF_TOKEN` with `--push-repo`) is a hard error.

### Manual building blocks

The launchers can also be run directly (each imports `maniguard.openpi_sft`
first, then delegates to the pristine openpi script):

```bash
# norm stats (config name is positional, per openpi's script)
python tools/openpi_sft/compute_norm_stats.py pi05_base_dusty_transfer_joint_2cam_lora

# training (all openpi train.py flags pass through)
python tools/openpi_sft/train.py pi05_base_dusty_transfer_joint_2cam_lora \
  --exp-name dusty_joint_2cam [--overwrite]
```

---

## 6. Checkpoints → HF

Two uploaders share one de-dup rule, so they never double-push and resume
safely. "Already pushed" = the local `<step>/params/` filename set is a subset
of the live HF repo's `<remote_label>/params/` — checked against HF itself (the
authoritative source), not a local marker. Both upload `params/` + `assets/`
(norm stats) and skip `train_state/`. The 0-indexed final step dir
(`num_train_steps - 1`) is relabeled to the round `num_train_steps` on HF.

- **`hf_push_watcher.py`** — launched in the background by `run_sft.sh`
  (`--push-repo`). Polls the checkpoint dir and uploads each checkpoint the
  moment it finalizes, so checkpoints land on HF *during* the run without
  blocking the GPU (separate process, filesystem reads only). Exits once the
  final step is confirmed complete on HF.

- **`hf_push.py`** — one-shot, run by hand any time to backfill anything the
  watcher missed; it skips whatever is already complete on HF.

```bash
# backfill / verify after the run (skips everything already up)
python tools/openpi_sft/hf_push.py \
  --ckpt-dir "$OPENPI_ROOT/checkpoints/pi05_base_dusty_transfer_joint_2cam_lora/dusty_joint_2cam" \
  --repo IDEAS-Lab-Northwestern/<model-repo> \
  --num-train-steps 10000 [--readme docs/cards/<card>.md] [--private]
```

---

## 7. Gotchas

- **openpi must stay pristine.** Never edit the openpi clone; add task configs in
  `maniguard/openpi_sft` and let `register()` inject them. Mirrors how
  `maniguard/_omnigibson_patches.py` keeps ManiGuard code out of OmniGibson.
- **norm stats are per-dataset.** `pi05_base` ships none, so `--norm-stats` is
  required the first time for each new dataset.
- **Final checkpoint is 0-indexed** (`steps-1`, e.g. `9999/`); the uploaders
  relabel it to the round number on HF. Keep `--num-train-steps` consistent
  between training and the manual `hf_push.py` so the relabel matches.
- **Concurrent uploads are safe but don't fight the watcher** — if you run
  `hf_push.py` while the watcher is still live, both consult HF and skip what's
  already there; nothing is duplicated.
- **Training scale in the config is a placeholder** sized for the 30-episode
  dusty set; rescale `num_train_steps` / `decay_steps` / `keep_period` /
  `peak_lr` together for your dataset and hardware.

---

## 8. File pointers

| Purpose | Path |
|---|---|
| Data config (joint, 2-cam LIBERO) | `maniguard/openpi_sft/data_configs.py` |
| Policy transforms (2-cam mapping) | `maniguard/openpi_sft/policies/sim_2cam_policy.py` |
| Train configs + `register()` | `maniguard/openpi_sft/train_configs.py` |
| Train / norm-stats launchers | `tools/openpi_sft/{train,compute_norm_stats}.py` |
| Run orchestration | `tools/openpi_sft/run_sft.sh` |
| HF uploaders + shared de-dup | `tools/openpi_sft/{hf_push_watcher,hf_push,_hf_push_common}.py` |
| Dataset render → LeRobot (upstream of SFT) | `maniguard/data/playback.py`, `maniguard/data/lerobot/multitask_lerobot_export.py` |
| EEF/LIBERO track (sibling) | `docs/openpi_sim_teleop_sft.md` |
