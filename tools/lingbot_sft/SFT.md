# ManiGuard SFT on LingBot-VLA 2.0

Post-trains **LingBot-VLA 2.0** on the six ManiGuard datagen-v1 families — the fifth base
model of the ManiGuard VLA benchmark (alongside pi0.5, pi0, GR00T N1.6, SmolVLA), trained on
identical data, cameras, and controller so the comparison isolates the model.

This repo is a **vendored fork** of [`robbyant/lingbot-vla-v2`](https://github.com/robbyant/lingbot-vla-v2)
(upstream `be27333`). Upstream files are untouched; the ManiGuard layer is exactly:

```
configs/robot_configs/maniguard.yaml    feature mapping (8-D single arm, 2 cameras)
configs/vla/maniguard/maniguard.yaml    post-training recipe (8-GPU shape, 2 epochs)
tools/lingbot_sft/                      download_weights.sh · run_sft.sh · push_to_hf.py
```

## Why so little glue

Upstream consumes **LeRobot v2.1 directly** — the exact format our datagen datasets already
ship — and registers a new robot through a declarative YAML whose `origin_keys` are free-form
lookups into the LeRobot item. So the six datasets stay **read-only and byte-identical** to
what the other four base models train on; there is no conversion, copy, or key remap step.

## Recipe

| | |
|---|---|
| Warm start | `robbyant/lingbot-vla-v2-6b` — the **pretrain** release. **Not** `…-6b-robotwin`, which is already post-trained 50k steps on RoboTwin. |
| Inputs | 2 cameras (`camera_top` ← `image_left` overview, `camera_wrist_left` ← `wrist_image`), 8-D joint state (7 arm + 1 gripper) into the 55-D unified vector |
| Actions | **absolute** joint targets (`subtract_state: false`), following upstream's simulation recipe; eval applies them straight to a JointController |
| Distillation | **on** (depth + DINO-video alignment) — upstream's default post-training setup |
| Scale | global batch 256 = micro 32 × 8 GPUs, lr 5e-5 cosine (upstream's own 8-GPU shape), **2 epochs**, 4-rung ladder |

Steps per family (2 epochs at global batch 256 — the same numbers the pi0.5 / pi0 tracks use):

| family | frames | steps | save every |
|---|---|---|---|
| clutter | 901,520 | 7,100 | 1,775 |
| cabinet | 4,172,962 | 32,650 | 8,163 |
| stack | 2,652,083 | 20,750 | 5,188 |
| jar | 946,870 | 7,400 | 1,850 |
| lid | 1,055,142 | 8,250 | 2,063 |
| dusty | 1,879,498 | 14,700 | 3,675 |

## Setup

```bash
bash tools/create_train_env.sh              # upstream's env builder (conda + flash-attn 2.8.3)
conda activate <the env it created>
export HF_TOKEN=...  WANDB_API_KEY=...
bash tools/lingbot_sft/download_weights.sh  # ~38 GB into assets/pretrained/
```

Two things that bite on a bare container:

- **flash-attn** is built from source by the env script and needs `nvcc`. If CUDA is not on
  the box, skip the compile entirely with a prebuilt wheel matching your torch/ABI:
  `bash tools/create_train_env.sh --resume --flash-attn-wheel /path/to/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
  (check the ABI with `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`).
- **`import lingbotvla` fails**: upstream launches `torchrun scripts/<x>.py`, which puts
  `scripts/` on `sys.path` instead of the repo root, and the env builder never runs
  `pip install -e .`. `run_sft.sh` exports `PYTHONPATH=<repo root>` to cover its own children;
  for other entrypoints (upstream's `train.sh`, the deploy scripts) install the package once:
  `python -m pip install -e . --no-deps`.
- **Video decode**: torchcodec dlopen()s FFmpeg at runtime; a bare container has none and the
  error only appears inside the dataloader workers. Export `FFMPEG_LIB_DIR=<dir with libav*>`
  (any conda env that ships FFmpeg works) and `run_sft.sh` puts it on `LD_LIBRARY_PATH`.
  Verify once with
  `python -c "from torchcodec.decoders import VideoDecoder; print('ok')"`.

⚠️ **The ~38 GB of weights land in `assets/pretrained/` inside the repo by default.** On a
cluster the repo usually sits on a small home volume, so first point them at the big
filesystem — either `export PRETRAIN_DIR=/big/vol/pretrained`, or symlink
`assets/pretrained -> /big/vol/pretrained` (the same treatment `outputs/` gets). The script
prints its destination and the free space there before downloading.

Weights fetched: `lingbot-vla-v2-6b` (28 GB, pretrain + both distillation teachers),
`Qwen3-VL-4B-Instruct` (tokenizer / base VLM), `moge-2-vitb-normal` (419 MB, depth teacher).

## Run

```bash
# --data-root = the shared read-only root holding IDEAS-Lab-Northwestern/datagen-*-v1-joint-5cam
bash tools/lingbot_sft/run_sft.sh --family clutter --data-root /path/to/shared/lerobot
```

Per family: norm stats (computed once into `assets/norm_stats/maniguard_<fam>.json`, reused
afterwards; force with `--norm-stats`) → 2-epoch 8-GPU training → checkpoints under
`outputs/lingbot_sft/runs/<family>/`.

All six, serially:

```bash
for FAM in clutter cabinet stack jar lid dusty; do
  bash tools/lingbot_sft/run_sft.sh --family "$FAM" --data-root /path/to/shared/lerobot
done
```

## Push

```bash
python tools/lingbot_sft/push_to_hf.py --family clutter \
  --run-dir outputs/lingbot_sft/runs/clutter \
  --repo IDEAS-Lab-Northwestern/lingbot-vla2-datagen-v1-clutter-joint-2cam-yanZ
```

Pushes the newest `global_step_*/hf_ckpt` (DCP shards and optimizer state are excluded) plus
`maniguard/norm_stats.json` and `maniguard/robot_config.yaml`, so the repo is self-contained
for serving.

## ⚠️ Verify on the first family before committing to the full sweep

1. **Loader**: the run prints the resolved dataset and step count — confirm the frame count
   matches the table above (proves the LeRobot dataset and the 8-D mapping are read correctly).
2. **VRAM**: upstream ships this micro-batch for its own 8-GPU config; read the actual peak on
   the first run. If a card is tight, set `--train.enable_gradient_checkpointing true`
   (slower but much cheaper), which is upstream's documented lever.
3. **Cameras**: we declare 2 views where upstream's samples use 3. Confirm the first batch
   builds (the policy pads/masks the unused slot) before launching a long run.
