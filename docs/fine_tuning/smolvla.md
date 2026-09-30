# SmolVLA SFT

SFT of HuggingFace LeRobot's **SmolVLA** on the same ManiGuard joint LeRobot v2.1
datasets used for [openpi](openpi.md) and [GR00T](gr00t.md) — the dataset is
model-agnostic; SmolVLA only differs in how the shared cameras/state/action are
presented to it.

SmolVLA is **LeRobot-native**: it is fine-tuned with the upstream `lerobot-train`
CLI. The dataset uses standard observation/action keys; the training launcher
maps its two camera names to the pretrained policy's expected keys. Saved
processors carry this mapping and the dataset normalization statistics into
evaluation. The model pads state/action vectors to its internal width.

## Why SmolVLA needs a data-prep (rename) step

The ManiGuard datagen export uses **flat, non-standard keys** (`image_left`,
`wrist_image`, `state`, `actions`, …) and ships **5 camera streams**. openpi and
GR00T consume those flat keys through an indirection layer (openpi's
`RepackTransform`, GR00T's `modality.json` `original_key`). `lerobot-train` has no
such indirection — it classifies features purely by standard key prefix. So a
one-time prep step creates a **2-camera, standard-keyed v3.0** copy of the dataset:

| datagen (source) | → SmolVLA (standard) | note |
|---|---|---|
| `image_<external_cam>` | `observation.images.top` | chosen overview (default `left`) |
| `wrist_image` | `observation.images.wrist` | wrist |
| `state` (8-D) | `observation.state` | absolute joint |
| `actions` (8-D) | `action` | absolute joint target |
| `image_opposite` / `image_right` / `image_left_shoulder` | — | dropped |
| `actions_commanded` | — | dropped |

The preparation script copies the selected videos without re-encoding, renames
Parquet columns and statistics without changing state/action values, and converts
the copy from v2.1 to v3.0 with one episode per video file. The five-camera release
remains v2.1. Choose a fresh output directory; existing output and converter
intermediate directories are not overwritten.

## Embodiment & schema

- **Base model:** [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (SmolVLM2 backbone + flow-matching action expert)
- **State / action:** absolute joint 8-D (7 arm joints + 1 gripper), padded to SmolVLA's internal width; the model outputs absolute joint targets fed straight to a `JointController` at eval (no delta transform — see [end to end](end_to_end.md))
- **Cameras (2):** `observation.images.top` (overview) + `observation.images.wrist`
- **Tuning:** SmolVLA default — vision encoder **frozen**, action expert trained (**no LoRA**, same "freeze VLM" strategy as the GR00T N1.6 path)

## Tooling

| Purpose | Path |
|---|---|
| Embodiment contract (flat → standard key map, dims) | `maniguard/smolvla_sft/embodiment.py` |
| Dataset prep (5-cam flat → 2-cam standard-keyed copy) | `tools/smolvla_sft/prepare_dataset.py` |
| SFT launcher (wraps `lerobot-train`) | `tools/smolvla_sft/run_sft.sh` |
| Push checkpoint + card to HF | `tools/smolvla_sft/push_to_hf.py` |
| End-to-end 6-family driver | `tools/smolvla_sft/run_all.sh` |

## Runtime setup

Use Python 3.12 and the pinned upstream LeRobot source (package version 0.5.1):
[`1396b9fab7aecddd10006c33c47a487ffdcb54b4`](https://github.com/huggingface/lerobot/commit/1396b9fab7aecddd10006c33c47a487ffdcb54b4).
The supplied runtime patch bounds video-decoder caching and applies the configured
tokenizer length when loading pretrained processors.

From the ManiGuard repository root, in a dedicated training environment:

```bash
MANIGUARD_ROOT="$(pwd)"
git clone https://github.com/huggingface/lerobot.git /path/to/lerobot
git -C /path/to/lerobot checkout 1396b9fab7aecddd10006c33c47a487ffdcb54b4
git -C /path/to/lerobot apply "$MANIGUARD_ROOT/tools/smolvla_sft/lerobot-runtime.patch"
python -m pip install -e '/path/to/lerobot[smolvla]'
```

FFmpeg must be available for dataset conversion. The default training video
backend is `torchcodec`; set `VIDEO_BACKEND=pyav` to use the PyAV backend.

## Running

Prepare a training copy in the LeRobot environment:

```bash
python tools/smolvla_sft/prepare_dataset.py \
  --src /path/to/datagen-clutter-v1-joint-5cam \
  --out /path/to/clutter-smolvla-v3 \
  --repo-id maniguard/clutter --external-cam left
```

Then launch training with the chosen schedule:

```bash
bash tools/smolvla_sft/run_sft.sh \
  --dataset /path/to/clutter-smolvla-v3 --repo-id maniguard/clutter \
  --output /path/to/checkpoints --steps 20000 --batch 64 --gpus 1 --exp-name clutter
```

`--batch` is per GPU. `--gpus` defaults to 1; larger values launch Accelerate DDP.
The launcher maps `observation.images.top` and `observation.images.wrist` to the
base model's `camera1` and `camera2` keys and saves that mapping with the processors.
State/action vectors remain eight-dimensional; state padding occurs inside the
model. `BASE_MODEL` can name a Hub model or a local checkpoint directory.

The public launcher uses `HF_TOKEN` for base-model access and `WANDB_API_KEY` for
its online training logs. Training saves locally with automatic Hub publication
disabled; publishing checkpoints is a separate operation.
