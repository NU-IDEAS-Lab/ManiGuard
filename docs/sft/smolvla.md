# SmolVLA SFT

SFT of HuggingFace LeRobot's **SmolVLA** on the same ManiGuard joint LeRobot v2.1
datasets used for [openpi](openpi.md) and [GR00T](gr00t.md) — the dataset is
model-agnostic; SmolVLA only differs in how the shared cameras/state/action are
presented to it.

SmolVLA is **LeRobot-native**: it is fine-tuned with the upstream `lerobot-train`
CLI and has **no config registry and no embodiment registration**. `lerobot-train`
derives the policy's input/output features straight from the dataset's *standard*
key prefixes (`observation.images.*` → visual, `observation.state` → state,
`action` → action), and the architecture is agnostic to camera count and pads the
state/action vectors to its internal width. So the ManiGuard side is the thinnest
of the three tracks: a small embodiment *contract* plus CLI-wrapping tools.

## Why SmolVLA needs a data-prep (rename) step

The ManiGuard datagen export uses **flat, non-standard keys** (`image_left`,
`wrist_image`, `state`, `actions`, …) and ships **5 camera streams**. openpi and
GR00T consume those flat keys through an indirection layer (openpi's
`RepackTransform`, GR00T's `modality.json` `original_key`). `lerobot-train` has no
such indirection — it classifies features purely by standard key prefix. So a
one-time prep step rebuilds a **2-camera, standard-keyed** copy of the dataset:

| datagen (source) | → SmolVLA (standard) | note |
|---|---|---|
| `image_<external_cam>` | `observation.images.top` | chosen overview (default `left`) |
| `wrist_image` | `observation.images.wrist` | wrist |
| `state` (8-D) | `observation.state` | absolute joint |
| `actions` (8-D) | `action` | absolute joint target |
| `image_opposite` / `image_right` / `image_left_shoulder` | — | dropped |
| `actions_commanded` | — | dropped |

Two cameras (overview + wrist) keep parity with the pi0.5 and GR00T tracks
(identical visual input → a fair benchmark). Videos are passthrough (H.264 copied
as-is; AV1 transcoded to H.264 so LeRobot's decoder works everywhere). The mapping
is defined once in `maniguard/smolvla_sft/embodiment.py` (the single source of
truth) and applied by `prepare_dataset.py`.

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

Training runs against a **`huggingface/lerobot` clone** (pinned tag, installed with
the `smolvla` extra) in the same environment that produces the prepared dataset.
Model repos are pushed as
`IDEAS-Lab-Northwestern/smolvla-base-datagen-v1-<fam>-joint-2cam`; run/experiment
names follow `smolvla-base_datagen_v1_<fam>_joint_2cam` (no `_lora` suffix — SmolVLA
does not use LoRA).

## Running

One family, or all six serially (download → prepare → train → push):

```bash
export HF_TOKEN=...  WANDB_API_KEY=...          # both required (pre-flight checked)
bash tools/smolvla_sft/run_all.sh --family clutter
bash tools/smolvla_sft/run_all.sh --all
```

Each family downloads its `datagen-<fam>-v1-joint-5cam` dataset once into the shared
cache (`MANIGUARD_SFT_DATA_ROOT`, default `outputs/sft_datasets/`, shared with the
openpi and GR00T tracks), prepares the 2-cam copy, trains ~2 epochs, and pushes.
Steps derive from each dataset's frame count as `ceil(frames × 2 / batch)`.

## Status / open items

The code layer is complete; the following are confirmed **on the box** at SFT time
(left open here intentionally — see the version-pin caveat in `run_sft.sh`):

- **LeRobot version pin.** The `lerobot-train` flag surface has changed across
  releases (e.g. `--policy.path` vs the older `--policy.type`). Clone
  `huggingface/lerobot` at a pinned tag, `pip install -e .[smolvla]`, and verify the
  flags in `run_sft.sh` against `lerobot-train --help` for that tag before a real
  run. *(Pinned tag: TBD.)*
- **Compute knobs** (batch size, num_workers, save frequency, LR schedule): the
  defaults in `run_all.sh` / `run_sft.sh` are starting points; tune per box.
- **Trained checkpoints + eval numbers**: pending compute.
