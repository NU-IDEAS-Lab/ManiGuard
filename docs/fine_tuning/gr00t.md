# GR00T (N1.6) SFT

SFT of NVIDIA **GR00T N1.6** on the same ManiGuard joint LeRobot v2.1 datasets
used for [openpi](openpi.md) — the dataset is model-agnostic, GR00T only differs
in how it declares the embodiment and maps the shared cameras/state/action.

## Embodiment & schema

GR00T consumes the ManiGuard sim Franka as a **`NEW_EMBODIMENT`** in **joint
space** (no EEF/IK), matching the dataset's absolute-joint `state` / `actions`
and its cameras. The modality config lives in
`maniguard/gr00t_sft/maniguard_embodiment.py`.

- state / action: absolute joint (8-D), as in [Dataset & config](dataset_and_config.md)
- cameras: one third-person overview (`image_left`) + wrist — **2-cam**, matching the pi0.5 and SmolVLA tracks for benchmark parity (GR00T reads the datagen `image_*` names directly via `modality.json` `original_key`; adding a view back is a one-line change to `VIDEO_KEYS`)
- trainer: PyTorch / HF Trainer (component-freeze rather than LoRA)

## Tooling

| Purpose | Path |
|---|---|
| `NEW_EMBODIMENT` modality config | `maniguard/gr00t_sft/maniguard_embodiment.py` |
| Dataset prep (symlink view of the ManiGuard LeRobot dataset + GR00T `modality.json` / stats; no copy, no transcode) | `tools/gr00t_sft/prepare_dataset.py` |
| Pre-bake norm stats (optional, reused by `prepare_dataset.py --stats-dir`) | `tools/gr00t_sft/bake_stats.py` |
| SFT launcher | `tools/gr00t_sft/run_sft.sh` |
| End-to-end 6-family driver | `tools/gr00t_sft/run_all.sh` |
| Push checkpoints to HF | `tools/gr00t_sft/push_to_hf.py` |

Training runs against an Isaac-GR00T clone (`n1.6-release`). Model repos follow
`<org>/gr00t-n16-datagen-v1-<fam>-joint-2cam` (e.g.
`IDEAS-Lab-Northwestern/gr00t-n16-datagen-v1-clutter-joint-2cam`); training runs
log to the `maniguard-gr00tN1d6` wandb project with the family as the
experiment name. `run_all.sh` drives one family
(`--family <fam>`) or all six serially (download → prepare → ~2-epoch train → push),
sharing the dataset cache (`MANIGUARD_SFT_DATA_ROOT`) with the openpi + SmolVLA tracks.
