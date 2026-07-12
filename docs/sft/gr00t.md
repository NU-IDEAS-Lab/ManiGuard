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
- cameras: third-person overview(s) + wrist (3-cam datasets)
- trainer: PyTorch / HF Trainer (component-freeze rather than LoRA)

## Tooling

| Purpose | Path |
|---|---|
| `NEW_EMBODIMENT` modality config | `maniguard/gr00t_sft/maniguard_embodiment.py` |
| Dataset prep (ManiGuard LeRobot → GR00T layout) | `tools/gr00t_sft/prepare_dataset.py` |
| SFT launcher | `tools/gr00t_sft/run_sft.sh` |
| Rollout / eval driver | `tools/gr00t_sft/run_rollout.sh` |
| Push checkpoints to HF | `tools/gr00t_sft/push_to_hf.py` |

Training runs against an Isaac-GR00T clone (N1.6). Model repos are pushed as
`IDEAS-Lab-Northwestern/gr00t-n16-base-<task>-joint-3cam`.

!!! note "Status"
    The GR00T track is used for a GR00T-vs-pi0.5 benchmark across the 6 families.
    Fill in the concrete per-box recipe here as it stabilizes.
