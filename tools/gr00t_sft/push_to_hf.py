#!/usr/bin/env python
"""Push a GR00T-N1.6 SFT checkpoint + a generated model card to a public HF repo.

Uploads only the inference-relevant files (model safetensors + experiment_cfg +
processor + config); skips the optimizer / scheduler / rng / trainer-state files.
Generates a concise model card from the per-family metadata.

Run inside the Isaac-GR00T venv (needs HF_TOKEN). Usage:

    python tools/gr00t_sft/push_to_hf.py --ckpt <checkpoint_dir> \
        --repo IDEAS-Lab-Northwestern/gr00t-n16-base-stack-retrieve-joint-3cam \
        --title "Stack-Retrieve" --task stack-retrieve \
        --data-repo IDEAS-Lab-Northwestern/sim-stack-retrieve-60-joint-3cam \
        --frames 48208 --epochs 4 --steps 3013 --batch 64
"""

import argparse

from huggingface_hub import HfApi

# Training-only artifacts — not needed for inference, skip to keep the repo lean.
_IGNORE = [
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "wandb_config.json",
]

_CARD = """---
license: apache-2.0
base_model: nvidia/GR00T-N1.6-3B
pipeline_tag: robotics
tags: [robotics, vla, gr00t, gr00t-n1.6, manipulation, maniguard, franka]
---

# GR00T-N1.6 - {title} (joint, 2-cam)

NVIDIA Isaac **GR00T-N1.6-3B** fine-tuned on the ManiGuard **{task}** base task (sim
Franka Panda). Part of the ManiGuard VLA benchmark - GR00T vs pi0.5 on the same task
families with identical data, cameras, and controller.

## Model
- **Base:** [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) - Cosmos-Reason VLM + flow-matching DiT action head
- **Embodiment:** NEW_EMBODIMENT - Franka Panda, **8-D joint** state/action (7 arm joints + 1 gripper)
- **Cameras (2):** image_left (overview) + wrist (256x256)
- **Action:** arm = state-relative chunks, gripper = absolute; 16-step horizon; NON_EEF (joint space)
- **Tuning:** GR00T-N1.6 default - VLM (LLM + visual) **frozen**, train projector + diffusion action head (**no LoRA**)

## Training
- 1x H100, bf16, global batch {batch}, {steps} steps (~{epochs} epochs over {frames:,} frames), cosine LR (peak 1e-4)
- Data: [{data_repo}](https://huggingface.co/datasets/{data_repo}); videos decoded as H.264 for GR00T's torchcodec loader

## Usage
Load with `Gr00tPolicy` from [Isaac-GR00T (n1.6-release)](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release), `--embodiment-tag NEW_EMBODIMENT`. The included `experiment_cfg/` carries the modality config + normalization stats.

> WARNING - Convention (must match at eval): joint-space JointController (absolute joint targets, NON_EEF) + 2 cameras (image_left overview + wrist). A mismatched controller or camera set silently feeds an out-of-distribution input.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir to upload")
    ap.add_argument("--repo", required=True, help="target HF model repo (org/name)")
    ap.add_argument("--title", required=True, help='card title, e.g. "Stack-Retrieve"')
    ap.add_argument("--task", required=True, help="task slug, e.g. stack-retrieve")
    ap.add_argument("--data-repo", required=True, help="source HF dataset repo")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=args.ckpt,
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=_IGNORE,
        commit_message=f"GR00T-N1.6 SFT on {args.task} ({args.epochs} epochs, {args.steps} steps)",
    )
    card = _CARD.format(
        title=args.title,
        task=args.task,
        data_repo=args.data_repo,
        frames=args.frames,
        epochs=args.epochs,
        steps=args.steps,
        batch=args.batch,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="model card",
    )
    print("PUSHED", args.repo)


if __name__ == "__main__":
    main()
