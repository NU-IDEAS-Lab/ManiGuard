#!/usr/bin/env python
"""Push a GR00T-N1.6 SFT checkpoint + a generated model card to a public HF repo.

Points at the run dir whose ROOT holds the FINAL saved model; uploads only the
inference bundle (model safetensors + config.json + processor/) and skips the
intermediate checkpoint-*/ dirs, the training-only experiment_cfg/ (its conf.yaml/
config.yaml leak absolute server paths and are never read at inference), DeepSpeed
ZeRO state (global_step*/, *_states.pt), and optimizer/scheduler/rng/trainer-state.
Inference (Gr00tPolicy) loads config.json + safetensors via AutoModel and the
normalization stats + modality config from processor/ via AutoProcessor.
Generates a concise model card from the per-family metadata.

Run inside the Isaac-GR00T venv (needs HF_TOKEN). Usage:

    python tools/gr00t_sft/push_to_hf.py --ckpt <checkpoint_dir> \
        --repo IDEAS-Lab-Northwestern/gr00t-n16-datagen-v1-jar-joint-2cam \
        --title "Jar" --task jar \
        --data-repo IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam \
        --frames 946870 --epochs 2 --steps 7400 --batch 256
"""

import argparse

from huggingface_hub import HfApi

# Training-only artifacts — not needed for inference; skip to keep the repo lean. When --ckpt
# is a run dir, "checkpoint-*" drops every intermediate step checkpoint (each re-saves the full
# weights + a ~30GB DeepSpeed global_step*/); the DeepSpeed + resume-state patterns below also
# strip those if --ckpt points straight at a single checkpoint-<step>/ dir.
_IGNORE = [
    "checkpoint-*",       # intermediate HF-Trainer step checkpoints (keep only the final root model)
    "experiment_cfg/*",   # training-only config dump — conf.yaml/config.yaml leak absolute server
    #                       paths (dataset_paths, output_dir); NOT read at inference (Gr00tPolicy
    #                       loads stats + modality from processor/, not here)
    "global_step*",       # DeepSpeed consolidated state dirs
    "*optim_states.pt",   # DeepSpeed ZeRO optimizer shards
    "*model_states.pt",   # DeepSpeed model-state shards
    "zero_to_fp32.py",
    "latest",
    "optimizer.pt",
    "rng_state*.pth",
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

NVIDIA Isaac **GR00T-N1.6-3B** fine-tuned on the ManiGuard **{task}** base task ({domain}).
Part of the ManiGuard VLA benchmark - GR00T vs pi0.5 on the same task families with
identical data, cameras, and controller.

## Model
- **Base:** [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) - Eagle (nvidia/Eagle-Block2A-2B-v2) VLM + flow-matching DiT action head
- **Embodiment:** NEW_EMBODIMENT - Franka Panda, **8-D joint** state/action (7 arm joints + 1 gripper)
- **Cameras (2):** image_left (overview) + wrist ({resolution})
- **Action:** {action_desc}
- **Tuning:** GR00T-N1.6 default - VLM (LLM + visual) **frozen**, train projector + diffusion action head (**no LoRA**)

## Training
- {hardware}, bf16, global batch {batch}, {steps} steps (~{epochs} epochs over {frames:,} frames), cosine LR (peak {lr}, sqrt-scaled), warmup 0.05
- Data: [{data_repo}](https://huggingface.co/datasets/{data_repo}); videos decoded as H.264 for GR00T's torchcodec loader

## Usage
Load with `Gr00tPolicy` from [Isaac-GR00T (n1d6)](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6), `--embodiment-tag NEW_EMBODIMENT`. The included `processor/` carries the normalization stats + modality config.

> WARNING - Convention (must match at eval): {warning} A mismatched controller or camera set silently feeds an out-of-distribution input.
"""

# Domain-dependent card text. The action semantics are NOT cosmetic: a real checkpoint emits
# joint VELOCITY (the arm group is ABSOLUTE, so nothing is added back to the state), while a
# sim one emits absolute joint targets reconstructed from state-relative chunks. Describing
# one as the other misinstructs whoever serves it.
_DOMAIN = {
    False: dict(
        domain="sim Franka Panda",
        resolution="256x256",
        action_desc="arm = state-relative chunks (reconstructed to absolute at inference), "
                    "gripper = absolute; 16-step horizon at 30 fps (0.53 s); NON_EEF (joint space)",
        warning="joint-space JointController (absolute joint targets, NON_EEF) + 2 cameras "
                "(image_left overview + wrist).",
    ),
    True: dict(
        domain="real Franka Panda, DROID-schema teleop",
        resolution="180x320, 16:9 centre-cropped at conversion time",
        action_desc="arm = joint VELOCITY (rad/s, ABSOLUTE representation - no state delta is "
                    "applied), gripper = next-frame target; 16-step horizon at 15 fps (1.07 s); "
                    "NON_EEF (joint space)",
        warning="the policy emits joint VELOCITY in rad/s - the client must apply "
                "`delta = action / 15` with NO clip, and must send 16:9 centre-cropped frames "
                "(the crop is baked into the training data), from 2 cameras "
                "(exterior_image_1_left + wrist_image_left).",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", help="checkpoint dir to upload (not needed with --card-only)")
    ap.add_argument("--repo", required=True, help="target HF model repo (org/name)")
    ap.add_argument("--title", help='card title, e.g. "Stack-Retrieve"')
    ap.add_argument("--task", help="task slug, e.g. stack-retrieve")
    ap.add_argument("--data-repo", help="source HF dataset repo")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--path-in-repo", default="",
                    help="upload into this SUBFOLDER instead of the repo root, e.g. "
                         "'checkpoint-30000'. Use it to add an intermediate rung beside the "
                         "final bundle without overwriting it. Implies no model card: the "
                         "card describes the repo as a whole and lives at the root.")
    ap.add_argument("--rungs", default="",
                    help="comma-separated intermediate steps present as subfolders, e.g. "
                         "'10000,20000,30000'. Adds a ladder section to the card so the rungs "
                         "are discoverable without guessing folder names.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=8, help="cards the run actually used (card text)")
    ap.add_argument("--lr", default="2e-4", help="peak LR the run actually used (card text)")
    ap.add_argument("--real", action="store_true",
                    help="real-robot (DROID schema) checkpoint: the card must state joint "
                         "VELOCITY actions and the 16:9 crop, not sim's absolute joint targets")
    ap.add_argument("--card-only", action="store_true",
                    help="regenerate and upload README.md only; do not re-upload the weights")
    args = ap.parse_args()

    # The card describes the whole repo, so a subfolder upload must not rewrite it -- and the
    # metadata it needs is then irrelevant. Everything else still requires the full set.
    writes_card = not args.path_in_repo
    if writes_card:
        need = [n for n in ("title", "task", "data_repo", "frames", "epochs", "steps")
                if getattr(args, n) is None]
        if need:
            ap.error("missing required card metadata: " + ", ".join("--" + n.replace("_", "-") for n in need))

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    if args.card_only:
        print("card-only: skipping the weight upload")
    elif not args.ckpt:
        ap.error("--ckpt is required unless --card-only is given")
    else:
        msg = (f"GR00T-N1.6 SFT on {args.task} ({args.epochs} epochs, {args.steps} steps)"
               if writes_card else f"add intermediate rung {args.path_in_repo}")
        api.upload_folder(
            folder_path=args.ckpt,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=_IGNORE,
            commit_message=msg,
        )
    if not writes_card:
        print("PUSHED", f"{args.repo}/{args.path_in_repo}")
        return
    hardware = (f"{args.gpus}-card config, DeepSpeed ZeRO-2" if args.gpus > 1
                else "single-card config")
    card = _CARD.format(
        title=args.title,
        task=args.task,
        data_repo=args.data_repo,
        frames=args.frames,
        epochs=args.epochs,
        steps=args.steps,
        batch=args.batch,
        hardware=hardware,
        lr=args.lr,
        **_DOMAIN[args.real],
    )
    if args.rungs:
        rungs = [s.strip() for s in args.rungs.split(",") if s.strip()]
        card += (
            "\n## Checkpoint ladder\n\n"
            f"The repo root is the FINAL model ({args.steps} steps). Earlier rungs are kept as "
            "subfolders so a later checkpoint that overfits can be compared against them:\n\n"
            + "".join(f"- `checkpoint-{s}/`\n" for s in rungs)
            + "\nEach subfolder is a complete inference bundle (weights + `config.json` + the "
            "processor files); load one by pointing `Gr00tPolicy` at "
            "`<repo>/checkpoint-<step>`. The processor is identical across rungs -- the "
            "normalization stats and modality config are fixed at the start of training.\n"
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
