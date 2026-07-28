#!/usr/bin/env python
"""Push a ManiGuard LingBot-VLA 2.0 checkpoint (+ a model card) to the Hugging Face Hub.

Training writes DCP shards plus an exported HF folder per save:
``<output_dir>/**/global_step_<N>/hf_ckpt/``. Only the exported ``hf_ckpt`` is pushed --
the DCP shards and optimizer state are training-only bulk (and would leak local paths).

The upload is made SELF-CONTAINED for eval by adding, alongside the weights:
  * ``maniguard/norm_stats.json``  -- the family's normalization statistics
  * ``maniguard/robot_config.yaml`` -- the feature mapping the policy was trained with
Serving needs both, and pairing them with the weights removes any chance of an eval run
loading a mismatched pair.

Usage:
  python tools/lingbot_sft/push_to_hf.py --run-dir outputs/lingbot_sft/runs/clutter \
      --family clutter --repo IDEAS-Lab-Northwestern/lingbot-vla2-datagen-v1-clutter-joint-2cam-yanZ
  # a specific rung instead of the last one:
  #   --step 1775        (or --ckpt <path to a hf_ckpt dir>)
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys

from huggingface_hub import HfApi

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Never upload: training-only bulk and anything that embeds local filesystem paths.
_IGNORE = [
    "*.distcp", "*.metadata", ".metadata", "optimizer*", "training_state*", "*optim*",
    "rng_state*", "scheduler*", "trainer_state.json", "wandb*", "images/*", "*.log",
]

CARD = """---
license: apache-2.0
base_model: robbyant/lingbot-vla-v2-6b
tags: [robotics, vla, lingbot-vla-2, maniguard]
---

# LingBot-VLA 2.0 — ManiGuard datagen-v1 `{family}` (joint, 2-cam)

LingBot-VLA 2.0 post-trained on the ManiGuard `{family}` family, one of five base models
evaluated on ManiGuard-Bench under identical data, cameras, and controller.

- **Warm start:** [`robbyant/lingbot-vla-v2-6b`]( https://huggingface.co/robbyant/lingbot-vla-v2-6b )
  — the **pretrain** release (not the RoboTwin post-trained variant).
- **Data:** `{data_repo}` (LeRobot v2.1, consumed directly — no format conversion).
- **Inputs:** 2 cameras (`camera_top` = the `left` overview, `camera_wrist_left` = wrist),
  8-D joint state (7 arm + gripper) mapped into the 55-D unified vector.
- **Actions:** **absolute** joint targets (`subtract_state: false` on both features, per
  LingBot's own simulation recipe) — apply directly to a JointController, no delta step.
- **Recipe:** upstream post-training config unchanged (MoE + depth/DINO distillation on);
  global batch 256 = micro 32 x 8 GPUs, lr 5e-5 cosine, **2 epochs** ({steps} steps).
- **Ladder:** checkpoints every {save_steps} steps.

Serving needs `maniguard/norm_stats.json` + `maniguard/robot_config.yaml`, both included here.
"""

FRAMES = {"clutter": 901_520, "cabinet": 4_172_962, "stack": 2_652_083,
          "jar": 946_870, "lid": 1_055_142, "dusty": 1_879_498}


def find_ckpt(run_dir: pathlib.Path, step: int | None) -> pathlib.Path:
    cands = [pathlib.Path(p) for p in glob.glob(str(run_dir / "**" / "global_step_*" / "hf_ckpt"), recursive=True)]
    cands = [c for c in cands if c.is_dir()]
    if not cands:
        sys.exit(f"no */global_step_*/hf_ckpt under {run_dir}")

    def step_of(p: pathlib.Path) -> int:
        m = re.search(r"global_step_(\d+)", str(p))
        return int(m.group(1)) if m else -1

    if step is not None:
        for c in cands:
            if step_of(c) == step:
                return c
        sys.exit(f"step {step} not found; available: {sorted(step_of(c) for c in cands)}")
    return max(cands, key=step_of)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="target HF model repo id")
    ap.add_argument("--family", required=True, choices=sorted(FRAMES), help="ManiGuard family")
    ap.add_argument("--run-dir", help="training output_dir (the newest rung is pushed)")
    ap.add_argument("--ckpt", help="explicit hf_ckpt dir (overrides --run-dir/--step)")
    ap.add_argument("--step", type=int, help="push this rung instead of the newest")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else find_ckpt(pathlib.Path(args.run_dir), args.step)
    if not (ckpt / "config.json").is_file():
        sys.exit(f"{ckpt} does not look like an exported hf_ckpt (no config.json)")
    step = int(re.search(r"global_step_(\d+)", str(ckpt)).group(1)) if re.search(r"global_step_(\d+)", str(ckpt)) else 0

    norm = REPO_ROOT / "assets" / "norm_stats" / f"maniguard_{args.family}.json"
    robot_cfg = REPO_ROOT / "configs" / "robot_configs" / "maniguard.yaml"
    for p in (norm, robot_cfg):
        if not p.is_file():
            sys.exit(f"missing {p} -- eval needs it packaged with the weights")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    print(f"[push] {ckpt}  (step {step}) -> {args.repo}")
    api.upload_folder(repo_id=args.repo, folder_path=str(ckpt), repo_type="model",
                      ignore_patterns=_IGNORE, commit_message=f"{args.family}: step {step}")
    for src, dst in ((norm, "maniguard/norm_stats.json"), (robot_cfg, "maniguard/robot_config.yaml")):
        api.upload_file(repo_id=args.repo, path_or_fileobj=str(src), path_in_repo=dst,
                        repo_type="model", commit_message=f"{args.family}: {dst}")
        print(f"[push] + {dst}")

    steps = {"clutter": 7100, "cabinet": 32650, "stack": 20750,
             "jar": 7400, "lid": 8250, "dusty": 14700}[args.family]
    card = CARD.format(family=args.family, data_repo=f"IDEAS-Lab-Northwestern/datagen-{args.family}-v1-joint-5cam",
                       steps=f"{steps:,}", save_steps=f"{(steps + 3) // 4:,}")
    api.upload_file(repo_id=args.repo, path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_type="model", commit_message=f"{args.family}: model card")
    print(f"[push] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
