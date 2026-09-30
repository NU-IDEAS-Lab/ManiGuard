#!/usr/bin/env python
"""Upload an exported LingBot checkpoint and serving metadata to Hugging Face.

Choose a global_step_<N>/hf_ckpt directory or an explicit checkpoint. Exclude
training state, sanitize path strings in config.json, and add family norm
statistics, a robot mapping with a checkpoint-relative statistics path, and
the supplied VLM configuration. Generate a model card after upload.
Serving still requires compatible LingBot software and a data configuration.

Example:
    python tools/lingbot_sft/push_to_hf.py --family clutter --run-dir outputs/lingbot_sft/runs/clutter --repo organization/model-name"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys

from huggingface_hub import HfApi

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Never upload: training-only bulk and anything that embeds local filesystem paths.
# config.json is excluded here and re-uploaded scrubbed -- see sanitize_config.
_IGNORE = [
    "*.distcp", "*.metadata", ".metadata", "optimizer*", "training_state*", "*optim*",
    "rng_state*", "scheduler*", "trainer_state.json", "wandb*", "images/*", "*.log",
    "config.json",
]


def retarget_norm_stats(path: pathlib.Path) -> bytes:
    """Set the robot mapping's norm_stats field to maniguard/norm_stats.json.

    Retain the field because FeatureTransform expects it even when a loader
    provides an explicit statistics path. Resolve this relative path against
    the downloaded checkpoint root when serving."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    kept = [ln for ln in lines if not ln.startswith("norm_stats:")]
    if len(kept) == len(lines):
        sys.exit(f"{path}: no top-level `norm_stats:` line -- refusing to guess")
    while kept and (kept[-1].startswith("# norm_stats") or not kept[-1].strip()):
        kept.pop()
    block = (
        "# norm_stats points at THIS checkpoint's own statistics (the file shipped beside this\n"
        "# one). It must stay present even when a loader passes an explicit path: LingBot's\n"
        "# FeatureTransform does `robot_config.pop('norm_stats')` with no default on both\n"
        "# branches, so a missing key raises KeyError. The path is relative to this checkpoint's\n"
        "# root -- resolve it yourself, or pass an absolute `norm_stats_path`\n"
        "# (maniguard/serve/lingbot_native.py does the latter).\n"
        "norm_stats: maniguard/norm_stats.json\n"
    )
    return ("".join(kept).rstrip() + "\n\n" + block).encode()


def sanitize_config(path: pathlib.Path) -> bytes:
    """Replace slash-prefixed string values in config.json with outputs/<basename>.

    Walk nested dictionaries and lists, report replacements, and reject output
    that still matches the configured absolute-path patterns."""
    cfg = json.loads(path.read_text())
    scrubbed: list[str] = []

    def walk(node, where=""):
        items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, val in items:
            at = f"{where}.{key}"
            if isinstance(val, (dict, list)):
                walk(val, at)
            elif isinstance(val, str) and val.startswith("/"):
                node[key] = os.path.join("outputs", os.path.basename(val.rstrip("/")) or "run")
                scrubbed.append(f"{at}: {val} -> {node[key]}")

    walk(cfg)
    for line in scrubbed:
        print(f"[push] scrubbed {line}")

    blob = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
    leftover = [ln for ln in blob.splitlines() if re.search(r'"\s*/|/home/|/mnt/|/workspace/|/root/', ln)]
    if leftover:
        sys.exit("refusing to push: absolute paths survived sanitization:\n  " + "\n  ".join(leftover))
    return blob.encode()

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

## Files needed to serve

Everything a rollout needs is in this repo:

| Path | What it is |
| --- | --- |
| `maniguard/norm_stats.json` | this family's normalization statistics |
| `maniguard/robot_config.yaml` | the feature mapping the policy was trained with; its `norm_stats:` points at the file above |
| `vlm/config.json` | the [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) config (Apache-2.0), vendored unchanged — LingBot builds the VLM skeleton from it before loading these weights |
| `tokenizer*.json`, `*preprocessor_config.json`, `merges.txt`, `vocab.json`, `added_tokens.json`, `special_tokens_map.json`, `chat_template.jinja` | the processor / tokenizer, byte-identical to training |

Serve it with ManiGuard's shim, which reads all of the above from the checkpoint directory and
passes the norm statistics explicitly:

```bash
python maniguard/serve/lingbot_native.py \\
    --checkpoint <this repo, downloaded> --qwen-config <dir holding vlm/config.json> \\
    --device cuda:0 --port 8000
```
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
    ap.add_argument("--vlm-config",
                    help="Qwen3-VL-4B-Instruct config.json to vendor as vlm/config.json "
                         "(default: assets/pretrained/Qwen3-VL-4B-Instruct/config.json)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else find_ckpt(pathlib.Path(args.run_dir), args.step)
    if not (ckpt / "config.json").is_file():
        sys.exit(f"{ckpt} does not look like an exported hf_ckpt (no config.json)")
    step = int(re.search(r"global_step_(\d+)", str(ckpt)).group(1)) if re.search(r"global_step_(\d+)", str(ckpt)) else 0

    norm = REPO_ROOT / "assets" / "norm_stats" / f"maniguard_{args.family}.json"
    robot_cfg = REPO_ROOT / "configs" / "robot_configs" / "maniguard.yaml"
    vlm_cfg = pathlib.Path(args.vlm_config) if args.vlm_config else (
        REPO_ROOT / "assets" / "pretrained" / "Qwen3-VL-4B-Instruct" / "config.json")
    for p in (norm, robot_cfg, vlm_cfg):
        if not p.is_file():
            sys.exit(f"missing {p} -- eval needs it packaged with the weights")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    print(f"[push] {ckpt}  (step {step}) -> {args.repo}")
    api.upload_folder(repo_id=args.repo, folder_path=str(ckpt), repo_type="model",
                      ignore_patterns=_IGNORE, commit_message=f"{args.family}: step {step}")
    api.upload_file(repo_id=args.repo, path_or_fileobj=sanitize_config(ckpt / "config.json"),
                    path_in_repo="config.json", repo_type="model",
                    commit_message=f"{args.family}: config (local paths scrubbed)")
    for src, dst in ((norm, "maniguard/norm_stats.json"), (vlm_cfg, "vlm/config.json")):
        api.upload_file(repo_id=args.repo, path_or_fileobj=str(src), path_in_repo=dst,
                        repo_type="model", commit_message=f"{args.family}: {dst}")
        print(f"[push] + {dst}")
    api.upload_file(repo_id=args.repo, path_or_fileobj=retarget_norm_stats(robot_cfg),
                    path_in_repo="maniguard/robot_config.yaml", repo_type="model",
                    commit_message=f"{args.family}: robot_config (norm_stats retargeted)")
    print("[push] + maniguard/robot_config.yaml  (norm_stats -> maniguard/norm_stats.json)")

    steps = {"clutter": 7100, "cabinet": 32650, "stack": 20750,
             "jar": 7400, "lid": 8250, "dusty": 14700}[args.family]
    card = CARD.format(family=args.family, data_repo=f"IDEAS-Lab-Northwestern/datagen-{args.family}-v1-joint-5cam",
                       steps=f"{steps:,}", save_steps=f"{(steps + 3) // 4:,}")
    api.upload_file(repo_id=args.repo, path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_type="model", commit_message=f"{args.family}: model card")
    print(f"[push] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
