#!/usr/bin/env python3
"""One-shot: push the SFT checkpoint ladder to HF, skipping anything already there.

Run AFTER training (or any time) to backfill checkpoints the sidecar
``hf_push_watcher.py`` did not get to. Shares the de-dup logic in
``_hf_push_common``: each step is uploaded only if its local ``params/`` files
are not already complete on HF, so re-running is safe and fast -- if everything
is already up, it uploads nothing.

Uploads ``params/`` + ``assets/`` (norm stats), skips ``train_state/``. The
0-indexed final step is relabeled to the round ``--num-train-steps``.

Usage:
    HF_TOKEN=... python tools/openpi_sft/hf_push.py \
        --ckpt-dir <openpi>/checkpoints/<cfg>/<exp> \
        --repo IDEAS-Lab-Northwestern/<model-repo> \
        --num-train-steps 10000 [--readme path/to/card.md] [--private]
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hf_push_common import (  # noqa: E402
    finalized_steps,
    is_pushed_complete,
    push_step,
    remote_label,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="exp dir containing <step>/ subfolders")
    ap.add_argument("--repo", required=True, help="HF model repo id")
    ap.add_argument("--num-train-steps", type=int, required=True,
                    help="Used to relabel the 0-indexed final step.")
    ap.add_argument("--readme", default=None, help="Path to a model card to upload as README.md")
    ap.add_argument("--private", action="store_true", help="Create the repo private (default public).")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

    steps = finalized_steps(args.ckpt_dir)
    if not steps:
        sys.exit(f"ERROR: no finalized checkpoints under {args.ckpt_dir}")

    uploaded = skipped = 0
    for step in steps:
        label = remote_label(step, args.num_train_steps)
        local_dir = os.path.join(args.ckpt_dir, str(step))
        if is_pushed_complete(api, args.repo, local_dir, label):
            print(f"  skip {step} -> {label} (already complete on HF)")
            skipped += 1
            continue
        print(f"  upload {step} -> {label}")
        push_step(api, args.repo, local_dir, label)
        uploaded += 1

    if args.readme and os.path.isfile(args.readme):
        api.upload_file(
            path_or_fileobj=args.readme, path_in_repo="README.md",
            repo_id=args.repo, repo_type="model", commit_message="model card",
        )
        print("  uploaded README.md")

    print(f"\nDone. uploaded={uploaded} skipped={skipped} (total {len(steps)}) -> {args.repo}")


if __name__ == "__main__":
    main()
