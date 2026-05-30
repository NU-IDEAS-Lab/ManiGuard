#!/usr/bin/env python3
"""Upload SFT checkpoints to HF as they finalize, in parallel with training.

Runs as a sidecar process (``run_sft.sh`` launches it in the background before
training): it polls the checkpoint dir, and the moment a checkpoint finalizes on
disk -- and is not already complete on HF -- it uploads that checkpoint's
``params/`` + ``assets/`` (skipping ``train_state/``). This way checkpoints reach
HF during the run, not only after it finishes, without ever blocking the GPU
(pure filesystem reads + a separate process).

De-dup is shared with the one-shot ``hf_push.py`` via ``_hf_push_common``: both
decide "already pushed" by comparing the local ``params/`` filename set against
the live HF repo, so nothing is uploaded twice. Run ``hf_push.py`` afterwards to
backfill anything the watcher missed (it will skip everything already complete).

Exits automatically once the final step (relabeled to ``num_train_steps``) is
complete on HF, so ``run_sft.sh`` can ``wait`` on it.

Usage:
    HF_TOKEN=... python tools/openpi_sft/hf_push_watcher.py \
        --ckpt-dir <openpi>/checkpoints/<cfg>/<exp> \
        --repo IDEAS-Lab-Northwestern/<model-repo> \
        --num-train-steps 10000 [--poll-interval 30] [--private]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
                    help="Used to relabel the 0-indexed final step and to know when to exit.")
    ap.add_argument("--poll-interval", type=int, default=30, help="Seconds between scans.")
    ap.add_argument("--private", action="store_true", help="Create the repo private (default public).")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

    final_label = remote_label(args.num_train_steps - 1, args.num_train_steps)
    print(f"[watcher] {args.ckpt_dir} -> {args.repo}  (poll {args.poll_interval}s, "
          f"exit after final step -> {final_label})", flush=True)

    final_done = False
    while not final_done:
        for step in finalized_steps(args.ckpt_dir):
            label = remote_label(step, args.num_train_steps)
            local_dir = os.path.join(args.ckpt_dir, str(step))
            if is_pushed_complete(api, args.repo, local_dir, label):
                continue
            print(f"[watcher] uploading step {step} -> {label}", flush=True)
            try:
                push_step(api, args.repo, local_dir, label)
                print(f"[watcher] done {label}", flush=True)
            except Exception as e:  # noqa: BLE001 -- keep watching; retry next scan
                print(f"[watcher] FAILED {label}: {e} (will retry)", flush=True)
        # Exit once the final checkpoint is confirmed complete on HF.
        final_local = os.path.join(args.ckpt_dir, str(args.num_train_steps - 1))
        if os.path.isdir(final_local) and is_pushed_complete(api, args.repo, final_local, final_label):
            final_done = True
            break
        time.sleep(args.poll_interval)

    print(f"[watcher] final step {final_label} on HF; exiting.", flush=True)


if __name__ == "__main__":
    main()
