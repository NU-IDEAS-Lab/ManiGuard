#!/usr/bin/env python
"""Assemble base SmolVLA weights with a family's SFT serving metadata.

Copy top-level metadata files from the family checkpoint and model.safetensors
from the selected base checkpoint. Then compare base/SFT tensor names and
shapes and check a projection-weight example. These checks occur after files
are copied. Refuse an already populated destination.

This baseline uses target-family preprocessing and normalization statistics
with base-model weights. It is not a data-free zero-shot setup. Structural
compatibility does not establish successful closed-loop behavior."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

from huggingface_hub import snapshot_download

BASE_REPO = "lerobot/smolvla_base"
SFT_REPO = "IDEAS-Lab-Northwestern/smolvla-base-datagen-v1-{family}-joint-2cam"
FAMILIES = ("clutter", "cabinet", "stack", "jar", "lid", "dusty")

# Everything except the weights is taken from the SFT side.
WEIGHTS = "model.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--out", required=True, help="serving directory to create")
    ap.add_argument("--base-repo", default=BASE_REPO)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty -- refusing to overwrite a serving directory")

    sft_repo = SFT_REPO.format(family=args.family)
    print(f"[zs] SFT side  : {sft_repo}")
    print(f"[zs] base side : {args.base_repo}")
    sft_dir = pathlib.Path(snapshot_download(sft_repo))
    base_dir = pathlib.Path(snapshot_download(args.base_repo, allow_patterns=[WEIGHTS, "config.json"]))

    out.mkdir(parents=True, exist_ok=True)
    # 1. everything that describes OUR robot comes from the SFT checkpoint
    copied = []
    for f in sorted(sft_dir.iterdir()):
        if not f.is_file() or f.name.startswith(".") or f.name == WEIGHTS:
            continue
        if f.name in ("README.md",):
            continue
        shutil.copy2(f, out / f.name)
        copied.append(f.name)
    print(f"[zs] from SFT  : {', '.join(copied)}")

    # 2. the weights come from the base model
    shutil.copy2(base_dir / WEIGHTS, out / WEIGHTS)
    print(f"[zs] from base : {WEIGHTS}")

    # 3. prove the swap is what we think it is, rather than trusting the file names
    from safetensors import safe_open

    with safe_open(str(base_dir / WEIGHTS), framework="pt") as fb, \
         safe_open(str(sft_dir / WEIGHTS), framework="pt") as fs:
        kb, ks = set(fb.keys()), set(fs.keys())
        if kb != ks:
            sys.exit(f"key sets differ ({len(kb - ks)} base-only, {len(ks - kb)} sft-only) -- "
                     "the checkpoints are not structurally identical, do not serve this")
        shape_bad = [k for k in kb if fb.get_slice(k).get_shape() != fs.get_slice(k).get_shape()]
        if shape_bad:
            sys.exit(f"{len(shape_bad)} tensors differ in shape, e.g. {shape_bad[:3]}")
        # the weights must actually BE the base ones, not the SFT ones
        probe = "model.action_out_proj.weight"
        if probe in kb:
            import torch
            same = torch.equal(fb.get_tensor(probe), fs.get_tensor(probe))
            if same:
                sys.exit(f"{probe} is identical in base and SFT -- the SFT run did not change it, "
                         "so this assembly would not be a zero-shot baseline. Investigate.")
        print(f"[zs] verified  : {len(kb)} tensors, identical keys and shapes, "
              f"{probe} differs between base and SFT as expected")

    cfg = json.loads((out / "config.json").read_text())
    print(f"[zs] config    : type={cfg.get('type')} "
          f"tokenizer_max_length={cfg.get('tokenizer_max_length')} "
          f"chunk={cfg.get('chunk_size')} n_action_steps={cfg.get('n_action_steps')}")
    print(f"[zs] ready     : {out}")
    print("[zs] NOTE: smoke one cell before committing a wave -- the arm should move without "
          "approaching the target, not freeze and not saturate.")


if __name__ == "__main__":
    main()
