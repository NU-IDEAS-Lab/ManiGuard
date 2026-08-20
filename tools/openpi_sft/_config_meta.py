#!/usr/bin/env python3
"""Print one field from a registered config, for run_sft.sh to read defaults.

Resolves a key against the config's policy_metadata first, then falls back to a
top-level TrainConfig attribute. Prints an empty line if absent (so the shell
can test for "unset"). Used by run_sft.sh to default --push-repo / --exp / the
watcher's run length from the config itself, so launching only needs --config.

Usage:
    python tools/openpi_sft/_config_meta.py <config_name> <key>
      hf_repo | hf_private | default_exp   -> from policy_metadata
      num_train_steps                       -> TrainConfig attribute
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OPENPI_ROOT = os.environ.get("OPENPI_ROOT", str(REPO_ROOT.parent / "openpi"))
sys.path.insert(0, os.path.join(OPENPI_ROOT, "src"))

from openpi.training.config import get_config

import maniguard.openpi_sft  # noqa: F401  -- registers configs


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: _config_meta.py <config_name> <key>")
    cfg = get_config(sys.argv[1])
    key = sys.argv[2]
    meta = cfg.policy_metadata or {}
    if key in meta:
        val = meta[key]
    elif hasattr(cfg, key):
        val = getattr(cfg, key)
    else:
        val = ""
    # Booleans -> lowercase for clean shell comparison ([[ "$x" == "true" ]]).
    if isinstance(val, bool):
        val = str(val).lower()
    print(val if val is not None else "")


if __name__ == "__main__":
    main()
