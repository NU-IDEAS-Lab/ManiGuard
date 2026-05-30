#!/usr/bin/env python3
"""Print a registered config's num_train_steps. Used by run_sft.sh to tell the
HF-push watcher the run length when ``--steps`` is not overridden on the CLI."""

from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OPENPI_ROOT = os.environ.get("OPENPI_ROOT", str(REPO_ROOT.parent / "openpi"))
sys.path.insert(0, os.path.join(OPENPI_ROOT, "src"))

import maniguard.openpi_sft  # noqa: E402,F401  -- registers configs
from openpi.training.config import get_config  # noqa: E402

if __name__ == "__main__":
    print(get_config(sys.argv[1]).num_train_steps)
