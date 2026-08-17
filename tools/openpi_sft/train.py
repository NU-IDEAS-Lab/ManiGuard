#!/usr/bin/env python3
"""Launcher: register ManiGuard SFT configs, then run openpi's ``scripts/train.py``.

openpi is consumed as a pristine, parallel clone (never edited). This wrapper:
  1. puts ManiGuard on ``sys.path`` and imports ``maniguard.openpi_sft``, which
     registers ManiGuard's pi0.5 SFT TrainConfigs into openpi's ``_CONFIGS_DICT``;
  2. delegates to openpi's ``scripts/train.py`` via ``runpy`` with argv intact,
     so every openpi train.py flag works unchanged (the config registry is read
     at ``cli()`` call time, after our registration).

Usage (run with openpi's venv, e.g. ``uv run python``):
    OPENPI_ROOT=/path/to/openpi uv run python tools/openpi_sft/train.py \
        pi05_base_dusty_transfer_joint_2cam_lora \
        --exp-name=... [--num-train-steps=...] [--batch-size=...] [--overwrite]

``OPENPI_ROOT`` defaults to a sibling ``openpi/`` next to the ManiGuard repo.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import maniguard.openpi_sft  # noqa: F401  -- registers TrainConfigs on import

OPENPI_ROOT = os.environ.get("OPENPI_ROOT", str(REPO_ROOT.parent / "openpi"))
_train = os.path.join(OPENPI_ROOT, "scripts", "train.py")
if not os.path.isfile(_train):
    raise FileNotFoundError(
        f"openpi train.py not found at {_train}; set OPENPI_ROOT to your openpi clone."
    )

runpy.run_path(_train, run_name="__main__")
