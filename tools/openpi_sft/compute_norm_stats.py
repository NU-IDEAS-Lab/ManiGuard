#!/usr/bin/env python3
"""Launcher: register ManiGuard SFT configs, then run openpi's
``scripts/compute_norm_stats.py``.

Same mechanism as ``train.py`` (this dir): import ``maniguard.openpi_sft`` to
register the TrainConfigs into openpi's ``_CONFIGS_DICT``, then delegate to
openpi's pristine script. Norm stats must be recomputed for each ManiGuard
dataset because the sim configs warm-start from ``pi05_base`` (which ships no
norm_stats).

Usage (run with openpi's venv):
    OPENPI_ROOT=/path/to/openpi uv run python tools/openpi_sft/compute_norm_stats.py \
        pi05_base_dusty_transfer_joint_2cam_lora

(openpi's compute_norm_stats.py takes the config name as a positional arg.)
``OPENPI_ROOT`` defaults to a sibling ``openpi/`` next to the ManiGuard repo.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import maniguard.openpi_sft  # noqa: E402,F401  -- registers TrainConfigs on import

OPENPI_ROOT = os.environ.get("OPENPI_ROOT", str(REPO_ROOT.parent / "openpi"))
_script = os.path.join(OPENPI_ROOT, "scripts", "compute_norm_stats.py")
if not os.path.isfile(_script):
    raise FileNotFoundError(
        f"openpi compute_norm_stats.py not found at {_script}; set OPENPI_ROOT to your openpi clone."
    )

runpy.run_path(_script, run_name="__main__")
