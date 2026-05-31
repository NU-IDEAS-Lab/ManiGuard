#!/usr/bin/env python3
"""Launcher: register ManiGuard SFT configs, then run openpi's
``scripts/compute_norm_stats.py``.

Same mechanism as ``train.py`` (this dir): import ``maniguard.openpi_sft`` to
register the TrainConfigs into openpi's ``_CONFIGS_DICT``, then delegate to
openpi's pristine script. Norm stats must be recomputed for each ManiGuard
dataset because the sim configs warm-start from ``pi05_base`` (which ships no
norm_stats).

openpi's compute_norm_stats.py is ``tyro.cli(main)`` over
``main(config_name, max_frames)``, so tyro exposes the config name as the FLAG
``--config-name`` (not a positional). It writes stats to ``config.assets_dirs``
(= ``assets_base_dir / config_name``) and does NOT expose ``--assets-base-dir``.
To keep norm-stats and training writing/reading the SAME assets dir we intercept
``--assets-base-dir`` here, apply it to the registered config, strip it from
argv, and pass ``--config-name`` straight through (openpi stays pristine).

Usage (run with openpi's venv):
    OPENPI_ROOT=/path/to/openpi uv run python tools/openpi_sft/compute_norm_stats.py \
        --config-name pi05_base_dusty_transfer_joint_2cam_lora [--assets-base-dir DIR]

``OPENPI_ROOT`` defaults to a sibling ``openpi/`` next to the ManiGuard repo.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import runpy
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import maniguard.openpi_sft  # noqa: E402,F401  -- registers TrainConfigs on import

# Intercept --assets-base-dir (openpi's norm-stats script doesn't accept it) and
# apply it to the registered config so its output dir matches what training reads.
# Everything else -- notably --config-name VALUE -- is passed through unchanged.
_argv = sys.argv[1:]
_assets_base_dir = None
_config_name = None
_kept = []
_i = 0
while _i < len(_argv):
    if _argv[_i] == "--assets-base-dir":
        _assets_base_dir = _argv[_i + 1]
        _i += 2
    elif _argv[_i] == "--config-name":
        _config_name = _argv[_i + 1]
        _kept.extend(_argv[_i : _i + 2])  # keep it for openpi's tyro.cli
        _i += 2
    else:
        _kept.append(_argv[_i])
        _i += 1
sys.argv = [sys.argv[0], *_kept]

if _assets_base_dir is not None:
    if _config_name is None:
        raise SystemExit("compute_norm_stats: --assets-base-dir requires --config-name")
    from openpi.training.config import _CONFIGS_DICT

    _CONFIGS_DICT[_config_name] = dataclasses.replace(
        _CONFIGS_DICT[_config_name], assets_base_dir=_assets_base_dir
    )

OPENPI_ROOT = os.environ.get("OPENPI_ROOT", str(REPO_ROOT.parent / "openpi"))
_script = os.path.join(OPENPI_ROOT, "scripts", "compute_norm_stats.py")
if not os.path.isfile(_script):
    raise FileNotFoundError(
        f"openpi compute_norm_stats.py not found at {_script}; set OPENPI_ROOT to your openpi clone."
    )

runpy.run_path(_script, run_name="__main__")
