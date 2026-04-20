"""Register Sentinel-Lite's Pi0.5 TrainConfigs with RLinf.

RLinf keeps all OpenPI configs in a module-level dict
(`_CONFIGS_DICT` in `rlinf.models.embodiment.openpi.dataconfig`). Rather
than editing that file directly -- which would conflate our deltas with
the vendored RLinf source -- we mutate the dict at import time from a
module that lives in our own repo.

The two entries we add mirror what used to live in RLinf:
  * pi05_sentinel_goblet         - the concrete goblet-on-plate SFT config.
  * pi05_sentinel_clutter_family - a parametric family config that reads
                                   repo_id / asset_id from env vars, so
                                   one TrainConfig can back every task
                                   family (clutter, stack, transfer, ...).

This module registers on import. Import order:
    python -c "import sentinel; import rlinf..."
or
    PYTHONPATH=$REPO_ROOT python ... and set a hydra pre-hook.
The launcher scripts in tools/run_{sft,rl}.sh do the former.
"""

from __future__ import annotations

import os

import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

# Pull RLinf's registry dict + its OmniGibsonDataConfig. These are the
# only things we touch from the RLinf source tree.
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
from sentinel.openpi.omnigibson_dataconfig import OmniGibsonDataConfig


# Pi0.5 base checkpoint path -- can be overridden via env var so the same
# config works across workstations / clusters without editing code.
_PI05_BASE = os.environ.get(
    "SENTINEL_PI05_BASE",
    "/home/nu-ideas-4080/Desktop/projects/SENTINEL-Lite/vla_models/RLinf-pi05-SFT-Stack-cube",
)


def _build_goblet_config() -> TrainConfig:
    """Concrete SFT for the current goblet→plate teleop dataset."""
    return TrainConfig(
        name="pi05_sentinel_goblet",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=OmniGibsonDataConfig(
            repo_id="sentinel/goblet_pick_place",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir=f"{_PI05_BASE}/assets",
                asset_id="sentinel_goblet_pick_place",
            ),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
        pytorch_weight_path=_PI05_BASE,
        num_train_steps=5_000,
    )


def _build_clutter_family_config() -> TrainConfig:
    """Parametric SFT config for any Sentinel task family.

    Reads SENTINEL_LEROBOT_REPO_ID and SENTINEL_ASSET_ID from the
    environment so the same config handles clutter, stack, transfer, etc.
    """
    repo_id = os.environ.get(
        "SENTINEL_LEROBOT_REPO_ID", "sentinel/clutter_pickup_v1"
    )
    asset_id = os.environ.get(
        "SENTINEL_ASSET_ID", "sentinel_clutter_pickup_v1"
    )
    return TrainConfig(
        name="pi05_sentinel_clutter_family",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=OmniGibsonDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir=f"{_PI05_BASE}/assets",
                asset_id=asset_id,
            ),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
        pytorch_weight_path=_PI05_BASE,
        num_train_steps=5_000,
    )


def register_sentinel_configs() -> None:
    """Mutate RLinf's _CONFIGS_DICT to include our entries.

    Idempotent: re-registering is a no-op overwrite. Safe to import
    multiple times.
    """
    for builder in (_build_goblet_config, _build_clutter_family_config):
        cfg = builder()
        _CONFIGS_DICT[cfg.name] = cfg


# Register on import so `import sentinel` is enough to make the
# configs visible to RLinf.
register_sentinel_configs()
