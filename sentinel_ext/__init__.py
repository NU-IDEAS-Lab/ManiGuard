"""Sentinel-Lite extensions to RLinf.

Import this package before RLinf's training entry points so that:

  * ``sentinel_ext.rlinf_patches`` patches RLinf's env registry, env
    factory, and config validator to recognise ``env_type: sentinel``
    and bring up our :class:`SentinelEnv` without touching RLinf source.
  * ``sentinel_ext.openpi_configs`` registers the Sentinel-specific
    Pi0.5 ``TrainConfig`` entries into RLinf's OpenPI config dict.

Both submodules apply their changes as import side-effects, so a single
``import sentinel_ext`` is enough.
"""

from sentinel_ext import rlinf_patches  # noqa: F401  # patch RLinf surfaces
from sentinel_ext import openpi  # noqa: F401  # register OmniGibson transforms
from sentinel_ext import openpi_configs  # noqa: F401  # register TrainConfigs

__all__ = ["openpi", "openpi_configs", "rlinf_patches"]
