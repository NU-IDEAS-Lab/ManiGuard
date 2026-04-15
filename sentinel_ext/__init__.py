"""Sentinel-Lite extensions to RLinf.

Import this package before RLinf's training entry points to register any
Sentinel-specific OpenPI configs / data configs into RLinf's global
registries. The training scripts themselves live in RLinf and are
unmodified -- we just mutate the registries at import time.
"""

from sentinel_ext import openpi_configs  # noqa: F401  # register side-effect

__all__ = ["openpi_configs"]
