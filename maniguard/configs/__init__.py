"""ManiGuard-owned OmniGibson / RL config files.

These used to live under ``OmniGibson/omnigibson/configs/`` as vendored
extensions. Use :func:`config_path` to resolve a filename to an absolute path.
"""

from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent


def config_path(filename: str) -> str:
    """Return the absolute path of a config file shipped with ``maniguard.configs``."""
    return str(_CONFIG_DIR / filename)


__all__ = ["config_path"]
