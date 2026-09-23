"""ManiGuard tools for manipulation benchmarks, safety monitoring, and data collection.

Importing the package registers the OmniGibson integration hooks. Set
MANIGUARD_SKIP_OMNIGIBSON_PATCH=1 to import lightweight utilities without them.
"""

try:
    # Read the package version supplied at build time.
    from maniguard._version import __version__  # type: ignore[import-not-found]
    from maniguard._version import version as _scm_version
except ImportError:
    # Not installed (e.g. running directly from a fresh clone before
    # ``pip install -e .``). Fall back so ``maniguard.__version__`` is still
    # defined.
    __version__ = "0.0.0+unknown"

from maniguard._omnigibson_patches import apply as _apply_omnigibson_patches

_apply_omnigibson_patches()

del _apply_omnigibson_patches
