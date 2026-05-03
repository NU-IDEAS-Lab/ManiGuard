"""Sentinel-Lite Python package.

Importing ``sentinel`` installs a small set of runtime patches on OmniGibson
(new ``Dropped`` / ``Upright`` object states, a ``Grasped`` alias for
``IsGrasping``, a tensor-safe ``draw_debug_markers``, a ``hold_steps`` option on
``GraspGoal``, and a fallback-link-aware ``GraspReward``). The hooks are the
minimal cost of keeping Sentinel-specific code out of the OmniGibson tree so
OmniGibson can be pinned as an upstream dependency; see
:mod:`sentinel._omnigibson_patches` for details.

Set ``SENTINEL_SKIP_OMNIGIBSON_PATCH=1`` to skip the patches entirely (e.g.
for lightweight pure-Python consumers that don't need OmniGibson).

Heavier opt-in side effects (RLinf patches, OpenPI TrainConfig registration)
still live behind explicit submodule imports — ``sentinel.rlinf.patches`` and
``sentinel.openpi``. RLinf-bound processes register those via
``sentinel/_autoimport/sitecustomize.py`` at Python startup.
"""

try:
    # Written by setuptools-scm at build/install time from the latest ``v*`` tag.
    from sentinel._version import __version__, version as _scm_version  # type: ignore[import-not-found]
except ImportError:
    # Not installed (e.g. running directly from a fresh clone before
    # ``pip install -e .``). Fall back so ``sentinel.__version__`` is still
    # defined.
    __version__ = "0.0.0+unknown"

from sentinel._omnigibson_patches import apply as _apply_omnigibson_patches

_apply_omnigibson_patches()

del _apply_omnigibson_patches
