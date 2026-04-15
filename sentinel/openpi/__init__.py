"""Sentinel openpi-compatible data/policy transforms + TrainConfigs.

Importing this subpackage triggers three side-effects in order:

  1. ``omnigibson_policy``   - pure transforms, no global registration.
  2. ``omnigibson_dataconfig`` - same.
  3. ``configs``             - registers pi05_sentinel_* TrainConfigs
                               into RLinf's ``_CONFIGS_DICT``.
"""

from sentinel.openpi import omnigibson_dataconfig  # noqa: F401
from sentinel.openpi import omnigibson_policy  # noqa: F401
from sentinel.openpi import configs  # noqa: F401  # registers TrainConfigs
