"""Initialize ManiGuard's openpi training configuration support.

Apply the augmentation guard and supported LeRobot decoder fallback, then
register ManiGuard TrainConfigs. The training and normalization wrappers use
these configurations with an external openpi checkout. Importing this package
requires the corresponding openpi, JAX, and augmentation dependencies.
"""

from maniguard.openpi_sft._augmax_patch import apply as _apply_augmax_guard
from maniguard.openpi_sft._lerobot_video_patch import apply as _apply_pyav_backend
from maniguard.openpi_sft._episode_subset_patch import apply as _apply_episode_subset
from maniguard.openpi_sft.train_configs import register

# Neutralize the rare non-finite output of openpi's training-time image
# augmentation (augmax) before any training runs. See _augmax_patch for details.
_apply_augmax_guard()
# Fall back LeRobot video decode to PyAV where torchcodec's system FFmpeg is
# unavailable (no-op where torchcodec works). See _lerobot_video_patch.
_apply_pyav_backend()
_apply_episode_subset()
register()
