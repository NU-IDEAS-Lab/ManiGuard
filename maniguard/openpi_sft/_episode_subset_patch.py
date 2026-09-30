"""Select a per-task demonstration subset for ManiGuard openpi training.

For each consecutive block of 40 demonstrations, retain the first
ceil(40 * episode_fraction) episodes. The same selection is applied to
training and normalization-statistic computation. The loader checks task-block
size and homogeneity, and maps filtered episode indices to their local positions
when querying LeRobot action windows. Configurations without a fraction use
the standard openpi data loader.
"""

from __future__ import annotations

import logging
import math
from numbers import Real

logger = logging.getLogger(__name__)

# The finalized datagen datasets store exactly 40 episodes per base task,
# written consecutively (task-homogeneous 40-blocks). Asserted at load time.
EPISODES_PER_BASE_TASK = 40


def select_episode_subset(episodes_meta: dict, fraction: float) -> list[int]:
    """Per-base-task first-``ceil(40*fraction)`` episode indices.

    ``episodes_meta``: LeRobotDatasetMetadata.episodes — {episode_index: record}
    with record["tasks"] (list of task strings) and record["length"].
    """
    if isinstance(fraction, bool) or not isinstance(fraction, Real) or not (0.0 < fraction < 1.0):
        raise ValueError(f"episode_fraction must be in (0, 1), got {fraction!r}")
    n = len(episodes_meta)
    if n == 0 or n % EPISODES_PER_BASE_TASK != 0:
        raise ValueError(
            f"episode_fraction requires the {EPISODES_PER_BASE_TASK}-per-base-task layout; "
            f"dataset has {n} episodes; expected a positive multiple of {EPISODES_PER_BASE_TASK}."
        )
    if (any(type(i) is not int for i in episodes_meta)
            or set(episodes_meta) != set(range(n))):
        raise ValueError("episode metadata must contain consecutive integer indices starting at zero")
    ordered = [episodes_meta[i] for i in range(n)]
    for i, rec in enumerate(ordered):
        if not isinstance(rec, dict) or type(rec.get("episode_index")) is not int or rec["episode_index"] != i:
            raise ValueError(f"episode metadata index mismatch at {i}")
        if type(rec.get("length")) is not int or rec["length"] <= 0:
            raise ValueError(f"episode {i} must have a positive integer length")
        tasks = rec.get("tasks")
        tasks = [tasks] if isinstance(tasks, str) else tasks
        if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str) or not tasks[0].strip():
            raise ValueError(f"episode {i} must have one nonempty task label")
    keep = math.ceil(EPISODES_PER_BASE_TASK * fraction)
    selected: list[int] = []
    for b in range(n // EPISODES_PER_BASE_TASK):
        block = ordered[b * EPISODES_PER_BASE_TASK : (b + 1) * EPISODES_PER_BASE_TASK]
        tasks = {t for rec in block for t in (rec["tasks"] if isinstance(rec["tasks"], list) else [rec["tasks"]])}
        if len(tasks) != 1:
            raise ValueError(
                f"episode block {b} (episodes {block[0]['episode_index']}..{block[-1]['episode_index']}) "
                f"mixes {len(tasks)} task strings — not the per-base-task layout; refusing to subset."
            )
        selected.extend(rec["episode_index"] for rec in block[:keep])
    return selected


def _fix_lerobot_filtered_query_indices() -> None:
    """Fix lerobot v2.1's ``episodes=`` filter for non-prefix selections.

    Upstream bug: with ``episodes=[...]``, ``episode_data_index["from"/"to"]``
    are POSITIONAL arrays over the selected episodes (size = len(selected)),
    but ``__getitem__`` passes the ORIGINAL ``episode_index`` stored in the
    parquet row into ``_get_query_indices`` — correct only when the selection
    is a 0-based prefix. Any other selection silently reads the wrong episode
    boundaries (wrong action-chunk windows) and raises IndexError once an
    original index >= len(selected) is reached. Our per-40-block subsets are
    non-contiguous, so we translate original -> position here. ``_query_videos``
    keeps the original index (video paths are named by it) — untouched.
    """
    import lerobot.common.datasets.lerobot_dataset as _lrd

    if getattr(_lrd.LeRobotDataset._get_query_indices, "_maniguard_subset_patch", False):
        return
    _orig_gqi = _lrd.LeRobotDataset._get_query_indices

    def _get_query_indices(self, idx, ep_idx):
        if self.episodes is not None:
            pos_map = getattr(self, "_maniguard_ep_pos", None)
            if pos_map is None:
                pos_map = {orig: pos for pos, orig in enumerate(self.episodes)}
                self._maniguard_ep_pos = pos_map
            ep_idx = pos_map[ep_idx]
        return _orig_gqi(self, idx, ep_idx)

    _get_query_indices._maniguard_subset_patch = True  # type: ignore[attr-defined]
    _lrd.LeRobotDataset._get_query_indices = _get_query_indices


def apply() -> None:
    """Install the wrappers (idempotent): subset creation + filtered-index fix."""
    import openpi.training.data_loader as _dl

    if getattr(_dl.create_torch_dataset, "_maniguard_subset_patch", False):
        return
    _orig = _dl.create_torch_dataset

    def create_torch_dataset(data_config, action_horizon, model_config):
        fraction = getattr(data_config, "episode_fraction", None)
        if fraction is None:
            return _orig(data_config, action_horizon, model_config)

        # Construct the dataset with the selected episode indices.
        try:
            import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        except ImportError as exc:
            raise RuntimeError(
                "ManiGuard episode subsets require the LeRobot v2.1 OpenPI environment "
                "with lerobot.common.datasets"
            ) from exc
        _fix_lerobot_filtered_query_indices()

        import openpi.transforms as _transforms

        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        episodes = select_episode_subset(dataset_meta.episodes, fraction)
        frames = sum(dataset_meta.episodes[i]["length"] for i in episodes)
        logger.info(
            "[episode_subset] %s: fraction=%.2f -> %d/%d episodes "
            "(first %d of every %d-block), %d frames",
            repo_id, fraction, len(episodes), len(dataset_meta.episodes),
            math.ceil(EPISODES_PER_BASE_TASK * fraction), EPISODES_PER_BASE_TASK, frames,
        )
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            episodes=episodes,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        if data_config.prompt_from_task:
            dataset = _dl.TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        return dataset

    create_torch_dataset._maniguard_subset_patch = True  # type: ignore[attr-defined]
    _dl.create_torch_dataset = create_torch_dataset
