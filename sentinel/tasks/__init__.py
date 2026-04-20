"""Sentinel-owned OmniGibson task subclasses.

Importing this package registers the sentinel task classes with
``omnigibson.tasks.task_base.REGISTERED_TASKS`` via ``Registerable``, so task
configs (yaml / dicts) can reference them by class name (e.g.
``type: SentinelGraspTask``).
"""

from sentinel.tasks.sentinel_grasp_task import SentinelGraspTask

__all__ = ["SentinelGraspTask"]
