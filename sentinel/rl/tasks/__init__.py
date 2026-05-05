"""Custom sentinel RL tasks. Importing this package auto-registers them
with OmniGibson's task registry so configs like
``{"type": "PickAndLiftTask", ...}`` resolve via the standard OG path.
"""

from omnigibson.tasks.task_base import REGISTERED_TASKS

from sentinel.rl.tasks.pick_and_lift import PickAndLiftTask
from sentinel.rl.tasks.pick_and_lift_privileged import PickAndLiftPrivilegedTask

REGISTERED_TASKS[PickAndLiftTask.__name__] = PickAndLiftTask
REGISTERED_TASKS[PickAndLiftPrivilegedTask.__name__] = PickAndLiftPrivilegedTask

__all__ = ["PickAndLiftTask", "PickAndLiftPrivilegedTask"]
