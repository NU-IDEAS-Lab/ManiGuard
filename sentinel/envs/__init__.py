"""Sentinel env-package public surface.

SentinelEnv (the RLinf-bound env) and EmbodimentProfile were removed when
the RLinf integration was excised. The active RL stack uses
``sentinel.rl.tasks.pick_and_lift.PickAndLiftTask`` directly via
``sentinel/rl/algorithms/{ppo,eval}.py``. For loading frozen scene
snapshots into a plain OmniGibson env, see
``sentinel.envs.frozen_task_runtime.build_env_config``.
"""

__all__: list[str] = []
