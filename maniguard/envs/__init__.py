"""ManiGuard env-package public surface.

ManiGuardEnv (the RLinf-bound env) and EmbodimentProfile were removed when
the RLinf integration was excised. The active RL stack uses
``maniguard.rl.tasks.pick_and_lift.PickAndLiftTask`` directly via
``maniguard/rl/algorithms/{ppo,eval}.py``. For loading frozen scene
snapshots into a plain OmniGibson env, see
``maniguard.envs.frozen_task_runtime.build_env_config``.
"""

__all__: list[str] = []
