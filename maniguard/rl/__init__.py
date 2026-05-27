"""ManiGuard RL — PPO training on OmniGibson with OmniReset-style grasp resets.

Subpackages:

  - ``envs``     — OG env config builders (benchmark scene + grasp-reset scene)
    and SB3 ``VecEnv`` adapters.
  - ``grasps``   — grasp-dataset lifecycle: antipodal sampling, physics-
    validated collection, per-reset IK teleport, gripper metrology.
  - ``tasks``    — task definitions (reward + termination + obs); currently
    ``PickAndLiftTask`` with physical-grasp-mode "holding" detection.
  - ``training`` — PPO entry points + SB3 support (callbacks, extractors,
    rollout sanity test).
"""
