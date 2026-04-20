"""OmniGibson environment builders + SB3 adapters.

  - ``benchmark_scene.build_config``   — config for a BEHAVIOR benchmark scene
    dir (``scene_ep1.json`` + ``diagnostics.jsonl``).
  - ``grasp_reset_scene.build_config`` — empty ``Scene`` + ``FrankaMounted`` +
    ``breakfast_table`` + one ``DatasetObject`` target; used by the grasp-reset
    RL training / rollout / smoke test.
  - ``sb3_vec``                        — ``SentinelSB3VectorEnvironment`` for
    multi-env rollouts under SB3's ``VecEnv`` API.
  - ``wrappers``                       — gym wrappers (image transposition,
    action clipping, etc.).
"""
