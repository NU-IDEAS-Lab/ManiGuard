"""Training entry points + SB3 support (callbacks, feature extractors).

  - ``ppo``               — benchmark-scene PPO training
    (``python -m sentinel.rl.training.ppo --scene-dir ...``).
  - ``ppo_grasp_reset``   — grasp-reset setup PPO training
    (``python -m sentinel.rl.training.ppo_grasp_reset --category ... --model ...``).
  - ``rollout_test``      — hand-coded lift policy sanity check on the
    grasp-reset env (no learning, verifies step + reward + termination).
  - ``callbacks``         — GC + safety callbacks invoked during PPO learn.
  - ``extractors``        — ``RGBCombinedExtractor`` feature network.
"""
