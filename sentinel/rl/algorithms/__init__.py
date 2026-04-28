"""Algorithm entry points (PPO, PPO-Lag, FOCOPS, CUPS, …).

Each module is a standalone ``python -m`` entry point that:
  1. Builds CLI via ``sentinel.rl.cli.common`` + algorithm-specific knobs.
  2. Builds the env via ``sentinel.rl.envs.wrappers.build_vec_env``.
  3. Wires shared callbacks (GC, Checkpoint, Wandb, Video).
  4. Constructs the algorithm and calls ``.learn(total_timesteps, callback)``.

Nothing here imports simulation-heavy modules at package-import time — each
entry point imports omnigibson lazily inside ``main()`` so importing the
package for CLI ``--help`` stays fast.
"""
