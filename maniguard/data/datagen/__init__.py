"""ManiGuard datagen — cuRobo-driven automatic SFT trajectory collection on the
finalized bench base tasks. A clean refactor of the earlier pnp/cuRobo pipeline.

Layered like ``bench_builder`` (primitives ↔ per-family skeleton ↔ driver):
  - ``primitives/`` — family-agnostic reusable primitives (scene / cuRobo segment /
    grasp / move-holding / contact / execute / record / obstacles / cameras).
  - ``families/``   — per-family manip skeletons (clutter / lid / jar / cabinet /
    stack / dusty): subtask sequence + per-step waypoints derived from diagnostics.
  - ``driver.py``   — single-task + batch-sweep orchestration.
  - ``data_format`` — the single source of truth for the dataset schema (joint-native;
    5 cameras; state8 / actions8 [achieved] / actions_commanded8; MimicGen sidecar).
"""
