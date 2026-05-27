"""Shared CLI-arg builders for maniguard RL algorithm entry points.

Every algorithm (PPO, PPO-Lag, FOCOPS, CUPS, …) wires the same env / grasp
dataset / wandb / video knobs. Those live here so each algorithm file only
adds its algorithm-specific hyperparameters on top.
"""
