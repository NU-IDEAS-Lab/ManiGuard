"""Diverse-reset state generation for sentinel.rl training.

Modeled on UW Lab OmniReset (ICLR 2026) — "Emergent Dexterity via Diverse
Resets and Large-Scale Reinforcement Learning". The idea: seed PPO episodes
with states sampled from a mixture of difficulty levels (e.g.
object-anywhere-ee-anywhere, object-anywhere-ee-grasped, object-lifted-ee-grasped)
so the policy sees positive reward signal from easier starts before learning
to solve the hardest distribution.
"""
