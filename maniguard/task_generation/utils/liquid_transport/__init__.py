"""Liquid-transport-specific object pools.

Separate from the clutter pipeline's pools because the liquid pipeline
runs with ``gm.USE_GPU_DYNAMICS = True``, under which any spawned object
carrying a BEHAVIOR particle-modifier ability
(``particleSource``/``particleSink``/``particleApplier``/``particleRemover``)
triggers eager particle-system initialization at ``obj.initialize`` time
that races with physx's GPU pose-buffer setup for newly-added prims.
The clutter pipeline runs CPU dynamics and so isn't affected.

This package mirrors the clutter pool layout: ``build_*.py`` regenerates
``*_pool.json``; ``select.py`` exposes uniform-by-category samplers.
"""
