"""Sentinel-Lite Python package (namespace only).

No side-effects on import. Submodules with RLinf/OpenPI dependencies
(``sentinel.rlinf.patches``, ``sentinel.openpi``) are opt-in -- import
them explicitly when you need RLinf patched or TrainConfigs
registered. This keeps lightweight consumers like ``sentinel.eval``
and ``sentinel.data`` usable from the ``behavior`` conda env, which
doesn't have ``openpi`` / ``rlinf`` installed.

RLinf-bound processes (main launcher + Ray workers) register via
``sentinel/_autoimport/sitecustomize.py``, which does the explicit
imports so the behavior is triggered once at Python startup.
"""
