"""Auto-import sentinel into every Python subprocess under this PYTHONPATH.

Ray spawns worker processes that import RLinf afresh; those workers do
NOT run our launcher's ``import sentinel``, so the TrainConfig /
env_type / validate_cfg patches are absent and SFT worker init fails
with ``Config 'pi05_sentinel_goblet' not found``.

Python's ``site`` module auto-imports any module named ``sitecustomize``
that it can find on ``sys.path``. By putting this file in a dedicated
directory and prepending that directory to PYTHONPATH, every Python
interpreter launched with that PYTHONPATH (main process, Ray workers,
subprocesses) runs ``import sentinel`` at startup.

Kept intentionally quiet on success and verbose on failure so the patch
chain never silently disappears.
"""

try:
    import sentinel.rlinf.patches  # noqa: F401  # patch RLinf surfaces
    import sentinel.openpi.configs  # noqa: F401  # register TrainConfigs
except Exception as _e:  # pragma: no cover
    import sys

    print(
        f"[sentinel sitecustomize] failed to register: {_e!r}",
        file=sys.stderr,
    )
