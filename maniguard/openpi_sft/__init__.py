"""ManiGuard pi0.5 LoRA SFT task configs for the openpi-native trainer.

openpi is consumed as a PRISTINE, parallel clone (``$OPENPI_ROOT``, default
``../openpi``) — it is never edited. This package defines ManiGuard's own task
DataConfigs / policies / TrainConfigs and, at import time, inserts the
TrainConfigs into openpi's ``_CONFIGS_DICT`` via :func:`train_configs.register`.

The launchers in ``tools/openpi_sft/`` import this package first (triggering the
registration below) and then hand off to openpi's own
``scripts/train.py`` / ``scripts/compute_norm_stats.py`` — so all the heavy
lifting (data loading, model, norm-stats, training loop) stays in openpi and we
only contribute the task-specific config.

Importing this package does NOT require OmniGibson: it pulls in
``maniguard/__init__.py`` (which calls the OmniGibson patch ``apply()``), but
that gracefully no-ops when OmniGibson is absent — exactly the case on an
SFT-only compute box.
"""

from maniguard.openpi_sft.train_configs import register

register()
