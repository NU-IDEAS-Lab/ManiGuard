"""GR00T N1.6 SFT support for ManiGuard.

Thin wrapper layer around the upstream NVIDIA Isaac-GR00T finetuning code,
consumed as a pristine, separate clone. This package only holds the
ManiGuard-side artifacts:

- ``maniguard_embodiment``: the self-contained GR00T ``NEW_EMBODIMENT`` modality
  config for our sim Franka (8-D joint state/action, 3 cameras) + the matching
  ``meta/modality.json`` body. Depends only on ``gr00t`` so the Isaac-GR00T uv
  venv can exec it directly (it does not have ``maniguard`` installed).

The runnable tooling lives in ``tools/gr00t_sft/`` (``prepare_dataset.py``,
``run_sft.sh``). See ``docs/fine_tuning`` for the end-to-end recipe.

NOTE: importing ``maniguard_embodiment`` triggers ``gr00t`` imports and registers
the embodiment config, so it is intentionally NOT imported here.
"""
