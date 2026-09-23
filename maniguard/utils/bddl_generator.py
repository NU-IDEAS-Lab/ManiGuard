"""Re-export task-specification helpers from maniguard.utils.task_spec."""

from maniguard.utils.task_spec import *
from maniguard.utils.task_spec import (  # noqa: F401 — explicit re-exports for type checkers
    _load_footprint_catalog,
    _make_spawn_spec,
    _pick_model_for_synset,
    _synset_to_category,
)
