"""bench_builder — build the finalized ManiGuard-Bench task dataset.

Standalone module for the 6fam-base -> ManiGuard-Bench rebuild: produces, per family,
one clean empty-scene `base` task plus 4 perturbation variants (env / target / language /
location), all FrankaPanda+longfinger, all rendered from the shared camera_setup (4 views
incl. left_shoulder). Reuses only the shared infra (camera_setup, task_generation.utils.video,
replay_empty); it does NOT import the legacy perturbation generator
(perturbation_scaling / render_task_variants / perturbation_runtime).

Canonical design spec: the Obsidian doc "ManiGuard 6fam_dataset_rebuild_design".
"""

from maniguard.data.bench_builder.render import render_task

__all__ = ["render_task"]
