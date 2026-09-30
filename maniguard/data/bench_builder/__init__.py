"""Build and validate ManiGuard-Bench task instances.

The builder finalizes base snapshots and constructs environment, appearance,
language, and object-location variants. Each instance contains a scene snapshot
and diagnostics; construction runs can also produce four review videos.
Robot configuration and camera placement use the shared project utilities.
"""

from maniguard.data.bench_builder.finalize_base import finalize_base_task
from maniguard.data.bench_builder.render import render_task, render_views
from maniguard.data.bench_builder.validate_base import validate_base_task

__all__ = ["finalize_base_task", "render_task", "render_views", "validate_base_task"]
