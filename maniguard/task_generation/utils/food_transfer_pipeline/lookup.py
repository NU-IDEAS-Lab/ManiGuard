"""Runtime lookup of a container's opening drop-XY.

Looks up the offline-derived opening centroid offset stored in
``container_openings.json`` and applies it to the live AABB center.
The offset is in the container's AABB-relative frame, so no orientation
sensitivity and no per-call simulator work — pure JSON lookup + arithmetic.

Cache: the JSON is loaded lazily on first call and reused.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_OPENINGS_PATH = Path(__file__).resolve().parent / "container_openings.json"
_offsets_cache: dict[tuple[str, str], tuple[float, float]] | None = None


def _load_offsets() -> dict[tuple[str, str], tuple[float, float]]:
    global _offsets_cache
    if _offsets_cache is not None:
        return _offsets_cache
    out: dict[tuple[str, str], tuple[float, float]] = {}
    raw = json.loads(_OPENINGS_PATH.read_text())
    for cat, info in raw.get("containers", {}).items():
        for m in info.get("models", []):
            if m.get("scan_status") != "ok":
                continue
            dx, dy = m.get("opening_centroid_xy_relative_to_aabb_center_m",
                           [0.0, 0.0])
            out[(cat, m["model"])] = (float(dx), float(dy))
    _offsets_cache = out
    return out


def container_drop_xy(container) -> tuple[float, float]:
    """Return world XY to drop food at, above the container's cavity opening.

    Looks up the container's (category, model) in container_openings.json
    and applies the cached AABB-relative centroid offset to the live AABB
    center. Falls back to the AABB center if the model has no entry
    (e.g. closed container or scan_status != "ok").
    """
    aabb_min, aabb_max = container.aabb
    cx = 0.5 * (float(aabb_min[0]) + float(aabb_max[0]))
    cy = 0.5 * (float(aabb_min[1]) + float(aabb_max[1]))
    offsets = _load_offsets()
    key = (getattr(container, "category", None), getattr(container, "model", None))
    off = offsets.get(key)
    if off is None:
        log.info("container_drop_xy(%s): no opening entry for %s, "
                 "falling back to AABB center",
                 getattr(container, "name", "?"), key)
        return cx, cy
    return cx + off[0], cy + off[1]
