"""Resolve annotation-database object membership across benchmark families.

Prefer the per-object families list. If it is absent, infer a single family
from source_task. This fallback supports existing annotation databases without
requiring simulation imports.
"""
from __future__ import annotations

# family CLI key -> bench-family directory name (with trailing "/" for source_task prefix matching).
FAMILY_STEMS = {
    "clutter": "clutter_pickup/", "jar": "jar_transport/", "lid": "lid_transport/",
    "dusty": "dusty_transfer/", "stack": "stack_retrieve/", "cabinet": "cabinet_pickup/",
}


def obj_families(obj: dict) -> set:
    """Bench-family dir names (e.g. ``'clutter_pickup'``) that grasp this mesh_db object. Prefers the
    multi-family ``families`` list; falls back to the single ``source_task`` prefix."""
    fams = obj.get("families")
    if fams:
        return set(fams)
    st = str(obj.get("source_task", ""))
    return {st.split("/", 1)[0]} if st else set()


def obj_in_family(obj: dict, family_keys) -> bool:
    """True if the mesh_db object belongs to ANY of the given family CLI keys (e.g. ``['clutter']``)."""
    want = {FAMILY_STEMS[f].rstrip("/") for f in family_keys}
    return bool(want & obj_families(obj))
