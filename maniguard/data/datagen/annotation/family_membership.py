"""Family membership for annotation-tool filtering (which bench families grasp a mesh_db object).

An object ``(category/model)`` can be grasped in MULTIPLE bench families (e.g. ``bowl/wtepsx`` is a
grasp target in BOTH ``clutter_pickup`` and ``stack_retrieve``). mesh_db therefore stores a multi-family
``families`` list per object, so the annotation tools (``annotate_tool`` / ``mesh_review`` /
``fix_approach_tags``) can show a shared object under EVERY family it belongs to — not just the one that
happened to extract its mesh last.

The legacy single ``source_task`` string is KEPT (it still names one valid task for ``validate_grasps``
to load the object into) and is used here only as a FALLBACK when ``families`` is absent (a mesh_db
written before this field existed). Nothing here imports sim / heavy deps — every annotation tool can
import it cheaply.
"""
from __future__ import annotations

# family CLI key -> bench-family directory name (with trailing "/" for the legacy prefix match).
FAMILY_STEMS = {
    "clutter": "clutter_pickup/", "jar": "jar_transport/", "lid": "lid_transport/",
    "dusty": "dusty_transfer/", "stack": "stack_retrieve/", "cabinet": "cabinet_pickup/",
}


def obj_families(obj: dict) -> set:
    """Bench-family dir names (e.g. ``'clutter_pickup'``) that grasp this mesh_db object. Prefers the
    multi-family ``families`` list; falls back to the single legacy ``source_task`` prefix."""
    fams = obj.get("families")
    if fams:
        return set(fams)
    st = str(obj.get("source_task", ""))
    return {st.split("/", 1)[0]} if st else set()


def obj_in_family(obj: dict, family_keys) -> bool:
    """True if the mesh_db object belongs to ANY of the given family CLI keys (e.g. ``['clutter']``)."""
    want = {FAMILY_STEMS[f].rstrip("/") for f in family_keys}
    return bool(want & obj_families(obj))
