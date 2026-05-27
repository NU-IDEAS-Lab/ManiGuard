"""Build lid_cap_container_pairs.json — inventory of every (lid|cap, container)
attachment pair with per-side status + verdict.

Source of pairings:
  behavior-1k/bddl3/bddl/generated_data/object_inventory.json
  ``attachment_pairs`` field. Each entry is keyed ``<cat>-<model>`` and has
  M/F lists of "parent" attachment ids; we resolve M-side items (lid/cap)
  back to their F-side containers.

Per-side fields:
  * ``status``               — from ``docs/graspability_classified.csv``
                               (graspable / no_grasp / not_ready /
                               too_large / no_metadata / degenerate_bbox /
                               <not in CSV>). The literal value
                               ``manually_added`` is reserved: hand-edit a
                               status to that and the downstream compat
                               builder will accept the pair regardless.
  * ``complaint_unresolved`` — bool, from ``complaints.json``
                               (entries where ``processed: false``).

Verdict logic (per pair):
  * ``kept``               — both sides graspable, neither has a complaint.
  * ``kept_via_relax``     — item graspable + no complaint, container in
                             ``LID_TRANSPORT_NO_GRASP_OK`` (kettle,
                             hingeless_jar) and ready (status ∈ {graspable,
                             no_grasp, not_ready} + no complaint). The
                             container is the *destination* in lid-transport
                             tasks and isn't itself manipulated by the
                             robot, so no_grasp on the container is
                             tolerable for these wide-mouth categories.
  * ``dropped_item_*``      — item graspable check failed.
  * ``dropped_container_*`` — item OK but container check failed.

Output schema (category-first → model-keyed):
  {
    "metadata": {... source paths, counts ...},
    "lid": {
      "<lid_model>": {
        "item_status": "...", "complaint_unresolved": bool,
        "containers": [
          {"category": "...", "model": "...",
           "status": "...", "complaint_unresolved": bool,
           "verdict": "kept" | "kept_via_relax" | "dropped_..." }
        ]
      }, ...
    },
    "cap": { ... }
  }
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
GRASP_CSV = ROOT / "docs" / "graspability_classified.csv"
COMPLAINTS_PATH = (
    ROOT / "behavior-1k" / "bddl3" / "bddl" / "generated_data" / "complaints.json"
)
INVENTORY_PATH = (
    ROOT / "behavior-1k" / "bddl3" / "bddl" / "generated_data" / "object_inventory.json"
)
OUT_PATH = HERE / "lid_cap_container_pairs.json"

ITEM_CATEGORIES = ("lid", "cap")

# Container categories where the lid sits on a destination the robot does
# not need to grasp. Bypass the container-graspability check for these
# (provided the model is otherwise ready).
LID_TRANSPORT_NO_GRASP_OK = frozenset({"kettle", "hingeless_jar"})

# Non-failing statuses the relax accepts (excludes too_large, no_metadata,
# degenerate_bbox — those mean the asset itself is unusable).
RELAX_ACCEPTABLE_STATUSES = frozenset({"graspable", "no_grasp", "not_ready"})

MANUAL_STATUS = "manually_added"


def load_status_map():
    out = {}
    with open(GRASP_CSV) as f:
        for r in csv.DictReader(f):
            out[(r["category"], r["model"])] = r["status"]
    return out


def load_unresolved_complaints():
    out = set()
    for c in json.load(open(COMPLAINTS_PATH)):
        if c.get("processed", False):
            continue
        key = c["object"]
        if "-" not in key:
            continue
        cat, model = key.rsplit("-", 1)
        out.add((cat, model))
    return out


def resolve_pairs(item_cat, ap, female_by_parent):
    """For an item category, return [(item_model, [(cont_cat, cont_model), ...])]."""
    out = []
    for obj_key, sides in ap.items():
        ck, _, mk = obj_key.rpartition("-")
        if ck != item_cat:
            continue
        containers = []
        for parent_id in sides.get("M", []):
            containers.extend(female_by_parent.get(parent_id, []))
        # Dedupe while preserving order
        seen = set()
        deduped = []
        for c in containers:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        out.append((mk, deduped))
    return out


def classify_item(item_cat, item_model, status_map, unresolved):
    st = status_map.get((item_cat, item_model), "<not in CSV>")
    return st, ((item_cat, item_model) in unresolved)


def classify_container(cont_cat, cont_model, status_map, unresolved):
    st = status_map.get((cont_cat, cont_model), "<not in CSV>")
    return st, ((cont_cat, cont_model) in unresolved)


def verdict_for(item_st, item_unres, cont_cat, cont_st, cont_unres):
    # Item-side gates first.
    if item_st == MANUAL_STATUS:
        item_ok = True
    elif item_st != "graspable":
        return f"dropped_item_{item_st}"
    elif item_unres:
        return "dropped_item_complaint"
    else:
        item_ok = True

    # Container-side: graspable wins outright.
    if cont_st == "graspable" and not cont_unres:
        return "kept"
    if cont_st == MANUAL_STATUS:
        return "kept_via_relax"
    # Relax for the lid-transport destination categories.
    if cont_cat in LID_TRANSPORT_NO_GRASP_OK \
            and cont_st in RELAX_ACCEPTABLE_STATUSES \
            and not cont_unres:
        return "kept_via_relax"
    # Otherwise: explain why container dropped.
    if cont_unres:
        return "dropped_container_complaint"
    return f"dropped_container_{cont_st}"


def main():
    status_map = load_status_map()
    unresolved = load_unresolved_complaints()
    inv = json.load(open(INVENTORY_PATH))
    ap = inv["attachment_pairs"]

    # F-side index: parent_id → list of (cat, model) containers.
    female_by_parent = defaultdict(list)
    for obj_key, sides in ap.items():
        cat, _, model = obj_key.rpartition("-")
        for parent_id in sides.get("F", []):
            female_by_parent[parent_id].append((cat, model))

    out_doc = {"metadata": {
        "source_attachment_pairs":
            "behavior-1k/bddl3/bddl/generated_data/object_inventory.json",
        "graspability_csv": "docs/graspability_classified.csv",
        "complaints": "behavior-1k/bddl3/bddl/generated_data/complaints.json",
        "lid_transport_no_grasp_ok": sorted(LID_TRANSPORT_NO_GRASP_OK),
        "manual_status_marker": MANUAL_STATUS,
    }}

    verdict_counts = defaultdict(lambda: defaultdict(int))
    for item_cat in ITEM_CATEGORIES:
        pairs = resolve_pairs(item_cat, ap, female_by_parent)
        per_model = {}
        for item_model, containers in pairs:
            item_st, item_unres = classify_item(
                item_cat, item_model, status_map, unresolved)
            cont_entries = []
            for cont_cat, cont_model in containers:
                cont_st, cont_unres = classify_container(
                    cont_cat, cont_model, status_map, unresolved)
                v = verdict_for(item_st, item_unres,
                                cont_cat, cont_st, cont_unres)
                verdict_counts[item_cat][v] += 1
                cont_entries.append({
                    "category": cont_cat,
                    "model": cont_model,
                    "status": cont_st,
                    "complaint_unresolved": cont_unres,
                    "verdict": v,
                })
            per_model[item_model] = {
                "item_status": item_st,
                "complaint_unresolved": item_unres,
                "containers": cont_entries,
            }
        # Sort by model id for diff-friendly output.
        out_doc[item_cat] = {k: per_model[k] for k in sorted(per_model)}

    out_doc["metadata"]["verdict_counts"] = {
        k: dict(v) for k, v in verdict_counts.items()
    }
    out_doc["metadata"]["totals"] = {
        item_cat: {
            "item_models": len(out_doc[item_cat]),
            "container_pairs": sum(
                len(v["containers"]) for v in out_doc[item_cat].values()),
        }
        for item_cat in ITEM_CATEGORIES
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)

    # Console summary.
    print(f"Wrote {OUT_PATH}")
    for item_cat in ITEM_CATEGORIES:
        t = out_doc["metadata"]["totals"][item_cat]
        print(f"\n{item_cat}: {t['item_models']} item models, "
              f"{t['container_pairs']} pair entries")
        for v, n in sorted(verdict_counts[item_cat].items()):
            print(f"  {v:42s} {n}")


if __name__ == "__main__":
    main()
