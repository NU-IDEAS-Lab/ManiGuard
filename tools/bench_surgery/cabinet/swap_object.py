"""Swap a cabinet_pickup task's target/obstacle for a donor object already present in the bench.

Rewrites the task's ``base/scene_ep1.json`` (init_info args + object_registry pose, with the role
object RENAMED consistently across the snapshot) + ``base/diagnostics.jsonl`` (``{role}_info`` /
``selection`` / ``goal_conditions`` / ``ltl_safety`` over-globs / ``prompt``). Re-finalize afterward
with ``tools.bench_surgery.cabinet.rerender_base --tasks task_NNNN`` (recomputes cameras/gate/LTL/surface + re-renders
the 4 review videos). Idempotent per (task, role); backs both files up to ``*.bak_swap`` on first touch.

The donor's geometry (scale + expected_file_hash) and its resting clearance are read from a SOURCE base
task where the donor already sits on a cabinet table:
``bottom_clearance = donor_centre_z - source_surface_top_z``; then ``new_z = this.surface_top_z + clearance``
re-seats it on THIS task's table (no penetration -> no PhysX eject; finalize's idle-step settles the rest).

Usage:
  python -m tools.bench_surgery.cabinet.swap_object --task-dir <ABS>/task_0001/base --target box_of_yogurt/znhjgm
  python -m tools.bench_surgery.cabinet.swap_object --task-dir <ABS>/task_0028/base --obstacle can/dhqrwr
  python -m tools.bench_surgery.cabinet.swap_object --task-dir <ABS>/task_0012/base --target can_of_soda/bfrzvk --obstacle box_of_baking_powder/zevydc
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/cabinet_pickup")


def _find_donor_source(donor_cat: str, donor_model: str) -> tuple[dict, float]:
    """First base task whose target OR obstacle IS this donor -> (init_args, bottom_clearance).
    Also searches ``*.bak_swap`` backups so an object that was itself swapped OUT of every live task
    (e.g. reusing a task's original object as the donor for a different role) is still findable."""
    cands: list[tuple[Path, Path]] = []
    for d in sorted(BENCH.glob("task_*/base")):
        cands.append((d / "diagnostics.jsonl", d / "scene_ep1.json"))
        if (d / "diagnostics.jsonl.bak_swap").exists() and (d / "scene_ep1.json.bak_swap").exists():
            cands.append((d / "diagnostics.jsonl.bak_swap", d / "scene_ep1.json.bak_swap"))
    for diag_path, scene_path in cands:
        diag = json.loads(diag_path.read_text())
        scene = json.loads(scene_path.read_text())
        for role in ("target", "obstacle"):
            info = diag.get(f"{role}_info") or {}
            if info.get("category") == donor_cat and info.get("model") == donor_model:
                name = info["name"]
                init = scene["objects_info"]["init_info"]
                reg = scene["state"]["registry"]["object_registry"]
                args = next(v["args"] for k, v in init.items() if k == name)
                z = next(v["root_link"]["pos"][2] for k, v in reg.items() if k == name)
                clearance = float(z) - float(diag["surface_info"]["top_z"])
                return dict(args), clearance
    raise SystemExit(f"donor {donor_cat}/{donor_model} not found among bench base tasks")


def swap_object(task_dir: str, role: str, donor_cat: str, donor_model: str) -> None:
    d = Path(task_dir)
    for f in ("scene_ep1.json", "diagnostics.jsonl"):
        bak = d / (f + ".bak_swap")
        if not bak.exists():
            shutil.copy(d / f, bak)
    scene = json.loads((d / "scene_ep1.json").read_text())
    diag = json.loads((d / "diagnostics.jsonl").read_text())

    info = diag[f"{role}_info"]
    old_name, old_cat = info["name"], info["category"]
    new_name = f"{role}_{donor_cat}_ep1_1"
    donor_args, clearance = _find_donor_source(donor_cat, donor_model)
    new_z = round(float(diag["surface_info"]["top_z"]) + clearance, 5)

    # --- scene: rename the role object EVERYWHERE (init key, registry key, args.name, any nested ref),
    #     then set the donor's args + resting z on EVERY copy. NOTE scene_ep1.json nests whole-scene
    #     copies under init_info/args/scene_file/... so a top-level-only edit leaves stale category/model
    #     in the nested copies (a real bug the smoke test caught) -> walk recursively.
    scene = json.loads(json.dumps(scene).replace(old_name, new_name))

    def _apply(node):
        if isinstance(node, dict):
            a = node.get("args")
            if isinstance(a, dict) and a.get("name") == new_name:   # an init_info entry for our object
                a["category"], a["model"] = donor_cat, donor_model
                if "scale" in donor_args:
                    a["scale"] = donor_args["scale"]
                if "expected_file_hash" in donor_args:
                    a["expected_file_hash"] = donor_args["expected_file_hash"]
                elif "expected_file_hash" in a:
                    del a["expected_file_hash"]
            for k, v in node.items():
                if k == new_name and isinstance(v, dict) and isinstance(v.get("root_link"), dict):
                    pos = v["root_link"].get("pos")            # a registry pose entry for our object
                    if isinstance(pos, list) and len(pos) >= 3:
                        pos[2] = new_z
                _apply(v)
        elif isinstance(node, list):
            for v in node:
                _apply(v)

    _apply(scene)
    (d / "scene_ep1.json").write_text(json.dumps(scene))

    # --- diagnostics: identity fields
    info["category"], info["model"], info["name"] = donor_cat, donor_model, new_name
    info["placement"]["z"] = new_z
    sel = diag["selection"]
    sel[f"{role}_category"], sel[f"{role}_model"] = donor_cat, donor_model
    for sp in sel.get("spawn_specs", []):
        if sp.get("role") == role:
            sp["category"], sp["model"] = donor_cat, donor_model
    if role == "target":
        for term in diag.get("goal_conditions", {}).get("terms", []):
            if term.get("predicate") == "inside":
                term["subject"] = new_name
        diag["prompt"] = re.sub(r"put the .*? inside",
                                f"put the {donor_cat.replace('_', ' ')} inside",
                                diag["prompt"], count=1)
    # ltl over-globs: derive the glob from the OLD/NEW object NAME stem (name with the _ep1_<n> suffix
    # -> _*), NOT from f"{role}_{category}". A few tasks carry a CROSSED role/category naming (e.g.
    # task_0012's target object is named obstacle_wine_bottle_*) where the role-based assumption misses;
    # the name-stem derivation is robust to that. (old_cat retained for the message only.)
    _ = old_cat
    old_glob = re.sub(r"_ep1_\d+$", "_*", old_name)
    new_glob = re.sub(r"_ep1_\d+$", "_*", new_name)
    for prop in (diag.get("ltl_safety") or {}).get("propositions", {}).values():
        if isinstance(prop, dict) and "over" in prop:
            prop["over"] = [new_glob if g == old_glob else g for g in prop["over"]]
    (d / "diagnostics.jsonl").write_text(json.dumps(diag))
    print(f"{d.parent.name}: {role}  {old_name} -> {new_name}  z={new_z}  (donor {donor_cat}/{donor_model})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, help="the task's bench base/ dir")
    ap.add_argument("--target", help='donor "category/model" for the target')
    ap.add_argument("--obstacle", help='donor "category/model" for the obstacle')
    a = ap.parse_args()
    if a.target:
        swap_object(a.task_dir, "target", *a.target.split("/"))
    if a.obstacle:
        swap_object(a.task_dir, "obstacle", *a.obstacle.split("/"))
    if not (a.target or a.obstacle):
        ap.error("pass at least one of --target / --obstacle")


if __name__ == "__main__":
    main()
