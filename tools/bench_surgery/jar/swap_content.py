"""Swap a jar_transport task's CONTENT item for a donor object (the jar itself is untouched).

Mirrors ``tools.bench_surgery.stack.swap_object`` for the jar family: rewrites the task's ``base/scene_ep1.json``
(item init_info args + registry entry renamed to the donor, spawn pose ABOVE the jar mouth so the
bench re-finalize settle drops it INTO the cavity) + ``base/diagnostics.jsonl`` (selection item_* /
spawn_specs / item_info / prompt). Then re-finalize with the family-generic bench finalize
(``finalize_base_task``) to settle physics, re-render the review videos and re-stamp runtime stats.

Constraint honoured by the caller: the (jar_model, item_category) pair must stay UNIQUE across the
family. Donor geometry (scale + expected_file_hash) is read from any bench/6fam base scene where the
donor already appears. Backs both files up to ``*.bak_swap`` on first touch; idempotent per task.

Usage:
  python -m tools.bench_surgery.jar.swap_content --task-dir <ABS>/task_0013/base --item jar_of_cumin/tsktnz
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/jar_transport")
SIXFAM = Path("outputs/lerobot_datasets/6fam-base/jar_transport")


def _donor_args(cat: str, model: str) -> dict:
    for root in (BENCH, SIXFAM):
        for scene_path in sorted(glob.glob(str(root / "task_*" / "*" / "scene_ep1.json"))):
            scene = json.loads(Path(scene_path).read_text())
            for _k, v in scene.get("objects_info", {}).get("init_info", {}).items():
                a = v.get("args", {})
                if a.get("category") == cat and a.get("model") == model:
                    return {"scale": a.get("scale", [1.0, 1.0, 1.0]),
                            "expected_file_hash": a.get("expected_file_hash")}
    raise SystemExit(f"donor {cat}/{model} not found in any bench/6fam jar base scene")


def swap(task_dir: str, new_cat: str, new_model: str) -> None:
    d = Path(task_dir)
    for f in ("scene_ep1.json", "diagnostics.jsonl"):
        bak = d / f"{f}.bak_swap"
        if not bak.exists():
            shutil.copy2(d / f, bak)

    scene = json.loads((d / "scene_ep1.json").read_text())
    diag = json.loads((d / "diagnostics.jsonl").read_text().splitlines()[0])
    sel = diag["selection"]
    old_cat, old_model = sel["item_category"], sel["item_model"]
    old_name = diag["item_info"]["name"]
    new_name = f"food_{new_cat}_ep1_1"
    print(f"[swap] {old_cat}/{old_model} ({old_name}) -> {new_cat}/{new_model} ({new_name})")

    donor = _donor_args(new_cat, new_model)
    reg = scene["state"]["registry"]["object_registry"]
    jar_name = diag["jar_info"]["name"]
    jar_pos = reg[jar_name]["root_link"]["pos"]

    # registry: rename the item entry; KEEP the old item's settled position (it sat at/in the mouth,
    # a bench-legal start) — the smaller donor gently settles from there INTO the cavity. Spawning
    # high above the mouth free-falls onto the jar and knocks it over (upright violation at settle).
    entry = reg.pop(old_name)
    entry["root_link"]["ori"] = [0.0, 0.0, 0.0, 1.0]
    entry["root_link"]["lin_vel"] = [0.0, 0.0, 0.0]
    entry["root_link"]["ang_vel"] = [0.0, 0.0, 0.0]
    if "joint_pos" in entry:      # donor is a rigid (non-articulated) item
        entry.pop("joint_pos", None); entry.pop("joint_vel", None)
    reg[new_name] = entry

    # init_info: retarget the item object
    ii = scene["objects_info"]["init_info"]
    obj = ii.pop(old_name)
    a = obj["args"]
    a["name"] = new_name
    a["category"], a["model"] = new_cat, new_model
    a["scale"] = donor["scale"]
    if donor["expected_file_hash"]:
        a["expected_file_hash"] = donor["expected_file_hash"]
    ii[new_name] = obj
    (d / "scene_ep1.json").write_text(json.dumps(scene))

    # diagnostics: selection / spawn_specs / item_info / prompt
    new_synset = f"{new_cat}.n.01"
    sel["item_category"], sel["item_model"], sel["item_synset"] = new_cat, new_model, new_synset
    for sp in sel.get("spawn_specs", []):
        if sp.get("category") == old_cat:
            sp["category"], sp["model"], sp["synset"] = new_cat, new_model, new_synset
    diag["item_info"] = {"name": new_name, "category": new_cat, "model": new_model}
    diag["prompt"] = diag.get("prompt", "").replace(old_cat.replace("_", " "), new_cat.replace("_", " "))
    (d / "diagnostics.jsonl").write_text(json.dumps(diag) + "\n")
    print(f"[swap] prompt: {diag['prompt']}")
    print("[swap] done — now re-finalize (settle + re-render) via the bench finalize")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--item", required=True, help="category/model, e.g. jar_of_cumin/tsktnz")
    a = ap.parse_args()
    cat, model = a.item.split("/")
    swap(a.task_dir, cat, model)
