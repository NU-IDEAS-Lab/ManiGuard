"""Swap a dusty_transfer task's FOOD (bench surgery, last resort — full rename cascade).

The food's instance NAME carries its category (e.g. ``cherry_124``) and is referenced by
goal_conditions (subject) and the LTL proposition ``over`` patterns (``cherry_*``), and
the category is spoken in the prompt — so unlike source/dest swaps this renames the
instance everywhere, structurally:
  - objects_info.init_info key + args (category/model/hash from the donor)
  - state.registry.object_registry key (pose kept; z bumped, gravity settle fixes rest)
  - goal_conditions subject
  - ltl_safety propositions ``over`` patterns + selection food fields + prompt phrase
Both json trees (top-level + nested init_info.args.scene_file) are processed.

Usage:
  python -m tools.bench_surgery.dusty.swap_food --task task_0023 --model potato/lgupkq \\
      --donor-task task_0011 [--apply]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/dusty_transfer")

PHRASE = {"cherry": "cherry", "potato": "potato", "half_blackberry": "blackberry",
          "garlic_clove": "garlic clove"}


def _food_name(diag: dict, tree: dict) -> str:
    sel = {x["role"]: x for x in diag["selection"]["spawn_specs"]}
    want = (sel["food"]["category"], sel["food"]["model"])
    for nm, info in tree["objects_info"]["init_info"].items():
        a = info.get("args", {})
        if (a.get("category"), a.get("model")) == want:
            return nm
    raise ValueError("food instance not found")


def _trees(scene):
    yield scene
    nested = (scene.get("init_info", {}).get("args", {}) or {}).get("scene_file")
    if isinstance(nested, dict):
        yield nested


def swap(task: str, model_key: str, donor_task: str, apply: bool) -> str:
    cat, model = model_key.split("/")
    base = BENCH / task / "base"
    scene_p, diag_p = base / "scene_ep1.json", base / "diagnostics.jsonl"
    diag = json.loads(diag_p.read_text())
    scene = json.loads(scene_p.read_text())

    ddiag = json.loads((BENCH / donor_task / "base" / "diagnostics.jsonl").read_text())
    dscene = json.loads((BENCH / donor_task / "base" / "scene_ep1.json").read_text())
    d_food = _food_name(ddiag, dscene)
    d_args = dscene["objects_info"]["init_info"][d_food]["args"]
    if d_args["model"] != model:
        return f"{task}: donor food is {d_args['model']}, not {model}"
    d_sel = {x["role"]: x for x in ddiag["selection"]["spawn_specs"]}

    old_name = _food_name(diag, scene)
    old_cat = old_name.rsplit("_", 1)[0]
    suffix = old_name.rsplit("_", 1)[1]
    new_name = f"{cat}_{suffix}"
    if new_name in scene["objects_info"]["init_info"]:
        return f"{task}: name collision {new_name}"

    n_edit = 0
    for tree in _trees(scene):
        ii = tree["objects_info"]["init_info"]
        reg = tree["state"]["registry"]["object_registry"]
        if old_name not in ii or old_name not in reg:
            continue
        entry = ii.pop(old_name)
        entry["args"] = dict(entry["args"], category=cat, model=model,
                             expected_file_hash=d_args.get("expected_file_hash"),
                             name=new_name)
        if "scale" in d_args:
            entry["args"]["scale"] = d_args["scale"]
        ii[new_name] = entry
        st = reg.pop(old_name)
        st["root_link"]["pos"][2] = float(st["root_link"]["pos"][2]) + 0.02
        for k in ("lin_vel", "ang_vel"):
            if k in st["root_link"]:
                st["root_link"][k] = [0.0, 0.0, 0.0]
        reg[new_name] = st
        n_edit += 1

    # diagnostics: goal subject, LTL over-patterns, selection, prompt
    for g in diag.get("goal_conditions") or []:
        if isinstance(g, dict) and g.get("subject") == old_name:
            g["subject"] = new_name
    props = (diag.get("ltl_safety") or {}).get("propositions") or {}
    for pdef in props.values():
        if isinstance(pdef, dict) and "over" in pdef:
            pdef["over"] = [p.replace(f"{old_cat}_", f"{cat}_") for p in pdef["over"]]
    for sp in diag["selection"]["spawn_specs"]:
        if sp.get("role") == "food":
            sp.update(category=cat, model=model, synset=d_sel["food"].get("synset"))
    diag["selection"]["food_synset"] = d_sel["food"].get("synset")
    oldp, newp = PHRASE.get(old_cat, old_cat), PHRASE.get(cat, cat)
    if "prompt" in diag and oldp != newp:
        diag["prompt"] = diag["prompt"].replace(oldp, newp)

    if not apply:
        return (f"{task}: food {old_name} -> {new_name} ({model}) trees={n_edit} DRY-RUN; "
                f"prompt: {diag['prompt'][:80]}")
    bak = scene_p.with_suffix(".json.bak_foodswap")
    if not bak.exists():
        shutil.copy2(scene_p, bak)
        shutil.copy2(diag_p, diag_p.with_suffix(".jsonl.bak_foodswap"))
    scene_p.write_text(json.dumps(scene))
    diag_p.write_text(json.dumps(diag) + "\n")
    return f"{task}: food {old_name} -> {new_name} ({model}) APPLIED trees={n_edit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--donor-task", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(swap(a.task, a.model, a.donor_task, a.apply))


if __name__ == "__main__":
    main()
