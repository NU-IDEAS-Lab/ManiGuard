"""Swap a dusty_transfer task's SOURCE carrier model (bench surgery, stage 2).

User-authorized: any of (source, dest, food) may swap as long as the family-wide
(food, source, dest) triple stays unique. The FOOD rides the source, so the food's
xy offset from the source centre is KEPT (new sources are chosen bigger) and its z is
raised onto the new source top + buffer — the rerender's gravity settle drops it to
true rest.

Donor-based dossier: init args (model/hash) come from a task that already uses the
new model; geometry from the family probe table.

Usage:
  python -m tools.bench_surgery.dusty.swap_source --task task_0017 --model plate/pjinwe \\
      --donor-task task_0013 [--apply]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/dusty_transfer")


def _src_food_names(diag: dict, tree: dict) -> tuple[str, str]:
    sel = {x["role"]: x for x in diag["selection"]["spawn_specs"]}
    names = {}
    for role in ("source", "food"):
        want = (sel[role]["category"], sel[role]["model"])
        for nm, info in tree["objects_info"]["init_info"].items():
            a = info.get("args", {})
            if (a.get("category"), a.get("model")) == want:
                names[role] = nm
                break
    return names["source"], names["food"]


def _trees(scene: dict):
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
    support_top = float(diag["surface_info"]["top_z"])

    # donor: init args + root-to-bottom for the new model
    dscene = json.loads((BENCH / donor_task / "base" / "scene_ep1.json").read_text())
    ddiag = json.loads((BENCH / donor_task / "base" / "diagnostics.jsonl").read_text())
    d_src, _ = _src_food_names(ddiag, dscene)
    d_args = dscene["objects_info"]["init_info"][d_src]["args"]
    if d_args["model"] != model:
        return f"{task}: donor {donor_task} source is {d_args['model']}, not {model}"
    d_root_z = float(dscene["state"]["registry"]["object_registry"][d_src]["root_link"]["pos"][2])
    d_support = float(ddiag["surface_info"]["top_z"])
    root_above_support = d_root_z - d_support

    src_name, food_name = _src_food_names(diag, scene)
    cross = not src_name.startswith(cat)
    # cross-category: allowed — the source is NOT referenced by LTL/goal_conditions
    # (only the food/dest are); the instance NAME keeps its old prefix (cosmetic).
    # We update args.category, the spawn spec category+synset (copied from the donor),
    # and the PROMPT phrase (the source category is spoken in the prompt).
    d_sel = {x["role"]: x for x in ddiag["selection"]["spawn_specs"]}

    n_edit = 0
    old = None
    for tree in _trees(scene):
        init = tree["objects_info"]["init_info"].get(src_name)
        reg = tree["state"]["registry"]["object_registry"].get(src_name)
        freg = tree["state"]["registry"]["object_registry"].get(food_name)
        if not (init and reg and freg):
            continue
        old = init["args"].get("model")
        old_cat = init["args"].get("category")
        init["args"]["model"] = model
        init["args"]["category"] = cat
        init["args"]["expected_file_hash"] = d_args.get("expected_file_hash")
        root = reg["root_link"]
        # keep xy; sit the new model at the donor's proven support offset
        root["pos"][2] = support_top + root_above_support + 0.002
        for k in ("lin_vel", "ang_vel"):
            if k in root:
                root[k] = [0.0, 0.0, 0.0]
            if k in freg["root_link"]:
                freg["root_link"][k] = [0.0, 0.0, 0.0]
        # food: keep its xy offset (new source is bigger); raise onto the new top +1cm
        # buffer — the rerender's gravity settle drops it to true rest
        food_z_old = float(freg["root_link"]["pos"][2])
        freg["root_link"]["pos"][2] = max(food_z_old + 0.01, support_top + 0.05 + 0.01)
        n_edit += 1

    for sp in diag["selection"]["spawn_specs"]:
        if sp.get("role") == "source":
            sp["model"] = model
            sp["category"] = cat
            sp["synset"] = d_sel["source"].get("synset", sp.get("synset"))
    if cross:
        diag["selection"]["source_synset"] = d_sel["source"].get("synset",
                                                                 diag["selection"].get("source_synset"))
        PHRASE = {"chopping_board": "chopping board", "cutting_board": "cutting board",
                  "tray": "tray", "platter": "platter", "plate": "plate", "saucer": "saucer"}
        oldp, newp = PHRASE.get(old_cat, old_cat), PHRASE.get(cat, cat)
        if oldp and oldp != newp and "prompt" in diag:
            diag["prompt"] = diag["prompt"].replace(oldp, newp)

    if not apply:
        return f"{task}: {src_name} {old} -> {model} (trees={n_edit}, DRY-RUN)"
    bak = scene_p.with_suffix(".json.bak_srcswap")
    if not bak.exists():
        shutil.copy2(scene_p, bak)
        shutil.copy2(diag_p, diag_p.with_suffix(".jsonl.bak_srcswap"))
    scene_p.write_text(json.dumps(scene))
    diag_p.write_text(json.dumps(diag) + "\n")
    return f"{task}: {src_name} {old} -> {model} APPLIED (trees={n_edit})"


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
