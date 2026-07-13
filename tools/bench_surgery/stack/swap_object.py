"""Swap a stack_retrieve (same-mode) task's stacked objects for a donor object.

Same idea as ``tools.bench_surgery.cabinet.swap_object`` but for the stack family: a same-mode task has 4 task
objects — the bottom ``target`` + 3 ``stack`` instances — ALL of one category/model. This rewrites the
task's ``base/scene_ep1.json`` (init_info args + registry poses, every object RENAMED to the donor and
RE-STACKED at the donor's thickness) + ``base/diagnostics.jsonl`` (selection / spawn_specs / goal_region
names / ltl over-globs / prompt). Then re-finalize with ``tools.bench_surgery.stack.rerender_base --tasks task_NNNN``
(settles physics + re-renders the 4 review videos + recomputes cameras/gate/LTL/surface).

The donor's geometry (scale + expected_file_hash) is read from ANY base scene where it already appears
(bench first, then 6fam-base). Its stacking thickness is the upright bbox z-extent from the dataset
object metadata. Idempotent per task; backs both files up to ``*.bak_swap`` on first touch.

Usage:
  python -m tools.bench_surgery.stack.swap_object --task-dir <ABS>/task_0022/base --object toy_dice/ievnsq
  python -m tools.bench_surgery.stack.swap_object --task-dir <ABS>/task_0026/base --object folder/lktggf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/stack_retrieve")
SIXFAM = Path("outputs/lerobot_datasets/6fam-base/stack_retrieve")
DATA = os.environ.get("OMNIGIBSON_DATA_PATH", "")
GAP = 0.003          # small load-time gap between stacked objects (gravity closes it on settle)


def _donor_args(cat: str, model: str) -> dict:
    """First base scene (bench, then 6fam) that spawns cat/model -> its init_info args (scale + hash)."""
    for root in (BENCH, SIXFAM):
        for scene_path in sorted(glob.glob(str(root / "task_*" / "*" / "scene_ep1.json"))
                                 + glob.glob(str(root / "task_*" / "*" / "scene_ep1_replay.json"))):
            scene = json.loads(Path(scene_path).read_text())
            for _k, v in scene.get("objects_info", {}).get("init_info", {}).items():
                a = v.get("args", {})
                if a.get("category") == cat and a.get("model") == model:
                    return {"scale": a.get("scale", [1.0, 1.0, 1.0]),
                            "expected_file_hash": a.get("expected_file_hash")}
    raise SystemExit(f"donor {cat}/{model} not found in any bench/6fam base scene")


def _donor_thickness(cat: str, model: str) -> float:
    """Upright z-extent (stacking thickness) = bbox_size[2] from the dataset object metadata."""
    fs = glob.glob(f"{DATA}/**/{cat}/{model}/misc/metadata.json", recursive=True)
    if not fs:
        raise SystemExit(f"no metadata.json for {cat}/{model} under OMNIGIBSON_DATA_PATH={DATA}")
    return float(json.loads(Path(fs[0]).read_text())["bbox_size"][2])


def _synset(diag_selection: dict) -> str:
    return diag_selection.get("target_synset") or diag_selection.get("stack_synset") or ""


def swap(task_dir: str, new_cat: str, new_model: str) -> None:
    d = Path(task_dir)
    for f in ("scene_ep1.json", "diagnostics.jsonl"):
        bak = d / (f + ".bak_swap")
        if not bak.exists():
            shutil.copy(d / f, bak)
    scene = json.loads((d / "scene_ep1.json").read_text())
    diag = json.loads((d / "diagnostics.jsonl").read_text())

    sel = diag["selection"]
    old_cat = sel["target_category"]
    old_model = sel["target_model"]
    old_synset = _synset(sel)
    new_synset = f"{new_cat}.n.01"
    donor = _donor_args(new_cat, new_model)
    thick = _donor_thickness(new_cat, new_model)
    top_z = float(diag["surface_info"]["top_z"])

    # bottom target + 3 stack instances share ONE xy (a clean vertical stack); index by the ep1_<n>
    # suffix: target=_ep1_1 (k=0, bottom), stack=_ep1_2.. (k=1..). centre_z(k)=top + half + k*(thick+GAP).
    reg = scene["state"]["registry"]["object_registry"]
    task_objs = {k: v for k, v in reg.items()
                 if (k.startswith("target_") or k.startswith("stack_")) and old_cat in k}
    xy = None
    for k in sorted(task_objs):
        if k.startswith("target_"):
            xy = list(reg[k]["root_link"]["pos"][:2])
    if xy is None:
        xy = list(next(iter(task_objs.values()))["root_link"]["pos"][:2])

    # --- SCENE: global rename (object names + category + model), then per-entry fix the donor's
    #     scale/expected_file_hash and RE-STACK the task objects at the donor thickness.
    scene = json.loads(json.dumps(scene).replace(old_cat, new_cat).replace(old_model, new_model))

    def _apply(node):
        if isinstance(node, dict):
            a = node.get("args")
            if isinstance(a, dict) and a.get("category") == new_cat:
                a["scale"] = donor["scale"]
                if donor.get("expected_file_hash"):
                    a["expected_file_hash"] = donor["expected_file_hash"]
                elif "expected_file_hash" in a:
                    del a["expected_file_hash"]
            for k, v in node.items():
                m = re.search(r"_ep1_(\d+)$", k)
                if (k.startswith("target_") or k.startswith("stack_")) and new_cat in k and m \
                        and isinstance(v, dict) and isinstance(v.get("root_link"), dict):
                    kk = int(m.group(1)) - 1
                    pos = v["root_link"]["pos"]
                    pos[0], pos[1] = xy[0], xy[1]
                    pos[2] = round(top_z + 0.5 * thick + kk * (thick + GAP), 5)
                    v["root_link"]["ori"] = [0.0, 0.0, 0.0, 1.0]
                _apply(v)
        elif isinstance(node, list):
            for v in node:
                _apply(v)

    _apply(scene)
    (d / "scene_ep1.json").write_text(json.dumps(scene))

    # --- DIAGNOSTICS: identity fields (targeted; keep everything else).
    for key in ("target_synset", "stack_synset"):
        if key in sel:
            sel[key] = new_synset
    sel["target_category"] = sel["stack_category"] = new_cat
    sel["target_model"] = sel["stack_model"] = new_model
    for sp in sel.get("spawn_specs", []):
        sp["synset"], sp["category"], sp["model"] = new_synset, new_cat, new_model
    gr = diag.get("goal_region", {})
    for key in ("target_name", "marker_name"):
        if gr.get(key):
            gr[key] = gr[key].replace(old_cat, new_cat)
    for prop in (diag.get("ltl_safety") or {}).get("propositions", {}).values():
        if isinstance(prop, dict) and "over" in prop:
            prop["over"] = [g.replace(old_cat, new_cat) for g in prop["over"]]
    diag["prompt"] = diag.get("prompt", "").replace(old_cat.replace("_", " "), new_cat.replace("_", " "))
    (d / "diagnostics.jsonl").write_text(json.dumps(diag))

    n = len(task_objs)
    print(f"{d.parent.name}: {old_cat}/{old_model} -> {new_cat}/{new_model}  ({n} objs restacked "
          f"@thick={thick:.3f} on top_z={top_z:.3f}, xy={[round(x,3) for x in xy]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, help="the task's bench base/ dir")
    ap.add_argument("--object", required=True, help='donor "category/model" for all 4 stacked objects')
    a = ap.parse_args()
    swap(a.task_dir, *a.object.split("/"))


if __name__ == "__main__":
    main()
