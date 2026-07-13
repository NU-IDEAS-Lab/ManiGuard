"""Same-category (container, lid) PAIR swap for lid_transport bench tasks.

For a task whose container+lid geometry is unworkable (gqwnfv canister tips at any
contact), swap BOTH models to a donor task's proven pair. Same-category only —
instance names, prompt wording, and LTL patterns all stay valid; only init args
(model/hash/scale) change. Positions keep xy; z is re-seated via the donor's
root-above-support offset; the rerender's gravity settle finishes the rest.

Usage:
  python -m tools.bench_surgery.lid.swap_pair --task task_0008 --donor-task task_0001 --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/lid_transport")


def _names(diag: dict, tree: dict) -> dict:
    sel = {x["role"]: x for x in diag["selection"]["spawn_specs"]}
    out = {}
    for role in ("lid", "container"):
        spec = sel.get(role) or (sel.get("target") if role == "container" else None)
        for nm, info in tree["objects_info"]["init_info"].items():
            a = info.get("args", {})
            if (a.get("category"), a.get("model")) == (spec["category"], spec["model"]):
                out[role] = nm
                break
    return out


def _trees(scene):
    yield scene
    nested = (scene.get("init_info", {}).get("args", {}) or {}).get("scene_file")
    if isinstance(nested, dict):
        yield nested


def swap(task: str, donor: str, apply: bool) -> str:
    base = BENCH / task / "base"
    dbase = BENCH / donor / "base"
    diag = json.loads((base / "diagnostics.jsonl").read_text())
    scene = json.loads((base / "scene_ep1.json").read_text())
    ddiag = json.loads((dbase / "diagnostics.jsonl").read_text())
    dscene = json.loads((dbase / "scene_ep1.json").read_text())

    d_names = _names(ddiag, dscene)
    d_sel = {x["role"]: x for x in ddiag["selection"]["spawn_specs"]}
    my_sel = {x["role"]: x for x in diag["selection"]["spawn_specs"]}
    cont_role = "container" if "container" in my_sel else "target"
    for role in ("lid", "container"):
        if (d_sel[role]["category"]
                != (my_sel.get(role) or my_sel[cont_role])["category"]):
            return f"{task}: donor category mismatch on {role} — same-category swaps only"

    d_sup = float(ddiag["surface_info"]["top_z"])
    my_sup = float(diag["surface_info"]["top_z"])
    n_edit = 0
    for tree in _trees(scene):
        names = _names(diag, tree)
        reg = tree["state"]["registry"]["object_registry"]
        ii = tree["objects_info"]["init_info"]
        for role in ("lid", "container"):
            d_args = dscene["objects_info"]["init_info"][d_names[role]]["args"]
            d_rootz = float(dscene["state"]["registry"]["object_registry"]
                            [d_names[role]]["root_link"]["pos"][2])
            args = ii[names[role]]["args"]
            args["model"] = d_args["model"]
            args["expected_file_hash"] = d_args.get("expected_file_hash")
            if "scale" in d_args:
                args["scale"] = d_args["scale"]
            root = reg[names[role]]["root_link"]
            root["pos"][2] = my_sup + (d_rootz - d_sup) + 0.003
            for k in ("lin_vel", "ang_vel"):
                if k in root:
                    root[k] = [0.0, 0.0, 0.0]
        n_edit += 1

    for sp in diag["selection"]["spawn_specs"]:
        role = "container" if sp.get("role") in ("container", "target") else sp.get("role")
        if role in ("lid", "container"):
            sp["model"] = d_sel[role]["model"]
    if isinstance(diag.get("lid_info"), dict):
        diag["lid_info"]["container"]["model"] = d_sel["container"]["model"]
        diag["lid_info"]["lid"]["model"] = d_sel["lid"]["model"]

    if not apply:
        return f"{task}: pair -> {d_sel['container']['model']}+{d_sel['lid']['model']} DRY-RUN"
    bak = (base / "scene_ep1.json").with_suffix(".json.bak_pairswap")
    if not bak.exists():
        shutil.copy2(base / "scene_ep1.json", bak)
        shutil.copy2(base / "diagnostics.jsonl",
                     (base / "diagnostics.jsonl").with_suffix(".jsonl.bak_pairswap"))
    (base / "scene_ep1.json").write_text(json.dumps(scene))
    (base / "diagnostics.jsonl").write_text(json.dumps(diag) + "\n")
    return (f"{task}: pair -> {d_sel['container']['model']}+{d_sel['lid']['model']} "
            f"APPLIED trees={n_edit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--donor-task", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(swap(a.task, a.donor_task, a.apply))


if __name__ == "__main__":
    main()
