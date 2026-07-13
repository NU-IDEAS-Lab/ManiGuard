"""Swap a dusty_transfer task's DEST container model (bench surgery, 2026-07-08 batch).

The 6-task batch replaces geometrically un-wipeable dests (hand-width/depth ceiling —
see the family wipeability census) with probed-wipeable models, then RE-SCATTERS the
dust group onto the new model's inner bottom (the old particle local positions were
fitted to the old bottom). (food, source, dest) triple uniqueness was verified for the
whole batch before running.

Edits BOTH json trees (top-level + the nested ``init_info.args.scene_file`` copy):
  - dest init_info args: model + expected_file_hash (category unchanged)
  - registry root z: support_top - lo_local_z + 2mm (xy/ori kept; velocities zeroed)
  - dust system: n kept, positions re-scattered in a disk on the NEW inner bottom
  - diagnostics: selection.spawn_specs dest model

Run with the behavior env python. Usage:
  python -m tools.bench_surgery.dusty.swap_dest --task task_0011 --model bowl/pihjqa [--apply]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/dusty_transfer")

# model dossiers: file hash + geometry (spawn-probe 2026-07-08; azoiaq from task_0000 donor
# + its dedicated probe). scatter_r <= 0.065 keeps every particle inside the simple
# centred-peck footprint (no boom needed -> faster, gentler collection).
DOSSIER = {
    "bowl/qzodht": {"hash": "da091e2cbce1158d69c740ea84c09e65", "lo_z": -0.0340,
                    "inner_bottom": -0.0333, "scatter_r": 0.055},
    "bowl/pihjqa": {"hash": "ac6fb637c1dfa980846e0813539ae3a4", "lo_z": -0.0321,
                    "inner_bottom": -0.0321, "scatter_r": 0.045, "z_pad": 0.010},
    "bowl/lgaxzt": {"hash": "912ee871342f3191f60518260ac01ec9", "lo_z": -0.0319,
                    "inner_bottom": -0.0286, "scatter_r": 0.045, "z_pad": 0.010},
    "bowl/haewxp": {"hash": "7af92fd098de5366abd7a8f062a16478", "lo_z": -0.0269,
                    "inner_bottom": -0.0269, "scatter_r": 0.065},
    "mixing_bowl/xifive": {"hash": "87aaec094d1e581111be8a2042d04440", "lo_z": -0.0587,
                          "inner_bottom": -0.0545, "scatter_r": 0.055},
    "saucepan/fsinsu": {"hash": "6380475ae8dd9455da99c50aed9d6b95", "lo_z": -0.0265,
                        "inner_bottom": -0.0260, "scatter_r": 0.050},
    "stockpot/azoiaq": {"hash": "203acf68b1353b0bcf18c3788840e28c", "lo_z": -0.0576,
                        "inner_bottom": -0.0560, "scatter_r": 0.065},
}


def _dest_name(diag: dict) -> str:
    for g in diag.get("goal_conditions") or []:
        if isinstance(g, dict) and g.get("predicate") in ("inside", "ontop"):
            return g["reference"]
    raise ValueError("no inside/ontop goal condition")


def _trees(scene: dict):
    yield scene
    nested = (scene.get("init_info", {}).get("args", {}) or {}).get("scene_file")
    if isinstance(nested, dict):
        yield nested


def swap(task: str, model_key: str, apply: bool) -> str:
    cat, model = model_key.split("/")
    d = DOSSIER[model_key]
    base = BENCH / task / "base"
    scene_p = base / "scene_ep1.json"
    diag_p = base / "diagnostics.jsonl"
    diag = json.loads(diag_p.read_text())
    dest = _dest_name(diag)
    if not dest.startswith(cat):
        return f"{task}: CATEGORY MISMATCH dest={dest} vs {cat} — same-category swaps only"
    scene = json.loads(scene_p.read_text())
    support_top = float(diag["surface_info"]["top_z"])
    rng = np.random.default_rng(abs(hash((task, model_key))) % (2**32))

    n_edit = 0
    for tree in _trees(scene):
        init = tree.get("objects_info", {}).get("init_info", {}).get(dest)
        reg = (tree.get("state", {}).get("registry", {})
               .get("object_registry", {}).get(dest))
        dust = (tree.get("state", {}).get("registry", {})
                .get("system_registry", {}) or {}).get("dust")
        if not (init and reg and dust):
            continue
        old = init["args"].get("model")
        init["args"]["model"] = model
        init["args"]["expected_file_hash"] = d["hash"]
        root = reg["root_link"]
        root["pos"][2] = support_top - d["lo_z"] + 0.002
        for k in ("lin_vel", "ang_vel"):
            if k in root:
                root[k] = [0.0, 0.0, 0.0]
        # dust re-scatter on the NEW inner bottom (local/base_link frame)
        n = int(dust["n_particles"])
        r = d["scatter_r"] * np.sqrt(rng.uniform(0, 1, n))
        th = rng.uniform(0, 2 * np.pi, n)
        z = d["inner_bottom"] + d.get("z_pad", 0.004) + rng.uniform(0, 0.003, n)
        dust["positions"] = [[float(r[i] * np.cos(th[i])), float(r[i] * np.sin(th[i])),
                              float(z[i])] for i in range(n)]
        n_edit += 1

    # diagnostics: dest spawn spec model
    for s in diag["selection"]["spawn_specs"]:
        if s.get("role") == "dest":
            s["model"] = model
    if "dest_synset" in diag["selection"]:
        pass  # category unchanged -> synset unchanged

    if not apply:
        return f"{task}: {dest} {old} -> {model} (trees={n_edit}, dust n kept, DRY-RUN)"
    bak = scene_p.with_suffix(".json.bak_destswap")
    if not bak.exists():
        shutil.copy2(scene_p, bak)
        shutil.copy2(diag_p, diag_p.with_suffix(".jsonl.bak_destswap"))
    scene_p.write_text(json.dumps(scene))
    diag_p.write_text(json.dumps(diag) + "\n")
    return f"{task}: {dest} {old} -> {model} APPLIED (trees={n_edit})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True, choices=sorted(DOSSIER))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(swap(a.task, a.model, a.apply))


if __name__ == "__main__":
    main()
