"""Swap a jar_transport task's TARGET JAR MODEL for a donor jar (the content item is untouched —
use ``tools.bench_surgery.jar.swap_content`` for that; combine both for a double swap).

Mirrors ``tools.bench_surgery.jar.swap_content`` on the jar side: rewrites ``base/scene_ep1.json`` (jar init_info
args retargeted to the donor model; registry keeps the task's XY but takes an EXPLICIT ``--root-z``
because root-to-bottom offsets differ per jar model — jnjtrl's root rides 8cm above the support
while gqtsam's is flush, so reusing the old z buries or floats the new jar; the donor's root ori +
lid joint rest angle come along so the lid starts at the model's natural open pose) +
``base/diagnostics.jsonl`` (selection jar_* / spawn_specs / jar_info). Re-finalize afterwards with
``tools/bench_surgery/jar/rerender_base.py``, then PROBE the lid-hang direction and yaw-fix via
``tools.bench_surgery.jar.move_task`` if it faces a bad side (hinge frames differ per model).

Constraint honoured by the caller: (jar_model, item_category) stays UNIQUE family-wide.

Usage:
  python -m tools.bench_surgery.jar.swap_model --task-dir <ABS>/task_0014/base --model jnjtrl --root-z 0.782
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

BENCH = Path("outputs/lerobot_datasets/maniguard-bench/jar_transport")
JAR = "target_hinged_jar_ep1_1"


def _donor(model: str) -> dict:
    """Find a bench task using the donor jar model; return its init args + registry rest state."""
    for scene_path in sorted(glob.glob(str(BENCH / "task_*" / "base" / "scene_ep1.json"))):
        scene = json.loads(Path(scene_path).read_text())
        a = scene.get("objects_info", {}).get("init_info", {}).get(JAR, {}).get("args", {})
        if a.get("model") != model:
            continue
        reg = scene["state"]["registry"]["object_registry"][JAR]
        diag_p = Path(scene_path).parent / "diagnostics.jsonl"
        jar_info = json.loads(diag_p.read_text().splitlines()[0])["jar_info"]
        return {"scale": a.get("scale", [1.0, 1.0, 1.0]),
                "expected_file_hash": a.get("expected_file_hash"),
                "joint_pos": reg.get("joint_pos", [2.62]),
                "ori": reg["root_link"]["ori"],
                "min_dim_m": jar_info.get("min_dim_m"),
                "src": scene_path.split("/")[-3]}
    raise SystemExit(f"donor jar model {model} not found in any bench jar base scene")


def swap(task_dir: str, new_model: str, root_z: float) -> None:
    d = Path(task_dir)
    for f in ("scene_ep1.json", "diagnostics.jsonl"):
        bak = d / f"{f}.bak_jarswap"
        if not bak.exists():
            shutil.copy2(d / f, bak)

    scene = json.loads((d / "scene_ep1.json").read_text())
    diag = json.loads((d / "diagnostics.jsonl").read_text().splitlines()[0])
    old_model = diag["jar_info"]["model"]
    donor = _donor(new_model)
    print(f"[jarswap] {old_model} -> {new_model} (donor task: {donor['src']})")

    reg = scene["state"]["registry"]["object_registry"][JAR]
    old_pos = reg["root_link"]["pos"]
    reg["root_link"]["pos"] = [float(old_pos[0]), float(old_pos[1]), float(root_z)]
    reg["root_link"]["ori"] = list(donor["ori"])
    reg["root_link"]["lin_vel"] = [0.0, 0.0, 0.0]
    reg["root_link"]["ang_vel"] = [0.0, 0.0, 0.0]
    reg["joint_pos"] = list(donor["joint_pos"])
    reg["joint_vel"] = [0.0]
    print(f"[jarswap] root z {old_pos[2]:.3f} -> {root_z:.3f}; lid rest joint {donor['joint_pos']}")

    a = scene["objects_info"]["init_info"][JAR]["args"]
    a["model"] = new_model
    a["scale"] = donor["scale"]
    if donor["expected_file_hash"]:
        a["expected_file_hash"] = donor["expected_file_hash"]
    (d / "scene_ep1.json").write_text(json.dumps(scene))

    sel = diag["selection"]
    sel["jar_model"] = new_model
    for sp in sel.get("spawn_specs", []):
        if sp.get("role") == "target" or sp.get("category") == "hinged_jar":
            sp["model"] = new_model
    diag["jar_info"]["model"] = new_model
    if donor["min_dim_m"] is not None:
        diag["jar_info"]["min_dim_m"] = donor["min_dim_m"]
    (d / "diagnostics.jsonl").write_text(json.dumps(diag) + "\n")
    print("[jarswap] done — re-finalize, then PROBE lid-hang direction (yaw-fix if needed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--root-z", type=float, required=True,
                    help="new jar root z = target support top + donor's root_above_support (probe both)")
    a = ap.parse_args()
    swap(a.task_dir, a.model, a.root_z)
