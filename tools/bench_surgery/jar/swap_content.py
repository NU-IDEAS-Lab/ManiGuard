"""Replace a jar task's content object with a selected donor asset.

Search the configured donor roots for asset scale and hash, rename the content
object, preserve its position, reset its orientation and velocities, and
update item metadata and prompt text. Create one-time .bak_swap backups.
The command does not check family-wide pair uniqueness or simulate fit.
Re-finalize afterward to refresh runtime checks and review videos."""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "outputs/lerobot_datasets/maniguard-bench/jar_transport"


def _donor_args(cat: str, model: str, donor_roots=None) -> dict:
    roots = [Path(p).expanduser() for p in donor_roots] if donor_roots else [BENCH]
    for root in roots:
        for scene_path in sorted(glob.glob(str(root / "task_*" / "*" / "scene_ep1.json"))):
            scene = json.loads(Path(scene_path).read_text())
            for _k, v in scene.get("objects_info", {}).get("init_info", {}).items():
                a = v.get("args", {})
                if a.get("category") == cat and a.get("model") == model:
                    return {"scale": a.get("scale", [1.0, 1.0, 1.0]),
                            "expected_file_hash": a.get("expected_file_hash")}
    raise SystemExit(f"donor {cat}/{model} not found under {roots}")


def swap(task_dir: str, new_cat: str, new_model: str, *, donor_roots=None) -> None:
    donor = _donor_args(new_cat, new_model, donor_roots)
    d = Path(task_dir)
    for f in ("scene_ep1.json", "diagnostics.jsonl"):
        bak = d / f"{f}.bak_swap"
        if not bak.exists():
            shutil.copy2(d / f, bak)

    scene = json.loads((d / "scene_ep1.json").read_text())
    diag = json.JSONDecoder().raw_decode((d / "diagnostics.jsonl").read_text().lstrip())[0]
    sel = diag["selection"]
    old_cat, old_model = sel["item_category"], sel["item_model"]
    old_name = diag["item_info"]["name"]
    new_name = f"food_{new_cat}_ep1_1"
    print(f"[swap] {old_cat}/{old_model} ({old_name}) -> {new_cat}/{new_model} ({new_name})")

    reg = scene["state"]["registry"]["object_registry"]
    jar_name = diag["jar_info"]["name"]
    jar_pos = reg[jar_name]["root_link"]["pos"]

    # Rename the item registry entry while preserving its existing position.
    # Reset orientation, velocities, and articulated-state fields for the rigid donor.
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
    else:
        a.pop("expected_file_hash", None)
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
    ap.add_argument("--donor-root", action="append", type=Path,
                    help="Family directory containing donor task scenes; repeat to search multiple roots. "
                         "Default: this family's ManiGuard-Bench directory.")
    a = ap.parse_args()
    cat, model = a.item.split("/")
    swap(a.task_dir, cat, model, donor_roots=a.donor_root)
