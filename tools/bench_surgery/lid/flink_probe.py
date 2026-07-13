"""One-boot probe: per (container, lid) pair of the lid_transport bench, record the
attachment geometry needed for (a) auditing container grasp annotations against the
lid-on envelope and (b) the family skeleton's release-pose math.

Spawns each unique pair in an empty scene on a grid (visual placement, no physics
needed beyond load), then records per pair:
  - container root -> F meta-link offset, in the CONTAINER's local frame
  - lid root -> M meta-link offset, in the LID's local frame
  - lid AABB extents (world-aligned at identity orientation)
  - container AABB extents + category/model, lid category/model

Output: outputs/grasp_annotation/lid_flink_db.json

Usage:
  OMNIGIBSON_HEADLESS=1 python -m tools.bench_surgery.lid.flink_probe
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BENCH = _REPO / "outputs/lerobot_datasets/maniguard-bench/lid_transport"
OUT = _REPO / "outputs/grasp_annotation/lid_flink_db.json"


def unique_pairs() -> list[dict]:
    pairs, seen = [], set()
    for d in sorted(BENCH.glob("task_*/base/diagnostics.jsonl")):
        r = json.loads(d.read_text().splitlines()[0])
        sel = {x["role"]: x for x in r["selection"]["spawn_specs"]}
        cont = sel.get("container") or sel.get("target")
        lid = sel["lid"]
        key = (cont["category"], cont["model"], lid["category"], lid["model"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"container": {"category": cont["category"], "model": cont["model"]},
                      "lid": {"category": lid["category"], "model": lid["model"]},
                      "task": d.parts[-3]})
    return pairs


def main() -> int:
    from maniguard.data.datagen.primitives import scene as scenemod
    from maniguard.utils.lid_attach import find_F_link, find_M_link

    pairs = unique_pairs()
    print(f"[flink] {len(pairs)} unique (container, lid) pairs", flush=True)

    og = scenemod.init_omnigibson(headless=True)
    import omnigibson as og_mod
    from omnigibson.objects.dataset_object import DatasetObject

    env = og_mod.Environment(configs={"scene": {"type": "Scene"}, "robots": []})
    objs = []
    for i, p in enumerate(pairs):
        x, y = 3.0 * (i % 6), 3.0 * (i // 6)
        c = DatasetObject(name=f"cont_{i}", category=p["container"]["category"],
                          model=p["container"]["model"])
        l = DatasetObject(name=f"lid_{i}", category=p["lid"]["category"],
                          model=p["lid"]["model"])
        env.scene.add_object(c)
        env.scene.add_object(l)
        c.set_position_orientation(position=[x, y, 1.0])
        l.set_position_orientation(position=[x + 1.2, y, 1.0])
        objs.append((p, c, l))
    for _ in range(3):
        og_mod.sim.step()

    db = {}
    for p, c, l in objs:
        m = find_M_link(l)
        f = find_F_link(c, m.meta_link_id) if m is not None else None
        entry = {"task": p["task"], "lid": p["lid"], "m_link": None, "f_link": None}
        c_pos, c_quat = (np.asarray(v, float) for v in c.get_position_orientation())
        l_pos, l_quat = (np.asarray(v, float) for v in l.get_position_orientation())
        from scipy.spatial.transform import Rotation as Rot
        if f is not None:
            f_pos, f_quat = (np.asarray(v, float) for v in f.get_position_orientation())
            rc = Rot.from_quat(c_quat).inv()
            entry["f_link"] = {"id": f.meta_link_id,
                               "local_offset": rc.apply(f_pos - c_pos).tolist(),
                               "local_quat": (rc * Rot.from_quat(f_quat)).as_quat().tolist()}
        if m is not None:
            m_pos, m_quat = (np.asarray(v, float) for v in m.get_position_orientation())
            rl = Rot.from_quat(l_quat).inv()
            entry["m_link"] = {"id": m.meta_link_id,
                               "local_offset": rl.apply(m_pos - l_pos).tolist(),
                               "local_quat": (rl * Rot.from_quat(m_quat)).as_quat().tolist()}
        entry["lid_aabb_extent"] = (np.asarray(l.aabb_extent, float)).tolist()
        entry["container_aabb_extent"] = (np.asarray(c.aabb_extent, float)).tolist()
        key = f"{p['container']['category']}/{p['container']['model']}"
        db[key] = entry
        ok = "OK" if (entry["f_link"] and entry["m_link"]) else "MISSING-META-LINK"
        print(f"[flink] {key:40s} lid={p['lid']['model']} {ok}", flush=True)

    OUT.write_text(json.dumps(db, indent=1))
    n_ok = sum(1 for v in db.values() if v["f_link"] and v["m_link"])
    print(f"[flink] DONE {n_ok}/{len(db)} pairs with full meta-links -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
