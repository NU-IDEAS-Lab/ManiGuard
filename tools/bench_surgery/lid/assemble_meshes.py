"""Build container-and-lid composite meshes for grasp annotation.

Read local attachment frames from lid_flink_db.json, transform the lid mesh
into the container root frame with T_F @ inverse(T_M), and concatenate the
meshes. Update mesh_db.json to use the composite; mesh_bare retains the
previous container mesh path. Lid entries keep their existing meshes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

ANN = _REPO / "outputs/grasp_annotation"
FLINK = ANN / "lid_flink_db.json"
MESH_DB = ANN / "mesh_db.json"


def _pose_mat(pos, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos, float)
    return T


def main() -> int:
    flink = json.loads(FLINK.read_text())
    mesh_db = json.loads(MESH_DB.read_text())
    objs = mesh_db["objects"]

    n_ok = 0
    for cont_key, entry in sorted(flink.items()):
        if not (entry.get("f_link") and entry.get("m_link")):
            print(f"[assemble] {cont_key}: missing meta-link data — SKIP", flush=True)
            continue
        lid_key = f"{entry['lid']['category']}/{entry['lid']['model']}"
        cont_e, lid_e = objs.get(cont_key), objs.get(lid_key)
        if not (cont_e and lid_e):
            print(f"[assemble] {cont_key}: mesh_db entries missing — SKIP", flush=True)
            continue
        cont_mesh_path = cont_e.get("mesh_bare") or cont_e["mesh"]
        cont_mesh = trimesh.load(ANN / cont_mesh_path, force="mesh")
        lid_mesh = trimesh.load(ANN / lid_e["mesh"], force="mesh")

        # attached lid root in container-local frame: M-frame ≡ F-frame at snap
        T_f = _pose_mat(entry["f_link"]["local_offset"], entry["f_link"]["local_quat"])
        T_m = _pose_mat(entry["m_link"]["local_offset"], entry["m_link"]["local_quat"])
        T_lid = T_f @ np.linalg.inv(T_m)
        lid_mesh.apply_transform(T_lid)

        combo = trimesh.util.concatenate([cont_mesh, lid_mesh])
        out_rel = f"meshes/{cont_key.replace('/', '__')}__lidon.glb"
        combo.export(ANN / out_rel)

        cont_e.setdefault("mesh_bare", cont_e["mesh"])
        cont_e["mesh"] = out_rel
        n_ok += 1
        lid_z = float(T_lid[2, 3])
        print(f"[assemble] {cont_key:38s} + {lid_key:12s} -> {out_rel} "
              f"(lid root z={lid_z:+.3f} in cont frame)", flush=True)

    MESH_DB.write_text(json.dumps(mesh_db, indent=1))
    print(f"[assemble] DONE {n_ok}/{len(flink)} composites; mesh_db updated "
          f"(mesh_bare keeps originals)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
