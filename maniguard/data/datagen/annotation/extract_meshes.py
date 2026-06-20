"""Phase A — batch-extract grasp-target meshes + metadata for the annotation tool.

OmniGibson assets are encrypted USD, so meshes can only be read via OG. This enumerates
the distinct ``(category, model)`` grasp TARGETS across the bench families (from each
task's ``diagnostics`` goal target + ``scene_ep1.json`` object model — pure JSON), then
in ONE OG session spawns every distinct target (``visual_only``, in a grid), extracts its
object-local visual mesh, exports a GLB, and records bbox + the upright world orientation
(parsed from the scene state, = how it stands in the task scene). Output feeds the viser
annotation tool (Phase B). Distractors are NOT extracted (never grasped).

  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
  OMNIGIBSON_HEADLESS=1 PYTHONPATH=$HOME/project/ManiGuard \
  python -u -m maniguard.data.datagen.annotation.extract_meshes \
      [--families clutter_pickup] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ALL_FAMILIES = ["clutter_pickup", "jar_transport", "lid_transport",
                "dusty_transfer", "stack_retrieve", "cabinet_pickup"]
BENCH = Path("outputs/lerobot_datasets/maniguard-bench")
OUT = Path("outputs/grasp_annotation")


def _target_name(diag: dict) -> str | None:
    """Grasp target object name from a task's first diagnostics row. Handles the two
    goal encodings: goal_region.target_name (clutter/jar/lid/dusty/stack); and
    goal_conditions — list form (grasping -> reference) or dict/terms form
    (cabinet: inside/ontop -> subject = the placed object)."""
    gr = diag.get("goal_region") or {}
    if gr.get("target_name"):
        return gr["target_name"]
    gc = diag.get("goal_conditions")
    if isinstance(gc, list):
        for t in gc:
            if isinstance(t, dict) and t.get("predicate") == "grasping":
                return t.get("reference")
        return gc[0].get("reference") if gc and isinstance(gc[0], dict) else None
    if isinstance(gc, dict):
        for t in gc.get("terms", []):
            if t.get("predicate") in ("inside", "ontop", "on_top", "under"):
                return t.get("subject")
        return gc.get("reference") or gc.get("subject")
    return None


def _state_ori(scene: dict, name: str):
    """Object's recorded world quaternion (xyzw) = its upright pose in the scene."""
    try:
        return scene["state"]["registry"]["object_registry"][name]["root_link"]["ori"]
    except Exception:  # noqa: BLE001
        return [0.0, 0.0, 0.0, 1.0]


def enumerate_targets(families) -> dict:
    """{``cat/model``: {category, model, upright_orientation_xyzw, source_task}} via pure
    JSON (no sim). First occurrence per distinct (cat, model) wins."""
    db: dict = {}
    for fam in families:
        for tdir in sorted(glob.glob(str(BENCH / fam / "task_*/base"))):
            tdir = Path(tdir)
            try:
                with open(tdir / "diagnostics.jsonl") as f:
                    diag = json.loads(f.readline())
                scene = json.load(open(tdir / "scene_ep1.json"))
            except Exception:  # noqa: BLE001
                continue
            tname = _target_name(diag)
            if not tname:
                continue
            args = (scene.get("objects_info", {}).get("init_info", {})
                    .get(tname, {}).get("args", {}))
            cat, model = args.get("category"), args.get("model")
            if not (cat and model):
                continue
            key = f"{cat}/{model}"
            if key in db:
                continue
            db[key] = {"category": cat, "model": model,
                       "upright_orientation_xyzw": _state_ori(scene, tname),
                       "source_task": f"{fam}/{tdir.parent.name}"}
    return db


def extract_gripper(og) -> None:
    """Dump the longfinger gripper mesh (panda_hand + 2 fingers) in the EEF_LINK frame so
    the viser tool can render the real gripper at a candidate eef-target pose."""
    import numpy as np
    import trimesh
    import omnigibson as og_mod
    from maniguard._omnigibson_patches import _patch_franka_longfinger
    from maniguard.data.datagen.primitives.grasp_obb import _to_np, _pose_to_mat

    _patch_franka_longfinger()
    env_cfg = {"env": {"action_frequency": 30, "rendering_frequency": 30},
               "scene": {"type": "Scene"},
               "robots": [{"type": "FrankaPanda", "name": "robot", "position": [0, 0, 0]}]}
    env = og_mod.Environment(configs=env_cfg)
    og.sim.step()
    robot = env.robots[0]
    arm = robot.default_arm
    ep, eq = robot.eef_links[arm].get_position_orientation()
    T_w_eef = np.linalg.inv(_pose_to_mat(_to_np(ep), _to_np(eq)))     # world -> eef_link

    parts = []
    for ln in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        link = robot.links.get(ln)
        if link is None:
            continue
        for geom in link.visual_meshes.values():
            if geom.geom_type != "Mesh" or geom.faces is None or len(geom.points) == 0:
                continue
            pts_w = _to_np(geom.transform_local_points_to_world(geom.points)).reshape(-1, 3)
            pts_eef = (T_w_eef[:3, :3] @ pts_w.T).T + T_w_eef[:3, 3]
            parts.append(trimesh.Trimesh(vertices=pts_eef,
                                         faces=_to_np(geom.faces).reshape(-1, 3),
                                         process=False))
    OUT.mkdir(parents=True, exist_ok=True)
    g = trimesh.util.concatenate(parts)
    g.export(OUT / "gripper_longfinger.glb")
    print(f"[extract] gripper mesh ({len(g.vertices)} verts, eef_link frame) -> "
          f"{OUT}/gripper_longfinger.glb", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["clutter_pickup"],
                    choices=ALL_FAMILIES + ["all"])
    ap.add_argument("--limit", type=int, default=0, help="cap # targets (0 = all)")
    ap.add_argument("--grid", type=int, default=10, help="objects per grid row")
    ap.add_argument("--gripper", action="store_true",
                    help="extract ONLY the longfinger gripper mesh (spawns a robot)")
    args = ap.parse_args()

    if args.gripper:
        from maniguard.data.datagen.primitives import scene as scenemod
        og = scenemod.init_omnigibson(headless=True)
        extract_gripper(og)
        try:
            og.sim.stop()
        except Exception:  # noqa: BLE001
            pass
        return 0
    families = ALL_FAMILIES if "all" in args.families else args.families

    targets = enumerate_targets(families)
    keys = list(targets)
    if args.limit:
        keys = keys[: args.limit]
    print(f"[extract] {len(keys)} distinct targets from {families}", flush=True)

    from maniguard.data.datagen.primitives import scene as scenemod
    from maniguard.data.datagen.primitives.grasp_obb import mesh_from_og_object

    og = scenemod.init_omnigibson(headless=True)
    import omnigibson as og_mod

    # One empty Scene env with every target spawned visual-only in a grid (no physics,
    # no robot — we only read meshes).
    obj_cfgs = []
    for i, key in enumerate(keys):
        t = targets[key]
        x, y = (i % args.grid) * 0.6, (i // args.grid) * 0.6
        obj_cfgs.append({"type": "DatasetObject", "name": f"obj_{i}",
                         "category": t["category"], "model": t["model"],
                         "position": [x, y, 0.5], "fixed_base": True,
                         "visual_only": True})
    env_cfg = {"env": {"action_frequency": 30, "rendering_frequency": 30},
               "scene": {"type": "Scene"}, "objects": obj_cfgs, "robots": []}
    print("[extract] building env with all targets ...", flush=True)
    env = og_mod.Environment(configs=env_cfg)
    og.sim.step()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    db = {"schema_version": "1.0", "gripper": "franka_panda_longfinger",
          "convention": {"pose_means": "eef_link target pose",
                         "frame": "object_local (obj.get_position_orientation)",
                         "note": "runtime: T_eef_world = object_world_pose @ pose"},
          "objects": {}}
    ok = bad = 0
    for i, key in enumerate(keys):
        obj = env.scene.object_registry("name", f"obj_{i}")
        try:
            mesh = mesh_from_og_object(obj, use_visual=True)   # object-local trimesh
            fname = key.replace("/", "__") + ".glb"
            mesh.export(OUT / "meshes" / fname)
            db["objects"][key] = {
                **targets[key],
                "bbox_size": [float(v) for v in mesh.extents],
                "mesh": f"meshes/{fname}",
                "grasps": [],
            }
            ok += 1
            if ok % 10 == 0:
                print(f"[extract] {ok}/{len(keys)} ...", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"[extract] FAIL {key}: {type(e).__name__}: {e}", flush=True)

    json.dump(db, open(OUT / "mesh_db.json", "w"), indent=2)
    print(f"[extract] DONE — {ok} meshes -> {OUT}/meshes, db -> {OUT}/mesh_db.json "
          f"({bad} failed)", flush=True)
    try:
        og.sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
