"""Re-home a stack_retrieve task's OBJECTS onto a DONOR task's scaffold (surface + canonical robot +
cameras), to escape a too-small / oblique surface while KEEPING the task's own objects (object-type
richness preserved). Inverse of ``tools.bench_surgery.stack.swap_object`` (which keeps the table, swaps the objects).

The donor (default ``task_0005``: ``breakfast_table_uhrsex_0``, 1.06 m², robot orthogonal @0.61 m) gives a
big axis-aligned surface. ``finalize_base_task`` STRIPS the source robot and bakes a CANONICAL robot
relative to the surface, so an axis-aligned donor table => an orthogonal robot automatically — which is
exactly the fix for the too-small (0001/0023/0025) and oblique-square (0018/0020) / tiny (0019) surfaces.

Per task T (all read BEFORE overwrite):
  * from T's diag: selection / prompt / ltl_safety / goal_conditions / goal_region (target_name + width +
    radius) + T's object names (target = goal_region.target_name, stack = the other 3 task objects);
  * from T's scene: each object's init args (scale + expected_file_hash);
  * thickness (metadata bbox_size[2]) for the target + stack models.
Then: copy the donor scene+diag, REMAP the donor's (1 target + 3 stack) objects to T's target/stack
models (T's ORIGINAL names, re-stacked at T's thicknesses on the donor's stack xy), rename the donor goal
marker to T's target + T's radius, and patch the diag identity to T (surface / scene_model stay donor's;
surface_info/cameras/gate/LTL are recomputed by the subsequent finalize).

Backs up T's originals to ``*.bak_rehome``. Re-finalise afterwards with ``tools.bench_surgery.stack.rerender_base``.

Usage:
  python -m tools.bench_surgery.stack.rehome_task --task-dir <ABS>/task_0018/base
  python -m tools.bench_surgery.stack.rehome_task --task-dir <ABS>/task_0001/base --donor <ABS>/task_0005/base
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
from pathlib import Path

DATA = os.environ.get("OMNIGIBSON_DATA_PATH", "")
DONOR_DEFAULT = "outputs/lerobot_datasets/maniguard-bench/stack_retrieve/task_0005/base"
GAP = 0.003          # small load-time gap between stacked objects (gravity closes it on settle)


def _thickness(cat: str, model: str) -> float:
    """Upright z-extent (stacking thickness) = bbox_size[2] from the dataset object metadata."""
    fs = glob.glob(f"{DATA}/**/{cat}/{model}/misc/metadata.json", recursive=True)
    if not fs:
        raise SystemExit(f"no metadata.json for {cat}/{model} under OMNIGIBSON_DATA_PATH={DATA!r}")
    return float(json.loads(Path(fs[0]).read_text())["bbox_size"][2])


def _task_object_keys(reg: dict, diag: dict) -> list[str]:
    """Registry keys of the 4 task objects (exclude surface / goal marker / robot)."""
    surface = diag.get("surface")
    return [k for k in reg
            if not (k == surface or k.startswith("goal_region") or k.startswith("agent")
                    or "robot" in k.lower())]


def _obj_args(scene: dict, cat: str, model: str) -> dict:
    """scale + expected_file_hash for cat/model from a scene's init_info (first match)."""
    for _k, v in scene.get("objects_info", {}).get("init_info", {}).items():
        a = v.get("args", {})
        if a.get("category") == cat and a.get("model") == model:
            return {"scale": a.get("scale", [1.0, 1.0, 1.0]),
                    "expected_file_hash": a.get("expected_file_hash")}
    raise SystemExit(f"{cat}/{model} not found in scene init_info")


def _spec(diag: dict, role: str) -> dict:
    for sp in diag["selection"].get("spawn_specs", []):
        if sp.get("role") == role:
            return sp
    raise SystemExit(f"no spawn_spec with role={role}")


def rehome(task_dir: str, donor_dir: str) -> None:
    d = Path(task_dir)
    donor = Path(donor_dir)

    # --- read T's ORIGINALS (backup first so a re-run reads the pristine source) ---
    for f in ("scene_ep1.json", "diagnostics.jsonl"):
        bak = d / (f + ".bak_rehome")
        if not bak.exists():
            shutil.copy(d / f, bak)
    t_scene = json.loads((d / "scene_ep1.json.bak_rehome").read_text())
    t_diag = json.loads((d / "diagnostics.jsonl.bak_rehome").read_text())
    t_reg = t_scene["state"]["registry"]["object_registry"]
    t_goal = t_diag["goal_region"]
    t_target_name = t_goal["target_name"]
    t_task_keys = _task_object_keys(t_reg, t_diag)
    if t_target_name not in t_task_keys:
        raise SystemExit(f"target {t_target_name} not among task objects {t_task_keys}")
    t_stack_names = [k for k in t_task_keys if k != t_target_name]
    if len(t_stack_names) != 3:
        raise SystemExit(f"expected 3 stack objects, got {t_stack_names}")

    tgt_spec, stk_spec = _spec(t_diag, "target"), _spec(t_diag, "stack")
    tgt_cat, tgt_model = tgt_spec["category"], tgt_spec["model"]
    stk_cat, stk_model = stk_spec["category"], stk_spec["model"]
    tgt_a = _obj_args(t_scene, tgt_cat, tgt_model)
    stk_a = _obj_args(t_scene, stk_cat, stk_model)
    thick_t, thick_s = _thickness(tgt_cat, tgt_model), _thickness(stk_cat, stk_model)

    # --- donor scaffold ---
    donor_scene = json.loads((donor / "scene_ep1.json").read_text())
    donor_diag = json.loads((donor / "diagnostics.jsonl").read_text())
    d_reg = donor_scene["state"]["registry"]["object_registry"]
    d_ii = donor_scene["objects_info"]["init_info"]
    d_goal = donor_diag["goal_region"]
    d_target_key = d_goal["target_name"]
    d_task_keys = _task_object_keys(d_reg, donor_diag)
    d_marker_key = next(k for k in d_reg if k.startswith("goal_region"))
    top_z = float(donor_diag["surface_info"]["top_z"])
    xy = list(d_reg[d_target_key]["root_link"]["pos"][:2])

    # templates (structure) from the donor's own target object + goal marker
    reg_tmpl = copy.deepcopy(d_reg[d_target_key])
    ii_tmpl = copy.deepcopy(d_ii[d_target_key])
    marker_reg_tmpl = copy.deepcopy(d_reg[d_marker_key])
    marker_ii_tmpl = copy.deepcopy(d_ii[d_marker_key])

    # --- build the new scene: donor scaffold, task objects + goal marker replaced by T's ---
    new_scene = copy.deepcopy(donor_scene)
    nreg = new_scene["state"]["registry"]["object_registry"]
    nii = new_scene["objects_info"]["init_info"]
    for k in d_task_keys + [d_marker_key]:
        nreg.pop(k, None)
        nii.pop(k, None)

    def _place(name, cat, model, args, z, is_target):
        r = copy.deepcopy(reg_tmpl)
        r["root_link"]["pos"] = [round(xy[0], 6), round(xy[1], 6), round(z, 6)]
        r["root_link"]["ori"] = [0.0, 0.0, 0.0, 1.0]
        for v in ("lin_vel", "ang_vel"):
            if v in r["root_link"]:
                r["root_link"][v] = [0.0, 0.0, 0.0]
        nreg[name] = r
        it = copy.deepcopy(ii_tmpl)
        a = it["args"]
        a["name"], a["category"], a["model"] = name, cat, model
        a["scale"] = args["scale"]
        if args.get("expected_file_hash"):
            a["expected_file_hash"] = args["expected_file_hash"]
        elif "expected_file_hash" in a:
            del a["expected_file_hash"]
        nii[name] = it

    # bottom target, then 3 stack objects rising by their thickness
    _place(t_target_name, tgt_cat, tgt_model, tgt_a, top_z + 0.5 * thick_t, True)
    for i, sname in enumerate(t_stack_names):
        z = top_z + thick_t + GAP + i * (thick_s + GAP) + 0.5 * thick_s
        _place(sname, stk_cat, stk_model, stk_a, z, False)

    # goal marker: donor position (left of the stack, on uhrsex) + T's radius, named for T's target
    marker_name = f"goal_region__{t_target_name}"
    mr = copy.deepcopy(marker_reg_tmpl)
    if "radius" in mr:
        mr["radius"] = t_goal["radius_m"]
    nreg[marker_name] = mr
    mi = copy.deepcopy(marker_ii_tmpl)
    ma = mi["args"]
    ma["name"] = marker_name
    ma["relative_prim_path"] = f"/{marker_name}"
    if "radius" in ma:
        ma["radius"] = t_goal["radius_m"]
    nii[marker_name] = mi

    (d / "scene_ep1.json").write_text(json.dumps(new_scene))

    # --- build the new diag: donor scaffold (surface/scene) + T identity (selection/prompt/ltl/goal) ---
    new_diag = copy.deepcopy(donor_diag)
    new_diag["surface"] = donor_diag["surface"]
    new_diag["scene_model"] = donor_diag.get("scene_model")
    new_diag["selection"] = t_diag["selection"]
    new_diag["prompt"] = t_diag["prompt"]
    new_diag["ltl_safety"] = t_diag.get("ltl_safety")
    new_diag["goal_conditions"] = t_diag.get("goal_conditions")
    ng = copy.deepcopy(d_goal)                              # donor centre / support / mode / colour
    ng["target_name"] = t_target_name
    ng["marker_name"] = marker_name
    ng["radius_m"] = t_goal["radius_m"]
    ng["target_width_m"] = t_goal.get("target_width_m")
    new_diag["goal_region"] = ng
    # surface_info/cameras/gate/ltl are recomputed by finalize; drop the donor's so it's obviously fresh
    for k in ("surface_info", "cameras", "gate_pass", "ltl_violated", "steps_executed", "ltl_summary"):
        new_diag.pop(k, None)
    (d / "diagnostics.jsonl").write_text(json.dumps(new_diag))

    print(f"{d.parent.name}: rehomed onto {donor.parent.name} ({donor_diag['surface']})  "
          f"target={t_target_name}({tgt_cat}/{tgt_model} t={thick_t:.3f})  "
          f"stack={stk_cat}/{stk_model}(t={thick_s:.3f}) x3  xy={[round(v, 3) for v in xy]} top_z={top_z:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, help="the task's bench base/ dir (gets re-homed in place)")
    ap.add_argument("--donor", default=DONOR_DEFAULT, help="donor task's base/ dir (surface scaffold)")
    a = ap.parse_args()
    rehome(a.task_dir, a.donor)


if __name__ == "__main__":
    main()
