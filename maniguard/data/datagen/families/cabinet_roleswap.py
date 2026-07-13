"""Reusable target<->obstacle role swap for cabinet tasks (spec §13).

A diagnostics-only edit (the scene object names + ``scene_ep1.json`` are left
untouched — every consumer resolves roles through the diagnostics, not the scene
name prefixes: ``GoalChecker`` by ``inside`` subject, ``build_active_objects_for_ltl``
by the ``over`` name-glob, datagen by ``target_info.name``). Used to make the
COMPACT, drawer-fitting object the target on the §15 NOFIT tasks.

Swaps the 6 role-bearing fields:
  1. target_info <-> obstacle_info
  2. selection target/obstacle category+model + spawn_specs roles
  3. goal_conditions 'inside' subject -> new target scene name
  4. ltl_safety.propositions target_dropped <-> obstacle_dropped 'over' globs
  5. prompt object phrases ("Place the X ... knock over the Y")

For a ``both``-mode task the swap is purely this. For a ``target``-mode task the
NEW target (old off-side obstacle) must ALSO be re-laid-out in-path afterwards
(run ``cabinet_bothfront --task ...``) — this module only does the diagnostics.

  python -u -m maniguard.data.datagen.families.cabinet_roleswap --task task_0002 [--apply]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BENCH_ROOT = Path("outputs/lerobot_datasets/maniguard-bench/cabinet_pickup")


def swap_roles(diag: dict) -> dict:
    """Return a deep-copied diagnostics dict with target<->obstacle roles swapped."""
    d = json.loads(json.dumps(diag))

    # 1. target_info <-> obstacle_info
    d["target_info"], d["obstacle_info"] = d["obstacle_info"], d["target_info"]
    new_t_name = d["target_info"]["name"]

    # 2. selection categories/models + spawn_specs roles
    sel = d["selection"]
    sel["target_category"], sel["obstacle_category"] = sel["obstacle_category"], sel["target_category"]
    sel["target_model"], sel["obstacle_model"] = sel["obstacle_model"], sel["target_model"]
    for spec in sel.get("spawn_specs", []):
        if spec.get("role") == "target":
            spec["role"] = "obstacle"
        elif spec.get("role") == "obstacle":
            spec["role"] = "target"

    # 3. goal_conditions 'inside' subject -> new target scene name
    for term in d.get("goal_conditions", {}).get("terms", []):
        if term.get("predicate") == "inside":
            term["subject"] = new_t_name

    # 4. ltl propositions target_dropped <-> obstacle_dropped 'over' globs
    props = d.get("ltl_safety", {}).get("propositions", {})
    if "target_dropped" in props and "obstacle_dropped" in props:
        props["target_dropped"]["over"], props["obstacle_dropped"]["over"] = \
            props["obstacle_dropped"]["over"], props["target_dropped"]["over"]

    # 5. prompt object phrases (NEW target = sel.target_category after the swap)
    tcat = sel["target_category"].replace("_", " ")    # new target words
    ocat = sel["obstacle_category"].replace("_", " ")   # new obstacle words
    p = d["prompt"]
    # old prompt: "Place the <old target=ocat> ... knock over the <old obstacle=tcat>"
    repl_t = f"Place the {ocat}"
    repl_o = f"knock over the {tcat}"
    if repl_t not in p or repl_o not in p:
        raise ValueError(
            f"prompt phrase mismatch — expected {repl_t!r} and {repl_o!r} in prompt:\n  {p}")
    p = p.replace(repl_t, "Place the @@TGT@@").replace(repl_o, "knock over the @@OBS@@")
    d["prompt"] = p.replace("@@TGT@@", tcat).replace("@@OBS@@", ocat)
    return d


def _apply(task: str, do_apply: bool) -> None:
    path = BENCH_ROOT / task / "base" / "diagnostics.jsonl"
    diag = json.loads(path.read_text().splitlines()[0])
    new = swap_roles(diag)
    print(f"[roleswap] {task}: mode={new.get('blocker_mode')}")
    print(f"  target : {new['target_info']['category']} ({new['target_info']['name']})")
    print(f"  obstacle: {new['obstacle_info']['category']} ({new['obstacle_info']['name']})")
    print(f"  goal inside subject: "
          f"{[t for t in new['goal_conditions']['terms'] if t.get('predicate')=='inside'][0]['subject']}")
    props = new["ltl_safety"]["propositions"]
    print(f"  ltl target_dropped.over={props['target_dropped']['over']} "
          f"obstacle_dropped.over={props['obstacle_dropped']['over']}")
    print(f"  prompt: {new['prompt']}")
    if do_apply:
        bak = path.with_suffix(".jsonl.bak_roleswap")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(json.dumps(new) + "\n")
        print(f"  APPLIED (backup: {bak.name})")
    else:
        print("  DRY-RUN — no write (pass --apply)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="task_NNNN")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    _apply(args.task, args.apply)


if __name__ == "__main__":
    main()
