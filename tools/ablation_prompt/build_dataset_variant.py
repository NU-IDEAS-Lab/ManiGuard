#!/usr/bin/env python
"""Build a prompt-ablation variant of a datagen LeRobot dataset — WITHOUT copying data.

The Q2 ablation (clutter, base only) trains the same policy on the same trajectories three
times, varying only how the safety constraint is conveyed in the language instruction:
``no_instruction`` (today's data, unchanged), ``natural_language``, ``ltl``.

All three models read their prompt from ONE place — ``meta/tasks.jsonl``:
  * openpi (pi0.5): ``PromptFromLeRobotTask(dataset_meta.tasks)``
  * GR00T:          ``annotation.human.action.task_description`` <- ``task_index`` -> tasks.jsonl
so a variant only needs a rewritten ``tasks.jsonl``. This script therefore produces a
*lightweight* dataset: ``meta/`` is a real copy with the prompts substituted, while
``data/`` and ``videos/`` are RELATIVE SYMLINKS to the source dataset. Cost is a few MB
instead of tens of GB, and the source dataset is never written to.

Prompts come from ``configs/ablation_prompt/clutter_base_prompts.json``, generated from the
finalized bench (each task's own ``ltl_safety`` block), so the task instruction and the
underlying automaton are identical across conditions — only the conveyance differs.

Usage (on the SFT box, from the ManiGuard repo):
  python tools/ablation_prompt/build_dataset_variant.py \
      --src  <HF_LEROBOT_HOME>/IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam \
      --condition natural_language \
      [--out <...>/datagen-clutter-v1-joint-5cam-promptnl]   # default: <src>-prompt{nl,ltl}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO_ROOT / "configs" / "ablation_prompt" / "clutter_base_prompts.json"
SUFFIX = {"natural_language": "-promptnl", "ltl": "-promptltl", "no_instruction": "-promptnone"}
LINKED = ("data", "videos")  # symlinked; never copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source LeRobot dataset dir (read-only)")
    ap.add_argument("--condition", required=True, choices=sorted(SUFFIX))
    ap.add_argument("--out", help="output dir (default: <src><suffix>, i.e. a sibling)")
    ap.add_argument("--prompt-map", default=str(DEFAULT_MAP))
    ap.add_argument("--force", action="store_true", help="rebuild if the output already exists")
    args = ap.parse_args()

    src = pathlib.Path(args.src).resolve()
    out = pathlib.Path(args.out).resolve() if args.out else src.with_name(src.name + SUFFIX[args.condition])
    if not (src / "meta" / "tasks.jsonl").is_file():
        sys.exit(f"not a LeRobot dataset: {src} (no meta/tasks.jsonl)")
    if out.exists():
        if not args.force:
            sys.exit(f"{out} already exists (pass --force to rebuild)")
        shutil.rmtree(out)

    pm = json.loads(pathlib.Path(args.prompt_map).read_text())["instruction_map"]

    # 1) meta/ : real copy, then substitute the prompts in tasks.jsonl
    shutil.copytree(src / "meta", out / "meta")
    rows, missing = [], []
    for line in (src / "meta" / "tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        instr = r["task"].strip()
        if instr not in pm:
            missing.append(instr)
            continue
        r["task"] = pm[instr][args.condition]
        rows.append(r)
    if missing:
        shutil.rmtree(out)
        sys.exit(f"{len(missing)} instruction(s) absent from the prompt map, e.g.:\n  {missing[0]!r}")
    (out / "meta" / "tasks.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    # keep info.json's repo_id consistent with the variant's directory name
    info_p = out / "meta" / "info.json"
    if info_p.is_file():
        info = json.loads(info_p.read_text())
        if "repo_id" in info:
            info["repo_id"] = f"{out.parent.name}/{out.name}"
            info_p.write_text(json.dumps(info, indent=4) + "\n")

    # 2) data/ + videos/ : relative symlinks -> zero data duplication, source untouched
    for name in LINKED:
        if (src / name).is_dir():
            os.symlink(os.path.relpath(src / name, out), out / name)

    n_meta = sum(f.stat().st_size for f in (out / "meta").rglob("*") if f.is_file())
    print(f"[build] {args.condition}: {out}")
    print(f"[build]   meta copied ({n_meta / 1e6:.1f} MB), {len(rows)} prompts substituted")
    print(f"[build]   {', '.join(n + '/ -> symlink' for n in LINKED if (out / n).exists())}")
    # GR00T resolves its language annotation through meta/modality.json; openpi does not
    # need it. It is copied along with the rest of meta/ when the source has one.
    has_mod = (out / "meta" / "modality.json").is_file()
    print(f"[build]   meta/modality.json: {'present -> usable by GR00T too' if has_mod else 'ABSENT -> openpi only'}")
    print(f"[build]   example: {rows[0]['task'][:120]}...")


if __name__ == "__main__":
    main()
