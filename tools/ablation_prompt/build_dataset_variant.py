#!/usr/bin/env python
"""Build a prompt-ablation variant of a datagen LeRobot dataset — WITHOUT copying data.

The Q2 ablation (clutter, base only) trains the same policy on the same trajectories three
times, varying only how the safety constraint is conveyed in the language instruction:
``no_instruction`` (today's data, unchanged), ``natural_language``, ``ltl``.

Every model resolves its prompt from ONE small file in ``meta/``, so a variant only needs
that file rewritten:

  * LeRobot **v2.1** — ``meta/tasks.jsonl``, one JSON object per line
      openpi (pi0.5 / pi0): ``PromptFromLeRobotTask(dataset_meta.tasks)``
      GR00T: ``annotation.human.action.task_description`` <- ``task_index`` -> tasks.jsonl
  * LeRobot **v3.0** — ``meta/tasks.parquet``, a DataFrame INDEXED BY THE TASK STRING
      SmolVLA: ``dataset_reader.py`` does ``item["task"] = meta.tasks.iloc[task_idx].name``,
      i.e. the prompt IS the index value at positional row ``task_index``. Rewriting the
      variant therefore replaces the index strings and must PRESERVE ROW ORDER -- ``iloc``
      is positional, so reordering would hand every episode a different task's prompt.

This script produces a *lightweight* dataset: ``meta/`` is a real copy with the prompts
substituted, while ``data/`` and ``videos/`` are RELATIVE SYMLINKS to the source dataset
(both directory names are unchanged between v2.1 and v3.0). Cost is a few MB instead of
tens of GB, and the source dataset is never written to.

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

SUFFIX = {"natural_language": "-promptnl", "ltl": "-promptltl", "no_instruction": "-promptnone"}
LINKED = ("data", "videos")  # symlinked; never copied

# The SFT boxes do not carry the ManiGuard repo -- this script and the prompt tables are
# copied there as loose files. So resolve the default map relative to the SCRIPT, checking the
# in-repo layout first and then a sibling file, and fall back to requiring --prompt-map rather
# than silently pointing at a path that does not exist.
_HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MAP = next(
    (p for p in (_HERE.parents[1] / "configs" / "ablation_prompt" / "clutter_base_prompts.json",
                 _HERE / "clutter_base_prompts.json") if p.is_file()),
    None,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source LeRobot dataset dir (read-only)")
    ap.add_argument("--condition", required=True, choices=sorted(SUFFIX))
    ap.add_argument("--out", help="output dir (default: <src><suffix>, i.e. a sibling)")
    ap.add_argument("--prompt-map", default=(str(DEFAULT_MAP) if DEFAULT_MAP else None),
                    required=DEFAULT_MAP is None,
                    help="prompt-variant table; MUST match the dataset's family "
                         "(configs/ablation_prompt/{clutter,jar,stack}_base_prompts.json). "
                         "A mismatch is caught -- every instruction would miss the map.")
    ap.add_argument("--force", action="store_true", help="rebuild if the output already exists")
    args = ap.parse_args()

    src = pathlib.Path(args.src).resolve()
    out = pathlib.Path(args.out).resolve() if args.out else src.with_name(src.name + SUFFIX[args.condition])
    jsonl, parquet = src / "meta" / "tasks.jsonl", src / "meta" / "tasks.parquet"
    if jsonl.is_file():
        fmt = "v2.1"
    elif parquet.is_file():
        fmt = "v3.0"
    else:
        sys.exit(f"not a LeRobot dataset: {src} (no meta/tasks.jsonl nor meta/tasks.parquet)")
    if out.exists():
        if not args.force:
            sys.exit(f"{out} already exists (pass --force to rebuild)")
        shutil.rmtree(out)

    pm = json.loads(pathlib.Path(args.prompt_map).read_text())["instruction_map"]

    def variant_of(instr: str, missing: list) -> str:
        """Look the instruction up in the map, recording a miss rather than guessing."""
        instr = instr.strip()
        if instr not in pm:
            missing.append(instr)
            return instr
        return pm[instr][args.condition]

    # 1) meta/ : real copy, then substitute the prompts in the task table
    shutil.copytree(src / "meta", out / "meta")
    missing: list[str] = []
    if fmt == "v2.1":
        rows = []
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["task"] = variant_of(r["task"], missing)
            rows.append(r)
        n_sub = len(rows)
        example = rows[0]["task"] if rows else ""
        if not missing:
            (out / "meta" / "tasks.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    else:
        import pandas as pd

        df = pd.read_parquet(parquet)
        # The prompt is the INDEX; task_index is a column. Rewrite the index in place so row
        # ORDER is untouched -- SmolVLA reads `tasks.iloc[task_index].name`, so a reordered
        # table would silently give every episode a different task's prompt.
        new_index = [variant_of(str(i), missing) for i in df.index]
        if len(set(new_index)) != len(new_index):
            shutil.rmtree(out)
            sys.exit("the substitution collapsed two distinct instructions onto one variant; "
                     "task_index would become ambiguous")
        df.index = pd.Index(new_index, name=df.index.name or "task")
        n_sub = len(df)
        example = str(df.index[0]) if len(df) else ""
        if not missing:
            df.to_parquet(out / "meta" / "tasks.parquet")

    if missing:
        shutil.rmtree(out)
        sys.exit(f"{len(missing)} instruction(s) absent from the prompt map, e.g.:\n  {missing[0]!r}")

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
    print(f"[build] {args.condition}: {out}  [{fmt}]")
    print(f"[build]   meta copied ({n_meta / 1e6:.1f} MB), {n_sub} prompts substituted")
    print(f"[build]   {', '.join(n + '/ -> symlink' for n in LINKED if (out / n).exists())}")
    # GR00T resolves its language annotation through meta/modality.json; openpi does not
    # need it. It is copied along with the rest of meta/ when the source has one.
    has_mod = (out / "meta" / "modality.json").is_file()
    print(f"[build]   meta/modality.json: {'present -> usable by GR00T too' if has_mod else 'ABSENT -> openpi only'}")
    print(f"[build]   example: {example[:120]}...")


if __name__ == "__main__":
    main()
