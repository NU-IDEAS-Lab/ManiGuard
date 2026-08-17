#!/usr/bin/env python
"""Generate a family's prompt-variant table for the constraint-format study.

The study trains the same policy on the same trajectories under three conditions that differ
ONLY in how the safety requirement is conveyed in the language instruction:

  no_instruction    the benchmark's released instruction, unchanged
  natural_language  + the specification's own per-clause ``description`` fields
  ltl               + the specification's own ``ltl`` formulas

Both constraint texts are taken VERBATIM from each task's ``ltl_safety`` block -- we do not
paraphrase or rewrite them -- so the task instruction and the underlying monitor automaton are
identical across conditions and only the conveyance differs. That is the whole point of the
comparison, and it is why this script generates rather than authors the table.

The finalized benchmark is READ-ONLY here: this reads ``<task>/base/diagnostics.jsonl`` and
writes a separate table under ``configs/ablation_prompt/``. Nothing under the bench is touched,
by this script or by anything downstream of it -- eval substitutes the prompt in memory at load
time, and the SFT dataset variants rewrite only a copied ``meta/tasks.jsonl``.

The table is keyed by INSTRUCTION, not by task: several tasks in a family share one instruction,
and a condition must map an instruction to the same variant wherever it appears. The script
fails if two tasks sharing an instruction disagree on their constraint set, since that would
make the mapping ambiguous.

Usage:
  python tools/ablation_prompt/build_prompt_table.py --family clutter_pickup   # verify
  python tools/ablation_prompt/build_prompt_table.py --family jar_transport --write
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "outputs" / "lerobot_datasets" / "maniguard-bench"
OUT_DIR = REPO_ROOT / "configs" / "ablation_prompt"

NL_LABEL = "Safety constraints that you should follow:"
LTL_LABEL = "Safety constraints (LTLf) that you should follow:"
NOTE = ("Task instruction and LTL automaton are identical across conditions; "
        "only how the constraint is conveyed differs.")

# family dir -> output basename (the shipped clutter table set this precedent)
STEM = {"clutter_pickup": "clutter", "jar_transport": "jar", "stack_retrieve": "stack",
        "cabinet_pickup": "cabinet", "lid_transport": "lid", "dusty_transfer": "dusty"}


def build(family: str) -> dict:
    fam_dir = BENCH / family
    if not fam_dir.is_dir():
        sys.exit(f"no such family under the bench: {fam_dir}")

    task_to_instruction: dict[str, str] = {}
    instruction_map: dict[str, dict] = {}
    seen_ids: dict[str, list[str]] = {}

    for diag in sorted(fam_dir.glob("task_*/base/diagnostics.jsonl")):
        task = diag.parent.parent.name
        row = json.loads(diag.open(encoding="utf-8").readline())
        base = row["prompt"]
        constraints = (row.get("ltl_safety") or {}).get("constraints") or []
        if not constraints:
            sys.exit(f"{task}: no ltl_safety constraints -- cannot build a constraint-bearing prompt")

        ids = [c["id"] for c in constraints]
        # An instruction shared by several tasks must carry the same constraint set, or the
        # instruction-keyed map would silently give one of them the other's specification.
        if base in seen_ids and seen_ids[base] != ids:
            sys.exit(f"{task}: instruction shared with an earlier task but the constraint sets "
                     f"differ ({seen_ids[base]} vs {ids}); the table cannot be keyed by instruction")
        seen_ids[base] = ids

        task_to_instruction[task] = base
        instruction_map.setdefault(base, {
            "no_instruction": base,
            "natural_language": f"{base} {NL_LABEL} " + " ".join(c["description"] for c in constraints),
            "ltl": f"{base} {LTL_LABEL} " + " & ".join(c["ltl"] for c in constraints),
            "constraint_ids": ids,
        })

    return {
        "_meta": {
            "family": family,
            "level": "base",
            "n_tasks": len(task_to_instruction),
            "n_unique_instructions": len(instruction_map),
            "source": "ManiGuard-Bench <task>/base/diagnostics.jsonl",
            "nl_label": NL_LABEL,
            "ltl_label": LTL_LABEL,
            "note": NOTE,
        },
        "task_to_instruction": task_to_instruction,
        "instruction_map": instruction_map,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(STEM))
    ap.add_argument("--write", action="store_true",
                    help="write the table; default is a dry run that diffs against what is on disk")
    a = ap.parse_args()

    table = build(a.family)
    out = OUT_DIR / f"{STEM[a.family]}_base_prompts.json"
    blob = json.dumps(table, indent=2, ensure_ascii=False) + "\n"
    m = table["_meta"]
    print(f"[prompt-table] {a.family}: {m['n_tasks']} tasks -> {m['n_unique_instructions']} unique instructions")
    n_con = sorted({len(v["constraint_ids"]) for v in table["instruction_map"].values()})
    print(f"[prompt-table] constraints per task: {n_con}")

    if out.exists():
        existing = out.read_text(encoding="utf-8")
        if existing == blob:
            print(f"[prompt-table] {out.name}: IDENTICAL to what is on disk")
        else:
            print(f"[prompt-table] {out.name}: DIFFERS from what is on disk"
                  f"{' -- overwriting' if a.write else ' (dry run; pass --write to overwrite)'}")
    elif not a.write:
        print(f"[prompt-table] {out.name}: does not exist yet (dry run; pass --write to create)")

    if a.write:
        out.write_text(blob, encoding="utf-8")
        print(f"[prompt-table] wrote {out}")


if __name__ == "__main__":
    main()
