#!/usr/bin/env bash
# eval_family.sh — full-family eval over ManiGuard-Bench, level-based ID / OOD.
#
# Runs maniguard.eval.benchmark on every task instance of ONE family, bucketed by
# perturbation LEVEL (the finalized ManiGuard-Bench ID/OOD definition):
#   ID  = the base level  (datagen collected 40 demos on every base task)
#   OOD = the 4 variant levels — target / language / location / env — one bucket each
# Logs (per-instance folders + results/summary under each bucket):
#   outputs/eval_logs/<leaf>/ID/
#   outputs/eval_logs/<leaf>/OOD/{target,language,location,env}/
# One OS process per instance (OmniGibson segfaults on og.clear()); per-task PhysX
# GPU-dynamics (liquid scenes) is set automatically; NaN cascades are recorded as
# clean nan_terminated failures by benchmark.py. The engagement metric is built in.
#
# Usage:
#   bash scripts/eval_family.sh <family> [output_leaf] [config]
#     <family>       ManiGuard-Bench family dir, e.g. clutter_pickup
#     [output_leaf]  leaf under outputs/eval_logs/ (default: <family>_joint)
#     [config]       eval config       (default: configs/eval/<family>_joint.yaml)
#   LEVELS="base target language location env"   restrict which levels to run
#                                                (e.g. LEVELS=base for ID only).
#   REPEAT=N   run every instance N times (eval is stochastic; default 1).
#   FORCE=1    clobber a non-empty output dir (guards against wiping finalized logs).
#
# Requires the family's policy server already serving on 127.0.0.1:8000.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # repo root (ManiGuard)

FAMILY="${1:?usage: eval_family.sh <family> [output_leaf] [config]}"
OUT_LEAF="${2:-${FAMILY}_joint}"
CFG="${3:-configs/eval/${FAMILY}_joint.yaml}"
PY="${PYTHON_CMD:-$HOME/miniconda3/envs/behavior/bin/python}"
FAM_ROOT="outputs/lerobot_datasets/maniguard-bench/${FAMILY}"
OUT_ROOT="outputs/eval_logs/${OUT_LEAF}"
LEVELS="${LEVELS:-base target language location env}"

export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

[ -d "$FAM_ROOT" ] || { echo "ERROR: no family dir: $FAM_ROOT"; exit 1; }
[ -f "$CFG" ]      || { echo "ERROR: no config: $CFG"; exit 1; }

# --- plan: for every task, emit (task/level, bucket, gpu) for each requested level. ---
# ID = base; OOD = target/language/location/env. GPU-dynamics is a per-TASK property
# (liquid scenes), read once from the task's base diagnostics and applied to all its levels.
PLAN_FILE="$(mktemp)"
$PY - "$FAM_ROOT" "$LEVELS" > "$PLAN_FILE" <<'PYEOF'
import sys, json, pathlib
fam_root = pathlib.Path(sys.argv[1])
levels = sys.argv[2].split()

def load(p):
    # diagnostics may be single-line OR pretty-printed multi-line (dusty) -> raw_decode
    return json.JSONDecoder().raw_decode(p.read_text().lstrip())[0]

tasks = sorted(p.name for p in fam_root.iterdir()
               if p.name.startswith("task_") and (p / "base" / "diagnostics.jsonl").exists())
plan = []
for t in tasks:
    base = load(fam_root / t / "base" / "diagnostics.jsonl")
    gpu = 1 if "liquid" in (base.get("pipeline") or "") else 0   # liquid task -> GPU dynamics
    for lvl in levels:
        d = fam_root / t / lvl / "diagnostics.jsonl"
        if not d.exists():
            continue
        if not (load(d).get("prompt") or "").strip():
            continue                       # skip genuinely unusable (no prompt)
        bucket = "ID" if lvl == "base" else f"OOD/{lvl}"
        plan.append((f"{t}/{lvl}", bucket, gpu))
for key, bucket, gpu in sorted(plan, key=lambda x: (x[2], x[1])):   # gpu=0 first, then by bucket
    print(f"{key}\t{bucket}\t{gpu}")
PYEOF

# REPEAT=N: repeat every planned instance N times (eval is stochastic).
if [ "${REPEAT:-1}" -gt 1 ] 2>/dev/null; then
  awk -v n="${REPEAT}" '{for(i=0;i<n;i++)print}' "$PLAN_FILE" > "$PLAN_FILE.x" && mv "$PLAN_FILE.x" "$PLAN_FILE"
  echo "REPEAT=${REPEAT}: each instance run ${REPEAT}x"
fi

N=$(grep -c . "$PLAN_FILE" || echo 0)
[ "$N" -gt 0 ] || { echo "ERROR: no instances planned for $FAMILY (levels: $LEVELS)"; rm -f "$PLAN_FILE"; exit 1; }
echo "@@@@@ eval_family $FAMILY  ($N instances)  ->  $OUT_ROOT @@@@@"
echo "--- plan (count | bucket | gpu) ---"; awk -F'\t' '{print $2" gpu="$3}' "$PLAN_FILE" | sort | uniq -c

# Guard: never silently clobber an existing (e.g. finalized) log dir.
if [ -d "$OUT_ROOT" ] && [ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]; then
  if [ "${FORCE:-0}" = "1" ]; then echo "FORCE=1: clearing $OUT_ROOT"; rm -rf "$OUT_ROOT"
  else echo "ERROR: $OUT_ROOT exists and is non-empty — pass a fresh output_leaf or FORCE=1."; rm -f "$PLAN_FILE"; exit 1; fi
fi

i=0
while IFS=$'\t' read -r key bucket gpu; do
  i=$((i+1))
  parent="$(dirname "$bucket")"; leaf="$(basename "$bucket")"
  outdir="$OUT_ROOT"; [ "$parent" != "." ] && outdir="$OUT_ROOT/$parent"
  if [ "$gpu" = "1" ]; then export EVAL_USE_GPU_DYNAMICS=1; else unset EVAL_USE_GPU_DYNAMICS; fi
  echo "--- hold 5s before $key ($bucket, gpu=$gpu) ---"; sleep 5
  echo "############ [$i/$N] $key START $(date +%H:%M:%S) -> $bucket ############"
  $PY -m maniguard.eval.benchmark --config "$CFG" --host 127.0.0.1 --port 8000 \
      --scenes "$key" --max-scenes 1 \
      --output-dir "$outdir" --run-name "$leaf" \
      --metrics success safety --headless \
    && echo "############ [$i/$N] $key OK   $(date +%H:%M:%S) ############" \
    || echo "############ [$i/$N] $key exit=$? $(date +%H:%M:%S) ############"
done < "$PLAN_FILE"
rm -f "$PLAN_FILE"

echo "@@@@@ eval_family $FAMILY DONE $(date +%H:%M:%S) @@@@@"
$PY - "$OUT_ROOT" <<'PYEOF'
import sys, json, pathlib, collections
root = pathlib.Path(sys.argv[1])
for rj in sorted(root.rglob("results.jsonl")):
    rows = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]
    d = [r for r in rows if r["status"] == "completed"]
    oc = collections.Counter(r.get("outcome") for r in d)
    nc = sum(1 for r in d if r.get("ever_contacted")); cv = sum(1 for r in d if r.get("counted_violation"))
    succ = sum(1 for r in d if r.get("success")); viol = sum(1 for r in d if r.get("ltl_violated"))
    print(f"{str(rj.parent.relative_to(root)):28} n={len(d)} succ={succ} viol={viol} | "
          f"idle={oc['idle']} reached={oc['reached']} manip={oc['manipulated']} success={oc['success']} | "
          f"contacted={nc} cviol={cv} gated={cv/max(nc,1):.2f} vacuous_safe={len(d)-nc}")
PYEOF
