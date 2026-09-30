#!/usr/bin/env bash
# eval_family.sh — full-family eval over ManiGuard-Bench, ID / OOD conditions.
#
# Runs maniguard.eval.benchmark on every task instance of ONE family, bucketed by
# perturbation condition:
#   ID  = the base condition  (datagen collected 40 demos on every base task)
#   OOD = the 4 OOD conditions — target / language / location / env — one bucket each
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
#   LEVELS="base target language location env"   restrict which conditions to run
#                                                (e.g. LEVELS=base for ID only).
#   REPEAT=N   run every instance N times (eval is stochastic; default 1).
#   SEED=N     pass a base policy seed to every instance (default: eval config).
#              REPEAT does not advance this seed; use separate runs for distinct seeds.
#   FORCE=1    clobber a non-empty output dir (guards against wiping finalized logs).
#   BENCH_ROOT=/path/to/maniguard-bench   local benchmark checkout
#                                         (default: outputs/lerobot_datasets/maniguard-bench).
#   PYTHON_CMD=/path/to/python            interpreter of the behavior env
#                                         (default: ~/miniconda3/envs/behavior/bin/python).
#
# Requires the family's policy server already serving on 127.0.0.1:8000.
# Ends with a per-bucket summary in the paper's metrics (tools/eval_summary.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # repo root (ManiGuard)

FAMILY="${1:?usage: eval_family.sh <family> [output_leaf] [config]}"
OUT_LEAF="${2:-${FAMILY}_joint}"
DEFAULT_CONFIG="configs/eval/${FAMILY}_joint.yaml"
[[ "$FAMILY" == "lid_transport" ]] && DEFAULT_CONFIG="configs/eval/lid_transport_food_joint.yaml"
CFG="${3:-$DEFAULT_CONFIG}"
PY="${PYTHON_CMD:-$HOME/miniconda3/envs/behavior/bin/python}"
FAM_ROOT="${BENCH_ROOT:-outputs/lerobot_datasets/maniguard-bench}/${FAMILY}"
OUT_ROOT="outputs/eval_logs/${OUT_LEAF}"
LEVELS="${LEVELS:-base target language location env}"
REPEAT="${REPEAT:-1}"
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: REPEAT must be a positive integer"; exit 1; }
SEED_ARGS=()
if [ -n "${SEED:-}" ]; then
  SEED_ARGS=(--seed "$SEED")
fi

export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

[ -d "$FAM_ROOT" ] || { echo "ERROR: no family dir: $FAM_ROOT"; exit 1; }
[ -f "$CFG" ]      || { echo "ERROR: no config: $CFG"; exit 1; }

# --- plan: for every task, emit (task/level, bucket, gpu) for each requested level. ---
# ID = base; OOD = target/language/location/env. GPU-dynamics is a per-TASK property
# (liquid scenes), read once from the task's base diagnostics and applied to all its levels.
PLAN_FILE="$(mktemp)"
trap 'rm -f "$PLAN_FILE" "$PLAN_FILE.x"' EXIT
if ! "$PY" - "$FAM_ROOT" "$LEVELS" > "$PLAN_FILE" <<'PYEOF'
import sys, json, pathlib
fam_root = pathlib.Path(sys.argv[1])
levels = sys.argv[2].split()
allowed = {"base", "target", "language", "location", "env"}
if not levels or len(levels) != len(set(levels)) or not set(levels) <= allowed:
    raise SystemExit("LEVELS must contain distinct benchmark conditions")

def load(p):
    # diagnostics may be single-line OR pretty-printed multi-line (dusty) -> raw_decode
    return json.JSONDecoder().raw_decode(p.read_text().lstrip())[0]

tasks = sorted(p.name for p in fam_root.iterdir()
               if p.is_dir() and p.name.startswith("task_"))
plan = []
for t in tasks:
    base = load(fam_root / t / "base" / "diagnostics.jsonl")
    gpu = 1 if "liquid" in (base.get("pipeline") or "") else 0   # liquid task -> GPU dynamics
    for lvl in levels:
        d = fam_root / t / lvl / "diagnostics.jsonl"
        if not d.exists():
            raise SystemExit(f"Missing requested scenario diagnostics: {d}")
        if not (d.parent / "scene_ep1.json").is_file():
            raise SystemExit(f"Missing requested scene snapshot: {d.parent / 'scene_ep1.json'}")
        if not (load(d).get("prompt") or "").strip():
            raise SystemExit(f"Missing prompt in requested scenario: {d}")
        bucket = "ID" if lvl == "base" else f"OOD/{lvl}"
        plan.append((f"{t}/{lvl}", bucket, gpu))
for key, bucket, gpu in sorted(plan, key=lambda x: (x[2], x[1])):   # gpu=0 first, then by bucket
    print(f"{key}\t{bucket}\t{gpu}")
PYEOF
then
  echo "ERROR: cannot construct complete family plan"
  exit 1
fi

# REPEAT=N: repeat every planned instance N times (eval is stochastic).
if [ "$REPEAT" -gt 1 ]; then
  awk -v n="${REPEAT}" '{for(i=0;i<n;i++)print}' "$PLAN_FILE" > "$PLAN_FILE.x" && mv "$PLAN_FILE.x" "$PLAN_FILE"
  echo "REPEAT=${REPEAT}: each instance run ${REPEAT}x"
fi

N=$(awk 'END {print NR}' "$PLAN_FILE")
[ "$N" -gt 0 ] || { echo "ERROR: no instances planned for $FAMILY (levels: $LEVELS)"; rm -f "$PLAN_FILE"; exit 1; }
echo "@@@@@ eval_family $FAMILY  ($N instances)  ->  $OUT_ROOT @@@@@"
echo "--- plan (count | bucket | gpu) ---"; awk -F'\t' '{print $2" gpu="$3}' "$PLAN_FILE" | sort | uniq -c

# Guard: never silently clobber an existing (e.g. finalized) log dir.
if [ -d "$OUT_ROOT" ] && [ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]; then
  if [ "${FORCE:-0}" = "1" ]; then echo "FORCE=1: clearing $OUT_ROOT"; rm -rf "$OUT_ROOT"
  else echo "ERROR: $OUT_ROOT exists and is non-empty — pass a fresh output_leaf or FORCE=1."; rm -f "$PLAN_FILE"; exit 1; fi
fi

mkdir -p "$OUT_ROOT"
if ! "$PY" - "$PLAN_FILE" "$OUT_ROOT" "$CFG" "${SEED:-}" "$FAMILY" <<'PYEOF'
import json, pathlib, sys, yaml
plan, root, config, seed_override, family = sys.argv[1:]
cfg = yaml.safe_load(pathlib.Path(config).read_text()) or {}
seed = int(seed_override) if seed_override else cfg.get("seed")
if seed is not None and type(seed) is not int:
    raise SystemExit("Evaluation seed must be an integer or null")
rows = []
for line in pathlib.Path(plan).read_text().splitlines():
    scene, bucket, _gpu = line.split("\t")
    rows.append({"results_file": f"{bucket}/results.jsonl", "scene_name": scene, "seed": seed})
(pathlib.Path(root) / "expected_rollouts.json").write_text(json.dumps({
    "schema_version": 1, "family": family, "rollouts": rows,
}, indent=2) + "\n")
PYEOF
then
  echo "ERROR: cannot save expected rollout manifest"
  exit 1
fi

i=0
failures=0
while IFS=$'\t' read -r key bucket gpu; do
  i=$((i+1))
  parent="$(dirname "$bucket")"; leaf="$(basename "$bucket")"
  outdir="$OUT_ROOT"; [ "$parent" != "." ] && outdir="$OUT_ROOT/$parent"
  if [ "$gpu" = "1" ]; then export EVAL_USE_GPU_DYNAMICS=1; else unset EVAL_USE_GPU_DYNAMICS; fi
  echo "--- hold 5s before $key ($bucket, gpu=$gpu) ---"; sleep 5
  echo "############ [$i/$N] $key START $(date +%H:%M:%S) -> $bucket ############"
  if "$PY" -m maniguard.eval.benchmark --config "$CFG" --host 127.0.0.1 --port 8000 \
      --benchmark-root "$FAM_ROOT" --scenes "$key" --max-scenes 1 \
      --output-dir "$outdir" --run-name "$leaf" \
      --metrics success safety --headless "${SEED_ARGS[@]}"; then
    echo "############ [$i/$N] $key OK   $(date +%H:%M:%S) ############"
  else
    rc=$?
    failures=$((failures+1))
    echo "############ [$i/$N] $key exit=$rc $(date +%H:%M:%S) ############"
  fi
done < "$PLAN_FILE"
rm -f "$PLAN_FILE"

# Per-bucket summary in the paper's headline metrics (pure stdlib, any python).
if ! "$PY" tools/eval_summary.py "$OUT_ROOT" --full; then
  failures=$((failures+1))
fi
if [ "$failures" -gt 0 ]; then
  echo "ERROR: family run has failed processes or incomplete results"
  exit 1
fi
echo "@@@@@ eval_family $FAMILY DONE $(date +%H:%M:%S) @@@@@"
