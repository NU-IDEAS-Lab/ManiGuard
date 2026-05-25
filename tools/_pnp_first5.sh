#!/usr/bin/env bash
# One-off: sweep first 5 clutter_pickup tasks (0000..0004), stop on
# first SFT-recorded success per task. Output: outputs/pnp_first5/.
set -uo pipefail

ROOT=/data/Projects/SENTINEL-Lite
DATA="$ROOT/datasets/6fam-base-20260513/clutter_pickup"
OUT="$ROOT/outputs/pnp_first5"
MAX_SEEDS=${MAX_SEEDS:-10}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1200}
mkdir -p "$OUT"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
export OMNIGIBSON_HEADLESS=1

check_success() {
  python3 -c "
import json, sys
try:
    d = json.load(open('$1'))
    print('True' if d.get('phase_b', {}).get('success') else 'False')
except Exception:
    print('False')
"
}

t_overall=$(date +%s)
for task in task_0000 task_0001 task_0002 task_0003 task_0004; do
  td="$DATA/$task/base"
  if [ ! -f "$td/diagnostics.jsonl" ]; then
    echo "[first5] $task: SKIP (no diagnostics.jsonl)"
    continue
  fi
  if [ ! -f "$td/scene_ep1.json" ] && [ -f "$td/scene_ep1_replay.json" ]; then
    ln -s scene_ep1_replay.json "$td/scene_ep1.json" 2>/dev/null || true
  fi
  echo ""
  echo "[first5] ============================================================"
  echo "[first5] === $task  ($(date '+%H:%M:%S')) ==="
  echo "[first5] ============================================================"
  succeeded=False
  for seed in $(seq 0 $((MAX_SEEDS-1))); do
    sd="$OUT/$task/seed_$(printf '%02d' "$seed")"
    if [ -f "$sd/result.json" ]; then
      prev=$(check_success "$sd/result.json")
      if [ "$prev" = "True" ]; then
        echo "[first5] $task seed=$seed: prior SUCCESS on disk, skipping rest"
        succeeded=True
        break
      fi
    fi
    mkdir -p "$sd"
    t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" conda run -n behavior --no-capture-output \
      python -m tools.pick_and_place_from_dataset \
        --task-dir "$td" --episode 1 \
        --max-candidates 400 --pick-timeout 600 \
        --transport-timeout 60 --ik-precheck \
        --seed "$seed" --out-dir "$sd" \
        --record-sft --record-resolution 256 \
        > "$sd/run.log" 2>&1
    rc=$?
    wall=$(($(date +%s) - t0))
    if [ -f "$sd/result.json" ]; then
      success=$(check_success "$sd/result.json")
      fail_step=$(python3 -c "
import json
try:
    d = json.load(open('$sd/result.json'))
    print(d.get('fail_step', '-'))
except Exception:
    print('-')
")
    else
      success=False
      fail_step="no_result_json"
    fi
    echo "[first5] $task seed=$seed  wall=${wall}s  rc=$rc  success=$success  fail_step=$fail_step"
    if [ "$success" = "True" ]; then
      succeeded=True
      break
    fi
  done
  if [ "$succeeded" = "False" ]; then
    echo "[first5] $task: NO SUCCESS in $MAX_SEEDS seeds"
  fi
done

t_total=$(( $(date +%s) - t_overall ))
echo ""
echo "[first5] ============================================================"
echo "[first5] DONE (total wall=${t_total}s)"
echo "[first5] ============================================================"
for task in task_0000 task_0001 task_0002 task_0003 task_0004; do
  found="-"
  for seed in $(seq 0 $((MAX_SEEDS-1))); do
    sd="$OUT/$task/seed_$(printf '%02d' "$seed")"
    if [ -f "$sd/result.json" ]; then
      s=$(check_success "$sd/result.json")
      if [ "$s" = "True" ]; then
        found="seed_$(printf '%02d' "$seed")"
        break
      fi
    fi
  done
  echo "[first5]   $task: $found"
done
