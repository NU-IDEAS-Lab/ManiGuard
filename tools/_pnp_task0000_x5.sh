#!/usr/bin/env bash
# One-off: collect 5 successful PnP trajectories on task_0000 by sweeping
# seeds until quota is met. Output: outputs/pnp_task0000_x5/.
set -uo pipefail

ROOT=/data/Projects/SENTINEL-Lite
TASK_DIR="$ROOT/datasets/6fam-base-20260513/clutter_pickup/task_0000/base"
OUT="$ROOT/outputs/pnp_task0000_x5"
QUOTA=${QUOTA:-5}
MAX_SEEDS=${MAX_SEEDS:-30}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1200}
mkdir -p "$OUT"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
export OMNIGIBSON_HEADLESS=1

check_success() {
  python3 -c "
import json
try:
    d = json.load(open('$1'))
    print('True' if d.get('phase_b', {}).get('success') else 'False')
except Exception:
    print('False')
"
}

# Count any pre-existing successes (resumable).
n_succ=0
for d in "$OUT"/seed_*; do
  [ -d "$d" ] || continue
  if [ -f "$d/result.json" ] && [ "$(check_success "$d/result.json")" = "True" ]; then
    n_succ=$((n_succ + 1))
  fi
done
echo "[task0000_x5] starting with $n_succ existing successes (quota=$QUOTA)"

t_overall=$(date +%s)
for seed in $(seq 0 $((MAX_SEEDS-1))); do
  if [ "$n_succ" -ge "$QUOTA" ]; then
    echo "[task0000_x5] quota $QUOTA met, stopping"
    break
  fi
  sd="$OUT/seed_$(printf '%02d' "$seed")"
  if [ -f "$sd/result.json" ]; then
    prev=$(check_success "$sd/result.json")
    if [ "$prev" = "True" ]; then
      echo "[task0000_x5] seed=$seed: prior SUCCESS on disk, skip"
      continue
    fi
  fi
  mkdir -p "$sd"
  t0=$(date +%s)
  timeout "$PER_RUN_TIMEOUT" conda run -n behavior --no-capture-output \
    python -m tools.pick_and_place_from_dataset \
      --task-dir "$TASK_DIR" --episode 1 \
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
  if [ "$success" = "True" ]; then
    n_succ=$((n_succ + 1))
  fi
  echo "[task0000_x5] seed=$seed wall=${wall}s rc=$rc success=$success fail=$fail_step  (succ=$n_succ/$QUOTA)"
done

t_total=$(( $(date +%s) - t_overall ))
echo ""
echo "[task0000_x5] DONE total_wall=${t_total}s  successes=$n_succ/$QUOTA"
ls "$OUT" | grep "^seed_" | sort | while read sd; do
  rj="$OUT/$sd/result.json"
  if [ -f "$rj" ]; then
    s=$(check_success "$rj")
    echo "[task0000_x5]   $sd: success=$s"
  fi
done
