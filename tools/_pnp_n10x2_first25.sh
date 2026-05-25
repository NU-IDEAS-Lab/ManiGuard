#!/usr/bin/env bash
# Run --n-transport-variants 10 over the first 25 clutter_pickup tasks,
# collecting 2 different held grasps per task (each via its own
# pick_and_place_from_dataset invocation with a distinct seed). Per task
# the driver sweeps seeds 0..MAX_SEEDS-1 and stops once QUOTA_GRASPS
# seeds have Phase A held. Output: outputs/variants_n10x2_first25/.
#
# 2 grasps × 10 variants = 20 trajectories per task, same volume as
# n_variants=20 single-grasp but with more grasp diversity.
set -uo pipefail

ROOT=/data/Projects/SENTINEL-Lite
DATA="$ROOT/datasets/6fam-base-20260513/clutter_pickup"
OUT="$ROOT/outputs/variants_n10x2_first25"
N_VARIANTS=${N_VARIANTS:-10}
QUOTA_GRASPS=${QUOTA_GRASPS:-2}            # held grasps per task
MAX_SEEDS=${MAX_SEEDS:-5}                  # seed budget per task
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1500}   # 25 min per seed (N=10 ~ 12 min + buffer)
LIFT_MIN=0.08
LIFT_MAX=0.15
HOVER_MIN=0.08
HOVER_MAX=0.15
RECORD_RESOLUTION=256
TASKS_LO=${TASKS_LO:-0}
TASKS_HI=${TASKS_HI:-24}   # inclusive
mkdir -p "$OUT"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
export OMNIGIBSON_HEADLESS=1

seed_phase_a_held() {
  # Returns "True" if the given seed dir has at least one successful variant.
  python3 - <<EOF
import json, pathlib
p = pathlib.Path("$1")
vs = p / "variants_summary.json"
if vs.is_file():
    try:
        s = json.load(open(vs))
        print("True" if s.get("n_succ", 0) > 0 else "False")
    except Exception:
        print("False")
else:
    held = False
    for r in p.glob("variant_*/result.json"):
        try:
            d = json.load(open(r))
            if d.get("phase_a", {}).get("held"):
                held = True
                break
        except Exception:
            pass
    print("True" if held else "False")
EOF
}

count_task_held_grasps() {
  # Count how many seed_* dirs under the task have Phase A held.
  local task_dir="$1"
  local n=0
  if [ -d "$task_dir" ]; then
    for sd in "$task_dir"/seed_*; do
      [ -d "$sd" ] || continue
      r=$(seed_phase_a_held "$sd")
      if [ "$r" = "True" ]; then n=$((n + 1)); fi
    done
  fi
  echo "$n"
}

t_overall=$(date +%s)
for ti in $(seq "$TASKS_LO" "$TASKS_HI"); do
  task=$(printf "task_%04d" "$ti")
  td="$DATA/$task/base"
  if [ ! -f "$td/diagnostics.jsonl" ]; then
    echo "[n10x2] $task: SKIP (no diagnostics.jsonl)"
    continue
  fi
  if [ ! -f "$td/scene_ep1.json" ] && [ -f "$td/scene_ep1_replay.json" ]; then
    ln -s scene_ep1_replay.json "$td/scene_ep1.json" 2>/dev/null || true
  fi
  task_out_root="$OUT/$task"
  mkdir -p "$task_out_root"

  echo ""
  echo "[n10x2] ============================================================"
  echo "[n10x2] === $task  $(date '+%H:%M:%S') ==="
  echo "[n10x2] ============================================================"

  held_count=$(count_task_held_grasps "$task_out_root")
  if [ "$held_count" -ge "$QUOTA_GRASPS" ]; then
    echo "[n10x2] $task: already has $held_count held grasp(s), skip"
    continue
  fi

  for seed in $(seq 0 $((MAX_SEEDS-1))); do
    if [ "$held_count" -ge "$QUOTA_GRASPS" ]; then break; fi
    sd="$task_out_root/seed_$(printf '%02d' "$seed")"

    # Resume: if this seed has prior held output, count + skip.
    if [ -d "$sd" ] && [ "$(seed_phase_a_held "$sd")" = "True" ]; then
      held_count=$((held_count + 1))
      echo "[n10x2] $task seed=$seed: prior held, count=$held_count/$QUOTA_GRASPS"
      continue
    fi
    # Wipe a partial seed dir before retry to keep things clean.
    rm -rf "$sd"
    mkdir -p "$sd"

    t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" conda run -n behavior --no-capture-output \
      python -m tools.pick_and_place_from_dataset \
        --task-dir "$td" --episode 1 \
        --max-candidates 400 --pick-timeout 600 \
        --transport-timeout 60 --ik-precheck \
        --seed "$seed" --out-dir "$sd" \
        --record-sft --record-resolution "$RECORD_RESOLUTION" \
        --n-transport-variants "$N_VARIANTS" \
        --lift-z-min "$LIFT_MIN" --lift-z-max "$LIFT_MAX" \
        --hover-z-min "$HOVER_MIN" --hover-z-max "$HOVER_MAX" \
        > "$sd/run.log" 2>&1
    rc=$?
    wall=$(($(date +%s) - t0))
    held=$(seed_phase_a_held "$sd")
    n_succ=$(python3 -c "
import json, pathlib
p = pathlib.Path('$sd') / 'variants_summary.json'
if p.is_file():
    try:
        print(json.load(open(p)).get('n_succ', 0))
    except Exception:
        print(0)
else:
    print(0)
" 2>/dev/null)
    if [ "$held" = "True" ]; then
      held_count=$((held_count + 1))
    fi
    df_avail=$(df -h /data | awk 'NR==2{print $4}')
    echo "[n10x2] $task seed=$seed wall=${wall}s rc=$rc held=$held n_succ=$n_succ/$N_VARIANTS  grasps=$held_count/$QUOTA_GRASPS  free=$df_avail"
  done
  if [ "$held_count" -lt "$QUOTA_GRASPS" ]; then
    echo "[n10x2] $task: only $held_count/$QUOTA_GRASPS grasps held in $MAX_SEEDS seeds"
  fi
done

t_total=$(( $(date +%s) - t_overall ))
echo ""
echo "[n10x2] DONE total_wall=${t_total}s"
echo "[n10x2] task summary:"
for ti in $(seq "$TASKS_LO" "$TASKS_HI"); do
  task=$(printf "task_%04d" "$ti")
  task_out_root="$OUT/$task"
  if [ -d "$task_out_root" ]; then
    n=$(count_task_held_grasps "$task_out_root")
    n_var=0
    for sd in "$task_out_root"/seed_*; do
      [ -d "$sd" ] || continue
      ns=$(python3 -c "
import json, pathlib
p = pathlib.Path('$sd') / 'variants_summary.json'
if p.is_file():
    try:
        print(json.load(open(p)).get('n_succ', 0))
    except: print(0)
else: print(0)
")
      n_var=$((n_var + ns))
    done
    echo "[n10x2]   $task: $n/$QUOTA_GRASPS grasps, $n_var variants total"
  else
    echo "[n10x2]   $task: -"
  fi
done
