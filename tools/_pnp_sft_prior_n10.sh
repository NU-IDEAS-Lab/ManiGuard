#!/usr/bin/env bash
# Run 10 transport variants per task using grasp poses from a prior SFT
# dataset (default outputs/sft_dataset_2026-05-16/success_balanced).
#
# For each task with a successful prior:
#   - Phase A loads the prior grasp pose, skips OBB sampling (~22s vs ~75s)
#   - Variant 0 runs Phase A replay (cache captured), Phase B EXECUTE
#   - Variants 1..9 restore post-Phase-A state, inject cached frames, run
#     Phase B EXECUTE only
#   - Early-exit-after-hover skips descend whenever the target is already
#     inside the goal region at end of hover
#
# All variants append to a single LeRobot v2.1 dataset under --lerobot-root.
# Per-task expected wall: ~4-5 min. 47 tasks -> ~3-4 hours.
set -uo pipefail

ROOT=/data/Projects/SENTINEL-Lite
DATA="$ROOT/datasets/6fam-base-20260513/clutter_pickup"
SFT_PRIOR="$ROOT/outputs/sft_dataset_2026-05-16/success_balanced"
OUT="$ROOT/outputs/pnp_sft_prior_n10"
LEROBOT_REPO=${LEROBOT_REPO:-sentinel/pnp_sft_prior_n10}
LEROBOT_ROOT="$ROOT/outputs/lerobot_pnp_sft_prior_n10"

N_VARIANTS=${N_VARIANTS:-10}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-900}   # 15 min/task cap
LIFT_MIN=0.08
LIFT_MAX=0.15
HOVER_MIN=0.08
HOVER_MAX=0.15
RECORD_RESOLUTION=256

mkdir -p "$OUT"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMNIGIBSON_HEADLESS=1

# Enumerate tasks that have a prior in the SFT dataset.
TASKS=$(ls "$SFT_PRIOR" | sed 's/__seed.*//' | sort -u)
n_tasks=$(echo "$TASKS" | wc -l)
echo "[sft_prior_n10] $n_tasks tasks with priors  target=$LEROBOT_ROOT"

t_overall=$(date +%s)
n_done=0
n_skipped=0
for task in $TASKS; do
  td="$DATA/$task/base"
  if [ ! -f "$td/diagnostics.jsonl" ]; then
    echo "[sft_prior_n10] $task: SKIP (no diagnostics.jsonl)"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  if [ ! -f "$td/scene_ep1.json" ] && [ -f "$td/scene_ep1_replay.json" ]; then
    ln -s scene_ep1_replay.json "$td/scene_ep1.json" 2>/dev/null || true
  fi
  task_out="$OUT/$task"
  if [ -f "$task_out/variants_summary.json" ]; then
    echo "[sft_prior_n10] $task: already done; skip (delete $task_out to retry)"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  mkdir -p "$task_out"

  echo ""
  echo "[sft_prior_n10] ============================================================"
  echo "[sft_prior_n10] === $task  $(date '+%H:%M:%S') ==="
  echo "[sft_prior_n10] ============================================================"

  t0=$(date +%s)
  timeout "$PER_RUN_TIMEOUT" conda run -n behavior --no-capture-output \
    python -m tools.pick_and_place_from_dataset \
      --task-dir "$td" --episode 1 \
      --max-candidates 400 --pick-timeout 300 \
      --transport-timeout 60 --ik-precheck \
      --seed 0 --out-dir "$task_out" \
      --record-sft --record-resolution "$RECORD_RESOLUTION" \
      --n-transport-variants "$N_VARIANTS" \
      --lift-z-min "$LIFT_MIN" --lift-z-max "$LIFT_MAX" \
      --hover-z-min "$HOVER_MIN" --hover-z-max "$HOVER_MAX" \
      --phase-a-grasp-from-dataset "$SFT_PRIOR" \
      --lerobot-repo-id "$LEROBOT_REPO" \
      --lerobot-root "$LEROBOT_ROOT" \
      > "$task_out/run.log" 2>&1
  rc=$?
  wall=$(($(date +%s) - t0))
  n_succ=$(python3 -c "
import json, pathlib
p = pathlib.Path('$task_out') / 'variants_summary.json'
if p.is_file():
    try: print(json.load(open(p)).get('n_succ', 0))
    except: print(0)
else: print(0)
" 2>/dev/null)
  total_eps=$(python3 -c "
import json, pathlib
p = pathlib.Path('$LEROBOT_ROOT') / 'meta' / 'info.json'
if p.is_file():
    try: print(json.load(open(p))['total_episodes'])
    except: print('?')
else: print(0)
" 2>/dev/null)
  df_avail=$(df -h /data | awk 'NR==2{print $4}')
  n_done=$((n_done + 1))
  echo "[sft_prior_n10] $task: wall=${wall}s rc=$rc n_succ=$n_succ/$N_VARIANTS  ds_eps=$total_eps  free=$df_avail  ($n_done/$n_tasks)"
done

t_total=$(( $(date +%s) - t_overall ))
echo ""
echo "[sft_prior_n10] DONE total_wall=${t_total}s  tasks_done=$n_done  skipped=$n_skipped"
echo "[sft_prior_n10] LeRobot dataset: $LEROBOT_ROOT"
python3 -c "
import json, pathlib
p = pathlib.Path('$LEROBOT_ROOT') / 'meta' / 'info.json'
if p.is_file():
    d = json.load(open(p))
    print(f'  total_episodes={d[\"total_episodes\"]}  total_frames={d[\"total_frames\"]}')
" 2>/dev/null
