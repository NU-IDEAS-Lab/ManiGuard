#!/usr/bin/env bash
# Drive tools.render_pnp_for_sft over every successful seed in task_0000.
set -u
REPO_ROOT=/data/Projects/SENTINEL-Lite
cd "$REPO_ROOT"
COLLECT=$REPO_ROOT/outputs/pnp_multitask_collect/task_0000
OUT_ROOT=$REPO_ROOT/outputs/pnp_sft/task_0000
mkdir -p "$OUT_ROOT"

# Enumerate seeds whose result.json reports success=True.
seeds=()
for sd in $(ls -d "$COLLECT"/seed_* 2>/dev/null | sort); do
  rj="$sd/result.json"
  [ -f "$rj" ] || continue
  ok=$(python -c "import json,sys; print(json.load(open('$rj'))['phase_b']['success'])" 2>/dev/null)
  [ "$ok" = "True" ] || continue
  seeds+=("$(basename "$sd")")
done
echo "[driver] found ${#seeds[@]} successful seeds in $COLLECT"

n_done=0; n_fail=0
for seed in "${seeds[@]}"; do
  out="$OUT_ROOT/$seed"
  if [ -f "$out/rollout.hdf5" ] && [ -f "$out/rollout_wrist.mp4" ]; then
    n_done=$((n_done+1))
    echo "[driver] [$n_done/${#seeds[@]}] $seed already rendered — skip"
    continue
  fi
  rm -rf "$out" && mkdir -p "$out"
  echo "[driver] $seed → $out"
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \
    timeout 900 conda run -n behavior --no-capture-output \
    python -m tools.render_pnp_for_sft \
      --collect-dir "$COLLECT/$seed" --out-dir "$out" \
      > "$out/render.log" 2>&1
  rc=$?
  if [ -f "$out/rollout.hdf5" ] && [ -f "$out/rollout_wrist.mp4" ]; then
    n_done=$((n_done+1))
    n_steps=$(python -c "import h5py; f=h5py.File('$out/rollout.hdf5','r'); print(int(f.attrs.get('n_steps',-1)))" 2>/dev/null)
    echo "[driver]   ok  rc=$rc  n_steps=$n_steps  done=$n_done/${#seeds[@]}"
  else
    n_fail=$((n_fail+1))
    echo "[driver]   FAIL rc=$rc — see $out/render.log"
  fi
done
echo
echo "[driver] === FINAL TALLY ==="
echo "[driver]   rendered: $n_done"
echo "[driver]   failed  : $n_fail"
echo "[driver]   out_root: $OUT_ROOT"
