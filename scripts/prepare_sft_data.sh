#!/usr/bin/env bash
# End-to-end data prep for Pi0.5 SFT on SENTINEL teleop demos.
#
# Stages (each idempotent):
#   1. Render obs (main + wrist @ 256x256) + 7D eef state per teleop HDF5.
#   2. Concatenate rendered episodes into a single LeRobot v2.1 dataset.
#   3. Compute per-feature norm_stats and drop them next to the Pi0.5 ckpt
#      so OpenPI's DataLoader can pick them up.
#
# Expects:
#   outputs/teleop/traj_*.hdf5            recorded via so101_franka_teleop.py
#   $SENTINEL_PI05_BASE                   Pi0.5 checkpoint root (has config.json,
#                                         model.safetensors). We create
#                                         $SENTINEL_PI05_BASE/assets/<asset_id>/
#                                         and write norm_stats.json there.
#
# Usage:
#   cd /home/nu-ideas-4080/Desktop/projects/SENTINEL-Lite
#   export SENTINEL_PI05_BASE=$PWD/RLinf-pi05-SFT-Stack-cube
#   bash scripts/prepare_sft_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${SENTINEL_PI05_BASE:?Set SENTINEL_PI05_BASE to the Pi0.5 checkpoint root}"

PY="/home/nu-ideas-4080/miniconda3/envs/behavior/bin/python"
INPUT_DIR="${INPUT_DIR:-outputs/teleop}"
RENDERED_DIR="${RENDERED_DIR:-outputs/teleop_rendered}"
LEROBOT_ROOT="${LEROBOT_ROOT:-outputs/lerobot_datasets}"
REPO_ID="${REPO_ID:-sentinel/goblet_pick_place}"
ASSET_ID="${ASSET_ID:-sentinel_goblet_pick_place}"
PROMPT="${PROMPT:-pick up the goblet and place it on the plate}"

echo "[Prep] input_dir     = $INPUT_DIR"
echo "[Prep] rendered_dir  = $RENDERED_DIR"
echo "[Prep] lerobot_root  = $LEROBOT_ROOT"
echo "[Prep] repo_id       = $REPO_ID"
echo "[Prep] asset_id      = $ASSET_ID"
echo "[Prep] pi05_base     = $SENTINEL_PI05_BASE"

# -- Stage 1: render obs for each teleop HDF5 -------------------------------
mkdir -p "$RENDERED_DIR"
for in_path in "$INPUT_DIR"/traj_*.hdf5; do
    name=$(basename "$in_path")
    out_path="$RENDERED_DIR/$name"
    if [[ -f "$out_path" && -s "$out_path" ]]; then
        echo "[Prep] Skip Stage 1 (exists): $out_path"
        continue
    fi
    echo "[Prep] Stage 1: $in_path"
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
    "$PY" python -m sentinel.data.playback \
        --input "$in_path" --output "$out_path" --save-mp4
done

# -- Stage 2: one LeRobot dataset from all rendered HDF5 --------------------
DATASET_ROOT="$LEROBOT_ROOT/$REPO_ID"
if [[ -d "$DATASET_ROOT/data" ]]; then
    echo "[Prep] Dataset already exists at $DATASET_ROOT (delete to regenerate)"
else
    echo "[Prep] Stage 2: building LeRobot dataset at $DATASET_ROOT"
    "$PY" python -m sentinel.data.lerobot_export \
        --input-dir "$RENDERED_DIR" \
        --repo-id "$REPO_ID" \
        --prompt "$PROMPT" \
        --root "$DATASET_ROOT"
fi

# -- Stage 3: norm_stats.json next to the Pi0.5 ckpt ------------------------
NORM_DIR="$SENTINEL_PI05_BASE/assets/$ASSET_ID"
echo "[Prep] Stage 3: writing norm_stats to $NORM_DIR"
"$PY" python -m sentinel.data.norm_stats \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$NORM_DIR"

echo
echo "[Prep] Done. Launch SFT with:"
echo
echo "    export SENTINEL_PI05_BASE=$SENTINEL_PI05_BASE"
echo "    export SENTINEL_LEROBOT_ROOT=$REPO_ROOT/$LEROBOT_ROOT"
echo "    export EMBODIED_PATH=$REPO_ROOT/RLinf/examples/sft"
echo "    sudo -E bash RLinf/examples/sft/run_embodiment_sft.sh sentinel_goblet_sft_openpi runner.max_steps=1"
