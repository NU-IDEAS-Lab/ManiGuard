#!/usr/bin/env bash
# Watchdog wrapper for maniguard.rl.algorithms.ppo: relaunches the training
# from the latest checkpoint whenever the process exits non-zero. Required
# because OmniGibson + PhysX has multiple articulation-corruption pathways
# (AG-fire-during-step, multi-env shared-view invalidation, scene-query
# NaN inputs) that periodically segfault long-running PPO. See
# maniguard/_omnigibson_patches.py:_patch_create_joint_skip_render for the
# one we already patched at the source.
#
# Usage:
#   tools/ppo_watchdog.sh <output-dir> -- <ppo-args...>
# Example:
#   tools/ppo_watchdog.sh outputs/rl_runs/task0022_wd -- \
#       --category goblet --model nawrfs \
#       --diagnostics-file datasets/.../task_0022/base/diagnostics.jsonl \
#       --grasp-dataset-dir outputs/grasp_datasets/task0022_inscene \
#       --num-envs 1 --max-steps 200 --total-timesteps 1000000 \
#       --n-steps 256 --batch-size 128 --save-freq 10000 \
#       --wandb --wandb-project sentinel-grasp-reset

set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <output-dir> -- <ppo args...>" >&2
    exit 2
fi

OUTPUT_DIR="$1"; shift
if [ "$1" != "--" ]; then
    echo "expected -- separator before PPO args, got: $1" >&2
    exit 2
fi
shift  # drop the --

PROJECT_ROOT="/data/Projects/ManiGuard"
PYTHON="/home/simonzhan/anaconda3/envs/behavior/bin/python"
MAX_RETRIES=100
COOLDOWN_S=5

cd "$PROJECT_ROOT"

# Pre-flight: warn if the user passed --output-dir or --resume-from in PPO_ARGS.
# We manage --output-dir ourselves; --resume-from gets injected per attempt.
for arg in "$@"; do
    case "$arg" in
        --output-dir|--output-dir=*|--resume-from|--resume-from=*)
            echo "[watchdog] do not pass $arg in PPO args; the watchdog manages these" >&2
            exit 2 ;;
    esac
done

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))

    # Pick the highest-step ckpt under <output-dir>/ckpts. SB3's
    # CheckpointCallback names files "ppo_<n>_steps.zip".
    LATEST=""
    if [ -d "$OUTPUT_DIR/ckpts" ]; then
        LATEST=$(ls -1 "$OUTPUT_DIR/ckpts"/ppo_*_steps.zip 2>/dev/null \
            | sed -E 's|.*/ppo_([0-9]+)_steps\.zip|\1 &|' \
            | sort -n \
            | tail -1 \
            | awk '{print $2}')
    fi

    RESUME_ARGS=()
    if [ -n "$LATEST" ]; then
        echo "[watchdog $(date +%H:%M:%S)] attempt $attempt — resuming from $LATEST" >&2
        RESUME_ARGS=(--resume-from "$LATEST")
    else
        echo "[watchdog $(date +%H:%M:%S)] attempt $attempt — starting fresh" >&2
    fi

    PYTHONPATH="$PROJECT_ROOT" \
    TMPDIR="$PROJECT_ROOT/outputs/_og_tmp" \
    OMNI_KIT_ACCEPT_EULA=yes \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
    OMNIGIBSON_HEADLESS=1 \
    "$PYTHON" -m maniguard.rl.algorithms.ppo \
        --output-dir "$OUTPUT_DIR" \
        "$@" \
        "${RESUME_ARGS[@]}"

    EXIT_CODE=$?

    # Successful completion: PPO writes ppo_final.zip and calls os._exit(0).
    if [ -f "$OUTPUT_DIR/ppo_final.zip" ]; then
        echo "[watchdog $(date +%H:%M:%S)] training complete — ppo_final.zip exists, exiting" >&2
        exit 0
    fi

    # Anything else: crash. Loop and resume.
    echo "[watchdog $(date +%H:%M:%S)] attempt $attempt exited code=$EXIT_CODE — restarting in ${COOLDOWN_S}s" >&2
    sleep "$COOLDOWN_S"
done

echo "[watchdog $(date +%H:%M:%S)] hit MAX_RETRIES=$MAX_RETRIES, giving up" >&2
exit 1
