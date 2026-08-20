#!/usr/bin/env bash
# Fine-tune GR00T N1.6 (NEW_EMBODIMENT) on a ManiGuard joint LeRobot dataset.
#
# Wraps the upstream Isaac-GR00T launcher (examples/finetune.sh -> launch_finetune.py).
# Default tuning per the official N1.6 §3 recipe: freeze the VLM (LLM + visual),
# train projector + diffusion action head (no LoRA in N1.6).
#
# Prereqs (once per shell):
#   source "$GR00T_HOME/.venv/bin/activate"
#   # ensure FFmpeg libs are on PATH (torchcodec needs them to decode H.264)
#   export WANDB_API_KEY=...             # USE_WANDB=1 by default
#   # base model nvidia/GR00T-N1.6-3B is fetched via HF (needs HF_TOKEN)
#   # flash-attn needs NO special runtime env; CUDA is bundled in torch.
#   # The dataset is already H.264; prepare_dataset.py symlinks it (no transcode).
#
# The dataset must already be GR00T-ready (run tools/gr00t_sft/prepare_dataset.py
# first to add meta/modality.json + stats).
#
# Usage:
#   bash tools/gr00t_sft/run_sft.sh --dataset <lerobot_dir> --output <ckpt_dir> \
#        [--steps 3000] [--batch 32] [--save-steps 1000] [--save-limit 3] \
#        [--workers 6] [--exp-name stack] [--modality-config sim|real|<path>] \
#        [-- <extra launch_finetune.py args>...]
#
#   --modality-config: 'sim' (default, datagen datasets -- absolute joint targets) or
#   'real' (DROID-schema teleop -- joint VELOCITY). It MUST match the --embodiment-config
#   used by prepare_dataset.py; the mismatch check below refuses to start otherwise.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GR00T_HOME="${GR00T_HOME:-$REPO_ROOT}"          # self-contained: the fork IS the GR00T repo
MODALITY_CONFIG="$REPO_ROOT/maniguard/gr00t_sft/maniguard_embodiment.py"
MODALITY_CONFIG_REAL="$REPO_ROOT/maniguard/gr00t_sft/maniguard_embodiment_real.py"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
WANDB_PROJECT="${WANDB_PROJECT:-maniguard-gr00tN1d6}"

DATASET=""
OUTPUT=""
STEPS=3000
BATCH=256          # 8-card config: GLOBAL batch (divided across GPUS)
GPUS=8
LR=2e-4            # peak LR (cosine); sqrt-scaled from 1e-4@batch64 for global batch 256
SAVE_STEPS=1000
SAVE_LIMIT=4
WORKERS=6          # PER-GPU dataloader workers (torchrun: total = WORKERS*GPUS)
EXP_NAME=""
EXTRA=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --save-limit) SAVE_LIMIT="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --exp-name)   EXP_NAME="$2"; shift 2 ;;
        --modality-config)
            case "$2" in
                sim)  MODALITY_CONFIG="$REPO_ROOT/maniguard/gr00t_sft/maniguard_embodiment.py" ;;
                real) MODALITY_CONFIG="$MODALITY_CONFIG_REAL" ;;
                *)    MODALITY_CONFIG="$2" ;;   # explicit path
            esac
            shift 2 ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help)    sed -n '1,27p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$DATASET" ] || { echo "Missing --dataset" >&2; exit 1; }
[ -n "$OUTPUT" ]  || { echo "Missing --output" >&2; exit 1; }
# Absolutize before we cd into GR00T_HOME, else relative paths resolve there.
DATASET="$(realpath -m "$DATASET")"
OUTPUT="$(realpath -m "$OUTPUT")"
[ -f "$DATASET/meta/modality.json" ] || {
    echo "ERROR: $DATASET/meta/modality.json missing — run prepare_dataset.py first." >&2
    exit 1
}
[ -f "$MODALITY_CONFIG" ] || { echo "ERROR: modality config not found: $MODALITY_CONFIG" >&2; exit 1; }

# The dataset's baked meta/modality.json and the modality config passed here MUST describe
# the same schema. A mismatch does not crash -- it silently trains on the wrong columns, or
# on `velocity - joint_position` for the arm. Detect it from a key only the real schema has.
ds_is_real=$(grep -q 'exterior_image_1_left' "$DATASET/meta/modality.json" && echo 1 || echo 0)
cfg_is_real=$(case "$MODALITY_CONFIG" in *_real.py) echo 1 ;; *) echo 0 ;; esac)
[ "$ds_is_real" = "$cfg_is_real" ] || {
    echo "ERROR: schema mismatch — dataset modality.json is $([ "$ds_is_real" = 1 ] && echo REAL || echo SIM)," >&2
    echo "       but --modality-config resolved to $([ "$cfg_is_real" = 1 ] && echo REAL || echo SIM): $MODALITY_CONFIG" >&2
    echo "       Re-run prepare_dataset.py with the matching --embodiment-config." >&2
    exit 1
}
EXP_NAME="${EXP_NAME:-$(basename "$OUTPUT")}"
[ $(( BATCH % GPUS )) -eq 0 ] || { echo "ERROR: BATCH ($BATCH) must be divisible by GPUS ($GPUS)." >&2; exit 1; }

# cap per-worker math threads so the DATALOADER_NUM_WORKERS processes don't oversubscribe
# the CPU (the openpi SFT lesson); respects an already-set value.
: "${OMP_NUM_THREADS:=1}"; : "${MKL_NUM_THREADS:=1}"; : "${OPENBLAS_NUM_THREADS:=1}"; : "${NUMEXPR_NUM_THREADS:=1}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
export NUM_GPUS="$GPUS"
export MAX_STEPS="$STEPS"
export GLOBAL_BATCH_SIZE="$BATCH"
export SAVE_STEPS="$SAVE_STEPS"
export DATALOADER_NUM_WORKERS="$WORKERS"
export USE_WANDB="${USE_WANDB:-1}"
# wandb online by default for live training visibility (aligned with the openpi SFT).
# If a specific server drops online history, set WANDB_MODE=offline to record locally;
# the block after training then syncs the full run.
export WANDB_MODE="${WANDB_MODE:-online}"

echo "[run_sft] family=$EXP_NAME steps=$STEPS batch=$BATCH save_steps=$SAVE_STEPS save_limit=$SAVE_LIMIT"
echo "[run_sft] dataset=$DATASET"
echo "[run_sft] output=$OUTPUT"
echo "[run_sft] base=$BASE_MODEL  modality=$MODALITY_CONFIG"

# Override finetune.sh's hardcoded --save_total_limit (5) via the `--` passthrough
# (last value wins); append any user EXTRA after it.
PASSTHRU=(--save_total_limit "$SAVE_LIMIT" --learning_rate "$LR")
if [ "${#EXTRA[@]}" -gt 0 ]; then
    PASSTHRU+=("${EXTRA[@]}")
fi

cd "$GR00T_HOME"
bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --output-dir "$OUTPUT" \
    --experiment-name "$EXP_NAME" \
    --wandb-project "$WANDB_PROJECT" \
    -- "${PASSTHRU[@]}"
rc=$?

# Upload the offline-recorded wandb run (full history) now that training finished.
if [ "$rc" = "0" ] && [ "${USE_WANDB:-1}" = "1" ] && [ "${WANDB_MODE:-}" = "offline" ]; then
    latest=$(ls -dt "$GR00T_HOME"/wandb/offline-run-* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "[run_sft] syncing wandb offline run: $latest"
        wandb sync "$latest" || echo "[run_sft] wandb sync failed (training itself completed OK)"
    fi
fi
exit "$rc"
