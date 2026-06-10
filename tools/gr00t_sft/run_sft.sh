#!/usr/bin/env bash
# Fine-tune GR00T N1.6 (NEW_EMBODIMENT) on a ManiGuard joint LeRobot dataset.
#
# Wraps the upstream Isaac-GR00T launcher (examples/finetune.sh -> launch_finetune.py).
# Default tuning per the official N1.6 §3 recipe: freeze the VLM (LLM + visual),
# train projector + diffusion action head (no LoRA in N1.6). Single H100.
#
# Prereqs (once per shell), e.g. on Quest:
#   source "$GR00T_HOME/.venv/bin/activate"
#   module load ffmpeg/6.1-gcc-11.2.0   # torchcodec needs FFmpeg libs to load + decode H.264
#   export WANDB_API_KEY=...             # USE_WANDB=1 by default
#   # base model nvidia/GR00T-N1.6-3B is fetched via HF (needs HF_TOKEN)
#   # flash-attn (built from source) needs NO special runtime env; CUDA is bundled in torch.
#   # The dataset must be H.264 (run prepare_dataset.py, which transcodes AV1->H.264).
#
# The dataset must already be GR00T-ready (run tools/gr00t_sft/prepare_dataset.py
# first to add meta/modality.json + stats).
#
# Usage:
#   bash tools/gr00t_sft/run_sft.sh --dataset <lerobot_dir> --output <ckpt_dir> \
#        [--steps 3000] [--batch 32] [--save-steps 1000] [--save-limit 3] \
#        [--workers 8] [--exp-name stack] [-- <extra launch_finetune.py args>...]
set -euo pipefail

MANIGUARD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GR00T_HOME="${GR00T_HOME:-$HOME/projects/Isaac-GR00T}"
MODALITY_CONFIG="$MANIGUARD_HOME/maniguard/gr00t_sft/maniguard_embodiment.py"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
WANDB_PROJECT="${WANDB_PROJECT:-maniguard-gr00t-sft}"

DATASET=""
OUTPUT=""
STEPS=3000
BATCH=32
SAVE_STEPS=1000
SAVE_LIMIT=3
WORKERS=8
EXP_NAME=""
EXTRA=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --save-limit) SAVE_LIMIT="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --exp-name)   EXP_NAME="$2"; shift 2 ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help)    sed -n '1,28p' "$0"; exit 0 ;;
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
EXP_NAME="${EXP_NAME:-$(basename "$OUTPUT")}"

export NUM_GPUS=1
export MAX_STEPS="$STEPS"
export GLOBAL_BATCH_SIZE="$BATCH"
export SAVE_STEPS="$SAVE_STEPS"
export DATALOADER_NUM_WORKERS="$WORKERS"
export USE_WANDB="${USE_WANDB:-1}"

echo "[run_sft] family=$EXP_NAME steps=$STEPS batch=$BATCH save_steps=$SAVE_STEPS save_limit=$SAVE_LIMIT"
echo "[run_sft] dataset=$DATASET"
echo "[run_sft] output=$OUTPUT"
echo "[run_sft] base=$BASE_MODEL  modality=$MODALITY_CONFIG"

# Override finetune.sh's hardcoded --save_total_limit (5) via the `--` passthrough
# (last value wins); append any user EXTRA after it.
PASSTHRU=(--save_total_limit "$SAVE_LIMIT")
if [ "${#EXTRA[@]}" -gt 0 ]; then
    PASSTHRU+=("${EXTRA[@]}")
fi

cd "$GR00T_HOME"
exec bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --output-dir "$OUTPUT" \
    --experiment-name "$EXP_NAME" \
    --wandb-project "$WANDB_PROJECT" \
    -- "${PASSTHRU[@]}"
