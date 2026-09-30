#!/usr/bin/env bash
# Fine-tune SmolVLA on a ManiGuard 2-cam joint LeRobot dataset.
#
# Wraps LeRobot's native `lerobot-train` CLI (SmolVLA is LeRobot-native — no config
# registry / embodiment registration; the policy's features are derived straight
# from the dataset's standard `observation.*` / `action` keys). The dataset must
# already be a 2-cam, standard-keyed copy — run tools/smolvla_sft/prepare_dataset.py
# first. Warm-starts from `lerobot/smolvla_base`; the default recipe freezes the
# vision encoder + trains the action expert (SmolVLA's built-in `train_expert_only`,
# no LoRA — same "freeze VLM" strategy as the GR00T N1.6 path).
#
# Use the pinned LeRobot runtime and compatibility patch documented in
# docs/fine_tuning/smolvla.md. Dataset preparation produces a v3.0 copy.
# --batch is per GPU; --gpus selects the number of training processes.
#
# Prereqs (once per shell, in the lerobot env):
#   pip install -e <lerobot_clone>[smolvla]     # provides the `lerobot-train` entry point
#   export HF_TOKEN=...  WANDB_API_KEY=...        # base-model pull + online logs (checked below)
#
# Usage:
#   bash tools/smolvla_sft/run_sft.sh --dataset <prepared_lerobot_dir> --repo-id <id> \
#        --output <ckpt_dir> [--steps 20000] [--batch 64] [--gpus 1] [--workers 16] \
#        [--exp-name clutter] [--save-freq 5000] [-- <extra lerobot-train args>...]
#
# Pushing the trained policy to HF is a separate step (tools/smolvla_sft/push_to_hf.py,
# invoked by run_all.sh), mirroring the GR00T path.
set -euo pipefail

MANIGUARD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-base-joint-2cam}"
# Saved base-model camera names; the prepared dataset uses top/wrist.
RENAME_MAP="${RENAME_MAP:-}"
if [ -z "$RENAME_MAP" ]; then
    RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
fi
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
TOLERANCE_S="${TOLERANCE_S:-0.01}"

DATASET=""
REPO_ID=""
OUTPUT=""
STEPS=20000
BATCH=64
GPUS=1
WORKERS=16
SAVE_FREQ=5000
EXP_NAME=""
EXTRA=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --repo-id)    REPO_ID="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --save-freq)  SAVE_FREQ="$2"; shift 2 ;;
        --exp-name)   EXP_NAME="$2"; shift 2 ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help)    sed -n '1,32p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$DATASET" ] || { echo "Missing --dataset" >&2; exit 1; }
[ -n "$REPO_ID" ] || { echo "Missing --repo-id" >&2; exit 1; }
[ -n "$OUTPUT" ]  || { echo "Missing --output" >&2; exit 1; }
DATASET="$(realpath -m "$DATASET")"
OUTPUT="$(realpath -m "$OUTPUT")"
[ -f "$DATASET/meta/info.json" ] || {
    echo "ERROR: $DATASET/meta/info.json missing — run prepare_dataset.py first." >&2
    exit 1
}
EXP_NAME="${EXP_NAME:-$(basename "$OUTPUT")}"

# --- pre-flight (fail fast) ---
[ -n "${WANDB_API_KEY:-}" ] || { echo "ERROR: WANDB_API_KEY unset (online training logs)." >&2; exit 1; }
[ -n "${HF_TOKEN:-}" ]      || { echo "ERROR: HF_TOKEN unset (base-model $BASE_MODEL pull)." >&2; exit 1; }

# Shared dataset cache: same root every model track resolves LeRobot datasets from,
# so each model-agnostic datagen dataset is downloaded once (aligned with openpi's
# HF_LEROBOT_HOME and gr00t's hf-download). We pass the prepared copy by explicit
# path below, so this only keeps LeRobot's default resolution consistent.
export HF_LEROBOT_HOME="${MANIGUARD_SFT_DATA_ROOT:-$MANIGUARD_HOME/outputs/sft_datasets}"
mkdir -p "$HF_LEROBOT_HOME"

echo "[run_sft] family/exp=$EXP_NAME steps=$STEPS batch=$BATCH workers=$WORKERS save_freq=$SAVE_FREQ"
echo "[run_sft] dataset=$DATASET (repo_id=$REPO_ID)"
echo "[run_sft] output=$OUTPUT  base=$BASE_MODEL  wandb_project=$WANDB_PROJECT"

# wandb online by default for live visibility (aligned with the openpi + gr00t paths).
# `--wandb.enable=true` with WANDB_API_KEY set logs online; set WANDB_MODE=offline for
# a specific box that drops the online connection, then `wandb sync` after.
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be a positive integer" >&2; exit 1; }
LAUNCH=(lerobot-train)
if [ "$GPUS" -gt 1 ]; then
    LAUNCH=(accelerate launch --multi_gpu --num_processes "$GPUS"
            --main_process_port "${MASTER_PORT:-29500}" --mixed_precision bf16 "$(command -v lerobot-train)")
fi
"${LAUNCH[@]}" \
    --policy.path="$BASE_MODEL" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET" \
    --dataset.video_backend="$VIDEO_BACKEND" \
    --rename_map="$RENAME_MAP" \
    --tolerance_s="$TOLERANCE_S" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --num_workers="$WORKERS" \
    --save_freq="$SAVE_FREQ" \
    --output_dir="$OUTPUT" \
    --job_name="$EXP_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    "${EXTRA[@]+"${EXTRA[@]}"}"

echo "[run_sft] training done -> $OUTPUT"
