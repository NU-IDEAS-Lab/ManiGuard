#!/usr/bin/env bash
# Post-train one ManiGuard family from an installed LingBot source tree.
# Compute missing normalization statistics, then invoke upstream train.sh
# with the family dataset, checkpoint schedule, and output directory.
# The fixed step table targets approximately two passes at global batch 256;
# --steps can override it. Training YAML controls the global batch.
# Usage: bash tools/lingbot_sft/run_sft.sh --family clutter --data-root /path/to/data















set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

ORG="IDEAS-Lab-Northwestern"
CONFIG="configs/vla/maniguard/maniguard.yaml"
# compute_norm_stats.py accepts only the data/train sections -- passing the training config
# (which has a model section) is rejected by its parser, so it gets its own config.
NORM_CONFIG="configs/vla/norm_compute/maniguard.yaml"

# upstream's entrypoints are `torchrun scripts/<x>.py`, which puts scripts/ (not the repo
# root) on sys.path, so `import lingbotvla` fails unless the package is installed. The env
# builder does not `pip install -e .`, so put the repo root on PYTHONPATH for the children.
export PYTHONPATH="$(cd "$HERE/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Prepend FFMPEG_LIB_DIR when supplied so video decoders can locate
# FFmpeg shared libraries. This script does not choose a decoder fallback.



if [ -n "${FFMPEG_LIB_DIR:-}" ]; then
  export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi
ROBOT_CONFIG_ROOT="./configs/robot_configs"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/outputs/lingbot_sft}"
PRETRAIN_DIR="assets/pretrained"

# family -> total frames (the ONLY per-dataset value; steps derive from it).
declare -A FRAMES=(
  [clutter]=901520  [cabinet]=4172962  [stack]=2652083
  [jar]=946870      [lid]=1055142      [dusty]=1879498
)
# Configured per-family training steps for approximately two dataset passes
# at global batch 256. These values are not recomputed when GPUS changes.
declare -A STEPS=(
  [clutter]=7100  [cabinet]=32650  [stack]=20750
  [jar]=7400      [lid]=8250       [dusty]=14700
)

FAMILY=""; DATA_ROOT=""; GPUS="${GPUS:-8}"; FORCE_NORM=0; STEPS_OVERRIDE=""; OUT_OVERRIDE=""
EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --family)     FAMILY="$2"; shift 2 ;;
    --data-root)  DATA_ROOT="$2"; shift 2 ;;
    --gpus)       GPUS="$2"; shift 2 ;;
    --steps)      STEPS_OVERRIDE="$2"; shift 2 ;;
    --out)        OUT_OVERRIDE="$2"; shift 2 ;;
    --norm-stats) FORCE_NORM=1; shift ;;
    --)           shift; EXTRA=("$@"); break ;;
    -h|--help)    sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[ -n "$FAMILY" ]    || { echo "Missing --family (one of: ${!FRAMES[*]})" >&2; exit 1; }
[ -n "${FRAMES[$FAMILY]:-}" ] || { echo "Unknown family '$FAMILY' (want: ${!FRAMES[*]})" >&2; exit 1; }
[ -n "$DATA_ROOT" ] || { echo "Missing --data-root (dir holding $ORG/datagen-<fam>-v1-joint-5cam)" >&2; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || { echo "ERROR: WANDB_API_KEY unset (config sets use_wandb=true)." >&2; exit 1; }

SRC="$DATA_ROOT/$ORG/datagen-$FAMILY-v1-joint-5cam"
[ -f "$SRC/meta/info.json" ] || { echo "ERROR: dataset not found at $SRC" >&2; exit 1; }

# The pretrain checkpoint + its distillation teachers + the Qwen3-VL tokenizer must be local.
for P in "$PRETRAIN_DIR/lingbot-vla-v2-6b/model.safetensors.index.json" \
         "$PRETRAIN_DIR/lingbot-vla-v2-6b/depth/model.pt" \
         "$PRETRAIN_DIR/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
         "$PRETRAIN_DIR/Qwen3-VL-4B-Instruct/config.json" \
         "$PRETRAIN_DIR/moge-2-vitb-normal/model.pt"; do
  [ -e "$P" ] || { echo "ERROR: missing $P -- run tools/lingbot_sft/download_weights.sh first." >&2; exit 1; }
done

NSTEPS="${STEPS_OVERRIDE:-${STEPS[$FAMILY]}}"
SAVE_STEPS=$(( (NSTEPS + 3) / 4 ))            # 4-rung ladder, as for the other base models
OUT="${OUT_OVERRIDE:-$RUN_ROOT/runs/$FAMILY}"
NORM_JSON="assets/norm_stats/maniguard_${FAMILY}.json"
mkdir -p "$(dirname "$OUT")" assets/norm_stats

echo "[run_sft] family=$FAMILY frames=${FRAMES[$FAMILY]} gpus=$GPUS"
echo "[run_sft] steps=$NSTEPS (2 epochs @ global batch $((32*GPUS))) save_steps=$SAVE_STEPS"
echo "[run_sft] data=$SRC"
echo "[run_sft] out=$OUT  norm_stats=$NORM_JSON"

# --- norm stats: computed once per family, then reused (recompute with --norm-stats) ---
if [ "$FORCE_NORM" = "1" ] || [ ! -f "$NORM_JSON" ]; then
  echo "[run_sft] computing norm stats -> $NORM_JSON"
  CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py "$NORM_CONFIG" \
    --data.data_name maniguard \
    --data.train_path "$SRC" \
    --data.robot_config_root "$ROBOT_CONFIG_ROOT" \
    --data.norm_path "$NORM_JSON" \
    --data.data_ratio_for_norm_compute 1
  [ -f "$NORM_JSON" ] || { echo "ERROR: norm stats not produced at $NORM_JSON" >&2; exit 1; }
else
  echo "[run_sft] reusing existing norm stats: $NORM_JSON"
fi

# --- train: upstream train.sh derives nproc from CUDA_VISIBLE_DEVICES / nvidia-smi ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS-1)))}"
# Supply a default WandB project when the environment does not specify one.

export WANDB_PROJECT="${WANDB_PROJECT:-maniguard-lingbot-sft-yanZ}"
bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG" \
  --data.data_name maniguard \
  --data.train_path "$SRC" \
  --data.robot_config_root "$ROBOT_CONFIG_ROOT" \
  --data.norm_stats_file "$NORM_JSON" \
  --train.output_dir "$OUT" \
  --train.max_steps "$NSTEPS" \
  --train.save_steps "$SAVE_STEPS" \
  --train.wandb_name "lingbot-vla2_datagen_v1_${FAMILY}_joint_2cam" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "[run_sft] $FAMILY done -> $OUT"
