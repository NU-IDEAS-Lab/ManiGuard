#!/usr/bin/env bash
# Post-train LingBot-VLA 2.0 on ONE ManiGuard datagen-v1 family, 8-GPU.
#
# Wraps upstream's own entrypoints (train.sh -> torchrun -> tasks/vla/train_lingbotvla.py)
# and only overrides the per-family knobs on the command line, exactly the way upstream's
# lingbotvla/data/vla_data/README.md documents. Upstream code is never edited; our layer is
# configs/robot_configs/maniguard.yaml + configs/vla/maniguard/maniguard.yaml + this script.
#
# Per family it: [computes norm stats if absent] -> trains 2 epochs -> leaves a 4-rung
# checkpoint ladder under <run root>/<family>/checkpoints/.
#
# Scale (identical across the ManiGuard base models): global batch 256 = micro 32 x 8 GPUs;
# steps = 2 epochs of that family = the same numbers the pi0.5 / pi0 tracks use.
#
# Prereqs (once per shell):
#   conda activate <lingbot env>          # built by tools/create_train_env.sh
#   export HF_TOKEN=...  WANDB_API_KEY=...
#   bash tools/lingbot_sft/download_weights.sh     # the 3 weight sets under assets/pretrained
#
# Usage:
#   bash tools/lingbot_sft/run_sft.sh --family clutter --data-root <shared_lerobot_root> \
#        [--gpus 8] [--norm-stats] [--steps N] [--out DIR] [-- <extra train overrides>...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

ORG="IDEAS-Lab-Northwestern"
CONFIG="configs/vla/maniguard/maniguard.yaml"

# upstream's entrypoints are `torchrun scripts/<x>.py`, which puts scripts/ (not the repo
# root) on sys.path, so `import lingbotvla` fails unless the package is installed. The env
# builder does not `pip install -e .`, so put the repo root on PYTHONPATH for the children.
export PYTHONPATH="$(cd "$HERE/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"

# LeRobot decodes the dataset's mp4s through torchcodec, which dlopen()s FFmpeg's shared
# libraries at runtime. A bare container has none, and the failure surfaces inside the
# dataloader workers (not at import), so it looks like a data bug. Point FFMPEG_LIB_DIR at a
# dir holding libav*/libsw* (e.g. a conda env's lib) and it is prepended here, before any
# worker forks. pyav is the fallback if this is ever unavailable -- it bundles its own FFmpeg.
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
# family -> 2-epoch step count at global batch 256, rounded up; identical to the pi0.5 / pi0
# tracks so the five base models see the same data for the same number of updates.
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
  CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py "$CONFIG" \
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
export WANDB_PROJECT="${WANDB_PROJECT:-maniguard-lingbot-sft}"
bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG" \
  --data.data_name maniguard \
  --data.train_path "$SRC" \
  --data.robot_config_root "$ROBOT_CONFIG_ROOT" \
  --data.norm_stats_file "$NORM_JSON" \
  --train.output_dir "$OUT" \
  --train.max_steps "$NSTEPS" \
  --train.save_steps "$SAVE_STEPS" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "[run_sft] $FAMILY done -> $OUT"
