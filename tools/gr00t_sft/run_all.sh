#!/usr/bin/env bash
# End-to-end GR00T N1.6 SFT over the 6 ManiGuard-Bench families on the finalized
# datagen v1 datasets. Two phases (mirrors the openpi track: train to completion,
# push separately):
#
#   TRAIN (default):  ensure dataset -> build GR00T VIEW -> train ~2 epochs.
#                     Needs WANDB_API_KEY only. HF_TOKEN only matters if the base
#                     model / a dataset still has to be fetched from the Hub.
#   PUSH  (--push):   upload each family's latest checkpoint + card to HF. Needs HF_TOKEN.
#
# Usage:
#   bash tools/gr00t_sft/run_all.sh [--data-root <dir>] [--family <fam> | --all]
#   bash tools/gr00t_sft/run_all.sh --push [--family <fam> | --all]
#   # compute knobs via env: BATCH GPUS WORKERS LR TAG WANDB_PROJECT MANIGUARD_SFT_DATA_ROOT
#   TAG=-yanZ bash tools/gr00t_sft/run_all.sh --data-root /path/to/lerobot --all
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

# --- identity (unique per dataset); compute knobs via env (8-card config) ---
ORG="IDEAS-Lab-Northwestern"
EPOCHS=2
BATCH="${BATCH:-256}"      # GLOBAL batch (divided across GPUS)
GPUS="${GPUS:-8}"
WORKERS="${WORKERS:-6}"    # PER-GPU dataloader workers (torchrun: total = WORKERS*GPUS = 48)
LR="${LR:-2e-4}"           # peak LR (cosine); sqrt-scaled for global batch 256
TAG="${TAG:-}"             # empty by default; e.g. TAG=-yanZ -> HF repo + wandb project suffix
# wandb: project = track+TAG (distinguishes model track + runner); run = per family.
export WANDB_PROJECT="${WANDB_PROJECT:-maniguard-gr00tN1d6$TAG}"
RUN_ROOT="$REPO_ROOT/outputs/gr00t_sft"

# family -> total frame count (the only per-dataset value; steps derive from it).
declare -A FRAMES=(
  [clutter]=901520  [cabinet]=4172962  [stack]=2652083
  [jar]=946870      [lid]=1055142      [dusty]=1879498
)
ORDER=(clutter cabinet stack jar lid dusty)

# --- args: [--push] [--data-root <dir>] [--family <fam> | --all] ---
PUSH=0
DATA_ROOT_ARG=""
SELECT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --push)      PUSH=1; shift ;;
    --data-root) [[ -n "${2:-}" ]] || { echo "Usage: $0 --data-root <dir>" >&2; exit 1; }; DATA_ROOT_ARG="$2"; shift 2 ;;
    --family)    [[ -n "${2:-}" ]] || { echo "Usage: $0 --family <fam>" >&2; exit 1; }; SELECT="$2"; shift 2 ;;
    --all)       SELECT="all"; shift ;;
    -h|--help)   sed -n '1,21p' "$0"; exit 0 ;;
    *) echo "Usage: $0 [--push] [--data-root <dir>] [--family <fam> | --all]" >&2; exit 1 ;;
  esac
done
[[ -n "$SELECT" ]] || SELECT="all"   # default: all families

# dataset root precedence: --data-root  >  $MANIGUARD_SFT_DATA_ROOT  >  repo default
SFT_DATA_ROOT="${DATA_ROOT_ARG:-${MANIGUARD_SFT_DATA_ROOT:-$REPO_ROOT/outputs/sft_datasets}}"

# --- pre-flight (fail fast, per phase) ---
if [ "$PUSH" = 1 ]; then
  [[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN unset (checkpoint push)." >&2; exit 1; }
else
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY unset (online training logs)." >&2; exit 1; }
  [[ -n "${HF_TOKEN:-}" ]] || echo "[run_all] note: HF_TOKEN unset — OK if the base model + datasets are already local/cached." >&2
fi

train_family() {
  local fam="$1"
  local frames="${FRAMES[$fam]:-}"
  [[ -n "$frames" ]] || { echo "ERROR: unknown family '$fam' (want: ${ORDER[*]})." >&2; exit 1; }
  local data_repo="$ORG/datagen-$fam-v1-joint-5cam"
  local src="$SFT_DATA_ROOT/$data_repo"
  local prepped="$RUN_ROOT/data/${fam}_gr00t"
  local out="$RUN_ROOT/runs/$fam"
  local steps=$(( (frames * EPOCHS + BATCH - 1) / BATCH ))

  echo "==================== TRAIN $fam ===================="
  echo "[run_all] data=$data_repo frames=$frames epochs=$EPOCHS batch=$BATCH -> steps=$steps  workers=${WORKERS}/gpu  wandb=$WANDB_PROJECT"

  # 1. dataset present in the SHARED root? (reused across openpi / gr00t / smolvla)
  if [[ -f "$src/meta/info.json" ]]; then
    echo "[run_all] dataset present, skip download: $src"
  else
    echo "[run_all] downloading $data_repo -> $src"
    hf download "$data_repo" --repo-type dataset --revision v2.1 --local-dir "$src"
  fi

  # 2. GR00T-ready VIEW: symlink videos/parquet + real meta with baked stats (src untouched)
  python tools/gr00t_sft/prepare_dataset.py --src "$src" --out "$prepped" --stats-dir "gr00t_stats/$fam"

  # 3. train ~EPOCHS epochs (8-card, online wandb; run name = datagen_v1_<fam>_joint_2cam)
  bash tools/gr00t_sft/run_sft.sh \
    --dataset "$prepped" --output "$out" \
    --steps "$steps" --batch "$BATCH" --gpus "$GPUS" --lr "$LR" --workers "$WORKERS" \
    --exp-name "datagen_v1_${fam}_joint_2cam"
  echo "[run_all] $fam TRAIN DONE -> $out"
}

push_family() {
  local fam="$1"
  local frames="${FRAMES[$fam]:-}"
  [[ -n "$frames" ]] || { echo "ERROR: unknown family '$fam' (want: ${ORDER[*]})." >&2; exit 1; }
  local data_repo="$ORG/datagen-$fam-v1-joint-5cam"
  local model_repo="$ORG/gr00t-n16-datagen-v1-$fam-joint-2cam$TAG"   # no -base; optional TAG suffix
  local exp="datagen_v1_${fam}_joint_2cam"          # launch_finetune nests the output under this name
  local run_dir="$RUN_ROOT/runs/$fam/$exp"          # its ROOT = the FINAL saved model + experiment_cfg/ + processor/
  local steps=$(( (frames * EPOCHS + BATCH - 1) / BATCH ))
  # Push the run dir's ROOT only. push_to_hf.py ignores checkpoint-*/ + DeepSpeed state, so just
  # the final model's ~6.6GB inference bundle uploads (no intermediate ckpts, no optimizer/resume state).
  [[ -f "$run_dir/config.json" ]] || { echo "ERROR: no final model at $run_dir — did $fam finish training?" >&2; exit 1; }

  echo "==================== PUSH $fam ===================="
  echo "[run_all] $fam final-model=$run_dir -> $model_repo"
  python tools/gr00t_sft/push_to_hf.py \
    --ckpt "$run_dir" --repo "$model_repo" --title "${fam^}" --task "$fam" \
    --data-repo "$data_repo" --frames "$frames" --epochs "$EPOCHS" \
    --steps "$steps" --batch "$BATCH"
  echo "[run_all] $fam PUSH DONE -> $model_repo"
}

# --- dispatch ---
if [ "$SELECT" = "all" ]; then fams=("${ORDER[@]}"); else fams=("$SELECT"); fi
for fam in "${fams[@]}"; do
  if [ "$PUSH" = 1 ]; then push_family "$fam"; else train_family "$fam"; fi
done
