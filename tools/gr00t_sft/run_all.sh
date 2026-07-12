#!/usr/bin/env bash
# End-to-end GR00T N1.6 SFT over the 6 ManiGuard-Bench families on the finalized
# datagen v1 datasets. Run one family (--family <fam>) or all serially (--all /
# no arg: clutter -> cabinet -> stack -> jar -> lid -> dusty).
#
# Per family, end to end (each step is idempotent / resumable):
#   1. ensure the model-agnostic LeRobot dataset is present in the SHARED root
#      (download once; reused across openpi / gr00t / smolvla);
#   2. build a GR00T-ready copy (H.264 + meta/modality.json + stats);
#   3. train ~2 epochs (steps = ceil(frames * 2 / batch));
#   4. push the checkpoint + a model card to HF.
#
# Identity (dataset, frames) is fixed in the table below; compute knobs are flags.
#
# Prereqs (once per shell, on the GR00T box):
#   source "$GR00T_HOME/.venv/bin/activate"   # GR00T_HOME defaults to ../Isaac-GR00T
#   module load ffmpeg/6.1-...                # torchcodec needs FFmpeg for H.264 decode
#   export HF_TOKEN=...  WANDB_API_KEY=...     # both required (checked below)
#
# Usage:
#   bash tools/gr00t_sft/run_all.sh --all
#   bash tools/gr00t_sft/run_all.sh --family dusty
#   BATCH=64 WORKERS=16 bash tools/gr00t_sft/run_all.sh --all
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIGUARD_HOME="$(cd "$HERE/../.." && pwd)"
cd "$MANIGUARD_HOME"

# --- identity (unique per dataset); compute knobs via env ---
ORG="IDEAS-Lab-Northwestern"
EPOCHS=2
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-16}"
SFT_DATA_ROOT="${MANIGUARD_SFT_DATA_ROOT:-$MANIGUARD_HOME/outputs/sft_datasets}"
RUN_ROOT="$MANIGUARD_HOME/outputs/gr00t_sft"

# family -> total frame count (the only per-dataset value; steps derive from it).
declare -A FRAMES=(
  [clutter]=901520  [cabinet]=4172962  [stack]=2652083
  [jar]=946870      [lid]=1055142      [dusty]=1879498
)
ORDER=(clutter cabinet stack jar lid dusty)

# --- pre-flight (fail fast) ---
[[ -n "${HF_TOKEN:-}" ]]      || { echo "ERROR: HF_TOKEN unset (dataset pull + checkpoint push)." >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY unset (online training logs)." >&2; exit 1; }

run_family() {
  local fam="$1"
  local frames="${FRAMES[$fam]:-}"
  [[ -n "$frames" ]] || { echo "ERROR: unknown family '$fam' (want: ${ORDER[*]})." >&2; exit 1; }

  local data_repo="$ORG/datagen-$fam-v1-joint-5cam"
  local model_repo="$ORG/gr00t-n16-base-datagen-v1-$fam-joint-2cam"
  local src="$SFT_DATA_ROOT/$data_repo"
  local prepped="$RUN_ROOT/data/${fam}_gr00t"
  local out="$RUN_ROOT/runs/$fam"
  local steps=$(( (frames * EPOCHS + BATCH - 1) / BATCH ))

  echo "==================== $fam ===================="
  echo "[run_all] data=$data_repo frames=$frames epochs=$EPOCHS batch=$BATCH -> steps=$steps"

  # 1. shared, idempotent download (reused across model tracks)
  if [[ -f "$src/meta/info.json" ]]; then
    echo "[run_all] dataset present, skip download: $src"
  else
    echo "[run_all] downloading $data_repo -> $src"
    hf download "$data_repo" --repo-type dataset --revision v2.1 --local-dir "$src"
  fi

  # 2. GR00T-ready copy (skips already-transcoded videos / existing stats internally)
  python tools/gr00t_sft/prepare_dataset.py --src "$src" --out "$prepped"

  # 3. train ~EPOCHS epochs (online wandb)
  bash tools/gr00t_sft/run_sft.sh \
    --dataset "$prepped" --output "$out" \
    --steps "$steps" --batch "$BATCH" --workers "$WORKERS" --exp-name "$fam"

  # 4. push checkpoint + card. --ckpt = the latest checkpoint dir the run wrote
  #    (HF-Trainer layout: <out>/checkpoint-<step>); fall back to <out>.
  local ckpt
  ckpt="$(ls -dt "$out"/checkpoint-* 2>/dev/null | head -1 || true)"
  [[ -n "$ckpt" ]] || ckpt="$out"
  local title="${fam^}"
  python tools/gr00t_sft/push_to_hf.py \
    --ckpt "$ckpt" --repo "$model_repo" --title "$title" --task "$fam" \
    --data-repo "$data_repo" --frames "$frames" --epochs "$EPOCHS" \
    --steps "$steps" --batch "$BATCH"

  echo "[run_all] $fam DONE -> $model_repo"
}

# --- dispatch ---
case "${1:-}" in
  --family) [[ -n "${2:-}" ]] || { echo "Usage: $0 --family <fam>" >&2; exit 1; }; run_family "$2" ;;
  --all|"") for fam in "${ORDER[@]}"; do run_family "$fam"; done ;;
  -h|--help) sed -n '1,24p' "$0"; exit 0 ;;
  *) echo "Usage: $0 [--family <fam> | --all]" >&2; exit 1 ;;
esac
