#!/usr/bin/env bash
# End-to-end SmolVLA SFT over the 6 ManiGuard-Bench families on the finalized
# datagen v1 datasets. Run one family (--family <fam>) or all serially (--all /
# no arg: clutter -> cabinet -> stack -> jar -> lid -> dusty).
#
# Per family, end to end (each step is idempotent / resumable):
#   1. ensure the model-agnostic 5-cam LeRobot dataset is present in the SHARED root
#      (download once; reused across openpi / gr00t / smolvla);
#   2. build a SmolVLA-ready 2-cam, standard-keyed copy (prepare_dataset.py);
#   3. train ~2 epochs (steps = ceil(frames * 2 / batch)) via lerobot-train;
#   4. push the checkpoint + a model card to HF.
#
# Identity (dataset, frames, overview view) is fixed in the tables below; compute
# knobs are flags. Run-name convention: smolvla-base_datagen_v1_<fam>_joint_2cam
# (no _lora suffix — SmolVLA freezes the VLM + trains the expert, no LoRA, like GR00T).
#
# ⚠️ LeRobot version pin: clone huggingface/lerobot at a PINNED release tag and
# `pip install -e <clone>[smolvla]` in this env. The `lerobot-train` flag surface has
# changed across releases — verify run_sft.sh's flags against `lerobot-train --help`
# for that tag before a real run (see docs/sft/smolvla.md).
#
# Prereqs (once per shell, in the lerobot env):
#   pip install -e <lerobot_clone>[smolvla]      # provides `lerobot-train` + LeRobotDataset
#   pip install -e <maniguard>                    # prepare_dataset.py imports maniguard.*
#   export HF_TOKEN=...  WANDB_API_KEY=...         # both required (checked below)
#
# Usage:
#   bash tools/smolvla_sft/run_all.sh --all
#   bash tools/smolvla_sft/run_all.sh --family dusty
#   BATCH=64 WORKERS=16 bash tools/smolvla_sft/run_all.sh --all
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
RUN_ROOT="$MANIGUARD_HOME/outputs/smolvla_sft"

# family -> total frame count (the only per-dataset value; steps derive from it).
declare -A FRAMES=(
  [clutter]=901520  [cabinet]=4172962  [stack]=2652083
  [jar]=946870      [lid]=1055142      [dusty]=1879498
)
# family -> overview view fed to observation.images.top. MUST match the openpi /
# gr00t datagen-v1 SFT (both use image_left) for benchmark parity — change a single
# entry here only if that family's config also changes.
declare -A EXTERNAL_CAM=(
  [clutter]=left  [cabinet]=left  [stack]=left
  [jar]=left      [lid]=left      [dusty]=left
)
ORDER=(clutter cabinet stack jar lid dusty)

# --- pre-flight (fail fast) ---
[[ -n "${HF_TOKEN:-}" ]]      || { echo "ERROR: HF_TOKEN unset (dataset pull + base model + checkpoint push)." >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY unset (online training logs)." >&2; exit 1; }
command -v lerobot-train >/dev/null 2>&1 || { echo "ERROR: lerobot-train not on PATH — pip install the pinned lerobot clone." >&2; exit 1; }

run_family() {
  local fam="$1"
  local frames="${FRAMES[$fam]:-}"
  [[ -n "$frames" ]] || { echo "ERROR: unknown family '$fam' (want: ${ORDER[*]})." >&2; exit 1; }
  local cam="${EXTERNAL_CAM[$fam]:-left}"

  local data_repo="$ORG/datagen-$fam-v1-joint-5cam"
  local prep_repo="$ORG/datagen-$fam-v1-joint-2cam"       # id stamped into the prepared copy
  local model_repo="$ORG/smolvla-base-datagen-v1-$fam-joint-2cam"
  local exp_name="smolvla-base_datagen_v1_${fam}_joint_2cam"
  local src="$SFT_DATA_ROOT/$data_repo"
  local prepped="$RUN_ROOT/data/${fam}_smolvla"
  local out="$RUN_ROOT/runs/$fam"
  local steps=$(( (frames * EPOCHS + BATCH - 1) / BATCH ))
  local save_freq=$(( steps / 5 > 0 ? steps / 5 : 1 ))

  echo "==================== $fam ===================="
  echo "[run_all] data=$data_repo frames=$frames epochs=$EPOCHS batch=$BATCH -> steps=$steps (cam=$cam)"

  # 1. shared, idempotent download (reused across model tracks)
  if [[ -f "$src/meta/info.json" ]]; then
    echo "[run_all] dataset present, skip download: $src"
  else
    echo "[run_all] downloading $data_repo -> $src"
    hf download "$data_repo" --repo-type dataset --revision v2.1 --local-dir "$src"
  fi

  # 2. SmolVLA-ready 2-cam standard-keyed copy (skips if already complete)
  python tools/smolvla_sft/prepare_dataset.py \
    --src "$src" --out "$prepped" --repo-id "$prep_repo" --external-cam "$cam"

  # 3. train ~EPOCHS epochs (online wandb)
  bash tools/smolvla_sft/run_sft.sh \
    --dataset "$prepped" --repo-id "$prep_repo" --output "$out" \
    --steps "$steps" --batch "$BATCH" --workers "$WORKERS" \
    --save-freq "$save_freq" --exp-name "$exp_name"

  # 4. push checkpoint + card. LeRobot writes <out>/checkpoints/<step>/pretrained_model
  #    with a `last` symlink to the newest; prefer that, else the newest numbered one.
  local ckpt="$out/checkpoints/last/pretrained_model"
  if [[ ! -d "$ckpt" ]]; then
    ckpt="$(ls -dt "$out"/checkpoints/*/pretrained_model 2>/dev/null | head -1 || true)"
  fi
  [[ -n "$ckpt" && -d "$ckpt" ]] || { echo "ERROR: no pretrained_model checkpoint under $out/checkpoints." >&2; exit 1; }
  local title="${fam^}"
  python tools/smolvla_sft/push_to_hf.py \
    --ckpt "$ckpt" --repo "$model_repo" --title "$title" --task "$fam" \
    --data-repo "$data_repo" --frames "$frames" --epochs "$EPOCHS" \
    --steps "$steps" --batch "$BATCH" --external-cam "$cam"

  echo "[run_all] $fam DONE -> $model_repo"
}

# --- dispatch ---
case "${1:-}" in
  --family) [[ -n "${2:-}" ]] || { echo "Usage: $0 --family <fam>" >&2; exit 1; }; run_family "$2" ;;
  --all|"") for fam in "${ORDER[@]}"; do run_family "$fam"; done ;;
  -h|--help) sed -n '1,34p' "$0"; exit 0 ;;
  *) echo "Usage: $0 [--family <fam> | --all]" >&2; exit 1 ;;
esac
