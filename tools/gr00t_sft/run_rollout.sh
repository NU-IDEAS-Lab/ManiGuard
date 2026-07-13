#!/usr/bin/env bash
# Drive the full GR00T-N1.6 SFT rollout across the ManiGuard families:
#   for each family -> train (run_sft.sh) -> push checkpoint + model card to HF.
# A per-family failure is logged and skipped so the rest still run (continuous +
# stable). Long-running: hours, dominated by clutter's ~53k-step / 2-epoch run.
#
# Prereqs: run inside the Isaac-GR00T venv with the FFmpeg module loaded (so
# torchcodec can decode H.264) and HF_TOKEN/WANDB_API_KEY set. In a tmux pane:
#     module load ffmpeg/6.1-gcc-11.2.0
#     bash tools/gr00t_sft/run_rollout.sh 2>&1 | tee outputs/gr00t_runs/rollout.log
set -uo pipefail

MANIGUARD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GR00T_HOME="${GR00T_HOME:-$HOME/projects/Isaac-GR00T}"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "$GR00T_HOME/.venv/bin/activate" ] && source "$GR00T_HOME/.venv/bin/activate"

GR00T_DS="$MANIGUARD_HOME/outputs/gr00t_datasets"
RUNS="$MANIGUARD_HOME/outputs/gr00t_runs"
RUN_SFT="$MANIGUARD_HOME/tools/gr00t_sft/run_sft.sh"
PUSH="$MANIGUARD_HOME/tools/gr00t_sft/push_to_hf.py"
ORG="IDEAS-Lab-Northwestern"
BATCH="${BATCH:-64}"

# name | ds_dir | task | title | data_repo(name) | frames | epochs | delete_repo_first
FAMILIES=(
  "dusty|sim-dusty-transfer-30-joint-3cam|dusty-transfer|Dusty-Transfer|sim-dusty-transfer-30-joint-3cam|20265|10|0"
  "lid|sim-lid-transport-food-30-joint-3cam|lid-transport-food|Lid-Transport-Food|sim-lid-transport-food-30-joint-3cam|12312|10|0"
  "cab|sim-cabinet-pickup-30-joint-3cam|cabinet-pickup|Cabinet-Pickup|sim-cabinet-pickup-30-joint-3cam|22684|10|0"
  "jar|sim-jar-transport-30-joint-3cam|jar-transport|Jar-Transport|sim-jar-transport-30-joint-3cam|12967|10|0"
  "stack|sim-stack-retrieve-60-joint-3cam|stack-retrieve|Stack-Retrieve|sim-stack-retrieve-60-joint-3cam|48208|10|1"
  "clutter|sentinel-pnp-clutter-joint|clutter|Clutter|sentinel-pnp-clutter-joint|1699175|2|0"
)

for entry in "${FAMILIES[@]}"; do
  IFS='|' read -r name dsdir task title datarepo frames epochs delfirst <<<"$entry"
  steps=$(( (frames * epochs + BATCH - 1) / BATCH ))   # ceil(frames*epochs/batch)
  ds="$GR00T_DS/$dsdir"
  out="$RUNS/$name"
  repo="$ORG/gr00t-n16-base-$task-joint-3cam"
  echo "################ [$(date '+%F %H:%M')] $name: epochs=$epochs frames=$frames steps=$steps repo=$repo ################"

  if [ ! -f "$ds/meta/modality.json" ]; then
    echo "[rollout] SKIP $name: dataset not prepared ($ds/meta/modality.json missing)"
    continue
  fi

  if [ "$delfirst" = "1" ]; then
    echo "[rollout] deleting existing HF repo before re-training: $repo"
    python -c "from huggingface_hub import HfApi; HfApi().delete_repo('$repo', repo_type='model', missing_ok=True)" \
      || echo "[rollout] repo delete failed/absent (continuing)"
  fi

  # final checkpoint only for short runs; a few intermediates on the long clutter run.
  if [ "$steps" -gt 20000 ]; then save_steps=10000; else save_steps="$steps"; fi

  if ! bash "$RUN_SFT" --dataset "$ds" --output "$out" --steps "$steps" --batch "$BATCH" \
        --save-steps "$save_steps" --exp-name "$name"; then
    echo "[rollout] TRAIN FAILED: $name (continuing to next family)"
    continue
  fi

  ck=$(ls -dt "$out"/*/checkpoint-* 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    echo "[rollout] NO CHECKPOINT found for $name under $out (continuing)"
    continue
  fi

  echo "[rollout] pushing $ck -> $repo"
  python "$PUSH" --ckpt "$ck" --repo "$repo" --title "$title" --task "$task" \
      --data-repo "$ORG/$datarepo" --frames "$frames" --epochs "$epochs" --steps "$steps" --batch "$BATCH" \
      || echo "[rollout] PUSH FAILED: $name (continuing)"
  echo "[rollout] DONE $name"
done
echo "ROLLOUT_ALL_DONE_$(date +%s)"
