#!/usr/bin/env bash
# Fetch the three weight sets LingBot-VLA 2.0 post-training needs, into assets/pretrained/.
#
#   lingbot-vla-v2-6b        28 GB  the PRETRAIN checkpoint + both distillation teachers
#                                   (depth/model.pt, dino_video/teacher_step_10000.pth)
#   Qwen3-VL-4B-Instruct     ~9 GB  tokenizer / base VLM the config points at
#   moge-2-vitb-normal      419 MB  MoGe teacher used by the depth alignment loss
#
# ⚠️ We warm-start from `robbyant/lingbot-vla-v2-6b` (the PRETRAIN release), NOT from
# `robbyant/lingbot-vla-v2-6b-robotwin`, which is that model already post-trained 50k steps
# on RoboTwin — starting there would confound the benchmark with another task suite.
#
# Usage:  export HF_TOKEN=...;  bash tools/lingbot_sft/download_weights.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DEST="$REPO_ROOT/assets/pretrained"
mkdir -p "$DEST"

# hf_transfer makes the multi-GB safetensors download far faster where it is installed.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

fetch() {  # <repo_id> <local_subdir>
  echo "==> $1 -> $DEST/$2"
  hf download "$1" --local-dir "$DEST/$2" --max-workers 8
}

fetch robbyant/lingbot-vla-v2-6b    lingbot-vla-v2-6b
fetch Qwen/Qwen3-VL-4B-Instruct     Qwen3-VL-4B-Instruct
fetch Ruicheng/moge-2-vitb-normal   moge-2-vitb-normal

echo "==> verifying"
ok=1
for P in "lingbot-vla-v2-6b/model.safetensors.index.json" \
         "lingbot-vla-v2-6b/depth/model.pt" \
         "lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
         "lingbot-vla-v2-6b/dino_video/config.yaml" \
         "Qwen3-VL-4B-Instruct/config.json" \
         "moge-2-vitb-normal/model.pt"; do
  if [ -e "$DEST/$P" ]; then echo "  OK   $P"; else echo "  MISS $P"; ok=0; fi
done
[ "$ok" = "1" ] && echo "all weights present ($(du -sh "$DEST" | cut -f1))" || { echo "INCOMPLETE" >&2; exit 1; }
