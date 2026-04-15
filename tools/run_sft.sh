#!/usr/bin/env bash
# Pi0.5 SFT launcher for a Sentinel task family.
#
# Imports sentinel_ext before invoking RLinf's SFT entry point so our
# Sentinel-specific TrainConfigs are in RLinf's registry. RLinf itself
# is not patched.
#
# Prep (once per task family):
#   1. Record teleop demos (so101_franka_teleop.py).
#   2. tools/prepare_sft_data.sh -> LeRobot dataset + norm_stats.json.
#
# Launch:
#   export SENTINEL_PI05_BASE=/path/to/pi05_base_or_sft_ckpt
#   export SENTINEL_LEROBOT_ROOT=$PWD/outputs/lerobot_datasets
#   export SENTINEL_LEROBOT_REPO_ID=sentinel/clutter_pickup_v1
#   export SENTINEL_ASSET_ID=sentinel_clutter_pickup_v1        # optional
#   export SENTINEL_LOG_ROOT=$PWD/outputs/rlinf_logs           # optional
#   bash tools/run_sft.sh sentinel_clutter_sft_openpi
#
# Pass hydra overrides after the config name, e.g.
#   bash tools/run_sft.sh sentinel_goblet_sft_openpi runner.max_steps=1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${1:-sentinel_clutter_sft_openpi}"
shift || true

: "${SENTINEL_PI05_BASE:?Set SENTINEL_PI05_BASE to the Pi0.5 checkpoint root}"
: "${SENTINEL_LEROBOT_ROOT:?Set SENTINEL_LEROBOT_ROOT to the HF_LEROBOT_HOME dir}"
: "${SENTINEL_LEROBOT_REPO_ID:?Set SENTINEL_LEROBOT_REPO_ID, e.g. sentinel/clutter_pickup_v1}"

export SENTINEL_ASSET_ID="${SENTINEL_ASSET_ID:-$(basename "$SENTINEL_LEROBOT_REPO_ID")}"
export SENTINEL_LOG_ROOT="${SENTINEL_LOG_ROOT:-$REPO_ROOT/outputs/rlinf_logs}"

# Hydra's yaml uses ${oc.env:EMBODIED_PATH}/config/ in its searchpath to
# resolve the model/env/training_backend defaults vendored with RLinf.
export EMBODIED_PATH="$REPO_ROOT/RLinf/examples/sft"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/RLinf:${PYTHONPATH:-}"

VENV_PY="$REPO_ROOT/RLinf/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "RLinf .venv python not found at $VENV_PY" >&2
    exit 1
fi
source "$REPO_ROOT/RLinf/.venv/bin/activate"

LOG_DIR="$SENTINEL_LOG_ROOT/sft_$(date +'%Y%m%d-%H%M%S')"
mkdir -p "$LOG_DIR"

# --config-path points at our configs dir; --config-name is the yaml
# filename (without .yaml). sentinel_ext registers OpenPI configs on
# import, which runpy invokes before hydra parses the CLI.
exec sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" \
    python -m sentinel_ext.launchers sft \
    --config-path "$REPO_ROOT/configs/sft" \
    --config-name "$CONFIG_NAME" \
    "runner.logger.log_path=$LOG_DIR" \
    "$@"
