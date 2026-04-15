#!/usr/bin/env bash
# Pi0.5 PPO post-train launcher for a Sentinel task family.
#
# Imports sentinel before invoking RLinf's RL entry point so our
# Sentinel-specific TrainConfigs are in RLinf's registry.
#
# Prep:
#   1. A scene corpus -- either datasets/safety-benchmark/ (checked-in
#      seed scene) or a larger corpus you generated via a task pipeline.
#   2. An SFT checkpoint to warm-start from.
#
# Launch:
#   export SENTINEL_PI05_SFT_CKPT=/path/to/sft/ckpt
#   # Optional overrides (defaults: datasets/safety-benchmark +
#   #                     bddl3/bddl/activity_definitions)
#   export SENTINEL_BENCHMARK_ROOT=/path/to/custom_benchmark
#   export SENTINEL_ACTIVITY_ROOT=/path/to/custom_activity_defs
#   # Needed by OmniGibson at import time:
#   export OMNIGIBSON_DATA_PATH=/path/to/behavior-1k-assets_parent
#   export ISAAC_PATH=/path/to/isaac-sim
#   bash scripts/run_rl.sh sentinel_clutter_ppo_openpi_pi05

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${1:-sentinel_clutter_ppo_openpi_pi05}"
shift || true

: "${SENTINEL_PI05_SFT_CKPT:?Set SENTINEL_PI05_SFT_CKPT to the SFT checkpoint root}"
: "${OMNIGIBSON_DATA_PATH:?OmniGibson needs OMNIGIBSON_DATA_PATH}"
: "${ISAAC_PATH:?OmniGibson needs ISAAC_PATH}"

export SENTINEL_BENCHMARK_ROOT="${SENTINEL_BENCHMARK_ROOT:-$REPO_ROOT/datasets/safety-benchmark}"
export SENTINEL_ACTIVITY_ROOT="${SENTINEL_ACTIVITY_ROOT:-$REPO_ROOT/bddl3/bddl/activity_definitions}"
export SENTINEL_LOG_ROOT="${SENTINEL_LOG_ROOT:-$REPO_ROOT/outputs/rlinf_logs}"
export EMBODIED_PATH="$REPO_ROOT/RLinf/examples/embodiment"
export SENTINEL_CONFIG_ROOT="$REPO_ROOT/configs"
# sitecustomize.py in _autoimport/ makes every Python subprocess
# (including Ray workers) auto-import sentinel on startup, so our
# patches reach all processes instead of just the launcher.
export PYTHONPATH="$REPO_ROOT/sentinel/_autoimport:$REPO_ROOT:$REPO_ROOT/RLinf:${PYTHONPATH:-}"

# OmniGibson launch-time env from run_embodiment.sh.
export OMNIGIBSON_DATASET_PATH="${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}"
export OMNIGIBSON_KEY_PATH="${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}"
export OMNIGIBSON_ASSET_PATH="${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}"
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

VENV_PY="$REPO_ROOT/RLinf/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "RLinf .venv python not found at $VENV_PY" >&2
    exit 1
fi
source "$REPO_ROOT/RLinf/.venv/bin/activate"

LOG_DIR="$SENTINEL_LOG_ROOT/rl_$(date +'%Y%m%d-%H%M%S')"
mkdir -p "$LOG_DIR"

exec sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" \
    python -m sentinel.launchers rl \
    --config-path "$REPO_ROOT/configs/rl" \
    --config-name "$CONFIG_NAME" \
    "runner.logger.log_path=$LOG_DIR" \
    "$@"
