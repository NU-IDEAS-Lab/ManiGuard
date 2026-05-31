#!/usr/bin/env bash
# Reusable pi0.5 LoRA SFT runner for the ManiGuard openpi_sft configs.
#
# Pipeline:
#   [norm-stats] -> 100-step smoke (auto-deleted) -> background HF-push watcher
#   -> full training -> wait for watcher to finish uploading the final ckpt.
#
# openpi stays pristine; configs come from maniguard.openpi_sft via the tools/
# launchers (which register them into openpi's _CONFIGS_DICT). Run from the
# openpi venv (e.g. `uv run`).
#
# HANDOFF: only --config is strictly required. The experiment/run name, HF push
# repo and its visibility all default from the config's policy_metadata
# (default_exp / hf_repo / hf_private), so a fresh box / another person / an
# agent reading the doc can launch with just --config. CLI flags override.
#
# All run artifacts go under ManiGuard/outputs/sft_runs/<exp>/ (gitignored):
#   checkpoints/  assets/(norm stats)  logs/   -- one self-contained folder per
#   run, identical regardless of where you launch from. The pi05_base warm-start
#   download is cached in ManiGuard/outputs/openpi_cache (shared across runs).
#
# The watcher (hf_push_watcher.py) runs in parallel and uploads each checkpoint
# to HF the moment it finalizes -- training never blocks on uploads. After the
# run, hf_push.py can be invoked manually to backfill anything missed (it skips
# whatever is already complete on HF; same de-dup as the watcher).
#
# Usage:
#   tools/openpi_sft/run_sft.sh --config <name> [options]
#
# Required:
#   --config NAME      openpi config name (registered by maniguard.openpi_sft)
#
# Options (all optional -- sensible defaults from the config):
#   --exp NAME         experiment/run name (default: config policy_metadata.default_exp)
#   --steps N          num_train_steps override
#   --batch N          batch_size override
#   --keep-period N    keep_period override (checkpoint cadence)
#   --norm-stats       run compute_norm_stats before training (sim configs need this)
#   --no-smoke         skip the 100-step smoke test
#   --smoke-only       run only the smoke test, then exit
#   --resume           pass --resume to openpi (continue from last ckpt)
#   --overwrite        pass --overwrite to openpi (wipe ckpt dir)
#   --push-repo REPO   HF model repo to stream checkpoints to
#                      (default: config policy_metadata.hf_repo; empty disables push)
#   --no-push          disable the HF push watcher even if the config sets hf_repo
#   --push-private     force the push repo private (default: config hf_private)
#   --poll-interval N  watcher scan interval seconds (default 30)
#
# Env (export once in your shell rc; no need to prefix on the command line):
#   OPENPI_ROOT       openpi clone (default: sibling ../openpi)
#   HF_TOKEN          required if pushing to HF
#   WANDB_API_KEY     required (training logs)
set -euo pipefail

CONFIG=""; EXP=""; STEPS=""; BATCH=""; KEEP_PERIOD=""
NORM_STATS=0; SMOKE=1; SMOKE_ONLY=0; RESUME=0; OVERWRITE=0
PUSH_REPO=""; NO_PUSH=0; PUSH_PRIVATE=""; POLL_INTERVAL=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --exp) EXP="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --keep-period) KEEP_PERIOD="$2"; shift 2;;
    --norm-stats) NORM_STATS=1; shift;;
    --no-smoke) SMOKE=0; shift;;
    --smoke-only) SMOKE_ONLY=1; shift;;
    --resume) RESUME=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --push-repo) PUSH_REPO="$2"; shift 2;;
    --no-push) NO_PUSH=1; shift;;
    --push-private) PUSH_PRIVATE=1; shift;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# OPENPI_ROOT / HF_TOKEN / WANDB_API_KEY are read straight from the environment
# (export them once in your shell rc). OPENPI_ROOT falls back to ../openpi.
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/openpi}"
export OPENPI_ROOT

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: --config is required" >&2
  exit 1
fi
if [[ ! -d "$OPENPI_ROOT" ]]; then
  echo "ERROR: OPENPI_ROOT does not exist: $OPENPI_ROOT" >&2
  echo "       set it in your shell rc (export OPENPI_ROOT=/path/to/openpi)." >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is unset." >&2
  echo "       export WANDB_API_KEY in your shell rc (~/.bashrc / ~/.zshrc)." >&2
  exit 1
fi

meta() { python "$HERE/_config_meta.py" "$CONFIG" "$1"; }

# --- resolve defaults from the config's policy_metadata --------------------
[[ -z "$EXP" ]] && EXP="$(meta default_exp)"
if [[ -z "$EXP" ]]; then
  echo "ERROR: no --exp and config has no policy_metadata.default_exp" >&2
  exit 1
fi
if [[ "$NO_PUSH" == "0" && -z "$PUSH_REPO" ]]; then
  PUSH_REPO="$(meta hf_repo)"
fi
[[ "$NO_PUSH" == "1" ]] && PUSH_REPO=""
# Visibility: --push-private forces private; otherwise follow config hf_private.
if [[ -z "$PUSH_PRIVATE" ]]; then
  [[ "$(meta hf_private)" == "true" ]] && PUSH_PRIVATE=1 || PUSH_PRIVATE=0
fi

if [[ -n "$PUSH_REPO" && -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: pushing to $PUSH_REPO but HF_TOKEN is unset." >&2
  echo "       export HF_TOKEN in your shell rc, or pass --no-push." >&2
  exit 1
fi

# --- run layout: one self-contained, gitignored folder per run -------------
RUN_DIR="$REPO_ROOT/outputs/sft_runs/$EXP"
CKPT_BASE="$RUN_DIR/checkpoints"
ASSETS_BASE="$RUN_DIR/assets"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
# pi05_base (and other GCS) warm-start downloads cached here, shared across runs.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/outputs/openpi_cache}"
mkdir -p "$OPENPI_DATA_HOME"

# openpi composes checkpoints as <checkpoint_base_dir>/<config_name>/<exp_name>/.
CKPT_DIR="$CKPT_BASE/$CONFIG/$EXP"

echo "[run_sft] config=$CONFIG exp=$EXP"
echo "[run_sft] run_dir=$RUN_DIR"
echo "[run_sft] openpi_root=$OPENPI_ROOT data_home=$OPENPI_DATA_HOME"
[[ -n "$PUSH_REPO" ]] && echo "[run_sft] push -> $PUSH_REPO (private=$PUSH_PRIVATE)" \
                      || echo "[run_sft] push disabled"

BASE_ARGS=( --assets-base-dir "$ASSETS_BASE" --checkpoint-base-dir "$CKPT_BASE" )
CMD_NORM=( python "$HERE/compute_norm_stats.py" "$CONFIG" )
TRAIN_ARGS=( "$CONFIG" --exp-name "$EXP" "${BASE_ARGS[@]}" )
[[ -n "$STEPS" ]] && TRAIN_ARGS+=( --num-train-steps "$STEPS" )
[[ -n "$BATCH" ]] && TRAIN_ARGS+=( --batch-size "$BATCH" )
[[ -n "$KEEP_PERIOD" ]] && TRAIN_ARGS+=( --keep-period "$KEEP_PERIOD" )
[[ "$RESUME" == "1" ]] && TRAIN_ARGS+=( --resume )
[[ "$OVERWRITE" == "1" ]] && TRAIN_ARGS+=( --overwrite )

if [[ "$NORM_STATS" == "1" ]]; then
  echo "[run_sft] computing norm stats for $CONFIG ..."
  "${CMD_NORM[@]}" 2>&1 | tee "$LOG_DIR/normstats.log"
fi

if [[ "$SMOKE" == "1" || "$SMOKE_ONLY" == "1" ]]; then
  echo "[run_sft] smoke test (100 steps) ..."
  SMOKE_EXP="${EXP}_smoke"
  python "$HERE/train.py" "$CONFIG" --exp-name "$SMOKE_EXP" "${BASE_ARGS[@]}" \
    --num-train-steps 100 --overwrite 2>&1 | tee "$LOG_DIR/smoke.log"
  echo "[run_sft] smoke OK; removing smoke ckpt dir"
  rm -rf "$CKPT_BASE/$CONFIG/$SMOKE_EXP" || true
  [[ "$SMOKE_ONLY" == "1" ]] && { echo "[run_sft] smoke-only done"; exit 0; }
fi

# --- background HF-push watcher --------------------------------------------
WATCHER_PID=""
cleanup() { [[ -n "$WATCHER_PID" ]] && kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -n "$PUSH_REPO" ]]; then
  WSTEPS="$STEPS"
  [[ -z "$WSTEPS" ]] && WSTEPS="$(meta num_train_steps)"
  WATCH_ARGS=( --ckpt-dir "$CKPT_DIR" --repo "$PUSH_REPO"
               --num-train-steps "$WSTEPS" --poll-interval "$POLL_INTERVAL" )
  [[ "$PUSH_PRIVATE" == "1" ]] && WATCH_ARGS+=( --private )
  echo "[run_sft] starting HF-push watcher -> $PUSH_REPO (num_train_steps=$WSTEPS)"
  python "$HERE/hf_push_watcher.py" "${WATCH_ARGS[@]}" 2>&1 | tee "$LOG_DIR/watcher.log" &
  WATCHER_PID=$!
fi

echo "[run_sft] full training: $CONFIG / $EXP"
python "$HERE/train.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
echo "[run_sft] training done."

if [[ -n "$WATCHER_PID" ]]; then
  echo "[run_sft] waiting for watcher to finish uploading final checkpoint ..."
  wait "$WATCHER_PID" || true
  WATCHER_PID=""
  echo "[run_sft] watcher finished."
fi
echo "[run_sft] done. ckpts: $CKPT_DIR"
[[ -n "$PUSH_REPO" ]] && cat <<EOF
[run_sft] to verify/backfill the HF push later:
  python $HERE/hf_push.py --ckpt-dir "$CKPT_DIR" \\
    --repo "$PUSH_REPO" --num-train-steps "${STEPS:-$(meta num_train_steps)}"$([[ "$PUSH_PRIVATE" == "1" ]] && echo " --private")
EOF
