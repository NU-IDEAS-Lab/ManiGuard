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
# The watcher (hf_push_watcher.py) runs in parallel and uploads each checkpoint
# to HF the moment it finalizes -- training never blocks on uploads. After the
# run, hf_push.py can be invoked manually to backfill anything missed (it skips
# whatever is already complete on HF; same de-dup as the watcher).
#
# Usage:
#   tools/openpi_sft/run_sft.sh --config <name> --exp <exp> [options]
#
# Required:
#   --config NAME      openpi config name (registered by maniguard.openpi_sft)
#   --exp NAME         experiment name (checkpoint subdir)
#
# Options:
#   --steps N          num_train_steps override
#   --batch N          batch_size override
#   --keep-period N    keep_period override (checkpoint cadence)
#   --norm-stats       run compute_norm_stats before training (sim configs need this)
#   --no-smoke         skip the 100-step smoke test
#   --smoke-only       run only the smoke test, then exit
#   --resume           pass --resume to openpi (continue from last ckpt)
#   --overwrite        pass --overwrite to openpi (wipe ckpt dir)
#   --push-repo REPO   HF model repo to stream checkpoints to (enables the watcher)
#   --push-private     create the push repo private (default public)
#   --poll-interval N  watcher scan interval seconds (default 30)
#
# Env: OPENPI_ROOT (default: sibling ../openpi next to the ManiGuard repo),
#      HF_TOKEN (required if --push-repo), WANDB_* as usual.
set -euo pipefail

CONFIG=""; EXP=""; STEPS=""; BATCH=""; KEEP_PERIOD=""
NORM_STATS=0; SMOKE=1; SMOKE_ONLY=0; RESUME=0; OVERWRITE=0
PUSH_REPO=""; PUSH_PRIVATE=0; POLL_INTERVAL=30

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
    --push-private) PUSH_PRIVATE=1; shift;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# All of OPENPI_ROOT / HF_TOKEN / WANDB_API_KEY are read straight from the
# environment, so exporting them once in your shell rc (~/.bashrc / ~/.zshrc) is
# enough -- no need to prefix them on the command line. OPENPI_ROOT falls back to
# a sibling ../openpi if unset.
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/openpi}"
export OPENPI_ROOT

if [[ -z "$CONFIG" || -z "$EXP" ]]; then
  echo "ERROR: --config and --exp are required" >&2
  exit 1
fi

if [[ ! -d "$OPENPI_ROOT" ]]; then
  echo "ERROR: OPENPI_ROOT does not exist: $OPENPI_ROOT" >&2
  echo "       set it in your shell rc (export OPENPI_ROOT=/path/to/openpi)." >&2
  exit 1
fi

# Preflight on env-provided secrets. These are read straight from the
# environment, so exporting them once in your shell rc is enough -- the shell
# has already sourced the rc, so $HF_TOKEN / $WANDB_API_KEY are populated without
# being passed on the command line.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is unset." >&2
  echo "       export WANDB_API_KEY in your shell rc (~/.bashrc / ~/.zshrc)." >&2
  exit 1
fi
if [[ -n "$PUSH_REPO" && -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: --push-repo given but HF_TOKEN is unset." >&2
  echo "       export HF_TOKEN in your shell rc, or drop --push-repo." >&2
  exit 1
fi

# openpi's compute_norm_stats.py takes the config name positionally.
CMD_NORM=( python "$HERE/compute_norm_stats.py" "$CONFIG" )
TRAIN_ARGS=( "$CONFIG" --exp-name "$EXP" )
[[ -n "$STEPS" ]] && TRAIN_ARGS+=( --num-train-steps "$STEPS" )
[[ -n "$BATCH" ]] && TRAIN_ARGS+=( --batch-size "$BATCH" )
[[ -n "$KEEP_PERIOD" ]] && TRAIN_ARGS+=( --keep-period "$KEEP_PERIOD" )
[[ "$RESUME" == "1" ]] && TRAIN_ARGS+=( --resume )
[[ "$OVERWRITE" == "1" ]] && TRAIN_ARGS+=( --overwrite )

if [[ "$NORM_STATS" == "1" ]]; then
  echo "[run_sft] computing norm stats for $CONFIG ..."
  "${CMD_NORM[@]}"
fi

if [[ "$SMOKE" == "1" || "$SMOKE_ONLY" == "1" ]]; then
  echo "[run_sft] smoke test (100 steps) ..."
  SMOKE_EXP="${EXP}_smoke"
  python "$HERE/train.py" "$CONFIG" --exp-name "$SMOKE_EXP" \
    --num-train-steps 100 --overwrite
  echo "[run_sft] smoke OK; removing smoke ckpt dir"
  rm -rf "$OPENPI_ROOT/checkpoints/$CONFIG/$SMOKE_EXP" || true
  [[ "$SMOKE_ONLY" == "1" ]] && { echo "[run_sft] smoke-only done"; exit 0; }
fi

# --- background HF-push watcher (optional) -----------------------------------
WATCHER_PID=""
cleanup() { [[ -n "$WATCHER_PID" ]] && kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -n "$PUSH_REPO" ]]; then
  CKPT_DIR="$OPENPI_ROOT/checkpoints/$CONFIG/$EXP"
  # num_train_steps the watcher should expect: CLI override or the config default.
  WSTEPS="$STEPS"
  if [[ -z "$WSTEPS" ]]; then
    WSTEPS="$(python "$HERE/_config_steps.py" "$CONFIG")"
  fi
  WATCH_ARGS=( --ckpt-dir "$CKPT_DIR" --repo "$PUSH_REPO"
               --num-train-steps "$WSTEPS" --poll-interval "$POLL_INTERVAL" )
  [[ "$PUSH_PRIVATE" == "1" ]] && WATCH_ARGS+=( --private )
  echo "[run_sft] starting HF-push watcher -> $PUSH_REPO (num_train_steps=$WSTEPS)"
  python "$HERE/hf_push_watcher.py" "${WATCH_ARGS[@]}" &
  WATCHER_PID=$!
fi

echo "[run_sft] full training: $CONFIG / $EXP"
python "$HERE/train.py" "${TRAIN_ARGS[@]}"
echo "[run_sft] training done."

if [[ -n "$WATCHER_PID" ]]; then
  echo "[run_sft] waiting for watcher to finish uploading final checkpoint ..."
  wait "$WATCHER_PID" || true
  WATCHER_PID=""
  echo "[run_sft] watcher finished. Run hf_push.py to verify/backfill if needed."
fi
echo "[run_sft] done."
