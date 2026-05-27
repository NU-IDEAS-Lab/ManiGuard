#!/bin/bash
# Auto-restart wrapper for maniguard.data.teleop.gello_grasp_batch.
#
# Why: OG's articulation_view nulls out after ~7-15 add/remove cycles
# (same bug render_grasps_loop.sh handles). gello_grasp_batch detects it
# and exits 2; we relaunch. Resume logic skips objects whose .pt already
# exists, so saved grasps are not lost.
#
# Usage:
#   bash scripts/gello_grasp_batch_loop.sh
#   LIMIT=100 OUTPUT_DIR=outputs/grasp_datasets/teleop/tensors \
#       bash scripts/gello_grasp_batch_loop.sh

set -u
cd "$(dirname "$0")/.."

: "${LIMIT:=50}"
: "${OUTPUT_DIR:=outputs/grasp_datasets/teleop/tensors}"
: "${LOG:=logs/gello_grasp_batch.log}"
: "${EXTRA_ARGS:=}"

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "" | tee -a "$LOG"
    echo "[$(date '+%F %T')] watchdog attempt #$attempt" | tee -a "$LOG"

    set +e
    CUDA_VISIBLE_DEVICES=0 \
        conda run -n behavior --no-capture-output \
            python -u -m maniguard.data.teleop.gello_grasp_batch \
                --limit "$LIMIT" --output-dir "$OUTPUT_DIR" \
                $EXTRA_ARGS 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e

    if [ "$rc" = "2" ] || [ "$rc" = "139" ] || [ "$rc" = "134" ]; then
        # rc=2 → our FATAL articulation_view detector
        # rc=139 → SIGSEGV (PhysX/USD shutdown crash after sys.exit, common
        #          when articulation is corrupted)
        # rc=134 → SIGABRT (assertion in OG / Isaac Sim during cleanup)
        echo "[$(date '+%F %T')] watchdog: exit $rc (recoverable crash), relaunching" \
            | tee -a "$LOG"
        sleep 3
        continue
    fi

    # Resume said no pending → empty CSV section → done. Operator pressed Q
    # or any other clean exit (rc 0) also stops the loop.
    if grep -q "Nothing to teleop." <(tail -20 "$LOG" 2>/dev/null); then
        echo "[$(date '+%F %T')] watchdog: nothing pending, done." | tee -a "$LOG"
        break
    fi

    echo "[$(date '+%F %T')] watchdog: clean exit (rc=$rc), stopping." | tee -a "$LOG"
    break
done
