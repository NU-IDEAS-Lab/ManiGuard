#!/bin/bash
# Auto-restart wrapper for sentinel.rl.grasps.render_grasps.
#
# Why: long full-dataset runs accumulate memory in OG / cuRobo / PhysX
# until the Linux OOM killer steps in (~60-100 objects in our setup).
# The pipeline's resume logic (skip rows whose .pt / _pcd_top.png / .mp4
# already exists) makes restart safe; this script just keeps relaunching
# until every row is done.
#
# Strategy:
#   1. Preemptive restart every $RESTART_EVERY new completions.
#   2. SIGKILL + restart if no completions for $NO_GROWTH_TIMEOUT seconds.
#   3. Exit when render_grasps says "Nothing to render." (resume found
#      no pending rows) or pending count reaches 0.
#
# Usage:
#   OUTPUT_DIR=outputs/grasp_datasets/graspgen_full bash scripts/render_grasps_loop.sh

set -u
cd "$(dirname "$0")/.."

# Required by render_grasps.
: "${OUTPUT_DIR:=outputs/grasp_datasets/graspgen_full}"
: "${CSV_PATH:=sentinel/utils/franka_graspability_full.csv}"
: "${EXCLUDE_STATUSES:=too_large,degenerate_bbox,no_metadata,not_ready}"
: "${NUM_TARGET_GRASPS:=5}"
: "${PER_OBJECT_TIMEOUT:=150}"
: "${SAVE_VIDEO:=1}"      # set to 0 to skip Phase B / MP4
: "${RESTART_EVERY:=50}"  # completions between preemptive restarts
: "${NO_GROWTH_TIMEOUT:=300}"  # seconds of zero progress before SIGKILL
: "${POLL_INTERVAL:=30}"

DATASET_DIR="$OUTPUT_DIR/datasets"
LOG="$OUTPUT_DIR/run.log"
mkdir -p "$DATASET_DIR"

count_done() {
    # A row is "done" iff it has a .pt (success) or _pcd_top.png (failure).
    local pt fail
    pt=$(find "$DATASET_DIR" -maxdepth 1 -name "grasps_*.pt" 2>/dev/null | wc -l)
    fail=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_pcd_top.png" 2>/dev/null | wc -l)
    echo $((pt + fail))
}

kill_run() {
    # conda run -> python; signal both. -9 because hung sim.step ignores SIGTERM.
    pkill -9 -f "sentinel.rl.grasps.render_grasps" 2>/dev/null || true
    sleep 2
}

video_flag=""
if [ "$SAVE_VIDEO" = "1" ]; then
    video_flag="--save-video"
fi

while true; do
    done_count=$(count_done)
    target=$((done_count + RESTART_EVERY))
    echo "[$(date '+%F %T')] watchdog: launching (done=$done_count, restart_at=$target)" >> "$LOG"

    SENTINEL_SKIP_LONGFINGER=1 \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \
        conda run -n behavior --no-capture-output \
            python -u -m sentinel.rl.grasps.render_grasps \
                --csv "$CSV_PATH" \
                --exclude-statuses "$EXCLUDE_STATUSES" \
                --output-dir "$OUTPUT_DIR" \
                --save-grasp-dataset "$DATASET_DIR" \
                --num-target-grasps "$NUM_TARGET_GRASPS" \
                --per-object-timeout "$PER_OBJECT_TIMEOUT" \
                $video_flag \
                --limit 0 >> "$LOG" 2>&1 &
    BG_PID=$!

    no_growth=0
    last_count=$done_count
    while kill -0 $BG_PID 2>/dev/null || pgrep -f "sentinel.rl.grasps.render_grasps" >/dev/null; do
        sleep "$POLL_INTERVAL"
        cur=$(count_done)

        if [ "$cur" -ge "$target" ]; then
            echo "[$(date '+%F %T')] watchdog: hit $cur done (>=$target), preemptive restart" >> "$LOG"
            kill_run
            break
        fi

        if [ "$cur" -le "$last_count" ]; then
            no_growth=$((no_growth + POLL_INTERVAL))
            if [ "$no_growth" -ge "$NO_GROWTH_TIMEOUT" ]; then
                echo "[$(date '+%F %T')] watchdog: ${no_growth}s no progress, SIGKILL" >> "$LOG"
                kill_run
                break
            fi
        else
            last_count=$cur
            no_growth=0
        fi
    done

    # If the process exited cleanly (no pending rows), the latest log line
    # is "Nothing to render." — bail out of the watchdog loop.
    if tail -5 "$LOG" 2>/dev/null | grep -q "Nothing to render"; then
        echo "[$(date '+%F %T')] watchdog: render_grasps reported nothing pending, done." >> "$LOG"
        break
    fi

    # Brief pause so OS can reclaim file handles / GPU contexts.
    sleep 5
done

echo "[$(date '+%F %T')] watchdog: exited (done=$(count_done))" >> "$LOG"
