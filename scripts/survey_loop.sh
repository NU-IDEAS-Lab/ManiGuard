#!/bin/bash
# Auto-restart wrapper for maniguard.rl.grasps.survey_graspability.
#
# Why: the survey runs OmniGibson + cuRobo for 5+ hours, and Isaac Sim
# accumulates state that occasionally triggers a PhysX hang or per-IK
# wallclock creep, after which subsequent objects either timeout or get
# stuck in an uninterruptible sim.step. The script's per-object deadline
# isn't enforceable inside a stuck sim.step, so a fresh process is the
# only reliable fix.
#
# Strategy:
#   1. Restart the survey every ~200 rows of CSV growth (preemptive).
#   2. SIGKILL + restart if CSV doesn't grow for 5 minutes (degradation).
#   3. Exit when CSV reaches the full target count (6469 + 1 header).
#
# Usage:
#   bash scripts/survey_loop.sh
set -u
cd "$(dirname "$0")/.."

CSV=outputs/grasp_datasets/survey/graspability.csv
LOG=outputs/grasp_datasets/survey/run.log
TOTAL_TARGETS=6469
RESTART_EVERY=200      # rows of CSV growth between preemptive restarts
NO_GROWTH_TIMEOUT=300  # seconds of zero CSV growth before SIGKILL
POLL_INTERVAL=30       # seconds between watchdog checks

mkdir -p "$(dirname "$CSV")"

kill_survey() {
    # The conda-run wrapper doesn't always propagate signals; kill anything
    # whose cmdline matches the survey module.
    pkill -9 -f "maniguard.rl.grasps.survey_graspability" 2>/dev/null || true
    sleep 1
}

while true; do
    last_count=$(wc -l < "$CSV" 2>/dev/null || echo 0)
    if [ "$last_count" -ge "$((TOTAL_TARGETS + 1))" ]; then
        echo "[$(date '+%F %T')] watchdog: CSV has $last_count rows, target reached. Done." >> "$LOG"
        break
    fi

    target=$((last_count + RESTART_EVERY))
    echo "[$(date '+%F %T')] watchdog: launching survey (rows=$last_count target=$target)" >> "$LOG"

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \
        conda run -n behavior --no-capture-output \
            python -u -m maniguard.rl.grasps.survey_graspability \
                --output "$CSV" >> "$LOG" 2>&1 &
    BG_PID=$!

    no_growth=0
    progress_count=$last_count
    while kill -0 $BG_PID 2>/dev/null || pgrep -f "maniguard.rl.grasps.survey_graspability" >/dev/null; do
        sleep "$POLL_INTERVAL"
        cur=$(wc -l < "$CSV" 2>/dev/null || echo 0)

        if [ "$cur" -ge "$((TOTAL_TARGETS + 1))" ]; then
            echo "[$(date '+%F %T')] watchdog: target reached mid-run, stopping survey" >> "$LOG"
            kill_survey
            break
        fi

        if [ "$cur" -ge "$target" ]; then
            echo "[$(date '+%F %T')] watchdog: hit $cur rows (>=$target), preemptive restart" >> "$LOG"
            kill_survey
            break
        fi

        if [ "$cur" -le "$progress_count" ]; then
            no_growth=$((no_growth + POLL_INTERVAL))
            if [ "$no_growth" -ge "$NO_GROWTH_TIMEOUT" ]; then
                echo "[$(date '+%F %T')] watchdog: ${no_growth}s no CSV growth, SIGKILL" >> "$LOG"
                kill_survey
                break
            fi
        else
            progress_count=$cur
            no_growth=0
        fi
    done

    # Brief pause so the OS can reclaim file handles / GPU contexts.
    sleep 5
done

echo "[$(date '+%F %T')] watchdog: exited" >> "$LOG"
