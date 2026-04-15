#!/usr/bin/env bash
set -euo pipefail

# Run evaluate_benchmark.py for each scene in a separate process.
# OmniGibson can't reliably reload scenes in the same process.
#
# Usage:
#   bash scripts/run_benchmark_all_scenes.sh \
#       --benchmark-root outputs/local_eval_benchmark/clutter_all_scene_20260319 \
#       --host 192.168.x.x --port 8000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse --benchmark-root from args
BENCHMARK_ROOT=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$BENCHMARK_ROOT" ]]; then
    echo "Usage: $0 --benchmark-root <path> [--host <ip>] [--port <port>] [other args]"
    exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/outputs/benchmark_eval"
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="${OUTPUT_DIR}/results.jsonl"

# Clear previous results
> "$RESULTS_FILE"

# Discover valid scene directories
SCENES=()
for scene_dir in "${BENCHMARK_ROOT}"/*/; do
    scene_name="$(basename "$scene_dir")"
    if [[ -f "${scene_dir}/scene_ep1.json" ]] && [[ -f "${scene_dir}/diagnostics.jsonl" ]]; then
        SCENES+=("$scene_name")
    fi
done

echo "Found ${#SCENES[@]} scenes in ${BENCHMARK_ROOT}"
echo "Results will be saved to ${RESULTS_FILE}"
echo ""

TOTAL=${#SCENES[@]}
SUCCESS=0
FAILED_LOAD=0
IDX=0

for scene_name in "${SCENES[@]}"; do
    IDX=$((IDX + 1))
    echo "============================================================"
    echo "Scene ${IDX}/${TOTAL}: ${scene_name}"
    echo "============================================================"

    OMNI_KIT_ACCEPT_EULA=yes \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CUDA_VISIBLE_DEVICES=0 \
    python -m sentinel.eval.benchmark \
        --benchmark-root "$BENCHMARK_ROOT" \
        --scenes "$scene_name" \
        --max-scenes 1 \
        --output-dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}" 2>&1 | tail -5

    # Check last result
    if [[ -f "$RESULTS_FILE" ]]; then
        last_line=$(tail -1 "$RESULTS_FILE")
        status=$(echo "$last_line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline()).get('status','?'))" 2>/dev/null || echo "?")
        success=$(echo "$last_line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline()).get('success', False))" 2>/dev/null || echo "False")
        if [[ "$success" == "True" ]]; then
            SUCCESS=$((SUCCESS + 1))
        fi
        if [[ "$status" == "load_failed" ]]; then
            FAILED_LOAD=$((FAILED_LOAD + 1))
        fi
    fi
    echo ""
done

COMPLETED=$((TOTAL - FAILED_LOAD))
echo "============================================================"
echo "FINAL SUMMARY"
echo "============================================================"
echo "Total scenes: ${TOTAL}"
echo "Loaded: ${COMPLETED}"
echo "Failed to load: ${FAILED_LOAD}"
echo "Success: ${SUCCESS}/${COMPLETED}"
echo "Results: ${RESULTS_FILE}"
