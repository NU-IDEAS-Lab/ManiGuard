#!/usr/bin/env bash
set -euo pipefail

# Run sentinel.eval.benchmark for each scene in a SEPARATE process.
# OmniGibson can't reliably reload scenes in the same process (og.clear()
# segfaults on the second scene), so we spawn one python per scene and
# aggregate the per-scene results.jsonl at the end.
#
# Supports both local paths and HuggingFace dataset repo_ids as
# --benchmark-root. For HF repos, a one-shot snapshot_download resolves
# the repo to a local cache dir, then scenes are enumerated from there.
#
# Usage:
#   bash scripts/run_benchmark_all_scenes.sh \
#       --benchmark-root IDEAS-Lab-Northwestern/sentinel-lite-taskgen-staging-20260415-v1 \
#       --host 192.168.x.x --port 8000 --max-steps 300 --save-video
#
#   bash scripts/run_benchmark_all_scenes.sh \
#       --benchmark-root datasets/safety-benchmark \
#       --host 127.0.0.1 --port 8000 --max-steps 100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse --benchmark-root and --benchmark-revision from args; pass rest through.
BENCHMARK_ROOT=""
BENCHMARK_REVISION="main"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
        --benchmark-revision) BENCHMARK_REVISION="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$BENCHMARK_ROOT" ]]; then
    echo "Usage: $0 --benchmark-root <local_path_or_hf_repo_id> [--benchmark-revision main] [--host <ip>] [--port <port>] [other args]"
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/RLinf:${PYTHONPATH:-}"

# The per-scene eval needs the behavior conda env (OmniGibson + Isaac Sim).
# Auto-detect or accept override via PYTHON_CMD.
if [[ -n "${PYTHON_CMD:-}" ]]; then
    PY="$PYTHON_CMD"
elif command -v conda &>/dev/null && conda run -n behavior python --version &>/dev/null 2>&1; then
    PY="conda run --no-capture-output -n behavior python"
elif [[ -x "/home/nu-ideas-4080/miniconda3/envs/behavior/bin/python" ]]; then
    PY="/home/nu-ideas-4080/miniconda3/envs/behavior/bin/python"
else
    PY="python"
fi
echo "[batch] Using python: $PY"

OUTPUT_DIR="${REPO_ROOT}/outputs/benchmark_eval"
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="${OUTPUT_DIR}/results.jsonl"

# Clear previous results
> "$RESULTS_FILE"

# -----------------------------------------------------------------------
# Discover scenes via Python (handles local 1-level, HF 2-level layouts,
# and snapshot_download for HF repo_ids — all in one call).
# -----------------------------------------------------------------------
echo "[batch] Discovering scenes from: ${BENCHMARK_ROOT} @ ${BENCHMARK_REVISION}"

SCENE_LIST=$(python -c "
import sys, json
sys.path.insert(0, '${REPO_ROOT}')
from sentinel.data.hf_benchmark import resolve_benchmark_root
from sentinel.eval.scene_discovery import discover_scenes
root = resolve_benchmark_root('${BENCHMARK_ROOT}', revision='${BENCHMARK_REVISION}')
scenes = discover_scenes(str(root))
for s in scenes:
    print(json.dumps({'name': s['name'], 'root': str(root)}))
" 2>/dev/null)

if [[ -z "$SCENE_LIST" ]]; then
    echo "[batch] No scenes discovered. Check --benchmark-root and diagnostics."
    exit 1
fi

# Parse into arrays
SCENES=()
RESOLVED_ROOT=""
while IFS= read -r line; do
    name=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline())['name'])")
    root=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline())['root'])")
    SCENES+=("$name")
    RESOLVED_ROOT="$root"
done <<< "$SCENE_LIST"

TOTAL=${#SCENES[@]}
echo "[batch] Found ${TOTAL} scenes (resolved root: ${RESOLVED_ROOT})"
echo "[batch] Results will be saved to ${RESULTS_FILE}"
echo ""

SUCCESS=0
FAILED_LOAD=0
IDX=0

for scene_name in "${SCENES[@]}"; do
    IDX=$((IDX + 1))
    echo "============================================================"
    echo "[batch] Scene ${IDX}/${TOTAL}: ${scene_name}"
    echo "============================================================"

    # Each scene gets its own python process. --scenes filters to just
    # this one scene; --benchmark-root is the resolved local path (not
    # the HF repo_id) so we skip re-downloading inside the child.
    # Isaac Sim segfaults on og.clear() shutdown (exit 139) even when
    # the eval itself succeeded. Tolerate non-zero exits here — the
    # results.jsonl tally below determines real success/failure.
    OMNI_KIT_ACCEPT_EULA=yes \
    VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    $PY -m sentinel.eval.benchmark \
        --benchmark-root "$RESOLVED_ROOT" \
        --scenes "$scene_name" \
        --max-scenes 1 \
        --output-dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}" 2>&1 | tail -5 || true

    # Tally from the last result line
    if [[ -f "$RESULTS_FILE" ]]; then
        last_line=$(tail -1 "$RESULTS_FILE" 2>/dev/null || echo "{}")
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
if [[ $COMPLETED -gt 0 ]]; then
    echo "Success rate: $(python3 -c "print(f'{${SUCCESS}/${COMPLETED}*100:.1f}%')")"
fi
echo "Results: ${RESULTS_FILE}"
