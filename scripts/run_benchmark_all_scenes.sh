#!/usr/bin/env bash
set -euo pipefail

# Run maniguard.eval.benchmark for each scene in a SEPARATE process.
# OmniGibson segfaults on og.clear() between scenes, so one python per scene.
#
# Usage:
#   bash scripts/run_benchmark_all_scenes.sh --config configs/eval/sim_table_25k.yaml
#   bash scripts/run_benchmark_all_scenes.sh --config configs/eval/sim_table_25k.yaml --max-steps 500

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse --config from args; pass rest through as CLI overrides.
CONFIG=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 --config <path_to_eval_config.yaml> [CLI overrides...]"
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Auto-detect python from behavior conda env.
if [[ -n "${PYTHON_CMD:-}" ]]; then
    PY="$PYTHON_CMD"
elif command -v conda &>/dev/null && conda run -n behavior python --version &>/dev/null 2>&1; then
    PY="conda run --no-capture-output -n behavior python"
else
    PY="python"
fi
echo "[batch] Using python: $PY"
echo "[batch] Config: ${CONFIG}"

# Read benchmark_root, scene_filter, and output_dir from the config.
# Use tail -1 to skip conda activation noise printed to stdout.
read -r BENCHMARK_ROOT SCENE_FILTER OUTPUT_DIR < <($PY -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('benchmark_root', ''), cfg.get('scene_filter', ''), cfg.get('output_dir', 'outputs/benchmark_eval'))
" 2>/dev/null | tail -1 || echo "")

if [[ -z "$BENCHMARK_ROOT" ]]; then
    echo "[batch] Could not read benchmark_root from config."
    exit 1
fi

# Resolve relative paths from repo root.
[[ "$BENCHMARK_ROOT" = /* ]] || BENCHMARK_ROOT="${REPO_ROOT}/${BENCHMARK_ROOT}"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="${OUTPUT_DIR}/results.jsonl"
> "$RESULTS_FILE"

# Discover scenes.
BENCHMARK_REVISION=$($PY -c "import yaml; print(yaml.safe_load(open('${CONFIG}')).get('benchmark_revision', 'main'))" 2>/dev/null | tail -1 || echo "main")
echo "[batch] Discovering scenes from: ${BENCHMARK_ROOT}"

SCENE_LIST=$($PY -c "
import sys, json
sys.path.insert(0, '${REPO_ROOT}')
from maniguard.data.scene.hf_benchmark import resolve_benchmark_root
from maniguard.eval.scene_discovery import discover_scenes
root = resolve_benchmark_root('${BENCHMARK_ROOT}', revision='${BENCHMARK_REVISION}')
scenes = discover_scenes(str(root))
scene_filter = '${SCENE_FILTER}'
if scene_filter:
    import fnmatch
    scenes = [s for s in scenes if fnmatch.fnmatch(s['name'], scene_filter)]
for s in scenes:
    print(json.dumps({'name': s['name'], 'root': str(root)}))
" 2>/dev/null)

if [[ -z "$SCENE_LIST" ]]; then
    echo "[batch] No scenes discovered."
    exit 1
fi

# Parse into arrays (filter out non-JSON lines from conda run noise).
SCENES=()
RESOLVED_ROOT=""
while IFS= read -r line; do
    [[ "$line" == "{"* ]] || continue
    name=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline())['name'])")
    root=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline())['root'])")
    SCENES+=("$name")
    RESOLVED_ROOT="$root"
done <<< "$SCENE_LIST"

TOTAL=${#SCENES[@]}
echo "[batch] Found ${TOTAL} scenes"
echo "[batch] Output: ${OUTPUT_DIR}"
echo ""

SUCCESS=0
FAILED_LOAD=0
IDX=0

for scene_name in "${SCENES[@]}"; do
    IDX=$((IDX + 1))
    echo "============================================================"
    echo "[batch] Scene ${IDX}/${TOTAL}: ${scene_name}"
    echo "============================================================"

    OMNI_KIT_ACCEPT_EULA=yes \
    VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    $PY -m maniguard.eval.benchmark \
        --config "$CONFIG" \
        --benchmark-root "$RESOLVED_ROOT" \
        --scenes "$scene_name" \
        --max-scenes 1 \
        --output-dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}" 2>&1 | tail -5 || true

    if [[ -f "$RESULTS_FILE" ]]; then
        last_line=$(tail -1 "$RESULTS_FILE" 2>/dev/null || echo "{}")
        status=$(echo "$last_line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline()).get('status','?'))" 2>/dev/null || echo "?")
        success=$(echo "$last_line" | python3 -c "import json,sys; print(json.loads(sys.stdin.readline()).get('success', False))" 2>/dev/null || echo "False")
        [[ "$success" == "True" ]] && SUCCESS=$((SUCCESS + 1))
        [[ "$status" == "load_failed" ]] && FAILED_LOAD=$((FAILED_LOAD + 1))
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
