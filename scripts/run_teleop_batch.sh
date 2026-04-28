#!/usr/bin/env bash
# Batch teleop collection for the mug-into-bowl empty-scene task.
#
# Iterates snapshots in a pipeline run directory (ground_truth first, then
# ep1..epN), launching so101_franka_teleop per episode. After each episode
# the operator is prompted to advance, retry, or quit. Retry deletes the
# partial HDF5 and re-runs the same snapshot.
#
# Usage:
#   scripts/run_teleop_batch.sh [--task <NAME>] [SCENE_DIR] [OUT_DIR]
#
# Args:
#   SCENE_DIR  — directory of scene_ep*.json snapshots (default: stack_same family)
#   OUT_DIR    — explicit output dir (default: outputs/jixing_teleop2_hdf5/<scene_basename>)
#
# Flags:
#   --task <NAME>   Shorthand for SCENE_DIR=$TASK_ROOT/<NAME>, where TASK_ROOT
#                   defaults to outputs/teleop_scenes (override via env var).
#                   Examples (each batches every scene_ep*.json in that family):
#                       --task table             → outputs/teleop_scenes/table
#                       --task transfer          → outputs/teleop_scenes/transfer
#                       --task lid_transport_food → outputs/teleop_scenes/lid_transport_food
#                   Errors if both --task and a positional SCENE_DIR are given.
#   --stock-franka  Forwarded to so101_franka_teleop. Falls back to the stock
#                   BEHAVIOR FrankaPanda asset; default is the long-finger
#                   variant under omnigibson-robot-assets/models/franka/
#                   franka_panda_longfinger/.
#   --scene <FILE>  Run only this single scene snapshot (full filename, must
#                   match the on-disk name exactly — e.g. "scene_ep0005.json"
#                   or "scene_ground_truth.json"). Skips the rest of the
#                   family and exits after this one episode. Errors if the
#                   file is missing.
#   --grasping-mode <MODE>  Forwarded to so101_franka_teleop. One of
#                   {physical, assisted, sticky}. Default 'physical'.
#                   'assisted' welds a grasped object to the gripper via a
#                   force-limited FixedJoint when both fingers contact it
#                   between the AG raycast endpoints — useful for thin/flat
#                   object teleop where physical contact friction isn't
#                   enough.
#
# Hotkeys inside teleop (unchanged from so101_franka_teleop):
#   S = toggle success flag
#   C = checkpoint   R = rollback to checkpoint
#   Q = save HDF5 and exit (returns to this script for next episode)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Parse args (supports --task flag interleaved with positional args) ----
TASK_NAME=""
SCENE_NAME=""
POS_ARGS=()
TELEOP_EXTRA_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --task)
            [[ $# -lt 2 ]] && { echo "[Batch] --task requires an argument" >&2; exit 2; }
            TASK_NAME="$2"; shift 2 ;;
        --task=*)
            TASK_NAME="${1#*=}"; shift ;;
        --scene)
            [[ $# -lt 2 ]] && { echo "[Batch] --scene requires an argument" >&2; exit 2; }
            SCENE_NAME="$2"; shift 2 ;;
        --scene=*)
            SCENE_NAME="${1#*=}"; shift ;;
        --stock-franka)
            TELEOP_EXTRA_ARGS+=(--stock-franka); shift ;;
        --grasping-mode)
            [[ $# -lt 2 ]] && { echo "[Batch] --grasping-mode requires an argument" >&2; exit 2; }
            TELEOP_EXTRA_ARGS+=(--grasping-mode "$2"); shift 2 ;;
        --grasping-mode=*)
            TELEOP_EXTRA_ARGS+=(--grasping-mode "${1#*=}"); shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        --)
            shift; while (( $# > 0 )); do POS_ARGS+=("$1"); shift; done ;;
        --*)
            echo "[Batch] Unknown flag: $1" >&2; exit 2 ;;
        *)
            POS_ARGS+=("$1"); shift ;;
    esac
done

# Resolve SCENE_DIR. --task <name> is a shorthand for $TASK_ROOT/<name>.
TASK_ROOT="${TASK_ROOT:-outputs/teleop_scenes}"
if [[ -n "$TASK_NAME" ]]; then
    if [[ -n "${POS_ARGS[0]:-}" ]]; then
        echo "[Batch] Cannot use --task and a positional SCENE_DIR together" >&2
        exit 2
    fi
    SCENE_DIR="$TASK_ROOT/$TASK_NAME"
    if [[ ! -d "$SCENE_DIR" ]]; then
        echo "[Batch] --task '$TASK_NAME' resolves to '$SCENE_DIR' which does not exist." >&2
        echo "[Batch] Available tasks under $TASK_ROOT:" >&2
        for d in "$TASK_ROOT"/*/; do [[ -d "$d" ]] && echo "  $(basename "$d")" >&2; done
        exit 1
    fi
else
    SCENE_DIR="${POS_ARGS[0]:-outputs/teleop_scenes/stack_same}"
fi
# Default OUT_DIR nests under a subdirectory named after SCENE_DIR's basename
# (e.g. SCENE_DIR=outputs/teleop_scenes/table -> OUT_DIR=outputs/jixing_teleop2_hdf5/table).
# This prevents cross-task collisions when the same scene_ep<NNNN>.json filenames
# exist in multiple SCENE_DIRs (e.g. table/scene_ep0000.json and transfer/scene_ep0000.json
# would otherwise both write to scene_ep0000.hdf5 in a flat OUT_DIR and overwrite each other).
# Pass an explicit second positional arg to override and write directly into a chosen path.
OUT_ROOT="${POS_ARGS[1]:-outputs/jixing_teleop2_hdf5}"
if [[ -n "${POS_ARGS[1]:-}" ]]; then
    OUT_DIR="$OUT_ROOT"
else
    OUT_DIR="$OUT_ROOT/$(basename "$SCENE_DIR")"
fi
mkdir -p "$OUT_DIR"

# Build ordered snapshot list: ground_truth first, then scene_ep1..scene_epN
# (sorted numerically, not lexically, so ep2 comes before ep10).
SNAPSHOTS=()
if [[ -f "$SCENE_DIR/scene_ground_truth.json" ]]; then
    SNAPSHOTS+=("$SCENE_DIR/scene_ground_truth.json")
fi
while IFS= read -r f; do
    SNAPSHOTS+=("$f")
done < <(ls -1 "$SCENE_DIR"/scene_ep*.json 2>/dev/null \
            | grep -E '/scene_ep[0-9]+\.json$' \
            | sed -E 's|.*scene_ep([0-9]+)\.json|\1 &|' \
            | sort -n | awk '{print $2}')

total=${#SNAPSHOTS[@]}
if (( total == 0 )); then
    echo "[Batch] No snapshots found in $SCENE_DIR" >&2
    exit 1
fi

# --scene narrows the list to a single snapshot. The arg is the full
# filename (e.g. "scene_ep5.json") and must already exist in SCENE_DIR.
if [[ -n "$SCENE_NAME" ]]; then
    target="$SCENE_DIR/$SCENE_NAME"
    if [[ ! -f "$target" ]]; then
        echo "[Batch] --scene '$SCENE_NAME' not found at $target" >&2
        echo "[Batch] Available snapshots in $SCENE_DIR:" >&2
        for s in "${SNAPSHOTS[@]}"; do echo "  $(basename "$s")" >&2; done
        exit 1
    fi
    SNAPSHOTS=("$target")
    total=1
    echo "[Batch] --scene set: running only $SCENE_NAME"
fi

echo "[Batch] Scene dir: $SCENE_DIR"
echo "[Batch] Output dir: $OUT_DIR"
echo "[Batch] Episodes to collect: $total"
echo "[Batch] Hotkeys inside teleop: S=success, C=checkpoint, R=rollback, Q=save+exit"
echo

min_hdf5_bytes=8192  # Empty DataCollectionWrapper HDF5 header is ~2-4 KB; real trajectories >>.

for ((i=0; i<total; i++)); do
    snap="${SNAPSHOTS[$i]}"
    name="$(basename "$snap" .json)"
    hdf5="$OUT_DIR/${name}.hdf5"

    if [[ -f "$hdf5" ]]; then
        existing_size=$(stat -c%s "$hdf5")
        if (( existing_size >= min_hdf5_bytes )); then
            echo "[Batch] Skipping $name — already collected ($hdf5, ${existing_size} B)."
            continue
        fi
        echo "[Batch] Found stale/empty $hdf5 (${existing_size} B); will re-run."
        rm -f "$hdf5"
    fi

    while :; do
        echo "============================================================"
        echo "[Batch] Episode $((i+1))/$total: $name"
        echo "[Batch] Snapshot: $snap"
        echo "[Batch] Output:   $hdf5"
        echo "============================================================"

        VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        python -m sentinel.teleop.so101_franka_teleop \
            --snapshot "$snap" \
            --output-hdf5 "$hdf5" \
            --only-successes \
            ${TELEOP_EXTRA_ARGS[@]+"${TELEOP_EXTRA_ARGS[@]}"}
        teleop_status=$?

        size=0
        [[ -f "$hdf5" ]] && size=$(stat -c%s "$hdf5")

        if [[ $teleop_status -ne 0 ]]; then
            echo "[Batch] Teleop exited with status $teleop_status."
        fi

        if (( size >= min_hdf5_bytes )); then
            echo "[Batch] $name recorded OK (${size} B)."
            status_tag="OK"
        else
            echo "[Batch] $name has no/empty HDF5 (${size} B — success flag not set?)."
            status_tag="EMPTY"
        fi

        echo
        read -r -p "  [Enter]=next / r=retry / s=skip / q=quit  > " choice
        case "${choice:-}" in
            ""|n|N)
                if [[ "$status_tag" == "EMPTY" ]]; then
                    read -r -p "  No HDF5 yet. Advance anyway? [y/N]  > " confirm
                    if [[ "${confirm:-N}" != "y" && "${confirm:-N}" != "Y" ]]; then
                        rm -f "$hdf5"
                        continue
                    fi
                fi
                break ;;
            r|R)
                rm -f "$hdf5"
                echo "[Batch] Retrying $name..."
                continue ;;
            s|S)
                rm -f "$hdf5"
                echo "[Batch] Skipped $name."
                break ;;
            q|Q)
                echo "[Batch] Aborting batch at episode $((i+1))/$total."
                exit 0 ;;
            *)
                echo "[Batch] Unrecognized choice '$choice'; retrying." ;;
        esac
    done
done

echo
echo "[Batch] Finished. HDF5 files:"
ls -la "$OUT_DIR" | awk 'NR>1 {print "  " $NF " (" $5 " B)"}'