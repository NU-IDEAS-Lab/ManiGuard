#!/usr/bin/env bash
# Batch teleop collection for the mug-into-bowl empty-scene task.
#
# Iterates snapshots in a pipeline run directory (ground_truth first, then
# ep1..epN), launching so101_franka_teleop per episode. After each episode
# the operator is prompted to advance, retry, or quit. Retry deletes the
# partial HDF5 and re-runs the same snapshot.
#
# Usage:
#   scripts/collect_teleop_batch.sh [SCENE_DIR] [OUT_DIR]
#
# Defaults:
#   SCENE_DIR = outputs/pipeline_runs/mug_into_bowl_empty_20260418_132924
#   OUT_DIR   = outputs/teleop_hdf5/<timestamp>
#
# Hotkeys inside teleop (unchanged from so101_franka_teleop):
#   S = toggle success flag
#   C = checkpoint   R = rollback to checkpoint
#   Q = save HDF5 and exit (returns to this script for next episode)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCENE_DIR="${1:-outputs/pipeline_runs/mug_into_bowl_empty_20260419_202321}"
OUT_DIR="${2:-outputs/jixing_teleop_hdf5}"
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

    while :; do
        echo "============================================================"
        echo "[Batch] Episode $((i+1))/$total: $name"
        echo "[Batch] Snapshot: $snap"
        echo "[Batch] Output:   $hdf5"
        echo "============================================================"

        python -m sentinel.teleop.so101_franka_teleop \
            --snapshot "$snap" \
            --output-hdf5 "$hdf5" \
            --only-successes
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
