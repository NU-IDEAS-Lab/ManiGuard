#!/usr/bin/env bash
# Template: raw GELLO teleop HDF5  ->  joint + 3-cam rendered HDF5  ->  LeRobot
# v2.1 multitask dataset. Reusable across task families: edit ONLY the CONFIG
# block below (paths + repo id), the script body is family-agnostic.
#
#   Stage 1  re-render each raw teleop trajectory through OmniGibson, recording
#            joint-8d state/action + 3 cameras (image_left, image_right,
#            wrist_image) at 256x256. One process per trajectory (Isaac Sim
#            scene-teardown is flaky across episodes in a single process; it
#            also segfaults at og.clear() AFTER the HDF5 is fully written, so
#            success is judged by a non-empty action dataset, not the exit code).
#            Skips a trajectory whose rendered HDF5 already exists (resume).
#
#   Stage 2  ingest the rendered HDF5s into a LeRobot v2.1 dataset. Each task's
#            language prompt is read from <DIAG_ROOT>/<task_id>/diagnostics.jsonl,
#            so a multitask family becomes one dataset with per-frame task_index.
#            Built locally; HF push is a separate, explicit step (see footer).
#
# Envs (two, separate):
#   - Stage 1 runs in the `behavior` conda env (OmniGibson).
#   - Stage 2 runs in the lerobot uv venv ($LEROBOT_PY).
#
# Usage:
#   conda activate behavior          # Stage 1 needs it on PATH for `conda run`
#   bash scripts/render_teleop_to_lerobot.sh                 # both stages
#   bash scripts/render_teleop_to_lerobot.sh --stage1        # render only
#   bash scripts/render_teleop_to_lerobot.sh --stage2        # convert only
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ============ CONFIG (edit per family) ============
FAMILY=jar_transport
IN_DIR="outputs/teleop_collected/${FAMILY}"                  # raw teleop HDF5s
RENDER_DIR="outputs/teleop_rendered_joint_3cam/${FAMILY}"    # Stage 1 output (flat)
# NOTE: a family's raw dir name may differ from its 6fam-base diag-tree name
# (e.g. raw 'lid_transport_food' vs diag 'lid_transport'). When they differ,
# write DIAG_ROOT out explicitly instead of reusing ${FAMILY}.
DIAG_ROOT="outputs/lerobot_datasets/6fam-base/${FAMILY}"     # <task_id>/base/diagnostics.jsonl (per-task prompt)
REPO_ID="IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam"   # LeRobot repo id (metadata)
LEROBOT_ROOT="outputs/lerobot_datasets/sim-jar-transport-30-joint-3cam"   # local dataset dir
# lerobot uv venv (build once with:
#   uv venv --python 3.11 .venv-lerobot
#   uv pip install --python .venv-lerobot/bin/python 'lerobot<0.4' h5py pyarrow opencv-python)
LEROBOT_PY=".venv-lerobot/bin/python"
GPU=0                                                        # CUDA device for Stage 1 render
# ==================================================

DO_STAGE1=1; DO_STAGE2=1
case "${1:-}" in
  --stage1) DO_STAGE2=0 ;;
  --stage2) DO_STAGE1=0 ;;
  "") ;;
  *) echo "usage: $0 [--stage1|--stage2]" >&2; exit 2 ;;
esac

# ---- Stage 1: render raw teleop -> joint + 3-cam HDF5 ----------------------
if [ "$DO_STAGE1" -eq 1 ]; then
  mkdir -p "$RENDER_DIR"
  mapfile -t FILES < <(ls "$IN_DIR"/*.hdf5 2>/dev/null | sort)
  n=${#FILES[@]}
  if [ "$n" -eq 0 ]; then
    echo "[stage1] no HDF5 under $IN_DIR" >&2; exit 1
  fi
  echo "[stage1] $n trajectories under $IN_DIR -> $RENDER_DIR"
  ok=0; skip=0; fail=0; failed_list=(); i=0
  for f in "${FILES[@]}"; do
    i=$((i + 1)); base=$(basename "$f"); out="$RENDER_DIR/$base"
    echo "[stage1] ($i/$n) $base"
    if [ -f "$out" ]; then echo "         -> exists, skip"; skip=$((skip + 1)); continue; fi
    # default controller=joint, cams=3 (no flags). Teardown segfault rc ignored.
    # OMNIGIBSON_HEADLESS=1: playback renders via offscreen external VisionSensors
    # (not the GUI viewport), so re-render runs fully headless on any box.
    OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES="$GPU" conda run -n behavior python -m maniguard.data.playback \
      --input "$f" --output "$out"
    # Judge success by output existing + non-empty action dataset, not rc.
    n_act=$("$LEROBOT_PY" - "$out" <<'PY' 2>/dev/null
import sys, h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        d = f["data"][sorted(f["data"].keys())[0]]
        print(d["action"].shape[0])
except Exception:
    print(0)
PY
)
    if [ -f "$out" ] && [ "${n_act:-0}" -gt 0 ]; then
      echo "         -> OK (${n_act} frames; teardown rc ignored)"; ok=$((ok + 1))
    else
      echo "         -> FAIL (no valid output)"; fail=$((fail + 1)); failed_list+=("$base")
    fi
  done
  echo "[stage1] DONE: ok=$ok skip=$skip fail=$fail / total=$n"
  if [ "$fail" -gt 0 ]; then
    echo "[stage1] FAILED: ${failed_list[*]}" >&2; exit 1
  fi
fi

# ---- Stage 2: rendered HDF5 -> LeRobot v2.1 multitask dataset --------------
if [ "$DO_STAGE2" -eq 1 ]; then
  echo "[stage2] $RENDER_DIR -> $LEROBOT_ROOT  (repo_id=$REPO_ID)"
  "$LEROBOT_PY" -m maniguard.data.lerobot.multitask_lerobot_export \
    --input-root "$RENDER_DIR" \
    --diag-root "$DIAG_ROOT" \
    --repo-id "$REPO_ID" \
    --root "$LEROBOT_ROOT" \
    --fps 30 --resolution 256
  echo "[stage2] DONE -> $LEROBOT_ROOT"
  echo
  echo "[hint] to push to HF later, re-run Stage 2 adding --push-to-hub $REPO_ID"
  echo "       (and --hub-private for a private repo), or call the exporter directly."
fi
