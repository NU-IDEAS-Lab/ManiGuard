#!/usr/bin/env bash
# Interactive single-family teleop collection with per-task, runtime-decided
# trajectory counts.
#
# You pick a list of task IDs for ONE family and a total-trajectory target.
# The script walks the list one task at a time; after every trajectory you
# decide at runtime whether to stay on the same task or move to the next one.
#
# Per trajectory:
#   1. Launch maniguard.data.teleop.gello_franka_teleop with --only-successes
#      on the family's scene snapshot. A file is written only on success
#      (S key / auto-success). Q without success writes nothing.
#   2. If a file was written, prompt "Keep this trajectory?".
#        y (default): kept, counts toward the total.
#        n: rm the file.
#   3. Print progress: trajectories saved so far, count for this task,
#      and how many tasks in the list have not been started yet.
#   4. Navigation prompt:
#        [s] same task again   [n] next task   [q] finish session
#      On the LAST task in the list, [n] is dropped.
#
# TARGET_TOTAL is only a display reference — the script never auto-stops on
# reaching it. The session ends when you choose [q] (or [n] on the last task).
#
# Resume: existing task_NNNN_traj_*.hdf5 in OUT_DIR are counted toward the
# total on startup, and each task's trajectory index continues past the
# highest one already on disk.
#
# Output files: <OUT_DIR>/task_NNNN_traj_NNN.hdf5  (3-digit zero-padded index).
# HF push is intentionally NOT done here — collect locally, push later.
#
# Run (teleop uses the `behavior` conda env, separate from the SFT uv env):
#   conda activate behavior
#   bash scripts/teleop_family_collect.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ============ CONFIG (edit these per session) ============
FAMILY=jar_transport
DATASET=outputs/lerobot_datasets/6fam-base
OUT_DIR=outputs/teleop_collected/${FAMILY}
SNAPSHOT_NAME=scene_ep1.json         # snapshot lives at $DATASET/$FAMILY/task_NNNN/base/$SNAPSHOT_NAME
GRASPING_MODE=sticky                 # physical | assisted | sticky
TARGET_TOTAL=30                      # display reference only — not a hard stop

# Task IDs to collect for this family (4-digit zero-padded).
# jar_transport has task_0000 .. task_0026 (27 tasks); browse each dir's
# base/rollout_*.mp4 to preview. Trim this list to collect a subset.
TASK_IDS=(
  0001 0004 0005 0006 0007 0008
  0009 0010 0011 0012 0013 0014 0015 0016 0017
  0018 0019 0020 0021 0022 0023 0024 0025 0026
)
# =========================================================

mkdir -p "$OUT_DIR"

total_tasks=${#TASK_IDS[@]}
if [ "$total_tasks" -eq 0 ]; then
  echo "TASK_IDS is empty — nothing to collect."
  exit 1
fi

# yes/no prompt with default
prompt_yes() {
  local q="$1" default="${2:-y}" hint ans
  [ "$default" = "y" ] && hint="[Y/n]" || hint="[y/N]"
  read -r -p "$q $hint " ans
  ans="${ans:-$default}"
  [[ "$ans" =~ ^[Yy] ]]
}

# Number of trajectory files on disk for a task ID.
count_existing() {
  local tid="$1" n=0 f
  for f in "$OUT_DIR/task_${tid}_traj_"*.hdf5; do
    [ -f "$f" ] && n=$((n + 1))
  done
  echo "$n"
}

# Next free 3-digit trajectory index for a task ID (max-on-disk + 1).
next_traj_idx() {
  local tid="$1" max=0 f base num
  for f in "$OUT_DIR/task_${tid}_traj_"*.hdf5; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .hdf5)   # task_NNNN_traj_NNN
    num=${base##*_traj_}          # NNN
    num=$((10#$num))              # strip leading zeros
    [ "$num" -gt "$max" ] && max=$num
  done
  printf "%03d" $((max + 1))
}

print_progress() {
  local tid="$1" cur="$2" task_count remaining
  task_count=$(count_existing "$tid")
  remaining=$((total_tasks - cur - 1))
  echo
  echo "  ── progress ──────────────────────────────"
  echo "    trajectories saved : $total_saved / $TARGET_TOTAL"
  echo "    this task ($tid)    : $task_count on disk"
  if [ "$remaining" -gt 0 ]; then
    echo "    tasks not yet started : $remaining  → ${TASK_IDS[*]:$((cur + 1))}"
  else
    echo "    tasks not yet started : 0  (this is the LAST task in the list)"
  fi
  echo "  ──────────────────────────────────────────"
}

# Navigation prompt. Sets global NAV to s | n | q.
nav_prompt() {
  local is_last="$1" opts ans
  if [ "$is_last" -eq 1 ]; then
    opts="[s] same task again   [q] finish session"
  else
    opts="[s] same task again   [n] next task   [q] finish session"
  fi
  while true; do
    read -r -p "  → $opts : " ans
    case "${ans:-}" in
      s | S) NAV=s; return ;;
      n | N)
        if [ "$is_last" -eq 1 ]; then
          echo "    (already on the last task — no next; use [s] or [q])"
        else
          NAV=n; return
        fi
        ;;
      q | Q) NAV=q; return ;;
      *) echo "    please type s / n / q" ;;
    esac
  done
}

trap 'echo; echo "Interrupted. Current trajectory left as-is on disk."; exit 130' INT

# ----- Resume scan -----
total_saved=0
for tid in "${TASK_IDS[@]}"; do
  total_saved=$((total_saved + $(count_existing "$tid")))
done
echo "Family: $FAMILY   |   tasks: $total_tasks   |   target: $TARGET_TOTAL trajectories"
if [ "$total_saved" -gt 0 ]; then
  echo "Resume: $total_saved existing trajectory file(s) found across configured tasks."
fi

# ----- Main loop -----
cur=0
while true; do
  tid="${TASK_IDS[$cur]}"
  is_last=0
  [ "$cur" -eq $((total_tasks - 1)) ] && is_last=1

  snapshot="$DATASET/$FAMILY/task_${tid}/base/$SNAPSHOT_NAME"
  if [ ! -f "$snapshot" ]; then
    echo
    echo "  [ERROR] snapshot not found: $snapshot"
    if [ "$is_last" -eq 1 ]; then
      echo "  Last task — nothing else to do."
      break
    fi
    echo "  Skipping task $tid → next."
    cur=$((cur + 1))
    continue
  fi

  tag=$(next_traj_idx "$tid")
  outfile="$OUT_DIR/task_${tid}_traj_${tag}.hdf5"

  echo
  echo "════════ task $tid  ($((cur + 1))/$total_tasks)  →  traj_${tag} ════════"

  python -m maniguard.data.teleop.gello_franka_teleop \
    --snapshot "$snapshot" \
    --output-hdf5 "$outfile" \
    --grasping-mode "$GRASPING_MODE" \
    --only-successes

  if [ -f "$outfile" ]; then
    sz_kb=$(( $(stat -c %s "$outfile" 2>/dev/null || echo 0) / 1024 ))
    echo "  Recorded: $outfile (${sz_kb} KB)"
    if prompt_yes "  Keep this trajectory?" "y"; then
      total_saved=$((total_saved + 1))
      echo "  → kept"
    else
      rm -f "$outfile"
      echo "  → discarded"
    fi
  else
    echo "  No trajectory saved (no success / Q without S)."
  fi

  print_progress "$tid" "$cur"
  if [ "$is_last" -eq 1 ]; then
    echo "  ⚠  This is the LAST task in your list — top it up, or [q] and"
    echo "     re-run later with a fresh TASK_IDS to backfill the total."
  fi

  nav_prompt "$is_last"
  case "$NAV" in
    s) : ;;                  # stay on same task
    n) cur=$((cur + 1)) ;;   # advance to next task
    q) break ;;
  esac
done

# ----- Session summary -----
echo
echo "════════ session done ════════"
echo "  trajectories on disk (configured tasks): $total_saved / $TARGET_TOTAL"
for tid in "${TASK_IDS[@]}"; do
  echo "    task $tid : $(count_existing "$tid")"
done
echo "  files in: $OUT_DIR/"
echo "  (HF push not done here — push when you're satisfied with the set.)"
