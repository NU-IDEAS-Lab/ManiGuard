#!/usr/bin/env bash
# Interactive teleop collection over ManiGuard-Bench tasks, with either
# supported leader arm (GELLO or SO-101).
#
# You pick a family (and optionally a task list); the script walks the tasks
# one at a time, launching one teleop episode per trajectory. After every
# trajectory you decide at runtime whether to stay on the same task or move on.
#
# Leader arms:
#   gello  (default)  joint-space leader — launches
#                     maniguard.data.teleop.gello_franka_teleop (JointController,
#                     8D absolute joint action).
#   so101             EEF-delta leader over the ZMQ bridge — launches
#                     maniguard.data.teleop.so101_franka_teleop (IK controller,
#                     7D delta action). Requires teleop_bridge/so101_server.py
#                     running first (lerobot venv; --mock works for a dry test).
#                     The raw 7D recording is converted to the joint-native SFT
#                     format by the Stage-1 re-render (maniguard.data.playback).
#
# Per trajectory:
#   1. Launch the arm's teleop module with --only-successes on the task's scene
#      snapshot. A file is written only on success (S key / auto-success).
#   2. If a file was written, prompt "Keep this trajectory?" (n deletes it).
#   3. Print progress and prompt: [s] same task  [n] next task  [q] finish.
#
# Resume: existing task_NNNN_traj_*.hdf5 in the output dir are counted on
# startup and each task's trajectory index continues past the highest on disk.
# Output: <out-dir>/task_NNNN_traj_NNN.hdf5   (out-dir is arm-tagged by default,
# so GELLO and SO-101 collections never mix). HF push is NOT done here.
#
# Usage (behavior conda env):
#   bash scripts/teleop_family_collect.sh --family jar_transport
#   bash scripts/teleop_family_collect.sh --arm so101 --family stack_retrieve \
#        --tasks 0001 0004 0013 --grasping-mode assisted
#   bash scripts/teleop_family_collect.sh --family lid_transport --dry-run
#
# Options:
#   --arm gello|so101        leader arm (default: gello)
#   --family NAME            bench family dir, e.g. jar_transport  (required)
#   --tasks ID [ID...]       4-digit task IDs; default: every task_* of the family
#   --target N               session target, display reference only (default: 30)
#   --bench-root DIR         benchmark checkout
#                            (default: $BENCH_ROOT or outputs/lerobot_datasets/maniguard-bench)
#   --out-dir DIR            output dir (default: outputs/teleop_collected/<family>_<arm>)
#   --snapshot-name NAME     snapshot file under task_NNNN/base/ (default: scene_ep1.json)
#   --grasping-mode MODE     physical | assisted | sticky (default: sticky)
#   --gello-port PORT        [gello] serial port override
#   --zmq-host HOST          [so101] bridge host (default: 127.0.0.1)
#   --zmq-port PORT          [so101] bridge port (default: 5557)
#   --pos-scale X            [so101] position scaling (default: 5.0)
#   --rot-scale X            [so101] rotation scaling (default: 1.0)
#   --dry-run                resolve config, print the launch command, and exit
#   -- ARGS...               everything after -- is forwarded to the teleop module
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON_CMD:-python}"

# ----- Defaults -----
ARM=gello
FAMILY=""
TASK_IDS=()
TARGET_TOTAL=30
BENCH="${BENCH_ROOT:-outputs/lerobot_datasets/maniguard-bench}"
OUT_DIR=""
SNAPSHOT_NAME=scene_ep1.json
GRASPING_MODE=sticky
GELLO_PORT=""
ZMQ_HOST=127.0.0.1
ZMQ_PORT=5557
POS_SCALE=5.0
ROT_SCALE=1.0
DRY_RUN=0
EXTRA_ARGS=()

usage() { sed -n '2,53p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --arm)            ARM="$2"; shift 2 ;;
    --family)         FAMILY="$2"; shift 2 ;;
    --tasks)          shift
                      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do TASK_IDS+=("$1"); shift; done ;;
    --target)         TARGET_TOTAL="$2"; shift 2 ;;
    --bench-root)     BENCH="$2"; shift 2 ;;
    --out-dir)        OUT_DIR="$2"; shift 2 ;;
    --snapshot-name)  SNAPSHOT_NAME="$2"; shift 2 ;;
    --grasping-mode)  GRASPING_MODE="$2"; shift 2 ;;
    --gello-port)     GELLO_PORT="$2"; shift 2 ;;
    --zmq-host)       ZMQ_HOST="$2"; shift 2 ;;
    --zmq-port)       ZMQ_PORT="$2"; shift 2 ;;
    --pos-scale)      POS_SCALE="$2"; shift 2 ;;
    --rot-scale)      ROT_SCALE="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h | --help)      usage ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1 (see --help)"; exit 2 ;;
  esac
done

[ -n "$FAMILY" ] || { echo "ERROR: --family is required (e.g. --family jar_transport)"; exit 2; }
case "$ARM" in gello | so101) : ;; *) echo "ERROR: --arm must be gello or so101"; exit 2 ;; esac
FAM_ROOT="$BENCH/$FAMILY"
[ -d "$FAM_ROOT" ] || { echo "ERROR: no family dir: $FAM_ROOT (set --bench-root?)"; exit 2; }
[ -n "$OUT_DIR" ] || OUT_DIR="outputs/teleop_collected/${FAMILY}_${ARM}"

# Default task list: every task with a base snapshot.
if [ ${#TASK_IDS[@]} -eq 0 ]; then
  for d in "$FAM_ROOT"/task_*/; do
    [ -f "$d/base/$SNAPSHOT_NAME" ] || continue
    tid="$(basename "$d")"; TASK_IDS+=("${tid#task_}")
  done
fi
total_tasks=${#TASK_IDS[@]}
[ "$total_tasks" -gt 0 ] || { echo "ERROR: no tasks resolved under $FAM_ROOT"; exit 2; }

# ----- Per-arm launch command (argv array; snapshot/output appended per episode) -----
if [ "$ARM" = "gello" ]; then
  LAUNCH=("$PY" -m maniguard.data.teleop.gello_franka_teleop
          --grasping-mode "$GRASPING_MODE" --only-successes)
  [ -n "$GELLO_PORT" ] && LAUNCH+=(--gello-port "$GELLO_PORT")
else
  LAUNCH=("$PY" -m maniguard.data.teleop.so101_franka_teleop
          --zmq-host "$ZMQ_HOST" --zmq-port "$ZMQ_PORT"
          --pos-scale "$POS_SCALE" --rot-scale "$ROT_SCALE"
          --grasping-mode "$GRASPING_MODE" --only-successes)
fi
LAUNCH+=("${EXTRA_ARGS[@]}")

echo "Arm: $ARM   |   Family: $FAMILY   |   tasks: $total_tasks   |   target: $TARGET_TOTAL trajectories"
echo "Bench: $FAM_ROOT"
echo "Out:   $OUT_DIR"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Tasks: ${TASK_IDS[*]}"
  echo "Launch template:"
  echo "  ${LAUNCH[*]} --snapshot $FAM_ROOT/task_${TASK_IDS[0]}/base/$SNAPSHOT_NAME \\"
  echo "    --output-hdf5 $OUT_DIR/task_${TASK_IDS[0]}_traj_NNN.hdf5"
  exit 0
fi

# ----- SO-101 pre-flight: the ZMQ bridge must be up BEFORE Isaac boots -----
if [ "$ARM" = "so101" ]; then
  if ! "$PY" -c "import socket; socket.create_connection(('$ZMQ_HOST', $ZMQ_PORT), 2).close()" 2>/dev/null; then
    echo "ERROR: SO-101 ZMQ bridge not reachable at $ZMQ_HOST:$ZMQ_PORT."
    echo "  Start it first (lerobot venv):"
    echo "    python teleop_bridge/so101_server.py --port /dev/ttyACM0   # real arm"
    echo "    python teleop_bridge/so101_server.py --mock                # no hardware"
    exit 1
  fi
  echo "SO-101 bridge reachable at $ZMQ_HOST:$ZMQ_PORT."
fi

mkdir -p "$OUT_DIR"

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
if [ "$total_saved" -gt 0 ]; then
  echo "Resume: $total_saved existing trajectory file(s) found across configured tasks."
fi

# ----- Main loop -----
cur=0
while true; do
  tid="${TASK_IDS[$cur]}"
  is_last=0
  [ "$cur" -eq $((total_tasks - 1)) ] && is_last=1

  snapshot="$FAM_ROOT/task_${tid}/base/$SNAPSHOT_NAME"
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
  echo "════════ task $tid  ($((cur + 1))/$total_tasks)  →  traj_${tag}  [$ARM] ════════"

  "${LAUNCH[@]}" --snapshot "$snapshot" --output-hdf5 "$outfile"

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
    echo "     re-run later with a fresh --tasks to backfill the total."
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
echo "  Next: Stage-1 re-render + Stage-2 LeRobot export — scripts/render_teleop_to_lerobot.sh"
