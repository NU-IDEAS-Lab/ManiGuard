#!/usr/bin/env bash
# eval_family.sh — full-family eval template for 6fam-base.
#
# Runs maniguard.eval.benchmark on EVERY task of ONE family, auto-classified into
# our ID / OOD taxonomy, saving logs in the standard structure
#   outputs/eval_logs/<leaf>/{ID, OOD/<tag>}/<per-task folders + results/summary>
# One OS process per task (OmniGibson segfaults on og.clear()); per-task PhysX
# GPU-dynamics (liquid scenes) is set automatically; NaN cascades are recorded as
# clean nan_terminated failures by benchmark.py. The engagement metric is built in.
#
# Usage:
#   bash scripts/eval_family.sh <family> [output_leaf] [config]
#     <family>       6fam-base family dir, e.g. clutter_pickup
#     [output_leaf]  leaf under outputs/eval_logs/ (default: <family>_joint)
#     [config]       eval config       (default: configs/eval/<family>_joint.yaml)
#   FORCE=1 to clobber a non-empty output dir (guards against wiping finalized logs).
#
# Requires the family's policy server already serving on 127.0.0.1:8000.
#
# Per-family taxonomy lives in the classifier below. clutter_pickup is implemented;
# add a branch per family as each one's ID/OOD structure is designed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # repo root (ManiGuard)

FAMILY="${1:?usage: eval_family.sh <family> [output_leaf] [config]}"
OUT_LEAF="${2:-${FAMILY}_joint}"
CFG="${3:-configs/eval/${FAMILY}_joint.yaml}"
PY="${PYTHON_CMD:-$HOME/miniconda3/envs/behavior/bin/python}"
FAM_ROOT="outputs/lerobot_datasets/6fam-base/${FAMILY}"
OUT_ROOT="outputs/eval_logs/${OUT_LEAF}"

export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

[ -d "$FAM_ROOT" ] || { echo "ERROR: no family dir: $FAM_ROOT"; exit 1; }
[ -f "$CFG" ]      || { echo "ERROR: no config: $CFG"; exit 1; }

# --- classify every task -> (task_key, bucket, gpu_dyn). Non-GPU tasks first. ---
PLAN_FILE="$(mktemp)"
$PY - "$FAMILY" "$FAM_ROOT" > "$PLAN_FILE" <<'PYEOF'
import sys, json, pathlib
family, fam_root = sys.argv[1], pathlib.Path(sys.argv[2])
# clutter: the 12 dry-SFT target categories. Object NOT in this set + liquid scene
# => novel-object OOD; in this set + liquid scene => OOD by scene; dry => in-domain.
CLUTTER_SFT = {"coffee_cup","teacup","bowl","mug","chalice","goblet","cocktail_glass",
               "beaker","gravy_boat","water_glass","beer_glass","decanter"}
# dusty: the 3 teleop'd (food, source, dest) triples (food is always potato). Any other
# usable task => OOD by object (novel source/dest container, or a new combo of seen ones).
DUSTY_TRIPLES = {("potato","chopping_board","stockpot"),
                 ("potato","platter","mixing_bowl"),
                 ("potato","platter","saucepan")}
# lid: SFT teleop'd only milk_carton (26 eps) + pitcher (4 eps); only milk_carton
# appears in 6fam-base. Food + a seen container => ID; food + novel container => OOD
# by object; liquid pipeline => OOD by target (the container holds liquid, not food).
LID_SFT = {"milk_carton", "pitcher"}
# jar: the manipulated object is ALWAYS hinged_jar and the motion is identical across
# every task (close lid -> carry to goal); only the food content INSIDE the jar varies
# (named in the prompt, partly visible). SFT teleop'd 2 contents. Content in this set
# => ID; any novel content => OOD by content (a soft visual/language shift only, since
# the motion never changes).
JAR_SFT = {"jar_of_cumin", "can_of_bay_leaves"}
tasks = sorted(p.name for p in fam_root.iterdir()
               if p.name.startswith("task_") and (p/"base"/"diagnostics.jsonl").exists())
plan = []
for t in tasks:
    # diagnostics may be single-line OR pretty-printed multi-line (dusty) -> raw_decode
    r = json.JSONDecoder().raw_decode(
        (fam_root/t/"base"/"diagnostics.jsonl").read_text().lstrip())[0]
    prompt = (r.get("prompt") or "").strip()
    scene = r.get("scene_model")
    if not prompt:
        continue                       # genuinely unusable (no prompt) -> skip. scene_model
                                       # may legitimately be None for empty-table families
                                       # (jar): benchmark.py loads those via the empty-Scene
                                       # path (build_og_config) from the task's saved scene_file.
    pipe = r.get("pipeline", "")
    gpu = 1 if "liquid" in pipe else 0                # liquid_transport / lid_transport_liquid
    sel = r.get("selection", {})
    def cat(role):
        return next((s.get("category") for s in sel.get("spawn_specs", [])
                     if s.get("role") == role), "")
    if family == "clutter_pickup":
        c = cat("target")
        if pipe == "table":          bucket = "ID"
        elif c in CLUTTER_SFT:       bucket = "OOD/by-scene_liquid-filled"
        else:                        bucket = "OOD/by-object_novel"
    elif family == "dusty_transfer":
        triple = (cat("food"), cat("source"), cat("dest"))
        bucket = "ID" if triple in DUSTY_TRIPLES else "OOD/by-object_novel"
    elif family == "lid_transport":
        cont = cat("container") or cat("target")
        if pipe == "lid_transport_food":
            bucket = "ID" if cont in LID_SFT else "OOD/by-object_novel"
        else:  # lid_transport_liquid -> the target container holds liquid, not the trained food
            bucket = "OOD/by-target_liquid"
    elif family == "jar_transport":
        food = cat("food")
        bucket = "ID" if food in JAR_SFT else "OOD/by-content_novel"
    else:
        # TODO: specialize this family's ID/OOD taxonomy. Until then, everything -> ID.
        sys.stderr.write(f"WARNING: taxonomy not specialized for '{family}'; all -> ID\n")
        bucket = "ID"
    plan.append((f"{t}/base", bucket, gpu))
for key, bucket, gpu in sorted(plan, key=lambda x: x[2]):   # gpu=0 first
    print(f"{key}\t{bucket}\t{gpu}")
PYEOF

# Optional ID_REPEAT=N: repeat every ID-bucket task N times, so a family with a tiny ID
# set gets an in-domain rate over N runs (eval is stochastic), not a single sample.
# Appends N rows to ID/results.jsonl (the per-task subdir keeps the last run's video).
if [ "${ID_REPEAT:-1}" -gt 1 ] 2>/dev/null; then
  awk -F'\t' -v n="${ID_REPEAT}" '{print} $2=="ID"{for(i=2;i<=n;i++)print}' "$PLAN_FILE" > "$PLAN_FILE.x" && mv "$PLAN_FILE.x" "$PLAN_FILE"
  echo "ID_REPEAT=${ID_REPEAT}: each ID task repeated ${ID_REPEAT}x"
fi

N=$(grep -c . "$PLAN_FILE" || echo 0)
[ "$N" -gt 0 ] || { echo "ERROR: no tasks classified for $FAMILY"; rm -f "$PLAN_FILE"; exit 1; }
echo "@@@@@ eval_family $FAMILY  ($N tasks)  ->  $OUT_ROOT @@@@@"
echo "--- plan (count | bucket | gpu) ---"; awk -F'\t' '{print $2" gpu="$3}' "$PLAN_FILE" | sort | uniq -c

# Guard: never silently clobber an existing (e.g. finalized) log dir.
if [ -d "$OUT_ROOT" ] && [ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]; then
  if [ "${FORCE:-0}" = "1" ]; then echo "FORCE=1: clearing $OUT_ROOT"; rm -rf "$OUT_ROOT"
  else echo "ERROR: $OUT_ROOT exists and is non-empty — pass a fresh output_leaf or FORCE=1."; rm -f "$PLAN_FILE"; exit 1; fi
fi

i=0
while IFS=$'\t' read -r key bucket gpu; do
  i=$((i+1))
  parent="$(dirname "$bucket")"; leaf="$(basename "$bucket")"
  outdir="$OUT_ROOT"; [ "$parent" != "." ] && outdir="$OUT_ROOT/$parent"
  if [ "$gpu" = "1" ]; then export EVAL_USE_GPU_DYNAMICS=1; else unset EVAL_USE_GPU_DYNAMICS; fi
  echo "--- hold 5s before $key ($bucket, gpu=$gpu) ---"; sleep 5
  echo "############ [$i/$N] $key START $(date +%H:%M:%S) -> $bucket ############"
  $PY -m maniguard.eval.benchmark --config "$CFG" --host 127.0.0.1 --port 8000 \
      --scenes "$key" --max-scenes 1 \
      --output-dir "$outdir" --run-name "$leaf" \
      --metrics success safety --headless \
    && echo "############ [$i/$N] $key OK   $(date +%H:%M:%S) ############" \
    || echo "############ [$i/$N] $key exit=$? $(date +%H:%M:%S) ############"
done < "$PLAN_FILE"
rm -f "$PLAN_FILE"

echo "@@@@@ eval_family $FAMILY DONE $(date +%H:%M:%S) @@@@@"
$PY - "$OUT_ROOT" <<'PYEOF'
import sys, json, pathlib, collections
root = pathlib.Path(sys.argv[1])
for rj in sorted(root.rglob("results.jsonl")):
    rows = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]
    d = [r for r in rows if r["status"] == "completed"]
    oc = collections.Counter(r.get("outcome") for r in d)
    nc = sum(1 for r in d if r.get("ever_contacted")); cv = sum(1 for r in d if r.get("counted_violation"))
    succ = sum(1 for r in d if r.get("success")); viol = sum(1 for r in d if r.get("ltl_violated"))
    print(f"{str(rj.parent.relative_to(root)):28} n={len(d)} succ={succ} viol={viol} | "
          f"idle={oc['idle']} reached={oc['reached']} manip={oc['manipulated']} success={oc['success']} | "
          f"contacted={nc} cviol={cv} gated={cv/max(nc,1):.2f} vacuous_safe={len(d)-nc}")
PYEOF
