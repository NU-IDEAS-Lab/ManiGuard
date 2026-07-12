# Datagen — scripted demo collection pipeline

This is the **`maniguard/data/datagen/`** pipeline: it turns the read-only
**ManiGuard-Bench** base tasks into large numbers of **success + safe** manipulation
demonstrations for SFT, fully scripted (no teleop, no per-trajectory human review).

The grasp source is **per-instance human annotation** (RoboTwin-style): each grasp is authored once
in a GUI and stored as an eef-target pose in the object-local frame (see §B).

> **Design in one line:** a *generic executor* plans / executes / gates / records / scales any
> family identically; each *family skeleton* only declares **which motion segments make up the
> task**. The two meet at one interface (`MotionSegment` + `FamilySkeleton`), so adding a family
> is "new task semantics, everything else reused."

---

## 0. The whole flow

```
   ┌─ one-time, per object set ──────────────────────────────────────────┐
A. extract_meshes      bench tasks ─► object-local GLB + bbox + gripper mesh   (1 OG session)
B. annotate_tool       viser GUI   ─► grasp_annotations.json (eef-target poses) (NO sim)
   fix_approach_tags / mesh_review  ─► tag cleanup + fast per-object review     (NO sim)
   validate_grasps     optional     ─► sim closeup that eef lands on each grasp (sim)
   └─────────────────────────────────────────────────────────────────────┘
   ┌─ collection, per base task (one OG process each) ───────────────────┐
D. driver.run_task     base task   ─► N success+safe demos (reset-reuse loop) (sim)
   (executor: grasp_db → grasp_select → variation → engine[+gate])
E. sweep               many tasks  ─► one subprocess per task, sharded/parallel
   └─────────────────────────────────────────────────────────────────────┘
   ┌─ offline, no sim ───────────────────────────────────────────────────┐
F. review              demos       ─► per-task montage MP4 (quick visual review)
G. to_lerobot          demos       ─► LeRobot v2.1 per family (offline; video passthrough)
   └─────────────────────────────────────────────────────────────────────┘
```

Environment for every step: `conda activate behavior`; sim steps also need
`OMNIGIBSON_HEADLESS=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` and run with
`python -u` (the `exit 139` segfault at `og.sim.stop()` is a benign teardown). All commands
assume `PYTHONPATH=$HOME/project/ManiGuard`.

**Prerequisites — install the bench tasks + robot asset first.** This pipeline collects
new demos *against ManiGuard-Bench tasks*, so it reads base tasks straight from a local copy
of the bench (e.g. `outputs/lerobot_datasets/maniguard-bench/<family>/task_NNNN/base/`, passed
as `--task-dir` in §D / discovered by `sweep` in §E), and every sim step loads the longfinger
Franka. Install both if you have not already (full details in
[Installation](../getting-started/installation.md)):

```bash
# bench tasks — the source scenes the executor replays
hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset \
  --local-dir outputs/lerobot_datasets/maniguard-bench
# robot asset — required by every sim step (auto-picked up on `import maniguard`)
hf download IDEAS-Lab-Northwestern/franka-panda-longfinger --repo-type dataset \
  --local-dir behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger
```

---

> The grasp-source stages (A–B + QC below) have a dedicated page with the full
> workflow, conventions, and gotchas: **[Grasp annotation](grasp_annotation.md)**.

## A. Mesh extraction — `annotation/extract_meshes.py`

**Why.** OmniGibson assets are *encrypted USD* — meshes can only be read through OG. We extract
each grasp target's mesh ONCE so the annotation tool (next stage) loads instantly with no sim.

**What it does.** Enumerates the distinct `(category, model)` grasp **targets** across the bench
families (pure JSON: each task's `diagnostics` goal target + `scene_ep1` model), then in **one OG
session** spawns every target `visual_only` in a grid, exports its **object-local** GLB, and
records `bbox_size` + the **upright world orientation** (how it stands in the scene). Also dumps
the longfinger gripper mesh in the **eef_link frame** (`--gripper`). Distractors are not extracted.

```bash
# meshes + metadata DB for one or all families
python -u -m maniguard.data.datagen.annotation.extract_meshes --families clutter_pickup
python -u -m maniguard.data.datagen.annotation.extract_meshes --families all
# the gripper mesh (once)
python -u -m maniguard.data.datagen.annotation.extract_meshes --gripper
```

Output: `outputs/grasp_annotation/{meshes/*.glb, gripper_longfinger.glb, mesh_db.json}`.

---

## B. Grasp annotation — `annotation/annotate_tool.py`

**The grasp source.** Each grasp is stored as an **eef_link TARGET pose** in the object-LOCAL
frame `(position, quat_xyzw)`. At runtime `T_eef_world = object_world_pose @ pose` feeds straight
to cuRobo IK — **zero conversion**. Verified exact (~1e-16) and object-relative (a grasp follows
the object's pose in any task instance).

**Gripper eef-local frame (sim-measured).** fingertips / **approach = eef +Z**, **closing
(between fingers) = eef −Y**. The annotation tool renders the real longfinger gripper at the draft
pose using this frame.

**The tool.** A [viser](https://viser.studio) web GUI on `:8080`. The object is shown **upright**
(its scene orientation) with the world XYZ axes and the gripper. Two input modes, both producing
the **same** stored record:
- **Guided** (default): click a surface point + pick an approach preset (top_down / side_*) +
  yaw/depth sliders. Fast for regular grasps.
- **Free** (toggle): a 6-DoF drag gizmo for odd semantic grasps (rim / handle / stem).

Incremental save + resume from `grasp_annotations.json`. `--family clutter` restricts the object
list to one family.

```bash
python -m maniguard.data.datagen.annotation.annotate_tool --family clutter   # browser :8080
```

Schema (`outputs/grasp_annotation/grasp_annotations.json`, gitignored — it is user data):

```json
{ "objects": { "<category>/<model>": {
    "bbox_size": [..], "upright_orientation_xyzw": [..], "mesh": "meshes/...glb",
    "grasps": [ { "id": 0, "position": [..], "orientation_xyzw": [..],
                  "approach_hint": "top_down|side", "label": "" } ] } } }
```

### Tag cleanup + fast review (NO sim)

- **`fix_approach_tags.py`** rewrites each grasp's `approach_hint` from its actual pose
  (`classify_approach`: top_down 0–60° / side 60–120° / bottom_up 120–180°; the 60° cutoff is a
  tunable param). Default dry-run; `--apply` writes back.
- **`mesh_review.py`** is the **default per-family review** — pure trimesh + matplotlib (the same
  geometry viser shows), so it renders in **seconds** with no sim. Per object: the upright object +
  gripper at each grasp from 3 viewpoints (oblique / side / top), a fixed-inset world-XYZ triad,
  and the **true** approach derived from the pose.

```bash
python -m maniguard.data.datagen.annotation.fix_approach_tags --family clutter --apply
python -m maniguard.data.datagen.annotation.mesh_review       --family clutter
# -> outputs/grasp_annotation/mesh_review/<cat__model>.png
```

### Optional sim validation — `annotation/validate_grasps.py`

Heavier confirmation, run later in a big time block: loads each grasp into its source task,
**teleports the robot base so eef_link lands exactly on the grasp (0.0 mm)**, hides the non-gripper
arm, and renders the 4 bench cameras + a matplotlib 3D closeup, plus a per-object `_summary.png`.
Run **one object per process** (`--object cat/model`) — `og.clear()` multi-task reload breaks the
cameras.

---

## D. The executor — one base task → N demos

`maniguard/data/datagen/{executor/, families/, grasp_db.py, driver.py}`. The decoupling:

| layer | files | role |
|---|---|---|
| **generic** (`executor/`) | `contracts` `engine` `variation` `grasp_select` `gate` `geometry` | plan / execute / gate / record / scale — family-agnostic |
| **specific** (`families/`) | `clutter.py` … | implement `FamilySkeleton` only: the motion segments |
| bridge | `grasp_db.py` | annotation DB → world eef poses |
| L3 | `driver.py` | orchestrate variants → engine → keep success+safe |

**`MotionSegment`** = `(name, eef_pos, eef_quat, mode{free|linear_servo}, attach, ignore_objects,
ignore_clutter, grip, grip_steps, min_clearance_m, target_clearance_m, compute, is_terminal)`. A
family's `derive_segments` returns an ordered list; the engine never knows which family produced it.
`compute` tags (`lift_to_clearance` / `over_goal` / `aim_to_goal_center`) are runtime-resolved by
the engine from the live post-grasp state via `geometry` (so skeletons stay pure functions).

**`FamilySkeleton`** (the only thing a family writes): `grasp_candidates(ctx)` (where grasps come
from — clutter looks up the annotation DB), `derive_segments(ctx, grasp, params)` (the boxy
waypoints), `success_extra(ctx)` (extra success terms; clutter none), `variation_knobs(ctx)`.

### How one trajectory is produced (engine loop)

For each `MotionSegment`: resolve any `compute` target from the live state → `solve_segment`
(cuRobo; LINEAR mode falls back to an unconstrained solve when this cuRobo build rejects the
partial-pose query) → `execute_trajectory` (JointController PD, recording every step) **with the
`SafetyGate` ticking each env step** → re-command the final waypoint to settle (PD undershoots
under load) → verify clearance. A demo is kept ONLY if it ends **held-in-goal AND was never
LTL-violated**.

### Grasp scoring (filter, not pick-one) — `grasp_select.py`

cuRobo scores each annotated grasp by **pre-grasp-standoff reachability**. Unreachable grasps are
dropped (`reachable=False`); **all reachable grasps are used** (round-robin), score only orders
which to try first. NB a grasp can score reachable yet still miss the goal in the full demo — the
gate filters those.

### Diversity / scaling — `variation.py`

One **master seed `variant_seed = grasp_id*1000 + draw`** per variant drives ALL its randomness, so
every variant differs. 5 dimensions:

| dim | how | draw 0 (canonical) |
|---|---|---|
| **grasp** | each reachable annotated grasp (round-robin) | — |
| **cuRobo trajopt seed** | engine `torch.manual_seed(seed)` → different joint solution for same waypoints | seeded |
| **lift height** | aim = `min_clearance × uniform(1.0, 1.5)`, always ≥3 cm floor | exactly 1.0× |
| **standoff** | pre-grasp standoff along the approach axis | base |
| **above_xy** | lateral offset of the pre-grasp point | none |

### Safety + success gate — `gate.py`

Built once per task (Spot init is not cheap). **Success** (eval-consistent): target AG-held AND
its AABB intersects the goal sphere. **LTL**: `utils.safety_monitor.TaskLTLMonitor` stepped every
executed env step (patterns rebuilt against the loaded scene, replicated from the eval runner). A
violation **instantly voids** the demo. We NEVER collect "success but not safe."

### Driver — `driver.py`

```bash
# one task, collect until 50 success+safe demos (preferred for collection)
python -u -m maniguard.data.datagen.driver \
    --task-dir outputs/lerobot_datasets/maniguard-bench/clutter_pickup/task_0000/base \
    --dataset v1 --target 50 --score
```

`--target N` keeps drawing variants until N successes (or `max_attempts`, default `N*4`). Between
EVERY variant the driver restores a pristine scene snapshot (`og.sim.load_state`) — each demo moves
the target, so without reset variant 2+ would start corrupted. Writes `_summary.json` per task.

---

## E. Sweep across tasks — `sweep.py`

Parallelism unit = the **task** (one fresh OG process each, because `og.clear()` can't switch tasks
in-process). A sweep runs its assigned tasks sequentially, one subprocess each, on one GPU.

```bash
# first 5 clutter tasks, 50 demos each, single GPU single process
python -u -m maniguard.data.datagen.sweep \
    --family clutter --dataset v1 --limit-tasks 5 --target 50 --gpu 0 --skip-existing
```

**Multi-GPU / parallel:** launch one sweep per shard in separate tmux sessions —
`--shard i --num-shards N --gpu G`. Shards own disjoint tasks (round-robin), so output dirs / logs
never collide. On a single GPU two shards still help: the workload is **latency / CPU-sync bound**
(GPU-Util reads ~100% but power ~35% of TDP and memory bandwidth ~3%), so a second process fills
the idle cycles — expect ~1.5–1.8×, with VRAM the limiting constraint (stagger launches).

**Save isolation (no two processes ever write the same file):** demos under
`task_NNNN/traj_NNN/` (a task is owned by one shard), per-task `_summary.json`, per-task log under
`_logs/<task>.log`.

---

## F. Review montage — `review.py`

Tiles every demo of a task into one MP4 for efficient review. Default = **`left_shoulder` only,
10 cells/row** (50 demos → 5 rows, one file); add wrist with `--wrist` (5 `left_shoulder|wrist`
pairs/row). Native 256² pixels (source is already low-res — no spatial downscale); size is
controlled by frame rate (`--stride 3` = 10 fps). All demos play in lockstep; a shorter clip
freezes on its last frame until the longest finishes. Streaming decode → ~10 MB memory even at full
res. Each group is labeled with its trajectory index; the montage is titled with the task.

```bash
python -m maniguard.data.datagen.review --dataset v1 --family clutter_pickup --all      # one MP4 per task
python -m maniguard.data.datagen.review --dataset v1 --family clutter_pickup --task task_0000 --wrist
# -> outputs/datagen/v1/clutter_pickup/<task>/_review_ls.mp4  (or _review_lsw[_pN].mp4)
```

---

## G. Reader + LeRobot conversion — `reader.py` → `to_lerobot.py`

Each demo stores videos and numbers **separately**: 5 MP4 streams (pixels) + `traj.hdf5` (numeric
only — no pixels) + `meta.json`. So **LeRobot conversion needs no sim and no replay** — it is a
pure offline file repackage: read frames from the existing MP4s + joint arrays from the hdf5 +
fields from meta, write into LeRobot v2.1 (parquet + per-camera video), **videos passed through with
no re-encode**. `reader.py` (`iter_traj_dirs` / `load_meta` / `load_traj` / `read_frames`) is the
single entry point the converter uses — keep the on-disk layout and the reader in sync.

The converter is **built and shipped** (`to_lerobot.py` serial + `to_lerobot_parallel.py`
shard-by-task, proven byte-identical, ~18.5×). It runs in the **lerobot uv env**, not `behavior`.
Full details, CLI, the byte-identity guarantee, and HF publishing are in
**[lerobot_conversion.md](lerobot_conversion.md)**.

---

## Data layout & schema

```
outputs/datagen/<dataset>/<bench_family>/<task>/traj_<NNN>/
    image_opposite.mp4  image_left.mp4  image_right.mp4  image_left_shoulder.mp4  wrist_image.mp4
    traj.hdf5      state (N,8)=[arm_q(7),gripper] · actions (N,8)=[arm_q[t+1],gripper_cmd]
                   actions_commanded (N,8)=[cuRobo cmd] · states (N,*)=sim-state dump (MimicGen hook)
    meta.json      family/source_task/target_key/grasp_id/approach/draw/standoff_m/
                   lift_clearance_mult/jitter/grasp_score/success/held_in_goal/ltl_violated/n_steps/prompt
    _summary.json  (per task) n_success / n_attempts / elapsed_s
```

`<bench_family>` matches the bench dataset dir name (`clutter_pickup`, …); `<task>` = `task_NNNN`;
`traj_NNN` is sequential over **kept** demos only (gap-free — failed demos drop their folder). Each
demo ≈ 1.1 MB (mostly the 5 videos; the hdf5 is ~120 KB). Naming and resolution (256², 30 fps,
h264/yuv420p) are byte-for-byte consistent with the bench rollout videos.

---

## Family-specific high-level plans

Only the **family skeleton** changes per family; everything above is shared. As each family's
pipeline is finalized it is documented here.

### clutter (`families/clutter.py`) — boxy top-down pick-and-place

`grasp(target) → transport(target → goal)`. The "boxy" segments (each = one `MotionSegment`):

```
pre_grasp   FREE,   open    standoff back along the grasp approach axis
descend     LINEAR, close   straight in to the annotated grasp pose, close (ignore_clutter:
                            world-collision-off for this short descent into the cluttered region)
lift        LINEAR, hold    straight up until the held object's lowest point clears the tallest
                            clutter by min_clearance (compute=lift_to_clearance)
transport   FREE,   hold    over the goal at the cleared height (compute=over_goal)
to_goal     LINEAR, hold    drive the held object's CENTRE to the goal-sphere centre
                            (compute=aim_to_goal_center) — overshoot past first contact for a robust endpoint
```

Why boxy: top-down grasps are kinematically reachable (~1 mm / 0.1° IK error); failures are purely
desk/clutter collision. "Lift to a safe height, then translate" structurally clears clutter, so
cuRobo solves it directly.

> **SERVO vs LINEAR.** Later families use `linear_servo` segments (labelled SERVO below) — pure
> straight-line joint IK per waypoint — instead of `linear`, because this cuRobo fork's partial-pose
> LINEAR_SERVO query is unreliable; SERVO gives the same straight motion without it.

### cabinet (`families/cabinet.py`, bench `cabinet_pickup`) — drawer place

`relocate blocker(s) → open sliding drawer → place target inside → close drawer`. Four phases, ~25
segments: **P1 relocate** (pick blocker, carry off, inverted return-replay), **P2 open** (grasp
handle, pull drawer, release), **P3 place** (grasp target, inverted-U over the rail, lower inside,
release), **P4 close** (grasp handle/front, push shut). Gate = `inside(target, cabinet) &
closed(cabinet)`; no `success_extra`. **Mechanic:** obstacle is relocated FIRST — opening the drawer
into an un-moved object tips it → LTL upright violation → demo voided.

### stack (`families/stack.py`, bench `stack_retrieve`) — unstack then retrieve

`unstack the 3 identical top objects onto one right-side re-stack pile → retrieve the exposed bottom
target left into the goal sphere (held)`. Per top object ×3: `s{i}_up/over/descend/lift/carry/
reorient/realign/place` (shallow FREE-descend grasp, double-gated transfer at a fixed safe height,
upright re-stack onto a growing pile); then the target tail `t_up/over/descend/lift/transport/place`.
**`success_extra`:** reject the demo if any pick moved >1 object (`not _multigrab`). **Mechanic:**
grasps filtered top-down-only; two unreachable tasks revert to a minimal re-stack gap.

### jar (`families/jar.py`, bench `jar_transport`) — close lid, transport jar

`close the hinged lid → grasp the jar body (side) → transport jar → goal`. **Phase A (close lid):**
`lid_under` (bar into the wedge under the lid) → `lid_slip` → `lid_ride` (straight ride past the
tipping point) → `lid_back` (retreat; gravity seats the lid). **Phase B:** `side_pre_grasp → descend
(close) → lift → transport → to_goal`. **Mechanic:** the **lid-ride** — the open gripper pivots the
articulated lid about its hinge (`arc_about_hinge` compute), the lid resting unilaterally on a finger
bar; Phase B is restricted to SIDE grasps so the jar stays upright off the just-closed lid.

### dusty (`families/dusty.py`, bench `dusty_transfer`) — wipe then pour

`pick+place sponge → wipe the dest's dust → return the sponge → pick the source (food riding inside)
→ tilt-pour the food into the dest`. Segments: sponge P&P + `wipe_step_{i}` peck-wipe tour →
sponge return → source grasp + `pour_step_{i}` tilt ramp → `pour_settle`. **`success_extra`:** the
dest must end **upright** (`up_z ≥ cos20°`) AND not displaced (>0.15 m → fail). **Family-hard gates:**
dust must be 100% removed before pouring (`wipe_incomplete`), any finger brushing the food fails
(`food_touched_by_agent`), no drop during pour (`pour_no_drop`); the episode ends in the tilted pour
pose (teleop termination — no untilt/place-back). **Mechanic:** peck-wipe hops laterally *in air* so
the sponge never friction-drags the dest; a sticky-grasp guard chain protects the food mid-grab.

### lid (`families/lid.py`, bench `lid_transport`) — assemble then transport

`pick(lid) → place onto the container's F meta-link → release → LidSnapper auto-welds → pick the
container (or rim+lid sandwich) → transport → END HOLDING inside the goal sphere` (no place-down;
teleop termination). Segments: lid place (`lid_pre/descend/lift/transit/place/align_verify/release/
retreat/snap_verify`) → container transport (`cont_pre/descend/lift/goal_over/to_goal`).
**`success_extra`:** the lid is still `AttachedTo` the container at the end. **LTL:**
`(container_on_support U lid_on_container) & G(!container_dropped)`. **Mechanic:** three geometry
classes — **plain** (top-knob vertical descend), **handle** (kettle/teapot: side grasp + horizontal
insertion under the handle arch + replay-reverse exit), **cap** (small cap onto a tall mouth); a
pre-release M↔F alignment gate rejects crooked drops; fat containers (>32 collision prims) carry
with `attach=False`.

---

## Gotchas (hard-won)

- **LINEAR_SERVO rejected by this cuRobo build** → the engine falls back to an unconstrained solve
  for LINEAR-mode segments (`grasp.py:140-147` pattern).
- **Grasp descent blocked by collision** → `MotionSegment.ignore_clutter` drops every non-robot
  obstacle for that short controlled descent; safety = physics + AG + the LTL gate.
- **Lift undershoots clearance** (PD steady-state droop under load) → re-command the final waypoint
  to settle + over-lift by a margin; verify uses the real ≥3 cm floor.
- **Variant cross-contamination** → restore the pristine scene snapshot before every variant.
- **Nested meta in h5py attrs** ("Object dtype has no native HDF5 equivalent") → JSON-encode
  non-scalar attrs; keep real nested dicts in `meta.json`.
- **Multi-task in one process** → `og.clear()` breaks the cameras; always one task per subprocess.
- **`og.sim.stop()` exit 139** is a benign teardown segfault; always run with `python -u`.
```
