# Collection (steps 3–4)

With grasps annotated, collection is fully autonomous: the **executor** turns one base
task into N success+safe demos; **sweep** scales that across tasks / GPUs.

## 3. The executor — one base task → N demos

`maniguard/data/datagen/{executor/, families/, grasp_db.py, driver.py}`. The whole system
is one decoupling: a **generic** engine that never knows the task, and a tiny **family
skeleton** that only declares the motion segments.

| layer | files | role |
|---|---|---|
| **generic** (`executor/`) | `contracts` `engine` `variation` `grasp_select` `gate` `geometry` | plan / execute / gate / record / scale — family-agnostic |
| **specific** (`families/`) | `clutter.py` … | implement `FamilySkeleton` only: the motion segments |
| bridge | `grasp_db.py` | annotation DB → world eef poses |
| driver | `driver.py` | orchestrate variants → engine → keep success+safe |

A family's `derive_segments` returns an ordered list of **`MotionSegment`**s (waypoint +
mode + grip + clearance + a runtime-resolved `compute` tag); the engine executes them
identically for every family.

```bash
# one task, collect until 50 success+safe demos (preferred for collection)
python -u -m maniguard.data.datagen.driver \
    --task-dir outputs/lerobot_datasets/maniguard-bench/clutter_pickup/task_0000/base \
    --dataset v1 --target 50 --score
```

`--target N` keeps drawing variants until N successes (or `max_attempts`, default `N*4`),
restoring a pristine scene snapshot between **every** variant (each demo moves the target,
so without reset variant 2+ starts corrupted).

??? note "▸ MotionSegment & FamilySkeleton"
    **`MotionSegment`** = `(name, eef_pos, eef_quat, mode{free|linear_servo}, attach,
    ignore_objects, ignore_clutter, grip, grip_steps, min_clearance_m, target_clearance_m,
    compute, is_terminal)`. `compute` tags (`lift_to_clearance` / `over_goal` /
    `aim_to_goal_center`) are runtime-resolved by the engine from the live post-grasp state
    via `geometry`, so skeletons stay pure functions.

    **`FamilySkeleton`** (the only thing a family writes): `grasp_candidates(ctx)` (where
    grasps come from — clutter looks up the annotation DB), `derive_segments(ctx, grasp,
    params)` (the boxy waypoints), `success_extra(ctx)`, `variation_knobs(ctx)`.

??? note "▸ How one trajectory is produced (engine loop)"
    For each `MotionSegment`: resolve any `compute` target from the live state →
    `solve_segment` (cuRobo; LINEAR mode falls back to an unconstrained solve when this
    cuRobo build rejects the partial-pose query) → `execute_trajectory` (JointController PD,
    recording every step) **with the `SafetyGate` ticking each env step** → re-command the
    final waypoint to settle (PD undershoots under load) → verify clearance. A demo is kept
    ONLY if it ends **held-in-goal AND was never LTL-violated**.

??? note "▸ Grasp scoring — filter, not pick-one (`grasp_select.py`)"
    cuRobo scores each annotated grasp by **pre-grasp-standoff reachability**. Unreachable
    grasps are dropped; **all reachable grasps are used** (round-robin), score only orders
    which to try first. A grasp can score reachable yet still miss the goal in the full demo
    — the gate filters those.

??? note "▸ Diversity / scaling (`variation.py`)"
    One master seed per variant — `SeedSequence([grasp_id, draw])` — drives ALL its
    randomness, so every variant differs. 5 dimensions:

    | dim | how | draw 0 (canonical) |
    |---|---|---|
    | **grasp** | each reachable annotated grasp (round-robin) | — |
    | **cuRobo trajopt seed** | engine `torch.manual_seed(seed)` → different joint solution | seeded |
    | **lift height** | aim = `min_clearance × uniform(1.0, 1.5)`, always ≥3 cm floor | exactly 1.0× |
    | **standoff** | pre-grasp standoff along the approach axis | base |
    | **above_xy** | lateral offset of the pre-grasp point | none |

??? note "▸ Safety + success gate (`gate.py`)"
    Built once per task (Spot init is not cheap). **Success** (eval-consistent): target
    AG-held AND its AABB intersects the goal sphere. **LTL**:
    `utils.safety_monitor.TaskLTLMonitor` stepped every executed env step (patterns rebuilt
    against the loaded scene, replicated from the eval runner). A violation **instantly
    voids** the demo — we never collect "success but not safe."

## 4. Sweep across tasks

The parallelism unit is the **task** — one fresh OG process each, because `og.clear()`
can't switch tasks in-process. `sweep.py` runs its assigned tasks sequentially, one
subprocess each, on one GPU.

```bash
# first 5 clutter tasks, 50 demos each, single GPU single process
python -u -m maniguard.data.datagen.sweep \
    --family clutter --dataset v1 --limit-tasks 5 --target 50 --gpu 0 --skip-existing
```

**Multi-GPU / parallel:** launch one sweep per shard in separate tmux sessions —
`--shard i --num-shards N --gpu G`. Shards own disjoint tasks (round-robin), so output
dirs / logs never collide.

??? note "▸ Throughput & save isolation"
    On a single GPU two shards still help: the workload is **latency / CPU-sync bound**
    (GPU-Util ~100% but power ~35% of TDP, memory bandwidth ~3%), so a second process fills
    the idle cycles — expect ~1.5–1.8×, with VRAM the limiting constraint (stagger launches).
    No two processes ever write the same file: demos under `task_NNNN/traj_NNN/` (a task is
    owned by one shard), per-task `_summary.json`, per-task log under `_logs/<task>.log`.

**Next step →** [Review & conversion](conversion.md)
