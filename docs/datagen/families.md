# Families & gotchas

Only the **family skeleton** changes per family; the executor, gates, variation, and
conversion are shared. Each family's task semantics:

## Family-specific plans

??? note "▸ clutter — boxy top-down pick-and-place"
    `grasp(target) → transport(target → goal)`. Boxy segments: `pre_grasp` (FREE, standoff
    back along the approach axis) → `descend` (LINEAR, close; `ignore_clutter` for the short
    descent) → `lift` (straight up until the held object clears the tallest clutter) →
    `transport` (FREE, over the goal at height) → `to_goal` (drive the object's centre to the
    goal-sphere centre, overshoot past first contact). Why boxy: top-down grasps are
    kinematically reachable (~1 mm IK error); "lift then translate" structurally clears clutter.

    > **SERVO vs LINEAR.** Later families use `linear_servo` segments (straight-line joint IK
    > per waypoint) instead of `linear`, because this cuRobo fork's partial-pose LINEAR_SERVO
    > query is unreliable; SERVO gives the same straight motion without it.

??? note "▸ cabinet (`cabinet_pickup`) — drawer place"
    `relocate blocker(s) → open drawer → place target inside → close drawer`. Four phases,
    ~25 segments: **P1 relocate** (pick blocker, carry off, inverted return-replay), **P2
    open** (grasp handle, pull, release), **P3 place** (grasp target, inverted-U over the
    rail, lower inside, release), **P4 close** (push shut). Gate = `inside(target,cabinet) &
    closed(cabinet)`. **Mechanic:** the obstacle is relocated FIRST — opening into an un-moved
    object tips it → LTL upright violation → demo voided.

??? note "▸ stack (`stack_retrieve`) — unstack then retrieve"
    `unstack the 3 identical top objects onto one right-side re-stack pile → retrieve the
    exposed bottom target into the goal (held)`. Per top object ×3: shallow FREE-descend grasp,
    double-gated transfer at a fixed safe height, upright re-stack onto a growing pile; then the
    target tail. **`success_extra`:** reject if any pick moved >1 object. **Mechanic:** grasps
    filtered top-down-only; two unreachable tasks revert to a minimal re-stack gap.

??? note "▸ jar (`jar_transport`) — close lid, transport jar"
    `close the hinged lid → grasp the jar body (side) → transport → goal`. **Phase A:**
    `lid_under → lid_slip → lid_ride → lid_back`. **Phase B:** `side_pre_grasp → descend →
    lift → transport → to_goal`. **Mechanic:** the **lid-ride** — the open gripper pivots the
    articulated lid about its hinge, the lid resting unilaterally on a finger bar; Phase B is
    restricted to SIDE grasps so the jar stays upright off the just-closed lid.

??? note "▸ dusty (`dusty_transfer`) — wipe then pour"
    `pick+place sponge → wipe the dest's dust → return sponge → pick source (food inside) →
    tilt-pour the food into the dest`. **`success_extra`:** dest ends **upright** and not
    displaced. **Family-hard gates:** dust 100% removed before pouring, no finger brushing the
    food, no drop during pour; episode ends in the tilted pour pose. **Mechanic:** peck-wipe
    hops laterally *in air* so the sponge never friction-drags the dest.

??? note "▸ lid (`lid_transport`) — assemble then transport"
    `pick(lid) → place onto the container's F meta-link → LidSnapper auto-welds → pick the
    container → transport → END HOLDING inside the goal`. **`success_extra`:** the lid is still
    `AttachedTo` the container at the end. **Mechanic:** three geometry classes — **plain**
    (top-knob vertical descend), **handle** (side grasp + horizontal insertion under the arch +
    replay-reverse exit), **cap** (small cap onto a tall mouth); a pre-release M↔F alignment
    gate rejects crooked drops.

## Gotchas

- **LINEAR_SERVO rejected by this cuRobo build** → the engine falls back to an
  unconstrained solve for LINEAR-mode segments.
- **Grasp descent blocked by collision** → `MotionSegment.ignore_clutter` drops every
  non-robot obstacle for that short controlled descent; safety = physics + AG + the LTL gate.
- **Lift undershoots clearance** (PD steady-state droop under load) → re-command the final
  waypoint to settle + over-lift by a margin; verify uses the real ≥3 cm floor.
- **Variant cross-contamination** → restore the pristine scene snapshot before every variant.
- **Nested meta in h5py attrs** → JSON-encode non-scalar attrs; keep nested dicts in `meta.json`.
- **Multi-task in one process** → `og.clear()` breaks the cameras; always one task per subprocess.
- **`og.sim.stop()` exit 139** is a benign teardown segfault; always run with `python -u`.
