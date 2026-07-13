# Reach-aware intersection-relaxation fallback (far-goal clutter transport) — design

**Date:** 2026-07-01 (supersedes the 2026-06-30 "yaw-free rotate-in-place" approach, which targeted the
wrong quantity — see Root Cause)
**Status:** Phase 1 IMPLEMENTED + validated
**Scope:** `maniguard/data/datagen` — clutter family transport terminal placement

## Progress (2026-07-01)

**Phase 1 (pull-back translation) stably solves the problem — shipped.** Validated locally: saucepan
(task_0030) 0/8 → 2/2 and mixing_bowl (task_0032) 0/8 → 2/2 (both via the pull-back fallback, k=4–22 cm),
mug (task_0034) 2/2 with NO fallback (primary path, zero regression). A 6-task stability run (3 dry + 3
liquid) came back 6/6 at 2/2, the fallback firing only on the far-goal liquid tasks (saucepan, gravy_boat).

**Phase 2 (world-Z yaw) is deferred / not built.** Pull-back alone was sufficient for every task tested,
so the yaw scaffolding (`executor/yaw_search.py`, `geometry.yaw_rotated_quat`, and their unit tests) was
removed to keep the shipped code focused (YAGNI). If a later liquid task turns out to have a goal beyond
the pull-back budget (`k_max = radius + object_forward_extent`), revisit Phase 2 — the design below still
stands and the helpers can be re-added.

## Root cause (established by instrumented IK probes on task_0030 saucepan)

The clutter task **succeeds when the held target's AABB intersects the goal sphere** (`goal_checker` →
`object_intersects_goal_region`: closest AABB point to `center_world` within `radius_m`, measured
`radius = 0.09 m`). It does **not** require the eef, nor the object centre, to be at the goal centre.

But the pipeline **over-constrains** the terminal placement:
- `transport` (`compute="over_goal"`) drives the **eef** to `goal_xy` at lift height.
- `to_goal` (`compute="aim_to_goal_center"`) drives the **object centre** exactly onto the goal centre.

For a far goal this is infeasible even though a valid placement exists:
- `base→goal` = **0.90 m** (Franka wrist-down reach limit ≈ 0.82 m). Aiming the eef AT the goal → IK_FAIL.
- Measured: pulling the eef back just **8 cm** (to 0.82 m) makes wrist-down **reachable** (15/20/30 cm too);
  CONTROL IK at the current eef pose and a midpoint both solve, so the probe is trustworthy.
- At an 8 cm pull-back the held object still pokes the 0.09 m sphere (object forward-extent + radius ≫ 8 cm),
  so it still **succeeds**.

So the goal is reachable; the pipeline just demands a needlessly precise centre-to-centre placement.

(The earlier "yaw rotate-in-place" idea failed because it kept the eef pinned AT `goal_xy` and only spun
the wrist — for a top-down grasp that is joint-7 roll, which cannot change reach. Yaw only helps once the
target is the *object*, not the eef; see Phase 2.)

## Goal

Keep the precise centre-to-centre placement as the primary path (unchanged). When it cannot be planned,
fall back to the **closest-to-goal eef placement that is IK-reachable AND keeps the held object
intersecting the goal sphere** — i.e. relax "object centre on goal centre" to "object touches the goal
region", pulling the eef back toward the robot into its reach envelope.

**Hard constraints:** (1) no regression — the fallback fires ONLY after the primary plan fails, so close
goals are byte-identical; (2) upright preserved (liquid: no spill) — the fallback keeps the carry
orientation, only translates (Phase 1) or adds a world-Z yaw (Phase 2, upright-preserving); (3) the
accepted placement must actually intersect the sphere (verified, not assumed).

## Approach

### Phase 1 — pull-back translation (primary fix; no yaw)

Trigger: the `transport` segment's normal plan fails (`res is None`) on a `reach_fallback` segment.

1. `dir = unit(goal_xy − base_xy)`; `r = goal_radius`.
2. Sweep the terminal eef target back toward the robot along `dir`, at (near) goal height:
   `target_k = (goal_xy − k·dir, goal_z)` for `k ∈ {0.04, 0.08, 0.12, …, k_max}`.
   `k_max = r + object_forward_extent` (how far the held object's AABB reaches past the eef toward the
   goal — beyond this the object leaves the sphere).
3. For each `k` (smallest first = object most centred): plan `solve_segment(target_k, q_live)` from the
   lifted state; if it plans AND the held object's AABB (current AABB rigidly translated by
   `target_k − eef_now`) still intersects the sphere → accept. This placement is the terminal (object in
   sphere ⇒ success); it replaces the failed `transport`+`to_goal`.
4. If no `k` within `k_max` is reachable → Phase 2 (or genuine failure).

### Phase 2 — add world-Z yaw (only if Phase 1 exhausts)

Orient the object so its long axis points toward the goal, increasing `object_forward_extent` (hence
`k_max` and the pull-back budget). Reuse `geometry.yaw_rotated_quat` + `yaw_search` over `q_live`; for
each yaw recompute the forward extent and re-run the Phase-1 sweep. World-Z yaw preserves upright (no
spill) for any grasp. Deferred unless Phase 1 proves insufficient on some task.

## Files touched

- `executor/contracts.py` — repurpose the `MotionSegment.yaw_free` flag → `reach_fallback: bool`
  (clearer name for the generalized fallback).
- `families/clutter.py` — set `reach_fallback=True` on `transport` (the segment that owns the terminal
  reach); the trailing `to_goal` is skipped when the fallback fired (it already placed the object).
- `executor/geometry.py` — keep `yaw_rotated_quat`; add `object_forward_extent(obj, eef_pos, dir)` and a
  small AABB-translate intersection check helper (or reuse `utils.goal_region`).
- `executor/engine.py` — replace `_yaw_search_transport` with `_reach_fallback_transport` (Phase-1 sweep,
  optional Phase-2 yaw); **remove all `YAW_DEBUG` instrumentation** added during debugging.
- `executor/yaw_search.py` — keep (used by Phase 2); Phase-1 needs only the pull-back sweep.
- `curobo_seg.py` — unchanged.

## Testing

- **Rescue:** saucepan (task_0030) + mixing_bowl (task_0032) → `n_success > 0`, with a
  `[reach fallback] pull-back k=…` log and a final state where the object intersects the sphere.
- **Regression:** mug (task_0034) + a dry task → still ~100%, no fallback log (primary plan succeeds).
- **Success integrity:** the recorded demo must pass the SAME `object_intersects_goal_region` success
  check the eval/bench use (shared module) — the pulled-back placement is a real success, not a relaxed
  datagen-only one.
- Local, ≤ 2 sim processes.

## Out of scope / not changing

- Bench task definition (goal position, robot pose, spill threshold) — untouched.
- The lift height — verified legit (a real 15.8 cm on-table transparent cocktail-glass forces it; lowering
  it would not help, since the blocker is horizontal reach at any height).
