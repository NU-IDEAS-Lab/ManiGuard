# Reach-aware intersection-relaxation fallback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** When the clutter terminal placement (eef→goal / object-centre→goal-centre) cannot be planned
for a far goal, fall back to the closest-to-goal eef placement that is IK-reachable AND keeps the held
object intersecting the goal sphere — pulling the eef back toward the robot (Phase 1), adding an
upright-preserving world-Z yaw only if pull-back alone is insufficient (Phase 2).

**Architecture:** `MotionSegment.reach_fallback` on the `transport` segment. When its normal plan fails,
`engine._reach_fallback_transport` sweeps the terminal eef target back along `base→goal` (at goal height),
plans each, and accepts the first that is reachable AND leaves the object∩sphere; that placement replaces
`transport`+`to_goal` (the trailing `to_goal` is skipped). Primary path unchanged → no regression.

> **Status (2026-07-01): Phase 1 SHIPPED + validated (see spec Progress).** Pull-back alone solved every
> task tested (saucepan/mixing_bowl rescued, mug unchanged, 6/6 stability run at 2/2). **Phase 2 (yaw) is
> deferred and NOT built** — its scaffolding (`yaw_search.py`, `yaw_rotated_quat`, their tests, and the
> yaw Tasks below) was removed for YAGNI. Revisit Phase 2 only if a task's goal exceeds the pull-back
> budget. The Phase-2 task steps below are kept for reference should that day come.

**Tech stack:** Python, numpy, scipy, cuRobo (`solve_segment`, unchanged), OmniGibson (sim validation), pytest.

## Global Constraints

- Reuse `solve_segment`; do NOT modify `curobo_seg.py`.
- Fallback fires ONLY after the primary transport plan fails → close goals byte-identical.
- Accepted placement MUST intersect the sphere (verified via the SHARED `object_intersects_goal_region`
  logic on the rigidly-translated object AABB), so the demo is a real success under the eval/bench checker.
- Phase 1 keeps the carry orientation (translate only). Phase 2 world-Z yaw preserves upright (no spill).
- **Remove all `YAW_DEBUG` instrumentation** added during debugging.
- Commits batched at the end. Sim validation local, ≤ 2 processes.

## File structure

- `executor/contracts.py` — rename `MotionSegment.yaw_free` → `reach_fallback: bool = False`.
- `families/clutter.py` — `reach_fallback=True` on `transport` (was `yaw_free`).
- `executor/geometry.py` — keep `yaw_rotated_quat`; add `object_forward_extent(obj, eef_pos, direction)`
  and `aabb_intersects_sphere(obj, center, radius, translate)` (rigidly-translated AABB ∩ sphere).
- `executor/engine.py` — replace `_yaw_search_transport` with `_reach_fallback_transport`; add
  `_skip_to_goal` handling; delete the `YAW_DEBUG` block + IK sweep.
- `executor/yaw_search.py` — keep for Phase 2.
- Tests: update `tests/datagen/test_clutter_yaw_free.py` → `test_clutter_reach_fallback.py`; add
  `tests/datagen/test_reach_geometry.py` (forward extent + translated-AABB intersection, pure).

---

### Task 1: geometry — forward extent + translated-AABB∩sphere (pure, TDD)

**Files:** modify `executor/geometry.py`; test `tests/datagen/test_reach_geometry.py`.

**Interfaces:**
- `object_forward_extent(aabb_lo, aabb_hi, eef_pos, direction) -> float` — max signed projection of
  `(AABB corner − eef_pos)` onto unit `direction` (how far the object reaches past the eef toward the goal).
- `aabb_sphere_hit(aabb_lo, aabb_hi, center, radius, offset=(0,0,0)) -> bool` — does the AABB, rigidly
  shifted by `offset`, come within `radius` of `center` (closest-point test, same math as
  `utils.goal_region.object_intersects_goal_region`).

- [ ] **Step 1: failing test** — `test_reach_geometry.py`:
```python
import numpy as np
from maniguard.data.datagen.executor.geometry import object_forward_extent, aabb_sphere_hit

def test_forward_extent_points_along_direction():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.1, 0.1])
    eef = np.array([0.0, 0.05, 0.05])
    # object extends +0.2 along +x from the eef
    assert np.isclose(object_forward_extent(lo, hi, eef, np.array([1.0, 0, 0])), 0.2, atol=1e-6)
    assert np.isclose(object_forward_extent(lo, hi, eef, np.array([-1.0, 0, 0])), 0.0, atol=1e-6)

def test_aabb_sphere_hit_with_offset():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])
    c = np.array([0.3, 0.05, 0.05])
    assert not aabb_sphere_hit(lo, hi, c, 0.05)                       # gap 0.2-0.05... far -> miss
    assert aabb_sphere_hit(lo, hi, c, 0.05, offset=np.array([0.2, 0, 0]))  # shift AABB +0.2 -> touches
```
- [ ] **Step 2: run → fail** (`python -m pytest tests/datagen/test_reach_geometry.py -v`).
- [ ] **Step 3: implement** in `geometry.py`:
```python
def object_forward_extent(aabb_lo, aabb_hi, eef_pos, direction) -> float:
    """How far the AABB reaches past ``eef_pos`` along unit ``direction`` (max corner projection, >=0)."""
    lo = _np(aabb_lo); hi = _np(aabb_hi); e = _np(eef_pos); d = _np(direction)
    d = d / (np.linalg.norm(d) + 1e-9)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return float(max(0.0, np.max((corners - e) @ d)))

def aabb_sphere_hit(aabb_lo, aabb_hi, center, radius, offset=(0.0, 0.0, 0.0)) -> bool:
    """True if the AABB rigidly shifted by ``offset`` comes within ``radius`` of ``center``."""
    lo = _np(aabb_lo) + _np(offset); hi = _np(aabb_hi) + _np(offset); c = _np(center)
    closest = np.minimum(np.maximum(c, lo), hi)
    return bool(float(np.dot(c - closest, c - closest)) <= float(radius) * float(radius))
```
- [ ] **Step 4: run → pass.**

---

### Task 2: rename `yaw_free` → `reach_fallback` (contracts + clutter, TDD)

**Files:** `executor/contracts.py`, `families/clutter.py`; rename test → `tests/datagen/test_clutter_reach_fallback.py`.

- [ ] **Step 1:** rewrite the test (copy of the yaw_free test, asserting `transport.reach_fallback is True`,
      others False). Delete `tests/datagen/test_clutter_yaw_free.py`.
- [ ] **Step 2: run → fail** (no `reach_fallback` attr).
- [ ] **Step 3:** in `contracts.py` rename the field `yaw_free` → `reach_fallback` (keep the default
      `False`, update the comment to describe the pull-back+yaw fallback). In `clutter.py` change the
      transport kwarg `yaw_free=True` → `reach_fallback=True`.
- [ ] **Step 4: run → pass.**

---

### Task 3: engine — `_reach_fallback_transport` + skip `to_goal` (integration)

**Files:** `executor/engine.py`. Validated by sim (Task 4).

- [ ] **Step 1: delete the `YAW_DEBUG` block and the IK-sweep** inside `_yaw_search_transport`, and the
      `diagnose_on_fail=dbg` bits. (Everything gated on `os.environ.get("YAW_DEBUG")`.)
- [ ] **Step 2: replace `_yaw_search_transport` with `_reach_fallback_transport`:**
```python
    def _reach_fallback_transport(self, seg, ctx, q_full, pos_t, quat_t, attach):
        """Terminal-placement fallback when the precise transport plan fails on a far goal: keep the
        carry orientation, pull the eef target back toward the robot along base->goal (at goal height),
        and accept the first placement that is IK-reachable AND still leaves the held object intersecting
        the goal sphere (relaxes object-centre-on-goal-centre to object-touches-goal). Returns a
        SegmentResult (this placement is the terminal; the caller skips to_goal), or None."""
        import torch as th
        arm = self.robot.default_arm
        eef_now = _np(self.robot.eef_links[arm].get_position_orientation()[0])
        base = _np(self.robot.get_position_orientation()[0])
        g = _np(ctx.goal_center); r = float(ctx.goal_radius)
        held = self._held(seg, ctx)
        lo, hi = geometry.aabb_lo_hi(held)
        d = (g - base); d[2] = 0.0; d = d / (np.linalg.norm(d) + 1e-9)   # base->goal, horizontal
        fwd = geometry.object_forward_extent(lo, hi, eef_now, d)          # object reach toward goal
        k_max = r + fwd + 0.02                                            # +2cm slack; beyond -> leaves sphere
        q = _np(quat_t)
        for k in np.arange(0.02, k_max + 1e-6, 0.02):
            tgt = np.array([g[0] - k * d[0], g[1] - k * d[1], g[2]])      # pulled back, at goal height
            offset = tgt - eef_now
            if not geometry.aabb_sphere_hit(lo, hi, g, r, offset=offset):
                break                                                    # pulled too far, object left sphere
            res = solve_segment(self.world.motion_gen, self.robot,
                                th.as_tensor(tgt, dtype=th.float32),
                                th.as_tensor(q, dtype=th.float32), q_full,
                                timeout=self.timeout, attach_obj=attach,
                                label=f"{seg.name}:pullback{int(k * 100)}")
            if res is not None:
                print(f"[datagen.engine] {seg.name}: reach fallback pull-back k={k * 100:.0f}cm "
                      f"(object still in {r * 100:.0f}cm goal sphere)", flush=True)
                self._skip_to_goal = True
                return res
        # Phase 2 (yaw) deferred — orient object toward goal to grow k_max. Not implemented yet.
        print(f"[datagen.engine] {seg.name}: reach fallback exhausted (k_max={k_max * 100:.0f}cm)", flush=True)
        return None
```
- [ ] **Step 3: update the trigger** in `run()` (the `res is None and seg.yaw_free` block → `reach_fallback`
      + call `_reach_fallback_transport`). Initialise `self._skip_to_goal = False` at the top of `run()`
      (next to `self._last_servo = None`). In the segment loop, right after `for seg in segments:` /
      before processing, add: `if seg.compute == "aim_to_goal_center" and self._skip_to_goal: continue`.
- [ ] **Step 4: import-smoke** `python -c "import maniguard.data.datagen.executor.engine"`; pure suite green.

Note: confirm `ctx.goal_radius` is populated for clutter (driver sets it from `spec.radius_m`) and
`self._held(seg, ctx)` returns the target — both already used in `run_task`/engine.

---

### Task 4: Sim validation (rescue + regression), local ≤2 procs

- [ ] **Step 1: rescue saucepan** `task_0030` (target 2, max-attempts 8, KEEP_FAILED): expect
      `n_success ≥ 1` + a `reach fallback pull-back k=…` log; a kept demo whose final state has the object
      intersecting the sphere.
- [ ] **Step 2: rescue mixing_bowl** `task_0032`: expect `n_success ≥ 1`.
- [ ] **Step 3: regression mug** `task_0034`: `n_success == 2`, NO fallback log (primary plan solved).
- [ ] **Step 4: inspect a rescued video** — object carried to the goal, poking the region, upright.
- [ ] **Step 5: clean throwaway datasets.**

---

### Task 5: batch commit (yaw+reach fix + spec/plan; GPU-dyn fix separate)

- [ ] pytest (reach_geometry + reach_fallback + yaw_rotated_quat + yaw_search) green; ruff; show diff;
      on approval commit reach-fallback fix as one commit (Co-Authored-By only) and the GPU-dynamics fix
      (scene.py+driver.py) as a separate commit.

## Self-Review

Root cause (over-constrained centre-to-centre; success = object∩sphere) → Task 2/3; reachability via
pull-back → Task 3 Phase 1; intersection integrity → Task 1 (`aabb_sphere_hit`, shared math) + Task 3
gate; no-regression → Task 3 (fallback only on primary fail) + Task 4 Step 3; remove debug → Task 3 Step 1;
yaw Phase 2 explicitly deferred (helpers kept). Types: `object_forward_extent`/`aabb_sphere_hit` signatures
match Task-1 defs and Task-3 calls.
