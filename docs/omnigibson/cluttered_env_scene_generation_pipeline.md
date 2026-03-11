# MVP Cluttered Scene Generation Pipeline

## 1. Architecture Overview

```
BDDL Definition → Task Spec Parsing → Zone Computation → Object Classification → Clutter Pack Generation → Robot Placement → Gate Validation
```

---

## 2. Stage 1: Zone Computation (`kitchen_bar_workspace.py`)

**Core function**: `compute_kitchen_bar_zone()`

This stage computes the feasible placement region for the entire scene.

```python
# Inputs
bar_bounds_xy    # AABB boundary of the bar countertop
sink_bounds_xy   # AABB boundary of the sink

# Output: KitchenBarZoneSpec
- bar_bounds          # Full bar countertop boundary
- sink_keepout_bounds # Expanded sink exclusion zone
- red_zone_bounds     # Actual object placement region (core output)
- long_axis           # Long axis direction ("x" or "y")
```

**Zone computation logic**:

```
1. sink_keepout = sink_bounds + sink_keepout_margin_m (default 0.08–0.10 m)
2. red_zone = bar_bounds - edge_margin_m - sink_keepout - sink_side_clearance_m
```

**Key parameters** (from `CLUTTER_DENSITY_PRESETS`):

| Parameter | low | medium | high | ultra | Description |
|-----------|-----|--------|------|-------|-------------|
| `zone_edge_margin_m` | 0.05 | 0.05 | 0.04 | 0.03 | Distance from bar edge to red zone boundary |
| `sink_keepout_margin_m` | 0.10 | 0.10 | 0.08 | 0.06 | Expansion margin around the sink exclusion zone |
| `sink_side_clearance_m` | 0.02 | 0.02 | 0.015 | 0.01 | Gap between sink keepout boundary and red zone |

---

## 3. Stage 2: Object Classification (`manipulation_task_spec.py`)

**Core functions**: `build_manipulation_task_spec()` + `_build_task_object_sets()`

Parses the BDDL definition and classifies all objects into four roles:

```python
# Inferred from BDDL goal predicates
target_ids   # Objects to be manipulated (e.g., coffee_cup)
fragile_ids  # Breakable objects (queried via ObjectTaxonomy "breakable" ability)
support_ids  # Support/container objects (e.g., countertop, cabinet)

# Remaining objects
clutter_ids  # All objects not assigned to the above roles
```

**BDDL parsing example**:

```lisp
(:objects
    coffee_cup.n.01_1 - coffee_cup.n.01        ; → target (goal requires it inside cabinet)
    wineglass.n.01_1 ... wineglass.n.01_6      ; → fragile (breakable)
    plate.n.04_1 ... plate.n.04_4              ; → clutter
    countertop.n.01_1 - countertop.n.01        ; → support
    cabinet.n.01_1 - cabinet.n.01              ; → support
)
```

---

## 4. Stage 3: Clutter Pack Generation (`clutter_pack_layout.py`)

**Core function**: `build_clutter_pack()`

This is the central object layout algorithm, using **Frontier-based Greedy Packing**.

### 4.1 Placement Order

```python
placement_order = target_descriptors + non_target_descriptors
# Targets are placed first (near center), then fragile objects, then clutter.
# Non-target objects are shuffled to introduce randomness.
```

### 4.2 Target Placement (Center-first)

```python
# Target is forced near the pack center within the jitter range
cx = rng.uniform(-min(jitter_xy, 0.02), min(jitter_xy, 0.02))
cy = rng.uniform(-min(jitter_xy, 0.02), min(jitter_xy, 0.02))
```

### 4.3 Remaining Object Placement (Frontier Algorithm)

```python
def _frontier_candidate_pool():
    # 1. Generate dense grid points sorted by distance from center
    sorted_points = _generate_sorted_grid_points(bounds, step=grid_step_m)

    # 2. Iterate grid points; collect those that do not collide with placed objects
    for px, py in sorted_points:
        if not _collides_with_placed(candidate, descriptor, placed, min_clearance):
            # 3. Collect all feasible points within d_min + noise_margin
            if dist <= best_dist + frontier_noise_margin_m:
                pool.append(candidate)

    # 4. Randomly select one point from the frontier pool
    return rng.choice(pool)
```

**Collision detection** (circle approximation):

```python
def _collides_with_placed():
    # Uses the larger half-extent as the object radius
    radius = max(descriptor.half_extent_xy[0], descriptor.half_extent_xy[1])
    min_dist = radius + other_radius + min_clearance
    return hypot(cx - ox, cy - oy) < min_dist
```

### 4.4 Key Parameters

| Parameter | low | medium | high | ultra | Description |
|-----------|-----|--------|------|-------|-------------|
| `pack_jitter_xy` | 0.010 | 0.015 | 0.022 | 0.026 | XY position jitter applied to each object |
| `pack_min_clearance` | 0.040 | 0.025 | 0.008 | 0.004 | Minimum inter-object XY clearance |
| `zone_padding` | 0.030 | 0.020 | 0.008 | 0.004 | Per-object padding used in zone capacity estimation |
| `zone_util_cap` | 0.70 | 0.85 | 0.98 | 1.10 | Zone utilization warning threshold (not a hard failure limit) |

---

## 5. Stage 4: Fitting the Pack into the Zone (`_fit_pack_to_zone`)

```python
def _fit_pack_to_zone(pack_spec, descriptor_by_inst, red_zone_bounds):
    # 1. Compute the bounding box of the pack in local (relative) coordinates
    rel_bounds = _compute_pack_relative_bounds(pack_spec, descriptor_by_inst)

    # 2. Find a valid pack origin inside the red zone
    origin_xy = _choose_pack_origin_in_zone(red_zone_bounds, rel_bounds)
    # Prefers the zone center; clamps to the feasible range if the pack is large.

    return pack_spec, rel_bounds, origin_xy
```

---

## 6. Stage 5: Clearance Reduction and Culling

When the full object set cannot be packed, a **two-layer fallback mechanism** is applied.

### 6.1 Clearance Schedule (Progressive Reduction)

```python
clearance_schedule = _build_min_clearance_schedule(
    start_clearance=0.008,  # Starting value (e.g., high preset)
    floor_clearance=0.002,  # Minimum acceptable clearance
    step=0.005,             # Decrement per level
)
# Example output: [0.008, 0.003, 0.002]
```

For each clearance level, `pack_tries_per_clearance` randomized layout attempts are made before moving to the next level.

### 6.2 Culling Strategy (Object Removal)

```python
def _select_cull_candidate():
    # Removal priority: clutter > fragile > target
    # Within the same role, the outermost object (largest distance from center) is removed first.
    role_rank = 0 if role == "clutter" else (1 if role == "fragile" else 9)
    return sorted(candidates, key=lambda x: (role_rank, -radius, inst))[0]
```

### 6.3 Re-introduction Strategy (Recovering Culled Objects)

```python
def _reintroduce_culled_descriptors():
    # After a successful pack is found, attempts to re-add previously culled objects
    # using the same frontier algorithm at floor_clearance.
```

---

## 7. Stage 6: Robot Placement (`franka_edge_align.py`)

**Core function**: `place_franka_edge_aligned()`

```python
# 1. Select which edge of the bar to place the robot against (hardcoded to x_min)
edge_label = "x_min"

# 2. Compute a weighted anchor position along the edge
#    Weights: target=3.0, fragile=2.0, clutter=1.0
anchor_s = compute_weighted_edge_anchor(...)

# 3. Scan candidate positions with small offsets; pick the first collision-free pose
for offset in scan_offsets_m:  # (0.0, 0.05, -0.05, 0.10, ...)
    pose = _edge_pose_xy(edge_label, anchor_s + offset, ...)
    if collision_checker(pose) == []:
        return pose
```

---

## 8. Stage 7: Gate Validation (`_evaluate_gate`)

Final validation that the generated scene meets all placement requirements:

```python
@dataclass
class MVPGateReport:
    scene_sane: bool                       # All object poses are finite and valid
    base_on_ground: bool                   # Robot base is at floor level
    base_collision_free: bool              # Robot base has no collisions
    gap_ok: bool                           # Robot-to-bar gap matches target (±0.04 m)
    target_in_reach_band: bool             # Target is within robot reach [0.20, 1.10] m
    pack_integrity_ok: bool                # Object positions have not drifted from planned layout
    all_objects_in_red_zone: bool          # All objects are within the red zone
    all_objects_outside_sink_keepout: bool # No objects overlap the sink exclusion zone
```

---

## 9. Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MVP Scene Generation Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. BDDL Parsing                                                         │
│     └─> ManipulationTaskSpec (target/fragile/clutter/support IDs)       │
│                                                                          │
│  2. Zone Computation                                                     │
│     ├─> bar_bounds (countertop boundary)                                 │
│     ├─> sink_keepout_bounds (sink exclusion zone)                        │
│     └─> red_zone_bounds (valid placement region)                         │
│                                                                          │
│  3. Descriptor Construction                                              │
│     └─> ClutterObjectDescriptor (instance_id, role, half_extent, height)│
│                                                                          │
│  4. Clutter Pack Generation (core)                                       │
│     ├─> Generate dense grid points sorted by distance from center        │
│     ├─> Place target near center (within jitter range)                   │
│     ├─> Place remaining objects via Frontier Greedy Packing              │
│     │   ├─> Find nearest feasible point outside min_clearance radius     │
│     │   └─> Randomly sample from frontier pool (d_min + noise_margin)   │
│     └─> Clearance reduction + culling fallback                           │
│                                                                          │
│  5. Pack-to-Zone Fitting                                                 │
│     └─> Compute pack_origin_world (prefers zone center)                  │
│                                                                          │
│  6. Physics Validation                                                   │
│     ├─> apply_pack_transform() — set object poses in simulation          │
│     ├─> Detect object interpenetrations                                  │
│     └─> Detect zone / keepout violations                                 │
│                                                                          │
│  7. Robot Placement                                                      │
│     └─> place_franka_edge_aligned() (weighted anchor + collision scan)   │
│                                                                          │
│  8. Gate Validation                                                      │
│     └─> MVPGateReport (8 checks)                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Key Source Files

| File | Responsibility |
|------|----------------|
| `franka_mounted_mvp_runner_kitchen_bar.py` | Main entry point and orchestration |
| `kitchen_bar_workspace.py` | Zone computation |
| `clutter_pack_layout.py` | Frontier packing algorithm |
| `manipulation_task_spec.py` | BDDL parsing and object classification |
| `franka_edge_align.py` | Robot placement algorithm |
| `problem0.bddl` | Task definition (object list, init and goal conditions) |

---

## 11. Increasing Clutter Density

Two independent control layers are available:

1. **Object count (primary)**: Edit `:objects` and `:init` in the BDDL file to add more objects.
2. **Packing tightness (secondary)**: Use `--clutter-density ultra` or override individual parameters.

> **Note**: `zone_util_cap > 1.0` only triggers a warning log; it does not cause an immediate failure.
