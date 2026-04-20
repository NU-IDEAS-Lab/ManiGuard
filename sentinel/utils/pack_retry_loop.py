"""Pack retry loop with clearance decay and object culling.

Extracted from the kitchen bar runner. Sim-dependent operations are injected
as callbacks so this module has no direct simulator dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sentinel.utils.clutter_pack_layout import (
    ClutterObjectDescriptor,
    ClutterPackSpec,
    apply_pack_transform,
    build_clutter_pack,
    check_packing_feasibility,
    validate_pack_integrity,
)
from sentinel.utils.tabletop_workspace import (
    Bounds2D,
    ZoneCapacityStats,
    bounds_overlap,
    compute_zone_capacity,
    contains_point,
    normalize_bounds,
)


@dataclass
class PackRetryConfig:
    pack_jitter_xy: float = 0.022
    pack_min_clearance: float = 0.008
    pack_clearance_step_m: float = 0.005
    pack_clearance_floor_m: float = 0.002
    pack_clearance_search_mode: str = "shrink_from_start"
    frontier_noise_margin_m: float = 0.02
    pack_tries_per_clearance: int = 10
    zone_padding: float = 0.008
    zone_util_cap: float = 0.98
    interpenetration_tol: float = 0.003
    integrity_tol_xy: float = 0.035


@dataclass
class PackResult:
    pack_spec: ClutterPackSpec
    world_positions: Dict[str, Tuple[float, float, float]]
    pack_origin: Tuple[float, float, float]
    attempt_used: int
    clearance_used: float
    active_descriptors: List[ClutterObjectDescriptor]
    active_objects_by_inst: Dict[str, object]
    cull_history: List[dict]


def _build_min_clearance_schedule(
    start_clearance: float,
    floor_clearance: float,
    step: float,
    search_mode: str = "shrink_from_start",
) -> Tuple[float, ...]:
    start = max(0.0, float(start_clearance))
    floor = max(0.0, float(floor_clearance))
    decrement = max(1e-6, float(step))
    if floor > start:
        raise ValueError("floor_clearance must be <= start_clearance")
    if search_mode not in {"shrink_from_start", "expand_from_floor"}:
        raise ValueError(f"Unsupported search_mode: {search_mode}")
    values = []
    if search_mode == "shrink_from_start":
        cur = start
        while cur > floor + 1e-9:
            values.append(round(cur, 4))
            cur -= decrement
        values.append(round(floor, 4))
    else:
        cur = floor
        while cur < start - 1e-9:
            values.append(round(cur, 4))
            cur += decrement
        values.append(round(start, 4))
    dedup = []
    for v in values:
        if v not in dedup:
            dedup.append(v)
    return tuple(dedup)


def _local_bounds_from_zone(zone_bounds: Bounds2D) -> Bounds2D:
    (x0, y0), (x1, y1) = normalize_bounds(zone_bounds)
    half_x = 0.5 * (x1 - x0)
    half_y = 0.5 * (y1 - y0)
    return ((-half_x, -half_y), (half_x, half_y))


def _select_cull_candidate(
    active_descriptors: Sequence[ClutterObjectDescriptor],
    last_pack_spec: Optional[ClutterPackSpec],
) -> Optional[Tuple[str, str, float]]:
    if len(active_descriptors) == 0:
        return None
    candidates = []
    if last_pack_spec is not None:
        for entry in last_pack_spec.object_entries:
            if entry.role == "target":
                continue
            radius = math.hypot(entry.rel_pose[0], entry.rel_pose[1])
            candidates.append((entry.inst_id, entry.role, radius))
    else:
        for d in active_descriptors:
            if d.role == "target":
                continue
            candidates.append((d.instance_id, d.role, 0.0))
    if len(candidates) == 0:
        return None

    def _priority(item):
        inst, role, radius = item
        role_rank = 0 if role == "clutter" else (1 if role == "fragile" else 9)
        return (role_rank, -float(radius), inst)

    candidates.sort(key=_priority)
    return candidates[0]


def _compute_pack_relative_bounds(pack_spec, descriptor_by_inst):
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    for entry in pack_spec.object_entries:
        d = descriptor_by_inst[entry.inst_id]
        rel_x, rel_y = entry.rel_pose[0], entry.rel_pose[1]
        x_min = min(x_min, rel_x - d.half_extent_xy[0])
        x_max = max(x_max, rel_x + d.half_extent_xy[0])
        y_min = min(y_min, rel_y - d.half_extent_xy[1])
        y_max = max(y_max, rel_y + d.half_extent_xy[1])

    if not math.isfinite(x_min):
        raise RuntimeError("Failed to compute clutter-pack bounds.")

    return ((x_min, y_min), (x_max, y_max))


def _choose_pack_origin_in_zone(zone_bounds, rel_bounds):
    (zx0, zy0), (zx1, zy1) = normalize_bounds(zone_bounds)
    (rx0, ry0), (rx1, ry1) = normalize_bounds(rel_bounds)

    ox_lo = zx0 - rx0
    ox_hi = zx1 - rx1
    oy_lo = zy0 - ry0
    oy_hi = zy1 - ry1

    if ox_lo > ox_hi or oy_lo > oy_hi:
        raise RuntimeError(
            "zone_capacity_exceeded: clutter pack does not fit inside red zone "
            f"(ox=[{ox_lo:.3f},{ox_hi:.3f}], oy=[{oy_lo:.3f},{oy_hi:.3f}])"
        )

    desired_x = 0.5 * (zx0 + zx1)
    desired_y = 0.5 * (zy0 + zy1)
    origin_x = min(max(desired_x, ox_lo), ox_hi)
    origin_y = min(max(desired_y, oy_lo), oy_hi)
    return (origin_x, origin_y)


def _fit_pack_to_zone(pack_spec, descriptor_by_inst, red_zone_bounds):
    rel_bounds = _compute_pack_relative_bounds(pack_spec, descriptor_by_inst)
    origin_xy = _choose_pack_origin_in_zone(red_zone_bounds, rel_bounds)
    return pack_spec, rel_bounds, origin_xy


def run_pack_retry_loop(
    support_name: str,
    descriptors: List[ClutterObjectDescriptor],
    objects_by_inst: Dict[str, object],
    red_zone_bounds: Bounds2D,
    table_top_z: float,
    floor_z: float,
    config: PackRetryConfig,
    base_seed: int,
    episode: int,
    settle_fn: Callable[[Dict[str, object]], None],
    park_fn: Callable[[Dict[str, object]], None],
    validate_poses_fn: Callable[[Dict[str, object]], List[str]],
    check_interpenetration_fn: Callable[[Dict[str, object], float], List],
    obstacle_keepout_bounds: Optional[Bounds2D] = None,
    obstacle_keepout_bounds_seq: Optional[Sequence[Bounds2D]] = None,
    extra_validation_fn: Optional[Callable[[Dict[str, object]], Optional[str]]] = None,
) -> PackResult:
    clearance_schedule = _build_min_clearance_schedule(
        start_clearance=config.pack_min_clearance,
        floor_clearance=config.pack_clearance_floor_m,
        step=config.pack_clearance_step_m,
        search_mode=config.pack_clearance_search_mode,
    )
    placement_bounds_local = _local_bounds_from_zone(red_zone_bounds)

    active_descriptors = list(descriptors)
    world_positions = None
    pack_spec = None
    pack_origin = None
    pack_attempt_used = None
    clearance_used = None
    last_pack_error = None
    active_objects_by_inst = None
    cull_history: List[dict] = []
    last_pack_for_cull = None
    attempt_idx = 0
    solved = False

    keepout_bounds_seq = tuple(obstacle_keepout_bounds_seq or ())
    if obstacle_keepout_bounds is not None and obstacle_keepout_bounds not in keepout_bounds_seq:
        keepout_bounds_seq = (obstacle_keepout_bounds,) + keepout_bounds_seq

    while not solved:
        descriptor_by_inst = {d.instance_id: d for d in active_descriptors}
        active_objects = {
            d.instance_id: objects_by_inst[d.instance_id]
            for d in active_descriptors
            if d.instance_id in objects_by_inst
        }
        if len(active_objects) != len(active_descriptors):
            missing = sorted(set(descriptor_by_inst.keys()) - set(active_objects.keys()))
            raise RuntimeError(f"missing_scope_objects_for_active_set: {missing}")

        passive_objects = {inst: obj for inst, obj in objects_by_inst.items() if inst not in active_objects}
        park_fn(passive_objects)

        zone_capacity = compute_zone_capacity(
            red_zone_bounds=red_zone_bounds,
            half_extents_xy=[d.half_extent_xy for d in active_descriptors],
            per_object_padding=config.zone_padding,
        )
        print(
            f"[PackRetry] zone_capacity: active_count={len(active_descriptors)}, "
            f"utilization={zone_capacity.utilization:.3f}"
        )

        solved_this_round = False
        for level_idx, min_clearance in enumerate(clearance_schedule, start=1):
            for trial_idx in range(1, config.pack_tries_per_clearance + 1):
                attempt_idx += 1
                jitter_xy = max(0.002, config.pack_jitter_xy)
                try:
                    pack_spec_candidate = build_clutter_pack(
                        table_obj_name=support_name,
                        descriptors=active_descriptors,
                        seed=base_seed + episode + attempt_idx * 101,
                        jitter_xy=jitter_xy,
                        min_clearance=min_clearance,
                        placement_bounds_local=placement_bounds_local,
                        frontier_noise_margin_m=config.frontier_noise_margin_m,
                        shuffle_non_target=True,
                    )
                    last_pack_for_cull = pack_spec_candidate
                    pack_spec_candidate, _, origin_xy = _fit_pack_to_zone(
                        pack_spec=pack_spec_candidate,
                        descriptor_by_inst=descriptor_by_inst,
                        red_zone_bounds=red_zone_bounds,
                    )
                    pack_origin_candidate = (origin_xy[0], origin_xy[1], table_top_z)
                    world_positions_candidate = apply_pack_transform(
                        pack_spec=pack_spec_candidate,
                        objects_by_inst=active_objects,
                        pack_origin_world=pack_origin_candidate,
                        pack_yaw=0.0,
                        table_top_z=table_top_z,
                    )
                    settle_fn(active_objects)

                    invalid_after = validate_poses_fn(active_objects)
                    if invalid_after:
                        last_pack_error = f"invalid_object_pose_after_settle: attempt={attempt_idx}"
                        continue

                    penetrations = check_interpenetration_fn(active_objects, config.interpenetration_tol)
                    if penetrations:
                        last_pack_error = f"interpenetration: attempt={attempt_idx}, count={len(penetrations)}"
                        continue

                    # Zone constraint check.
                    zone_ok = True
                    for inst, obj in active_objects.items():
                        try:
                            aabb_min, aabb_max = obj.aabb
                            cx = 0.5 * (float(aabb_min[0]) + float(aabb_max[0]))
                            cy = 0.5 * (float(aabb_min[1]) + float(aabb_max[1]))
                        except Exception:
                            pos = obj.get_position_orientation()[0]
                            cx, cy = float(pos[0]), float(pos[1])
                        if not contains_point(red_zone_bounds, (cx, cy)):
                            zone_ok = False
                            break
                        if keepout_bounds_seq:
                            obj_bounds = ((cx, cy), (cx, cy))
                            if any(bounds_overlap(obj_bounds, keepout, tol=1e-6) for keepout in keepout_bounds_seq):
                                zone_ok = False
                                break
                    if not zone_ok:
                        last_pack_error = f"zone_constraint_violation: attempt={attempt_idx}"
                        continue

                    if extra_validation_fn is not None:
                        extra_error = extra_validation_fn(active_objects)
                        if extra_error:
                            last_pack_error = f"{extra_error}: attempt={attempt_idx}"
                            continue

                    pack_spec = pack_spec_candidate
                    world_positions = world_positions_candidate
                    pack_origin = pack_origin_candidate
                    pack_attempt_used = attempt_idx
                    clearance_used = min_clearance
                    active_objects_by_inst = active_objects
                    solved = True
                    solved_this_round = True
                    break
                except Exception as e:
                    last_pack_error = str(e)
            if solved_this_round:
                break

        if solved_this_round:
            break

        cull = _select_cull_candidate(active_descriptors, last_pack_for_cull)
        if cull is None:
            break
        cull_inst, cull_role, cull_radius = cull
        kept = [d for d in active_descriptors if d.instance_id != cull_inst]
        if len(kept) == len(active_descriptors):
            break
        active_descriptors = kept
        cull_history.append({
            "attempt": attempt_idx,
            "inst_id": cull_inst,
            "role": cull_role,
            "radius": float(cull_radius),
            "remaining": len(active_descriptors),
        })
        print(
            f"[PackRetry] cull: removed={cull_inst} role={cull_role}, "
            f"remaining={len(active_descriptors)}"
        )

    if world_positions is None or pack_spec is None or active_objects_by_inst is None:
        raise RuntimeError(f"pack_generation_failed_after_retries: {last_pack_error}")

    return PackResult(
        pack_spec=pack_spec,
        world_positions=world_positions,
        pack_origin=pack_origin,
        attempt_used=pack_attempt_used,
        clearance_used=clearance_used,
        active_descriptors=active_descriptors,
        active_objects_by_inst=active_objects_by_inst,
        cull_history=cull_history,
    )
