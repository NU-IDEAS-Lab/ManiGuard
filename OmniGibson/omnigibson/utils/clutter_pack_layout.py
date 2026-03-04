from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClutterObjectDescriptor:
    instance_id: str
    role: str
    half_extent_xy: Tuple[float, float]
    height: float


@dataclass(frozen=True)
class ClutterPackEntry:
    inst_id: str
    role: str
    rel_pose: Tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ClutterPackSpec:
    table_obj_name: str
    pack_origin_world: Tuple[float, float, float]
    object_entries: Tuple[ClutterPackEntry, ...]
    seed: int
    template_id: str


@dataclass(frozen=True)
class PackIntegrityReport:
    ok: bool
    max_position_error: float
    failure_reasons: Tuple[str, ...]


def build_clutter_pack(
    table_obj_name: str,
    descriptors: Sequence[ClutterObjectDescriptor],
    seed: int,
    template_id: str = "cup_first_v1",
    jitter_xy: float = 0.015,
    min_clearance: float = 0.025,
    placement_bounds_local: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    grid_step_m: float = 0.005,
    frontier_noise_margin_m: float = 0.02,
    shuffle_non_target: bool = True,
) -> ClutterPackSpec:
    if not descriptors:
        raise ValueError("descriptors must be non-empty")
    if grid_step_m <= 0.0:
        raise ValueError("grid_step_m must be > 0")
    if frontier_noise_margin_m < 0.0:
        raise ValueError("frontier_noise_margin_m must be >= 0")
    if min_clearance < 0.0:
        raise ValueError("min_clearance must be >= 0")

    rng = random.Random(seed)
    if placement_bounds_local is None:
        placement_bounds_local = ((-0.45, -0.45), (0.45, 0.45))
    sorted_points = _generate_sorted_grid_points(
        bounds=placement_bounds_local,
        step=grid_step_m,
    )
    if len(sorted_points) == 0:
        raise RuntimeError("No candidate points generated for clutter packing.")

    placed: List[Tuple[ClutterObjectDescriptor, float, float]] = []
    entries: List[ClutterPackEntry] = []

    ordered = _ordered_descriptors(descriptors)
    target_descriptors = [d for d in ordered if d.role == "target"]
    non_target_descriptors = [d for d in ordered if d.role != "target"]
    if shuffle_non_target:
        rng.shuffle(non_target_descriptors)
    placement_order = target_descriptors + non_target_descriptors

    for idx, descriptor in enumerate(placement_order):
        chosen_xy = None
        if idx == 0 and descriptor.role == "target":
            # Force target near center so surrounding clutter naturally forms a safety-critical neighborhood.
            cx = rng.uniform(-min(jitter_xy, 0.02), min(jitter_xy, 0.02))
            cy = rng.uniform(-min(jitter_xy, 0.02), min(jitter_xy, 0.02))
            candidate = (cx, cy)
            if not _collides_with_placed(candidate, descriptor, placed, min_clearance=min_clearance):
                chosen_xy = candidate

        if chosen_xy is None:
            pool = _frontier_candidate_pool(
                descriptor=descriptor,
                placed=placed,
                sorted_points=sorted_points,
                min_clearance=min_clearance,
                noise_margin=frontier_noise_margin_m,
            )
            if len(pool) == 0:
                raise RuntimeError(
                    "pack_no_feasible_point:"
                    f"inst={descriptor.instance_id}, role={descriptor.role}, "
                    f"min_clearance={min_clearance:.4f}"
                )
            chosen_xy = rng.choice(pool)

        x, y = chosen_xy
        z = max(0.008, 0.5 * max(descriptor.height, 0.01) + 0.004)
        yaw = 0.0 if descriptor.role == "target" else rng.uniform(-0.18, 0.18)
        qx, qy, qz, qw = _quat_from_yaw(yaw)
        entries.append(
            ClutterPackEntry(
                inst_id=descriptor.instance_id,
                role=descriptor.role,
                rel_pose=(x, y, z, qx, qy, qz, qw),
            )
        )
        placed.append((descriptor, x, y))

    return ClutterPackSpec(
        table_obj_name=table_obj_name,
        pack_origin_world=(0.0, 0.0, 0.0),
        object_entries=tuple(entries),
        seed=int(seed),
        template_id=template_id,
    )


def apply_pack_transform(
    pack_spec: ClutterPackSpec,
    objects_by_inst: Dict[str, object],
    pack_origin_world: Tuple[float, float, float],
    pack_yaw: float = 0.0,
    table_top_z: Optional[float] = None,
) -> Dict[str, Tuple[float, float, float]]:
    cos_y = math.cos(pack_yaw)
    sin_y = math.sin(pack_yaw)
    ox, oy, oz = pack_origin_world
    placements: Dict[str, Tuple[float, float, float]] = {}

    for entry in pack_spec.object_entries:
        obj = objects_by_inst.get(entry.inst_id, None)
        if obj is None:
            continue

        rel_x, rel_y, rel_z, _, _, rel_qz, rel_qw = entry.rel_pose
        wx = ox + cos_y * rel_x - sin_y * rel_y
        wy = oy + sin_y * rel_x + cos_y * rel_y
        wz = (table_top_z if table_top_z is not None else oz) + rel_z
        rel_yaw = _yaw_from_z_w(rel_qz, rel_qw)
        qx, qy, qz, qw = _quat_from_yaw(pack_yaw + rel_yaw)
        obj.set_position_orientation(position=(wx, wy, wz), orientation=(qx, qy, qz, qw))
        placements[entry.inst_id] = (wx, wy, wz)

    return placements


def validate_pack_integrity(
    pack_spec: ClutterPackSpec,
    world_positions: Dict[str, Tuple[float, float, float]],
    pack_origin_world: Tuple[float, float, float],
    pack_yaw: float = 0.0,
    tol_xy: float = 0.03,
) -> PackIntegrityReport:
    cos_y = math.cos(pack_yaw)
    sin_y = math.sin(pack_yaw)
    ox, oy, _ = pack_origin_world
    max_err = 0.0
    failures: List[str] = []

    expected_xy = {}
    observed_xy = {}
    for entry in pack_spec.object_entries:
        rel_x, rel_y = entry.rel_pose[0], entry.rel_pose[1]
        ex = ox + cos_y * rel_x - sin_y * rel_y
        ey = oy + sin_y * rel_x + cos_y * rel_y
        expected_xy[entry.inst_id] = (ex, ey)

        if entry.inst_id not in world_positions:
            failures.append(f"missing_world_pose:{entry.inst_id}")
            continue
        wx, wy, _ = world_positions[entry.inst_id]
        observed_xy[entry.inst_id] = (wx, wy)
        err = math.hypot(wx - ex, wy - ey)
        max_err = max(max_err, err)
        if err > tol_xy:
            failures.append(f"position_error:{entry.inst_id}:{err:.4f}")

    # Pairwise rigidity check.
    inst_ids = [entry.inst_id for entry in pack_spec.object_entries if entry.inst_id in observed_xy]
    for i in range(len(inst_ids)):
        for j in range(i + 1, len(inst_ids)):
            inst_i = inst_ids[i]
            inst_j = inst_ids[j]
            exp_d = math.hypot(
                expected_xy[inst_i][0] - expected_xy[inst_j][0],
                expected_xy[inst_i][1] - expected_xy[inst_j][1],
            )
            obs_d = math.hypot(
                observed_xy[inst_i][0] - observed_xy[inst_j][0],
                observed_xy[inst_i][1] - observed_xy[inst_j][1],
            )
            err = abs(obs_d - exp_d)
            max_err = max(max_err, err)
            if err > tol_xy:
                failures.append(f"pairwise_error:{inst_i}:{inst_j}:{err:.4f}")

    return PackIntegrityReport(
        ok=len(failures) == 0,
        max_position_error=float(max_err),
        failure_reasons=tuple(failures),
    )


def _ordered_descriptors(descriptors: Sequence[ClutterObjectDescriptor]) -> List[ClutterObjectDescriptor]:
    role_priority = {"target": 0, "fragile": 1, "clutter": 2}
    return sorted(
        descriptors,
        key=lambda d: (role_priority.get(d.role, 3), d.instance_id),
    )


def _collides_with_placed(
    candidate_xy: Tuple[float, float],
    descriptor: ClutterObjectDescriptor,
    placed: Iterable[Tuple[ClutterObjectDescriptor, float, float]],
    min_clearance: float,
) -> bool:
    cx, cy = candidate_xy
    radius = max(descriptor.half_extent_xy[0], descriptor.half_extent_xy[1])
    for other_desc, ox, oy in placed:
        other_r = max(other_desc.half_extent_xy[0], other_desc.half_extent_xy[1])
        min_dist = radius + other_r + min_clearance
        if math.hypot(cx - ox, cy - oy) < min_dist:
            return True
    return False


def _generate_sorted_grid_points(
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    step: float,
) -> Tuple[Tuple[float, float], ...]:
    (x0, y0), (x1, y1) = bounds
    x_lo, x_hi = min(x0, x1), max(x0, x1)
    y_lo, y_hi = min(y0, y1), max(y0, y1)
    if x_hi - x_lo <= 0.0 or y_hi - y_lo <= 0.0:
        return tuple()

    points: List[Tuple[float, float]] = []
    x = x_lo
    while x <= x_hi + 1e-9:
        y = y_lo
        while y <= y_hi + 1e-9:
            points.append((round(float(x), 6), round(float(y), 6)))
            y += step
        x += step

    points.sort(key=lambda p: (math.hypot(p[0], p[1]), abs(p[0]) + abs(p[1]), p[0], p[1]))
    return tuple(points)


def _frontier_candidate_pool(
    descriptor: ClutterObjectDescriptor,
    placed: Iterable[Tuple[ClutterObjectDescriptor, float, float]],
    sorted_points: Sequence[Tuple[float, float]],
    min_clearance: float,
    noise_margin: float,
) -> List[Tuple[float, float]]:
    best_dist = None
    pool: List[Tuple[float, float]] = []
    threshold = None
    for px, py in sorted_points:
        candidate = (px, py)
        if _collides_with_placed(candidate, descriptor, placed, min_clearance=min_clearance):
            continue
        dist = math.hypot(px, py)
        if best_dist is None:
            best_dist = dist
            threshold = dist + noise_margin + 1e-12
            pool.append(candidate)
            continue
        if dist <= threshold:
            pool.append(candidate)
            continue
        break
    return pool


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _yaw_from_z_w(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(float(qz), float(qw))
