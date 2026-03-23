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


def check_packing_feasibility(
    descriptors: Sequence[ClutterObjectDescriptor],
    placement_bounds_local: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    min_clearance: float = 0.0,
) -> Tuple[bool, float]:
    """Area-based feasibility pre-check for circle packing.

    Returns ``(feasible, utilization)`` where *utilization* is the ratio of
    total padded-circle area to zone area.  A utilization above ~0.85
    (conservative hexagonal-packing limit for mixed radii in a rectangle)
    is flagged as infeasible.
    """
    if placement_bounds_local is None:
        placement_bounds_local = ((-0.45, -0.45), (0.45, 0.45))
    (x0, y0), (x1, y1) = placement_bounds_local
    x_lo, x_hi = min(x0, x1), max(x0, x1)
    y_lo, y_hi = min(y0, y1), max(y0, y1)
    zone_area = (x_hi - x_lo) * (y_hi - y_lo)
    if zone_area <= 0:
        return False, float("inf")

    total_circle_area = 0.0
    max_r = 0.0
    for d in descriptors:
        r = _effective_radius(d) + 0.5 * min_clearance
        total_circle_area += math.pi * r * r
        max_r = max(max_r, r)

    if 2.0 * max_r > min(x_hi - x_lo, y_hi - y_lo):
        return False, float("inf")

    utilization = total_circle_area / zone_area
    return utilization <= 0.85, utilization


def build_clutter_pack(
    table_obj_name: str,
    descriptors: Sequence[ClutterObjectDescriptor],
    seed: int,
    template_id: str = "cup_first_v1",
    jitter_xy: float = 0.015,
    min_clearance: float = 0.025,
    placement_bounds_local: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    frontier_noise_margin_m: float = 0.02,
    shuffle_non_target: bool = True,
    # Deprecated: kept for backward compatibility, ignored.
    grid_step_m: float = 0.005,
) -> ClutterPackSpec:
    if not descriptors:
        raise ValueError("descriptors must be non-empty")
    if frontier_noise_margin_m < 0.0:
        raise ValueError("frontier_noise_margin_m must be >= 0")
    if min_clearance < 0.0:
        raise ValueError("min_clearance must be >= 0")

    rng = random.Random(seed)
    if placement_bounds_local is None:
        placement_bounds_local = ((-0.45, -0.45), (0.45, 0.45))

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
            pool = compute_candidate_pool(
                descriptor=descriptor,
                placed=placed,
                placement_bounds=placement_bounds_local,
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


def compute_candidate_pool(
    descriptor: ClutterObjectDescriptor,
    placed: Sequence[Tuple[ClutterObjectDescriptor, float, float]],
    placement_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    min_clearance: float,
    noise_margin: float = 0.02,
    n_angles: int = 36,
) -> List[Tuple[float, float]]:
    """Greedy ring-based candidate generation.

    Sweeps concentric rings outward from the origin (target centre).  On each
    ring, *n_angles* evenly-spaced angular slots are tested for collision with
    already-placed circles and zone bounds.  Returns all valid positions on the
    innermost ring(s) that contain at least one free slot, including rings
    within *noise_margin* of the best to add angular variety.
    """
    r_new = _effective_radius(descriptor)
    (x0, y0), (x1, y1) = placement_bounds
    # Shrink bounds so circle centres stay fully inside.
    x_lo = min(x0, x1) + r_new
    x_hi = max(x0, x1) - r_new
    y_lo = min(y0, y1) + r_new
    y_hi = max(y0, y1) - r_new
    if x_lo > x_hi or y_lo > y_hi:
        return []

    if not placed:
        cx = min(max(0.0, x_lo), x_hi)
        cy = min(max(0.0, y_lo), y_hi)
        return [(cx, cy)]

    # Minimum ring radius: must clear the centre-most placed object.
    min_ring_r = r_new + min_clearance
    for d_p, px, py in placed:
        if math.hypot(px, py) < 0.05:
            min_ring_r = max(min_ring_r, _effective_radius(d_p) + r_new + min_clearance)

    ring_step = max(0.005, 0.5 * r_new)
    max_radius = math.hypot(max(abs(x_lo), abs(x_hi)), max(abs(y_lo), abs(y_hi)))

    best_ring_r: Optional[float] = None
    pool: List[Tuple[float, float]] = []
    ring_r = min_ring_r
    while ring_r <= max_radius:
        ring_valid: List[Tuple[float, float]] = []
        for i in range(n_angles):
            theta = 2.0 * math.pi * i / n_angles
            cx = ring_r * math.cos(theta)
            cy = ring_r * math.sin(theta)
            if cx < x_lo or cx > x_hi or cy < y_lo or cy > y_hi:
                continue
            if not _collides_with_placed((cx, cy), descriptor, placed, min_clearance):
                ring_valid.append((cx, cy))

        if ring_valid:
            if best_ring_r is None:
                best_ring_r = ring_r
            if ring_r <= best_ring_r + noise_margin + 1e-12:
                pool.extend(ring_valid)
            else:
                break
        elif best_ring_r is not None and ring_r > best_ring_r + noise_margin:
            break

        ring_r += ring_step

    return pool


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _effective_radius(d: ClutterObjectDescriptor) -> float:
    # Use AABB diagonal so that circle-circle clearance guarantees no
    # axis-aligned bounding-box overlap — matches the 3D AABB interpenetration
    # check used during post-placement validation.
    return math.hypot(d.half_extent_xy[0], d.half_extent_xy[1])


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


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _yaw_from_z_w(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(float(qz), float(qw))


# ---------------------------------------------------------------------------
# Stack layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackObjectDescriptor:
    """Describes one object in a vertical stack."""
    instance_id: str
    role: str                       # "target", "stack", "base" (bottom-most)
    half_extent_xy: Tuple[float, float]
    height: float


@dataclass(frozen=True)
class StackEntry:
    """Computed world-relative pose for one stack element."""
    inst_id: str
    role: str
    rel_pose: Tuple[float, float, float, float, float, float, float]  # x,y,z, qx,qy,qz,qw
    supporter_inst_id: Optional[str]  # inst below this one (None for bottom)


@dataclass(frozen=True)
class StackLayoutSpec:
    """Complete stack layout specification."""
    support_obj_name: str           # table / counter the stack sits on
    stack_origin_world: Tuple[float, float, float]
    entries: Tuple[StackEntry, ...]
    seed: int


def build_stack_layout(
    support_obj_name: str,
    descriptors: Sequence[StackObjectDescriptor],
    seed: int,
    z_clearance: float = 0.003,
    xy_jitter: float = 0.002,
) -> StackLayoutSpec:
    """Compute analytical vertical poses for a stack of objects.

    Objects are stacked bottom-to-top in the order given by *descriptors*.
    Each object is placed centered on the one below with a small z clearance.

    Args:
        support_obj_name: Name of the furniture the stack sits on.
        descriptors: Objects ordered bottom-to-top.
        seed: RNG seed for small xy jitter.
        z_clearance: Gap (m) between the top of one object and the bottom of the next.
        xy_jitter: Max random XY offset (m) per object for slight imperfection.

    Returns:
        StackLayoutSpec with one StackEntry per descriptor.
    """
    if not descriptors:
        raise ValueError("descriptors must be non-empty")

    rng = random.Random(seed)
    entries: List[StackEntry] = []
    cumulative_z = 0.0

    for i, desc in enumerate(descriptors):
        # Centre of object = cumulative_z + half its height
        obj_z = cumulative_z + 0.5 * desc.height
        # Small XY jitter to make placement slightly imperfect
        jx = rng.uniform(-xy_jitter, xy_jitter) if i > 0 else 0.0
        jy = rng.uniform(-xy_jitter, xy_jitter) if i > 0 else 0.0

        supporter = descriptors[i - 1].instance_id if i > 0 else None
        entries.append(StackEntry(
            inst_id=desc.instance_id,
            role=desc.role,
            rel_pose=(jx, jy, obj_z, 0.0, 0.0, 0.0, 1.0),
            supporter_inst_id=supporter,
        ))
        # Advance z cursor: top of this object + clearance
        cumulative_z = obj_z + 0.5 * desc.height + z_clearance

    return StackLayoutSpec(
        support_obj_name=support_obj_name,
        stack_origin_world=(0.0, 0.0, 0.0),
        entries=tuple(entries),
        seed=seed,
    )


def apply_stack_transform(
    stack_spec: StackLayoutSpec,
    objects_by_inst: Dict[str, object],
    stack_origin_world: Tuple[float, float, float],
) -> Dict[str, Tuple[float, float, float]]:
    """Teleport objects to their computed stack positions.

    Args:
        stack_spec: Layout from build_stack_layout().
        objects_by_inst: Map of instance_id → OG object.
        stack_origin_world: (x, y, z) of the support surface top centre.

    Returns:
        Dict mapping instance_id → (wx, wy, wz) world position.
    """
    ox, oy, oz = stack_origin_world
    placements: Dict[str, Tuple[float, float, float]] = {}

    for entry in stack_spec.entries:
        obj = objects_by_inst.get(entry.inst_id)
        if obj is None:
            continue
        rx, ry, rz, qx, qy, qz, qw = entry.rel_pose
        wx = ox + rx
        wy = oy + ry
        wz = oz + rz
        obj.set_position_orientation(position=(wx, wy, wz), orientation=(qx, qy, qz, qw))
        placements[entry.inst_id] = (wx, wy, wz)

    return placements
