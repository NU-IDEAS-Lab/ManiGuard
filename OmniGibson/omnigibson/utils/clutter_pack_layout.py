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
) -> ClutterPackSpec:
    if not descriptors:
        raise ValueError("descriptors must be non-empty")

    rng = random.Random(seed)
    placed: List[Tuple[ClutterObjectDescriptor, float, float]] = []
    entries: List[ClutterPackEntry] = []

    for descriptor in _ordered_descriptors(descriptors):
        slots = _template_slots(descriptor.role)
        chosen_xy = None
        for slot_idx, (sx, sy) in enumerate(slots):
            dx = rng.uniform(-jitter_xy, jitter_xy) if slot_idx < 2 else rng.uniform(-0.5 * jitter_xy, 0.5 * jitter_xy)
            dy = rng.uniform(-jitter_xy, jitter_xy) if slot_idx < 2 else rng.uniform(-0.5 * jitter_xy, 0.5 * jitter_xy)
            candidate = (sx + dx, sy + dy)
            if _collides_with_placed(candidate, descriptor, placed, min_clearance=min_clearance):
                continue
            chosen_xy = candidate
            break

        if chosen_xy is None:
            # Last-resort expansion keeps the pack valid even for many objects.
            ring_radius = 0.28 + 0.05 * len(placed)
            theta = rng.uniform(-math.pi, math.pi)
            chosen_xy = (ring_radius * math.cos(theta), ring_radius * math.sin(theta))

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


def _template_slots(role: str) -> Tuple[Tuple[float, float], ...]:
    if role == "target":
        return ((0.0, 0.0), (-0.03, 0.02), (0.03, -0.02))
    if role == "fragile":
        return (
            (-0.14, 0.11),
            (0.15, 0.10),
            (-0.13, -0.13),
            (0.14, -0.12),
            (0.0, 0.19),
            (0.0, -0.19),
        )
    return (
        (-0.22, 0.00),
        (0.22, 0.00),
        (0.00, 0.22),
        (0.00, -0.22),
        (-0.18, -0.18),
        (0.18, 0.18),
        (-0.25, 0.14),
        (0.25, -0.14),
    )


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _yaw_from_z_w(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(float(qz), float(qw))
