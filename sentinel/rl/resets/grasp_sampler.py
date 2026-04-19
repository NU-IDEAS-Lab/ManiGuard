"""Mesh-based antipodal grasp sampler.

Ports the geometric core of UW Lab OmniReset's ``_sample_antipodal_grasps``
(``source/uwlab_tasks/.../mdp/events.py:199-323``) to a dependency-light
``trimesh``-only function. No physics, no sim, no cuRobo — just surface
sampling, antipodal ray casting, and per-axis orientation/standoff sweep.

Output is in **object-local frame** as ``(N, 4, 4)`` transform matrices
(scene-invariant). Compose with the object's world pose at sampling time to
get world-frame eef targets for IK.

OmniGibson's own ``get_grasp_poses_for_object_sticky`` only returns a single
bbox-top pose, which is unusable for narrow-neck objects like goblets. This
sampler finds genuine antipodal pairs across the mesh, yielding far richer
grasp candidates that matter for sticky-mode policies to actually engage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import trimesh
import trimesh.transformations as tra


@dataclass(frozen=True)
class AntipodalConfig:
    """Hyperparameters for antipodal grasp sampling.

    Matches OmniReset's grasp_sampling_event params 1:1 so the algorithm
    behaves the same; only defaults are tuned for Franka + tabletop objects.
    """

    num_surface_samples: int = 64
    """Surface points kept after top-bias filtering (OmniReset default: ~1e6 / (16*32) ≈ 2000)."""

    num_orientations: int = 16
    """Yaw rotations around the grasp axis per candidate."""

    num_standoff_samples: int = 8
    """Standoff distances along the approach direction per candidate."""

    gripper_max_aperture: float = 0.08
    """Max distance between two gripper fingers when open (Franka Panda ~ 0.08m)."""

    finger_offset: float = 0.10
    """Distance from eef origin to fingertip along approach axis (Franka ~ 0.10m).
    Min standoff — approach starts this far from the grasp center."""

    finger_clearance: float = 0.02
    """Extra clearance added to max standoff. OmniReset scales by mesh diagonal."""

    lateral_sigma: float = 0.0
    """If >0, perturb grasp center along grasp axis via truncated normal
    (OmniReset's default is 0, so grasp center is midpoint of antipodal pair)."""

    top_bias: bool = False
    """If True, prefer surface points with high z + upward-facing normals
    (OmniReset's default for top-down-grabbable convex objects). For complex
    shapes like goblet — where top-bias over-samples the wide cup rim and
    misses the stable-grip stem — default is False: uniform surface sampling,
    and the downstream physics + shake test filters out unstable grasps.
    OmniReset philosophy: sampler is permissive, validator is strict."""

    gripper_approach_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Unit vector in eef local frame pointing from eef origin toward fingertips.
    Franka convention: +Z points out of the gripper."""

    grasp_align_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Unit vector in eef local frame that should align with the grasp axis
    (line between the two gripper fingers). Franka convention: +X."""

    orientation_sample_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Axis (in eef local frame) around which to rotate for orientation sweep.
    Same as gripper_approach_axis for Franka — rotating around approach direction."""


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix (4x4) that rotates ``a`` onto ``b``.

    Thin wrapper around ``trimesh.geometry.align_vectors`` that handles the
    degenerate parallel/anti-parallel case without raising.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-6:
        return np.eye(4)
    if dot < -1.0 + 1e-6:
        # 180° rotation — pick any axis perpendicular to a
        perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return tra.rotation_matrix(np.pi, perp)
    return trimesh.geometry.align_vectors(a, b)


def sample_antipodal_grasps(
    mesh: trimesh.Trimesh,
    cfg: AntipodalConfig = AntipodalConfig(),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample antipodal grasp poses on ``mesh``, returned in object-local frame.

    Args:
        mesh: triangle mesh of the target object (vertices in object-local frame).
        cfg: sampling hyperparameters.
        rng: optional numpy Generator for reproducibility.

    Returns:
        ``(N, 4, 4)`` array of homogeneous transform matrices. Each matrix is
        the *eef pose* (not the gripper fingertip midpoint) in object-local
        coords. Apply the object's world pose to get world eef targets.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Surface sampling with 10× oversampling so top-bias can prune.
    initial = max(cfg.num_surface_samples * 10, 500)
    surface_pts, face_idx = trimesh.sample.sample_surface(mesh, initial)
    surface_pts = np.asarray(surface_pts)
    normals = np.asarray(mesh.face_normals[face_idx])

    # 2) Top-bias: score = normalized_z + max(0, normal_z).
    if cfg.top_bias:
        z = surface_pts[:, 2]
        z_range = z.max() - z.min() + 1e-8
        z_norm = (z - z.min()) / z_range
        score = z_norm + np.maximum(normals[:, 2], 0.0)
        keep = np.argsort(score)[-cfg.num_surface_samples:]
    else:
        keep = rng.choice(len(surface_pts), size=cfg.num_surface_samples, replace=False)
    surface_pts = surface_pts[keep]
    normals = normals[keep]

    # 3) Antipodal ray casting: shoot ray in −normal direction through mesh.
    ray_dirs = -normals
    hits, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=surface_pts, ray_directions=ray_dirs, multiple_hits=True
    )
    hits = np.asarray(hits)
    ray_ids = np.asarray(ray_ids)

    # Mesh-adaptive standoff (OmniReset: finger_offset .. finger_offset + diag + clearance/2)
    mesh_diag = float(np.linalg.norm(mesh.extents))
    standoffs = np.linspace(
        cfg.finger_offset,
        cfg.finger_offset + mesh_diag + cfg.finger_clearance / 2,
        cfg.num_standoff_samples,
    )

    # Orientation sweep
    yaws = np.linspace(-np.pi, np.pi, cfg.num_orientations, endpoint=False)

    # Static rotations (eef-frame, reused per candidate)
    align_axis = np.asarray(cfg.grasp_align_axis, dtype=np.float64)
    orient_axis = np.asarray(cfg.orientation_sample_axis, dtype=np.float64)
    approach = np.asarray(cfg.gripper_approach_axis, dtype=np.float64)
    approach = approach / np.linalg.norm(approach)

    out = []
    for i in range(len(surface_pts)):
        candidate_hits = hits[ray_ids == i]
        if len(candidate_hits) == 0:
            continue

        p0 = surface_pts[i]
        dists = np.linalg.norm(candidate_hits - p0, axis=1)
        # Accept only exit points within gripper aperture (drop p0 itself — distance ≈ 0)
        valid = (dists > 1e-4) & (dists <= cfg.gripper_max_aperture)
        if not np.any(valid):
            continue
        # OmniReset picks the furthest valid hit for more stable (deeper) grasps.
        p1 = candidate_hits[valid][np.argmax(dists[valid])]

        grasp_axis = p1 - p0
        axis_len = float(np.linalg.norm(grasp_axis))
        if axis_len < 1e-6:
            continue
        grasp_axis = grasp_axis / axis_len

        if cfg.lateral_sigma > 0.0:
            # Truncated normal centered at 0.5, clipped to [0, 1]
            center_ratio = float(np.clip(rng.normal(0.5, cfg.lateral_sigma / axis_len), 0.0, 1.0))
        else:
            center_ratio = 0.5
        grasp_center = p0 + grasp_axis * axis_len * center_ratio

        # Rotation that aligns gripper's grasp_align_axis with this grasp_axis.
        R_align = _align_vectors(align_axis, grasp_axis)

        # _align_vectors uses minimum-angle rotation, which can leave the
        # gripper approach axis (+Z local) pointing upward in world frame —
        # unreachable for arm-on-pedestal robots. Since the antipodal pair is
        # symmetric about grasp_axis, a 180° rotation about grasp_axis flips
        # the gripper (swaps which finger is left vs right) without changing
        # the two contact points, but flips the approach direction. Apply it
        # whenever approach would otherwise point upward.
        approach_world = (R_align[:3, :3] @ approach)
        if approach_world[2] > 0:
            R_flip = tra.rotation_matrix(np.pi, grasp_axis)
            R_align = R_flip @ R_align

        T_center = tra.translation_matrix(grasp_center)

        for yaw in yaws:
            R_yaw = tra.rotation_matrix(yaw, orient_axis)
            for standoff in standoffs:
                # Gripper standoff: translate backward along approach axis in
                # eef local frame, so after composing the grasp center becomes
                # finger_offset + standoff from eef origin.
                T_standoff = tra.translation_matrix(-approach * standoff)
                T_eef_in_obj = T_center @ R_align @ R_yaw @ T_standoff
                out.append(T_eef_in_obj)

    if not out:
        return np.empty((0, 4, 4))
    return np.stack(out).astype(np.float32)
