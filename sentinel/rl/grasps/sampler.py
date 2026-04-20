"""Mesh-based antipodal grasp sampler.

Ports the geometric core of UW Lab OmniReset's ``_sample_antipodal_grasps``
(``source/uwlab_tasks/.../mdp/events.py:199-323``) to a dependency-light
``trimesh``-only function. No physics, no sim, no cuRobo — just surface
sampling, antipodal ray casting, and per-axis orientation sweep.

Key differences from OmniReset's port:

- Standoff sweep dropped: without an approach phase and in a single-env +
  real-arm-IK setting, candidates with standoff > finger_offset put fingertips
  behind the object and close on air.
- Antipodal partner selection by raycast-hit parity: even-indexed sorted hits
  (0, 2, ...) are material-EXIT points where a fingertip placed at +aperture/2
  along grasp_axis lands in air (teleport-safe, contact on close). Odd indices
  are material-entry points that put the opposite fingertip inside solid
  material. OmniReset's ``furthest-valid-within-aperture`` silently falls back
  to a degenerate inner-surface hit when the true antipodal partner is beyond
  aperture — we break instead. Each surface point can therefore yield multiple
  valid candidates (e.g. wall-pinch + across-cavity for a cup).

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

    gripper_max_aperture: float = 0.08
    """Max distance between two gripper fingers when open (Franka Panda ~ 0.08m)."""

    finger_offset: float = 0.10
    """Distance from eef origin to fingertip along approach axis (Franka ~ 0.10m).
    Used as the fixed backward translation of eef along approach so fingertips
    land at the grasp center."""

    min_grasp_width: float = 0.002
    """Minimum grasp_axis length. Set just above the Franka Panda gripper's
    physical closed inner-face gap (~1.2-1.7 mm, measured via
    sentinel.rl.grasps.measure_gripper). Raycast exits closer than this are rejected
    as sub-mesh-precision artifacts rather than real grasp geometry."""

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

    # Orientation sweep
    yaws = np.linspace(-np.pi, np.pi, cfg.num_orientations, endpoint=False)

    # Static rotations (eef-frame, reused per candidate)
    align_axis = np.asarray(cfg.grasp_align_axis, dtype=np.float64)
    orient_axis = np.asarray(cfg.orientation_sample_axis, dtype=np.float64)
    approach = np.asarray(cfg.gripper_approach_axis, dtype=np.float64)
    approach = approach / np.linalg.norm(approach)

    # Pre-compute the single eef-local standoff translation: pull eef back
    # along approach by finger_offset so fingertips land at grasp_center.
    T_standoff = tra.translation_matrix(-approach * cfg.finger_offset)

    out = []
    for i in range(len(surface_pts)):
        candidate_hits = hits[ray_ids == i]
        if len(candidate_hits) == 0:
            continue

        p0 = surface_pts[i]
        dists = np.linalg.norm(candidate_hits - p0, axis=1)

        # Sort hits by distance so parity = material entry/exit.
        order = np.argsort(dists)
        sorted_hits = candidate_hits[order]
        sorted_dists = dists[order]

        # Trim tail: hits beyond gripper aperture are unreachable. Chopping the
        # tail keeps parity of remaining indices intact.
        within = sorted_dists <= cfg.gripper_max_aperture
        sorted_hits = sorted_hits[within]
        sorted_dists = sorted_dists[within]
        if len(sorted_dists) == 0:
            continue

        # Strip p0 self-intersection only. Tight epsilon (1e-6 = 1 micron): big
        # enough to absorb floating-point noise, small enough not to drop
        # legitimate thin-wall hits (which would shift parity and break the
        # even-index = exit invariant).
        if sorted_dists[0] < 1e-6:
            sorted_hits = sorted_hits[1:]
            sorted_dists = sorted_dists[1:]

        # Even indices (0, 2, 4, ...) are material-EXIT points: fingertip at
        # grasp_center + aperture/2 * grasp_axis lands in AIR, so closing
        # fingers sweep to p1 through air and make contact.
        # Odd indices are material-ENTRY points: fingertip would teleport
        # inside solid material — PhysX resolves by displacing the object,
        # breaking contact. Skip them.
        for k in range(0, len(sorted_dists), 2):
            axis_len = float(sorted_dists[k])
            if axis_len < cfg.min_grasp_width:
                continue  # degenerate (e.g. p0 near open top, ray clipped mesh)

            p1 = sorted_hits[k]
            grasp_axis = (p1 - p0) / axis_len

            if cfg.lateral_sigma > 0.0:
                center_ratio = float(
                    np.clip(rng.normal(0.5, cfg.lateral_sigma / axis_len), 0.0, 1.0)
                )
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
            approach_world = R_align[:3, :3] @ approach
            if approach_world[2] > 0:
                R_flip = tra.rotation_matrix(np.pi, grasp_axis)
                R_align = R_flip @ R_align

            T_center = tra.translation_matrix(grasp_center)

            for yaw in yaws:
                R_yaw = tra.rotation_matrix(yaw, orient_axis)
                T_eef_in_obj = T_center @ R_align @ R_yaw @ T_standoff
                out.append(T_eef_in_obj)

    if not out:
        return np.empty((0, 4, 4))
    return np.stack(out).astype(np.float32)
