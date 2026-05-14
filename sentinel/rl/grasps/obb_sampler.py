"""OBB-based assisted-grasp sampler.

Models the gripper as 3 oriented bounding boxes (left finger, right
finger, palm) plus 2 virtual swept-volume OBBs between the fingers.
All geometric checks unify into "box ∩ object = empty / non-empty"
queries over a pre-sampled point cloud of the mesh surface.

Conditions for a valid grasp pose:

    left_finger  ∩ surface = ∅      (left finger free of object)
    right_finger ∩ surface = ∅      (right finger free of object)
    palm         ∩ surface = ∅      (palm/wrist free of object)
    left_swept   ∩ surface ≠ ∅      (left finger will contact on closing)
    right_swept  ∩ surface ≠ ∅      (right finger will contact on closing)

This replaces the v2 ray-based "two inward fingertip rays + 8 corner
rays per pose" formulation with five uniform OBB volume queries that:
  * upgrade the chord test from 1D to 3D — tolerant to small orientation
    error;
  * add a palm/wrist clearance test that v2 didn't have;
  * share a single point cloud across all box tests (one preprocessing
    pass per mesh, then all per-pose work is vectorized numpy).

Margins:
    finger / palm boxes are shrunk by ``shrink_m`` (default 2 mm)
        so noise and IK jitter don't cause false-positive collisions.
    swept boxes are expanded by ``expand_m`` (default 2 mm) so
        tangent surface contact still registers.

Same output contract as the graspgen sampler:
``(poses (N, 4, 4) float32, scores (N,) float32)`` in mesh-local frame,
Franka convention (+Z approach, +Y closing, +X = Y×Z), origin shifted
to ``eef_link`` via ``hand_to_eef_offset``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Franka panda gripper geometry (long-finger bundle — matches the active
# OmniGibson Franka asset franka_panda_longfinger).
#
# Calibrated against the visual meshes in
#   omnigibson-robot-assets/models/franka/franka_panda_longfinger/urdf/meshes/visual/
# and verified against the cuRobo collision-sphere envelope at the same dataset path.
#
# Geometry summary (mm):
#   panda_hand body:  x = ±31.5,  y = -26 to +66,  z behind fingers ≈ 0 to 58
#   tri_finger:       x = ±18.8,  y = -40 to 0,    z = -13 to +150
#   cuRobo finger inner edge at 80 mm tip-to-tip: ±35.5 mm (71 mm clear corridor)
#   eef_link → fingertip distance along approach: ≈ 98 mm
_FRANKA_MAX_OPENING  = 0.070
_FRANKA_FINGER_LEN   = 0.145
_FRANKA_FINGER_BREAD = 0.038   # along perp
_FRANKA_FINGER_THICK = 0.040   # along closing (slab thickness)
_FRANKA_EEF_TO_TIP   = 0.098   # eef_link → fingertip along approach
# Palm = panda_hand link body (approximate cuboid behind the fingers).
_FRANKA_PALM_HALF_LEN   = 0.030   # along approach
_FRANKA_PALM_HALF_WIDTH = 0.046   # along closing (spans the gripper)
_FRANKA_PALM_HALF_BREAD = 0.032   # along perp

# Assisted-grasp raycast zone (matches sentinel/_omnigibson_patches.py
# LONG_AG_Z = (0.045, 0.085, 0.120, 0.140) in finger-local coords; eef_link
# sits at finger-local z = 0.105 − 0.0584 = +0.0466 m). The swept-volume box
# is confined to this zone — outside it, AG raycasts can't fire even if the
# OBB geometry check passes, so accepting candidates with object material
# outside this band produces "AG didn't engage" failures downstream.
_FRANKA_AG_Z_FROM_EEF_LOW  = 0.045 - 0.0466   # ≈ -0.0016
_FRANKA_AG_Z_FROM_EEF_HIGH = 0.140 - 0.0466   # ≈ +0.0934


@dataclass
class OBBConfig:
    n_surface_points: int = 6000
    """Surface point cloud size used by the OBB containment test (step D)."""

    n_support_directions: int = 200
    """Number of random unit directions used to extract support points
    (mesh.vertices @ d argmax) in step B. Each direction → one extremal
    vertex; after deduplication these approximate the convex-hull
    vertices, which are the natural "gripper-reachable" anchors."""

    n_poses_per_anchor: int = 1000
    """Number of candidate poses sampled around each anchor (support point
    or uniform surface point). Total candidates ≈ (n_support_points +
    n_uniform_anchors) × this; bounded by the OBB-check cost."""

    n_uniform_anchors: int = 30
    """Uniform-surface anchors mixed in alongside support points. Covers
    concave regions (handles, holes, dents) that the convex-hull-only
    sampler can't reach."""

    spread_radius_m: float = 0.02
    """Std-dev of Gaussian noise on anchor position before re-projection
    to the mesh surface. Spreads poses around each anchor."""

    cone_half_angle_rad: float = 1.05   # ≈ π/3 (60°)
    """Approach direction is sampled inside a cone of this half-angle
    around the inward-pointing surface normal at the projected anchor.
    π/3 admits "near-perpendicular" approaches without locking to the
    exact normal (which was the bug in the original sampler)."""

    shrink_m: float        = 0.002   # finger/palm/swept box inward margin
    max_candidates: int    = 400
    approach_standoff: float = 0.10
    expand_m: float        = 0.002   # legacy, unused after AG-zone swept rework
    hand_to_eef_offset: float = 0.1034  # legacy, retained for API compat; pose origin uses _FRANKA_EEF_TO_TIP directly


def sample_obb_assisted_grasps(
    mesh,
    config: OBBConfig | None = None,
    rng: np.random.Generator | None = None,
    pose_chunk: int = 4000,
):
    """OBB-based assisted-grasp sampler. Returns ``(poses, scores)``.

    Args:
        mesh: ``trimesh.Trimesh`` in object-local frame.
        config: tunable parameters; defaults match Franka panda.
        rng: numpy ``Generator`` for reproducibility.
        pose_chunk: chunk size for the batched box-mesh containment
            test. ``(pose_chunk × n_surface_points × 3)`` floats fit in
            RAM at once. 4000 → ~600 MB peak at 6000 points; lower if
            memory-constrained.

    Returns:
        ``(poses (N, 4, 4) float32, scores (N,) float32)`` in mesh-local
        frame, ranked by descending score.
    """
    import trimesh

    cfg = config or OBBConfig()
    if rng is None:
        rng = np.random.default_rng()

    # ── A. Preprocess: sample mesh surface once (shared by all box tests).
    surf, _ = trimesh.sample.sample_surface(
        mesh, cfg.n_surface_points, seed=int(rng.integers(0, 2**31 - 1)),
    )
    surf = np.asarray(surf, dtype=np.float64)
    n_surf = len(surf)

    # ── B. Pose seeding via support points + uniform anchors.
    # Sample K random unit directions, take mesh.vertices @ d argmax per
    # direction → convex-hull-extremal vertices. These are the "natural
    # gripper anchor" points (places the gripper reaches from outside).
    # Mix in a small batch of uniform surface points to cover concave
    # regions (handles, holes) that the convex-hull-only seeds miss.
    V = np.asarray(mesh.vertices, dtype=np.float64)
    center_mass = V.mean(axis=0)
    dirs = rng.standard_normal(size=(cfg.n_support_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    proj = dirs @ (V - center_mass).T                          # (K, V)
    extreme_idx = proj.argmax(axis=1)
    support_anchors = V[np.unique(extreme_idx)]

    if cfg.n_uniform_anchors > 0:
        uniform_anchors, _ = trimesh.sample.sample_surface(
            mesh, cfg.n_uniform_anchors,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        anchors = np.vstack([support_anchors, np.asarray(uniform_anchors, dtype=np.float64)])
    else:
        anchors = support_anchors

    # Fan out: each anchor → n_poses_per_anchor candidates with Gaussian
    # position spread; "project" each spread point back to the mesh surface
    # by finding the nearest pre-sampled surface point (via KDTree on
    # ``surf``, which was sampled in step A). Avoids
    # ``trimesh.proximity.closest_point`` which segfaults on some meshes
    # (BVH instability in the trimesh C extension).
    from scipy.spatial import cKDTree

    n_per = cfg.n_poses_per_anchor
    N = len(anchors) * n_per
    centers_raw = np.repeat(anchors, n_per, axis=0)
    centers_raw = centers_raw + rng.normal(0.0, cfg.spread_radius_m, size=centers_raw.shape)

    # Sample a denser surface-with-normals set for the projection step. We
    # can't reuse ``surf`` from step A because it lacks face_idx → normals.
    proj_pts, proj_face_idx = trimesh.sample.sample_surface(
        mesh, max(cfg.n_surface_points, 6000),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    proj_pts = np.asarray(proj_pts, dtype=np.float64)
    proj_normals = mesh.face_normals[proj_face_idx].astype(np.float64)
    proj_normals /= np.linalg.norm(proj_normals, axis=1, keepdims=True) + 1e-12
    proj_tree = cKDTree(proj_pts)
    _, nearest_idx = proj_tree.query(centers_raw, k=1)
    closest_pts = proj_pts[nearest_idx]
    surface_normals = proj_normals[nearest_idx]

    # Approach direction sampled in a cone around -surface_normal (the
    # inward direction at the projected surface point). cos(theta) uniform
    # in [cos(half_angle), 1], azimuth uniform; rotated into world via a
    # tangent basis built from the normal.
    cos_max = float(np.cos(cfg.cone_half_angle_rad))
    cos_theta = rng.uniform(cos_max, 1.0, size=N)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=N)
    local_cone = np.stack([sin_theta * np.cos(azimuth),
                           sin_theta * np.sin(azimuth),
                           cos_theta], axis=-1)               # (N, 3) in cone-axis-z frame
    cone_axis = -surface_normals                              # inward
    # Build per-point tangent basis (any orthonormal frame with z=cone_axis).
    helper = np.where(np.abs(cone_axis[:, 0:1]) < 0.9,
                      np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]))
    helper = np.broadcast_to(helper, cone_axis.shape).copy()
    tx = np.cross(helper, cone_axis)
    tx /= np.linalg.norm(tx, axis=1, keepdims=True) + 1e-12
    ty = np.cross(cone_axis, tx)
    R_cone = np.stack([tx, ty, cone_axis], axis=-1)           # (N, 3, 3)
    approach = np.einsum("nij,nj->ni", R_cone, local_cone)
    approach /= np.linalg.norm(approach, axis=1, keepdims=True) + 1e-12

    # Closing = random unit perpendicular to approach.
    raw2 = rng.standard_normal(size=(N, 3))
    proj2 = np.einsum("ij,ij->i", raw2, approach)[:, None] * approach
    closing = raw2 - proj2
    closing /= np.linalg.norm(closing, axis=1, keepdims=True) + 1e-12
    perp = np.cross(closing, approach)

    # eef_link target = closest surface point shifted BACK along approach
    # by eef_to_tip, so the fingertip lands ON the surface (not at the
    # noise-perturbed seed). This is the "snap-to-surface" trick — it
    # keeps the grasp geometrically clean even after position spread.
    mid = closest_pts - _FRANKA_EEF_TO_TIP * approach

    width_margin = np.ones(N, dtype=np.float64)
    n_cand = N

    # ── C. Define the 5 OBBs in the grasp frame (perp=X, closing=Y, approach=Z).
    # Each box: (center_offset, half_extents) in local (perp, closing, approach).
    # Grasp-frame origin = eef_link. Long-finger eef_link sits 98 mm behind
    # the fingertips, so we shift finger/swept boxes forward by eef_to_tip
    # so they span [eef_to_tip - fl, eef_to_tip] = actual finger volume.
    open_half = 0.5 * _FRANKA_MAX_OPENING
    fl = _FRANKA_FINGER_LEN
    fb = _FRANKA_FINGER_BREAD
    ft = _FRANKA_FINGER_THICK
    et = _FRANKA_EEF_TO_TIP
    s = cfg.shrink_m
    e = cfg.expand_m

    fz = et - fl / 2                                   # finger-body box z-center
    pz = et - fl - _FRANKA_PALM_HALF_LEN               # palm sits behind finger root

    # Swept-volume z range mirrors the AG raycast firing zone, not the full
    # finger length. Outside this band AG raycasts can't fire.
    sz_lo = _FRANKA_AG_Z_FROM_EEF_LOW
    sz_hi = _FRANKA_AG_Z_FROM_EEF_HIGH
    sz_center  = 0.5 * (sz_lo + sz_hi)
    sz_halfraw = 0.5 * (sz_hi - sz_lo)

    # Finger center (along closing) sits outside the open_half tip face by
    # half the finger thickness, so the inner face of the slab is at
    # ±open_half exactly.
    boxes = {
        "left_finger": (
            np.array([0.0, -(open_half + ft / 2), fz]),
            np.array([fb / 2 - s, ft / 2 - s, fl / 2 - s]),
            False,   # must be EMPTY
        ),
        "right_finger": (
            np.array([0.0, +(open_half + ft / 2), fz]),
            np.array([fb / 2 - s, ft / 2 - s, fl / 2 - s]),
            False,
        ),
        "palm": (
            np.array([0.0, 0.0, pz]),
            np.array([_FRANKA_PALM_HALF_BREAD - s, _FRANKA_PALM_HALF_WIDTH - s,
                      _FRANKA_PALM_HALF_LEN - s]),
            False,
        ),
        # Swept volume spans the gripping corridor BETWEEN the fingers,
        # confined in z to the AG raycast firing zone. Shrunk by ``s`` on
        # all sides (same as the must-be-empty boxes) — accepting candidates
        # whose object material falls outside the AG zone produced "AG
        # didn't engage" failures downstream even when the OBB geometry
        # check passed and cuRobo's motion plan succeeded.
        "swept": (
            np.array([0.0, 0.0, sz_center]),
            np.array([fb / 2 - s, open_half - s, sz_halfraw - s]),
            True,    # must be NON-EMPTY
        ),
    }

    # ── D. Batched OBB-vs-surface containment, in chunks over poses.
    # Per pose, build the local frame R = [perp, closing, approach] (3×3,
    # columns are box axes in world). Transform world surface points into
    # box-local coords via (p_world - center) @ R, then test AABB.
    keep = np.ones(n_cand, dtype=bool)
    swept_count = np.zeros(n_cand, dtype=np.int32)

    # Stack rotation cols: R[n,:,:] = [perp[n], closing[n], approach[n]].
    R_all = np.stack([perp, closing, approach], axis=-1)   # (n_cand, 3, 3)

    # Optimization: transform surface points into each pose's grasp frame
    # ONCE per chunk (instead of once per box). For each box, just shift by
    # its center_offset and run the AABB check — no extra rotation work.
    # Also pre-filter points by distance to mid: only points within R_max
    # can possibly lie in any gripper box. R_max = farthest box-corner
    # distance from mid across all 5 boxes (palm-rear corner usually wins).
    r_max = max(
        float(np.linalg.norm(c_off)) + float(np.linalg.norm(h_ext))
        for c_off, h_ext, _ in boxes.values()
    ) + 0.005   # 5 mm safety
    r_max_sq = r_max ** 2
    surf32 = surf.astype(np.float32)
    for start in range(0, n_cand, pose_chunk):
        end = min(start + pose_chunk, n_cand)
        R = R_all[start:end].astype(np.float32)        # (B, 3, 3)
        mid_chunk = mid[start:end].astype(np.float32)  # (B, 3)

        # Sphere pre-filter: per-pose mask of relevant surface points.
        # Vectorizing the pre-filter across the chunk would cost
        # B × S × 3 floats which is exactly what we're trying to avoid.
        # Instead, loop per pose for the cheap squared-distance filter
        # and accumulate kept counts. Empirically this is ~10× faster
        # than the full-chunk path for objects much smaller than R_max.
        for bi, b_idx in enumerate(range(start, end)):
            diff = surf32 - mid_chunk[bi]                       # (S, 3)
            d2 = np.einsum("ij,ij->i", diff, diff)              # (S,)
            near = d2 <= r_max_sq
            if not np.any(near):
                # No surface in reach → swept box empty → fail.
                keep[b_idx] = False
                continue
            diff_near = diff[near]                              # (S', 3)
            # Grasp-frame coords (relative to mid): diff_near @ R[bi].
            local_mid = diff_near @ R[bi]                       # (S', 3)

            for name, (center_offset, half_ext, must_be_nonempty) in boxes.items():
                local = local_mid - center_offset.astype(np.float32)
                inside = np.all(np.abs(local) <= half_ext.astype(np.float32), axis=-1)
                count = int(inside.sum())
                if must_be_nonempty:
                    if count == 0:
                        keep[b_idx] = False
                        break
                    if name == "swept":
                        swept_count[b_idx] = count
                else:
                    if count > 0:
                        keep[b_idx] = False
                        break

    if not np.any(keep):
        return _empty()

    # ── E. Score and pose assembly.
    z = approach[keep]
    y = closing[keep]
    y = y - np.einsum("ij,ij->i", y, z)[:, None] * z
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    x = np.cross(y, z)
    centers = mid[keep]
    s_width = width_margin[keep]

    # Swept-volume fill (normalized by total surface points) — only signal
    # we have without a chord width; higher = more object material captured.
    s_fill = swept_count[keep] / max(n_surf, 1)
    scores = (s_width * s_fill).astype(np.float32)

    N = len(centers)
    poses = np.zeros((N, 4, 4), dtype=np.float32)
    poses[:, :3, 0] = x
    poses[:, :3, 1] = y
    poses[:, :3, 2] = z
    poses[:, :3, 3] = centers   # seed center == eef_link target directly
    poses[:,  3, 3] = 1.0

    order = np.argsort(-scores)
    if len(order) > cfg.max_candidates:
        order = order[: cfg.max_candidates]
    return poses[order], scores[order]


def _empty():
    return np.empty((0, 4, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
