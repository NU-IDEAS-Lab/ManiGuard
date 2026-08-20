"""Geometric shallow-grab depth for the stack-retrieve family (trimesh mesh proximity).

One responsibility: *how far to retract a stack pick along its own approach axis so the gripper
grabs ONLY the top object and never the object below it.*

A stack pick descends to the annotated (deep) grasp; with the sticky assisted-grasp + long fingers
that descent can contact the object directly beneath the top one, so the pick lifts/drags two
objects. We fix it purely geometrically (no sim iterations): for each instance, line-search along
``-approach`` for the smallest retraction ``d`` at which the gripper mesh clears the BELOW object by
``margin`` while still contacting the TOP object within ``contact_tol``. If no such ``d`` exists (the
gripper is thicker than the stack gap) the caller clamps/fails the task.

Proximity uses trimesh ``mesh.nearest.on_surface`` (unsigned point-to-surface distance) — no
python-fcl (not installed) and no ``contains`` (the object meshes are non-watertight). The gripper
always approaches/retracts from above, so a penetrating fingertip sits near the object's TOP face and
the unsigned distance tracks the true clearance monotonically as it retracts upward.

trimesh lives here (mesh work) to keep ``stack_geom.py`` pure numpy.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

ANN_DIR = Path("outputs/grasp_annotation")


def min_surface_dist(points_world, mesh_world) -> float:
    """Min distance from ``points_world`` (N,3) to the surface of ``mesh_world`` (trimesh)."""
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    _, dists, _ = mesh_world.nearest.on_surface(pts)
    return float(np.asarray(dists).min())


def shallow_retraction(
    gripper_pts_eef,
    T_grasp,
    approach,
    below_mesh_world,
    top_mesh_world,
    *,
    margin: float = 0.005,
    contact_tol: float = 0.010,
    d_max: float = 0.08,
    d_step: float = 0.002,
):
    """Smallest retraction ``d in [0, d_max]`` (stepped by ``d_step``) that clears the below object.

    ``gripper(d) = (T_grasp . gripper_pts_eef) - d * approach_hat``. Returns the first ``d`` at which
    ``min_surface_dist(gripper(d), below) >= margin`` *iff* at that same ``d``
    ``min_surface_dist(gripper(d), top) <= contact_tol`` (the pick still touches the top object);
    otherwise ``None`` (clearing the below object has lost contact with the top, or no ``d <= d_max``
    clears the below object at all).
    """
    pts = np.asarray(gripper_pts_eef, dtype=float).reshape(-1, 3)
    T = np.asarray(T_grasp, dtype=float)
    world = (T[:3, :3] @ pts.T).T + T[:3, 3]

    a = np.asarray(approach, dtype=float)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        raise ValueError("approach axis must be non-zero")
    a_hat = a / n

    n_steps = int(round(d_max / d_step))
    for k in range(n_steps + 1):
        d = k * d_step
        g = world - d * a_hat
        if min_surface_dist(g, below_mesh_world) >= margin:
            # first d clearing the below object; the top distance only grows past here, so this d has
            # the best chance of still touching the top -> accept it or give up.
            if min_surface_dist(g, top_mesh_world) <= contact_tol:
                return float(d)
            return None
    return None


# --- real-mesh loading (trimesh) ---------------------------------------------
#
# Frame conventions match the annotation pipeline (see annotation/validate_grasps.py):
#   * the gripper GLB verts are EEF-LOCAL — transform by the grasp's eef world pose.
#   * each object GLB is OBJECT-LOCAL — transform by the object's live (pos, quat_xyzw).
# quat is xyzw; scipy ``Rotation.from_quat`` (scalar-last) matches OmniGibson's convention
# here (kept sim-free — no ``omnigibson.utils.transform_utils`` import).


def _load_mesh(path):
    import trimesh

    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


@functools.lru_cache(maxsize=1)
def _gripper_verts_full() -> np.ndarray:
    """ALL eef-local gripper vertices (``gripper_longfinger.glb``) — for exact extents."""
    return np.asarray(_load_mesh(ANN_DIR / "gripper_longfinger.glb").vertices, dtype=float)


@functools.lru_cache(maxsize=1)
def gripper_verts() -> np.ndarray:
    """EEF-local gripper vertices capped to 2000 (deterministic) — for the proximity line-search."""
    v = _gripper_verts_full()
    if len(v) > 2000:
        idx = np.random.RandomState(0).choice(len(v), 2000, replace=False)
        v = v[idx]
    return v


def gripper_drop_below_eef(quat_xyzw) -> float:
    """Downward (world −z) extent of the LOWEST gripper vertex below eef_link, for a grasp at
    orientation ``quat_xyzw`` (xyzw). ``= −min_v (R·v)_z`` over the FULL gripper mesh, so it captures
    the whole gripper (fingertips, finger breadth, rail) AND any grasp tilt. A transfer height of
    ``object_top + this + margin`` guarantees the entire gripper clears the object top by ``margin``."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    world_z = _gripper_verts_full() @ R[2, :]     # (R·v)_z for every vertex
    return float(-world_z.min())


@functools.lru_cache(maxsize=1)
def rail_half() -> float:
    """Gripper's max horizontal (perp-to-approach = eef X/Y) half-extent from the GLB (~0.10 m).

    The finger-rail sticks out this far to the side of the eef; the stack re-pile must clear the
    source by at least this (not just the object footprint) or the rail clips the residual source
    stack during the place-descent. Measured from the FULL mesh bounds (not the subsampled verts).
    """
    b = _load_mesh(ANN_DIR / "gripper_longfinger.glb").bounds  # (2,3) eef-local AABB
    span_xy = np.asarray(b[1] - b[0], dtype=float)[:2]
    return 0.5 * float(span_xy.max())


@functools.cache
def _object_mesh_local(key: str):
    """Cached OBJECT-LOCAL trimesh for ``key`` = ``"category/model"``."""
    cat, model = key.split("/")
    return _load_mesh(ANN_DIR / "meshes" / f"{cat}__{model}.glb")


def object_mesh_world(key, pos_xyz, quat_xyzw):
    """A COPY of ``key``'s mesh transformed to world by ``(pos_xyz, quat_xyzw)`` (xyzw)."""
    from scipy.spatial.transform import Rotation

    m = _object_mesh_local(key).copy()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=float).reshape(3)
    m.apply_transform(T)
    return m


def instance_descend_offset(
    target_key,
    T_grasp,
    approach,
    below_key,
    below_pose,
    top_pose,
    *,
    margin: float = 0.005,
    contact_tol: float = 0.010,
    d_max: float = 0.08,
):
    """Shallow-grab retraction ``d`` for one stack instance, from real GLB meshes.

    ``target_key``/``below_key`` = ``"category/model"``; ``top_pose``/``below_pose`` =
    ``(pos_xyz, quat_xyzw)`` live poses of the top (grasped) object and the object directly under
    it. Returns the retraction along ``-approach`` (see :func:`shallow_retraction`) or ``None``.
    """
    top = object_mesh_world(target_key, *top_pose)
    below = object_mesh_world(below_key, *below_pose)
    return shallow_retraction(
        gripper_verts(), T_grasp, approach, below, top,
        margin=margin, contact_tol=contact_tol, d_max=d_max,
    )
