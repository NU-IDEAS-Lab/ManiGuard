"""OBB-based grasp-pose sampler — Layer-1 primitive (family-agnostic geometry).

"Where should the two-finger gripper grab this object?" Models the open gripper as
five oriented boxes in the grasp frame (perp=X, closing=Y, approach=Z):

  left_finger / right_finger / palm   must be EMPTY of object material (open gripper
                                      straddles the object without pre-colliding)
  swept (the AG raycast firing zone   must be NON-EMPTY (closing the fingers WILL
   corridor between the fingers)       capture object material)

Candidate poses are seeded from convex-hull support points + uniform anchors, with
the approach direction sampled in a cone around the inward surface normal. Each pose
is scored by how much object material the swept box captures AND — NEW vs the
reference sampler — how CENTERED that material is along the closing axis, so the two
fingers contact the object near-simultaneously on closing. (An off-center grasp lets
the first-contacting finger shove the object out of the gripper before the second
finger lands — the assisted-grasp raycast then misses and the grasp fails; this was
the common push-away failure in teleop.)

Replicated clean from the team's `rl/grasps/obb_sampler.py` (SimonZhan) + `mesh.py`,
with the legacy/unused fields dropped and the centering criterion added. Franka
constants are calibrated for the active `franka_panda_longfinger` asset + our AG
raycast patch (`_omnigibson_patches.LONG_AG_Z`). datagen does not import the rl tree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Franka longfinger gripper geometry (metres) -----------------------------
_MAX_OPENING = 0.070
_FINGER_LEN = 0.145
_FINGER_BREAD = 0.038        # along perp
_FINGER_THICK = 0.040        # along closing (slab thickness)
_EEF_TO_TIP = 0.098          # eef_link -> fingertip along approach
_PALM_HALF_LEN = 0.030       # along approach
_PALM_HALF_WIDTH = 0.046     # along closing (spans the gripper)
_PALM_HALF_BREAD = 0.032     # along perp
# AG raycast firing zone in eef-frame z (matches _omnigibson_patches LONG_AG_Z;
# eef_link sits at finger-local z = +0.0466). Outside this band AG can't fire.
_AG_Z_LOW = 0.045 - 0.0466   # ~ -0.0016
_AG_Z_HIGH = 0.140 - 0.0466  # ~ +0.0934


@dataclass
class GraspConfig:
    n_surface_points: int = 6000      # surface cloud for the box-containment test
    n_support_directions: int = 200   # random dirs -> convex-hull support anchors
    n_poses_per_anchor: int = 1000    # candidate poses per anchor
    n_uniform_anchors: int = 30       # uniform anchors (cover concave regions)
    spread_radius_m: float = 0.02     # Gaussian spread on anchor position
    cone_half_angle_rad: float = 1.05  # approach cone around inward surface normal
    shrink_m: float = 0.002           # finger/palm/swept box inward margin
    max_candidates: int = 400
    center_sigma_m: float = 0.005     # closing-axis centering tolerance (push-away)


# --- mesh extraction (replicated clean from rl/grasps/mesh.py) ----------------
def _to_np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu().numpy()
    return np.asarray(t)


def _pose_to_mat(pos, quat_xyzw) -> np.ndarray:
    import omnigibson.utils.transform_utils as T
    import torch as th

    R = _to_np(T.quat2mat(th.as_tensor(np.asarray(quat_xyzw, dtype=np.float64)))).astype(np.float64)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return M


def mesh_from_og_object(obj, use_visual: bool = True):
    """Merged ``trimesh.Trimesh`` of an OG object in its OBJECT-LOCAL frame."""
    import trimesh

    root_pos = _to_np(obj.get_position_orientation()[0]).astype(np.float64).reshape(3)
    root_quat = _to_np(obj.get_position_orientation()[1]).astype(np.float64).reshape(4)
    T_world_root = np.linalg.inv(_pose_to_mat(root_pos, root_quat))

    parts = []
    for link in obj.links.values():
        mesh_dict = link.visual_meshes if use_visual else link.collision_meshes
        for geom in mesh_dict.values():
            if geom.geom_type != "Mesh":
                continue
            faces = geom.faces
            if faces is None or len(faces) == 0 or len(geom.points) == 0:
                continue
            pts_world = _to_np(geom.transform_local_points_to_world(geom.points)).astype(np.float64).reshape(-1, 3)
            tm = trimesh.Trimesh(vertices=pts_world,
                                 faces=_to_np(faces).astype(np.int64).reshape(-1, 3),
                                 process=False)
            tm.apply_transform(T_world_root)
            parts.append(tm)
    if not parts:
        raise ValueError(f"Object {obj.name} has no mesh-type {'visual' if use_visual else 'collision'} geoms.")
    return trimesh.util.concatenate(parts)


def gather_obstacle_surf_local(env, target_obj, points_per_obstacle: int = 1500,
                               rng_seed: int = 0) -> np.ndarray:
    """Sample every non-target, non-robot object's surface into the TARGET-local
    frame (for the OBB obstacle-aware reject). Returns ``(M, 3)`` or ``(0, 3)``."""
    import trimesh

    t_pos, t_quat = target_obj.get_position_orientation()
    T_w_to_local = np.linalg.inv(_pose_to_mat(_to_np(t_pos), _to_np(t_quat)))
    rng = np.random.default_rng(rng_seed)
    robot_set = set(env.robots) if env.robots else set()
    out = []
    for obj in env.scene.objects:
        if obj is target_obj or obj in robot_set:
            continue
        if "goal_region" in (getattr(obj, "name", "") or ""):
            continue
        try:
            obs_mesh = mesh_from_og_object(obj, use_visual=True)
        except Exception:  # noqa: BLE001
            continue
        if obs_mesh is None or len(obs_mesh.vertices) == 0:
            continue
        try:
            n = min(points_per_obstacle, max(200, int(obs_mesh.area * 5000)))
            pts_local, _ = trimesh.sample.sample_surface(obs_mesh, n,
                                                         seed=int(rng.integers(0, 2**31 - 1)))
        except Exception:  # noqa: BLE001
            continue
        o_pos, o_quat = obj.get_position_orientation()
        T_obs_w = _pose_to_mat(_to_np(o_pos), _to_np(o_quat))
        pts_world = (T_obs_w[:3, :3] @ np.asarray(pts_local, np.float64).T).T + T_obs_w[:3, 3]
        pts_tgt = (T_w_to_local[:3, :3] @ pts_world.T).T + T_w_to_local[:3, 3]
        out.append(pts_tgt.astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty((0, 3), dtype=np.float32)


def _gripper_boxes(cfg: GraspConfig):
    """The 5 OBBs (center_offset, half_extents, must_be_nonempty) in the grasp
    frame (perp=X, closing=Y, approach=Z), origin at eef_link."""
    open_half = 0.5 * _MAX_OPENING
    fl, fb, ft, et, s = _FINGER_LEN, _FINGER_BREAD, _FINGER_THICK, _EEF_TO_TIP, cfg.shrink_m
    fz = et - fl / 2
    pz = et - fl - _PALM_HALF_LEN
    sz_c = 0.5 * (_AG_Z_LOW + _AG_Z_HIGH)
    sz_h = 0.5 * (_AG_Z_HIGH - _AG_Z_LOW)
    return {
        "left_finger": (np.array([0.0, -(open_half + ft / 2), fz]),
                        np.array([fb / 2 - s, ft / 2 - s, fl / 2 - s]), False),
        "right_finger": (np.array([0.0, +(open_half + ft / 2), fz]),
                         np.array([fb / 2 - s, ft / 2 - s, fl / 2 - s]), False),
        "palm": (np.array([0.0, 0.0, pz]),
                 np.array([_PALM_HALF_BREAD - s, _PALM_HALF_WIDTH - s, _PALM_HALF_LEN - s]), False),
        "swept": (np.array([0.0, 0.0, sz_c]),
                  np.array([fb / 2 - s, open_half - s, sz_h - s]), True),
    }


def sample_grasp_poses(mesh, config: GraspConfig | None = None,
                       rng: np.random.Generator | None = None,
                       obstacle_surf_local: np.ndarray | None = None,
                       pose_chunk: int = 4000):
    """OBB-sample grasp poses for ``mesh`` (trimesh, object-local). Returns
    ``(poses (N,4,4) float32, scores (N,) float32)`` in mesh-local frame, ranked by
    descending score (swept-fill x closing-axis centering)."""
    import trimesh

    cfg = config or GraspConfig()
    if rng is None:
        rng = np.random.default_rng()

    # A. surface cloud.
    surf, _ = trimesh.sample.sample_surface(mesh, cfg.n_surface_points,
                                            seed=int(rng.integers(0, 2**31 - 1)))
    surf = np.asarray(surf, dtype=np.float64)
    n_surf = len(surf)

    # B. anchors: convex-hull support points + uniform anchors.
    V = np.asarray(mesh.vertices, dtype=np.float64)
    dirs = rng.standard_normal(size=(cfg.n_support_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    extreme = (dirs @ (V - V.mean(0)).T).argmax(axis=1)
    anchors = V[np.unique(extreme)]
    if cfg.n_uniform_anchors > 0:
        ua, _ = trimesh.sample.sample_surface(mesh, cfg.n_uniform_anchors,
                                              seed=int(rng.integers(0, 2**31 - 1)))
        anchors = np.vstack([anchors, np.asarray(ua, dtype=np.float64)])

    from scipy.spatial import cKDTree
    n_per = cfg.n_poses_per_anchor
    N = len(anchors) * n_per
    centers_raw = np.repeat(anchors, n_per, axis=0)
    centers_raw += rng.normal(0.0, cfg.spread_radius_m, size=centers_raw.shape)

    # Project each spread point to the surface (nearest pre-sampled point) + normal.
    proj_pts, proj_face = trimesh.sample.sample_surface(mesh, max(cfg.n_surface_points, 6000),
                                                        seed=int(rng.integers(0, 2**31 - 1)))
    proj_pts = np.asarray(proj_pts, dtype=np.float64)
    proj_normals = mesh.face_normals[proj_face].astype(np.float64)
    proj_normals /= np.linalg.norm(proj_normals, axis=1, keepdims=True) + 1e-12
    _, ni = cKDTree(proj_pts).query(centers_raw, k=1)
    closest_pts, surface_normals = proj_pts[ni], proj_normals[ni]

    # Approach sampled in a cone around the inward normal.
    cos_max = float(np.cos(cfg.cone_half_angle_rad))
    cos_t = rng.uniform(cos_max, 1.0, size=N)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    az = rng.uniform(0.0, 2.0 * np.pi, size=N)
    local_cone = np.stack([sin_t * np.cos(az), sin_t * np.sin(az), cos_t], axis=-1)
    cone_axis = -surface_normals
    helper = np.where(np.abs(cone_axis[:, 0:1]) < 0.9,
                      np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    helper = np.broadcast_to(helper, cone_axis.shape).copy()
    tx = np.cross(helper, cone_axis); tx /= np.linalg.norm(tx, axis=1, keepdims=True) + 1e-12
    ty = np.cross(cone_axis, tx)
    approach = np.einsum("nij,nj->ni", np.stack([tx, ty, cone_axis], axis=-1), local_cone)
    approach /= np.linalg.norm(approach, axis=1, keepdims=True) + 1e-12

    # Closing = random unit perpendicular to approach; perp completes the frame.
    raw2 = rng.standard_normal(size=(N, 3))
    closing = raw2 - np.einsum("ij,ij->i", raw2, approach)[:, None] * approach
    closing /= np.linalg.norm(closing, axis=1, keepdims=True) + 1e-12
    perp = np.cross(closing, approach)

    # eef_link target = surface point shifted back along approach by eef_to_tip.
    mid = closest_pts - _EEF_TO_TIP * approach

    boxes = _gripper_boxes(cfg)
    keep = np.ones(N, dtype=bool)
    swept_count = np.zeros(N, dtype=np.int32)
    swept_yoff = np.zeros(N, dtype=np.float32)   # closing-axis centering offset
    R_all = np.stack([perp, closing, approach], axis=-1)   # (N,3,3)
    r_max = max(float(np.linalg.norm(c)) + float(np.linalg.norm(h)) for c, h, _ in boxes.values()) + 0.005
    r_max_sq = r_max ** 2
    surf32 = surf.astype(np.float32)
    obs32 = (np.asarray(obstacle_surf_local, dtype=np.float32)
             if obstacle_surf_local is not None and len(obstacle_surf_local) > 0 else None)

    for start in range(0, N, pose_chunk):
        end = min(start + pose_chunk, N)
        R = R_all[start:end].astype(np.float32)
        mid_c = mid[start:end].astype(np.float32)
        for bi, b_idx in enumerate(range(start, end)):
            diff = surf32 - mid_c[bi]
            d2 = np.einsum("ij,ij->i", diff, diff)
            near = d2 <= r_max_sq
            if not np.any(near):
                keep[b_idx] = False
                continue
            local_mid = diff[near] @ R[bi]
            obs_local_mid = None
            if obs32 is not None:
                od = obs32 - mid_c[bi]
                onear = np.einsum("ij,ij->i", od, od) <= r_max_sq
                if np.any(onear):
                    obs_local_mid = od[onear] @ R[bi]
            for name, (c_off, h_ext, nonempty) in boxes.items():
                local = local_mid - c_off.astype(np.float32)
                inside = np.all(np.abs(local) <= h_ext.astype(np.float32), axis=-1)
                cnt = int(inside.sum())
                if nonempty:
                    if cnt == 0:
                        keep[b_idx] = False
                        break
                    if name == "swept":
                        swept_count[b_idx] = cnt
                        y = local_mid[inside, 1]   # closing-axis coords of captured pts
                        swept_yoff[b_idx] = float(0.5 * (y.min() + y.max()))
                else:
                    if cnt > 0:
                        keep[b_idx] = False
                        break
                    if obs_local_mid is not None:
                        obs_local = obs_local_mid - c_off.astype(np.float32)
                        if int(np.all(np.abs(obs_local) <= h_ext.astype(np.float32), axis=-1).sum()) > 0:
                            keep[b_idx] = False
                            break

    if not np.any(keep):
        return np.zeros((0, 4, 4), np.float32), np.zeros((0,), np.float32)

    z = approach[keep]
    y = closing[keep]
    y = y - np.einsum("ij,ij->i", y, z)[:, None] * z
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    x = np.cross(y, z)
    centers = mid[keep]

    # Score = swept-fill x closing-axis centering (both fingers contact together).
    s_fill = swept_count[keep] / max(n_surf, 1)
    centering = np.exp(-(np.abs(swept_yoff[keep]) / cfg.center_sigma_m) ** 2)
    scores = (s_fill * centering).astype(np.float32)

    n = len(centers)
    poses = np.zeros((n, 4, 4), dtype=np.float32)
    poses[:, :3, 0], poses[:, :3, 1], poses[:, :3, 2] = x, y, z
    poses[:, :3, 3] = centers
    poses[:, 3, 3] = 1.0
    order = np.argsort(-scores)[: cfg.max_candidates]
    return poses[order], scores[order]
