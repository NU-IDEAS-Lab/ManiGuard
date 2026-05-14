"""Extract a trimesh from a live OmniGibson BaseObject.

Must be called inside the ``behavior`` conda env while OG sim is running
(needs USD / prim access). Returns the object's visual mesh in **object-local
frame** so it can be cached to disk and reused for antipodal grasp sampling
independent of the object's placement in any given scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh as trimesh_t


def _to_numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def _quat_to_4x4(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from (pos, quat xyzw)."""
    import trimesh.transformations as tra

    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    # trimesh uses (w, x, y, z) order for quaternion_matrix
    T = tra.quaternion_matrix([w, x, y, z])
    T[:3, 3] = pos.astype(np.float64)
    return T


def mesh_from_og_object(obj, use_visual: bool = True) -> "trimesh_t.Trimesh":
    """Extract a single merged trimesh from an OG BaseObject.

    Args:
        obj: an OG ``BaseObject`` that's been loaded into a running sim.
        use_visual: if True, use ``link.visual_meshes``; otherwise
            ``link.collision_meshes``. Visual meshes are more detailed
            (better for grasp sampling), collision meshes are coarser but
            match what PhysX actually contacts.

    Returns:
        ``trimesh.Trimesh`` with vertices in **object-local frame** (relative
        to the object's root pose at call time), so the mesh is scene-invariant
        and can be cached to a PLY file.
    """
    import trimesh

    root_pos = _to_numpy(obj.get_position_orientation()[0]).astype(np.float64).reshape(3)
    root_quat = _to_numpy(obj.get_position_orientation()[1]).astype(np.float64).reshape(4)
    T_root_world = _quat_to_4x4(root_pos, root_quat)
    T_world_root = np.linalg.inv(T_root_world)

    parts: list[trimesh.Trimesh] = []
    for link_name, link in obj.links.items():
        mesh_dict = link.visual_meshes if use_visual else link.collision_meshes
        for geom_name, geom in mesh_dict.items():
            if geom.geom_type != "Mesh":
                # Skip primitive shapes; could be added via mesh_prim_shape_to_trimesh_mesh if needed.
                continue
            faces = geom.faces
            if faces is None or len(faces) == 0 or len(geom.points) == 0:
                continue

            # Transform geom-local points to world WITH scale applied. OG stores
            # mesh points as raw USD attrs (unit-scaled); ``transform_local_points_to_world``
            # composes the prim's LocalToWorld xform (incl. scale) so we get
            # real-world metric coordinates.
            pts_world_np = _to_numpy(geom.transform_local_points_to_world(geom.points)).astype(np.float64).reshape(-1, 3)
            faces_np = _to_numpy(faces).astype(np.int64).reshape(-1, 3)

            tm = trimesh.Trimesh(vertices=pts_world_np, faces=faces_np, process=False)
            # World → object-local (rotation + translation only; scale is already
            # baked into pts_world, and object.get_position_orientation() returns
            # the rotation+translation of the root frame at scale 1).
            tm.apply_transform(T_world_root)
            parts.append(tm)

    if not parts:
        raise ValueError(f"Object {obj.name} has no mesh-type {'visual' if use_visual else 'collision'} geoms.")
    return trimesh.util.concatenate(parts)




