"""
Quick demo: load external USD assets (desk + cup) from ~/.molmospaces/usd
into an empty OmniGibson scene with a ground plane.

These THOR assets use plain meshes with PhysicsRigidBodyAPI, not OmniGibson's
link/joint structure, so we load them via the USD stage API directly.

Usage:
    conda activate behavior
    python -m omnigibson.examples.environments.molmospaces_usd_demo [--save-video]
"""
import argparse
import os
import glob as glob_mod

import numpy as np
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------

MOLMO_USD_ROOT = os.path.expanduser("~/.molmospaces/usd/objects/thor/20260128")


def _find_asset(pattern):
    """Return (name, usd_path) for the first asset matching a glob pattern."""
    matches = sorted(glob_mod.glob(os.path.join(MOLMO_USD_ROOT, pattern)))
    for d in matches:
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        usda = os.path.join(d, f"{name}.usda")
        if os.path.isfile(usda):
            return name, usda
    raise FileNotFoundError(f"No USD asset found for pattern: {pattern}")


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------

def _pxr():
    """Lazy import of pxr (only available after Isaac Sim init)."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    return Usd, UsdGeom, UsdPhysics, Gf


def add_usd_reference(stage, prim_path, usd_path):
    """Add a USD file as a reference at the given prim path."""
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    return prim


def get_prim_aabb(stage, prim_path):
    """Compute world-space AABB for a prim and its descendants."""
    Usd, UsdGeom, _, _ = _pxr()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
    )
    prim = stage.GetPrimAtPath(prim_path)
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    return np.array(rng.GetMin()), np.array(rng.GetMax())


def set_prim_xform(stage, prim_path, translate=None, orient_quat=None):
    """Set translate and/or orientation on an Xformable prim."""
    _, UsdGeom, _, Gf = _pxr()
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xformable.ClearXformOpOrder()
    if translate is not None:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if orient_quat is not None:
        xformable.AddOrientOp().Set(
            Gf.Quatd(orient_quat[3], orient_quat[0], orient_quat[1], orient_quat[2])
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load molmospaces USD assets demo")
    parser.add_argument("--desk", default="Coffee_Table_201_*",
                        help="Glob pattern for desk/table asset")
    parser.add_argument("--cup", default="Cup_1",
                        help="Glob pattern for cup asset")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()

    # Resolve assets before launching the simulator.
    desk_name, desk_usd = _find_asset(args.desk)
    cup_name, cup_usd = _find_asset(args.cup)
    print(f"[Demo] Desk: {desk_name}  →  {desk_usd}")
    print(f"[Demo] Cup:  {cup_name}  →  {cup_usd}")

    # ---- Launch OmniGibson (minimal, for physics + rendering) ----
    import omnigibson as og
    from omnigibson.macros import gm

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False

    cfg = dict(
        scene=dict(type="Scene"),
        robots=[],
        objects=[],
    )
    og.Environment(configs=cfg)

    # Ground plane.
    og.sim.add_ground_plane(floor_plane_visible=True)
    og.sim.step()

    stage = og.sim.stage
    Usd, UsdGeom, UsdPhysics, _ = _pxr()

    # ---- Add desk + cup references (physics stripped initially) ----
    desk_path = "/World/demo_desk"
    cup_path = "/World/demo_cup"
    add_usd_reference(stage, desk_path, desk_usd)
    add_usd_reference(stage, cup_path, cup_usd)

    # Strip all physics APIs so PhysX doesn't choke on the THOR structure.
    for root_path in [desk_path, cup_path]:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if prim.HasAPI(UsdPhysics.MassAPI):
                prim.RemoveAPI(UsdPhysics.MassAPI)

    og.sim.step()  # safe now — no physics on these prims

    # ---- Position desk (bottom at z=0) ----
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_],
    )
    desk_bbox = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(desk_path))
    desk_range = desk_bbox.ComputeAlignedRange()
    desk_min_z = float(desk_range.GetMin()[2])
    set_prim_xform(stage, desk_path, translate=(0.0, 0.0, -desk_min_z))
    og.sim.step()

    # Re-read desk top after shift.
    bbox_cache.Clear()
    desk_bbox = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(desk_path))
    desk_top_z = float(desk_bbox.ComputeAlignedRange().GetMax()[2])
    print(f"[Demo] Desk placed: top_z={desk_top_z:.3f}m")

    # ---- Position cup (on top of desk + 2cm) ----
    bbox_cache.Clear()
    cup_bbox = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(cup_path))
    cup_min_z = float(cup_bbox.ComputeAlignedRange().GetMin()[2])
    cup_start_z = desk_top_z - cup_min_z + 0.02
    set_prim_xform(stage, cup_path, translate=(0.0, 0.0, cup_start_z))
    og.sim.step()
    print(f"[Demo] Cup placed at z={cup_start_z:.3f}m (desk top + 2cm)")

    # ---- Camera (looking at the desk from a 45° angle) ----
    cam = og.sim.viewer_camera
    # Compute look-at quaternion: camera at (1,1,0.8) looking toward origin.
    eye = np.array([1.0, 1.0, 0.8])
    target = np.array([0.0, 0.0, desk_top_z * 0.5])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    # Build rotation matrix (Z-up convention).
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # Rotation matrix → quaternion.
    rot = np.stack([right, up, -fwd], axis=1)  # columns
    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_matrix(rot).as_quat()  # (x,y,z,w)
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    cam.set_position_orientation(position=eye.tolist(), orientation=quat_wxyz)

    # ---- Video writer (PyAV, matches pipeline convention) ----
    video_writer = None
    video_path = None
    if args.save_video:
        try:
            import av
            run_dir = os.path.join("outputs", "molmospaces_demo")
            os.makedirs(run_dir, exist_ok=True)
            video_path = os.path.join(run_dir, "demo.mp4")
            container = av.open(video_path, mode="w")
            stream = container.add_stream("h264", rate=args.video_fps)
            stream.width = 1280
            stream.height = 720
            stream.pix_fmt = "yuv420p"
            video_writer = {"container": container, "stream": stream}
            print(f"[Demo] Recording video → {video_path}")
        except ImportError:
            print("[Demo] PyAV not available — video disabled.")

    # ---- Step physics ----
    print(f"[Demo] Running {args.steps} steps …")
    for _ in range(args.steps):
        og.sim.step()
        if video_writer is not None:
            try:
                import av as _av
                rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                frame = _av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for pkt in video_writer["stream"].encode(frame):
                    video_writer["container"].mux(pkt)
            except Exception as exc:
                log.warning("molmospaces_usd_demo: optional av import failed: %s", exc)
                pass

    # ---- Report final state ----
    cup_aabb_min, cup_aabb_max = get_prim_aabb(stage, cup_path)
    cup_center_z = float((cup_aabb_min[2] + cup_aabb_max[2]) / 2)
    print(f"[Demo] Cup final center z={cup_center_z:.3f}m  (desk top={desk_top_z:.3f}m)")
    if cup_center_z > desk_top_z * 0.5:
        print("[Demo] Cup is on the desk.")
    else:
        print("[Demo] Cup fell off the desk!")

    if video_writer is not None:
        import av as _av
        for pkt in video_writer["stream"].encode():
            video_writer["container"].mux(pkt)
        video_writer["container"].close()
        print(f"[Demo] Video saved: {video_path}")

    og.shutdown()


if __name__ == "__main__":
    main()
