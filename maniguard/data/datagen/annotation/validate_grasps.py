"""Phase C-lite — load annotated grasps into their object's task, drive the Franka eef
to each grasp pose, hold, and render the 4 bench third-person cameras for review.

For each annotated object: loads its source task (the bench task where it is the grasp
target, longfinger + the 4 bench cameras), finds the target instance, then per grasp:
``T_eef_world = T_object_world @ grasp_local`` and TELEPORTS the robot base so eef_link
lands EXACTLY at that pose (no IK confound — exact placement to verify the grasp POINT),
hides the (non-physical floating) arm links so only the gripper + scene show, opens the
gripper, and saves the 4 third-person views + a montage. Lets you confirm each annotated
grasp grabs the intended part before doing the full annotation.

  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
  OMNIGIBSON_HEADLESS=1 PYTHONPATH=$HOME/project/ManiGuard \
  python -u -m maniguard.data.datagen.annotation.validate_grasps [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANN_DIR = Path("outputs/grasp_annotation")
ANN = ANN_DIR / "grasp_annotations.json"
MESH_DB = ANN_DIR / "mesh_db.json"
OUT = ANN_DIR / "validate"
BENCH = Path("outputs/lerobot_datasets/maniguard-bench")


def _mat2quat_t(R):
    import omnigibson.utils.transform_utils as T
    import torch as th
    return T.mat2quat(th.as_tensor(np.asarray(R), dtype=th.float32)).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap # objects")
    ap.add_argument("--object", default=None,
                    help="process only this object key 'cat/model' (one task per process; "
                         "og.clear() multi-task reload breaks the cameras, so callers loop "
                         "this per object in separate processes)")
    args = ap.parse_args()

    ann = json.load(open(ANN))
    mesh_db = json.load(open(MESH_DB))
    items = [(k, v) for k, v in ann["objects"].items() if v.get("grasps")]
    if args.object:
        items = [(k, v) for k, v in items if k == args.object]
    if args.limit:
        items = items[: args.limit]
    n_g = sum(len(v["grasps"]) for _, v in items)
    print(f"[validate] {len(items)} annotated objects, {n_g} grasps", flush=True)
    if not items:
        print("[validate] nothing annotated yet — annotate in the viser tool first.")
        return 0

    import torch as th
    from omnigibson import lazy
    from scipy.spatial.transform import Rotation as Rot

    from maniguard._omnigibson_patches import _patch_franka_longfinger
    from maniguard.data.datagen import data_format
    from maniguard.data.datagen.annotation.extract_meshes import _target_name
    from maniguard.data.datagen.primitives import cameras
    from maniguard.data.datagen.primitives import scene as scenemod
    from maniguard.data.datagen.primitives.grasp_obb import _pose_to_mat, _to_np
    from maniguard.data.datagen.primitives.record import _sensor_rgb_uint8

    # TEMPORARY review-only close-up (NOT part of utils/camera_setup): rather than ADD a
    # 5th VisionSensor (adding an extra render target crashes the GPU here), we REUSE the
    # existing cam_opposite — grab its wide view, then park it close to the grasp (same
    # viewing direction) for a zoomed shot, then restore it. Repositioning an existing
    # camera is the reliable path.
    CLOSEUP_DIST = 0.30

    def _opposite_cam(diag):
        """(unit viewing dir, recorded eye, recorded orientation) of cam_opposite."""
        for c in (diag.get("cameras") or []):
            if c.get("sensor_name") == "cam_opposite" and c.get("eye") and c.get("lookat"):
                eye = np.asarray(c["eye"], float)
                d = np.asarray(c["lookat"], float) - eye
                n = np.linalg.norm(d)
                if n > 1e-6:
                    return d / n, eye, c.get("orientation")
        dd = np.array([0.0, -0.8, -0.6]); dd /= np.linalg.norm(dd)
        return dd, None, None
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    def _glb_verts(path):
        m = trimesh.load(path, force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        return np.asarray(m.vertices)

    def _closeup_rgb(objw, gripw, c, el, az):
        """Object(gray)+gripper(blue) point clouds from the opposite angle, rendered to a
        standalone RGB array — reused by both the per-grasp montage and the per-object
        summary (so the 3D closeup is rendered once)."""
        f = plt.figure(figsize=(4, 4))
        ax = f.add_subplot(111, projection="3d")
        ax.scatter(objw[:, 0], objw[:, 1], objw[:, 2], s=3, c="0.55", alpha=0.6)
        ax.scatter(gripw[:, 0], gripw[:, 1], gripw[:, 2], s=1, c="tab:blue", alpha=0.25)
        for setlim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(c[i] - 0.13, c[i] + 0.13)
        ax.view_init(elev=el, azim=az); ax.axis("off")
        f.tight_layout(pad=0)
        f.canvas.draw()
        arr = np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(f)
        return arr

    og = scenemod.init_omnigibson(headless=True)
    OUT.mkdir(parents=True, exist_ok=True)
    GRIP_V = _glb_verts(ANN_DIR / "gripper_longfinger.glb")   # eef-local gripper verts

    for ti, (key, rec) in enumerate(items):
        src = mesh_db["objects"][key]["source_task"]          # "clutter_pickup/task_0000"
        task_dir = BENCH / src / "base"
        if ti > 0:
            og.clear()
        # NOTE: install_wrist_camera (injects a USD Camera under panda_hand) was found to
        # intermittently trigger Vulkan ERROR_DEVICE_LOST during scene build on this box,
        # and the wrist view is too close to be useful here — so the review uses the 4
        # bench cams + a matplotlib 3D close-up only.
        bundle = scenemod.scene_from_task_dir(
            task_dir, external_sensors=cameras.external_camera_configs(),
            pre_build_hooks=[_patch_franka_longfinger])
        env, robot = bundle.env, bundle.robot
        cameras.place_and_resize_cameras(env, robot, og, bundle.diagnostics)
        opp_dir, _, _ = _opposite_cam(bundle.diagnostics)
        og.sim.render()

        tname = _target_name(bundle.diagnostics)
        target = env.scene.object_registry("name", tname)
        tp, tq = target.get_position_orientation()
        T_obj = _pose_to_mat(_to_np(tp), _to_np(tq))
        OBJ_V = _glb_verts(ANN_DIR / mesh_db["objects"][key]["mesh"])   # object-local
        print(f"[validate] {key}: task {src}, target={tname}", flush=True)

        arm = robot.default_arm
        ep, eq = robot.eef_links[arm].get_position_orientation()
        T_eef_home = _pose_to_mat(_to_np(ep), _to_np(eq))
        bp, bq = robot.get_position_orientation()
        T_base_home = _pose_to_mat(_to_np(bp), _to_np(bq))
        T_base_to_eef = np.linalg.inv(T_base_home) @ T_eef_home

        # open the gripper so the straddle is visible
        try:
            garm = robot.gripper_control_idx[arm]
            full = robot.get_joint_positions().clone()
            for gi in garm:
                full[gi] = float(robot.joint_upper_limits[gi])
            robot.set_joint_positions(full)
        except Exception as e:  # noqa: BLE001
            print(f"[validate] (gripper-open skip: {e})", flush=True)

        # hide non-gripper arm links (teleport leaves a non-physical floating arm)
        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        keep = {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
        for ln, link in robot.links.items():
            if ln in keep:
                continue
            prim = stage.GetPrimAtPath(link.prim_path)
            if prim and prim.IsValid():
                lazy.pxr.UsdGeom.Imageable(prim).MakeInvisible()

        obj_summary = []          # (opposite-cam frame, closeup array, caption) per grasp
        for gr in rec["grasps"]:
            T_local = np.eye(4)
            T_local[:3, :3] = Rot.from_quat(gr["orientation_xyzw"]).as_matrix()
            T_local[:3, 3] = gr["position"]
            T_eef_world = T_obj @ T_local
            T_base_new = T_eef_world @ np.linalg.inv(T_base_to_eef)
            robot.set_position_orientation(
                position=th.tensor(T_base_new[:3, 3], dtype=th.float32),
                orientation=_mat2quat_t(T_base_new[:3, :3]))
            robot.keep_still()
            for _ in range(3):
                og.sim.render()
            e2, _ = robot.eef_links[arm].get_position_orientation()
            err = float(np.linalg.norm(_to_np(e2) - T_eef_world[:3, 3]))

            sens = env.external_sensors or {}
            frames = {k: _sensor_rgb_uint8(sens.get(og_name), data_format.RESOLUTION)
                      for k, og_name in data_format.THIRD_PERSON_CAMS.items()}
            d = OUT / key.replace("/", "__") / f"grasp_{gr['id']}"
            d.mkdir(parents=True, exist_ok=True)
            for k, img in frames.items():
                if img is not None and np.asarray(img).ndim == 3:
                    imageio.imwrite(d / f"{k}.png", img)

            # reliable close-up = matplotlib 3D of the object + gripper meshes at the
            # grasp, viewed from cam_opposite's angle (sim cameras don't re-render after a
            # mid-run reposition in headless, so we draw the meshes instead).
            objw = (T_obj[:3, :3] @ OBJ_V.T).T + T_obj[:3, 3]
            gripw = (T_eef_world[:3, :3] @ GRIP_V.T).T + T_eef_world[:3, 3]
            c = T_eef_world[:3, 3]
            az = float(np.degrees(np.arctan2(-opp_dir[1], -opp_dir[0])))
            el = float(np.degrees(np.arcsin(np.clip(-opp_dir[2], -1, 1))))

            closeup = _closeup_rgb(objw, gripw, c, el, az)     # rendered once, reused below

            order = list(data_format.THIRD_PERSON_CAMS)   # 4 bench views
            fig = plt.figure(figsize=(14, 8.6))
            for si, k in enumerate(order):
                ax = fig.add_subplot(2, 3, si + 1)
                img = frames.get(k)
                if img is not None and np.asarray(img).ndim == 3:
                    ax.imshow(img)
                ax.set_title(k, fontsize=10); ax.axis("off")
            ax3 = fig.add_subplot(2, 3, (5, 6))
            ax3.imshow(closeup); ax3.axis("off")
            ax3.set_title("CLOSEUP: object(gray)+gripper(blue), opposite angle",
                          fontsize=10)
            cap = (f"#{gr['id']} [{gr.get('approach_hint', '')}/{gr.get('label', '')}] "
                   f"{err * 1000:.0f}mm")
            fig.suptitle(f"{key}  grasp#{gr['id']}  "
                         f"[{gr.get('approach_hint', '')}/{gr.get('label', '')}]  "
                         f"eef_err={err * 1000:.1f}mm", fontsize=12)
            fig.tight_layout()
            fig.savefig(d / "montage.png", dpi=110)
            plt.close(fig)
            obj_summary.append((frames.get("image_opposite"), closeup, cap))
            print(f"[validate]   grasp#{gr['id']}: eef_err={err * 1000:.1f}mm -> "
                  f"{d}/montage.png", flush=True)

        # per-object review grid: all grasps in ONE image (top = opposite bench view,
        # bottom = 3D closeup), same review style — one clear sheet per target object.
        valid = [(o, cl, cp) for (o, cl, cp) in obj_summary if cl is not None]
        if valid:
            ncol = len(valid)
            figs = plt.figure(figsize=(3.1 * ncol, 6.6))
            for j, (opp, cls, cp) in enumerate(valid):
                a1 = figs.add_subplot(2, ncol, j + 1)
                if opp is not None and np.asarray(opp).ndim == 3:
                    a1.imshow(opp)
                a1.set_title(cp, fontsize=9); a1.axis("off")
                a2 = figs.add_subplot(2, ncol, ncol + j + 1)
                a2.imshow(cls); a2.axis("off")
            figs.suptitle(f"{key}   ({ncol} grasps)   top = opposite cam · "
                          f"bottom = 3D closeup", fontsize=12)
            figs.tight_layout()
            spath = OUT / key.replace("/", "__") / "_summary.png"
            figs.savefig(spath, dpi=120)
            plt.close(figs)
            print(f"[validate] {key}: per-object summary ({ncol} grasps) -> {spath}",
                  flush=True)

    print("[validate] DONE", flush=True)
    try:
        og.sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
