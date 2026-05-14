"""SFT recorder shared between pick_and_place_from_dataset and render_pnp_for_sft.

Provides:
  - ``install_wrist_camera_patch`` — monkey-patch FrankaPanda._load_sensors so
    a Camera prim is injected under ``panda_hand`` before sensor discovery.
    The pose is copied verbatim from ``franka_mounted.usda`` so the captured
    wrist view matches the one the task-generation pipeline already uses.
  - ``SFTRecorder`` — captures (image_left, image_right, wrist, state, action)
    per env.step into an HDF5 plus three review MP4s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_WRIST_CAM_PATCHED = False


def install_wrist_camera_patch() -> None:
    """Attach a Camera prim under ``panda_hand`` before sensor discovery.

    Mimics ``behavior-1k/datasets/omnigibson-robot-assets/models/franka/
    franka_mounted/usd/franka_mounted.usda`` lines around 2918. Idempotent.
    """
    global _WRIST_CAM_PATCHED
    if _WRIST_CAM_PATCHED:
        return
    from omnigibson.robots.franka import FrankaPanda
    import omnigibson.lazy as lazy

    _orig = FrankaPanda._load_sensors

    def _patched(self):
        try:
            stage = lazy.isaacsim.core.utils.stage.get_current_stage()
            hand = self._links.get("panda_hand") if self._links else None
            if hand is not None:
                cam_path = f"{hand.prim_path}/Camera"
                if not stage.GetPrimAtPath(cam_path).IsValid():
                    cam = lazy.pxr.UsdGeom.Camera.Define(stage, cam_path).GetPrim()
                    xf = lazy.pxr.UsdGeom.Xformable(cam)
                    xf.ClearXformOpOrder()
                    t_op = xf.AddXformOp(
                        lazy.pxr.UsdGeom.XformOp.TypeTranslate,
                        lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
                    t_op.Set(lazy.pxr.Gf.Vec3d(0.05, 0.0, -0.13))
                    r_op = xf.AddXformOp(
                        lazy.pxr.UsdGeom.XformOp.TypeOrient,
                        lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
                    r_op.Set(lazy.pxr.Gf.Quatd(-0.0923, -0.701, -0.701, -0.0923))
                    s_op = xf.AddXformOp(
                        lazy.pxr.UsdGeom.XformOp.TypeScale,
                        lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
                    s_op.Set(lazy.pxr.Gf.Vec3d(1.0, 1.0, 1.0))
                    xf.SetXformOpOrder([t_op, r_op, s_op])
                    cam.CreateAttribute(
                        "focalLength", lazy.pxr.Sdf.ValueTypeNames.Float
                    ).Set(17.0)
                    cam.CreateAttribute(
                        "clippingRange", lazy.pxr.Sdf.ValueTypeNames.Float2
                    ).Set(lazy.pxr.Gf.Vec2f(0.001, 1000000.0))
                    print(f"[sft_recorder] injected wrist Camera prim at {cam_path}",
                          flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[sft_recorder] wrist-cam injection failed: {e}", flush=True)
        return _orig(self)

    FrankaPanda._load_sensors = _patched
    _WRIST_CAM_PATCHED = True


def _quat2axisangle_np(q_xyzw) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    q = np.clip(q, -1.0, 1.0)
    w = float(q[3])
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(max(-1.0, min(1.0, w))))
    axis = q[:3] / sin_half
    return (axis * angle).astype(np.float32)


def find_wrist_sensor(robot):
    from omnigibson.sensors import VisionSensor
    if not getattr(robot, "sensors", None):
        return None
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor) and "hand" in name.lower():
            return sensor
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            return sensor
    return None


def _sensor_rgb_uint8(sensor, resolution: int) -> np.ndarray:
    try:
        obs_tuple = sensor.get_obs()
        rgb = obs_tuple[0].get("rgb") if isinstance(obs_tuple, tuple) else obs_tuple.get("rgb")
    except Exception:  # noqa: BLE001
        rgb = None
    if rgb is None:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rgb = rgb[..., :3]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach()
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.asarray(rgb, dtype=np.uint8)


def grab_state(robot) -> np.ndarray:
    """8D SFT state: [eef_pos(3) | eef_aa(3) | gripper_q(2)] float32."""
    eef_pos = robot.get_relative_eef_position().detach().cpu().numpy().astype(np.float32)
    eef_quat = robot.get_relative_eef_orientation().detach().cpu().numpy().astype(np.float32)
    eef_aa = _quat2axisangle_np(eef_quat)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    if hasattr(gripper_idx, "cpu"):
        gripper_idx_np = gripper_idx.cpu().numpy()
    else:
        gripper_idx_np = np.asarray(gripper_idx)
    gripper_q = (robot.get_joint_positions().detach().cpu().numpy()[gripper_idx_np]
                 .astype(np.float32))
    return np.concatenate([eef_pos, eef_aa, gripper_q]).astype(np.float32)


class SFTRecorder:
    """Capture (obs, action) per env.step + stream three review MP4s.

    Usage::

        recorder = SFTRecorder(out_dir, resolution=256, fps=30)
        recorder.attach(env, robot)
        # ... in the OSC step loop after env.step(action):
        recorder.record_step(action_np, done=False)
        # at the end:
        recorder.finalize(success=True, attrs={"task_dir": ..., "seed": ...})
    """

    def __init__(self, out_dir, resolution: int = 256, fps: int = 30,
                 ext_cam_names=("cam_left", "cam_right")):
        self.out_dir = Path(out_dir)
        self.resolution = int(resolution)
        self.fps = int(fps)
        self._ext_cam_names = list(ext_cam_names)
        self.env = None
        self.robot = None
        self._left = self._right = self._wrist = None
        self._writers = None
        self._frames_l: list[np.ndarray] = []
        self._frames_r: list[np.ndarray] = []
        self._frames_w: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._dones: list[bool] = []
        self.n_video_frames = 0
        self.n_record_steps = 0

    def attach(self, env, robot) -> None:
        import imageio.v2 as imageio
        self.env = env
        self.robot = robot
        sens = env.external_sensors or {}
        self._left = sens.get(self._ext_cam_names[0])
        self._right = (sens.get(self._ext_cam_names[1])
                       if len(self._ext_cam_names) > 1 else None)
        self._wrist = find_wrist_sensor(robot)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writers = {
            "image_left": imageio.get_writer(
                str(self.out_dir / "rollout_image_left.mp4"),
                fps=self.fps, codec="libx264", quality=7),
            "image_right": imageio.get_writer(
                str(self.out_dir / "rollout_image_right.mp4"),
                fps=self.fps, codec="libx264", quality=7),
            "wrist": imageio.get_writer(
                str(self.out_dir / "rollout_wrist.mp4"),
                fps=self.fps, codec="libx264", quality=7),
        }
        wrist_name = getattr(self._wrist, "name", None) if self._wrist else None
        print(f"[sft_recorder] attached  left={self._ext_cam_names[0]}  "
              f"right={self._ext_cam_names[1] if len(self._ext_cam_names) > 1 else '-'}  "
              f"wrist={wrist_name}", flush=True)

    def _grab_frames(self):
        z = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        l = _sensor_rgb_uint8(self._left, self.resolution) if self._left is not None else z
        r = _sensor_rgb_uint8(self._right, self.resolution) if self._right is not None else z
        w = _sensor_rgb_uint8(self._wrist, self.resolution) if self._wrist is not None else z
        return l, r, w

    def record_video_only(self) -> None:
        """Append a frame to the three MP4s but skip HDF5 buffers (Phase A)."""
        l, r, w = self._grab_frames()
        self._writers["image_left"].append_data(l)
        self._writers["image_right"].append_data(r)
        self._writers["wrist"].append_data(w)
        self.n_video_frames += 1

    def record_step(self, action_np, done: bool = False) -> None:
        l, r, w = self._grab_frames()
        self._writers["image_left"].append_data(l)
        self._writers["image_right"].append_data(r)
        self._writers["wrist"].append_data(w)
        self._frames_l.append(l)
        self._frames_r.append(r)
        self._frames_w.append(w)
        self._states.append(grab_state(self.robot))
        self._actions.append(np.asarray(action_np, dtype=np.float32))
        self._dones.append(bool(done))
        self.n_video_frames += 1
        self.n_record_steps += 1

    def finalize(self, success: bool, attrs: dict | None = None):
        """Close MP4 writers. On success, also write rollout.hdf5. On miss/
        no-steps, delete the empty MP4 files so the seed dir doesn't carry
        zero-byte artifacts."""
        for w in (self._writers or {}).values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        self._writers = None
        if not success or self.n_record_steps == 0:
            for name in ("rollout_image_left.mp4", "rollout_image_right.mp4",
                         "rollout_wrist.mp4"):
                p = self.out_dir / name
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:  # noqa: BLE001
                        pass
            return None
        if self._dones:
            self._dones[-1] = True
        return self._write_hdf5(attrs or {})

    def _write_hdf5(self, attrs: dict) -> Path:
        import h5py
        path = self.out_dir / "rollout.hdf5"
        data = {
            "image_left": np.stack(self._frames_l, axis=0),
            "image_right": np.stack(self._frames_r, axis=0),
            "wrist_image": np.stack(self._frames_w, axis=0),
            "state": np.stack(self._states, axis=0),
            "action": np.stack(self._actions, axis=0),
            "done": np.array(self._dones, dtype=bool),
        }
        with h5py.File(str(path), "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["n_steps"] = int(self.n_record_steps)
            f.attrs["n_video_frames"] = int(self.n_video_frames)
            grp = f.create_group("data/demo_0")
            obs = grp.create_group("obs")
            obs.create_dataset("image_left", data=data["image_left"],
                               compression="gzip", compression_opts=4)
            obs.create_dataset("image_right", data=data["image_right"],
                               compression="gzip", compression_opts=4)
            obs.create_dataset("wrist_image", data=data["wrist_image"],
                               compression="gzip", compression_opts=4)
            obs.create_dataset("state", data=data["state"])
            grp.create_dataset("action", data=data["action"])
            grp.create_dataset("done", data=data["done"])
        print(f"[sft_recorder] wrote {path}  ({self.n_record_steps} steps)",
              flush=True)
        return path
