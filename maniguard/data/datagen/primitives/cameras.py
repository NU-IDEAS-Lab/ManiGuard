"""Camera configuration and placement helpers for demonstration collection.

Provide four benchmark external-camera configurations, optional wrist-camera
installation, wrist-sensor lookup, and post-construction resolution setup.
External views use the camera poses stored in task diagnostics. The wrist-camera
installer must be invoked before robot sensor discovery when its patch is needed.
"""
from __future__ import annotations

from typing import Any

from maniguard.data.datagen.data_format import RESOLUTION
from maniguard.utils.camera_setup import (
    EXTERNAL_CAMERA_NAMES,
    build_external_camera_configs,
)


def external_camera_configs(resolution: int = RESOLUTION) -> list[dict]:
    """The four bench third-person VisionSensor configs for the env's
    ``external_sensors`` (the ``scene_from_task_dir(external_sensors=)`` seam).

    Resolution is also forced post-construction (``place_and_resize_cameras``)
    because ``sensor_kwargs`` alone is unreliable (StanfordVL/OmniGibson#266/#1875);
    setting it here too keeps the config self-describing.
    """
    return build_external_camera_configs(
        names=EXTERNAL_CAMERA_NAMES,
        resolution=resolution,
        modalities=("rgb",),
    )


# Idempotency guard — the patch rebinds a class method, so install at most once.
_WRIST_CAM_PATCHED = False


def install_wrist_camera() -> None:
    """Monkeypatch ``FrankaPanda._load_sensors`` to (re)place a wrist Camera under
    ``panda_hand`` before sensor discovery (the ``pre_build_hooks`` seam). Idempotent.

    The Franka USD ships a wrist Camera at translate ``(0.05, 0, -0.05)`` (behind the
    hand, toward the wrist joint); with the maniguard longfinger patch the fingers
    reach further in +Z and that stock pose frames the back of the gripper. Flip Z to
    ``+0.05`` so the camera sits between wrist and fingertips, looking out at the
    grasp zone. If the USD shipped no wrist Camera, create one. Pose copied verbatim
    from ``franka_mounted.usda`` so the wrist view matches task-generation.
    """
    global _WRIST_CAM_PATCHED
    if _WRIST_CAM_PATCHED:
        return

    from omnigibson import lazy
    from omnigibson.robots.franka import FrankaPanda

    _orig = FrankaPanda._load_sensors
    target_translate = (0.05, 0.0, 0.05)
    target_orient = (-0.0923, -0.701, -0.701, -0.0923)  # (w, x, y, z)

    def _patched(self):
        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        hand = self._links.get("panda_hand") if self._links else None
        if hand is None:
            return _orig(self)

        cam_path = f"{hand.prim_path}/Camera"
        prim = stage.GetPrimAtPath(cam_path)
        if not prim.IsValid():
            cam_prim = lazy.pxr.UsdGeom.Camera.Define(stage, cam_path)
            prim = cam_prim.GetPrim()
            cam_prim.CreateFocalLengthAttr().Set(17.0)
            cam_prim.CreateClippingRangeAttr().Set(
                lazy.pxr.Gf.Vec2f(0.001, 1000000.0))
            print(f"[datagen.cameras] created wrist Camera at {cam_path}", flush=True)
        else:
            print(f"[datagen.cameras] using existing wrist Camera at {cam_path}",
                  flush=True)

        xf = lazy.pxr.UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        t_op = xf.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeTranslate,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        t_op.Set(lazy.pxr.Gf.Vec3d(*target_translate))
        r_op = xf.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeOrient,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        r_op.Set(lazy.pxr.Gf.Quatd(*target_orient))
        s_op = xf.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeScale,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        s_op.Set(lazy.pxr.Gf.Vec3d(1.0, 1.0, 1.0))
        xf.SetXformOpOrder([t_op, r_op, s_op])
        return _orig(self)

    FrankaPanda._load_sensors = _patched
    _WRIST_CAM_PATCHED = True


def find_wrist_sensor(robot) -> Any | None:
    """Return the hand-mounted VisionSensor, another robot VisionSensor as a fallback, or None."""
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


def place_and_resize_cameras(env, robot, og, diagnostics: dict,
                             resolution: int = RESOLUTION) -> int:
    """Post-construction camera finalize: position the four third-person cameras at
    the poses RECORDED in ``diagnostics["cameras"]`` (each matched to its
    ``sensor_name``), then force every VisionSensor (externals + wrist) to
    ``resolution``² and rebuild the obs space. Returns the number of cameras placed.

    Reading the recorded poses (vs recomputing robot-frame) keeps the datagen views
    byte-for-byte consistent with the bench render + eval. Call once after
    ``scene_from_task_dir`` returns. The explicit setter + ``load_observation_space``
    is the documented resolution workaround (``sensor_kwargs`` alone is unreliable —
    StanfordVL/OmniGibson#266/#1875).
    """
    from omnigibson.sensors import VisionSensor

    from maniguard.envs.frozen_task_runtime import position_diagnostics_cameras

    placed = position_diagnostics_cameras(env, og, diagnostics, set_viewer=True)
    expected = len([c for c in (diagnostics.get("cameras") or []) if c.get("sensor_name")])
    if placed != expected:
        print(f"[datagen.cameras] WARNING: positioned {placed}/{expected} recorded "
              f"third-person cameras (sensor_name mismatch?)", flush=True)

    for cam in (env.external_sensors or {}).values():
        cam.image_height = int(resolution)
        cam.image_width = int(resolution)
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = int(resolution)
            sensor.image_width = int(resolution)
    env.load_observation_space()
    return placed
