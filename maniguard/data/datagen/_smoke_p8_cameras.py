"""P8 smoke test: 5-stream camera rig (4 bench third-person + injected wrist).

Verifies the P1 seams (external_sensors + pre_build_hooks), wrist injection, robot-
frame placement, and 256x256 resolution enforcement by grabbing one frame from each
of the five streams. Throwaway harness for Step 1 P8 verification.

  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
  OMNIGIBSON_HEADLESS=1 PYTHONPATH=$HOME/project/ManiGuard \
  python -m maniguard.data.datagen._smoke_p8_cameras <task_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from maniguard.data.datagen.data_format import RESOLUTION, THIRD_PERSON_CAMS
from maniguard.data.datagen.primitives import cameras, scene


def _grab_rgb(sensor):
    if sensor is None:
        return None
    obs = sensor.get_obs()
    rgb = obs[0].get("rgb") if isinstance(obs, tuple) else obs.get("rgb")
    if rgb is None:
        return None
    rgb = rgb[..., :3]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.asarray(rgb)


def main() -> int:
    task_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "outputs/lerobot_datasets/maniguard-bench/clutter_pickup/task_0000/base")

    og = scene.init_omnigibson(headless=True)
    bundle = scene.scene_from_task_dir(
        task_dir,
        episode=1,
        external_sensors=cameras.external_camera_configs(),
        pre_build_hooks=[cameras.install_wrist_camera],
    )
    placed = cameras.place_and_resize_cameras(
        bundle.env, bundle.robot, bundle.og, bundle.diagnostics)
    print(f"[smoke] positioned {placed} recorded third-person cameras")

    env, robot = bundle.env, bundle.robot
    # Render ONLY — never env.step(action) here. The joint_position_impedance
    # controller reads the action as an ABSOLUTE joint target, so a zero action
    # commands the arm to the joint-zero pose and destroys the task's designed
    # init pose. Datagen starts from + holds the init pose; only curobo-planned
    # trajectories ever move the arm. og.sim.render() advances rendering without
    # issuing any controller command (the impedance drive holds the reset pose).
    for _ in range(3):
        og.sim.render()

    ok = True

    # Four third-person externals: present, positioned (not at origin), 256x256x3.
    print("[smoke] external cameras:")
    for og_name in THIRD_PERSON_CAMS.values():
        sensor = (env.external_sensors or {}).get(og_name)
        if sensor is None:
            print(f"[smoke]   {og_name}: MISSING"); ok = False; continue
        pos = sensor.get_position_orientation()[0]
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        rgb = _grab_rgb(sensor)
        shape = None if rgb is None else tuple(rgb.shape)
        nonzero = bool(rgb is not None and np.any(rgb))
        moved = bool(np.linalg.norm(pos) > 0.1)
        good = (shape == (RESOLUTION, RESOLUTION, 3)) and nonzero and moved
        ok = ok and good
        print(f"[smoke]   {og_name}: shape={shape} nonzero={nonzero} "
              f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) {'OK' if good else 'BAD'}")

    # Wrist: injected under panda_hand, 256x256x3, rendering.
    wrist = cameras.find_wrist_sensor(robot)
    wname = getattr(wrist, "name", None)
    wrgb = _grab_rgb(wrist)
    wshape = None if wrgb is None else tuple(wrgb.shape)
    wnonzero = bool(wrgb is not None and np.any(wrgb))
    wgood = (wrist is not None) and (wshape == (RESOLUTION, RESOLUTION, 3)) and wnonzero
    ok = ok and wgood
    print(f"[smoke] wrist: name={wname} shape={wshape} nonzero={wnonzero} "
          f"{'OK' if wgood else 'BAD'}")

    # 5-pane montage (4 third-person + wrist) for visual review.
    try:
        import imageio.v2 as imageio
        panes = []
        for og_name in THIRD_PERSON_CAMS.values():
            s = (env.external_sensors or {}).get(og_name)
            rgb = _grab_rgb(s)
            panes.append(np.zeros((RESOLUTION, RESOLUTION, 3), np.uint8)
                         if rgb is None else rgb.astype(np.uint8))
        wrgb2 = _grab_rgb(cameras.find_wrist_sensor(robot))
        panes.append(np.zeros((RESOLUTION, RESOLUTION, 3), np.uint8)
                     if wrgb2 is None else wrgb2.astype(np.uint8))
        montage = np.concatenate(panes, axis=1)
        out = Path("/tmp/p8_montage.png")
        imageio.imwrite(str(out), montage)
        print(f"[smoke] montage saved: {out}  "
              f"(order: opposite | left | right | left_shoulder | wrist)")
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] montage save skipped: {e}")

    print("[smoke] RESULT:", "PASS" if ok else "FAIL")
    try:
        og.sim.stop()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
