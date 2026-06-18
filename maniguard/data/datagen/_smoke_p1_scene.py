"""P1 smoke test: build the empty scene from a real base-task dump.

Verifies the stale-import fix + env construction + robot/objects/settle. Throwaway
harness for Step 1 P1 verification (not part of the datagen API).

  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
  OMNIGIBSON_HEADLESS=1 PYTHONPATH=$HOME/project/ManiGuard \
  python -m maniguard.data.datagen._smoke_p1_scene <task_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from maniguard.data.datagen.primitives.scene import init_omnigibson, scene_from_task_dir


def main() -> int:
    task_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "outputs/pipeline_runs/Rs_int_20260529_205132")

    og = init_omnigibson(headless=True)
    bundle = scene_from_task_dir(task_dir, episode=1)

    env = bundle.env
    scene_objs = {o.name for o in env.scene.objects}
    print("[smoke] robot:", bundle.robot.name)
    print("[smoke] task_names:", bundle.task_names)
    present = [n for n in bundle.task_names if n in scene_objs]
    missing = [n for n in bundle.task_names if n not in scene_objs]
    print(f"[smoke] task objects present in scene: {len(present)}/{len(bundle.task_names)}")
    if missing:
        print("[smoke] MISSING:", missing)
    print("[smoke] goal_spec:", bundle.goal_spec)

    # Step a few times with a zero action; confirm the env is steppable + read back
    # the surface + first task-object world z (sanity: nothing exploded to NaN).
    import numpy as np
    robot = bundle.robot
    action = np.zeros_like(np.asarray(robot.action_space.sample(), dtype=np.float32))
    for _ in range(15):
        env.step(action)
    surf_z = float(bundle.surface.get_position_orientation()[0][2])
    print(f"[smoke] surface z after 15 steps: {surf_z:.4f}")
    for n in bundle.task_names[1:]:
        obj = env.scene.object_registry("name", n)
        if obj is None:
            continue
        z = float(obj.get_position_orientation()[0][2])
        print(f"[smoke]   {n} z={z:.4f}")

    ok = (len(missing) == 0) and bundle.robot is not None and np.isfinite(surf_z)
    print("[smoke] RESULT:", "PASS" if ok else "FAIL")

    try:
        og.sim.stop()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
