#!/usr/bin/env python3
"""Batch grasp-dataset collector across a curated set of objects.

Boots one empty ``Scene`` with a FrankaMounted robot + a fixed support table,
then iterates through ``(category, model)`` targets: spawns each on the table,
runs the OmniReset-style physics-validated grasp pipeline
(``sentinel.rl.grasps.collector``), saves ``grasps_<category>_<model>.pt``
+ ``<category>_<model>.ply``, removes the object, and moves on.

One OG boot amortises across all targets (cuRobo stays warm).

Usage:
    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.grasps.collect_batch \\
            --targets cocktail_glass:fapson mug:wpvubp bowl:ajzltc \\
            --num-target-grasps 10 --max-attempts 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Collect validated grasps for a list of objects in an empty scene.")
    p.add_argument(
        "--targets", nargs="+", required=True,
        help="Space-separated 'category:model' pairs, e.g. 'mug:wpvubp bowl:ajzltc'.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_datasets/batch"))
    p.add_argument("--num-target-grasps", type=int, default=10)
    p.add_argument("--max-attempts", type=int, default=400)
    p.add_argument("--num-surface-samples", type=int, default=64)
    p.add_argument("--num-orientations", type=int, default=16)
    p.add_argument("--table-top-z", type=float, default=0.75,
                   help="Approx world-Z of table top (used as drop height reference).")
    p.add_argument("--object-xy", type=float, nargs=2, default=[0.7, 0.0],
                   help="World-XY of the target drop point (on the table, within Franka reach).")
    p.add_argument("--settle-steps", type=int, default=60,
                   help="Physics steps after drop before mesh extraction + grasping.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-collision-mesh", action="store_true",
                   help="Sample on the collision mesh (coarser) instead of visual.")
    return p.parse_args()


def _parse_targets(raw: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in raw:
        if ":" not in s:
            raise SystemExit(f"--targets entries must be 'category:model', got {s!r}")
        cat, mdl = s.split(":", 1)
        if not cat or not mdl:
            raise SystemExit(f"--targets entry has empty field: {s!r}")
        out.append((cat.strip(), mdl.strip()))
    return out


def _build_env_config() -> dict:
    """Empty Scene + FrankaMounted + DatasetObject breakfast_table support.

    Table z is set so its top surface sits at ~0.74 m (rjgmmy half-height ≈ 0.41 m),
    reachable from a FrankaMounted at the origin (pedestal height ~0.35 m, arm
    reach ~0.85 m).
    """
    return dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[dict(
            type="FrankaMounted",
            name="agent_0",
            obs_modalities=["rgb"],
            action_type="continuous",
            action_normalize=True,
            grasping_mode="physical",
            self_collisions=True,
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            controller_config={
                # OSC is required by the collector's lift phase
                # (drives eef +Z via pose-delta rather than joint teleport).
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        )],
        objects=[dict(
            type="DatasetObject",
            name="support_table",
            category="breakfast_table",
            model="rjgmmy",
            position=[0.8, 0.0, 0.41],
            orientation=[0.0, 0.0, 0.0, 1.0],
            fixed_base=True,
        )],
        task=dict(type="DummyTask"),
    )


def _spawn_target(env, category: str, model: str, xy: tuple[float, float], drop_z: float):
    """Add a DatasetObject to the running scene and settle it with gravity.

    Note: the ``position`` kwarg on the constructor isn't reliably applied when
    adding at runtime (vs via the scene config), so we explicitly call
    set_position_orientation after add_object.
    """
    import torch as th
    from omnigibson.objects import DatasetObject

    name = f"target_{category}_{model}"
    obj = DatasetObject(name=name, category=category, model=model)
    env.scene.add_object(obj)
    obj.set_position_orientation(
        position=th.tensor([xy[0], xy[1], drop_z], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    # Zero any stray velocity from loading.
    obj.root_link.set_linear_velocity(th.zeros(3))
    obj.root_link.set_angular_velocity(th.zeros(3))
    return obj


def _remove_target(env, obj) -> None:
    env.scene.remove_object(obj)


def _run_one(env, category: str, model: str, out_dir: Path, args) -> tuple[int, float]:
    """Sample + validate grasps for one target; return (num_valid, elapsed_s)."""
    import omnigibson as og
    import torch as th
    from sentinel.rl.grasps.collector import (
        GraspCollectorConfig, collect_valid_grasps, save_grasp_dataset,
    )
    from sentinel.rl.grasps.sampler import AntipodalConfig, sample_antipodal_grasps
    from sentinel.rl.grasps.mesh import franka_mounted_gripper_params, mesh_from_og_object

    robot = env.robots[0]
    initial_robot_joints = robot.get_joint_positions().clone()

    drop_z = args.table_top_z + 0.12
    obj = _spawn_target(env, category, model, tuple(args.object_xy), drop_z)
    spawn_pos = obj.get_position_orientation()[0]
    print(f"  spawned at {spawn_pos.tolist()}", flush=True)

    # Let gravity seat the object on the table.
    for _ in range(args.settle_steps):
        og.sim.step()
    settled_pos = obj.get_position_orientation()[0]
    print(f"  settled at {settled_pos.tolist()}", flush=True)

    try:
        mesh = mesh_from_og_object(obj, use_visual=not args.use_collision_mesh)
        print(f"  mesh: V={len(mesh.vertices)} F={len(mesh.faces)} "
              f"extents={[round(float(x), 3) for x in mesh.extents.tolist()]}",
              flush=True)

        gp = franka_mounted_gripper_params()
        acfg = AntipodalConfig(
            num_surface_samples=args.num_surface_samples,
            num_orientations=args.num_orientations,
            top_bias=False,
            **gp,
        )
        rng = np.random.default_rng(args.seed)
        candidates = sample_antipodal_grasps(mesh, acfg, rng=rng)
        # Shuffle so we don't burn attempts on clustered failures.
        candidates = candidates[rng.permutation(len(candidates))]
        print(f"  candidates: {len(candidates)}", flush=True)

        ccfg = GraspCollectorConfig(
            num_target_grasps=args.num_target_grasps,
            max_attempts=args.max_attempts,
        )
        t0 = time.time()
        valid = collect_valid_grasps(env, obj, candidates, cfg=ccfg, rng=rng, verbose=True)
        dt = time.time() - t0
        print(f"  validated {len(valid)} grasps in {dt:.1f}s", flush=True)

        if valid:
            stem = f"{category}_{model}"
            mesh.export(str(out_dir / f"{stem}.ply"))
            save_grasp_dataset(valid, out_dir / f"grasps_{stem}.pt", target_name=stem)
            print(f"  wrote {out_dir / f'grasps_{stem}.pt'}", flush=True)

        return len(valid), dt
    finally:
        # Always clean up, so next target starts from the known robot+table state.
        _remove_target(env, obj)
        robot.set_joint_positions(initial_robot_joints)
        og.sim.step()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = _parse_targets(args.targets)

    # OG boot.
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG (empty scene + Franka + table)...", flush=True)
    env = og.Environment(configs=_build_env_config())
    env.reset()
    robot = env.robots[0]
    print(f"  robot: {robot.name}  grasping_mode={robot.grasping_mode}", flush=True)
    table = env.scene.object_registry("name", "support_table")
    if table is not None:
        tp = table.get_position_orientation()[0]
        print(f"  table: pos={tp.tolist()} bbox_extent={table.aabb_extent.tolist()}", flush=True)
    if robot.grasping_mode == "sticky":
        raise RuntimeError("Need grasping_mode='physical' for real grasp validation.")

    summary: list[tuple[str, str, int, float]] = []
    for category, model in targets:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {category}/{model}", flush=True)
        print(f"{'=' * 60}", flush=True)
        try:
            n_valid, dt = _run_one(env, category, model, out, args)
            summary.append((category, model, n_valid, dt))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ! {category}/{model} failed: {exc}", flush=True)
            summary.append((category, model, -1, 0.0))

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}", flush=True)
    for category, model, n, dt in summary:
        status = "ERROR" if n < 0 else f"{n:3d} grasps"
        print(f"  {category:24s} {model:20s} {status}  ({dt:.1f}s)", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
