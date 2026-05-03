#!/usr/bin/env python3
"""Survey single-arm Franka graspability across every BEHAVIOR-1K object.

Boots one empty ``Scene`` with a fixed ``breakfast_table`` support and a
``FrankaPanda`` mounted on the table top (base z = table_top + 0.02 m).
For each ``(category, model)`` in ``behavior-1k/datasets/behavior-1k-assets/objects/``:

  1. Spawn the object at table center, settle under gravity.
  2. Extract the visual mesh, run :func:`sample_antipodal_grasps`.
  3. Iterate candidates with cuRobo IK (DEFAULT embodiment), teleport, open,
     close. Object grasping uses ``grasping_mode="assisted"``: contact between
     the AG ray segment and the target's mesh attaches the object.
  4. After close, query ``robot.is_grasping(arm, obj)``. If ``TRUE``, drive the
     eef +Z via OSC pose-delta and re-check both ``is_grasping`` and the
     object's vertical rise. First candidate that survives the lift = graspable;
     break and move on.
  5. Per object hard timeout (default 180 s) and resumable CSV output.

Why this is cheaper than ``collect_batch.py``: no shake test, no per-grasp
saving — assisted grasping snaps a stable bond on the first contact pair,
so a single hold + lift is sufficient to mark the object graspable.

Usage::

    conda activate behavior
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \\
        python -m sentinel.rl.grasps.survey_graspability \\
            --output outputs/grasp_datasets/survey/graspability.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Categories whose bbox has a zero dim (degenerate "flat" assets) — confirmed
# by sweeping all metadata.json files. Skipped wholesale rather than per-model.
_DEGENERATE_CATEGORIES = {"ceilings", "floors", "walls", "quilt"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Survey graspability of every BEHAVIOR-1K object with a Franka single-arm + assisted grasp."
    )
    p.add_argument("--output", type=Path,
                   default=Path("outputs/grasp_datasets/survey/graspability.csv"),
                   help="Resumable CSV. Existing rows are skipped.")
    p.add_argument("--dataset-root", type=Path,
                   default=Path("behavior-1k/datasets/behavior-1k-assets/objects"))
    p.add_argument("--per-object-timeout", type=float, default=180.0,
                   help="Hard wall-clock budget per (category, model) (seconds).")
    # Sampling
    p.add_argument("--num-surface-samples", type=int, default=64)
    p.add_argument("--num-orientations", type=int, default=16)
    p.add_argument("--max-candidates", type=int, default=400,
                   help="Cap on antipodal candidates iterated per object.")
    p.add_argument("--seed", type=int, default=0)
    # Scene geometry
    p.add_argument("--table-center-z", type=float, default=0.41,
                   help="World z of the breakfast_table center (its top is at center+halfH).")
    p.add_argument("--table-xy", type=float, nargs=2, default=[0.8, 0.0])
    p.add_argument("--object-xy", type=float, nargs=2, default=[0.85, 0.0],
                   help="World XY where each target is spawned (in front of the Franka base).")
    p.add_argument("--franka-xy", type=float, nargs=2, default=[0.55, 0.0],
                   help="World XY where the Franka panda_link0 is mounted on the table.")
    p.add_argument("--table-top-z-override", type=float, default=None,
                   help="Override the auto-computed table top z (= table_center_z + bbox_z/2 from metadata).")
    p.add_argument("--max-reach", type=float, default=0.95,
                   help="Skip candidates whose eef pos is further than this from the panda base.")
    p.add_argument("--prefilter-max-dim", type=float, default=0.5,
                   help="Skip objects whose largest bbox dim exceeds this (m). "
                        "0 disables the prefilter. 0.5 m catches typical "
                        "furniture/appliances; raise if you want them attempted anyway.")
    # Time-step counts
    p.add_argument("--settle-after-spawn-steps", type=int, default=40)
    p.add_argument("--settle-open-steps", type=int, default=8)
    p.add_argument("--close-steps", type=int, default=20)
    p.add_argument("--lift-steps", type=int, default=25)
    p.add_argument("--release-steps", type=int, default=6,
                   help="Open-gripper steps before removing object (releases AG bond).")
    p.add_argument("--min-lift-rise", type=float, default=0.04,
                   help="Object z must rise by at least this (m) to count as a successful lift.")
    return p.parse_args()


def _build_env_config(args, table_center_z: float, franka_base_z: float) -> dict:
    """Empty Scene + breakfast_table + FrankaPanda directly mounted on the table top.

    Notes:
      - ``grasping_mode="assisted"`` enables magnetic attachment when the AG ray
        between fingertip start/end points hits the target. ``is_grasping``
        then reads ``_ag_obj_in_hand`` rather than physical contact wrenches.
      - ``OperationalSpaceController`` is required for the lift phase
        (``_try_one_object`` drives +Z eef via pose-delta).
    """
    return dict(
        env={"action_frequency": 30, "physics_frequency": 300},
        scene=dict(type="Scene"),
        robots=[dict(
            type="FrankaPanda",
            name="agent_0",
            obs_modalities=["rgb"],
            action_type="continuous",
            action_normalize=True,
            grasping_mode="assisted",
            self_collisions=True,
            position=[args.franka_xy[0], args.franka_xy[1], franka_base_z],
            orientation=[0.0, 0.0, 0.0, 1.0],
            controller_config={
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        )],
        objects=[dict(
            type="DatasetObject",
            name="support_table",
            category="breakfast_table",
            model="rjgmmy",
            position=[args.table_xy[0], args.table_xy[1], table_center_z],
            orientation=[0.0, 0.0, 0.0, 1.0],
            fixed_base=True,
        )],
        task=dict(type="DummyTask"),
    )


def _enumerate_targets(dataset_root: Path):
    """Yield ``(category, model)`` for every model directory under ``dataset_root``,
    skipping degenerate categories (ceilings/floors/walls/quilt)."""
    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if cat_dir.name in _DEGENERATE_CATEGORIES:
            continue
        for model_dir in sorted(cat_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            yield cat_dir.name, model_dir.name


def _load_done(csv_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["category"], row["model"]))
    return done


def _curobo_ik_fast(motion_gen, robot, arm, eef_pos, eef_quat, skip_obstacle_update):
    """Survey-tuned IK: same DEFAULT-embodiment plumbing as
    :func:`sentinel.rl.grasps.collector._curobo_ik`, but with much smaller
    planning + retry budgets so a single IK call cannot eat the whole
    per-object 180 s deadline.

    Field-observed: with the collector's defaults
    (``timeout=60``, ``max_attempts=ceil(MAX_PLANNING_ATTEMPTS/bs)``,
    ``ik_fail_return=MAX_IK_FAILURES_BEFORE_RETURN``) cuRobo can spin > 600 s
    on hard configs (e.g. armchair surfaces). Survey just needs a yes/no, so
    we cap each call at ~10 s and a single attempt.
    """
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    eef_link = robot.eef_link_names[arm]
    bs = motion_gen.batch_size
    target_pos = {eef_link: th.stack([eef_pos for _ in range(bs)])}
    target_quat = {eef_link: th.stack([eef_quat for _ in range(bs)])}

    successes, joint_states = motion_gen.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        initial_joint_pos=None,
        is_local=False,
        max_attempts=1,
        timeout=10.0,
        ik_fail_return=1,
        enable_finetune_trajopt=False,
        finetune_attempts=0,
        return_full_result=False,
        success_ratio=1.0 / bs,
        attached_obj=None,
        attached_obj_scale=None,
        motion_constraint=None,
        skip_obstacle_update=skip_obstacle_update,
        ik_only=True,
        ik_world_collision_check=False,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    )
    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        return None
    joint_state = joint_states[success_idx[0]]
    joint_pos = motion_gen.path_to_joint_trajectory(
        joint_state, get_full_js=False, emb_sel=CuRoboEmbodimentSelection.DEFAULT
    )
    manip_idx = th.cat([robot.arm_control_idx[arm]])
    return joint_pos[manip_idx].cpu()


def _metadata_prefilter(dataset_root: Path, category: str, model: str,
                        max_dim_limit: float) -> str | None:
    """Cheap reject for objects guaranteed not to be Franka-graspable.

    Reads only ``misc/metadata.json``; no USD load. Returns a short reason
    string when the object should be skipped, else ``None``.

    Heuristic: if the object's largest bbox dim exceeds ``max_dim_limit``,
    it is likely furniture / appliance / fixture. Even if the antipodal
    sampler finds candidates on a thin sub-feature (e.g. armrest), the AG
    mass threshold (10 kg in OG) and the lift-against-gravity test will
    reject the whole object — but each candidate burns IK + sim time first.
    Cutting these at the metadata layer reclaims hours.
    """
    import json
    p = dataset_root / category / model / "misc" / "metadata.json"
    try:
        bbox = json.loads(p.read_text())["bbox_size"]
    except Exception:  # noqa: BLE001
        return None
    if any((x is None or x <= 0) for x in bbox):
        return "degenerate_bbox"
    if float(max(bbox)) > max_dim_limit:
        return f"too_large(max_dim={max(bbox):.2f}m)"
    return None


def _step_loop(robot, action, n_steps, deadline):
    """Run ``n_steps`` of ``apply_action`` + ``sim.step``, breaking on deadline.

    Returns ``True`` if the loop completed within budget, ``False`` if the
    deadline tripped before all steps ran. Used for settle / close / lift —
    each phase can blow up under heavy contact (PhysX substep count balloons),
    and a single 947 s candidate was observed in field testing when only the
    outer-loop deadline check was in place.
    """
    import omnigibson as og
    for _ in range(n_steps):
        if time.time() > deadline:
            return False
        robot.apply_action(action)
        og.sim.step()
    return True


def _try_one_object(env, primitives, category: str, model: str, args, deadline: float):
    """Spawn one object and try every antipodal candidate until one holds.

    Returns ``(status, n_candidates, n_tried, note)``.
    """
    import omnigibson as og
    import torch as th
    from omnigibson.controllers.controller_base import IsGraspingState
    from omnigibson.objects import DatasetObject

    from sentinel.rl.grasps.collector import (
        _build_action,
        _mat_to_pose,
        _pose_to_mat,
        _reset_controller_goals,
    )
    from sentinel.rl.grasps.mesh import franka_panda_gripper_params, mesh_from_og_object
    from sentinel.rl.grasps.sampler import AntipodalConfig, sample_antipodal_grasps

    robot = env.robots[0]
    arm = robot.default_arm
    arm_control_idx = robot.arm_control_idx[arm]
    gripper_control_idx = robot.gripper_control_idx[arm]
    initial_joint_pos = robot.get_joint_positions().clone()
    zero_arm_cmd = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)
    open_gripper_q = robot.joint_upper_limits[gripper_control_idx].clone()

    drop_z = args.table_top_z + 0.12
    name = f"survey_target_{category}_{model}"
    try:
        obj = DatasetObject(name=name, category=category, model=model)
        env.scene.add_object(obj)
    except Exception as e:  # noqa: BLE001
        # Try to scrub any partial-add state so the next iteration starts clean.
        try:
            existing = env.scene.object_registry("name", name)
            if existing is not None:
                env.scene.remove_object(existing)
        except Exception:  # noqa: BLE001
            pass
        return "spawn_failed", 0, 0, f"{type(e).__name__}: {str(e)[:160]}"

    obj.set_position_orientation(
        position=th.tensor([args.object_xy[0], args.object_xy[1], drop_z], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    obj.root_link.set_linear_velocity(th.zeros(3))
    obj.root_link.set_angular_velocity(th.zeros(3))

    try:
        # Settle under gravity.
        for _ in range(args.settle_after_spawn_steps):
            og.sim.step()
            if time.time() > deadline:
                return "timeout", 0, 0, "settle"

        init_pos, init_quat = obj.get_position_orientation()
        init_pos = init_pos.clone()
        init_quat = init_quat.clone()
        T_target_world = _pose_to_mat(init_pos.cpu().numpy(), init_quat.cpu().numpy())

        try:
            mesh = mesh_from_og_object(obj, use_visual=True)
        except Exception as e:  # noqa: BLE001
            return "mesh_failed", 0, 0, f"{type(e).__name__}: {str(e)[:120]}"

        gp = franka_panda_gripper_params()
        acfg = AntipodalConfig(
            num_surface_samples=args.num_surface_samples,
            num_orientations=args.num_orientations,
            top_bias=False,
            **gp,
        )
        rng = np.random.default_rng(args.seed)
        candidates = sample_antipodal_grasps(mesh, acfg, rng=rng)
        n_total = len(candidates)
        if n_total == 0:
            return "no_candidates", 0, 0, ""
        candidates = candidates[rng.permutation(n_total)]
        if n_total > args.max_candidates:
            candidates = candidates[:args.max_candidates]

        robot_base_np = (robot.get_position_orientation()[0]
                         .cpu().numpy().astype(np.float64).reshape(3))

        n_tried = 0
        first_call = True
        for i, T_local in enumerate(candidates):
            if time.time() > deadline:
                return "timeout", n_total, n_tried, f"after {n_tried} tries"

            T_local = np.asarray(T_local, dtype=np.float64)
            T_eef_world = T_target_world @ T_local
            eef_pos_np, eef_quat_np = _mat_to_pose(T_eef_world)

            if float(np.linalg.norm(eef_pos_np - robot_base_np)) > args.max_reach:
                continue
            if float(T_eef_world[2, 2]) > -0.3:  # approach must point downward
                continue

            n_tried += 1
            eef_pos = th.tensor(eef_pos_np, dtype=th.float32)
            eef_quat = th.tensor(eef_quat_np, dtype=th.float32)

            # Reset arm + object to spawn pose. Explicitly drop any AG bond
            # held over from the previous candidate; ``set_joint_positions``
            # alone leaves ``_ag_obj_in_hand`` populated and ``is_grasping``
            # would still report TRUE.
            robot.release_grasp_immediately(arm)
            robot.set_joint_positions(initial_joint_pos)
            obj.set_position_orientation(init_pos, init_quat)
            obj.root_link.set_linear_velocity(th.zeros(3))
            obj.root_link.set_angular_velocity(th.zeros(3))
            _reset_controller_goals(robot)
            og.sim.step()

            try:
                joint_pos = _curobo_ik_fast(
                    primitives._motion_generator, robot, arm,
                    eef_pos, eef_quat,
                    skip_obstacle_update=not first_call,
                )
            except Exception:  # noqa: BLE001
                continue
            first_call = False
            # Even with a 10 s cuRobo cap, cumulative IK across many
            # candidates can still cross the per-object deadline.
            if time.time() > deadline:
                return "timeout", n_total, n_tried, f"ik {n_tried}"
            if joint_pos is None:
                continue

            # Teleport arm + open gripper, then settle.
            robot.set_joint_positions(joint_pos, arm_control_idx)
            robot.set_joint_positions(open_gripper_q, gripper_control_idx)
            _reset_controller_goals(robot)

            settle_action = _build_action(robot, zero_arm_cmd, gripper_cmd=+1.0)
            if not _step_loop(robot, settle_action, args.settle_open_steps, deadline):
                return "timeout", n_total, n_tried, f"settle {n_tried}"

            # Close — assisted-grasp fires when the AG ray segment intersects the target.
            close_action = _build_action(robot, zero_arm_cmd, gripper_cmd=-1.0)
            if not _step_loop(robot, close_action, args.close_steps, deadline):
                return "timeout", n_total, n_tried, f"close {n_tried}"

            if robot.is_grasping(arm, obj) != IsGraspingState.TRUE:
                continue

            # Lift via OSC pose-delta. Same magnitude (0.1) used by collect_batch's lift phase.
            lift_arm = th.zeros(len(robot.arm_action_idx[arm]), dtype=th.float32)
            lift_arm[2] = 0.1
            lift_action = _build_action(robot, lift_arm, gripper_cmd=-1.0)
            z_before = float(obj.get_position_orientation()[0][2])
            if not _step_loop(robot, lift_action, args.lift_steps, deadline):
                return "timeout", n_total, n_tried, f"lift {n_tried}"
            z_after = float(obj.get_position_orientation()[0][2])

            still_grasping = robot.is_grasping(arm, obj) == IsGraspingState.TRUE
            z_rise = z_after - z_before
            if still_grasping and z_rise >= args.min_lift_rise:
                return ("graspable", n_total, n_tried,
                        f"i={i} z_rise={z_rise:.3f}")

        return "no_grasp", n_total, n_tried, ""
    finally:
        # Drop AG bond explicitly first; otherwise remove_object can race with
        # the AG joint that still references the (about-to-be-deleted) prim.
        try:
            robot.release_grasp_immediately(arm)
        except Exception:  # noqa: BLE001
            pass
        try:
            release_action = _build_action(robot, zero_arm_cmd, gripper_cmd=+1.0)
            for _ in range(args.release_steps):
                robot.apply_action(release_action)
                og.sim.step()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.scene.remove_object(obj)
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.set_joint_positions(initial_joint_pos)
            _reset_controller_goals(robot)
            og.sim.step()
        except Exception:  # noqa: BLE001
            pass


def _read_bbox_z(dataset_root: Path, category: str, model: str) -> float:
    """Read ``bbox_size[2]`` from the model's metadata.json (cheap, no sim)."""
    import json
    p = dataset_root / category / model / "misc" / "metadata.json"
    return float(json.loads(p.read_text())["bbox_size"][2])


def main():
    args = parse_args()
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    targets = list(_enumerate_targets(dataset_root))
    done = _load_done(out)
    remaining = [t for t in targets if t not in done]
    print(f"[{time.strftime('%H:%M:%S')}] {len(targets)} targets total, "
          f"{len(done)} already in CSV, {len(remaining)} to run.", flush=True)
    if not remaining:
        print("Nothing to do.", flush=True)
        return

    # Precompute table-top z from the breakfast_table metadata (avoids a post-boot
    # reposition of a fixed_base=True FrankaPanda, which is finicky in OG).
    if args.table_top_z_override is not None:
        table_top_z = float(args.table_top_z_override)
    else:
        table_bbox_z = _read_bbox_z(dataset_root, "breakfast_table", "rjgmmy")
        table_top_z = float(args.table_center_z) + table_bbox_z / 2.0
    franka_base_z = table_top_z + 0.02
    args.table_top_z = table_top_z  # consumed inside _try_one_object
    print(f"[{time.strftime('%H:%M:%S')}] table_top_z={table_top_z:.3f}  "
          f"franka_base_z={franka_base_z:.3f}", flush=True)

    # OG boot.
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    import omnigibson as og

    print(f"[{time.strftime('%H:%M:%S')}] Booting OG (FrankaPanda + breakfast_table)...",
          flush=True)
    env = og.Environment(configs=_build_env_config(
        args, table_center_z=args.table_center_z, franka_base_z=franka_base_z,
    ))
    env.reset()

    robot = env.robots[0]
    table = env.scene.object_registry("name", "support_table")
    print(f"  table: pos={table.get_position_orientation()[0].tolist()} "
          f"aabb_extent={table.aabb_extent.tolist()}", flush=True)
    print(f"  robot: {type(robot).__name__}  grasping_mode={robot.grasping_mode}  "
          f"base_pos={robot.get_position_orientation()[0].tolist()}", flush=True)
    if robot.grasping_mode != "assisted":
        raise RuntimeError("Need grasping_mode='assisted'.")

    # cuRobo init — first call is slow, subsequent reuses are fast.
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )
    print(f"  [{time.strftime('%H:%M:%S')}] initializing cuRobo motion generator...",
          flush=True)
    primitives = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
    print(f"  [{time.strftime('%H:%M:%S')}] cuRobo ready.", flush=True)

    is_new = not out.exists()
    csv_file = open(out, "a", newline="")
    writer = csv.writer(csv_file)
    if is_new:
        writer.writerow(["category", "model", "status",
                         "n_candidates", "n_tried", "elapsed_s", "note"])
        csv_file.flush()

    n_done = 0
    counts = {}
    try:
        for category, model in remaining:
            n_done += 1
            t0 = time.time()
            deadline = t0 + args.per_object_timeout
            print(f"\n[{time.strftime('%H:%M:%S')}] "
                  f"({n_done}/{len(remaining)}) {category}/{model}", flush=True)

            # Cheap metadata prefilter — skips guaranteed-too-large items
            # without paying the USD load + cuRobo IK cost.
            prefilter_reason = (
                _metadata_prefilter(dataset_root, category, model, args.prefilter_max_dim)
                if args.prefilter_max_dim > 0 else None
            )
            if prefilter_reason is not None:
                status, n_cand, n_tried, note = "too_large", 0, 0, prefilter_reason
            else:
                try:
                    status, n_cand, n_tried, note = _try_one_object(
                        env, primitives, category, model, args, deadline,
                    )
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    status, n_cand, n_tried, note = "error", 0, 0, repr(exc)[:200]
            elapsed = time.time() - t0
            writer.writerow([category, model, status, n_cand, n_tried,
                             f"{elapsed:.1f}", note])
            csv_file.flush()
            counts[status] = counts.get(status, 0) + 1
            print(f"  -> {status}  n_cand={n_cand} n_tried={n_tried} "
                  f"{elapsed:.1f}s  cum={dict(sorted(counts.items()))}", flush=True)
    finally:
        csv_file.close()

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. CSV: {out}\n  totals: {counts}",
          flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
